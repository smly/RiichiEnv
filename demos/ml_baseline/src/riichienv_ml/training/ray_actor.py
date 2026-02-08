import time

import ray
import torch
import numpy as np
from riichienv import RiichiEnv

from riichienv_ml.config import import_class
from riichienv_ml.models.grp_model import RewardPredictor


def sample_top_p(logits, p):
    """Nucleus (top-p) sampling from logits."""
    if p >= 1:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
    if p <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled


@ray.remote
class MahjongWorker:
    """
    DQN rollout worker. Collects transitions with decayed returns for
    off-policy training. Episodes are split at kyoku (round) boundaries.
    Per-kyoku rewards are computed using the GRP (GlobalRewardPredictor),
    matching the reward structure used during CQL pretraining.
    """
    def __init__(self, worker_id: int, device: str = "cpu",
                 gamma: float = 0.99,
                 exploration: str = "boltzmann",
                 # epsilon-greedy params
                 epsilon: float = 0.1,
                 # Boltzmann params
                 boltzmann_epsilon: float = 0.1,
                 boltzmann_temp: float = 1.0,
                 top_p: float = 0.9,
                 num_envs: int = 16,
                 model_config: dict | None = None,
                 model_class: str = "riichienv_ml.models.cql_model.QNetwork",
                 encoder_class: str = "riichienv_ml.data.cql_dataset.ObservationEncoder",
                 grp_model: str | None = None,
                 pts_weight: list[float] | None = None):
        torch.set_num_threads(1)
        self.worker_id = worker_id
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.envs = [RiichiEnv(game_mode="4p-red-half") for _ in range(num_envs)]
        self.gamma = gamma
        self.exploration = exploration
        self.epsilon = epsilon
        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

        mc = model_config or {}
        ModelClass = import_class(model_class)
        self.model = ModelClass(**mc).to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            self.model = torch.compile(self.model)
        self.encoder = import_class(encoder_class)
        self._compiled_warmup = False

        # GRP reward predictor
        pw = pts_weight or [10.0, 4.0, -4.0, -10.0]
        if grp_model:
            self.reward_predictor = RewardPredictor(grp_model, pw, device="cpu")
        else:
            self.reward_predictor = None

    def update_weights(self, state_dict):
        """Syncs weights from the Learner."""
        # torch.compile wraps the model; load into the underlying module
        target = self.model
        if hasattr(target, "_orig_mod"):
            target = target._orig_mod
        target.load_state_dict(state_dict)

    def set_epsilon(self, epsilon: float):
        """Updates epsilon for epsilon-greedy exploration."""
        self.epsilon = epsilon

    def set_boltzmann_temp(self, temp: float):
        """Updates Boltzmann temperature."""
        self.boltzmann_temp = temp

    def _warmup_compile(self):
        """Run a dummy forward pass to trigger torch.compile JIT."""
        if self._compiled_warmup:
            return
        target = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        in_ch = target.backbone.conv_in.in_channels
        dummy = torch.randn(1, in_ch, 34, device=self.device)
        with torch.no_grad():
            self.model(dummy)
        self._compiled_warmup = True

    def _compute_grp_expected_pts(self, prev_scores, cur_scores,
                                  round_wind, oya, honba, riichi_sticks,
                                  pid: int) -> float:
        """Compute GRP expected points for the current game state after a kyoku.

        Returns the expected ranking-based points (potential) for the given player.
        The per-kyoku reward is the *change* in this potential between
        consecutive kyoku boundaries (computed by the caller).
        """
        if self.reward_predictor is None:
            return 0.0

        deltas = [cur_scores[i] - prev_scores[i] for i in range(4)]
        grp_features = {
            "p0_init_score": prev_scores[0],
            "p1_init_score": prev_scores[1],
            "p2_init_score": prev_scores[2],
            "p3_init_score": prev_scores[3],
            "p0_end_score": cur_scores[0],
            "p1_end_score": cur_scores[1],
            "p2_end_score": cur_scores[2],
            "p3_end_score": cur_scores[3],
            "p0_delta_score": deltas[0],
            "p1_delta_score": deltas[1],
            "p2_delta_score": deltas[2],
            "p3_delta_score": deltas[3],
            "chang": round_wind,
            "ju": oya,
            "ben": honba,
            "liqibang": riichi_sticks,
        }
        pts, _ = self.reward_predictor.calc_pts_rewards([grp_features], pid)
        return pts[0].detach().cpu().item()

    def collect_episodes(self):
        """
        Runs N half-games (hanchan) in parallel using batched inference.
        All 4 players use the same model with exploration.

        Each hanchan is split into per-kyoku trajectories. At each kyoku boundary,
        GRP computes the reward for that kyoku. Per-kyoku decayed returns
        R * γ^(T-t-1) are computed matching the CQL pretraining reward structure.
        """
        t_start = time.time()

        if not self._compiled_warmup:
            self._warmup_compile()

        # Reset all envs
        obs_dicts = [
            env.reset(scores=[25000, 25000, 25000, 25000],
                      bakaze=0, oya=0, honba=0, kyotaku=0)
            for env in self.envs
        ]

        active = [True] * self.num_envs

        # Per-kyoku per-player trajectory buffer (reset at each kyoku boundary)
        kyoku_buffers = [{pid: [] for pid in range(4)} for _ in range(self.num_envs)]
        # Completed kyoku trajectories: list of (pid, trajectory, reward) per env
        completed_kyokus = [[] for _ in range(self.num_envs)]

        # Track kyoku state for boundary detection
        prev_kyoku_idx = [env.kyoku_idx for env in self.envs]
        kyoku_start_scores = [list(env.scores()) for env in self.envs]
        kyoku_start_meta = [(env.round_wind, env.oya, env.honba, env.riichi_sticks)
                            for env in self.envs]
        # GRP potential per player: expected pts at start of hanchan = mean(pts_weight) = 0.0
        init_pts = float(np.mean(self.reward_predictor.pts_weight)) if self.reward_predictor else 0.0
        prev_grp_pts = [[init_pts] * 4 for _ in range(self.num_envs)]

        while any(active):
            # Collect observations from all active envs
            batch_items = []  # (env_idx, pid, obs, legal_actions)
            for ei in range(self.num_envs):
                if not active[ei]:
                    continue
                for pid, obs in obs_dicts[ei].items():
                    la = obs.legal_actions()
                    if la:
                        batch_items.append((ei, pid, obs, la))

            if not batch_items:
                for ei in range(self.num_envs):
                    if active[ei] and self.envs[ei].done():
                        active[ei] = False
                break

            # Encode all observations
            feat_list = []
            mask_list = []
            for _, _, obs, _ in batch_items:
                feat = self.encoder.encode(obs)
                feat_list.append(feat)
                m = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
                mask_list.append(torch.from_numpy(m))

            # Batched to-device + forward
            feat_batch = torch.stack(feat_list).to(self.device)
            mask_batch = torch.stack(mask_list).to(self.device)

            with torch.no_grad():
                q_values = self.model(feat_batch)
                mask_bool = mask_batch.bool()
                q_values = q_values.masked_fill(~mask_bool, -torch.inf)

            # Batched action selection
            if torch.isnan(q_values).any():
                # Fallback: per-item random legal action
                actions_cpu = np.zeros(len(batch_items), dtype=np.int64)
                mask_np = mask_batch.cpu().numpy()
                for idx in range(len(batch_items)):
                    legal = np.where(mask_np[idx] > 0)[0]
                    actions_cpu[idx] = np.random.choice(legal)
            else:
                actions = q_values.argmax(dim=1)
                if self.exploration == "boltzmann" and self.boltzmann_epsilon > 0:
                    logits = q_values / self.boltzmann_temp
                    logits = logits.masked_fill(~mask_bool, -torch.inf)
                    probs = torch.softmax(logits, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    use_boltzmann = torch.rand(len(batch_items), device=self.device) < self.boltzmann_epsilon
                    actions = torch.where(use_boltzmann, sampled, actions)
                elif self.exploration == "epsilon_greedy" and self.epsilon > 0:
                    mask_np = mask_batch.cpu().numpy()
                    random_actions = np.array([
                        np.random.choice(np.where(mask_np[i] > 0)[0])
                        for i in range(len(batch_items))
                    ])
                    random_actions_t = torch.from_numpy(random_actions).to(self.device)
                    use_random = torch.rand(len(batch_items), device=self.device) < self.epsilon
                    actions = torch.where(use_random, random_actions_t, actions)
                actions_cpu = actions.cpu().numpy()

            # Store transitions + build env steps
            feat_cpu = feat_batch.cpu().numpy()
            mask_cpu = mask_batch.cpu().numpy()
            env_steps = {ei: {} for ei in range(self.num_envs)}

            for idx, (ei, pid, obs, la) in enumerate(batch_items):
                action_idx = int(actions_cpu[idx])
                kyoku_buffers[ei][pid].append({
                    "features": feat_cpu[idx],
                    "mask": mask_cpu[idx],
                    "action": action_idx,
                })
                found_action = obs.find_action(action_idx)
                if found_action is None:
                    found_action = la[0]
                env_steps[ei][pid] = found_action

            # Step all active envs
            for ei in range(self.num_envs):
                if not active[ei]:
                    continue
                if env_steps[ei]:
                    obs_dicts[ei] = self.envs[ei].step(env_steps[ei])

                env = self.envs[ei]

                # Detect kyoku boundary via kyoku_idx change
                cur_kyoku_idx = env.kyoku_idx
                if cur_kyoku_idx != prev_kyoku_idx[ei]:
                    cur_scores = list(env.scores())
                    rw, oya, honba, rsticks = kyoku_start_meta[ei]
                    for pid in range(4):
                        if kyoku_buffers[ei][pid]:
                            cur_pts = self._compute_grp_expected_pts(
                                kyoku_start_scores[ei], cur_scores,
                                rw, oya, honba, rsticks, pid)
                            reward = cur_pts - prev_grp_pts[ei][pid]
                            completed_kyokus[ei].append(
                                (pid, kyoku_buffers[ei][pid], reward))
                            kyoku_buffers[ei][pid] = []
                            prev_grp_pts[ei][pid] = cur_pts
                    # Update tracking for next kyoku
                    prev_kyoku_idx[ei] = cur_kyoku_idx
                    kyoku_start_scores[ei] = cur_scores
                    kyoku_start_meta[ei] = (env.round_wind, env.oya,
                                            env.honba, env.riichi_sticks)

                if env.done():
                    # Flush remaining kyoku buffer (final kyoku of the hanchan)
                    cur_scores = list(env.scores())
                    rw, oya, honba, rsticks = kyoku_start_meta[ei]
                    for pid in range(4):
                        if kyoku_buffers[ei][pid]:
                            cur_pts = self._compute_grp_expected_pts(
                                kyoku_start_scores[ei], cur_scores,
                                rw, oya, honba, rsticks, pid)
                            reward = cur_pts - prev_grp_pts[ei][pid]
                            completed_kyokus[ei].append(
                                (pid, kyoku_buffers[ei][pid], reward))
                            kyoku_buffers[ei][pid] = []
                    active[ei] = False

        # Compute per-kyoku decayed returns: R * γ^(T-t-1) matching CQL
        transitions = []
        episode_rewards = []
        episode_ranks = []
        kyoku_lengths = []
        kyoku_rewards = []

        for ei in range(self.num_envs):
            ranks = self.envs[ei].ranks()
            # Track rank 0 stats for logging
            rank = ranks[0]
            final_reward = 0.0
            if rank == 1: final_reward = 10.0
            elif rank == 2: final_reward = 4.0
            elif rank == 3: final_reward = -4.0
            elif rank == 4: final_reward = -10.0
            episode_rewards.append(final_reward)
            episode_ranks.append(rank)

            for pid, traj, kyoku_reward in completed_kyokus[ei]:
                T = len(traj)
                if T == 0:
                    continue

                kyoku_lengths.append(T)
                kyoku_rewards.append(kyoku_reward)

                for t, step in enumerate(traj):
                    decayed_return = kyoku_reward * (self.gamma ** (T - t - 1))
                    step["reward"] = np.array(decayed_return, dtype=np.float32)
                    step["done"] = bool(t == T - 1)
                    transitions.append(step)

        # Worker-level stats
        stats = {}
        if episode_rewards:
            stats["reward_mean"] = float(np.mean(episode_rewards))
            stats["reward_std"] = float(np.std(episode_rewards))
            stats["rank_mean"] = float(np.mean(episode_ranks))
        if kyoku_lengths:
            stats["kyoku_length_mean"] = float(np.mean(kyoku_lengths))
            stats["kyoku_reward_mean"] = float(np.mean(kyoku_rewards))
            stats["kyoku_reward_std"] = float(np.std(kyoku_rewards))
            stats["kyokus_per_hanchan"] = float(len(kyoku_lengths) / self.num_envs)

        t_end = time.time()
        episode_time = t_end - t_start
        if transitions:
            print(f"Worker {self.worker_id}: {self.num_envs} hanchan, "
                  f"{len(kyoku_lengths)} kyokus took {episode_time:.3f}s, "
                  f"{len(transitions)} transitions, {len(transitions)/episode_time:.1f} trans/s")

        return transitions, stats
