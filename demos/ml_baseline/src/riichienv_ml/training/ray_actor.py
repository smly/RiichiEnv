import random
import time

import ray
import torch
import numpy as np
from riichienv import RiichiEnv

from riichienv_ml.config import import_class
from riichienv_ml.models.grp_model import RewardPredictor


def _compute_ranks(scores: list) -> list[int]:
    """Compute ranks (0=1st, 3=4th) from scores for all 4 players."""
    arr = np.array(scores, dtype=np.float64)
    return (-arr).argsort(kind='stable').argsort(kind='stable').tolist()


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
class RVWorker:
    """
    DQN rollout worker. Collects transitions with decayed returns for
    off-policy training. Episodes are split at kyoku (round) boundaries.
    Per-kyoku rewards are computed using the GRP (GlobalRewardPredictor),
    matching the reward structure used during CQL pretraining.

    Also supports batched evaluation: hero (player 0, greedy) vs frozen
    baseline (players 1-3, greedy) using ``evaluate_episodes()``.
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

        # Baseline model (frozen opponents for evaluation)
        self.baseline_model = ModelClass(**mc).to(self.device)
        self.baseline_model.eval()

        if self.device.type == "cuda":
            self.model = torch.compile(self.model)
            self.baseline_model = torch.compile(self.baseline_model)
        self.encoder = import_class(encoder_class)
        self._compiled_warmup = False

        # GRP reward predictor
        pw = pts_weight or [10.0, 4.0, -4.0, -10.0]
        if grp_model:
            self.reward_predictor = RewardPredictor(grp_model, pw, device="cpu")
        else:
            self.reward_predictor = None

    def _apply_state_dict(self, model, state_dict):
        """Apply state dict by copying parameter data directly."""
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        for name, param in target.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name])
        for name, buf in target.named_buffers():
            if name in state_dict:
                buf.data.copy_(state_dict[name])

    def update_weights(self, state_dict):
        """Syncs hero model weights from the Learner."""
        self._apply_state_dict(self.model, state_dict)

    def update_baseline_weights(self, state_dict):
        """Syncs baseline model weights (called once at initialization)."""
        self._apply_state_dict(self.baseline_model, state_dict)

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
            self.baseline_model(dummy)
        self._compiled_warmup = True

    def _compute_kyoku_rewards(self, prev_scores, cur_scores,
                               round_wind, oya, honba, riichi_sticks) -> list[float]:
        """Compute absolute GRP rewards for all 4 players after a kyoku.

        Returns [reward_p0, ..., reward_p3] where each reward = GRP_pts - mean_pts.
        This matches the reward scale used during CQL offline pretraining
        (each kyoku's reward is independent, not incremental).
        """
        if self.reward_predictor is None:
            return [0.0] * 4

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
        return self.reward_predictor.calc_all_player_rewards(grp_features)

    def collect_episodes(self):
        """
        Runs N half-games (hanchan) in parallel using batched inference.
        Hero (random pid per env) uses current model with exploration;
        opponents use frozen baseline with greedy argmax.

        Only the hero's transitions are collected for training.
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
        hero_pids = [random.randint(0, 3) for _ in range(self.num_envs)]

        # Per-kyoku hero trajectory buffer (reset at each kyoku boundary)
        kyoku_buffers = [[] for _ in range(self.num_envs)]
        # Completed kyoku trajectories: list of (trajectory, reward, rank) per env
        completed_kyokus = [[] for _ in range(self.num_envs)]

        # Track kyoku state for boundary detection
        prev_kyoku_idx = [env.kyoku_idx for env in self.envs]
        kyoku_start_scores = [list(env.scores()) for env in self.envs]
        kyoku_start_meta = [(env.round_wind, env.oya, env.honba, env.riichi_sticks)
                            for env in self.envs]
        kyoku_count = [0] * self.num_envs  # for debug logging

        while any(active):
            # Separate hero and opponent observations
            hero_items = []  # (ei, obs, la)
            opp_items = []   # (ei, pid, obs, la)

            for ei in range(self.num_envs):
                if not active[ei]:
                    continue
                for pid, obs in obs_dicts[ei].items():
                    la = obs.legal_actions()
                    if la:
                        if pid == hero_pids[ei]:
                            hero_items.append((ei, obs, la))
                        else:
                            opp_items.append((ei, pid, obs, la))

            if not hero_items and not opp_items:
                for ei in range(self.num_envs):
                    if active[ei] and self.envs[ei].done():
                        active[ei] = False
                break

            env_steps = {ei: {} for ei in range(self.num_envs)}

            # === Hero inference (current model with exploration) ===
            if hero_items:
                feat_list = []
                mask_list = []
                for _, obs, _ in hero_items:
                    feat_list.append(self.encoder.encode(obs))
                    mask_list.append(torch.from_numpy(
                        np.frombuffer(obs.mask(), dtype=np.uint8).copy()))

                feat_batch = torch.stack(feat_list).to(self.device)
                mask_batch = torch.stack(mask_list).to(self.device)

                with torch.no_grad():
                    q_values = self.model(feat_batch)
                    mask_bool = mask_batch.bool()
                    q_values = q_values.masked_fill(~mask_bool, -torch.inf)

                # Action selection with exploration
                if torch.isnan(q_values).any():
                    actions_cpu = np.zeros(len(hero_items), dtype=np.int64)
                    mask_np = mask_batch.cpu().numpy()
                    for idx in range(len(hero_items)):
                        legal = np.where(mask_np[idx] > 0)[0]
                        actions_cpu[idx] = np.random.choice(legal)
                else:
                    actions = q_values.argmax(dim=1)
                    if self.exploration == "boltzmann" and self.boltzmann_epsilon > 0:
                        logits = q_values / self.boltzmann_temp
                        logits = logits.masked_fill(~mask_bool, -torch.inf)
                        probs = torch.softmax(logits, dim=-1)
                        sampled = torch.multinomial(probs, 1).squeeze(-1)
                        use_boltzmann = torch.rand(len(hero_items), device=self.device) < self.boltzmann_epsilon
                        actions = torch.where(use_boltzmann, sampled, actions)
                    elif self.exploration == "epsilon_greedy" and self.epsilon > 0:
                        mask_np = mask_batch.cpu().numpy()
                        random_actions = np.array([
                            np.random.choice(np.where(mask_np[i] > 0)[0])
                            for i in range(len(hero_items))
                        ])
                        random_actions_t = torch.from_numpy(random_actions).to(self.device)
                        use_random = torch.rand(len(hero_items), device=self.device) < self.epsilon
                        actions = torch.where(use_random, random_actions_t, actions)
                    actions_cpu = actions.cpu().numpy()

                # Store hero transitions
                feat_cpu = feat_batch.cpu().numpy()
                mask_cpu = mask_batch.cpu().numpy()
                for idx, (ei, obs, la) in enumerate(hero_items):
                    action_idx = int(actions_cpu[idx])
                    kyoku_buffers[ei].append({
                        "features": feat_cpu[idx],
                        "mask": mask_cpu[idx],
                        "action": action_idx,
                    })
                    found_action = obs.find_action(action_idx)
                    if found_action is None:
                        found_action = la[0]
                    env_steps[ei][hero_pids[ei]] = found_action

            # === Opponent inference (frozen baseline, greedy argmax) ===
            if opp_items:
                feat_list = []
                mask_list = []
                for _, _, obs, _ in opp_items:
                    feat_list.append(self.encoder.encode(obs))
                    mask_list.append(torch.from_numpy(
                        np.frombuffer(obs.mask(), dtype=np.uint8).copy()))

                feat_batch = torch.stack(feat_list).to(self.device)
                mask_batch = torch.stack(mask_list).to(self.device)

                with torch.no_grad():
                    q_values = self.baseline_model(feat_batch)
                    q_values = q_values.masked_fill(~mask_batch.bool(), -1e9)
                    actions = q_values.argmax(dim=1)

                actions_cpu = actions.cpu().numpy()
                for idx, (ei, pid, obs, la) in enumerate(opp_items):
                    action_idx = int(actions_cpu[idx])
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
                if cur_kyoku_idx != prev_kyoku_idx[ei] and kyoku_buffers[ei]:
                    cur_scores = list(env.scores())
                    ranks = _compute_ranks(cur_scores)
                    rw, oya, honba, rsticks = kyoku_start_meta[ei]
                    all_rewards = self._compute_kyoku_rewards(
                        kyoku_start_scores[ei], cur_scores, rw, oya, honba, rsticks)
                    reward = all_rewards[hero_pids[ei]]
                    completed_kyokus[ei].append(
                        (kyoku_buffers[ei], reward, ranks[hero_pids[ei]]))
                    kyoku_buffers[ei] = []
                    kyoku_count[ei] += 1
                    if ei == 0 and kyoku_count[ei] <= 3:
                        print(f"  [DEBUG] env0 kyoku {kyoku_count[ei]}: "
                              f"hero={hero_pids[ei]} reward={reward:.3f} "
                              f"all={[f'{r:.3f}' for r in all_rewards]}")
                    # Update tracking for next kyoku
                    prev_kyoku_idx[ei] = cur_kyoku_idx
                    kyoku_start_scores[ei] = cur_scores
                    kyoku_start_meta[ei] = (env.round_wind, env.oya,
                                            env.honba, env.riichi_sticks)

                if cur_kyoku_idx != prev_kyoku_idx[ei]:
                    prev_kyoku_idx[ei] = cur_kyoku_idx
                    kyoku_start_scores[ei] = list(env.scores())
                    kyoku_start_meta[ei] = (env.round_wind, env.oya,
                                            env.honba, env.riichi_sticks)

                if env.done():
                    # Flush remaining kyoku buffer (final kyoku)
                    if kyoku_buffers[ei]:
                        cur_scores = list(env.scores())
                        ranks = _compute_ranks(cur_scores)
                        rw, oya, honba, rsticks = kyoku_start_meta[ei]
                        all_rewards = self._compute_kyoku_rewards(
                            kyoku_start_scores[ei], cur_scores, rw, oya, honba, rsticks)
                        reward = all_rewards[hero_pids[ei]]
                        completed_kyokus[ei].append(
                            (kyoku_buffers[ei], reward, ranks[hero_pids[ei]]))
                        kyoku_buffers[ei] = []
                    active[ei] = False

        # Compute per-kyoku decayed returns: R * γ^(T-t-1) matching CQL
        transitions = []
        episode_rewards = []
        episode_ranks = []
        kyoku_lengths = []
        kyoku_rewards = []

        for ei in range(self.num_envs):
            ranks = self.envs[ei].ranks()
            rank = ranks[hero_pids[ei]]
            final_reward = 0.0
            if rank == 1: final_reward = 10.0
            elif rank == 2: final_reward = 4.0
            elif rank == 3: final_reward = -4.0
            elif rank == 4: final_reward = -10.0
            episode_rewards.append(final_reward)
            episode_ranks.append(rank)

            for traj, kyoku_reward, kyoku_rank in completed_kyokus[ei]:
                T = len(traj)
                if T == 0:
                    continue

                kyoku_lengths.append(T)
                kyoku_rewards.append(kyoku_reward)

                for t, step in enumerate(traj):
                    decayed_return = kyoku_reward * (self.gamma ** (T - t - 1))
                    step["reward"] = np.array(decayed_return, dtype=np.float32)
                    step["done"] = bool(t == T - 1)
                    step["rank"] = kyoku_rank
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

    def evaluate_episodes(self):
        """
        Runs num_envs hanchan games: hero (player 0, greedy) vs frozen
        baseline (players 1-3, greedy). Returns list of (reward, rank) tuples.
        """
        if not self._compiled_warmup:
            self._warmup_compile()

        obs_dicts = [
            env.reset(scores=[25000, 25000, 25000, 25000],
                      bakaze=0, oya=0, honba=0, kyotaku=0)
            for env in self.envs
        ]
        active = [True] * self.num_envs

        while any(active):
            hero_items = []   # (ei, obs, la)
            opp_items = []    # (ei, pid, obs, la)

            for ei in range(self.num_envs):
                if not active[ei]:
                    continue
                for pid, obs in obs_dicts[ei].items():
                    la = obs.legal_actions()
                    if la:
                        if pid == 0:
                            hero_items.append((ei, obs, la))
                        else:
                            opp_items.append((ei, pid, obs, la))

            if not hero_items and not opp_items:
                for ei in range(self.num_envs):
                    if active[ei] and self.envs[ei].done():
                        active[ei] = False
                break

            env_steps = {ei: {} for ei in range(self.num_envs)}

            # Hero inference (greedy argmax)
            if hero_items:
                feat_list = [self.encoder.encode(obs) for _, obs, _ in hero_items]
                mask_list = [torch.from_numpy(
                    np.frombuffer(obs.mask(), dtype=np.uint8).copy())
                    for _, obs, _ in hero_items]

                feat_batch = torch.stack(feat_list).to(self.device)
                mask_batch = torch.stack(mask_list).to(self.device)

                with torch.no_grad():
                    q_values = self.model(feat_batch)
                    q_values = q_values.masked_fill(~mask_batch.bool(), -1e9)
                    actions = q_values.argmax(dim=1)

                actions_cpu = actions.cpu().numpy()
                for idx, (ei, obs, la) in enumerate(hero_items):
                    action_idx = int(actions_cpu[idx])
                    found_action = obs.find_action(action_idx)
                    if found_action is None:
                        found_action = la[0]
                    env_steps[ei][0] = found_action

            # Opponent inference (frozen baseline, greedy argmax)
            if opp_items:
                feat_list = [self.encoder.encode(obs) for _, _, obs, _ in opp_items]
                mask_list = [torch.from_numpy(
                    np.frombuffer(obs.mask(), dtype=np.uint8).copy())
                    for _, _, obs, _ in opp_items]

                feat_batch = torch.stack(feat_list).to(self.device)
                mask_batch = torch.stack(mask_list).to(self.device)

                with torch.no_grad():
                    q_values = self.baseline_model(feat_batch)
                    q_values = q_values.masked_fill(~mask_batch.bool(), -1e9)
                    actions = q_values.argmax(dim=1)

                actions_cpu = actions.cpu().numpy()
                for idx, (ei, pid, obs, la) in enumerate(opp_items):
                    action_idx = int(actions_cpu[idx])
                    found_action = obs.find_action(action_idx)
                    if found_action is None:
                        found_action = la[0]
                    env_steps[ei][pid] = found_action

            for ei in range(self.num_envs):
                if not active[ei]:
                    continue
                if env_steps[ei]:
                    obs_dicts[ei] = self.envs[ei].step(env_steps[ei])
                if self.envs[ei].done():
                    active[ei] = False

        results = []
        for ei in range(self.num_envs):
            ranks = self.envs[ei].ranks()
            rank = ranks[0]
            reward = 0.0
            if rank == 1: reward = 10.0
            elif rank == 2: reward = 4.0
            elif rank == 3: reward = -4.0
            elif rank == 4: reward = -10.0
            results.append((reward, rank))
        return results
