import random
import time

import ray
import torch
import numpy as np
from loguru import logger
from riichienv import RiichiEnv

from riichienv_ml.config import import_class
from riichienv_ml.models.grp_model import RewardPredictor


@ray.remote
class PPOWorker:
    """
    PPO rollout worker. Collects trajectories with log_probs and values
    for on-policy training. Uses hero (random pid per env) with current policy
    and opponents with a frozen baseline model.

    Episodes are split at kyoku (round) boundaries. Each kyoku produces an
    independent trajectory with its own GAE computation. Per-kyoku rewards
    are computed using the GRP (GlobalRewardPredictor) model, matching the
    reward structure used during CQL pretraining.
    """
    def __init__(self, worker_id: int, device: str = "cpu",
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 num_envs: int = 16,
                 model_config: dict | None = None,
                 model_class: str = "riichienv_ml.models.actor_critic.ActorCriticNetwork",
                 encoder_class: str = "riichienv_ml.data.cql_dataset.ObservationEncoder",
                 grp_model: str | None = None,
                 pts_weight: list[float] | None = None):
        torch.set_num_threads(1)
        self.worker_id = worker_id
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.envs = [RiichiEnv(game_mode="4p-red-half") for _ in range(num_envs)]
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        mc = model_config or {}
        ModelClass = import_class(model_class)

        # Hero model (current policy, updated each step)
        self.model = ModelClass(**mc).to(self.device)
        self.model.eval()

        # Baseline model (frozen opponents)
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
        """Apply state dict by copying parameter data directly.

        torch.compile caches compiled graphs and load_state_dict() may not
        properly invalidate the cache, causing the compiled model to keep
        using old weights. Copying .data directly avoids this issue.
        """
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        for name, param in target.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name])
        for name, buf in target.named_buffers():
            if name in state_dict:
                buf.data.copy_(state_dict[name])

    def update_weights(self, state_dict):
        self._apply_state_dict(self.model, state_dict)

    def update_baseline_weights(self, state_dict):
        self._apply_state_dict(self.baseline_model, state_dict)

    def _warmup_compile(self):
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
        Hero (random pid per env) uses current policy; opponents use frozen baseline.

        Each hanchan is split into per-kyoku trajectories. At each kyoku boundary,
        GRP computes the reward for that kyoku. GAE is computed independently
        per kyoku (~30-50 hero steps), matching the CQL pretraining reward structure.
        """
        t_start = time.time()

        if not self._compiled_warmup:
            self._warmup_compile()

        obs_dicts = [
            env.reset(scores=[25000, 25000, 25000, 25000],
                      round_wind=0, oya=0, honba=0, kyotaku=0)
            for env in self.envs
        ]

        active = [True] * self.num_envs
        # Randomly assign hero pid per env (player id has game meaning: oya, wind)
        hero_pids = [random.randint(0, 3) for _ in range(self.num_envs)]

        # Per-kyoku hero trajectory buffer (reset at each kyoku boundary)
        kyoku_buffers = [[] for _ in range(self.num_envs)]
        # Completed kyoku trajectories: list of (trajectory, reward) per env
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

            # === Hero inference (current policy, with log_probs/values) ===
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
                    logits, values = self.model(feat_batch)
                    mask_bool = mask_batch.bool()
                    logits = logits.masked_fill(~mask_bool, -1e9)

                    probs = torch.softmax(logits, dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(-1)
                    log_probs_all = torch.log_softmax(logits, dim=-1)
                    log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

                actions_cpu = actions.cpu().numpy()
                log_probs_cpu = log_probs.cpu().numpy()
                values_cpu = values.cpu().numpy()
                feat_cpu = feat_batch.cpu().numpy()
                mask_cpu = mask_batch.cpu().numpy()

                for idx, (ei, obs, la) in enumerate(hero_items):
                    action_idx = int(actions_cpu[idx])
                    kyoku_buffers[ei].append({
                        "features": feat_cpu[idx],
                        "mask": mask_cpu[idx],
                        "action": action_idx,
                        "log_prob": float(log_probs_cpu[idx]),
                        "value": float(values_cpu[idx]),
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
                    output = self.baseline_model(feat_batch)
                    if isinstance(output, tuple):
                        opp_logits = output[0]
                    else:
                        opp_logits = output
                    opp_logits = opp_logits.masked_fill(~mask_batch.bool(), -1e9)
                    opp_actions = opp_logits.argmax(dim=1)

                opp_actions_cpu = opp_actions.cpu().numpy()
                for idx, (ei, pid, obs, la) in enumerate(opp_items):
                    action_idx = int(opp_actions_cpu[idx])
                    found_action = obs.find_action(action_idx)
                    if found_action is None:
                        found_action = la[0]
                    env_steps[ei][pid] = found_action

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
                    rw, oya, honba, rsticks = kyoku_start_meta[ei]
                    # Compute absolute GRP rewards for all 4 players (matches CQL offline)
                    all_rewards = self._compute_kyoku_rewards(
                        kyoku_start_scores[ei], cur_scores, rw, oya, honba, rsticks)
                    reward = all_rewards[hero_pids[ei]]
                    completed_kyokus[ei].append((kyoku_buffers[ei], reward))
                    kyoku_buffers[ei] = []
                    kyoku_count[ei] += 1
                    # Debug: log first 3 kyokus of env 0
                    if ei == 0 and kyoku_count[ei] <= 3:
                        logger.debug(f"env0 kyoku {kyoku_count[ei]}: "
                                     f"hero={hero_pids[ei]} reward={reward:.3f} "
                                     f"all={[f'{r:.3f}' for r in all_rewards]}")
                    # Update tracking for next kyoku
                    prev_kyoku_idx[ei] = cur_kyoku_idx
                    kyoku_start_scores[ei] = cur_scores
                    kyoku_start_meta[ei] = (env.round_wind, env.oya,
                                            env.honba, env.riichi_sticks)

                if env.done():
                    # Flush remaining kyoku buffer (final kyoku of the hanchan)
                    if kyoku_buffers[ei]:
                        cur_scores = list(env.scores())
                        rw, oya, honba, rsticks = kyoku_start_meta[ei]
                        all_rewards = self._compute_kyoku_rewards(
                            kyoku_start_scores[ei], cur_scores, rw, oya, honba, rsticks)
                        reward = all_rewards[hero_pids[ei]]
                        completed_kyokus[ei].append((kyoku_buffers[ei], reward))
                        kyoku_buffers[ei] = []
                    active[ei] = False

        # Compute GAE advantages per kyoku
        # Collect into lists for efficient batching (reduces Ray serialization overhead)
        feat_list = []
        mask_list = []
        action_list = []
        log_prob_list = []
        advantage_list = []
        return_list = []
        episode_rewards = []
        episode_ranks = []
        kyoku_lengths = []
        kyoku_rewards = []
        value_predictions = []

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

            for traj, kyoku_reward in completed_kyokus[ei]:
                T = len(traj)
                if T == 0:
                    continue

                kyoku_lengths.append(T)
                kyoku_rewards.append(kyoku_reward)

                # Compute GAE per kyoku
                values = [step["value"] for step in traj]
                value_predictions.extend(values)
                advantages = [0.0] * T
                returns = [0.0] * T

                gae = 0.0
                for t in reversed(range(T)):
                    if t == T - 1:
                        reward = kyoku_reward
                        next_value = 0.0  # terminal (end of kyoku)
                    else:
                        reward = 0.0
                        next_value = values[t + 1]

                    delta = reward + self.gamma * next_value - values[t]
                    gae = delta + self.gamma * self.gae_lambda * gae
                    advantages[t] = gae
                    returns[t] = gae + values[t]

                for t, step in enumerate(traj):
                    feat_list.append(step["features"])
                    mask_list.append(step["mask"])
                    action_list.append(step["action"])
                    log_prob_list.append(step["log_prob"])
                    advantage_list.append(advantages[t])
                    return_list.append(returns[t])

        # Pre-batch into contiguous arrays (much faster Ray serialization)
        n_transitions = len(feat_list)
        if n_transitions > 0:
            transitions = {
                "features": np.stack(feat_list),
                "mask": np.stack(mask_list),
                "action": np.array(action_list, dtype=np.int64),
                "log_prob": np.array(log_prob_list, dtype=np.float32),
                "advantage": np.array(advantage_list, dtype=np.float32),
                "return": np.array(return_list, dtype=np.float32),
            }
        else:
            transitions = {}

        # Worker-level stats
        stats = {}
        if episode_rewards:
            stats["reward_mean"] = float(np.mean(episode_rewards))
            stats["reward_std"] = float(np.std(episode_rewards))
            stats["rank_mean"] = float(np.mean(episode_ranks))
            stats["value_pred_mean"] = float(np.mean(value_predictions)) if value_predictions else 0.0
            stats["value_pred_std"] = float(np.std(value_predictions)) if value_predictions else 0.0
        if kyoku_lengths:
            stats["kyoku_length_mean"] = float(np.mean(kyoku_lengths))
            stats["kyoku_reward_mean"] = float(np.mean(kyoku_rewards))
            stats["kyoku_reward_std"] = float(np.std(kyoku_rewards))
            stats["kyokus_per_hanchan"] = float(len(kyoku_lengths) / self.num_envs)

        t_end = time.time()
        episode_time = t_end - t_start
        if n_transitions > 0:
            logger.info(f"Worker {self.worker_id}: {self.num_envs} hanchan, "
                        f"{len(kyoku_lengths)} kyokus took {episode_time:.3f}s, "
                        f"{n_transitions} transitions, {n_transitions/episode_time:.1f} trans/s")

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
                      round_wind=0, oya=0, honba=0, kyotaku=0)
            for env in self.envs
        ]
        active = [True] * self.num_envs

        while any(active):
            hero_items = []
            opp_items = []

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

            # Hero inference (greedy argmax on logits)
            if hero_items:
                feat_list = [self.encoder.encode(obs) for _, obs, _ in hero_items]
                mask_list = [torch.from_numpy(
                    np.frombuffer(obs.mask(), dtype=np.uint8).copy())
                    for _, obs, _ in hero_items]

                feat_batch = torch.stack(feat_list).to(self.device)
                mask_batch = torch.stack(mask_list).to(self.device)

                with torch.no_grad():
                    logits, _ = self.model(feat_batch)
                    logits = logits.masked_fill(~mask_batch.bool(), -1e9)
                    actions = logits.argmax(dim=1)

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
                    output = self.baseline_model(feat_batch)
                    if isinstance(output, tuple):
                        opp_logits = output[0]
                    else:
                        opp_logits = output
                    opp_logits = opp_logits.masked_fill(~mask_batch.bool(), -1e9)
                    opp_actions = opp_logits.argmax(dim=1)

                opp_actions_cpu = opp_actions.cpu().numpy()
                for idx, (ei, pid, obs, la) in enumerate(opp_items):
                    action_idx = int(opp_actions_cpu[idx])
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
