import random
import time

import ray
import torch
import numpy as np
from loguru import logger
from riichienv import RiichiEnv

from riichienv_ml.config import import_class


@ray.remote
class PPOWorker:
    """
    PPO rollout worker. Collects trajectories with log_probs and values
    for on-policy training. Uses hero (random pid per env) with current policy
    and opponents with a frozen baseline model.
    """
    def __init__(self, worker_id: int, device: str = "cpu",
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 num_envs: int = 16,
                 model_config: dict | None = None,
                 model_class: str = "riichienv_ml.models.actor_critic.ActorCriticNetwork",
                 encoder_class: str = "riichienv_ml.data.cql_dataset.ObservationEncoder"):
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

    def update_weights(self, state_dict):
        target = self.model
        if hasattr(target, "_orig_mod"):
            target = target._orig_mod
        target.load_state_dict(state_dict)

    def update_baseline_weights(self, state_dict):
        target = self.baseline_model
        if hasattr(target, "_orig_mod"):
            target = target._orig_mod
        target.load_state_dict(state_dict)

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

    def collect_episodes(self):
        """
        Runs N episodes in parallel using batched inference.
        Hero (random pid per env) uses current policy; opponents use frozen baseline.
        Only hero trajectories are returned for training.
        """
        t_start = time.time()

        if not self._compiled_warmup:
            self._warmup_compile()

        obs_dicts = [
            env.reset(scores=[25000, 25000, 25000, 25000],
                      bakaze=0, oya=0, honba=0, kyotaku=0)
            for env in self.envs
        ]

        active = [True] * self.num_envs
        # Randomly assign hero pid per env (player id has game meaning: oya, wind)
        hero_pids = [random.randint(0, 3) for _ in range(self.num_envs)]
        hero_buffers = [[] for _ in range(self.num_envs)]

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
                    hero_buffers[ei].append({
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
                if self.envs[ei].done():
                    active[ei] = False

        # Compute GAE advantages for hero only
        transitions = []
        episode_rewards = []
        episode_ranks = []
        episode_lengths = []
        value_predictions = []

        for ei in range(self.num_envs):
            ranks = self.envs[ei].ranks()
            rank = ranks[hero_pids[ei]]  # hero rank
            final_reward = 0.0
            if rank == 1: final_reward = 10.0
            elif rank == 2: final_reward = 4.0
            elif rank == 3: final_reward = -4.0
            elif rank == 4: final_reward = -10.0

            traj = hero_buffers[ei]
            T = len(traj)
            if T == 0:
                continue

            episode_rewards.append(final_reward)
            episode_ranks.append(rank)
            episode_lengths.append(T)

            # Compute GAE
            values = [step["value"] for step in traj]
            value_predictions.extend(values)
            advantages = [0.0] * T
            returns = [0.0] * T

            gae = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    reward = final_reward
                    next_value = 0.0  # terminal
                else:
                    reward = 0.0
                    next_value = values[t + 1]

                delta = reward + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages[t] = gae
                returns[t] = gae + values[t]

            for t, step in enumerate(traj):
                transitions.append({
                    "features": step["features"],
                    "mask": step["mask"],
                    "action": step["action"],
                    "log_prob": step["log_prob"],
                    "advantage": np.float32(advantages[t]),
                    "return": np.float32(returns[t]),
                })

        # Worker-level stats
        stats = {}
        if episode_rewards:
            stats["reward_mean"] = float(np.mean(episode_rewards))
            stats["reward_std"] = float(np.std(episode_rewards))
            stats["rank_mean"] = float(np.mean(episode_ranks))
            stats["episode_length_mean"] = float(np.mean(episode_lengths))
            stats["value_pred_mean"] = float(np.mean(value_predictions))
            stats["value_pred_std"] = float(np.std(value_predictions))

        t_end = time.time()
        episode_time = t_end - t_start
        if transitions:
            logger.info(f"Worker {self.worker_id}: {self.num_envs} episodes took {episode_time:.3f}s, "
                        f"{len(transitions)} transitions, {len(transitions)/episode_time:.1f} trans/s")

        return transitions, stats
