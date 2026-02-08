import time

import ray
import torch
import numpy as np
from riichienv import RiichiEnv

from riichienv_ml.config import import_class


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
                 encoder_class: str = "riichienv_ml.data.cql_dataset.ObservationEncoder"):
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
        mc = {}
        # Infer in_channels from model
        target = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        in_ch = target.backbone.conv_in.in_channels
        dummy = torch.randn(1, in_ch, 34, device=self.device)
        with torch.no_grad():
            self.model(dummy)
        self._compiled_warmup = True

    def collect_episodes(self):
        """
        Runs N episodes in parallel using batched inference.
        All environments are stepped in lockstep; observations are batched
        into a single forward pass for GPU efficiency.
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
        all_buffers = [{0: [], 1: [], 2: [], 3: []} for _ in range(self.num_envs)]

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
                all_buffers[ei][pid].append({
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
                if self.envs[ei].done():
                    active[ei] = False

        # Compute returns for all episodes
        transitions = []
        for ei in range(self.num_envs):
            ranks = self.envs[ei].ranks()
            for pid in range(4):
                rank = ranks[pid]
                obs_reward = 0.0
                if rank == 1: obs_reward = 10.0
                elif rank == 2: obs_reward = 4.0
                elif rank == 3: obs_reward = -4.0
                elif rank == 4: obs_reward = -10.0

                traj = all_buffers[ei][pid]
                T = len(traj)
                for t, step in enumerate(traj):
                    decayed_return = obs_reward * (self.gamma ** (T - t - 1))
                    step["reward"] = np.array(decayed_return, dtype=np.float32)
                    step["done"] = bool(t == T - 1)
                    transitions.append(step)

        t_end = time.time()
        episode_time = t_end - t_start
        if transitions:
            print(f"Worker {self.worker_id}: {self.num_envs} episodes took {episode_time:.3f}s, "
                  f"{len(transitions)} transitions, {len(transitions)/episode_time:.1f} trans/s")

        return transitions
