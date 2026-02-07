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
                 model_config: dict | None = None,
                 model_class: str = "riichienv_ml.models.cql_model.QNetwork",
                 encoder_class: str = "riichienv_ml.data.cql_dataset.ObservationEncoder"):
        torch.set_num_threads(1)
        self.worker_id = worker_id
        self.device = torch.device(device)
        self.env = RiichiEnv(game_mode="4p-red-half")
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
        self.encoder = import_class(encoder_class)

    def update_weights(self, state_dict):
        """Syncs weights from the Learner."""
        self.model.load_state_dict(state_dict)

    def set_epsilon(self, epsilon: float):
        """Updates epsilon for epsilon-greedy exploration."""
        self.epsilon = epsilon

    def set_boltzmann_temp(self, temp: float):
        """Updates Boltzmann temperature."""
        self.boltzmann_temp = temp

    def _encode_obs(self, obs):
        """Encodes Rust observation."""
        feat = self.encoder.encode(obs)
        mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()

        feat_tensor = feat.to(self.device)
        mask_tensor = torch.from_numpy(mask).to(self.device)

        return feat_tensor, mask_tensor

    def _select_action(self, q_values, mask_bool, mask_tensor):
        """Selects an action based on exploration strategy."""
        if torch.isnan(q_values).any():
            legal_indices = [i for i, m in enumerate(mask_tensor.cpu().numpy()) if m > 0]
            return np.random.choice(legal_indices)

        if self.exploration == "boltzmann":
            if np.random.random() < self.boltzmann_epsilon:
                logits = q_values / self.boltzmann_temp
                logits = logits.masked_fill(~mask_bool, -torch.inf)
                return sample_top_p(logits.squeeze(0), self.top_p).item()
            else:
                return q_values.argmax(dim=1).item()
        else:  # epsilon_greedy
            if np.random.random() < self.epsilon:
                legal_indices = [i for i, m in enumerate(mask_tensor.cpu().numpy()) if m > 0]
                return np.random.choice(legal_indices)
            else:
                return q_values.argmax(dim=1).item()

    def collect_episode(self):
        """
        Runs one full episode of self-play with configured exploration strategy.
        """
        import time
        t_start = time.time()

        obs_dict = self.env.reset(
            scores=[25000, 25000, 25000, 25000],
            bakaze=0,
            oya=0,
            honba=0,
            kyotaku=0
        )

        episode_buffer = {0: [], 1: [], 2: [], 3: []}

        while not self.env.done():
            steps = {}
            any_legal = False
            for pid, obs in obs_dict.items():
                legal_actions = obs.legal_actions()
                if not legal_actions:
                    continue

                any_legal = True
                feat_tensor, mask_tensor = self._encode_obs(obs)

                with torch.no_grad():
                    feat_batch = feat_tensor.unsqueeze(0)
                    q_values = self.model(feat_batch)
                    mask_bool = mask_tensor.unsqueeze(0).bool()
                    q_values = q_values.masked_fill(~mask_bool, -torch.inf)
                    action_idx = self._select_action(q_values, mask_bool, mask_tensor)

                found_action = obs.find_action(action_idx)
                if found_action is None:
                    found_action = legal_actions[0]

                steps[pid] = found_action

                episode_buffer[pid].append({
                    "features": feat_tensor.cpu().numpy(),
                    "mask": mask_tensor.cpu().numpy(),
                    "action": action_idx,
                })

            if not any_legal and not self.env.done():
                print(f"FATAL: Total Deadlock - No Legal Actions for any player.")
                print(f"  Phase: {self.env.phase}")
                print(f"  Current Player: {self.env.current_player}")
                print(f"  Hands: {self.env.hands}")
                print(f"  MJAI Log (Last 10): {self.env.mjai_log[-10:]}")
                return []

            obs_dict = self.env.step(steps)

        ranks = self.env.ranks()
        transitions = []

        for pid in range(4):
            rank = ranks[pid]

            obs_reward = 0.0
            if rank == 1: obs_reward = 10.0
            elif rank == 2: obs_reward = 4.0
            elif rank == 3: obs_reward = -4.0
            elif rank == 4: obs_reward = -10.0

            traj = episode_buffer[pid]
            T = len(traj)

            for t, step in enumerate(traj):
                decayed_return = obs_reward * (self.gamma ** (T - t - 1))
                step["reward"] = np.array(decayed_return, dtype=np.float32)
                step["done"] = bool(t == T - 1)
                transitions.append(step)

        t_end = time.time()
        episode_time = t_end - t_start
        if len(transitions) > 0:
            print(f"Worker {self.worker_id}: Episode took {episode_time:.3f}s, {len(transitions)} transitions, {len(transitions)/episode_time:.1f} trans/s")

        return transitions
