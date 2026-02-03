import ray
import torch
import numpy as np
from torch.distributions import Categorical
from riichienv import RiichiEnv

from unified_model import UnifiedNetwork
from cql_dataset import ObservationEncoder


@ray.remote
class MahjongWorker:
    def __init__(self, worker_id: int, device: str = "cpu", gamma: float = 0.99):
        torch.set_num_threads(1)
        self.worker_id = worker_id
        self.device = torch.device(device)
        self.env = RiichiEnv(game_mode="4p-red-half")
        self.gamma = gamma

        # Policy Model (Unified) with legacy feature dimensions (46 channels)
        self.model = UnifiedNetwork(num_actions=82).to(self.device)
        self.model.eval()

    def update_weights(self, state_dict):
        """Syncs weights from the Learner."""
        self.model.load_state_dict(state_dict)

        # Check for NaN in weights after loading
        has_nan = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"ERROR: Worker {self.worker_id} received NaN weights in {name}")
                has_nan = True

        if has_nan:
            print(f"WARNING: Worker {self.worker_id} has NaN weights after sync")

    def _encode_obs(self, obs):
        """Encodes Rust observation using legacy features (46, 34)."""
        # Use legacy encoding (46 channels only)
        feat = ObservationEncoder.encode_legacy(obs)
        mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()

        # Move to device
        feat_tensor = feat.to(self.device)
        mask_tensor = torch.from_numpy(mask).to(self.device)

        return feat_tensor, mask_tensor

    def collect_episode(self):
        """
        Runs one full episode of self-play.
        Returns a list of transitions for the learner.
        """
        import time
        t_start = time.time()

        obs_dict = self.env.reset(
            scores=[25000, 25000, 25000, 25000],
            bakaze=0,  # East
            oya=0,     # Player 0
            honba=0,
            kyotaku=0
        )

        # key: player_id, value: list of steps
        episode_buffer = {0: [], 1: [], 2: [], 3: []}
        
        while not self.env.done():
            steps = {}
            any_legal = False
            for pid, obs in obs_dict.items():
                legal_actions = obs.legal_actions()
                if not legal_actions:
                    continue
                
                any_legal = True
                # 1. Observation
                feat_tensor, mask_tensor = self._encode_obs(obs)

                # 2. Policy Step (Stochastic Sampling for AWAC)
                with torch.no_grad():
                    # Add batch dimension
                    feat_batch = feat_tensor.unsqueeze(0)  # (1, 46, 34)
                    logits, _ = self.model(feat_batch)  # (1, 82)
                    logits = logits.masked_fill(mask_tensor.unsqueeze(0) == 0, -1e9)

                    # Check for NaN in logits
                    if torch.isnan(logits).any():
                        print(f"ERROR: NaN detected in worker {self.worker_id} logits")
                        print(f"  logits: {logits}")
                        print(f"  Falling back to random legal action")
                        # Use random legal action as fallback
                        action_idx = np.random.choice([i for i, m in enumerate(mask_tensor.cpu().numpy()) if m > 0])
                        # Store dummy log_prob
                        log_prob_value = -2.0  # Approximate uniform log prob
                    else:
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        action_idx = action.item()
                        log_prob_value = log_prob.cpu().item()

                # 3. Step Environment
                found_action = obs.find_action(action_idx)
                if found_action is None:
                    # Should not happen given mask_fill, but for safety:
                    found_action = legal_actions[0]

                steps[pid] = found_action

                # 4. Store step (for AWAC)
                # Store legacy features as numpy array
                episode_buffer[pid].append({
                    "features": feat_tensor.cpu().numpy(),  # (46, 34)
                    "mask": mask_tensor.cpu().numpy(),
                    "action": action_idx,  # Scalar
                    "log_prob": log_prob_value,  # Scalar (not used in AWAC, but kept for compatibility)
                })

            if not any_legal and not self.env.done():
                # Fatal: No one has legal actions but game is not done
                print(f"FATAL: Total Deadlock - No Legal Actions for any player.")
                print(f"  Phase: {self.env.phase}")
                print(f"  Current Player: {self.env.current_player}")
                print(f"  Hands: {self.env.hands}")
                print(f"  MJAI Log (Last 10): {self.env.mjai_log[-10:]}")
                return []

            obs_dict = self.env.step(steps)
            
        ranks = self.env.ranks()
        transitions = []
        
        # Calculate Rewards
        for pid in range(4):
            rank = ranks[pid]
            
            # Scaled Rank Reward [10.0, 4.0, -4.0, -10.0]
            # RiichiEnv returns 1-based ranks (1, 2, 3, 4)
            obs_reward = 0.0
            if rank == 1: obs_reward = 10.0
            elif rank == 2: obs_reward = 4.0
            elif rank == 3: obs_reward = -4.0
            elif rank == 4: obs_reward = -10.0
            
            traj = episode_buffer[pid]
            T = len(traj)
            
            for t, step in enumerate(traj):
                # MC Return G_t with configured gamma
                decayed_return = obs_reward * (self.gamma ** (T - t - 1))
                
                # We store 'reward' = G_t because PPO Learner will sample this 
                # and use it as target.
                step["reward"] = np.array(decayed_return, dtype=np.float32)
                step["done"] = bool(t == T-1)
                
                transitions.append(step)

        t_end = time.time()
        episode_time = t_end - t_start
        if len(transitions) > 0:
            print(f"Worker {self.worker_id}: Episode took {episode_time:.3f}s, {len(transitions)} transitions, {len(transitions)/episode_time:.1f} trans/s")

        return transitions
