import torch
import numpy as np
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from tensordict import TensorDict


class GlobalReplayBuffer:
    def __init__(self,
                 actor_capacity: int = 50000,   # Increased: ~250 episodes
                 critic_capacity: int = 1000000, # Increased: ~5000 episodes for better offline-to-online transition
                 batch_size: int = 32,
                 device: str = "cpu"):

        self.device = torch.device(device)
        self.batch_size = batch_size

        # 1. Actor Buffer (FIFO / Sliding Window)
        # Keeps only recent data for PPO (on-policy)
        # Using LazyTensorStorage (In-Memory) for performance
        # Note: batch_size parameter in TensorDictReplayBuffer is NOT the sampling batch size
        # It's ignored when using LazyTensorStorage
        self.actor_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(actor_capacity),
            sampler=SamplerWithoutReplacement(),
        )

        # 2. Critic Buffer (Standard Replay Buffer)
        # Note: Using standard buffer instead of prioritized due to SumSegmentTreeFp32 build issues
        # Keeps long history for CQL (offline + online data)
        self.critic_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(critic_capacity),
            sampler=SamplerWithoutReplacement(),
        )

    def add(self, transitions: list[dict]):
        """
        Adds a list of transitions to both buffers.
        Optimized: Creates a single TensorDict for the whole episode at once.
        Features are legacy (46, 34) numpy arrays.
        """
        if not transitions:
            return

        batch_size = len(transitions)

        # Stack legacy features (46, 34)
        features = np.stack([t["features"] for t in transitions])

        batch_data = {
            "mask": np.stack([t["mask"] for t in transitions]),
            "action": np.array([t["action"] for t in transitions]),
            "reward": np.array([t["reward"] for t in transitions]),
            "done": np.array([t["done"] for t in transitions], dtype=bool),
            "log_prob": np.array([t["log_prob"] for t in transitions]),
        }

        batch = TensorDict({
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(batch_data["mask"]),
            "action": torch.from_numpy(batch_data["action"]),
            "reward": torch.from_numpy(batch_data["reward"]),
            "done": torch.from_numpy(batch_data["done"]),
            "log_prob": torch.from_numpy(batch_data["log_prob"]),
        }, batch_size=[batch_size])

        self.actor_buffer.extend(batch)
        self.critic_buffer.extend(batch)

    
    def update_beta(self, beta: float):
        """Update beta for Importance Sampling"""
        if hasattr(self.critic_buffer, "_sampler") and hasattr(self.critic_buffer._sampler, "_beta"):
             self.critic_buffer._sampler._beta = beta

    def sample_actor(self, batch_size=None):
        """Sample from recent data (PPO)"""
        if batch_size is None:
            batch_size = self.batch_size
        return self.actor_buffer.sample(batch_size=batch_size).to(self.device)

    def sample_critic(self, batch_size=None):
        """
        Sample from historical data (CQL) with Importance Sampling.
        Returns TensorDict with 'index' and '_weight' keys if using prioritized buffer.
        """
        if batch_size is None:
            batch_size = self.batch_size

        batch = self.critic_buffer.sample(batch_size=batch_size).to(self.device)

        # Get actual batch size from sampled data
        actual_batch_size = batch.batch_size[0]

        # Create new TensorDict with additional keys to avoid batch size mismatch
        if "_weight" not in batch.keys() or "index" not in batch.keys():
            batch_dict = {k: v for k, v in batch.items()}
            if "_weight" not in batch.keys():
                batch_dict["_weight"] = torch.ones(actual_batch_size, device=self.device)
            if "index" not in batch.keys():
                batch_dict["index"] = torch.zeros(actual_batch_size, dtype=torch.long, device=self.device)
            batch = TensorDict(batch_dict, batch_size=[actual_batch_size])

        return batch

    def update_priority(self, index, priority):
        """Update priorities for Critic Buffer"""
        if hasattr(self.critic_buffer, "update_priority"):
            self.critic_buffer.update_priority(index, priority)