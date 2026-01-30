import torch
import numpy as np
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from tensordict import TensorDict


class GlobalReplayBuffer:
    def __init__(self, 
                 actor_capacity: int = 10000, 
                 critic_capacity: int = 100000,
                 batch_size: int = 32,
                 device: str = "cpu"):
        
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # 1. Actor Buffer (FIFO / Sliding Window)
        # Keeps only recent data for PPO
        # Using LazyTensorStorage (In-Memory) for performance
        self.actor_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(actor_capacity),
            sampler=SamplerWithoutReplacement(),
            batch_size=batch_size,
        )
        
        # 2. Critic Buffer (Prioritized)
        # Keeps long history for CQL
        self.critic_buffer = TensorDictPrioritizedReplayBuffer(
            storage=LazyTensorStorage(critic_capacity),
            alpha=0.6,
            beta=0.4,
            batch_size=batch_size,
        )

    def add(self, transitions: list[dict]):
        """
        Adds a list of transitions to both buffers.
        Optimized: Creates a single TensorDict for the whole episode at once.
        """
        if not transitions:
            return

        batch_size = len(transitions)        
        batch_data = {
            "features": np.stack([t["features"] for t in transitions]),
            "mask": np.stack([t["mask"] for t in transitions]),
            "action": np.array([t["action"] for t in transitions]),
            "reward": np.array([t["reward"] for t in transitions]),
            "done": np.array([t["done"] for t in transitions], dtype=bool),
            "policy_version": np.array([t["policy_version"] for t in transitions]),
            "log_prob": np.array([t["log_prob"] for t in transitions]),
        }

        batch = TensorDict({
            "features": torch.from_numpy(batch_data["features"]),
            "mask": torch.from_numpy(batch_data["mask"]),
            "action": torch.from_numpy(batch_data["action"]),
            "reward": torch.from_numpy(batch_data["reward"]),
            "done": torch.from_numpy(batch_data["done"]),
            "policy_version": torch.from_numpy(batch_data["policy_version"]),
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
        
        if "_weight" not in batch.keys():
            batch["_weight"] = torch.ones(batch_size, device=self.device)
            
        if "index" not in batch.keys():
             batch["index"] = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        return batch

    def update_priority(self, index, priority):
        """Update priorities for Critic Buffer"""
        if hasattr(self.critic_buffer, "update_priority"):
            self.critic_buffer.update_priority(index, priority)