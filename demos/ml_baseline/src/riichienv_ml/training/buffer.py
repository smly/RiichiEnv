import torch
import numpy as np
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from tensordict import TensorDict


class GlobalReplayBuffer:
    def __init__(self,
                 capacity: int = 1000000,
                 batch_size: int = 32,
                 device: str = "cpu"):

        self.device = torch.device(device)
        self.batch_size = batch_size

        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(capacity),
            sampler=SamplerWithoutReplacement(),
        )

    def add(self, transitions: list[dict]):
        """Adds a list of transitions to the buffer."""
        if not transitions:
            return

        batch_size = len(transitions)

        features = np.stack([t["features"] for t in transitions])

        batch_data = {
            "mask": np.stack([t["mask"] for t in transitions]),
            "action": np.array([t["action"] for t in transitions]),
            "reward": np.array([t["reward"] for t in transitions]),
            "done": np.array([t["done"] for t in transitions], dtype=bool),
        }

        batch = TensorDict({
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(batch_data["mask"]),
            "action": torch.from_numpy(batch_data["action"]),
            "reward": torch.from_numpy(batch_data["reward"]),
            "done": torch.from_numpy(batch_data["done"]),
        }, batch_size=[batch_size])

        self.buffer.extend(batch)

    def sample(self, batch_size=None):
        """Sample a batch from the buffer."""
        if batch_size is None:
            batch_size = self.batch_size
        return self.buffer.sample(batch_size=batch_size).to(self.device)

    def __len__(self):
        return len(self.buffer)
