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

    def add(self, transitions):
        """Adds transitions to the buffer.

        Accepts either:
        - dict of pre-batched numpy arrays (from workers, fast path)
        - list of dicts (legacy format)
        """
        if not transitions:
            return

        if isinstance(transitions, dict):
            # Pre-batched format: dict of numpy arrays
            batch_size = len(transitions["action"])
            td = {
                "features": torch.from_numpy(transitions["features"]),
                "mask": torch.from_numpy(transitions["mask"]),
                "action": torch.from_numpy(transitions["action"]),
                "reward": torch.from_numpy(transitions["reward"]),
                "done": torch.from_numpy(transitions["done"]),
            }
            if "rank" in transitions:
                td["rank"] = torch.from_numpy(transitions["rank"])
        else:
            # Legacy list-of-dicts format
            batch_size = len(transitions)
            td = {
                "features": torch.from_numpy(np.stack([t["features"] for t in transitions])),
                "mask": torch.from_numpy(np.stack([t["mask"] for t in transitions])),
                "action": torch.from_numpy(np.array([t["action"] for t in transitions])),
                "reward": torch.from_numpy(np.array([t["reward"] for t in transitions])),
                "done": torch.from_numpy(np.array([t["done"] for t in transitions], dtype=bool)),
            }
            if "rank" in transitions[0]:
                td["rank"] = torch.from_numpy(
                    np.array([t["rank"] for t in transitions], dtype=np.int64))

        batch = TensorDict(td, batch_size=[batch_size])
        self.buffer.extend(batch)

    def sample(self, batch_size=None):
        """Sample a batch from the buffer."""
        if batch_size is None:
            batch_size = self.batch_size
        return self.buffer.sample(batch_size=batch_size).to(self.device)

    def __len__(self):
        return len(self.buffer)


class OnPolicyBuffer:
    """Single-use buffer for on-policy training. Data is used once and discarded."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.data = None
        self.size = 0

    def set_data(self, worker_results):
        """Concatenate transitions from all workers into one dataset.

        Args:
            worker_results: list of dicts, each with numpy array values
                (features, mask, action, reward, done, rank).
        """
        all_arrays = {}
        keys = ["features", "mask", "action", "reward", "done", "rank"]
        for key in keys:
            arrays = [r[key] for r in worker_results if key in r]
            if arrays:
                all_arrays[key] = torch.from_numpy(np.concatenate(arrays))

        self.data = all_arrays
        self.size = len(self.data["action"])

    def iter_batches(self, batch_size):
        """Yield shuffled batches. Each datum used exactly once."""
        perm = torch.randperm(self.size)
        for start in range(0, self.size, batch_size):
            indices = perm[start:start + batch_size]
            yield {k: v[indices].to(self.device) for k, v in self.data.items()}

    def clear(self):
        """Discard all data after training."""
        self.data = None
        self.size = 0
