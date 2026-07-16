"""
Fixed-capacity ring buffer for storing tensor samples for DPMM fitting.

Inspired by TransitionReplayBuffer in
``LIBERO/my_code/train_taskEnc_with_dpmm_mixed_replay.py``.

Design choices:
    - Pre-allocated CPU tensor (no repeated memory allocation).
    - Homogeneous data (all samples have the same dimensionality).
    - CPU storage keeps GPU memory free for model training.
    - When the buffer is full, ``is_full()`` returns True, signalling that
      the DPMM should be updated. After the update, the buffer continues
      to overwrite old samples in a circular fashion.
"""

import torch
import numpy as np


class TensorRingBuffer:
    """Fixed-capacity ring buffer for storing tensor samples for DPMM fitting.

    Attributes:
        capacity: Maximum number of samples the buffer can hold.
        data_dim: Dimensionality of each sample.
        storage: Pre-allocated tensor of shape (capacity, data_dim) on CPU.
        write_pos: Next write position (wraps around).
        num_filled: Number of valid samples currently in the buffer.
    """

    def __init__(self, capacity: int, data_dim: int):
        self.capacity = int(capacity)
        self.data_dim = int(data_dim)
        self.storage = torch.zeros(self.capacity, self.data_dim, dtype=torch.float32)
        self.write_pos = 0
        self.num_filled = 0

    def __len__(self) -> int:
        return self.num_filled

    def add_batch(self, batch: torch.Tensor) -> None:
        """Add a batch of samples to the buffer with circular wraparound.

        Args:
            batch: (N, data_dim) tensor. Will be detached and moved to CPU.
        """
        if self.capacity <= 0:
            return
        batch = batch.detach().to(dtype=torch.float32).cpu()
        if batch.ndim != 2 or batch.shape[1] != self.data_dim:
            raise ValueError(
                f"Expected (N, {self.data_dim}), got {tuple(batch.shape)}"
            )
        n = batch.shape[0]
        for i in range(n):
            self.storage[self.write_pos] = batch[i]
            self.write_pos = (self.write_pos + 1) % self.capacity
            self.num_filled = min(self.num_filled + 1, self.capacity)

    def is_full(self) -> bool:
        """Return True if the buffer has reached its capacity."""
        return self.num_filled >= self.capacity

    def get_all(self) -> torch.Tensor:
        """Return all valid samples as a contiguous tensor."""
        return self.storage[:self.num_filled].clone()

    def sample(self, n: int) -> torch.Tensor:
        """Randomly sample n items from the buffer (without replacement).

        Args:
            n: Number of samples to draw.

        Returns:
            (n, data_dim) tensor.
        """
        if self.num_filled == 0 or n <= 0:
            return torch.empty(0, self.data_dim, dtype=torch.float32)
        n = min(n, self.num_filled)
        idx = torch.randperm(self.num_filled)[:n]
        return self.storage[idx].clone()

    def reset(self) -> None:
        """Clear the buffer."""
        self.write_pos = 0
        self.num_filled = 0
        self.storage.zero_()