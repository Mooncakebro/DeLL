"""
Online DPMM manager for alternating network–DPMM training.

This module provides ``OnlineDPMMManager``, which encapsulates:
    - Two ``TensorRingBuffer`` instances (one per DPMM knowledge space).
    - Two ``BNPModel`` instances (traj DPMM and fuseFeat DPMM).
    - The alternating update schedule: when a buffer fills, the corresponding
      DPMM is fitted (on rank 0), new anchors are broadcast to all ranks, and
      the model's decoders/encoders are hot-swapped.

Usage (inside a training loop)::

    manager = OnlineDPMMManager(config, logdir, device, rank, world_size)
    manager.init_buffers()

    for batch in dataloader:
        # ... forward, loss, backward, optimizer step ...

        # Fill buffers with data from this batch
        manager.fill_traj_buffer(batch['route'][:, :10])
        manager.fill_fusefeat_buffer(model.last_joined_features)

        # Check if either DPMM should update
        manager.maybe_update_traj_dpmm(model)
        manager.maybe_update_fusefeat_dpmm(model)

    manager.save()
"""

import os
import json
import torch
import numpy as np

from dpmm_ring_buffer import TensorRingBuffer

# BNPModel lives in my_dpmm_model/ at the project root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'my_dpmm_model'))
from my_dpmm_model import BNPModel

# Utility functions from team_code/utils.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'team_code'))
from utils import convert_tensor_to_list, purge_invalid_values


class OnlineDPMMManager:
    """Manages two DPMMs and their ring buffers for online alternating training.

    Args:
        config: GlobalConfig instance with online DPMM parameters.
        logdir: Root log directory for this training run.
        device: torch.device for the current process.
        rank: Distributed rank (0 = master).
        world_size: Total number of distributed processes.
    """

    def __init__(self, config, logdir: str, device: torch.device,
                 rank: int = 0, world_size: int = 1):
        self.config = config
        self.logdir = logdir
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.global_step = 0
        self.update_count = 0

        # --- Trajectory DPMM ---
        self.traj_buffer = TensorRingBuffer(
            capacity=config.traj_dpmm_buffer_size,
            data_dim=config.predict_checkpoint_len * 2,  # 10 * 2 = 20
        )
        traj_save_dir = os.path.join(logdir, config.dpmm_log_dir, 'traj')
        self.traj_dpmm = BNPModel(
            save_dir=traj_save_dir,
            gamma0=config.traj_dpmm_gamma0,
            num_lap=config.traj_dpmm_num_lap,
            sF=config.traj_dpmm_sF,
        )

        # --- FuseFeat DPMM ---
        fusefeat_dim = config.gru_input_size * (config.predict_checkpoint_len + 1)  # 256 * 11 = 2816
        self.fusefeat_buffer = TensorRingBuffer(
            capacity=config.fusefeat_dpmm_buffer_size,
            data_dim=fusefeat_dim,
        )
        fusefeat_save_dir = os.path.join(logdir, config.dpmm_log_dir, 'fusefeat')
        self.fusefeat_dpmm = BNPModel(
            save_dir=fusefeat_save_dir,
            gamma0=config.fusefeat_dpmm_gamma0,
            num_lap=config.fusefeat_dpmm_num_lap,
            sF=config.fusefeat_dpmm_sF,
        )

        # Create log directories
        if rank == 0:
            os.makedirs(os.path.join(logdir, config.dpmm_log_dir, 'traj'), exist_ok=True)
            os.makedirs(os.path.join(logdir, config.dpmm_log_dir, 'fusefeat'), exist_ok=True)

    # ------------------------------------------------------------------
    # Buffer filling
    # ------------------------------------------------------------------

    def fill_traj_buffer(self, traj_batch: torch.Tensor) -> None:
        """Add ground-truth trajectory data to the traj buffer.

        Args:
            traj_batch: (B, predict_checkpoint_len, 2) or (B, 20) tensor.
        """
        if traj_batch is None:
            return
        traj_flat = traj_batch.detach().reshape(traj_batch.shape[0], -1)
        self.traj_buffer.add_batch(traj_flat)

    def fill_fusefeat_buffer(self, fusefeat_batch: torch.Tensor) -> None:
        """Add fused-feature data to the fuseFeat buffer.

        Args:
            fusefeat_batch: (B, 11, 256) or (B, 2816) tensor.
        """
        if fusefeat_batch is None:
            return
        fusefeat_flat = fusefeat_batch.detach().reshape(fusefeat_batch.shape[0], -1)
        self.fusefeat_buffer.add_batch(fusefeat_flat)

    def step(self) -> None:
        """Increment the global step counter."""
        self.global_step += 1

    # ------------------------------------------------------------------
    # Update scheduling
    # ------------------------------------------------------------------

    def _should_update(self, buffer, fit_min_samples: int) -> bool:
        """Check if a DPMM should be updated based on the schedule."""
        if self.global_step < self.config.dpmm_update_start_step:
            return False
        # Periodic update mode
        if self.config.dpmm_update_freq_steps > 0:
            if self.global_step % self.config.dpmm_update_freq_steps == 0:
                return buffer.num_filled >= fit_min_samples
            return False
        # Buffer-full mode (default)
        return buffer.is_full()

    def maybe_update_traj_dpmm(self, model) -> bool:
        """Check if the traj DPMM should update; if so, fit and hot-swap anchors.

        Args:
            model: The LidarCenterNet model (or DDP-wrapped).

        Returns:
            True if an update was performed, False otherwise.
        """
        if not self._should_update(self.traj_buffer, self.config.traj_dpmm_fit_min_samples):
            return False
        self._fit_and_swap(
            dpmm=self.traj_dpmm,
            buffer=self.traj_buffer,
            replay_ratio=self.config.traj_dpmm_replay_ratio,
            max_replay=self.config.traj_dpmm_max_replay_samples,
            tag='traj',
            anchor_dim_slice=20,  # traj anchors are 20-dim
            update_fn=model.update_traj_anchors,
        )
        self.update_count += 1
        return True

    def maybe_update_fusefeat_dpmm(self, model) -> bool:
        """Check if the fuseFeat DPMM should update; if so, fit and hot-swap.

        Args:
            model: The LidarCenterNet model (or DDP-wrapped).

        Returns:
            True if an update was performed, False otherwise.
        """
        if not self._should_update(self.fusefeat_buffer, self.config.fusefeat_dpmm_fit_min_samples):
            return False
        self._fit_and_swap(
            dpmm=self.fusefeat_dpmm,
            buffer=self.fusefeat_buffer,
            replay_ratio=self.config.fusefeat_dpmm_replay_ratio,
            max_replay=self.config.fusefeat_dpmm_max_replay_samples,
            tag='fusefeat',
            anchor_dim_slice=None,  # fuseFeat anchors use full mu
            update_fn=model.update_fusefeat_anchors,
        )
        self.update_count += 1
        return True

    # ------------------------------------------------------------------
    # Core fit + broadcast + hot-swap
    # ------------------------------------------------------------------

    def _fit_and_swap(self, dpmm, buffer, replay_ratio, max_replay,
                      tag, anchor_dim_slice, update_fn) -> None:
        """Fit a DPMM on rank 0, broadcast new anchors, and hot-swap into model.

        The buffer is reset AFTER the fit so it doesn't re-trigger on the
        very next step.

        Args:
            dpmm: BNPModel instance.
            buffer: TensorRingBuffer with data to fit on.
            replay_ratio: Ratio of replay samples to mix with buffer data.
            max_replay: Maximum number of replay samples.
            tag: 'traj' or 'fusefeat' (for logging).
            anchor_dim_slice: If not None, slice mu to first N dims (traj=20).
            update_fn: Model method to call with new anchors.
        """
        buffer_data = buffer.get_all()  # (N, dim) on CPU

        # Mix with replay samples from existing DPMM
        fit_data = self._mix_replay(dpmm, buffer_data, replay_ratio, max_replay)
        fit_data = purge_invalid_values(fit_data, f"{tag}_dpmm_fit")

        if fit_data.shape[0] == 0:
            print(f"[OnlineDPMM] Skipping {tag} update: no valid data")
            return

        # Fit on rank 0 only (bnpy is CPU-only, not distributed)
        new_anchors = None
        if self.rank == 0:
            print(f"[OnlineDPMM] Fitting {tag} DPMM with {fit_data.shape[0]} samples "
                  f"(buffer={buffer_data.shape[0]}, step={self.global_step})")
            dpmm.fit(fit_data)
            new_anchors = self._extract_anchors(dpmm, anchor_dim_slice)
            self._log_dpmm_state(dpmm, tag)
            print(f"[OnlineDPMM] {tag} DPMM updated: {new_anchors.shape[0]} clusters")

        # Broadcast anchors from rank 0 to all ranks
        new_anchors = self._broadcast_anchors(new_anchors)

        # Hot-swap anchors into model
        update_fn(new_anchors.to(self.device))

        # Reset buffer so it doesn't immediately re-trigger
        buffer.reset()

    def _mix_replay(self, dpmm, buffer_data, replay_ratio, max_replay):
        """Mix buffer data with replay samples from the existing DPMM.

        This follows the LEGION pattern: mixing replay prevents the DPMM
        from forgetting old clusters when new data arrives.
        """
        if len(dpmm.components) == 0:
            return buffer_data

        num_replay = int(len(buffer_data) * replay_ratio)
        num_replay = min(num_replay, max_replay)
        if num_replay <= 0:
            return buffer_data

        replay = dpmm.sample_all(num_samples=num_replay)
        replay = replay.cpu().to(dtype=torch.float32)
        return torch.cat([replay, buffer_data], dim=0)

    def _extract_anchors(self, dpmm, dim_slice):
        """Extract anchor means from DPMM clusters.

        Args:
            dpmm: BNPModel with current clusters.
            dim_slice: If not None, slice mu to first dim_slice dims.

        Returns:
            (num_anchors, dim) tensor of anchor means.
        """
        clusters = dpmm.get_current_cluster_list()
        if len(clusters) == 0:
            return torch.empty(0, 1, dtype=torch.float32)

        mus = []
        for c in clusters:
            mu = torch.tensor(c['mu'], dtype=torch.float32)
            if dim_slice is not None:
                mu = mu[:dim_slice]
            mus.append(mu)
        return torch.stack(mus, dim=0)

    def _broadcast_anchors(self, anchors):
        """Broadcast anchors from rank 0 to all ranks.

        Handles variable K (number of clusters) by broadcasting shape first.
        """
        if self.world_size <= 1:
            if anchors is None:
                raise RuntimeError("Rank 0 produced no anchors")
            return anchors

        # Broadcast shape (2,) from rank 0
        if self.rank == 0:
            if anchors is None:
                # Signal no anchors with shape (0, 0)
                shape = torch.tensor([0, 0], dtype=torch.long)
            else:
                shape = torch.tensor(anchors.shape, dtype=torch.long)
        else:
            shape = torch.empty(2, dtype=torch.long)

        torch.distributed.broadcast(shape, src=0)

        # If shape is (0, 0), no anchors to swap
        if shape[0] == 0 or shape[1] == 0:
            return torch.empty(0, 0, dtype=torch.float32)

        # Broadcast data
        if self.rank != 0:
            anchors = torch.empty(*shape, dtype=torch.float32)
        torch.distributed.broadcast(anchors, src=0)

        return anchors

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_dpmm_state(self, dpmm, tag):
        """Save tracked clusters as JSON for reproducibility."""
        tracked = sorted(
            [{'cluster_id': c['cluster_id'], 'mu': c['mu'], 'var': c['var']}
             for c in dpmm.get_current_cluster_list()],
            key=lambda x: x['cluster_id'],
        )
        tracked = convert_tensor_to_list(tracked)

        path = os.path.join(
            self.logdir, self.config.dpmm_log_dir, tag,
            f"step{self.global_step:06d}-tracked_clusters.json",
        )
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tracked, f, indent=2)

        # Also save components
        components = sorted(dpmm.components, key=lambda x: x['k'])
        comp_path = os.path.join(
            self.logdir, self.config.dpmm_log_dir, tag,
            f"step{self.global_step:06d}-components.json",
        )
        with open(comp_path, 'w', encoding='utf-8') as f:
            json.dump(components, f, indent=2)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self):
        """Save both DPMM models (rank 0 only)."""
        if self.rank != 0:
            return
        traj_dir = os.path.join(self.logdir, self.config.dpmm_log_dir, 'traj')
        fusefeat_dir = os.path.join(self.logdir, self.config.dpmm_log_dir, 'fusefeat')
        self.traj_dpmm.save_model(traj_dir)
        self.fusefeat_dpmm.save_model(fusefeat_dir)
        print(f"[OnlineDPMM] Saved DPMM models to {self.config.dpmm_log_dir}/")

    def load(self, traj_path=None, fusefeat_path=None):
        """Load pre-trained DPMM models (optional warm start).

        Args:
            traj_path: Path to saved traj DPMM directory.
            fusefeat_path: Path to saved fuseFeat DPMM directory.
        """
        if traj_path and os.path.isdir(traj_path):
            self.traj_dpmm.load_model(traj_path)
            print(f"[OnlineDPMM] Loaded traj DPMM from {traj_path}")
        if fusefeat_path and os.path.isdir(fusefeat_path):
            self.fusefeat_dpmm.load_model(fusefeat_path)
            print(f"[OnlineDPMM] Loaded fuseFeat DPMM from {fusefeat_path}")