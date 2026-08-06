"""
Online DPMM manager for task-boundary DPMM preparation.

This module provides ``OnlineDPMMManager``, which encapsulates:
    - Two ``TensorRingBuffer`` instances (one per DPMM knowledge space).
    - Two ``BNPModel`` instances (traj DPMM and fuseFeat DPMM).
    - A bounded-memory streaming update schedule used before each task.
    - Final anchor broadcast and hot-swap after every task dataset has been
      fully consumed, before network optimization begins.

Usage (inside a training loop)::

    manager = OnlineDPMMManager(config, logdir, device, rank, world_size)
    manager.begin_task('Emergency_Brake')
    for batch in dataloader:
        manager.consume_task_batch(batch['route'][:, :10], joined_features)
    manager.finalize_task(model)
    manager.save()
"""

import os
import json
import torch

from dpmm_ring_buffer import TensorRingBuffer

# BNPModel lives in my_dpmm_model/ at the project root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'my_dpmm_model'))
from my_dpmm_model import BNPModel

# Utility functions from team_code/utils.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'team_code'))
from utils import convert_tensor_to_list, purge_invalid_values


class OnlineDPMMManager:
    """Manages two DPMMs and bounded buffers at task boundaries.

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
        self.current_task = None
        self.tag_update_counts = {'traj': 0, 'fusefeat': 0}
        self.anchors_ready_for_task = False
        self.enable_traj = bool(config.use_controller_input_prediction)
        self.enable_fusefeat = bool(config.use_prior_fuseFeat)

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

    def step(self) -> None:
        """Increment the global step counter."""
        self.global_step += 1

    # ------------------------------------------------------------------
    # Task-boundary streaming preparation
    # ------------------------------------------------------------------

    def begin_task(self, task_name: str) -> None:
        """Start a fresh streaming pass for one task without resetting DPMMs."""
        self.current_task = str(task_name)
        self.anchors_ready_for_task = False
        self.traj_buffer.reset()
        self.fusefeat_buffer.reset()
        if self.rank == 0:
            print(f"[OnlineDPMM] Preparing anchors for task: {self.current_task}")

    def consume_task_batch(self, traj_batch=None, fusefeat_batch=None) -> None:
        """Stream one gathered task batch through bounded buffers on rank 0.

        Each sample is consumed exactly once. If adding a batch would overflow
        a buffer, the batch is split, the full buffer is fitted immediately,
        and the remainder continues into the emptied buffer.
        """
        if self.rank != 0:
            return

        if self.enable_traj and traj_batch is not None:
            traj_flat = traj_batch.detach().reshape(traj_batch.shape[0], -1)
            self._consume_stream(
                buffer=self.traj_buffer,
                batch=traj_flat,
                dpmm=self.traj_dpmm,
                replay_ratio=self.config.traj_dpmm_replay_ratio,
                max_replay=self.config.traj_dpmm_max_replay_samples,
                tag='traj',
                anchor_dim_slice=20,
            )

        if self.enable_fusefeat and fusefeat_batch is not None:
            fusefeat_flat = fusefeat_batch.detach().reshape(fusefeat_batch.shape[0], -1)
            self._consume_stream(
                buffer=self.fusefeat_buffer,
                batch=fusefeat_flat,
                dpmm=self.fusefeat_dpmm,
                replay_ratio=self.config.fusefeat_dpmm_replay_ratio,
                max_replay=self.config.fusefeat_dpmm_max_replay_samples,
                tag='fusefeat',
                anchor_dim_slice=None,
            )

    def _consume_stream(self, buffer, batch, dpmm, replay_ratio,
                        max_replay, tag, anchor_dim_slice) -> None:
        """Consume a CPU/GPU batch without dropping samples at wraparound."""
        batch = batch.detach().to(dtype=torch.float32).cpu()
        if batch.ndim != 2 or batch.shape[1] != buffer.data_dim:
            raise ValueError(
                f"Expected {tag} batch (N, {buffer.data_dim}), got {tuple(batch.shape)}"
            )

        offset = 0
        while offset < batch.shape[0]:
            if buffer.is_full():
                self._fit_local_buffer(
                    dpmm, buffer, replay_ratio, max_replay, tag, anchor_dim_slice)

            free_slots = buffer.capacity - buffer.num_filled
            take = min(free_slots, batch.shape[0] - offset)
            buffer.add_batch(batch[offset:offset + take])
            offset += take

            if buffer.is_full():
                self._fit_local_buffer(
                    dpmm, buffer, replay_ratio, max_replay, tag, anchor_dim_slice)

    def _fit_local_buffer(self, dpmm, buffer, replay_ratio, max_replay,
                          tag, anchor_dim_slice):
        """Fit one DPMM buffer locally on rank 0 and reset the buffer."""
        if self.rank != 0 or buffer.num_filled == 0:
            return None

        buffer_data = buffer.get_all()
        fit_data = self._mix_replay(dpmm, buffer_data, replay_ratio, max_replay)
        fit_data = purge_invalid_values(fit_data, f"{tag}_dpmm_fit")

        if fit_data.shape[0] == 0:
            print(f"[OnlineDPMM] Skipping {tag} update: no valid data")
            buffer.reset()
            return None

        print(f"[OnlineDPMM] Fitting {tag} DPMM with {fit_data.shape[0]} samples "
              f"(buffer={buffer_data.shape[0]}, task={self.current_task}, "
              f"step={self.global_step})")
        dpmm.fit(fit_data)
        new_anchors = self._extract_anchors(dpmm, anchor_dim_slice)
        self.tag_update_counts[tag] += 1
        self._log_dpmm_state(dpmm, tag)
        self.update_count += 1
        buffer.reset()
        print(f"[OnlineDPMM] {tag} DPMM updated: {new_anchors.shape[0]} clusters")
        return new_anchors

    def finalize_task(self, model):
        """Flush partial buffers, broadcast final anchors, and update the model.

        This collective method must be called by every distributed rank after
        all samples from the current task have been consumed.
        """
        traj_anchors = None
        fusefeat_anchors = None

        if self.rank == 0:
            if self.enable_traj:
                self._fit_local_buffer(
                    self.traj_dpmm, self.traj_buffer,
                    self.config.traj_dpmm_replay_ratio,
                    self.config.traj_dpmm_max_replay_samples,
                    'traj', 20)
                traj_anchors = self._extract_anchors(self.traj_dpmm, 20)

            if self.enable_fusefeat:
                self._fit_local_buffer(
                    self.fusefeat_dpmm, self.fusefeat_buffer,
                    self.config.fusefeat_dpmm_replay_ratio,
                    self.config.fusefeat_dpmm_max_replay_samples,
                    'fusefeat', None)
                fusefeat_anchors = self._extract_anchors(self.fusefeat_dpmm, None)

        result = {}
        if self.enable_traj:
            traj_anchors = self._broadcast_anchors(traj_anchors)
            if traj_anchors.numel() == 0:
                raise RuntimeError(
                    f"Trajectory DPMM produced no anchors for task {self.current_task}."
                )
            model.update_traj_anchors(traj_anchors.to(self.device))
            result['traj'] = tuple(traj_anchors.shape)

        if self.enable_fusefeat:
            fusefeat_anchors = self._broadcast_anchors(fusefeat_anchors)
            if fusefeat_anchors.numel() == 0:
                raise RuntimeError(
                    f"Fuse-feature DPMM produced no anchors for task {self.current_task}."
                )
            model.update_fusefeat_anchors(fusefeat_anchors.to(self.device))
            result['fusefeat'] = tuple(fusefeat_anchors.shape)

        if self.rank == 0:
            print(f"[OnlineDPMM] Frozen anchors ready for {self.current_task}: {result}")
        self.anchors_ready_for_task = True
        return result

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
        Tensors are moved to ``self.device`` because the NCCL backend
        (used for GPU distributed training) does not support CPU tensors.
        """
        if self.world_size <= 1:
            if anchors is None:
                return torch.empty(0, 0, dtype=torch.float32)
            return anchors

        # Broadcast shape (2,) from rank 0 — must be on GPU for NCCL
        if self.rank == 0:
            if anchors is None:
                shape = torch.tensor([0, 0], dtype=torch.long, device=self.device)
            else:
                shape = torch.tensor(anchors.shape, dtype=torch.long, device=self.device)
        else:
            shape = torch.empty(2, dtype=torch.long, device=self.device)

        torch.distributed.broadcast(shape, src=0)

        # If shape is (0, 0), no anchors to swap
        if shape[0] == 0 or shape[1] == 0:
            return torch.empty(0, 0, dtype=torch.float32)

        # Broadcast data — must be on GPU for NCCL
        if self.rank == 0:
            anchors = anchors.to(self.device)
        else:
            anchors = torch.empty(*shape, dtype=torch.float32, device=self.device)
        torch.distributed.broadcast(anchors, src=0)

        # Return on CPU; caller moves to device as needed
        return anchors.cpu()

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

        update_index = self.tag_update_counts.get(tag, 0)
        path = os.path.join(
            self.logdir, self.config.dpmm_log_dir, tag,
            f"step{self.global_step:06d}-update{update_index:04d}-tracked_clusters.json",
        )
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tracked, f, indent=2)

        # Also save components
        components = sorted(dpmm.components, key=lambda x: x['k'])
        comp_path = os.path.join(
            self.logdir, self.config.dpmm_log_dir, tag,
            f"step{self.global_step:06d}-update{update_index:04d}-components.json",
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
