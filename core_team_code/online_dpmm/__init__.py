"""
Online alternating Network-DPMM training package.

This package provides the infrastructure to co-train the TransFuser++ network
with two Dirichlet Process Mixture Models (DPMMs) — one for trajectory anchors
and one for fused-feature anchors — inside a single training loop.

Key components:
    - TensorRingBuffer: fixed-capacity ring buffer for DPMM fitting data.
    - OnlineDPMMManager: manages two DPMMs, two buffers, and anchor hot-swap.
    - my_train_ability_wTFFdeQtd_online: entry-point training script.
"""