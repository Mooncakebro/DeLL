"""
Task-boundary Network-DPMM training package.

This package prepares two Dirichlet Process Mixture Models (DPMMs) before each
task — one for trajectory anchors and one for fused-feature anchors — and then
keeps the resulting anchors fixed during model optimization.

Key components:
    - TensorRingBuffer: fixed-capacity ring buffer for DPMM fitting data.
    - OnlineDPMMManager: streams task data, fits two DPMMs, and installs anchors.
    - my_train_ability_wTFFdeQtd_online: entry-point training script.
"""
