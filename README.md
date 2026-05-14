# GeoScout / ShapeNBV

This repository contains the code for **GeoScout: Caption-Conditioned
Next-Best-View Policies for Generalizable 3D Reconstruction**.

The public entry point is [`ShapeNBV/`](ShapeNBV/).  It contains the
object-centric NBV environment, PPO training and evaluation scripts, caption
embedding utilities, tests, and reproduction documentation.

## Quick Links

- Code package: [`ShapeNBV/shapenbv/`](ShapeNBV/shapenbv/)
- Training and evaluation scripts: [`ShapeNBV/scripts/`](ShapeNBV/scripts/)
- Reproduction guide: [`ShapeNBV/REPRODUCE.md`](ShapeNBV/REPRODUCE.md)

Large external artifacts are intentionally not committed: ShapeNet meshes,
preprocessed voxel grids, PPO checkpoints, W&B logs, and full rollout dumps
should live outside git.  See the reproduction guide for the expected artifact
layout.
