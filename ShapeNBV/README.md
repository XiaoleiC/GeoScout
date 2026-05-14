# GeoScout: Caption-Conditioned NBV on ShapeNet

GeoScout studies next-best-view (NBV) policies for active 3D reconstruction
when the agent receives a compact language prior about the object geometry.
The core assumption is practical: before reconstructing an object in detail, a
robot or user often knows coarse facts such as "chair with thin legs and a
backrest" or "table with a central pedestal".  GeoScout encodes that prior with
a frozen text encoder and conditions a PPO NBV policy on it.

This codebase contains:

- a ShapeNet object-centric NBV simulator with mesh rendering, occupancy-grid
  belief updates, and coverage rewards;
- discrete and continuous camera-action PPO policies;
- caption preprocessing and sentence-transformer embedding utilities;
- numerical evaluation for learned policies and hand-designed baselines;
- compact aggregate CSV/JSON outputs for reproducing the reported numbers.

## Repository Layout

```text
ShapeNBV/
  shapenbv/                  core environment, renderer, encoder, voxel utils
  scripts/                   preprocessing, training, evaluation, Modal helpers
  tests/                     CPU/GPU smoke tests
  eval_*                     tracked aggregate evaluation summaries
  caption_*                  tracked caption reports and compact JSONL outputs
```

Large binary artifacts are not committed.  Keep ShapeNet, preprocessed `.pt`
grids, PPO checkpoints, W&B logs, and full rollout image dumps in an external
artifact directory.  See [`REPRODUCE.md`](REPRODUCE.md) for the expected layout
and exact commands.

## Install

Use Python 3.10 or 3.11.  Install the PyTorch build that matches your CUDA
driver first, then install the package:

```bash
cd ShapeNBV
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the fastest GPU renderer used in our main runs:

```bash
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

## Minimal Smoke Test

The smoke tests create synthetic cube meshes and do not require ShapeNet:

```bash
cd ShapeNBV
pytest -q tests/test_voxel_utils.py tests/test_renderer.py tests/test_tensor_env_smoke.py
```

CUDA-only tests automatically skip when CUDA or optional rasterizers are not
available.

The same checks are available through `make test`; `make compile` performs a
fast syntax/import-bytecode sanity check for the package and scripts.

## Main Reproduction Commands

The main experiment setting uses 600 in-distribution ShapeNet objects from
chair, sofa, and table, plus 300 held-out objects from the same categories:

```text
chair  03001627
sofa   04256520
table  04379243
```

Typical workflow:

```bash
# 1) Encode caption JSONL into 384-d MiniLM embeddings.
python -m scripts.precompute_object_caption_embeddings \
  --caption-jsonl "$CAPTION_JSONL" \
  --out "$CAPTION_EMB_PATH" \
  --device cuda

# 2) Preprocess ShapeNet meshes into reward grids and attach caption embeddings.
python -m scripts.preprocess \
  --shapenet_root "$SHAPENET_ROOT" \
  --out_dir "$PREPROC_DIR" \
  --synsets 03001627,04256520,04379243 \
  --limit_per_synset 200 \
  --grid_size 128 \
  --grid_storage_dtype uint8 \
  --caption_emb_path "$CAPTION_EMB_PATH" \
  --n_workers 16

# 3) Train a caption-conditioned discrete PPO policy.
python -m scripts.train \
  --shapenet_root "$SHAPENET_ROOT" \
  --preproc_dir "$PREPROC_DIR" \
  --synsets 03001627,04256520,04379243 \
  --limit_per_synset 200 \
  --tensor_env \
  --tensor_env_n_envs 128 \
  --caption_dim 384 \
  --auto_lookat_center \
  --action_space_type discrete \
  --renderer_backend nvdiffrast \
  --total_timesteps 24000000 \
  --checkpoint_freq_steps 1000000 \
  --log_dir runs/geoscout_discrete_revised \
  --wandb_mode "${WANDB_MODE:-disabled}"

# 4) Evaluate GeoScout and geometric baselines.
python -m scripts.evaluate_baselines \
  --shapenet_root "$SHAPENET_ROOT" \
  --preproc_dir "$PREPROC_DIR" \
  --out_dir runs/eval_geoscout_in_dist \
  --ckpt "$GEOSCOUT_CKPT" \
  --policies ppo,fibonacci,axis6,ring,random \
  --synsets 03001627,04256520,04379243 \
  --limit_per_synset 200 \
  --caption_dim 384 \
  --action_space_type discrete \
  --deterministic \
  --n_envs 64 \
  --n_episodes 600
```

For no-caption ablations, use `--caption_dim 0` consistently during training
and evaluation.  For continuous-policy ablations, use
`--action_space_type continuous_tanh`.

## Notes on Data and Licensing

This repository does not redistribute ShapeNet meshes.  Download ShapeNetCore
from the official source under its license, then point `SHAPENET_ROOT` to the
downloaded root.  Caption files, preprocessed grids, and checkpoints should be
released separately as project artifacts rather than committed to git.
