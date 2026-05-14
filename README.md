# GeoScout

**GeoScout: Caption-Conditioned Next-Best-View Policies for Generalizable
3D Reconstruction**

GeoScout is an object-centric next-best-view reconstruction codebase.  It
trains caption-conditioned RL policies that choose camera viewpoints from a
partial occupancy belief state, using coarse geometric language as a prior for
where an unseen object is likely to contain informative structure.

This repository is managed with **uv** and is organized directly from the repo
root:

```text
geoscout/       core environment, tensor rollout code, renderers, encoders
scripts/        preprocessing, training, evaluation, caption, and Modal jobs
tests/          smoke and unit tests
pyproject.toml  uv / Python project definition
REPRODUCE.md    end-to-end reproduction guide
```

Large artifacts are intentionally not committed: ShapeNet meshes, preprocessed
voxel grids, checkpoints, W&B logs, visual atlases, and rollout dumps should
live outside git.

## Setup With uv

Install uv if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo and create the managed environment:

```bash
git clone git@github.com:XiaoleiC/GeoScout.git
cd GeoScout

uv python install 3.10
uv sync --extra dev --extra caption --extra viz --extra cloud
```

For CUDA training, make sure the PyTorch wheel matches your driver/CUDA stack.
If you need to override uv's default PyTorch resolution, install the desired
wheel inside the uv environment, for example:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Optional GPU rasterization support:

```bash
uv pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

Copy the environment template if you want local paths and logging defaults:

```bash
cp .env.example .env
```

Never commit `.env`, API keys, checkpoints, datasets, or generated rollouts.

## Sanity Checks

Compile the package and run the lightweight tests through uv:

```bash
uv run make compile
uv run make test
```

CUDA-specific tests skip automatically when CUDA is unavailable.  Full training
and evaluation require a CUDA GPU for practical runtime.

## Main Workflow

Precompute caption embeddings:

```bash
uv run python -m scripts.precompute_object_caption_embeddings \
  --caption-jsonl "$CAPTION_JSONL" \
  --out "$CAPTION_EMB_PATH" \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device cuda
```

Preprocess ShapeNet objects:

```bash
uv run python -m scripts.preprocess \
  --shapenet_root "$SHAPENET_ROOT" \
  --out_dir "$PREPROC_DIR" \
  --synsets 03001627,04256520,04379243 \
  --limit_per_synset 200 \
  --grid_size 128 \
  --caption_emb_path "$CAPTION_EMB_PATH"
```

Train the main discrete GeoScout policy:

```bash
uv run python -m scripts.train \
  --shapenet_root "$SHAPENET_ROOT" \
  --preproc_dir "$PREPROC_DIR" \
  --synsets 03001627,04256520,04379243 \
  --limit_per_synset 200 \
  --tensor_env \
  --tensor_env_n_envs 128 \
  --caption_dim 384 \
  --action_space_type discrete \
  --renderer_backend nvdiffrast \
  --total_timesteps 24000000 \
  --log_dir runs/geoscout_discrete
```

Evaluate against learned and non-adaptive policies:

```bash
uv run python -m scripts.evaluate_baselines \
  --shapenet_root "$SHAPENET_ROOT" \
  --preproc_dir "$PREPROC_DIR" \
  --out_dir runs/eval_geoscout \
  --policies ppo,fibonacci,axis6,ring,random \
  --ckpt "$GEOSCOUT_CKPT" \
  --synsets 03001627,04256520,04379243 \
  --n_envs 64 \
  --caption_dim 384 \
  --action_space_type discrete \
  --deterministic
```

## Modal Jobs

The cloud entrypoints are also run through uv:

```bash
uv run modal run scripts/modal_app.py::preprocess
uv run modal run --detach scripts/modal_app.py::train
uv run modal run scripts/modal_app.py::evaluate_baselines_shapenet
```

See [REPRODUCE.md](REPRODUCE.md) for the full artifact layout, dataset
assumptions, and exact experiment commands.
