# GeoScout

**GeoScout: Caption-Conditioned Next-Best-View Policies for Generalizable
3D Reconstruction**

GeoScout is an object-centric next-best-view reconstruction codebase.  It
trains caption-conditioned RL policies that choose camera viewpoints from a
partial occupancy belief state, using coarse geometric language as a prior for
where an unseen object is likely to contain informative structure.

The repository is organized from the root.  There is no nested project folder:

```text
geoscout/       core environment, tensorized rollout code, renderers, encoders
scripts/        preprocessing, training, evaluation, caption, and Modal jobs
tests/          smoke and unit tests
REPRODUCE.md    end-to-end reproduction guide
```

Large artifacts are intentionally not committed: ShapeNet meshes, preprocessed
voxel grids, checkpoints, W&B logs, visual atlases, and rollout dumps should
live outside git.

## Setup

```bash
git clone git@github.com:XiaoleiC/GeoScout.git
cd GeoScout

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional GPU rasterization support:

```bash
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

Copy the environment template if you want local paths and logging defaults:

```bash
cp .env.example .env
```

Never commit `.env`, API keys, checkpoints, datasets, or generated rollouts.

## Quick Smoke Test

```bash
make compile
make test
```

CUDA-specific tests skip automatically when CUDA is unavailable.  Full training
and evaluation require a CUDA GPU for practical runtime.

## Main Workflow

Precompute caption embeddings:

```bash
python -m scripts.precompute_object_caption_embeddings \
  --caption-jsonl "$CAPTION_JSONL" \
  --out "$CAPTION_EMB_PATH" \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device cuda
```

Preprocess ShapeNet objects:

```bash
python -m scripts.preprocess \
  --shapenet_root "$SHAPENET_ROOT" \
  --out_dir "$PREPROC_DIR" \
  --synsets 03001627,04256520,04379243 \
  --limit_per_synset 200 \
  --grid_size 128 \
  --caption_emb_path "$CAPTION_EMB_PATH"
```

Train the main discrete GeoScout policy:

```bash
python -m scripts.train \
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
python -m scripts.evaluate_baselines \
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

See [REPRODUCE.md](REPRODUCE.md) for the full artifact layout and exact
experiment commands.
