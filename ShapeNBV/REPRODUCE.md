# Reproducing GeoScout

This guide documents the public reproduction path for the GeoScout experiments.
It separates code, licensed data, and generated artifacts so the repository can
stay lightweight while still making every reported number traceable.

## 1. Artifact Layout

Keep large artifacts outside git.  The commands below assume:

```text
geoscout_artifacts/
  ShapeNetCore.v2/                         licensed ShapeNet meshes
  captions/
    train600_revised.jsonl                 captions for the 600 training objects
    ood300_revised.jsonl                   captions for the 300 held-out objects
    train600_minilm_l6.pt                  sentence-transformer embeddings
    ood300_minilm_l6.pt
  preproc/
    train600_attr_v2/                      one <synset>_<model_id>.pt per object
    ood300_attr_v2/
  ckpts/
    geoscout_discrete_revised.zip
    geoscout_discrete_raw_caption.zip
    geoscout_discrete_nocap.zip
    geoscout_continuous_raw_caption.zip
  eval/
    in_dist_600/
    ood_300/
```

The repository tracks compact CSV/JSON summaries under `eval_*`, but not the
full rollout dumps, preprocessed grids, or checkpoints.

## 2. Environment

```bash
cd ShapeNBV
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the PyTorch wheel matching your CUDA setup first if needed.
pip install -r requirements.txt

# Optional but recommended for GPU mesh rasterization.
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

Create a local environment file if useful:

```bash
cp .env.example .env
```

Do not commit `.env` or any API keys.

## 3. Caption Embeddings

Caption JSONL rows are keyed by the same object id used for preprocessed files:
`<synset>_<model_id>`.  Each row should contain a compact geometry-only caption
under `caption.embedding_caption`.

```bash
python -m scripts.precompute_object_caption_embeddings \
  --caption-jsonl "$CAPTION_JSONL" \
  --out "$CAPTION_EMB_PATH" \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device cuda \
  --batch-size 128
```

The saved `.pt` payload stores normalized 384-d embeddings plus the original
text for auditability.

## 4. ShapeNet Preprocessing

The main experiments use chair, sofa, and table:

```text
03001627 chair
04256520 sofa
04379243 table
```

Build the in-distribution preprocessed set:

```bash
python -m scripts.preprocess \
  --shapenet_root "$SHAPENET_ROOT" \
  --out_dir "$PREPROC_DIR" \
  --synsets 03001627,04256520,04379243 \
  --limit_per_synset 200 \
  --grid_size 128 \
  --grid_storage_dtype uint8 \
  --caption_emb_path "$CAPTION_EMB_PATH" \
  --n_workers 16
```

For no-caption baselines, omit `--caption_emb_path` and train/evaluate with
`--caption_dim 0`.

## 5. Training

Main caption-conditioned discrete policy:

```bash
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
  --grid_size 128 \
  --obs_grid_size 32 \
  --episode_len 50 \
  --buffer_size 30 \
  --total_timesteps 24000000 \
  --checkpoint_freq_steps 1000000 \
  --checkpoint_keep_last 5 \
  --log_dir runs/geoscout_discrete_revised \
  --wandb_project "${WANDB_PROJECT:-shapenbv}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_mode "${WANDB_MODE:-disabled}"
```

Continuous-action ablation:

```bash
python -m scripts.train \
  --shapenet_root "$SHAPENET_ROOT" \
  --preproc_dir "$PREPROC_DIR" \
  --synsets 03001627,04256520,04379243 \
  --limit_per_synset 200 \
  --tensor_env \
  --tensor_env_n_envs 128 \
  --caption_dim 384 \
  --auto_lookat_center \
  --action_space_type continuous_tanh \
  --renderer_backend nvdiffrast \
  --total_timesteps 24000000 \
  --log_dir runs/geoscout_continuous_raw_caption
```

No-caption ablation:

```bash
python -m scripts.train \
  --shapenet_root "$SHAPENET_ROOT" \
  --preproc_dir "$PREPROC_DIR_NO_CAP" \
  --synsets 03001627,04256520,04379243 \
  --limit_per_synset 200 \
  --tensor_env \
  --tensor_env_n_envs 128 \
  --caption_dim 0 \
  --auto_lookat_center \
  --action_space_type discrete \
  --renderer_backend nvdiffrast \
  --total_timesteps 24000000 \
  --log_dir runs/geoscout_discrete_nocap
```

## 6. Numeric Evaluation

Evaluate learned PPO against non-adaptive baselines:

```bash
python -m scripts.evaluate_baselines \
  --shapenet_root "$SHAPENET_ROOT" \
  --preproc_dir "$PREPROC_DIR" \
  --out_dir runs/eval_geoscout_in_dist \
  --policies ppo,fibonacci,axis6,ring,random \
  --ckpt "$GEOSCOUT_CKPT" \
  --synsets 03001627,04256520,04379243 \
  --limit_per_synset 200 \
  --n_episodes 600 \
  --n_envs 64 \
  --caption_dim 384 \
  --action_space_type discrete \
  --deterministic \
  --renderer_backend voxel_cuda
```

Run the same command on the held-out preprocessed directory for OOD evaluation.
The output directory contains per-episode CSV files and `summary.json` files
used to reproduce the aggregate metrics.

## 7. Visual Rollouts

For qualitative trajectories and HTML atlases:

```bash
python -m scripts.evaluate_visual_rollouts \
  --mode run_policy \
  --policy discrete_s1_det \
  --shapenet_root "$SHAPENET_ROOT" \
  --preproc_dir "$PREPROC_DIR" \
  --caption_jsonl "$CAPTION_JSONL" \
  --out_dir runs/visual_rollouts/geoscout \
  --discrete_ckpt "$GEOSCOUT_CKPT" \
  --synsets 03001627,04256520,04379243 \
  --limit_per_synset 200 \
  --n_envs 64 \
  --caption_dim 384 \
  --renderer_backend voxel_cuda \
  --write_trajectory_sheets
```

Then build the static HTML atlas:

```bash
python -m scripts.evaluate_visual_rollouts \
  --mode build_report \
  --out_dir runs/visual_rollouts/geoscout \
  --report_workers 8
```

## 8. Sanity Checks

```bash
pytest -q tests/test_voxel_utils.py tests/test_renderer.py tests/test_tensor_env_smoke.py
```

Expected behavior:

- CPU tests pass without ShapeNet.
- CUDA-specific tests skip if CUDA is unavailable.
- Full training/evaluation requires a CUDA GPU for practical runtime.
