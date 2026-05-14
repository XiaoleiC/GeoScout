"""Modal cloud entrypoints for GeoScout.

Usage (one-time setup):
    modal run GeoScout/scripts/modal_app.py::download_shapenet
        — fetches ShapeNetCore.v2 to the persistent volume.
    modal run GeoScout/scripts/modal_app.py::preprocess
        — voxelizes meshes into <vol>/preproc/*.pt.
    modal run --detach GeoScout/scripts/modal_app.py::train
        — kicks off the GenNBV-faithful PPO trainer on L4:4.

Two persistent volumes mirror the object_nbv_zgr setup:
    "geoscout-data": ShapeNetCore.v2 raw + voxelized preproc + sqlite indexes
    "geoscout-runs": training run artefacts (ckpts, wandb offline, debug viz)
"""
from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("geoscout")

# Lightweight image: torch + trimesh + sb3. Mesh rendering is now a
# pure-PyTorch ray-triangle intersection (see geoscout/mesh_renderer.py),
# so no IsaacGym / gsplat compilation is required — image
# build is a few minutes instead of 15-20 min.
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    # awscli's transitive deps include tzdata which prompts interactively
    # without DEBIAN_FRONTEND=noninteractive — set here BEFORE apt_install
    # so the build doesn't hang on a `Configuring tzdata` dialog.
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0", "wget", "unzip",
                  "awscli")  # awscli used by ABO downloader; bundled here so the
                             # download function doesn't need a per-function image
                             # variant (would re-trigger Modal's "build step after
                             # add_local_*" error).
    .run_commands("python -m pip install --upgrade pip")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy", "scipy", "trimesh", "matplotlib", "open3d",
        "stable-baselines3==2.3.2", "gymnasium",
        "wandb", "tqdm", "pillow", "wheel",
        # torch.utils.cpp_extension uses ninja when building the optional
        # custom CUDA Bresenham scatter kernel.
        "ninja",
        # Phase 1 caption pipeline. Sentence-transformers is small
        # (~80MB inc. MiniLM-L6) and bundling it here avoids the
        # "build step after add_local_*" image error you'd otherwise
        # get if you tried to chain a separate pip_install on a
        # per-function basis.
        "transformers==4.44.2",
        "sentence-transformers==3.0.1",
        # Modal CLI tools (used by huggingface_hub for auth-gated
        # ShapeNet download from inside the container).
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "PYTHONPATH": "/workspace/GeoScout",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "CC": "gcc",
        "CXX": "g++",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.9;9.0",
    })
    .run_commands(
        "python -m pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation"
    )
    .add_local_dir(
        local_path=str(Path(__file__).resolve().parents[1]),
        remote_path="/workspace/GeoScout",
    )
)

caption_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0", "wget", "unzip")
    .run_commands("python -m pip install --upgrade pip")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy",
        "trimesh",
        "open3d",
        "pillow",
        "tqdm",
        "wheel",
        "ninja",
        "huggingface_hub[hf_transfer]==0.36.0",
        "accelerate>=0.34.0",
        "transformers==4.57.1",
        "qwen-vl-utils",
        "safetensors",
        "sentencepiece",
        "protobuf",
    )
    .env({
        "PYTHONPATH": "/workspace/GeoScout",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/model-cache/huggingface",
        "CC": "gcc",
        "CXX": "g++",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.9;9.0",
    })
    .run_commands(
        "python -m pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation"
    )
    .add_local_dir(
        local_path=str(Path(__file__).resolve().parents[1]),
        remote_path="/workspace/GeoScout",
    )
)

vol_data = modal.Volume.from_name("geoscout-data", create_if_missing=True)
vol_runs = modal.Volume.from_name("geoscout-runs", create_if_missing=True)
vol_model_cache = modal.Volume.from_name("geoscout-model-cache", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-key", required_keys=["WANDB_API_KEY"])
hf_secret = modal.Secret.from_name("hf-token", required_keys=["HF_TOKEN"])


def _check_call_with_periodic_runs_commit(
    cmd,
    *,
    env,
    commit_interval_s: int = 600,
    label: str = "subprocess",
):
    """Run a subprocess and periodically commit the /runs Modal volume.

    Modal volumes are committed at normal function exit, but long training
    jobs need intermediate checkpoints to be durable and visible while the
    job is still running. The training process writes the checkpoint files;
    this wrapper commits those writes from the Modal function process.
    """
    import subprocess
    import time

    interval = int(commit_interval_s)
    if interval <= 0:
        try:
            subprocess.check_call(cmd, env=env)
        finally:
            print(f"[{label}] final /runs volume commit...", flush=True)
            vol_runs.commit()
        return

    proc = subprocess.Popen(cmd, env=env)
    last_commit = time.monotonic()
    sleep_s = min(30.0, max(1.0, interval / 10.0))
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, cmd)
                return
            now = time.monotonic()
            if now - last_commit >= interval:
                print(f"[{label}] committing /runs volume...", flush=True)
                vol_runs.commit()
                last_commit = now
            time.sleep(sleep_s)
    finally:
        print(f"[{label}] final /runs volume commit...", flush=True)
        vol_runs.commit()


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


@app.function(
    image=image,
    volumes={"/data": vol_data},
    timeout=2 * 3600,
    cpu=2.0,
    memory=4 * 1024,
)
def inspect_data_volume(
    root: str = "/data",
    depth2: bool = True,
    largest_files: int = 30,
):
    """Read-only size/inode audit for the geoscout-data Modal volume."""
    import heapq
    import os
    from collections import Counter, defaultdict
    from pathlib import Path

    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f"{root} does not exist")

    def table(title, rows, headers):
        print(f"\n=== {title} ===", flush=True)
        print("\t".join(headers), flush=True)
        for row in rows:
            print("\t".join(str(x) for x in row), flush=True)

    st = os.statvfs(root)
    print("=== filesystem ===", flush=True)
    print(
        f"blocks={st.f_blocks} free_blocks={st.f_bfree} "
        f"files={st.f_files} free_files={st.f_ffree} "
        f"used_files={st.f_files - st.f_ffree} "
        f"inode_use={(st.f_files - st.f_ffree) / max(st.f_files, 1):.4f}",
        flush=True,
    )

    top = defaultdict(lambda: {"bytes": 0, "files": 0, "dirs": 0})
    second = defaultdict(lambda: {"bytes": 0, "files": 0, "dirs": 0})
    ext = defaultdict(lambda: {"bytes": 0, "files": 0})
    largest = []
    total_bytes = 0
    total_files = 0
    total_dirs = 0

    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        try:
            rel_parts = p.relative_to(root_path).parts
        except ValueError:
            rel_parts = ()

        if rel_parts:
            top_key = rel_parts[0]
            top[top_key]["dirs"] += len(dirnames)
            if len(rel_parts) >= 2:
                second_key = str(Path(rel_parts[0]) / rel_parts[1])
                second[second_key]["dirs"] += len(dirnames)
        else:
            total_dirs += len(dirnames)

        for name in filenames:
            fp = p / name
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            total_bytes += size
            total_files += 1
            ext_key = fp.suffix.lower() or "<none>"
            ext[ext_key]["files"] += 1
            ext[ext_key]["bytes"] += size
            if rel_parts:
                top_key = rel_parts[0]
                top[top_key]["bytes"] += size
                top[top_key]["files"] += 1
                if len(rel_parts) >= 2:
                    second_key = str(Path(rel_parts[0]) / rel_parts[1])
                    second[second_key]["bytes"] += size
                    second[second_key]["files"] += 1
            else:
                top[name]["bytes"] += size
                top[name]["files"] += 1

            item = (size, str(fp.relative_to(root_path)))
            if len(largest) < largest_files:
                heapq.heappush(largest, item)
            elif largest_files > 0 and size > largest[0][0]:
                heapq.heapreplace(largest, item)

    total_dirs += sum(v["dirs"] for v in top.values())
    print(
        f"\nTOTAL bytes={_human_bytes(total_bytes)} files={total_files} "
        f"dirs={total_dirs} entries={total_files + total_dirs}",
        flush=True,
    )

    rows = [
        (k, _human_bytes(v["bytes"]), v["files"], v["dirs"])
        for k, v in sorted(top.items(), key=lambda kv: kv[1]["bytes"], reverse=True)
    ]
    table("top-level by bytes", rows, ("path", "bytes", "files", "dirs"))

    rows = [
        (k, _human_bytes(v["bytes"]), v["files"], v["dirs"])
        for k, v in sorted(top.items(), key=lambda kv: kv[1]["files"], reverse=True)
    ]
    table("top-level by file count", rows, ("path", "bytes", "files", "dirs"))

    if depth2:
        rows = [
            (k, _human_bytes(v["bytes"]), v["files"], v["dirs"])
            for k, v in sorted(second.items(), key=lambda kv: kv[1]["bytes"], reverse=True)[:80]
        ]
        table("depth-2 by bytes (top 80)", rows, ("path", "bytes", "files", "dirs"))

        rows = [
            (k, _human_bytes(v["bytes"]), v["files"], v["dirs"])
            for k, v in sorted(second.items(), key=lambda kv: kv[1]["files"], reverse=True)[:80]
        ]
        table("depth-2 by file count (top 80)", rows, ("path", "bytes", "files", "dirs"))

    rows = [
        (k, _human_bytes(v["bytes"]), v["files"])
        for k, v in sorted(ext.items(), key=lambda kv: kv[1]["bytes"], reverse=True)[:30]
    ]
    table("extensions by bytes", rows, ("ext", "bytes", "files"))

    rows = [
        (k, _human_bytes(v["bytes"]), v["files"])
        for k, v in sorted(ext.items(), key=lambda kv: kv[1]["files"], reverse=True)[:30]
    ]
    table("extensions by file count", rows, ("ext", "bytes", "files"))

    rows = [
        (path, _human_bytes(size))
        for size, path in sorted(largest, reverse=True)
    ]
    table("largest files", rows, ("path", "bytes"))


@app.function(
    image=image,
    volumes={"/data": vol_data},
    timeout=8 * 3600,
    cpu=2.0,
    memory=4 * 1024,
)
def cleanup_shapenet_geometry_only_assets(
    root: str = "/data/ShapeNetCore.v2",
    dry_run: bool = True,
    delete_mtl: bool = False,
    delete_json: bool = False,
):
    """Remove ShapeNet assets unused by GeoScout geometry-only training.

    GeoScout renders depth/alpha from `model_normalized.obj` and reads the
    preprocessed `.pt` files. Texture images, preview images, binvox files, and
    extraction leftovers only consume space/inodes for the current pipeline.

    By default this keeps `.obj`, `.json`, and `.mtl`:
      * `.obj` is the required mesh geometry.
      * `.json` is tiny and may be useful metadata/debug context.
      * `.mtl` is not used by our renderer, but keeping it avoids noisy loaders
        that try to resolve OBJ material references. Pass `delete_mtl=True`
        only after validating loaders against missing material files.
    """
    import os
    import time
    from collections import defaultdict
    from pathlib import Path

    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f"{root} does not exist")

    delete_exts = {
        ".binvox",
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tga",
        ".tif",
        ".tiff",
        ".gif",
        ".psd",
        ".tmp",
    }
    if delete_mtl:
        delete_exts.add(".mtl")
    if delete_json:
        delete_exts.add(".json")

    st0 = os.statvfs("/data")
    stats = defaultdict(lambda: {"bytes": 0, "files": 0})
    total_bytes = 0
    total_files = 0
    t0 = time.time()

    print(
        f"[cleanup] root={root_path} dry_run={dry_run} "
        f"delete_mtl={delete_mtl} delete_json={delete_json}",
        flush=True,
    )
    print(f"[cleanup] deleting extensions: {sorted(delete_exts)}", flush=True)
    print(
        f"[cleanup] before: free_files={st0.f_ffree} "
        f"used_files={st0.f_files - st0.f_ffree}",
        flush=True,
    )

    for dirpath, _, filenames in os.walk(root_path):
        d = Path(dirpath)
        for name in filenames:
            p = d / name
            ext = p.suffix.lower()
            if ext not in delete_exts:
                continue
            try:
                size = p.stat().st_size
            except FileNotFoundError:
                continue
            stats[ext]["bytes"] += size
            stats[ext]["files"] += 1
            total_bytes += size
            total_files += 1
            if not dry_run:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

    print("[cleanup] by extension:", flush=True)
    for ext, row in sorted(stats.items(), key=lambda kv: kv[1]["files"], reverse=True):
        print(f"  {ext}\t{_human_bytes(row['bytes'])}\t{row['files']} files", flush=True)
    print(
        f"[cleanup] total target={_human_bytes(total_bytes)} "
        f"files={total_files} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )

    if dry_run:
        print("[cleanup] dry-run only; no files removed.", flush=True)
    else:
        vol_data.commit()
        st1 = os.statvfs("/data")
        print(
            f"[cleanup] after: free_files={st1.f_ffree} "
            f"used_files={st1.f_files - st1.f_ffree} "
            f"freed_files={st1.f_ffree - st0.f_ffree}",
            flush=True,
        )


# ----------------------------------------------------------------------
# Geometry-only multi-view VLM captioning
# ----------------------------------------------------------------------
@app.function(
    image=caption_image,
    gpu="A100-80GB",
    volumes={"/data": vol_data, "/model-cache": vol_model_cache},
    secrets=[hf_secret],
    timeout=24 * 3600,
    retries=0,
    cpu=8.0,
    memory=64 * 1024,
)
def caption_shapenet_objects_jsonl(
    preproc_dir: str = "/data/geoscout_preproc_g128",
    shapenet_root: str = "/data/ShapeNetCore.v2",
    out_jsonl: str = "/data/geoscout_captions/object_captions_g128_geom_v1.jsonl",
    manifest_jsonl: str = "/data/geoscout_captions/object_manifest_g128_geom_v1.jsonl",
    run_config_json: str = "/data/geoscout_captions/caption_run_config_g128_geom_v1.json",
    validation_json: str = "/data/geoscout_captions/validation_report_g128_geom_v1.json",
    debug_tar: str = "/data/geoscout_captions/debug_contact_sheets_sample.tar",
    sample_names: str = "",
    max_objects: int = 0,
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    prompt_version: str = "geom_attribute_prompt_v1",
    schema_version: str = "geoscout_geom_caption_v1",
    render_version: str = "geom_clay_10view_zup_v2",
    image_size: int = 384,
    num_views: int = 10,
    max_faces: int = 12000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 256,
    min_pixels: int = 262144,
    max_pixels: int = 1572864,
    vlm_batch_size: int = 64,
    save_debug_sheets: int = 12,
    render_only: bool = False,
    resume: bool = True,
):
    """Generate geometry-focused per-object JSON captions from ShapeNet OBJ.

    This writes exactly one JSONL caption dataset plus manifest/config/report
    files. It intentionally does not compute text embeddings.
    """
    import io
    import json
    import math
    import os
    import re
    import tarfile
    import time
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    from geoscout.data import SYNSET_TO_CATEGORY
    from geoscout.mesh_renderer import _load_obj_verts_faces

    data_root = Path(shapenet_root)
    pp_root = Path(preproc_dir)
    out_path = Path(out_jsonl)
    manifest_path = Path(manifest_jsonl)
    config_path = Path(run_config_json)
    validation_path = Path(validation_json)
    debug_path = Path(debug_tar) if debug_tar else None
    for p in [out_path, manifest_path, config_path, validation_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)

    def view_table(n: int):
        views = [
            ("front", [0.0, -1.8, 0.15], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ("back", [0.0, 1.8, 0.15], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ("left", [-1.8, 0.0, 0.15], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ("right", [1.8, 0.0, 0.15], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ("top", [0.0, -0.05, 1.8], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ("low_front", [0.0, -1.8, -0.35], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ("front_left_oblique", [-1.25, -1.25, 0.75], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ("front_right_oblique", [1.25, -1.25, 0.75], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ("back_left_oblique", [-1.25, 1.25, 0.75], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ("back_right_oblique", [1.25, 1.25, 0.75], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ]
        out = []
        for name, eye, at, up in views[:max(1, min(int(n), len(views)))]:
            out.append({
                "name": name,
                "camera_position": eye,
                "look_at": at,
                "up": up,
                "coordinate_frame": "display_zup",
                "fov_degrees": 60.0,
            })
        return out

    views = view_table(num_views)
    render_meta = {
        "render_version": render_version,
        "render_backend": "nvdiffrast_clay",
        "image_size": int(image_size),
        "contact_sheet_layout": "5x2" if len(views) > 5 else f"{len(views)}x1",
        "material_policy": "uniform_matte_clay_no_texture",
        "background": "plain_light_gray",
        "camera_convention": (
            "GeoScout display frame used by preprocessing visualizers: "
            "ShapeNet raw Y-up vertices are rotated to Z-up by "
            "(x_raw, y_raw, z_raw) -> (x_raw, -z_raw, y_raw)"
        ),
        "source_mesh_frame": "ShapeNetCore.v2 model_normalized.obj raw frame, usually Y-up",
        "display_transform": {
            "name": "shapenet_y_up_to_z_up",
            "direction": "training_raw_to_display_zup",
            "matrix_row_major": [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
        },
        "inverse_display_transform": {
            "name": "display_zup_to_training_raw",
            "formula": "(x_display, y_display, z_display) -> (x_display, z_display, -y_display)",
        },
        "caption_coordinate_policy": (
            "Caption text must not output numeric coordinates or raw axis signs. "
            "View labels are human-readable display-frame labels; each rendered "
            "view also records the corresponding training/raw-frame camera pose."
        ),
        "views": views,
    }
    vlm_meta = {
        "model": vlm_model,
        "model_revision": "",
        "inference_backend": "transformers",
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "prompt_version": prompt_version,
        "batch_size": int(vlm_batch_size),
    }

    def build_manifest():
        requested = [s.strip() for s in str(sample_names).split(",") if s.strip()]
        if requested:
            names = requested
        else:
            if not pp_root.exists():
                raise FileNotFoundError(f"preproc_dir not found: {pp_root}")
            names = sorted(p.stem for p in pp_root.glob("*.pt"))
        if int(max_objects) > 0:
            names = names[: int(max_objects)]
        rows = []
        for object_id in names:
            if "_" not in object_id:
                raise ValueError(f"object id must be <synset>_<model_id>: {object_id}")
            synset, model_id = object_id.split("_", 1)
            mesh = data_root / synset / model_id / "models" / "model_normalized.obj"
            pp = pp_root / f"{object_id}.pt"
            if not mesh.exists():
                raise FileNotFoundError(f"missing mesh for {object_id}: {mesh}")
            rows.append({
                "object_id": object_id,
                "synset": synset,
                "model_id": model_id,
                "category": SYNSET_TO_CATEGORY.get(synset, synset),
                "mesh_path": str(mesh),
                "preproc_path": str(pp) if pp.exists() else None,
            })
        return rows

    manifest = build_manifest()
    if not manifest:
        raise RuntimeError("caption manifest is empty")
    with manifest_path.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    run_config = {
        "schema_version": "geoscout_geom_caption_run_config_v1",
        "caption_schema_version": schema_version,
        "target_manifest": str(manifest_path),
        "output_jsonl": str(out_path),
        "prompt_version": prompt_version,
        "render": render_meta,
        "vlm": vlm_meta,
        "object_count": len(manifest),
        "render_only": bool(render_only),
        "max_faces": int(max_faces),
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "vlm_batch_size": int(vlm_batch_size),
    }
    config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True))

    geometry_attribute_keys = [
        "has_single_main_body",
        "has_flat_horizontal_surface",
        "has_vertical_panel",
        "has_curved_shell",
        "has_boxy_volume",
        "has_distinct_seat",
        "has_backrest",
        "has_armrests",
        "has_four_or_more_legs",
        "has_pedestal_or_central_support",
        "has_star_base_or_wheels",
        "has_thin_supports",
        "has_cross_braces_or_bars",
        "has_slats",
        "has_perforations_or_holes",
        "has_open_gaps",
        "has_concavity",
        "has_cylindrical_parts",
        "has_asymmetry",
        "has_occluded_or_hidden_supports",
    ]

    def prompt_text(category: str, object_id: str) -> str:
        if str(prompt_version).startswith("geom_attribute"):
            attr_order = ", ".join(geometry_attribute_keys)
            return f"""You are filling a geometry-attribute checklist for a ShapeNet 3D object used in next-best-view 3D reconstruction.

The input image is a 10-view contact sheet of one normalized 3D mesh. The mesh is rendered upright in a human-readable Z-up display frame after a fixed ShapeNet Y-up to Z-up display rotation. The object uses an artificial plain material. Ignore color, texture, material, lighting, style, brand, comfort, function, and real-world usage. Only judge visible or strongly implied 3D geometry.

View labels are contact-sheet labels. Use only these labels when citing evidence:
front, back, left, right, top, low_front, front_left_oblique, front_right_oblique, back_left_oblique, back_right_oblique.

Category hint: {category}
Object id: {object_id}

Return exactly one compact JSON object. Do not use markdown or code fences.
The first character must be {{ and the last character must be }}.
Use 0/1 integers for attributes. Do not write long explanations.

Use exactly this JSON structure:
{{
  "category": string,
  "shape_tags": [string],
  "attrs": [0 | 1],
  "priority_views": [string],
  "uncertainties": [string]
}}

Rules:
- "attrs" must contain exactly 20 integers, each 0 or 1, in this exact order:
  [{attr_order}]
- Set an attribute to 1 only when the geometry is visible or strongly implied; otherwise set it to 0.
- shape_tags must contain at most 5 tags chosen from: flat, curved, rectangular, cylindrical, boxy, shell-like, thin, thick, slatted, perforated, open, concave, convex, symmetric, asymmetric.
- priority_views must contain at most 4 view labels that reveal difficult geometry, such as holes, slats, thin supports, hidden legs, concavity, or asymmetry.
- uncertainties must contain at most 3 short phrases.
- Do not write material, style, comfort, or usage words such as modern, classic, industrial, decorative, stylish, ergonomic, upholstered, comfortable, stable, sitting, weight, or support weight.
- If no view is clearly important, return an empty priority_views list.
- Put ambiguity in uncertainties instead of guessing."""
        return f"""You are generating a geometry-focused JSON caption for a ShapeNet 3D object used in a next-best-view 3D reconstruction task.

The input image is a contact sheet of fixed camera views of one normalized 3D mesh. ShapeNet meshes are rendered after a fixed Y-up to Z-up display rotation, so the contact sheet shows the object upright in a human-readable frame. The object is rendered with an artificial plain material. Ignore color, texture, material, lighting, artistic style, and background. Focus only on 3D geometry and object structure.

The view labels are contact-sheet labels. Do not output numeric coordinates, axis names, coordinate signs, or raw coordinate-frame claims in the JSON. If a view matters, refer only to the provided view label.

Category hint: {category}
Object id: {object_id}

The small labels in the image are camera-view names, not text printed on the object.

Return exactly one compact JSON object. Do not use markdown. Do not wrap the
answer in ```json or any code fence. The first character must be {{ and the last
character must be }}. Do not include comments.

Use exactly this JSON structure:
{{
  "category": {{"name": string, "confidence": "high" | "medium" | "low", "evidence": string}},
  "global_geometry": {{"summary": string, "overall_shape": string, "proportions": string, "symmetry": string}},
  "parts": [{{"name": string, "description": string, "visibility": string, "reconstruction_importance": "high" | "medium" | "low"}}],
  "thin_structures": [{{"name": string, "reason": string}}],
  "openings_or_gaps": [{{"name": string, "description": string, "confidence": "high" | "medium" | "low"}}],
  "view_dependent_notes": {{"front": string, "back": string, "left": string, "right": string, "top": string, "bottom_or_low": string}},
  "nbv_relevance": {{"likely_hard_views": [string], "reason": string, "suggested_view_priorities": [string]}},
  "uncertainties": [string],
  "final_caption": string
}}

Rules:
- Mention only geometry visible or strongly implied by the views.
- The category hint is only a taxonomy hint. Do not infer comfort, usage,
  material, manufacturing style, or real-world function from the category.
- Use geometric words only, such as flat, curved, rectangular, cylindrical,
  vertical, horizontal, slatted, perforated, thin, thick, angled, splayed,
  connected, separated, concave, convex, open, closed, symmetric, asymmetric.
- First compare the labeled views against each other. Use view-specific evidence
  when describing thin structures, holes, supports, or asymmetries.
- When visible, list multiple structural parts separately, for example seat, backrest, supports, arms, tabletop, legs, base, posts, handles, or other category-specific components.
- Keep the JSON compact: at most 6 parts, at most 6 thin structures, at most 6
  openings_or_gaps, and at most 4 likely_hard_views.
- Keep every string short and factual. Use no more than 18 words for each part
  description, thin-structure reason, opening description, or view note.
- Do not mention color, texture, material, lighting, brand, or style.
- Never infer what a part is made of. Avoid words such as metal, plastic, wooden, fabric, leather, matte, shiny, colored, painted, modern, classic, industrial, decorative, elegant, stylish, or design-style words.
- Never infer comfort or function. Avoid words such as ergonomic, upholstered,
  comfortable, comfort, weight-bearing, stability, stable, user, sitting, or
  support weight. Describe only visible geometry, position, and connectivity.
- Avoid generic statements like "all views are crucial" or "views from all sides
  are needed". Name only the specific view labels that reveal nontrivial
  geometry, such as hidden supports, concavity, holes, thin bars, occluded legs,
  or strong asymmetry.
- In "nbv_relevance.likely_hard_views", include views that are useful because
  they reveal difficult geometry. Do not list a view merely because it is less
  informative. If no view is clearly hard, return an empty list.
- If something is ambiguous, put it in "uncertainties" rather than inventing.
- Keep "final_caption" to 1 or 2 sentences and 18 to 45 words. It should
  summarize the major geometric parts and only the most useful view labels.
- Make the caption useful for choosing camera views to reconstruct the 3D shape."""

    def build_basis(eye, at, up):
        eye = torch.as_tensor(eye, device="cuda", dtype=torch.float32)
        at = torch.as_tensor(at, device="cuda", dtype=torch.float32)
        up = torch.as_tensor(up, device="cuda", dtype=torch.float32)
        look = at - eye
        look = look / (torch.linalg.norm(look) + 1e-8)
        up = up / (torch.linalg.norm(up) + 1e-8)
        if abs(float((look * up).sum().detach().cpu())) > 0.995:
            up = torch.tensor([0.0, 1.0, 0.0], device="cuda", dtype=torch.float32)
        right = torch.cross(up, look, dim=0)
        right = right / (torch.linalg.norm(right) + 1e-8)
        true_up = torch.cross(look, right, dim=0)
        true_up = true_up / (torch.linalg.norm(true_up) + 1e-8)
        return torch.stack([right, true_up, look], dim=1)

    def display_to_training_vec(vec):
        x, y, z = [float(v) for v in vec]
        return [x, z, -y]

    @torch.no_grad()
    def render_contact_sheet(mesh_path: str, object_id: str):
        import nvdiffrast.torch as dr

        verts_np, faces_np = _load_obj_verts_faces(Path(mesh_path), max_faces=int(max_faces))
        # Keep the caption images in the exact display frame used by
        # preprocessing_viz_g128 and geoscout.viz: ShapeNet/ABO meshes are
        # Y-up, while all human-facing GeoScout visualizations are Z-up.
        # This prevents VLM captions from seeing sofas/chairs "lying down".
        verts_np = np.asarray(verts_np, dtype=np.float32).copy()
        raw_y = verts_np[:, 1].copy()
        raw_z = verts_np[:, 2].copy()
        verts_np[:, 1] = -raw_z
        verts_np[:, 2] = raw_y
        verts = torch.from_numpy(verts_np).to("cuda", dtype=torch.float32).contiguous()
        faces = torch.from_numpy(faces_np).to("cuda", dtype=torch.long).contiguous()
        faces_i32 = faces.to(torch.int32).contiguous()
        bbox_min = verts.min(dim=0).values
        bbox_max = verts.max(dim=0).values
        bbox_center = 0.5 * (bbox_min + bbox_max)
        bbox_extent = bbox_max - bbox_min
        max_extent = float(bbox_extent.max().detach().cpu().item())
        camera_radius = max(0.85, 1.28 * max_extent)

        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        face_n = torch.cross(v1 - v0, v2 - v0, dim=1)
        face_n = face_n / (torch.linalg.norm(face_n, dim=1, keepdim=True) + 1e-8)
        vert_n = torch.zeros_like(verts)
        vert_n.index_add_(0, faces[:, 0], face_n)
        vert_n.index_add_(0, faces[:, 1], face_n)
        vert_n.index_add_(0, faces[:, 2], face_n)
        vert_n = vert_n / (torch.linalg.norm(vert_n, dim=1, keepdim=True) + 1e-8)

        ctx = dr.RasterizeCudaContext(device=torch.device("cuda"))
        eye_list = []
        at_list = []
        actual_views = []
        for v in views:
            offset = torch.tensor(v["camera_position"], device="cuda", dtype=torch.float32)
            offset = offset / (torch.linalg.norm(offset) + 1e-8)
            eye_t = bbox_center + offset * camera_radius
            at_t = bbox_center
            eye_list.append(eye_t)
            at_list.append(at_t)
            eye_display = [float(x) for x in eye_t.detach().cpu().tolist()]
            at_display = [float(x) for x in at_t.detach().cpu().tolist()]
            actual_views.append({
                **v,
                "camera_position_actual": eye_display,
                "look_at_actual": at_display,
                "camera_position_actual_display": eye_display,
                "look_at_actual_display": at_display,
                "camera_position_actual_training_raw": display_to_training_vec(eye_display),
                "look_at_actual_training_raw": display_to_training_vec(at_display),
                "up_training_raw": display_to_training_vec(v["up"]),
            })
        eyes = torch.stack(eye_list, dim=0)
        ats = torch.stack(at_list, dim=0)
        bases = torch.stack([
            build_basis(eye_list[i], at_list[i], views[i]["up"])
            for i in range(len(views))
        ], dim=0)

        rel = verts.unsqueeze(0) - eyes[:, None, :]
        cam = torch.einsum("kji,kvj->kvi", bases, rel)
        z = cam[..., 2]
        near, far = 1e-4, 100.0
        w = z.clamp(min=near)
        H = W = int(image_size)
        tan_half = math.tan(math.radians(60.0) * 0.5)
        x_shift = 1.0 / max(float(W), 1.0)
        y_shift = -1.0 / max(float(H), 1.0)
        clip_x = cam[..., 0] / tan_half + x_shift * w
        clip_y = -cam[..., 1] / tan_half + y_shift * w
        z_ndc = ((w - near) / max(far - near, 1e-6)) * 2.0 - 1.0
        clip_z = z_ndc * w
        pos_clip = torch.stack([clip_x, clip_y, clip_z, w], dim=-1).contiguous()

        rast, _ = dr.rasterize(ctx, pos_clip, faces_i32, resolution=(H, W), grad_db=False)
        hit = rast[..., 3] > 0
        nrm, _ = dr.interpolate(vert_n.unsqueeze(0).expand(len(views), -1, -1).contiguous(), rast, faces_i32)
        pts, _ = dr.interpolate(verts.unsqueeze(0).expand(len(views), -1, -1).contiguous(), rast, faces_i32)
        nrm = nrm / (torch.linalg.norm(nrm, dim=-1, keepdim=True) + 1e-8)
        view_dir = eyes[:, None, None, :] - pts
        view_dir = view_dir / (torch.linalg.norm(view_dir, dim=-1, keepdim=True) + 1e-8)
        light_dir = torch.tensor([0.35, -0.45, 0.82], device="cuda", dtype=torch.float32)
        light_dir = light_dir / torch.linalg.norm(light_dir)
        ndotv = torch.abs((nrm * view_dir).sum(dim=-1))
        ndotl = torch.abs((nrm * light_dir).sum(dim=-1))
        shade = (0.42 + 0.42 * ndotv + 0.16 * ndotl).clamp(0.0, 1.0)
        depth = torch.linalg.norm(pts - eyes[:, None, None, :], dim=-1)
        depth = torch.where(hit, depth, torch.zeros_like(depth))
        depth_norm = torch.zeros_like(depth)
        for k in range(len(views)):
            valid = hit[k]
            if bool(valid.any()):
                d = depth[k][valid]
                lo, hi = torch.quantile(d, 0.02), torch.quantile(d, 0.98)
                depth_norm[k][valid] = ((depth[k][valid] - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
        shade = shade * (1.04 - 0.14 * depth_norm)

        bg = torch.tensor([242, 243, 245], device="cuda", dtype=torch.float32)
        clay = torch.tensor([184, 181, 170], device="cuda", dtype=torch.float32)
        rgb = bg.view(1, 1, 1, 3).expand(len(views), H, W, 3).clone()
        obj_rgb = (clay.view(1, 1, 1, 3) * shade[..., None]).clamp(0, 255)
        rgb = torch.where(hit[..., None], obj_rgb, rgb)
        rgb_np = rgb.to(torch.uint8).detach().cpu().numpy()

        pil_views = []
        for k, meta in enumerate(views):
            im = Image.fromarray(rgb_np[k], mode="RGB")
            draw = ImageDraw.Draw(im)
            label = meta["name"]
            tw = max(72, 7 * len(label) + 14)
            draw.rectangle((6, 6, tw, 29), fill=(255, 255, 255), outline=(30, 35, 45))
            draw.text((12, 11), label, fill=(10, 15, 25))
            pil_views.append(im)

        cols = 5 if len(pil_views) > 5 else len(pil_views)
        rows = int(math.ceil(len(pil_views) / cols))
        sheet = Image.new("RGB", (cols * W, rows * H), color=(242, 243, 245))
        for idx, im in enumerate(pil_views):
            sheet.paste(im, ((idx % cols) * W, (idx // cols) * H))
        return sheet, {
            "num_faces": int(faces.shape[0]),
            "num_vertices": int(verts.shape[0]),
            "bbox_center": [float(x) for x in bbox_center.detach().cpu().tolist()],
            "bbox_extent": [float(x) for x in bbox_extent.detach().cpu().tolist()],
            "camera_radius": float(camera_radius),
            "actual_views": actual_views,
        }

    def load_done_ids(path: Path):
        if not bool(resume) or not path.exists():
            return set()
        done = set()
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("quality", {}).get("json_parse_ok", False):
                    done.add(rec.get("object_id", ""))
        return done

    def load_vlm():
        print(f"[caption] loading VLM {vlm_model}", flush=True)
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except Exception:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_cls = Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(
            vlm_model,
            trust_remote_code=True,
            min_pixels=int(min_pixels),
            max_pixels=int(max_pixels),
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
        model = model_cls.from_pretrained(
            vlm_model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            print(
                f"[caption] gpu={torch.cuda.get_device_name(0)} "
                f"free={free_b / (1024 ** 3):.1f}GiB total={total_b / (1024 ** 3):.1f}GiB",
                flush=True,
            )
        return processor, model

    def build_vlm_messages(image: Image.Image, prompt: str):
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

    def run_vlm_batch(processor, model, images, prompts):
        messages_list = [
            build_vlm_messages(image, prompt)
            for image, prompt in zip(images, prompts)
        ]
        texts = [
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs = []
            video_inputs = []
            for messages in messages_list:
                kwargs = {}
                patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", None)
                if patch_size is not None:
                    kwargs["image_patch_size"] = patch_size
                vision_out = process_vision_info(messages, **kwargs)
                one_images = vision_out[0] if len(vision_out) > 0 else None
                one_videos = vision_out[1] if len(vision_out) > 1 else None
                if one_images is not None:
                    image_inputs.extend(one_images)
                if one_videos is not None:
                    video_inputs.extend(one_videos)
            inputs = processor(
                text=texts,
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            inputs = processor(text=texts, images=list(images), padding=True, return_tensors="pt")
        inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}
        gen_kwargs = {"max_new_tokens": int(max_new_tokens), "use_cache": True}
        if float(temperature) > 0:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
            })
        else:
            gen_kwargs["do_sample"] = False
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
        return processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def run_vlm(processor, model, image: Image.Image, prompt: str):
        return run_vlm_batch(processor, model, [image], [prompt])[0].strip()

    def extract_json(raw: str):
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fence is not None:
            text = fence.group(1).strip()
        elif text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "caption" in parsed and "category" not in parsed:
            parsed = parsed["caption"]
        return parsed

    appearance_terms = {
        "red", "blue", "green", "yellow", "black", "white", "gray", "grey",
        "brown", "wooden", "metal", "metallic", "plastic", "fabric", "leather",
        "glossy", "shiny", "matte", "transparent", "painted", "color",
        "texture", "material", "modern", "classic", "industrial", "decorative",
        "elegant", "stylish", "style", "ergonomic", "upholstered",
        "comfortable", "comfort", "weight-bearing", "stability", "stable",
        "sitting", "support weight",
    }
    required_caption_keys = {
        "category", "global_geometry", "parts", "thin_structures",
        "openings_or_gaps", "view_dependent_notes", "nbv_relevance",
        "uncertainties", "final_caption",
    }
    required_attribute_keys = {
        "category", "shape_tags", "attrs", "priority_views", "uncertainties",
    }

    def is_attribute_caption(caption) -> bool:
        return isinstance(caption, dict) and (
            isinstance(caption.get("attrs"), list)
            or isinstance(caption.get("attributes"), dict)
        )

    def normalized_attributes(caption):
        if not isinstance(caption, dict):
            return {}
        raw = caption.get("attrs")
        if isinstance(raw, list):
            return {
                key: (raw[i] if i < len(raw) else 0)
                for i, key in enumerate(geometry_attribute_keys)
            }
        raw = caption.get("attributes")
        return raw if isinstance(raw, dict) else {}

    def clean_geom_text(text: str, max_words: int = 18) -> str:
        s = str(text or "")
        for term in sorted(appearance_terms, key=len, reverse=True):
            s = re.sub(rf"\b{re.escape(term)}\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s+([,.;:])", r"\1", s)
        s = s.strip(" ,.;:")
        if not s:
            return ""
        words = s.split()
        if len(words) > int(max_words):
            s = " ".join(words[: int(max_words)])
        return s

    def attr_present(attr) -> bool:
        if isinstance(attr, bool):
            return attr
        if isinstance(attr, (int, float)):
            return bool(attr)
        if isinstance(attr, str):
            return attr.strip().lower() in {"true", "yes", "present", "1"}
        if not isinstance(attr, dict):
            return False
        val = attr.get("present", attr.get("value", attr.get("answer", False)))
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in {"true", "yes", "present", "1"}

    def join_phrases(items):
        items = [x for x in items if x]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def compose_attribute_caption(caption, category_hint: str):
        if not is_attribute_caption(caption):
            return caption
        attrs = normalized_attributes(caption)
        caption["attribute_order"] = list(geometry_attribute_keys)
        caption["attributes"] = {
            key: int(attr_present(attrs.get(key)))
            for key in geometry_attribute_keys
        }
        cat = str(category_hint or "")
        if isinstance(caption.get("category"), dict):
            cat = str(caption["category"].get("name") or cat)
        cat = clean_geom_text(cat.replace("_", " "), max_words=4) or "object"

        key_labels = {
            "has_single_main_body": "single main body",
            "has_flat_horizontal_surface": "flat horizontal surface",
            "has_vertical_panel": "vertical panel",
            "has_curved_shell": "curved shell",
            "has_boxy_volume": "boxy volume",
            "has_distinct_seat": "distinct seat",
            "has_backrest": "backrest",
            "has_armrests": "armrests",
            "has_four_or_more_legs": "four or more legs",
            "has_pedestal_or_central_support": "central support",
            "has_star_base_or_wheels": "star base or wheels",
            "has_thin_supports": "thin supports",
            "has_cross_braces_or_bars": "cross braces or bars",
            "has_slats": "slats",
            "has_perforations_or_holes": "perforations or holes",
            "has_open_gaps": "open gaps",
            "has_concavity": "concavity",
            "has_cylindrical_parts": "cylindrical parts",
            "has_asymmetry": "asymmetry",
            "has_occluded_or_hidden_supports": "partly hidden supports",
        }
        major_order = [
            "has_single_main_body", "has_boxy_volume", "has_curved_shell",
            "has_flat_horizontal_surface", "has_distinct_seat", "has_backrest",
            "has_armrests", "has_four_or_more_legs",
            "has_pedestal_or_central_support", "has_star_base_or_wheels",
        ]
        detail_order = [
            "has_thin_supports", "has_cross_braces_or_bars", "has_slats",
            "has_perforations_or_holes", "has_open_gaps", "has_concavity",
            "has_cylindrical_parts", "has_asymmetry",
            "has_occluded_or_hidden_supports",
        ]

        def describe_attr(key):
            attr = attrs.get(key) if isinstance(attrs, dict) else None
            if not attr_present(attr):
                return ""
            desc = ""
            for note in caption.get("attribute_notes", [])[:8]:
                if not isinstance(note, dict) or note.get("attribute") != key:
                    continue
                desc = clean_geom_text(note.get("description", ""), max_words=10)
                if desc:
                    break
            if not desc and isinstance(attr, dict):
                desc = clean_geom_text(attr.get("description", ""), max_words=10)
            return desc or key_labels.get(key, key.replace("_", " "))

        major = [describe_attr(k) for k in major_order]
        major = [m for m in major if m][:6]
        details = [describe_attr(k) for k in detail_order]
        details = [d for d in details if d][:5]

        global_shape = caption.get("global_shape") if isinstance(caption.get("global_shape"), dict) else {}
        profile = clean_geom_text(global_shape.get("profile", ""), max_words=12)
        volume = clean_geom_text(global_shape.get("overall_volume", ""), max_words=12)
        symmetry = clean_geom_text(global_shape.get("symmetry", ""), max_words=10)
        sentences = []
        if major:
            sentences.append(f"A {cat} with {join_phrases(major)}.")
        elif profile or volume:
            sentences.append(f"A {cat} with {join_phrases([volume, profile])}.")
        else:
            sentences.append(f"A {cat} with visible 3D geometric structure.")
        detail_bits = [x for x in [profile, symmetry] if x]
        detail_bits.extend(details)
        if detail_bits:
            sentences.append(
                "Reconstruction-relevant geometry includes "
                + join_phrases(detail_bits[:6])
                + "."
            )
        embedding_caption = " ".join(sentences)
        caption["embedding_caption"] = embedding_caption
        caption["final_caption"] = embedding_caption
        caption["composer_version"] = "geom_attribute_composer_v2_attrs_only"
        return caption

    def validate_caption(caption, category_hint: str, parse_ok: bool):
        quality = {
            "json_parse_ok": bool(parse_ok),
            "caption_schema_kind": "attribute" if is_attribute_caption(caption) else "freeform",
            "has_final_caption": False,
            "has_embedding_caption": False,
            "mentions_color": False,
            "mentions_material": False,
            "mentions_appearance_term": False,
            "embedding_caption_mentions_appearance_term": False,
            "category_mismatch": False,
            "too_generic": False,
            "needs_manual_review": False,
            "missing_required_fields": [],
            "num_present_attributes": 0,
            "quality_score_auto": 0,
        }
        if not parse_ok or not isinstance(caption, dict):
            quality["needs_manual_review"] = True
            return quality
        attr_schema = is_attribute_caption(caption)
        required = required_attribute_keys if attr_schema else required_caption_keys
        missing = sorted(required - set(caption.keys()))
        quality["missing_required_fields"] = missing
        final_caption = str(caption.get("embedding_caption") or caption.get("final_caption", "") or "")
        quality["has_final_caption"] = bool(final_caption.strip())
        quality["has_embedding_caption"] = bool(str(caption.get("embedding_caption", "") or "").strip())
        text = json.dumps(caption, ensure_ascii=False).lower()
        hits = sorted(
            t for t in appearance_terms
            if re.search(rf"\b{re.escape(t)}\b", text) is not None
        )
        emb_hits = sorted(
            t for t in appearance_terms
            if re.search(rf"\b{re.escape(t)}\b", final_caption.lower()) is not None
        )
        quality["mentions_appearance_term"] = bool(hits)
        quality["embedding_caption_mentions_appearance_term"] = bool(emb_hits)
        quality["mentions_color"] = any(t in hits for t in ["red", "blue", "green", "yellow", "black", "white", "gray", "grey", "brown", "color"])
        quality["mentions_material"] = any(t in hits for t in ["wooden", "metal", "metallic", "plastic", "fabric", "leather", "material", "texture"])
        pred_cat = ""
        if isinstance(caption.get("category"), dict):
            pred_cat = str(caption["category"].get("name", "")).lower()
        cat = str(category_hint).replace("_", " ").lower()
        if cat and pred_cat and cat not in pred_cat and pred_cat not in cat:
            quality["category_mismatch"] = True
        if attr_schema:
            attrs = normalized_attributes(caption)
            n_present = sum(1 for v in attrs.values() if attr_present(v)) if isinstance(attrs, dict) else 0
            quality["num_present_attributes"] = int(n_present)
            quality["too_generic"] = (len(final_caption.split()) < 12) or (n_present < 2)
        else:
            parts = caption.get("parts", [])
            n_parts = len(parts) if isinstance(parts, list) else 0
            thin_structures = caption.get("thin_structures", [])
            n_thin = len(thin_structures) if isinstance(thin_structures, list) else 0
            quality["too_generic"] = (len(final_caption.split()) < 18) or ((n_parts + n_thin) < 2)
        score = 3
        if missing:
            score -= 1
        if quality["mentions_appearance_term"] or quality["embedding_caption_mentions_appearance_term"]:
            score -= 1
        if quality["category_mismatch"] or quality["too_generic"]:
            score -= 1
        if not quality["has_final_caption"]:
            score = 0
        quality["quality_score_auto"] = max(0, int(score))
        quality["needs_manual_review"] = (
            bool(missing) or quality["mentions_appearance_term"]
            or quality["category_mismatch"] or quality["too_generic"]
            or not quality["has_final_caption"]
        )
        return quality

    done = load_done_ids(out_path)
    if done:
        print(f"[caption] resume enabled: {len(done)} completed records in {out_path}", flush=True)

    processor = model = None
    if not bool(render_only):
        processor, model = load_vlm()

    batch_size = max(1, int(vlm_batch_size))
    tar = tarfile.open(debug_path, "w") if debug_path is not None and int(save_debug_sheets) > 0 else None
    debug_written = 0
    rows_written = 0
    rows_skipped = 0
    rows_failed = 0
    t0 = time.time()

    try:
        jsonl_file = None if bool(render_only) else out_path.open("a")
        caption_batch = []

        def write_caption_record(item, raw: str, parsed, parse_ok: bool):
            nonlocal rows_written, rows_failed
            row = item["row"]
            object_id = row["object_id"]
            if parse_ok:
                parsed = compose_attribute_caption(parsed, row["category"])
            quality = validate_caption(parsed, row["category"], parse_ok)
            record = {
                "schema_version": schema_version,
                "object_id": object_id,
                "source": {
                    "dataset": "ShapeNetCore.v2",
                    "synset": row["synset"],
                    "model_id": row["model_id"],
                    "category_hint": row["category"],
                    "mesh_path": row["mesh_path"],
                    "preproc_path": row["preproc_path"],
                },
                "render": {**render_meta, "stats": item["render_stats"]},
                "vlm": vlm_meta,
                "caption": parsed,
                "quality": quality,
                "raw_vlm_response": raw,
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            jsonl_file.flush()
            os.fsync(jsonl_file.fileno())
            rows_written += 1
            if not parse_ok:
                rows_failed += 1
            elapsed = max(time.time() - t0, 1e-6)
            done_count = int(rows_written + rows_skipped)
            rate = done_count / elapsed
            remaining = max(0, len(manifest) - done_count)
            eta_s = remaining / rate if rate > 0 else float("inf")
            print(
                f"[caption] {item['idx']}/{len(manifest)} {object_id} "
                f"parse={parse_ok} score={quality.get('quality_score_auto')} "
                f"review={quality.get('needs_manual_review')} "
                f"faces={item['render_stats']['num_faces']} "
                f"batch={item['batch_len']} dt={time.time() - item['step_t']:.2f}s "
                f"done={done_count}/{len(manifest)} rate={rate:.3f}/s eta={eta_s/60:.1f}m",
                flush=True,
            )

        def process_caption_batch(items):
            if not items:
                return
            batch_t = time.time()
            for item in items:
                item["batch_len"] = len(items)
            print(
                f"[caption-batch] start count={len(items)} "
                f"range={items[0]['idx']}-{items[-1]['idx']} "
                f"done={rows_written + rows_skipped}/{len(manifest)}",
                flush=True,
            )
            images = [item["sheet"] for item in items]
            prompts = [
                prompt_text(item["row"]["category"], item["row"]["object_id"])
                for item in items
            ]
            try:
                raws = [
                    str(raw).strip()
                    for raw in run_vlm_batch(processor, model, images, prompts)
                ]
                if len(raws) != len(items):
                    raise RuntimeError(f"VLM batch returned {len(raws)} outputs for {len(items)} inputs")
            except Exception as exc:
                if len(items) == 1:
                    item = items[0]
                    raw = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[caption] WARN generate failed for {item['row']['object_id']}: {raw[:300]}",
                        flush=True,
                    )
                    write_caption_record(item, raw, {}, False)
                    return
                print(
                    f"[caption] WARN batch generate failed for {len(items)} items; "
                    f"falling back to single-item generation: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                for item in items:
                    try:
                        raw = run_vlm(
                            processor,
                            model,
                            item["sheet"],
                            prompt_text(item["row"]["category"], item["row"]["object_id"]),
                        )
                        parsed = extract_json(raw)
                        write_caption_record(item, raw, parsed, True)
                    except Exception as one_exc:
                        raw = f"{type(one_exc).__name__}: {one_exc}"
                        print(
                            f"[caption] WARN generate/parse failed for "
                            f"{item['row']['object_id']}: {raw[:300]}",
                            flush=True,
                        )
                        write_caption_record(item, raw, {}, False)
                return

            for item, raw in zip(items, raws):
                try:
                    parsed = extract_json(raw)
                    parse_ok = True
                except Exception as exc:
                    parsed = {}
                    parse_ok = False
                    raw = raw or f"{type(exc).__name__}: {exc}"
                    print(
                        f"[caption] WARN parse failed for {item['row']['object_id']}: {raw[:300]}",
                        flush=True,
                    )
                write_caption_record(item, raw, parsed, parse_ok)
            peak_gib = 0.0
            reserved_gib = 0.0
            if torch.cuda.is_available():
                peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
                reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(
                f"[caption-batch] generated {len(items)} captions in {time.time() - batch_t:.2f}s "
                f"peak_alloc={peak_gib:.1f}GiB peak_reserved={reserved_gib:.1f}GiB",
                flush=True,
            )

        for idx, row in enumerate(manifest, start=1):
            object_id = row["object_id"]
            if object_id in done:
                rows_skipped += 1
                continue
            step_t = time.time()
            sheet, render_stats = render_contact_sheet(row["mesh_path"], object_id)
            if tar is not None and debug_written < int(save_debug_sheets):
                buf = io.BytesIO()
                sheet.save(buf, format="PNG")
                data = buf.getvalue()
                info = tarfile.TarInfo(name=f"{object_id}.png")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                debug_written += 1
            if bool(render_only):
                print(
                    f"[caption-render] {idx}/{len(manifest)} {object_id} "
                    f"faces={render_stats['num_faces']} vertices={render_stats['num_vertices']} "
                    f"dt={time.time() - step_t:.2f}s",
                    flush=True,
                )
                continue

            caption_batch.append({
                "idx": idx,
                "row": row,
                "sheet": sheet,
                "render_stats": render_stats,
                "step_t": step_t,
                "batch_len": 1,
            })
            if len(caption_batch) >= batch_size:
                process_caption_batch(caption_batch)
                caption_batch = []
        process_caption_batch(caption_batch)
        if jsonl_file is not None:
            jsonl_file.close()
    finally:
        if tar is not None:
            tar.close()

    # Validation summary over the merged/current JSONL.
    records = []
    if out_path.exists() and not bool(render_only):
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        records.append({"object_id": "<json_decode_error>", "quality": {"json_parse_ok": False}})
    manifest_ids = [r["object_id"] for r in manifest]
    record_ids = [r.get("object_id", "") for r in records if r.get("object_id", "") in set(manifest_ids)]
    counts = Counter(record_ids)
    score_hist = Counter(str((r.get("quality") or {}).get("quality_score_auto", 0)) for r in records)
    report = {
        "schema_version": "geoscout_geom_caption_validation_v1",
        "caption_jsonl": str(out_path),
        "manifest_jsonl": str(manifest_path),
        "render_only": bool(render_only),
        "num_manifest": len(manifest_ids),
        "num_records": len(record_ids),
        "num_missing": len(sorted(set(manifest_ids) - set(record_ids))),
        "num_duplicate_records": sum(1 for _, n in counts.items() if n > 1),
        "json_parse_failures": sum(1 for r in records if not (r.get("quality") or {}).get("json_parse_ok", False)),
        "empty_final_caption": sum(1 for r in records if not (r.get("quality") or {}).get("has_final_caption", False)),
        "category_mismatch": sum(1 for r in records if (r.get("quality") or {}).get("category_mismatch", False)),
        "appearance_word_flags": sum(1 for r in records if (r.get("quality") or {}).get("mentions_appearance_term", False)),
        "too_generic": sum(1 for r in records if (r.get("quality") or {}).get("too_generic", False)),
        "needs_manual_review": sum(1 for r in records if (r.get("quality") or {}).get("needs_manual_review", False)),
        "quality_score_histogram": dict(sorted(score_hist.items())),
        "missing_object_ids": sorted(set(manifest_ids) - set(record_ids))[:50],
        "duplicate_object_ids": sorted([oid for oid, n in counts.items() if n > 1])[:50],
        "rows_written_this_run": rows_written,
        "rows_skipped_this_run": rows_skipped,
        "rows_failed_this_run": rows_failed,
        "debug_sheets_written": debug_written,
        "elapsed_s": time.time() - t0,
    }
    validation_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print("[caption-summary] " + json.dumps(report, sort_keys=True), flush=True)
    vol_data.commit()
    try:
        vol_model_cache.commit()
    except Exception:
        pass
    return report


# ----------------------------------------------------------------------
# One-time data download
# ----------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/data": vol_data},
    secrets=[hf_secret],
    timeout=2 * 3600,
    cpu=4.0,
)
def download_shapenet_hf(
    full: bool = False,
    repo_id: str = "",
    hf_token: str = "",            # passed at call time; pulls from env if empty
):
    """Pull ShapeNetCore.v2 from HuggingFace into the Modal `geoscout-data` volume.

    Auth: requires `hf-token` Modal secret (Hugging Face read token from
    a user account that has been approved for the gated ShapeNet repo).
    With `hf_transfer` enabled (set in the image env), downloads run at
    1+ GB/s on Modal's backbone — Core-13 (~10 GB) finishes in minutes,
    full 55 classes (~24 GB) in <10 min.

    Layout after download:
        /data/ShapeNetCore.v2/<synset>/<model_id>/models/model_normalized.obj

    Args:
        full: if True, pull the entire ShapeNetCore-archive (~24 GB,
              all 55 classes); else pull just the 13 categories most
              used in NBV/SLAM papers (chair, sofa, table, plane, car,
              bench, cabinet, display, lamp, speaker, rifle, telephone,
              watercraft) — saves ~50% space.
        repo_id: override the source repo (default: ShapeNet/ShapeNetCore
                 for category zips, ShapeNet/ShapeNetCore-archive for full).
    """
    import os, zipfile
    from pathlib import Path
    from huggingface_hub import snapshot_download

    out_root = Path("/data/ShapeNetCore.v2")
    out_root.mkdir(parents=True, exist_ok=True)
    if any(out_root.iterdir()):
        # Allow re-runs (idempotent). Tag a sentinel for skip detection.
        print(f"[download] {out_root} non-empty — checking layout, will extend.")

    core13 = ["02691156", "02828884", "02933112", "02958343", "03001627",
              "03211117", "03636649", "03691459", "04090263", "04256520",
              "04379243", "04401088", "04530566"]
    if full:
        repo = repo_id or "ShapeNet/ShapeNetCore-archive"
        print(f"[download] pulling FULL archive from {repo}")
        snapshot_download(
            repo_id=repo, repo_type="dataset",
            local_dir="/data/_shapenet_dl", token=hf_token or os.environ.get("HF_TOKEN"),
        )
        # The archive contains ShapeNetCore.v2.zip — unzip in place.
        for zp in Path("/data/_shapenet_dl").rglob("*.zip"):
            print(f"[download] unzipping {zp.name}...")
            with zipfile.ZipFile(zp) as z:
                z.extractall("/data")
            zp.unlink()
    else:
        repo = repo_id or "ShapeNet/ShapeNetCore"
        print(f"[download] pulling Core-13 ({len(core13)} classes) from {repo}")
        snapshot_download(
            repo_id=repo, repo_type="dataset",
            allow_patterns=[f"{c}.zip" for c in core13],
            local_dir="/data/_shapenet_dl", token=hf_token or os.environ.get("HF_TOKEN"),
        )
        # Each <synset>.zip extracts to <synset>/<model_id>/models/...
        for syn in core13:
            zp = Path(f"/data/_shapenet_dl/{syn}.zip")
            if not zp.exists():
                print(f"[download] WARN {syn}.zip missing")
                continue
            print(f"[download] unzipping {syn}.zip → {out_root}/...")
            with zipfile.ZipFile(zp) as z:
                z.extractall(out_root)
            zp.unlink()
    # Cleanup the staging dir if empty.
    staging = Path("/data/_shapenet_dl")
    if staging.exists():
        try:
            staging.rmdir()
        except OSError:
            pass
    vol_data.commit()
    n_synsets = sum(1 for p in out_root.iterdir() if p.is_dir() and p.name.isdigit())
    print(f"[download] ShapeNetCore.v2 ready: {n_synsets} synsets at {out_root}")


@app.function(
    image=image,
    volumes={"/data": vol_data},
    timeout=8 * 3600,
    cpu=4.0,
    memory=8 * 1024,
)
def download_abo(
    out_subdir: str = "ABO",
    skip_3dmodels: bool = False,
    skip_listings: bool = False,
):
    """Download Amazon Berkeley Objects (ABO) into the volume.

    Layout:
        /data/<out_subdir>/3dmodels/...   GLBs (~7,953 files)
        /data/<out_subdir>/listings/...   captions / metadata (NDJSON)

    No auth required — ABO is on AWS Open Data, served via anonymous
    S3 (`--no-sign-request`). Uses CC-BY-4.0 license.

    Sizes:
        abo-3dmodels.tar  ~154 GB (3D models + textures)
        abo-listings.tar  ~83 MB  (per-object metadata + captions)

    Stream-extract avoids 154 GB tar landing on disk first.
    """
    import os, subprocess
    out_root = f"/data/{out_subdir}"
    os.makedirs(out_root, exist_ok=True)

    if not skip_listings:
        print("[abo] downloading listings (small, ~83MB)...")
        subprocess.check_call(
            "aws s3 cp --no-sign-request "
            "s3://amazon-berkeley-objects/archives/abo-listings.tar - "
            f"| tar -xf - -C {out_root}",
            shell=True,
        )

    if not skip_3dmodels:
        print("[abo] downloading 3dmodels (~154GB) — stream-extract...")
        subprocess.check_call(
            "aws s3 cp --no-sign-request "
            "s3://amazon-berkeley-objects/archives/abo-3dmodels.tar - "
            f"| tar -xf - -C {out_root}",
            shell=True,
        )

    vol_data.commit()
    # Quick stats.
    import glob
    n_glb = len(glob.glob(f"{out_root}/3dmodels/original/*/*.glb"))
    print(f"[abo] DONE: {n_glb} GLBs at {out_root}/3dmodels/")


@app.function(
    image=image,
    volumes={"/data": vol_data},
    timeout=4 * 3600,
    cpu=4.0,
    memory=4 * 1024,
)
def download_abo_subset(
    out_subdir: str = "ABO",
    categories: str = "CHAIR,LAMP,SOFA,TABLE",
    limit_per_category: int = 500,
):
    """Download ONLY the ABO GLBs matching given categories.

    Saves bandwidth vs the full 154 GB tarball — for 4 categories ×
    500 each × ~6 MB / GLB ≈ 12 GB.

    Sequence:
        1. Pull `abo-listings.tar` (83 MB) → categorize all model_ids
        2. For each (category, model_id) match in our filter, fetch
           the corresponding `<C>/<model_id>.glb` directly from S3
        3. Also unpack the listings to keep captions accessible

    Output identical to `download_abo`'s layout for the matched IDs:
        /data/<out>/3dmodels/original/<C>/<model_id>.glb
        /data/<out>/3dmodels/metadata/3dmodels.csv.gz   (just for the subset)
        /data/<out>/listings/metadata/listings_*.json.gz
    """
    import os, subprocess, gzip, csv, io
    from pathlib import Path as _P
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out_root = _P(f"/data/{out_subdir}")
    out_root.mkdir(parents=True, exist_ok=True)
    cat_set = {c.strip().upper() for c in categories.split(",") if c.strip()}
    print(f"[abo-sub] target categories = {sorted(cat_set)}, "
          f"limit_per_cat = {limit_per_category}")

    # 1) Pull listings tar (small).
    print("[abo-sub] downloading listings (83MB)...")
    subprocess.check_call(
        f"aws s3 cp --no-sign-request "
        f"s3://amazon-berkeley-objects/archives/abo-listings.tar - "
        f"| tar -xf - -C {out_root}",
        shell=True,
    )
    # 2) Pull the small 3dmodels manifest (CSV.gz, < 1MB).
    print("[abo-sub] downloading 3dmodels manifest...")
    (out_root / "3dmodels" / "metadata").mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        f"aws s3 cp --no-sign-request "
        f"s3://amazon-berkeley-objects/3dmodels/metadata/3dmodels.csv.gz "
        f"{out_root}/3dmodels/metadata/3dmodels.csv.gz",
        shell=True,
    )

    # 3) Read listings + manifest, build {category → list of model_ids}
    # then sample `limit_per_category` from each.
    import json, glob
    cat_to_ids = {c: [] for c in cat_set}
    listings_files = sorted(glob.glob(f"{out_root}/listings/metadata/listings_*.json.gz"))
    print(f"[abo-sub] reading {len(listings_files)} listing shards...")
    for shard in listings_files:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                mid = rec.get("3dmodel_id")
                if not mid:
                    continue
                pt = rec.get("product_type", "")
                if isinstance(pt, list):
                    pt = pt[0].get("value", "") if pt and isinstance(pt[0], dict) else ""
                pt = str(pt).upper()
                if pt in cat_set and len(cat_to_ids[pt]) < limit_per_category:
                    cat_to_ids[pt].append(mid)

    selected = []
    for c, ids in cat_to_ids.items():
        print(f"[abo-sub]   {c}: {len(ids)} matched")
        selected.extend(ids)
    if not selected:
        raise RuntimeError(
            f"No models matched categories {cat_set}. "
            f"Use list_abo / list_categories to inspect."
        )

    # 4) Read manifest to get exact paths.
    paths_by_id = {}
    with gzip.open(out_root / "3dmodels" / "metadata" / "3dmodels.csv.gz",
                   "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths_by_id[row["3dmodel_id"]] = row.get("path", "")

    # 5) Parallel S3 download per GLB (~12-32 GB total).
    print(f"[abo-sub] downloading {len(selected)} GLBs in parallel (16 streams)...")
    (out_root / "3dmodels" / "original").mkdir(parents=True, exist_ok=True)

    def _fetch(mid):
        rel = paths_by_id.get(mid)
        if not rel:
            return (mid, "missing-path")
        target = out_root / "3dmodels" / "original" / rel
        if target.exists():
            return (mid, "skip")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.check_call([
                "aws", "s3", "cp", "--no-sign-request", "--quiet",
                f"s3://amazon-berkeley-objects/3dmodels/original/{rel}",
                str(target),
            ])
            return (mid, "ok")
        except subprocess.CalledProcessError as e:
            return (mid, f"err:{e.returncode}")

    n_ok = n_err = n_skip = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(_fetch, mid) for mid in selected]
        for i, fut in enumerate(as_completed(futures)):
            mid, status = fut.result()
            if status == "ok":     n_ok += 1
            elif status == "skip": n_skip += 1
            else:                  n_err += 1
            if (i + 1) % 50 == 0:
                print(f"[abo-sub] {i + 1}/{len(selected)}  "
                      f"ok={n_ok} skip={n_skip} err={n_err}")

    vol_data.commit()
    print(f"[abo-sub] DONE: ok={n_ok} skip={n_skip} err={n_err}")


@app.function(
    image=image,
    volumes={"/data": vol_data},
    timeout=8 * 3600,
    cpu=8.0,
    memory=16 * 1024,
)
def preprocess_abo(
    abo_root: str = "/data/ABO",
    out_dir: str = "/data/abo_preproc",
    captions_path: str = "/data/abo_preproc/object_captions.pt",
    categories: str = "",
    limit: int = 0,
    grid_size: int = 20,
    grid_storage_dtype: str = "float32",
    n_surface_points: int = 0,            # 0 = skip persistent points_canon
    n_workers: int = 8,
):
    """Voxelize every ABO mesh + attach its per-OBJECT caption_emb.

    Same schema as ShapeNet preproc — `<model_id>.pt` per object,
    consumable by env.py / tensor_env.py without code changes (they
    only key off the dict fields, not the dataset name).
    """
    import os, gzip, csv
    # ThreadPool: trimesh is mostly C-extension so GIL is released for
    # heavy I/O. ProcessPool fails because the local _process_one
    # closure isn't pickleable.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path as _P
    import torch
    from geoscout.abo import list_abo
    from geoscout.preprocess import preproc_file_matches_config, preprocess_mesh, save_preproc

    cat_list = [c.strip() for c in categories.split(",") if c.strip()] or None
    entries = list_abo(_P(abo_root), categories=cat_list, limit=limit)
    print(f"[abo-pp] {len(entries)} ABO objects after filter")
    if not entries:
        return

    # Load per-object caption embeddings (precompute_abo_captions output).
    caption_lookup = {}
    if captions_path and _P(captions_path).exists():
        cap_dict = torch.load(captions_path, weights_only=False, map_location="cpu")
        caption_lookup = cap_dict.get("model_id_to_emb", {})
        print(f"[abo-pp] caption_lookup: {len(caption_lookup)} entries")
    else:
        print(f"[abo-pp] no captions at {captions_path} — preprocing without caption_emb")

    out_root = _P(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    def _process_one(entry):
        out_pt = out_root / f"{entry.name}.pt"
        try:
            cap = caption_lookup.get(entry.model_id)
            if out_pt.exists():
                matches, reason = preproc_file_matches_config(
                    out_pt,
                    grid_size=grid_size,
                    grid_storage_dtype=grid_storage_dtype,
                    n_surface_points=n_surface_points,
                    require_caption_emb=cap is not None,
                )
                if matches:
                    return (entry.name, "skip", "")
                print(f"[abo-pp] regenerate stale {entry.name}: {reason}", flush=True)
            data = preprocess_mesh(
                entry.glb_path,
                grid_size=grid_size,
                n_surface_points=n_surface_points,
                caption_emb=cap,
                synset=entry.product_type,            # use category as proxy "synset"
                category=entry.product_type.lower(),
                grid_storage_dtype=grid_storage_dtype,
            )
            save_preproc(out_pt, data)
            return (entry.name, "ok", "")
        except Exception as e:
            return (entry.name, "err", f"{type(e).__name__}: {e}")

    n_ok = n_skip = n_err = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_process_one, e) for e in entries]
        for i, fut in enumerate(as_completed(futures)):
            name, status, msg = fut.result()
            if status == "ok": n_ok += 1
            elif status == "skip": n_skip += 1
            else: n_err += 1
            if (i + 1) % 100 == 0:
                print(f"[abo-pp] {i + 1}/{len(entries)} ok={n_ok} skip={n_skip} err={n_err}")
            if status == "err":
                print(f"  ERR {name}: {msg}")

    vol_data.commit()
    print(f"[abo-pp] DONE ok={n_ok} skip={n_skip} err={n_err}")


@app.function(
    image=image,
    volumes={"/data": vol_data},
    timeout=20 * 60,
    cpu=2.0,
)
def precompute_abo_captions(
    abo_root: str = "/data/ABO",
    out_path: str = "/data/abo_preproc/object_captions.pt",
    categories: str = "",
    limit: int = 0,
):
    """Encode every ABO object's per-product caption with sentence-
    transformer (Phase 1 = per-OBJECT caption, not per-category).

    Output schema: {model_id → torch.Tensor[384]}, ~7K entries × 1.5KB
    ≈ 11 MB.
    """
    from pathlib import Path as _P
    import torch
    from sentence_transformers import SentenceTransformer
    from geoscout.abo import list_abo

    cat_list = [c.strip() for c in categories.split(",") if c.strip()] or None
    entries = list_abo(_P(abo_root), categories=cat_list, limit=limit)
    print(f"[abo-cap] enumerated {len(entries)} ABO objects "
          f"(categories={cat_list}, limit={limit})")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    texts = [e.caption_text[:500] for e in entries]   # cap each at 500 chars
    print(f"[abo-cap] encoding {len(texts)} captions...")
    embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                        batch_size=64, show_progress_bar=True).cpu()

    payload = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "model_id_to_emb": {entries[i].model_id: embs[i] for i in range(len(entries))},
        "model_id_to_text": {entries[i].model_id: texts[i] for i in range(len(entries))},
        "model_id_to_category": {entries[i].model_id: entries[i].product_type for i in range(len(entries))},
    }
    out = _P(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"[abo-cap] wrote {out} ({len(payload['model_id_to_emb'])} entries)")
    vol_data.commit()


@app.function(
    image=image,
    volumes={"/data": vol_data},
    timeout=24 * 3600,
    cpu=4.0,
)
def download_shapenet(url_env_var: str = "SHAPENET_URL"):
    """Download ShapeNetCore.v2 to /data/ShapeNetCore.v2.

    ShapeNet requires manual licensing — this function expects the user
    to set the secret URL via Modal secret `shapenet-url`. Alternatively,
    upload a local copy to the volume with `modal volume put`.
    """
    import os, sys
    target = Path("/data/ShapeNetCore.v2")
    if target.exists() and any(target.iterdir()):
        print(f"[download] {target} already populated — skipping.")
        return
    url = os.environ.get(url_env_var)
    if not url:
        raise RuntimeError(
            f"Set Modal secret {url_env_var} with the ShapeNetCore.v2 zip URL "
            "(after manual license at shapenet.org), or upload via "
            "`modal volume put geoscout-data ./ShapeNetCore.v2`."
        )
    target.mkdir(parents=True, exist_ok=True)
    zip_path = "/tmp/shapenet.zip"
    subprocess.check_call(["wget", "-O", zip_path, url])
    subprocess.check_call(["unzip", "-q", zip_path, "-d", "/data"])
    vol_data.commit()
    print(f"[download] ShapeNetCore.v2 ready at {target}")


# ----------------------------------------------------------------------
# Preprocessing (mesh → voxel grid_gt)
# ----------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/data": vol_data},
    timeout=12 * 3600,
    cpu=8.0,
    memory=32 * 1024,
)
def preprocess(
    synsets: str = "",
    categories: str = "",
    limit_per_synset: int = 0,
    out_dir: str = "/data/geoscout_preproc_g128",
    grid_size: int = 128,
    grid_storage_dtype: str = "uint8",
    n_points: int = 0,                       # 0 = skip points_canon (35× space win)
    n_workers: int = 8,
    caption_emb_path: str = "/data/geoscout_preproc_g128/category_embeddings.pt",
):
    """Mesh → 128³ binary occupancy grid (+ optional caption_emb attach).

    Idempotent: skips entries whose .pt already exists. With
    `n_points=0` (default) we don't persist `points_canon` — saves
    ~1.2 MB / object × 51K ShapeNet ≈ 60 GB → 1.7 GB. Voxelization
    still uses 100k internal samples for grid_gt quality.

    Pass a `caption_emb_path` (produced by
    `precompute_category_embeddings`) to attach a per-synset
    sentence-transformer embedding into each `.pt`. The env reads
    this back via `_preproc["caption_emb"]` when `caption_dim > 0`.
    """
    import subprocess, sys
    cmd = [
        sys.executable, "-u", "-m", "scripts.preprocess",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--out_dir", out_dir,
        "--limit_per_synset", str(limit_per_synset),
        "--grid_size", str(grid_size),
        "--grid_storage_dtype", grid_storage_dtype,
        "--n_surface_points", str(n_points),
        "--n_workers", str(n_workers),
    ]
    if synsets:
        cmd += ["--synsets", synsets]
    if categories:
        cmd += ["--categories", categories]
    if caption_emb_path:
        cmd += ["--caption_emb_path", caption_emb_path]
    print("[preprocess] running:", " ".join(cmd))
    subprocess.check_call(cmd, env={**__import__("os").environ,
                                    "PYTHONPATH": "/workspace/GeoScout"})
    vol_data.commit()


@app.function(
    image=image,                             # sentence-transformers is in the base image now
    volumes={"/data": vol_data},
    timeout=20 * 60,
    cpu=2.0,
)
def precompute_category_embeddings(
    out_path: str = "/data/geoscout_preproc_g128/category_embeddings.pt",
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Encode all 55 ShapeNet category names with sentence-transformers.

    Output is one ~80 KB .pt that downstream `preprocess` uses to
    attach a per-object `caption_emb` (Phase 1: synset-level shared
    vector). Re-run only when the model changes.
    """
    import subprocess, sys
    cmd = [
        sys.executable, "-u", "-m", "scripts.precompute_category_embeddings",
        "--out", out_path,
        "--model", model,
        "--device", "cpu",
    ]
    print("[caption] running:", " ".join(cmd))
    subprocess.check_call(cmd, env={**__import__("os").environ,
                                    "PYTHONPATH": "/workspace/GeoScout"})
    vol_data.commit()


@app.function(
    image=image,
    volumes={"/data": vol_data, "/model-cache": vol_model_cache},
    timeout=45 * 60,
    cpu=4.0,
    memory=8 * 1024,
)
def precompute_object_caption_embeddings(
    caption_jsonl: str = "/data/geoscout_captions/full_attr_600_qwen25_7b_a100_batch64_tok256_array_v2_20260506_corrected.jsonl",
    out_path: str = "/data/geoscout_captions/object_caption_embeddings_attr_v2.pt",
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
):
    """Encode per-object VLM geometry captions into normalized text embeddings."""
    import os
    import subprocess
    import sys

    env = {
        **os.environ,
        "PYTHONPATH": "/workspace/GeoScout",
        "HF_HOME": "/model-cache/huggingface",
    }
    cmd = [
        sys.executable, "-u", "-m", "scripts.precompute_object_caption_embeddings",
        "--caption-jsonl", caption_jsonl,
        "--out", out_path,
        "--model", model,
        "--device", "cpu",
        "--batch-size", str(batch_size),
    ]
    print("[caption-emb] running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)
    vol_data.commit()


@app.function(
    image=image,
    volumes={"/data": vol_data},
    timeout=60 * 60,
    cpu=4.0,
    memory=16 * 1024,
)
def attach_object_caption_embeddings(
    preproc_dir: str = "/data/geoscout_preproc_g128",
    embedding_path: str = "/data/geoscout_captions/object_caption_embeddings_attr_v2.pt",
    caption_jsonl: str = "/data/geoscout_captions/full_attr_600_qwen25_7b_a100_batch64_tok256_array_v2_20260506_corrected.jsonl",
    out_dir: str = "/data/geoscout_preproc_g128_attr_v2",
    report_json: str = "/data/geoscout_captions/attach_object_caption_embeddings_attr_v2_report.json",
    overwrite: bool = True,
):
    """Attach object-level caption_emb to existing voxel preproc `.pt` files."""
    import os
    import subprocess
    import sys

    cmd = [
        sys.executable, "-u", "-m", "scripts.attach_object_caption_embeddings",
        "--preproc-dir", preproc_dir,
        "--embedding-path", embedding_path,
        "--caption-jsonl", caption_jsonl,
        "--out-dir", out_dir,
        "--report-json", report_json,
    ]
    if overwrite:
        cmd.append("--overwrite")
    print("[caption-emb] running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})
    vol_data.commit()


@app.function(
    image=image,
    volumes={"/data": vol_data},
    timeout=30 * 60,
    cpu=2.0,
    memory=8 * 1024,
)
def verify_preproc_caption_embeddings(
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2",
    embedding_path: str = "/data/geoscout_captions/object_caption_embeddings_attr_v2.pt",
    caption_jsonl: str = "/data/geoscout_captions/full_attr_600_qwen25_7b_a100_batch64_tok256_array_v2_20260506_corrected.jsonl",
    report_json: str = "/data/geoscout_captions/verify_preproc_caption_embeddings_attr_v2_report.json",
):
    """Load every captioned preproc file and verify caption_emb exactly."""
    import os
    import subprocess
    import sys

    cmd = [
        sys.executable, "-u", "-m", "scripts.verify_preproc_caption_embeddings",
        "--preproc-dir", preproc_dir,
        "--embedding-path", embedding_path,
        "--caption-jsonl", caption_jsonl,
        "--report-json", report_json,
    ]
    print("[caption-emb] running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})
    vol_data.commit()


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4:4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    secrets=[wandb_secret],
    timeout=24 * 3600,
    retries=0,
)
def train(
    total_timesteps: int = 14_000_000,
    n_envs: int = 32,
    synsets: str = "",
    categories: str = "",
    limit_per_synset: int = 0,
    seed: int = 0,
    wandb_run_name: str = "geoscout-firstrun",
    log_subdir: str = "geoscout-firstrun",
):
    """GenNBV-faithful PPO trainer for GeoScout.

    Keeps GenNBV's hyperparameters verbatim:
        - n_steps=128, batch_size=128, n_epochs=5, clip_range=0.2
        - learning_rate=1e-4, gamma=0.99, target_kl=0.05, ent_coef=0.0
        - reward: surface_coverage(20) + short_path(0.1) + termination(1) - collision(10)
        - episode_len=100, buffer_size=100, grid_size=128, obs_grid_size=32
        - action: MultiDiscrete([81,81,81,1,13,13])
    """
    import subprocess, sys
    log_dir = f"/runs/{log_subdir}"
    cmd = [
        sys.executable, "-u", "-m", "scripts.train",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", "/data/geoscout_preproc_g128",
        "--log_dir", log_dir,
        "--total_timesteps", str(total_timesteps),
        "--n_envs", str(n_envs),
        "--seed", str(seed),
        "--device", "cuda",
        # GenNBV effective reward scales — kept explicit for visibility.
        "--episode_len", "100",
        "--buffer_size", "100",
        "--grid_size", "128",
        "--obs_grid_size", "32",
        "--coverage_reward_scale", "20",
        "--short_path_grace_steps", "30",
        "--short_path_max_extra", "2",
        "--short_path_scale", "0.1",
        "--termination_bonus", "1",
        "--coverage_threshold", "0.99",
        "--n_steps", "128",
        "--batch_size", "128",
        "--n_epochs", "5",
        "--learning_rate", "1e-4",
        "--clip_range", "0.2",
        "--gamma", "0.99",
        "--ent_coef", "0.0",
        "--target_kl", "0.05",
        "--subproc",
    ]
    if synsets:
        cmd += ["--synsets", synsets]
    if categories:
        cmd += ["--categories", categories]
    if limit_per_synset > 0:
        cmd += ["--limit_per_synset", str(limit_per_synset)]
    print("[train] running:", " ".join(cmd))
    subprocess.check_call(cmd, env={**__import__("os").environ,
                                    "PYTHONPATH": "/workspace/GeoScout"})
    vol_runs.commit()


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": vol_data, "/runs": vol_runs},
    secrets=[wandb_secret],
    timeout=24 * 3600,
    retries=0,
)
def train_shapenet(
    total_timesteps: int = 4_000_000,      # 500K showed kl≈0.003 = no learning
    n_envs: int = 32,
    episode_len: int = 50,
    synsets: str = "03001627,04256520,04379243",   # chair, sofa, table
    categories: str = "",
    limit_per_synset: int = 200,
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2",
    out_subdir: str = "shapenet-train",
    caption_dim: int = 384,
    wandb_project: str = "geoscout",
    wandb_run_name: str = "shapenet-firstrun",
    wandb_entity: str = "xiaoleichu-university-of-california-berkeley",
    wandb_mode: str = "online",
    max_faces: int = 5000,
    renderer_backend: str = "nvdiffrast",
    free_raycast_backend: str = "auto",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
    grid_size: int = 128,
    coverage_reward_type: str = "linear",
    coverage_threshold: float = 0.99,
    termination_bonus: float = 1.0,
    collision_penalty: float = 10.0,
    obs_grid_size: int = 32,
    action_space_type: str = "discrete",
    auto_lookat_center: bool = False,
    n_steps: int = 128,
    batch_size: int = 128,
    n_epochs: int = 5,
    learning_rate: float = 1e-4,
    ent_coef: float = 0.0,
    target_kl: float = 0.05,
    seq_names: str = "",
    novelty_reward_scale: float = 0.0,
    remaining_reward_scale: float = 0.0,
    redundancy_penalty_scale: float = 0.0,
    view_revisit_penalty_scale: float = 0.0,
    view_revisit_angle_deg: float = 12.0,
    checkpoint_freq_steps: int = 1_000_000,
    checkpoint_keep_last: int = 5,
    resume_from: str = "",
    resume_latest: bool = False,
    volume_commit_interval_s: int = 600,
    seed: int = 0,
):
    """ShapeNet tensor-env trainer for the 600-sample furniture pool.

    The defaults match the current GeoScout setting: G=128 reward grid,
    O=32 policy grid, object-caption embeddings attached to preprocessing,
    Cube Mode actions, and no forced look-at-center shortcut unless explicitly
    requested.  PPO rollout parameters are exposed so throughput can be tuned
    for larger n_envs without changing the model architecture.
    """
    import os, sys
    log_dir = f"/runs/{out_subdir}"
    resolved_resume_from = str(resume_from or "")
    if bool(resume_latest) and not resolved_resume_from:
        candidates = [
            f"{log_dir}/checkpoints/ppo_geoscout_latest.zip",
            f"{log_dir}/ppo_geoscout.zip",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                resolved_resume_from = candidate
                break
        if resolved_resume_from:
            print(f"[shapenet-train] resume_latest resolved to {resolved_resume_from}", flush=True)
        else:
            print(
                f"[shapenet-train] resume_latest requested but no checkpoint "
                f"found under {log_dir}; starting fresh.",
                flush=True,
            )
    cmd = [
        sys.executable, "-u", "-m", "scripts.train",
        "--dataset", "shapenet",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--log_dir", log_dir,
        "--total_timesteps", str(total_timesteps),
        "--n_envs", str(n_envs),
        "--seed", str(seed),
        "--device", "cuda",
        "--image_size", "400",
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--coverage_reward_scale", "20",
        "--short_path_grace_steps", "30",
        "--short_path_max_extra", "2",
        "--short_path_scale", "0.1",
        "--termination_bonus", str(termination_bonus),
        "--coverage_threshold", str(coverage_threshold),
        "--coverage_reward_type", coverage_reward_type,
        "--collision_penalty", str(collision_penalty),
        "--novelty_reward_scale", str(novelty_reward_scale),
        "--remaining_reward_scale", str(remaining_reward_scale),
        "--redundancy_penalty_scale", str(redundancy_penalty_scale),
        "--view_revisit_penalty_scale", str(view_revisit_penalty_scale),
        "--view_revisit_angle_deg", str(view_revisit_angle_deg),
        "--n_steps", str(n_steps), "--batch_size", str(batch_size),
        "--n_epochs", str(n_epochs),
        "--learning_rate", str(learning_rate), "--clip_range", "0.2",
        "--gamma", "0.99", "--ent_coef", str(ent_coef),
        "--target_kl", str(target_kl),
        "--tensor_env",
        "--tensor_env_n_envs", str(n_envs),
        "--action_space_type", action_space_type,
        "--caption_dim", str(caption_dim),
        "--limit_per_synset", str(limit_per_synset),
        "--max_faces", str(max_faces),
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
        "--checkpoint_freq_steps", str(checkpoint_freq_steps),
        "--checkpoint_keep_last", str(checkpoint_keep_last),
        "--wandb_project", wandb_project,
        "--wandb_run_name", wandb_run_name,
        "--wandb_entity", wandb_entity,
        "--wandb_mode", wandb_mode,
    ]
    if resolved_resume_from:
        cmd += ["--resume_from", resolved_resume_from]
    if auto_lookat_center:
        cmd += ["--auto_lookat_center"]
    if synsets:
        cmd += ["--synsets", synsets]
    if categories:
        cmd += ["--categories", categories]
    if seq_names:
        cmd += ["--seq_names", seq_names]
    print("[shapenet-train] running:", " ".join(cmd))
    _check_call_with_periodic_runs_commit(
        cmd,
        env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"},
        commit_interval_s=volume_commit_interval_s,
        label="shapenet-train",
    )


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    secrets=[wandb_secret],
    timeout=8 * 3600,
    retries=0,
)
def train_preprocessing_examples_ppo(
    sample_names: str = (
        "03001627_1006be65e7bc937e9141f9b58470d646,"
        "03001627_1007e20d5e811b308351982a6e40cf41,"
        "03001627_100b18376b885f206ae9ad7e32c4139d,"
        "03001627_1013f70851210a618f2e765c4a8ed3d,"
        "04256520_1037fd31d12178d396f164a988ef37cc,"
        "04256520_103b76b2594a1582eaf14273fa406ffc,"
        "04256520_104256e5bb73b0b719fb4103277a6b93,"
        "04256520_1050790962944624febad4f49b26ec52,"
        "04379243_1011e1c9812b84d2a9ed7bb5b55809f8,"
        "04379243_10139657dfa9afe0c3bd24f986301745,"
        "04379243_1028a9cbaa7a333230bbd4cddd04c77b,"
        "04379243_102f0532f9f8bbcdcb503f63ed915ed2"
    ),
    total_timesteps: int = 200_000,
    n_envs: int = 32,
    episode_len: int = 50,
    preproc_dir: str = "/data/geoscout_preproc_g128",
    out_subdir: str = "ppo-preproc12-200k-emptycuda-0505",
    eval_out_subdir: str = "",
    eval_episodes: int = 96,
    eval_policies: str = "ppo,random,axis6",
    caption_dim: int = 384,
    max_faces: int = 5000,
    renderer_backend: str = "nvdiffrast",
    free_raycast_backend: str = "cuda",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    coverage_threshold: float = 0.99,
    coverage_reward_scale: float = 20.0,
    termination_bonus: float = 1.0,
    collision_penalty: float = 10.0,
    action_space_type: str = "discrete",
    auto_lookat_center: bool = False,
    run_deterministic_eval: bool = True,
    wandb_project: str = "geoscout",
    wandb_run_name: str = "ppo-preproc12-200k-emptycuda-0505",
    wandb_entity: str = "xiaoleichu-university-of-california-berkeley",
    wandb_mode: str = "online",
    seed: int = 0,
):
    """Train PPO exactly on the preprocessing visualization samples.

    This is intentionally a smoke-scale learning/infrastructure run: fixed
    12-mesh pool, full 400x400 renderer, 128^3 reward grid, 32^3 policy grid,
    CUDA free-space update, then a small PPO/random/axis6 evaluation.
    `action_space_type=continuous_tanh` keeps the same Cube Mode pose
    domain but trains PPO with a Gaussian policy over raw actions that the
    env maps through tanh.
    """
    import json
    import os
    import subprocess
    import sys

    names = [s.strip() for s in str(sample_names).split(",") if s.strip()]
    if not names:
        raise ValueError("sample_names must contain at least one sample")
    synsets = ",".join(sorted({name.split("_", 1)[0] for name in names}))
    log_dir = f"/runs/{out_subdir}"
    eval_dir_name = eval_out_subdir or f"{out_subdir}-eval"
    eval_dir = f"/runs/{eval_dir_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(
        f"[preproc-ppo] train samples={len(names)} "
        f"total_timesteps={total_timesteps} "
        f"n_envs={n_envs} episode_len={episode_len} renderer={renderer_backend} "
        f"free={free_raycast_backend}/{free_mask_apply_mode} "
        f"action_space_type={action_space_type} auto_lookat_center={auto_lookat_center}",
        flush=True,
    )
    print("[preproc-ppo] sample_names=" + ",".join(names), flush=True)

    train_cmd = [
        sys.executable, "-u", "-m", "scripts.train",
        "--dataset", "shapenet",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--log_dir", log_dir,
        "--total_timesteps", str(total_timesteps),
        "--n_envs", str(n_envs),
        "--seed", str(seed),
        "--device", "cuda",
        "--image_size", "400",
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--coverage_reward_scale", str(coverage_reward_scale),
        "--short_path_grace_steps", "30",
        "--short_path_max_extra", "2",
        "--short_path_scale", "0.1",
        "--termination_bonus", str(termination_bonus),
        "--coverage_threshold", str(coverage_threshold),
        "--coverage_reward_type", "linear",
        "--collision_penalty", str(collision_penalty),
        "--n_steps", "128",
        "--batch_size", "128",
        "--n_epochs", "5",
        "--learning_rate", "1e-4",
        "--clip_range", "0.2",
        "--gamma", "0.99",
        "--ent_coef", "0.0",
        "--target_kl", "0.05",
        "--tensor_env",
        "--tensor_env_n_envs", str(n_envs),
        "--action_space_type", action_space_type,
        "--caption_dim", str(caption_dim),
        "--synsets", synsets,
        "--seq_names", ",".join(names),
        "--limit_per_synset", "0",
        "--max_faces", str(max_faces),
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
        "--wandb_project", wandb_project,
        "--wandb_run_name", wandb_run_name,
        "--wandb_entity", wandb_entity,
        "--wandb_mode", wandb_mode,
    ]
    if auto_lookat_center:
        train_cmd += ["--auto_lookat_center"]
    print("[preproc-ppo] train running:", " ".join(train_cmd), flush=True)
    subprocess.check_call(train_cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})

    eval_cmd = [
        sys.executable, "-u", "-m", "scripts.evaluate_baselines",
        "--dataset", "shapenet",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--out_dir", eval_dir,
        "--policies", eval_policies,
        "--ckpt", f"{log_dir}/ppo_geoscout.zip",
        "--n_episodes", str(eval_episodes),
        "--n_envs", str(n_envs),
        "--seed", str(seed + 1000),
        "--device", "cuda",
        "--image_size", "400",
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--coverage_hit_dilate_radius", "1",
        "--action_space_type", action_space_type,
        "--caption_dim", str(caption_dim),
        "--synsets", synsets,
        "--seq_names", ",".join(names),
        "--limit_per_synset", "0",
        "--max_faces", str(max_faces),
        "--coverage_threshold", str(coverage_threshold),
        "--coverage_reward_scale", str(coverage_reward_scale),
        "--coverage_reward_type", "linear",
        "--termination_bonus", str(termination_bonus),
        "--collision_penalty", str(collision_penalty),
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
    ]
    if auto_lookat_center:
        eval_cmd += ["--auto_lookat_center"]
    print("[preproc-ppo] eval running:", " ".join(eval_cmd), flush=True)
    subprocess.check_call(eval_cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})

    det_eval_dir = f"/runs/{eval_dir_name}-det"
    det_summaries = []
    if run_deterministic_eval:
        det_eval_cmd = list(eval_cmd)
        out_idx = det_eval_cmd.index("--out_dir") + 1
        det_eval_cmd[out_idx] = det_eval_dir
        pol_idx = det_eval_cmd.index("--policies") + 1
        det_eval_cmd[pol_idx] = "ppo"
        det_eval_cmd += ["--deterministic"]
        print("[preproc-ppo] deterministic eval running:", " ".join(det_eval_cmd), flush=True)
        subprocess.check_call(det_eval_cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})

    summary_path = os.path.join(eval_dir, "summaries.json")
    summaries = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summaries = json.load(f)
        print("[preproc-ppo] eval summaries=" + json.dumps(summaries, sort_keys=True), flush=True)
    det_summary_path = os.path.join(det_eval_dir, "summaries.json")
    if run_deterministic_eval and os.path.exists(det_summary_path):
        with open(det_summary_path) as f:
            det_summaries = json.load(f)
        print("[preproc-ppo] deterministic eval summaries=" + json.dumps(det_summaries, sort_keys=True), flush=True)
    vol_runs.commit()
    return {
        "train_dir": log_dir,
        "eval_dir": eval_dir,
        "det_eval_dir": det_eval_dir if run_deterministic_eval else "",
        "wandb_run_name": wandb_run_name,
        "sample_names": names,
        "summaries": summaries,
        "deterministic_summaries": det_summaries,
    }


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data},
    timeout=2 * 3600,
    retries=0,
)
def validate_continuous_tanh_action_mode(
    sample_names: str = (
        "03001627_1006be65e7bc937e9141f9b58470d646,"
        "03001627_1007e20d5e811b308351982a6e40cf41,"
        "03001627_100b18376b885f206ae9ad7e32c4139d,"
        "03001627_1013f70851210a618f2e765c4a8ed3d,"
        "04256520_1037fd31d12178d396f164a988ef37cc,"
        "04256520_103b76b2594a1582eaf14273fa406ffc,"
        "04256520_104256e5bb73b0b719fb4103277a6b93,"
        "04256520_1050790962944624febad4f49b26ec52,"
        "04379243_1011e1c9812b84d2a9ed7bb5b55809f8,"
        "04379243_10139657dfa9afe0c3bd24f986301745,"
        "04379243_1028a9cbaa7a333230bbd4cddd04c77b,"
        "04379243_102f0532f9f8bbcdcb503f63ed915ed2"
    ),
    preproc_dir: str = "/data/geoscout_preproc_g128",
    n_envs: int = 4,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    caption_dim: int = 384,
    max_faces: int = 5000,
    renderer_backend: str = "nvdiffrast",
    free_raycast_backend: str = "cuda",
    free_mask_apply_mode: str = "triton",
):
    """Hard validation for the continuous tanh Gaussian action ablation.

    Checks the math before the expensive PPO run:
      1. Box action space is continuous R^5.
      2. raw -> tanh -> pose6 matches the intended bounded mapping.
      3. look directions use pitch/yaw when auto_lookat_center is False.
      4. one real 400x400 env step is finite and produces nonzero coverage.
      5. SB3 PPO builds a DiagGaussian policy and survives a tiny rollout.
    """
    import math
    from pathlib import Path

    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from geoscout.data import list_shapenet
    from geoscout.hybrid_encoder import Hybrid_Encoder
    from geoscout.tensor_env import POSE_DIM, TensorBatchEnv

    names = [s.strip() for s in str(sample_names).split(",") if s.strip()]
    entries = list_shapenet(Path("/data/ShapeNetCore.v2"))
    name_to_path = {e.name: Path(e.mesh_path) for e in entries}
    mesh_paths = []
    preproc_paths = []
    for name in names:
        pp = Path(preproc_dir) / f"{name}.pt"
        if name in name_to_path and pp.exists():
            mesh_paths.append(name_to_path[name])
            preproc_paths.append(pp)
    if not mesh_paths:
        raise RuntimeError("No preprocessing examples resolved for validation.")

    def make_env(num_envs: int) -> TensorBatchEnv:
        return TensorBatchEnv(
            num_envs=num_envs,
            mesh_paths=mesh_paths,
            preproc_paths=preproc_paths,
            device="cuda",
            buffer_size=30,
            grid_size=grid_size,
            obs_grid_size=obs_grid_size,
            episode_len=50,
            render_size=image_size,
            fov_deg=60.0,
            cr_success_threshold=0.99,
            coverage_reward_scale=20.0,
            short_path_grace=30,
            short_path_clip=2.0,
            short_path_scale=0.1,
            only_positive_rewards=True,
            update_empty_rays=True,
            coverage_hit_dilate_radius=1,
            caption_dim=caption_dim,
            auto_lookat_center=False,
            action_space_type="continuous_tanh",
            max_faces=max_faces,
            renderer_backend=renderer_backend,
            free_raycast_backend=free_raycast_backend,
            free_mask_apply_mode=free_mask_apply_mode,
            coverage_reward_type="linear",
            termination_bonus=1.0,
            collision_penalty=10.0,
            seed=123,
        )

    env = make_env(int(n_envs))
    assert env.action_space.shape == (5,), env.action_space
    assert env.action_space_type == "continuous_tanh"
    assert not env.auto_lookat_center

    # 1-3. Exact tanh decode and camera direction checks on synthetic raws.
    norm = torch.tensor(
        [[0.0, 0.25, -0.50, 0.30, -0.40],
         [0.75, -0.20, 0.40, -0.55, 0.85]],
        device=env.device,
        dtype=torch.float32,
    )
    raw = torch.atanh(norm.clamp(-0.9999, 0.9999))
    pose6, eyes, ats = env._decode_actions(raw)
    expected = torch.zeros(2, POSE_DIM, device=env.device)
    expected[:, 0:3] = norm[:, 0:3]
    expected[:, 4] = norm[:, 3] * (math.pi / 2.0)
    expected[:, 5] = (norm[:, 4] + 1.0) * math.pi
    max_pose_err = float((pose6 - expected).abs().max().detach().cpu().item())
    if max_pose_err > 2e-5:
        raise AssertionError(f"continuous tanh pose decode mismatch: {max_pose_err}")
    look_dirs = ats - eyes
    look_dirs = look_dirs / look_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cp, sp = torch.cos(expected[:, 4]), torch.sin(expected[:, 4])
    cy, sy = torch.cos(expected[:, 5]), torch.sin(expected[:, 5])
    expected_dirs = torch.stack([cp * cy, cp * sy, sp], dim=-1)
    max_dir_err = float((look_dirs - expected_dirs).abs().max().detach().cpu().item())
    if max_dir_err > 2e-5:
        raise AssertionError(f"continuous look-dir mismatch: {max_dir_err}")

    # 4. One real renderer/free-space/reward step with cameras looking at origin.
    positions = np.asarray(
        [[0.35, 0.00, 0.92],
         [-0.70, 0.18, 0.48],
         [0.55, -0.55, 0.42],
         [0.08, 0.92, 0.30]],
        dtype=np.float32,
    )
    positions = positions[: int(n_envs)]
    pose = np.zeros((positions.shape[0], 6), dtype=np.float32)
    pose[:, 0:3] = positions
    direction = -positions / np.linalg.norm(positions, axis=1, keepdims=True)
    pose[:, 4] = np.arcsin(np.clip(direction[:, 2], -1.0, 1.0))
    yaw = np.arctan2(direction[:, 1], direction[:, 0])
    pose[:, 5] = np.where(yaw >= 0.0, yaw, yaw + 2.0 * np.pi)
    raw_np = np.empty((positions.shape[0], 5), dtype=np.float32)
    raw_np[:, 0:3] = pose[:, 0:3]
    raw_np[:, 3] = pose[:, 4] / (0.5 * np.pi)
    raw_np[:, 4] = pose[:, 5] / np.pi - 1.0
    raw_np = np.arctanh(np.clip(raw_np, -0.9999, 0.9999)).astype(np.float32)
    obs0 = env.reset()
    obs1, rewards, dones, infos = env.step(raw_np)
    if obs0.shape != obs1.shape or obs1.shape[0] != positions.shape[0]:
        raise AssertionError(f"unexpected obs shapes: {obs0.shape} -> {obs1.shape}")
    if not np.isfinite(obs1).all() or not np.isfinite(rewards).all():
        raise AssertionError("non-finite observation or reward in continuous step")
    cr_values = np.asarray([float(info.get("cr", 0.0)) for info in infos], dtype=np.float32)
    collision_values = np.asarray([bool(info.get("collision", False)) for info in infos])
    if float(cr_values.max()) <= 0.0:
        raise AssertionError(f"continuous camera step saw zero coverage: cr={cr_values.tolist()}")
    if collision_values.all():
        raise AssertionError("all validation cameras collided; action mapping is suspect")

    # 5. Verify SB3 uses a Gaussian distribution and can collect one tiny rollout.
    train_env = make_env(2)
    policy_kwargs = dict(
        features_extractor_class=Hybrid_Encoder,
        features_extractor_kwargs=dict(
            encoder_param={"hidden_shapes": [256, 256], "visual_dim": 256},
            net_param={
                "transformer_params": [[1, 256], [1, 256]],
                "append_hidden_shapes": [256, 256],
            },
            state_input_shape=(30 * POSE_DIM,),
            visual_input_shape=(30, image_size, image_size),
            state_input_only=True,
            grid_size=obs_grid_size,
            caption_dim=caption_dim,
        ),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        learning_rate=1e-4,
        gamma=0.99,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        device="cuda",
        seed=123,
        verbose=0,
    )
    dist_name = model.policy.action_dist.__class__.__name__
    if "Gaussian" not in dist_name:
        raise AssertionError(f"expected Gaussian action distribution, got {dist_name}")
    model.learn(total_timesteps=8)
    train_env.close()
    env.close()

    result = {
        "samples": len(mesh_paths),
        "action_space_shape": tuple(env.action_space.shape),
        "distribution": dist_name,
        "max_pose_err": max_pose_err,
        "max_dir_err": max_dir_err,
        "step_cr": cr_values.tolist(),
        "step_collision": collision_values.tolist(),
        "reward": np.asarray(rewards).astype(float).tolist(),
    }
    print("[validate-continuous] " + __import__("json").dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    secrets=[wandb_secret],
    timeout=2 * 3600,
    retries=0,
)
def smoke_reward_probe_shapenet(
    seq_name: str = "03001627_1006be65e7bc937e9141f9b58470d646",
    synsets: str = "03001627",
    limit_per_synset: int = 1,
    pool_size: int = 1,
    preproc_dir: str = "/data/geoscout_preproc_g128",
    out_subdir: str = "smoke-reward-probe-g128-one-chair",
    wandb_project: str = "geoscout",
    wandb_run_name: str = "smoke-reward-probe-g128-one-chair",
    wandb_entity: str = "xiaoleichu-university-of-california-berkeley",
    wandb_mode: str = "online",
    n_envs: int = 1,
    max_steps: int = 6,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    coverage_hit_dilate_radius: int = 1,
    coverage_threshold: float = 0.99,
    coverage_reward_scale: float = 20.0,
    short_path_scale: float = 0.1,
    termination_bonus: float = 1.0,
    collision_penalty: float = 10.0,
    skip_free_raycast: bool = False,
    update_empty_rays: bool = True,
    no_empty_ray_pair_dedupe: bool = True,
    profile_timing: bool = False,
    caption_dim: int = 384,
    max_faces: int = 5000,
    renderer_backend: str = "nvdiffrast",
    no_renderer_bbox_cull: bool = False,
    no_triton_free_raycast: bool = False,
    free_raycast_backend: str = "auto",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
    seed: int = 0,
):
    """Minimal Modal/WandB smoke probe for reward scale and env fps.

    This bypasses PPO and steps TensorBatchEnv directly on one ShapeNet
    sample. It is intentionally tiny so slow free-space raycasting shows
    up immediately instead of being hidden inside a 128-step PPO rollout.
    """
    import os
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import wandb

    from geoscout.data import list_shapenet
    from geoscout.tensor_env import TensorBatchEnv

    log_dir = f"/runs/{out_subdir}"
    os.makedirs(log_dir, exist_ok=True)

    synset_list = [s.strip() for s in synsets.split(",") if s.strip()] or None
    entries = list_shapenet(
        Path("/data/ShapeNetCore.v2"),
        synsets=synset_list,
        limit_per_synset=limit_per_synset,
    )
    if seq_name:
        entries = [e for e in entries if e.name == seq_name] or [
            e for e in list_shapenet(Path("/data/ShapeNetCore.v2"))
            if e.name == seq_name
        ]
    if not entries:
        raise RuntimeError(
            f"No ShapeNet entry found for seq_name={seq_name!r}, synsets={synsets!r}"
        )
    pool_entries = entries[:max(1, int(pool_size))]
    mesh_paths = [e.mesh_path for e in pool_entries]
    preproc_paths = [Path(preproc_dir) / f"{e.name}.pt" for e in pool_entries]
    missing_pp = [str(p) for p in preproc_paths if not p.exists()]
    if missing_pp:
        raise FileNotFoundError(f"preproc missing for pool: {missing_pp[:5]}")

    print(
        f"[smoke-probe] samples={[e.name for e in pool_entries]} "
        f"n_envs={n_envs} max_steps={max_steps} image_size={image_size} "
        f"grid={grid_size} obs_grid={obs_grid_size} "
        f"skip_free_raycast={skip_free_raycast} "
        f"update_empty_rays={update_empty_rays} "
        f"dedupe_empty_ray_pairs={not no_empty_ray_pair_dedupe} "
        f"renderer_backend={renderer_backend} "
        f"renderer_bbox_ray_cull={not no_renderer_bbox_cull} "
        f"use_triton_free_raycast={not no_triton_free_raycast} "
        f"free_raycast_backend={free_raycast_backend} "
        f"free_mask_apply_mode={free_mask_apply_mode} "
        f"triton_bresenham_block_rays={triton_bresenham_block_rays} "
        f"profile_timing={profile_timing}",
        flush=True,
    )

    run = wandb.init(
        project=wandb_project,
        entity=(wandb_entity or None),
        name=wandb_run_name,
        id=wandb_run_name,
        resume="allow",
        mode=wandb_mode,
        dir=log_dir,
        config={
            "seq_name": pool_entries[0].name,
            "pool_size": len(pool_entries),
            "pool_names": [e.name for e in pool_entries],
            "n_envs": n_envs,
            "max_steps": max_steps,
            "image_size": image_size,
            "grid_size": grid_size,
            "obs_grid_size": obs_grid_size,
            "coverage_hit_dilate_radius": coverage_hit_dilate_radius,
            "coverage_threshold": coverage_threshold,
            "coverage_reward_scale": coverage_reward_scale,
            "short_path_scale": short_path_scale,
            "termination_bonus": termination_bonus,
            "collision_penalty": collision_penalty,
            "skip_free_raycast": skip_free_raycast,
            "update_empty_rays": update_empty_rays,
            "dedupe_empty_ray_pairs": not no_empty_ray_pair_dedupe,
            "profile_timing": profile_timing,
            "caption_dim": caption_dim,
            "max_faces": max_faces,
            "renderer_backend": renderer_backend,
            "renderer_bbox_ray_cull": not no_renderer_bbox_cull,
            "use_triton_free_raycast": not no_triton_free_raycast,
            "free_raycast_backend": free_raycast_backend,
            "free_mask_apply_mode": free_mask_apply_mode,
            "triton_bresenham_block_rays": triton_bresenham_block_rays,
            "seed": seed,
        },
    )
    print(f"[smoke-probe] wandb run: {run.url} (mode={wandb_mode})", flush=True)

    env = TensorBatchEnv(
        num_envs=n_envs,
        mesh_paths=mesh_paths,
        preproc_paths=preproc_paths,
        device="cuda",
        buffer_size=30,
        grid_size=grid_size,
        obs_grid_size=obs_grid_size,
        episode_len=max_steps,
        render_size=image_size,
        fov_deg=60.0,
        cr_success_threshold=coverage_threshold,
        coverage_reward_scale=coverage_reward_scale,
        short_path_grace=30,
        short_path_clip=2.0,
        short_path_scale=short_path_scale,
        only_positive_rewards=True,
        skip_free_raycast=skip_free_raycast,
        update_empty_rays=update_empty_rays,
        coverage_hit_dilate_radius=coverage_hit_dilate_radius,
        caption_dim=caption_dim,
        auto_lookat_center=True,
        max_faces=max_faces,
        renderer_backend=renderer_backend,
        renderer_bbox_ray_cull=not no_renderer_bbox_cull,
        use_triton_free_raycast=not no_triton_free_raycast,
        free_raycast_backend=free_raycast_backend,
        free_mask_apply_mode=free_mask_apply_mode,
        triton_bresenham_block_rays=triton_bresenham_block_rays,
        coverage_reward_type="linear",
        termination_bonus=termination_bonus,
        collision_penalty=collision_penalty,
        dedupe_empty_ray_pairs=not no_empty_ray_pair_dedupe,
        profile_timing=profile_timing,
        seed=seed,
    )
    obs = env.reset()
    print(f"[smoke-probe] reset obs_shape={obs.shape}", flush=True)

    # Six canonical Cube Mode viewpoints around the normalized object.
    action_cycle = np.array([
        [40, 40, 80, 0, 0, 0],
        [80, 40, 40, 0, 0, 0],
        [0, 40, 40, 0, 0, 0],
        [40, 80, 40, 0, 0, 0],
        [40, 0, 40, 0, 0, 0],
        [40, 40, 0, 0, 0, 0],
    ], dtype=np.int64)

    total_wall = 0.0
    final_cr = 0.0
    for step in range(max_steps):
        action = np.repeat(action_cycle[step % len(action_cycle)][None, :], n_envs, axis=0)
        t0 = time.perf_counter()
        env.step_async(action)
        obs, rewards, dones, infos = env.step_wait()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        total_wall += dt

        cr = np.array([float(info.get("cr", 0.0)) for info in infos], dtype=np.float32)
        collision = np.array([bool(info.get("collision", False)) for info in infos], dtype=bool)
        early = np.array([bool(info.get("early_stopped", False)) for info in infos], dtype=bool)
        timeout = np.array([bool(info.get("TimeLimit.truncated", False)) for info in infos], dtype=bool)
        cov_delta = np.array([float(info.get("coverage_delta", 0.0)) for info in infos], dtype=np.float32)
        new_gt = np.array([float(info.get("new_gt_voxels", 0.0)) for info in infos], dtype=np.float32)
        visible_gt = np.array([float(info.get("visible_gt_voxels", 0.0)) for info in infos], dtype=np.float32)
        final_cr = float(cr.mean()) if len(cr) else 0.0

        payload = {
            "probe/step_wall_s": float(dt),
            "probe/batch_steps_per_sec": float(1.0 / max(dt, 1e-9)),
            "probe/env_steps_per_sec": float(n_envs / max(dt, 1e-9)),
            "probe/reward_mean": float(np.mean(rewards)),
            "probe/reward_min": float(np.min(rewards)),
            "probe/reward_max": float(np.max(rewards)),
            "probe/cr_mean": float(cr.mean()),
            "probe/cr_max": float(cr.max()),
            "probe/coverage_delta_mean": float(cov_delta.mean()),
            "probe/new_gt_voxels_mean": float(new_gt.mean()),
            "probe/visible_gt_voxels_mean": float(visible_gt.mean()),
            "probe/collision_rate": float(collision.mean()),
            "probe/early_stop_rate": float(early.mean()),
            "probe/timeout_rate": float(timeout.mean()),
        }
        prof = getattr(env, "last_step_profile", {}) or {}
        for key, value in prof.items():
            if not isinstance(value, (float, int)):
                continue
            if key.endswith("_s"):
                payload[f"profile/{key[:-2]}_ms"] = float(value) * 1000.0
            else:
                payload[f"profile/{key}"] = float(value)
        wandb.log(payload, step=step + 1)
        profile_msg = ""
        if profile_timing and prof:
            render_ratio = float(prof.get("render/active_ratio", 1.0))
            pairs_before = int(float(prof.get("free/empty_ray_pairs_before_unique", 0.0)))
            pairs_after = int(float(prof.get("free/empty_ray_pairs_after_unique", 0.0)))
            miss_intersect = int(float(prof.get("free/miss_intersect_rays", 0.0)))
            triton_free = int(float(prof.get("free/triton_free_raycast", 0.0)))
            triton_hit = int(float(prof.get("free/triton_hit_raycast", 0.0)))
            cuda_free = int(float(prof.get("free/cuda_free_raycast", 0.0)))
            cuda_hit = int(float(prof.get("free/cuda_hit_raycast", 0.0)))
            triton_apply = int(float(prof.get("free/free_mask_apply_triton", 0.0)))
            dense_apply = int(float(prof.get("free/free_mask_apply_dense", 0.0)))
            index_apply = int(float(prof.get("free/free_mask_apply_index", 0.0)))
            hit_max_delta = int(float(prof.get("free/hit_max_delta", 0.0)))
            profile_msg = (
                f" render={float(prof.get('render_s', 0.0)) * 1000.0:.1f}ms"
                f" render_active={render_ratio:.3f}"
                f" free={float(prof.get('free_update_s', 0.0)) * 1000.0:.1f}ms"
                f" backend={prof.get('free/free_raycast_backend_used', '') or 'none'}"
                f" cuda_free={cuda_free}"
                f" cuda_hit={cuda_hit}"
                f" triton_free={triton_free}"
                f" triton_hit={triton_hit}"
                f" hit_dmax={hit_max_delta}"
                f" miss_hit_aabb={miss_intersect}"
                f" empty_pairs={pairs_before}->{pairs_after}"
                f" obs={float(prof.get('obs_build_s', 0.0)) * 1000.0:.1f}ms"
                f" free_vox={int(float(prof.get('free/union_free_voxels', 0.0)))}"
                f" apply={free_mask_apply_mode}:T{triton_apply}/D{dense_apply}/I{index_apply}"
                f" block={triton_bresenham_block_rays}"
                f" scatter_ms="
                f"{float(prof.get('free/hit_scatter_s', 0.0)) * 1000.0:.1f}+"
                f"{float(prof.get('free/empty_scatter_s', 0.0)) * 1000.0:.1f}"
                f" pair_ms={float(prof.get('free/empty_pairs_s', 0.0)) * 1000.0:.1f}"
                f" zero_ms={float(prof.get('free/mask_alloc_s', 0.0)) * 1000.0:.1f}"
                f" count_ms={float(prof.get('free/mask_count_s', 0.0)) * 1000.0:.1f}"
                f" apply_ms={float(prof.get('free/mask_apply_s', 0.0)) * 1000.0:.1f}"
            )
        print(
            f"[smoke-probe] step={step + 1}/{max_steps} "
            f"dt={dt:.3f}s env_fps={payload['probe/env_steps_per_sec']:.3f} "
            f"reward={payload['probe/reward_mean']:.4f} "
            f"cr={payload['probe/cr_mean']:.4f} "
            f"dcr={payload['probe/coverage_delta_mean']:.4f} "
            f"new_gt={payload['probe/new_gt_voxels_mean']:.1f} "
            f"vis_gt={payload['probe/visible_gt_voxels_mean']:.1f} "
            f"collision={payload['probe/collision_rate']:.2f} "
            f"done={bool(np.any(dones))}"
            f"{profile_msg}",
            flush=True,
        )
        if bool(np.all(dones)):
            break

    summary = {
        "final_cr": final_cr,
        "total_wall_s": total_wall,
        "mean_env_steps_per_sec": float((step + 1) * n_envs / max(total_wall, 1e-9)),
        "reached_threshold": bool(final_cr > coverage_threshold),
    }
    wandb.summary.update(summary)
    print(f"[smoke-probe] summary={summary}", flush=True)
    env.close()
    wandb.finish()
    vol_runs.commit()


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data},
    timeout=2 * 3600,
    retries=0,
)
def benchmark_free_update_blocks(
    block_sizes: str = "16,32,64,128",
    seq_name: str = "",
    synsets: str = "03001627",
    limit_per_synset: int = 8,
    pool_size: int = 8,
    preproc_dir: str = "/data/geoscout_preproc_g128",
    n_envs: int = 32,
    max_steps: int = 3,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    max_faces: int = 5000,
    seed: int = 0,
):
    """Benchmark Triton Bresenham vector width inside one Modal job."""
    import json
    import time
    from pathlib import Path

    import numpy as np
    import torch

    from geoscout.data import list_shapenet
    from geoscout.tensor_env import TensorBatchEnv

    parsed_blocks = [int(x.strip()) for x in block_sizes.split(",") if x.strip()]
    if not parsed_blocks:
        raise ValueError("block_sizes must contain at least one integer")

    synset_list = [s.strip() for s in synsets.split(",") if s.strip()] or None
    entries = list_shapenet(
        Path("/data/ShapeNetCore.v2"),
        synsets=synset_list,
        limit_per_synset=limit_per_synset,
    )
    if seq_name:
        entries = [e for e in entries if e.name == seq_name] or [
            e for e in list_shapenet(Path("/data/ShapeNetCore.v2"))
            if e.name == seq_name
        ]
    if not entries:
        raise RuntimeError(f"No ShapeNet entries for seq_name={seq_name!r}")
    pool_entries = entries[:max(1, int(pool_size))]
    mesh_paths = [e.mesh_path for e in pool_entries]
    preproc_paths = [Path(preproc_dir) / f"{e.name}.pt" for e in pool_entries]
    missing_pp = [str(p) for p in preproc_paths if not p.exists()]
    if missing_pp:
        raise FileNotFoundError(f"preproc missing for pool: {missing_pp[:5]}")

    print(
        f"[free-block-bench] samples={[e.name for e in pool_entries]} "
        f"blocks={parsed_blocks} n_envs={n_envs} max_steps={max_steps}",
        flush=True,
    )
    env = TensorBatchEnv(
        num_envs=n_envs,
        mesh_paths=mesh_paths,
        preproc_paths=preproc_paths,
        device="cuda",
        buffer_size=30,
        grid_size=grid_size,
        obs_grid_size=obs_grid_size,
        episode_len=max_steps,
        render_size=image_size,
        fov_deg=60.0,
        cr_success_threshold=0.99,
        coverage_reward_scale=20.0,
        short_path_grace=30,
        short_path_clip=2.0,
        short_path_scale=0.1,
        only_positive_rewards=True,
        skip_free_raycast=False,
        update_empty_rays=True,
        coverage_hit_dilate_radius=1,
        caption_dim=384,
        auto_lookat_center=True,
        max_faces=max_faces,
        renderer_backend="nvdiffrast",
        renderer_bbox_ray_cull=True,
        use_triton_free_raycast=True,
        free_raycast_backend="triton",
        free_mask_apply_mode="triton",
        coverage_reward_type="linear",
        termination_bonus=1.0,
        collision_penalty=10.0,
        dedupe_empty_ray_pairs=False,
        profile_timing=True,
        seed=seed,
    )

    action_cycle = np.array([
        [40, 40, 80, 0, 0, 0],
        [80, 40, 40, 0, 0, 0],
        [0, 40, 40, 0, 0, 0],
        [40, 80, 40, 0, 0, 0],
        [40, 0, 40, 0, 0, 0],
        [40, 40, 0, 0, 0, 0],
    ], dtype=np.int64)

    results = []
    for block in parsed_blocks:
        env.triton_bresenham_block_rays = max(1, int(block))
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        env.reset()
        block_steps = []
        for step in range(int(max_steps)):
            action = np.repeat(action_cycle[step % len(action_cycle)][None, :], n_envs, axis=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            env.step_async(action)
            _, rewards, dones, infos = env.step_wait()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            prof = getattr(env, "last_step_profile", {}) or {}
            cr = np.array([float(info.get("cr", 0.0)) for info in infos], dtype=np.float32)
            row = {
                "step": step + 1,
                "wall_s": float(dt),
                "env_fps": float(n_envs / max(dt, 1e-9)),
                "reward_mean": float(np.mean(rewards)),
                "cr_mean": float(cr.mean()),
                "free_ms": float(prof.get("free_update_s", 0.0)) * 1000.0,
                "render_ms": float(prof.get("render_s", 0.0)) * 1000.0,
                "hit_scatter_ms": float(prof.get("free/hit_scatter_s", 0.0)) * 1000.0,
                "empty_scatter_ms": float(prof.get("free/empty_scatter_s", 0.0)) * 1000.0,
                "mask_apply_ms": float(prof.get("free/mask_apply_s", 0.0)) * 1000.0,
                "empty_pairs_after": int(float(prof.get("free/empty_ray_pairs_after_unique", 0.0))),
                "free_voxels": int(float(prof.get("free/union_free_voxels", 0.0))),
                "done_any": bool(np.any(dones)),
            }
            block_steps.append(row)
            print(
                "[free-block-bench] "
                + json.dumps({"block": block, **row}, sort_keys=True),
                flush=True,
            )
            if bool(np.all(dones)):
                break

        steady = block_steps[1:] or block_steps
        result = {
            "block_rays": int(block),
            "steps": block_steps,
            "steady_free_ms_mean": float(np.mean([s["free_ms"] for s in steady])),
            "steady_wall_s_mean": float(np.mean([s["wall_s"] for s in steady])),
            "steady_env_fps_mean": float(np.mean([s["env_fps"] for s in steady])),
            "steady_hit_scatter_ms_mean": float(np.mean([s["hit_scatter_ms"] for s in steady])),
            "steady_empty_scatter_ms_mean": float(np.mean([s["empty_scatter_ms"] for s in steady])),
        }
        results.append(result)

    env.close()
    summary = {
        "samples": [e.name for e in pool_entries],
        "n_envs": int(n_envs),
        "image_size": int(image_size),
        "grid_size": int(grid_size),
        "results": results,
    }
    print("[free-block-bench-summary] " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data},
    timeout=2 * 3600,
    retries=0,
)
def validate_preprocessing_examples_cuda_infra(
    sample_names: str = (
        "03001627_1006be65e7bc937e9141f9b58470d646,"
        "03001627_1007e20d5e811b308351982a6e40cf41,"
        "03001627_100b18376b885f206ae9ad7e32c4139d,"
        "03001627_1013f70851210a618f2e765c4a8ed3d,"
        "04256520_1037fd31d12178d396f164a988ef37cc,"
        "04256520_103b76b2594a1582eaf14273fa406ffc,"
        "04256520_104256e5bb73b0b719fb4103277a6b93,"
        "04256520_1050790962944624febad4f49b26ec52,"
        "04379243_1011e1c9812b84d2a9ed7bb5b55809f8,"
        "04379243_10139657dfa9afe0c3bd24f986301745,"
        "04379243_1028a9cbaa7a333230bbd4cddd04c77b,"
        "04379243_102f0532f9f8bbcdcb503f63ed915ed2"
    ),
    preproc_dir: str = "/data/geoscout_preproc_g128",
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    max_steps: int = 4,
    max_faces: int = 5000,
    seed: int = 0,
):
    """Validate custom CUDA free-space update on the preprocessing examples.

    Each env is pinned to one of the real samples used by
    ``preprocessing_viz_g128``.  The function runs the full TensorBatchEnv
    step path twice, once with the custom CUDA Bresenham scatter and once
    with the previous Triton scatter, and asserts exact agreement in the
    state/reward surfaces that the policy and reward consume.
    """
    import json
    import time
    from pathlib import Path

    import numpy as np
    import torch

    from geoscout.tensor_env import TensorBatchEnv

    names = [s.strip() for s in str(sample_names).split(",") if s.strip()]
    if not names:
        raise ValueError("sample_names must contain at least one sample")

    shapenet_root = Path("/data/ShapeNetCore.v2")
    preproc_root = Path(preproc_dir)
    mesh_paths = []
    preproc_paths = []
    for name in names:
        synset, model_id = name.split("_", 1)
        mesh = shapenet_root / synset / model_id / "models" / "model_normalized.obj"
        pp = preproc_root / f"{name}.pt"
        if not mesh.exists():
            raise FileNotFoundError(f"missing mesh for {name}: {mesh}")
        if not pp.exists():
            raise FileNotFoundError(f"missing preproc for {name}: {pp}")
        mesh_paths.append(mesh)
        preproc_paths.append(pp)

    n_envs = len(names)
    print(
        f"[preproc-cuda-validate] samples={names} n_envs={n_envs} "
        f"image_size={image_size} grid={grid_size} obs_grid={obs_grid_size} "
        f"max_steps={max_steps}",
        flush=True,
    )

    def make_env(backend: str) -> TensorBatchEnv:
        return TensorBatchEnv(
            num_envs=n_envs,
            mesh_paths=mesh_paths,
            preproc_paths=preproc_paths,
            device="cuda",
            buffer_size=30,
            grid_size=grid_size,
            obs_grid_size=obs_grid_size,
            episode_len=max_steps + 1,
            render_size=image_size,
            fov_deg=60.0,
            cr_success_threshold=2.0,
            coverage_reward_scale=20.0,
            short_path_grace=30,
            short_path_clip=2.0,
            short_path_scale=0.1,
            only_positive_rewards=True,
            skip_free_raycast=False,
            update_empty_rays=True,
            coverage_hit_dilate_radius=1,
            caption_dim=384,
            auto_lookat_center=True,
            max_faces=max_faces,
            renderer_backend="nvdiffrast",
            renderer_bbox_ray_cull=True,
            use_triton_free_raycast=True,
            free_raycast_backend=backend,
            free_mask_apply_mode="triton",
            coverage_reward_type="linear",
            termination_bonus=1.0,
            collision_penalty=10.0,
            dedupe_empty_ray_pairs=False,
            profile_timing=True,
            seed=seed,
        )

    def pin_env_to_preprocessing_order(env: TensorBatchEnv) -> np.ndarray:
        device = env.device
        ids = torch.arange(n_envs, dtype=torch.long, device=device)
        env._prob_grid[ids] = 0.0
        env._scanned_gt_grid[ids] = 0.0
        env._step_idx[ids] = 0
        env._cr_prev[ids] = 0.0
        env._action_history[ids] = 0.0
        env._ep_step_count[ids] = 0
        env._ep_reward_sum[ids] = 0.0
        env._ep_new_gt_sum[ids] = 0.0
        env._ep_visible_gt_sum[ids] = 0.0
        env._ep_redundant_gt_sum[ids] = 0.0
        env._ep_revisit_sum[ids] = 0.0

        env._env_mesh_id[ids] = ids
        if env._pool_grid_on_device:
            selected_grid = env._pool_grid_gt[ids]
        else:
            selected_grid = env._pool_grid_gt[ids.detach().cpu()].to(
                device, dtype=torch.float32, non_blocking=True,
            )
        env._grid_gt[ids] = selected_grid
        env._bbox_min[ids] = env._pool_bbox_min[ids]
        env._voxel_size[ids] = env._pool_voxel_size[ids]
        env._num_valid_gt_per_env[ids] = env._pool_num_valid[ids].clamp(min=1.0)
        if env._caption_emb is not None:
            env._caption_emb[ids] = env._pool_caption_emb[ids]

        ix0 = env._idx_up[0] // 2
        iy0 = env._idx_up[1] // 2
        iz0 = env._idx_up[2]
        init_idx = torch.stack([
            ix0.expand(n_envs),
            iy0.expand(n_envs),
            iz0.expand(n_envs),
            torch.zeros(n_envs, dtype=torch.long, device=device),
            torch.zeros(n_envs, dtype=torch.long, device=device),
            torch.zeros(n_envs, dtype=torch.long, device=device),
        ], dim=-1)
        init_pose = init_idx.float() * env._action_unit + env._action_low
        env._action_history[ids, -1, :] = init_pose
        env._last_action_idx[ids] = init_idx
        return env._build_observation_np()

    cuda_env = make_env("cuda")
    triton_env = make_env("triton")
    obs_cuda = pin_env_to_preprocessing_order(cuda_env)
    obs_triton = pin_env_to_preprocessing_order(triton_env)
    reset_obs_diff = float(np.max(np.abs(obs_cuda - obs_triton)))
    if reset_obs_diff != 0.0:
        raise AssertionError(f"reset observation mismatch: {reset_obs_diff}")

    action_cycle = np.array([
        [40, 40, 80, 0, 0, 0],
        [80, 40, 40, 0, 0, 0],
        [0, 40, 40, 0, 0, 0],
        [40, 80, 40, 0, 0, 0],
        [40, 0, 40, 0, 0, 0],
        [40, 40, 0, 0, 0, 0],
    ], dtype=np.int64)

    rows = []
    for step in range(int(max_steps)):
        action = np.repeat(action_cycle[step % len(action_cycle)][None, :], n_envs, axis=0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cuda_env.step_async(action)
        obs_cuda, rew_cuda, done_cuda, info_cuda = cuda_env.step_wait()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cuda_wall = time.perf_counter() - t0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        triton_env.step_async(action)
        obs_triton, rew_triton, done_triton, info_triton = triton_env.step_wait()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        triton_wall = time.perf_counter() - t0

        prob_diff = float((cuda_env._prob_grid - triton_env._prob_grid).abs().max().detach().cpu().item())
        scanned_diff = float((cuda_env._scanned_gt_grid - triton_env._scanned_gt_grid).abs().max().detach().cpu().item())
        cr_diff = float(abs(
            (cuda_env._scanned_gt_grid.sum(dim=(1, 2, 3)) / cuda_env._num_valid_gt_per_env)
            - (triton_env._scanned_gt_grid.sum(dim=(1, 2, 3)) / triton_env._num_valid_gt_per_env)
        ).max().detach().cpu().item())
        reward_diff = float(np.max(np.abs(np.asarray(rew_cuda) - np.asarray(rew_triton))))
        done_cuda_arr = np.asarray(done_cuda, dtype=bool)
        done_triton_arr = np.asarray(done_triton, dtype=bool)
        done_match = bool(np.array_equal(done_cuda_arr, done_triton_arr))
        post_reset_obs_diff = float(np.max(np.abs(np.asarray(obs_cuda) - np.asarray(obs_triton))))
        active_obs_diff = 0.0
        if (~done_cuda_arr).any():
            active_obs_diff = float(np.max(np.abs(
                np.asarray(obs_cuda)[~done_cuda_arr] - np.asarray(obs_triton)[~done_cuda_arr]
            )))
        terminal_obs_diff = 0.0
        if done_cuda_arr.any():
            terminal_diffs = []
            for i, is_done in enumerate(done_cuda_arr):
                if not is_done:
                    continue
                term_cuda = info_cuda[i].get("terminal_observation")
                term_triton = info_triton[i].get("terminal_observation")
                if term_cuda is None or term_triton is None:
                    raise AssertionError(f"step {step + 1} missing terminal_observation for env {i}")
                terminal_diffs.append(float(np.max(np.abs(
                    np.asarray(term_cuda) - np.asarray(term_triton)
                ))))
            terminal_obs_diff = float(max(terminal_diffs)) if terminal_diffs else 0.0
        obs_diff = max(active_obs_diff, terminal_obs_diff)
        seq_match = [
            cu.get("seq_name", "") == tr.get("seq_name", "")
            for cu, tr in zip(info_cuda, info_triton)
        ]
        if prob_diff != 0.0 or scanned_diff != 0.0 or reward_diff != 0.0 or obs_diff != 0.0 or cr_diff != 0.0:
            raise AssertionError(
                f"step {step + 1} mismatch: prob={prob_diff} scanned={scanned_diff} "
                f"reward={reward_diff} obs={obs_diff} cr={cr_diff}"
            )
        if not done_match or not all(seq_match):
            raise AssertionError(
                f"step {step + 1} done/seq mismatch: done_match={done_match} "
                f"seq_match={seq_match}"
            )

        prof_cuda = getattr(cuda_env, "last_step_profile", {}) or {}
        prof_triton = getattr(triton_env, "last_step_profile", {}) or {}
        row = {
            "step": step + 1,
            "action": action_cycle[step % len(action_cycle)].tolist(),
            "done_any": bool(done_cuda_arr.any()),
            "done_count": int(done_cuda_arr.sum()),
            "cuda_wall_ms": float(cuda_wall * 1000.0),
            "triton_wall_ms": float(triton_wall * 1000.0),
            "cuda_env_fps": float(n_envs / max(cuda_wall, 1e-9)),
            "triton_env_fps": float(n_envs / max(triton_wall, 1e-9)),
            "cuda_render_ms": float(prof_cuda.get("render_s", 0.0)) * 1000.0,
            "triton_render_ms": float(prof_triton.get("render_s", 0.0)) * 1000.0,
            "cuda_free_ms": float(prof_cuda.get("free_update_s", 0.0)) * 1000.0,
            "triton_free_ms": float(prof_triton.get("free_update_s", 0.0)) * 1000.0,
            "cuda_hit_unique_ms": float(prof_cuda.get("free/hit_unique_s", 0.0)) * 1000.0,
            "triton_hit_unique_ms": float(prof_triton.get("free/hit_unique_s", 0.0)) * 1000.0,
            "cuda_hit_scatter_ms": float(prof_cuda.get("free/hit_scatter_s", 0.0)) * 1000.0,
            "triton_hit_scatter_ms": float(prof_triton.get("free/hit_scatter_s", 0.0)) * 1000.0,
            "cuda_hit_fallback_ms": float(prof_cuda.get("free/hit_fallback_s", 0.0)) * 1000.0,
            "triton_hit_fallback_ms": float(prof_triton.get("free/hit_fallback_s", 0.0)) * 1000.0,
            "cuda_empty_scatter_ms": float(prof_cuda.get("free/empty_scatter_s", 0.0)) * 1000.0,
            "triton_empty_scatter_ms": float(prof_triton.get("free/empty_scatter_s", 0.0)) * 1000.0,
            "cuda_clear_hit_ms": float(prof_cuda.get("free/clear_hit_endpoints_s", 0.0)) * 1000.0,
            "triton_clear_hit_ms": float(prof_triton.get("free/clear_hit_endpoints_s", 0.0)) * 1000.0,
            "cuda_scatter_ms": (
                float(prof_cuda.get("free/hit_scatter_s", 0.0))
                + float(prof_cuda.get("free/empty_scatter_s", 0.0))
            ) * 1000.0,
            "triton_scatter_ms": (
                float(prof_triton.get("free/hit_scatter_s", 0.0))
                + float(prof_triton.get("free/empty_scatter_s", 0.0))
            ) * 1000.0,
            "cuda_pair_ms": float(prof_cuda.get("free/empty_pairs_s", 0.0)) * 1000.0,
            "triton_pair_ms": float(prof_triton.get("free/empty_pairs_s", 0.0)) * 1000.0,
            "cuda_mask_alloc_ms": float(prof_cuda.get("free/mask_alloc_s", 0.0)) * 1000.0,
            "triton_mask_alloc_ms": float(prof_triton.get("free/mask_alloc_s", 0.0)) * 1000.0,
            "cuda_count_ms": float(prof_cuda.get("free/mask_count_s", 0.0)) * 1000.0,
            "triton_count_ms": float(prof_triton.get("free/mask_count_s", 0.0)) * 1000.0,
            "cuda_apply_ms": float(prof_cuda.get("free/mask_apply_s", 0.0)) * 1000.0,
            "triton_apply_ms": float(prof_triton.get("free/mask_apply_s", 0.0)) * 1000.0,
            "cuda_hit_valid_pixels": int(float(prof_cuda.get("free/hit_valid_pixels", 0.0))),
            "triton_hit_valid_pixels": int(float(prof_triton.get("free/hit_valid_pixels", 0.0))),
            "cuda_hit_unique_targets": int(float(prof_cuda.get("free/hit_unique_targets", 0.0))),
            "triton_hit_unique_targets": int(float(prof_triton.get("free/hit_unique_targets", 0.0))),
            "cuda_miss_pixels": int(float(prof_cuda.get("free/miss_pixels", 0.0))),
            "triton_miss_pixels": int(float(prof_triton.get("free/miss_pixels", 0.0))),
            "cuda_empty_pairs_after": int(float(prof_cuda.get("free/empty_ray_pairs_after_unique", 0.0))),
            "triton_empty_pairs_after": int(float(prof_triton.get("free/empty_ray_pairs_after_unique", 0.0))),
            "cuda_free_voxels": int(float(prof_cuda.get("free/union_free_voxels", 0.0))),
            "triton_free_voxels": int(float(prof_triton.get("free/union_free_voxels", 0.0))),
            "prob_diff": prob_diff,
            "scanned_diff": scanned_diff,
            "reward_diff": reward_diff,
            "obs_diff": obs_diff,
            "active_obs_diff": active_obs_diff,
            "terminal_obs_diff": terminal_obs_diff,
            "post_reset_obs_diff": post_reset_obs_diff,
            "cr_diff": cr_diff,
            "done_match": done_match,
        }
        rows.append(row)
        print("[preproc-cuda-validate-step] " + json.dumps(row, sort_keys=True), flush=True)

    steady = rows[1:] if len(rows) > 1 else rows
    nonterminal_steady = [r for r in steady if not r.get("done_any", False)]
    if not nonterminal_steady:
        nonterminal_steady = steady
    summary = {
        "samples": names,
        "n_envs": n_envs,
        "image_size": int(image_size),
        "grid_size": int(grid_size),
        "obs_grid_size": int(obs_grid_size),
        "max_prob_diff": float(max(r["prob_diff"] for r in rows)),
        "max_scanned_diff": float(max(r["scanned_diff"] for r in rows)),
        "max_reward_diff": float(max(r["reward_diff"] for r in rows)),
        "max_obs_diff": float(max(r["obs_diff"] for r in rows)),
        "max_cr_diff": float(max(r["cr_diff"] for r in rows)),
        "steady_cuda_wall_ms_mean": float(np.mean([r["cuda_wall_ms"] for r in steady])),
        "steady_triton_wall_ms_mean": float(np.mean([r["triton_wall_ms"] for r in steady])),
        "steady_cuda_free_ms_mean": float(np.mean([r["cuda_free_ms"] for r in steady])),
        "steady_triton_free_ms_mean": float(np.mean([r["triton_free_ms"] for r in steady])),
        "steady_cuda_scatter_ms_mean": float(np.mean([r["cuda_scatter_ms"] for r in steady])),
        "steady_triton_scatter_ms_mean": float(np.mean([r["triton_scatter_ms"] for r in steady])),
        "steady_cuda_env_fps_mean": float(np.mean([r["cuda_env_fps"] for r in steady])),
        "steady_triton_env_fps_mean": float(np.mean([r["triton_env_fps"] for r in steady])),
        "nonterminal_cuda_wall_ms_mean": float(np.mean([r["cuda_wall_ms"] for r in nonterminal_steady])),
        "nonterminal_triton_wall_ms_mean": float(np.mean([r["triton_wall_ms"] for r in nonterminal_steady])),
        "nonterminal_cuda_free_ms_mean": float(np.mean([r["cuda_free_ms"] for r in nonterminal_steady])),
        "nonterminal_triton_free_ms_mean": float(np.mean([r["triton_free_ms"] for r in nonterminal_steady])),
        "nonterminal_cuda_scatter_ms_mean": float(np.mean([r["cuda_scatter_ms"] for r in nonterminal_steady])),
        "nonterminal_triton_scatter_ms_mean": float(np.mean([r["triton_scatter_ms"] for r in nonterminal_steady])),
        "nonterminal_cuda_env_fps_mean": float(np.mean([r["cuda_env_fps"] for r in nonterminal_steady])),
        "nonterminal_triton_env_fps_mean": float(np.mean([r["triton_env_fps"] for r in nonterminal_steady])),
        "rows": rows,
    }
    print("[preproc-cuda-validate-summary] " + json.dumps(summary, sort_keys=True), flush=True)
    cuda_env.close()
    triton_env.close()
    return summary


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data},
    timeout=2 * 3600,
    retries=0,
)
def validate_cuda_empty_pair_builder(
    sample_names: str = (
        "03001627_1006be65e7bc937e9141f9b58470d646,"
        "04256520_1037fd31d12178d396f164a988ef37cc,"
        "04379243_1011e1c9812b84d2a9ed7bb5b55809f8"
    ),
    preproc_dir: str = "/data/geoscout_preproc_g128",
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    max_steps: int = 5,
    max_faces: int = 5000,
    no_empty_ray_pair_dedupe: bool = True,
    seed: int = 0,
):
    """A/B the batched CUDA empty-ray pair builder against the old loop."""
    import json
    import os
    import time
    from pathlib import Path

    import numpy as np
    import torch

    from geoscout.cuda_bresenham import empty_ray_pairs_cuda
    from geoscout.tensor_env import TensorBatchEnv

    names = [s.strip() for s in str(sample_names).split(",") if s.strip()]
    shapenet_root = Path("/data/ShapeNetCore.v2")
    preproc_root = Path(preproc_dir)
    mesh_paths = []
    preproc_paths = []
    for name in names:
        synset, model_id = name.split("_", 1)
        mesh = shapenet_root / synset / model_id / "models" / "model_normalized.obj"
        pp = preproc_root / f"{name}.pt"
        if not mesh.exists():
            raise FileNotFoundError(mesh)
        if not pp.exists():
            raise FileNotFoundError(pp)
        mesh_paths.append(mesh)
        preproc_paths.append(pp)

    n_envs = len(names)
    print(
        f"[empty-pair-validate] samples={names} n_envs={n_envs} "
        f"image_size={image_size} max_steps={max_steps} "
        f"dedupe_empty_ray_pairs={not no_empty_ray_pair_dedupe}",
        flush=True,
    )

    def make_env() -> TensorBatchEnv:
        return TensorBatchEnv(
            num_envs=n_envs,
            mesh_paths=mesh_paths,
            preproc_paths=preproc_paths,
            device="cuda",
            buffer_size=30,
            grid_size=grid_size,
            obs_grid_size=obs_grid_size,
            episode_len=max_steps + 1,
            render_size=image_size,
            fov_deg=60.0,
            cr_success_threshold=2.0,
            coverage_reward_scale=20.0,
            short_path_grace=30,
            short_path_clip=2.0,
            short_path_scale=0.1,
            only_positive_rewards=True,
            skip_free_raycast=False,
            update_empty_rays=True,
            coverage_hit_dilate_radius=1,
            caption_dim=384,
            auto_lookat_center=True,
            max_faces=max_faces,
            renderer_backend="nvdiffrast",
            renderer_bbox_ray_cull=True,
            use_triton_free_raycast=True,
            free_raycast_backend="cuda",
            free_mask_apply_mode="triton",
            coverage_reward_type="linear",
            termination_bonus=1.0,
            collision_penalty=10.0,
            dedupe_empty_ray_pairs=not no_empty_ray_pair_dedupe,
            profile_timing=True,
            seed=seed,
        )

    def pin_env_to_preprocessing_order(env: TensorBatchEnv) -> np.ndarray:
        device = env.device
        ids = torch.arange(n_envs, dtype=torch.long, device=device)
        env._prob_grid[ids] = 0.0
        env._scanned_gt_grid[ids] = 0.0
        env._step_idx[ids] = 0
        env._cr_prev[ids] = 0.0
        env._action_history[ids] = 0.0
        env._ep_step_count[ids] = 0
        env._ep_reward_sum[ids] = 0.0
        env._ep_new_gt_sum[ids] = 0.0
        env._ep_visible_gt_sum[ids] = 0.0
        env._ep_redundant_gt_sum[ids] = 0.0
        env._ep_revisit_sum[ids] = 0.0
        env._env_mesh_id[ids] = ids
        selected_grid = env._pool_grid_gt[ids.detach().cpu()].to(
            device, dtype=torch.float32, non_blocking=True,
        )
        env._grid_gt[ids] = selected_grid
        env._bbox_min[ids] = env._pool_bbox_min[ids]
        env._voxel_size[ids] = env._pool_voxel_size[ids]
        env._num_valid_gt_per_env[ids] = env._pool_num_valid[ids].clamp(min=1.0)
        if env._caption_emb is not None:
            env._caption_emb[ids] = env._pool_caption_emb[ids]

        ix0 = env._idx_up[0] // 2
        iy0 = env._idx_up[1] // 2
        iz0 = env._idx_up[2]
        init_idx = torch.stack([
            ix0.expand(n_envs),
            iy0.expand(n_envs),
            iz0.expand(n_envs),
            torch.zeros(n_envs, dtype=torch.long, device=device),
            torch.zeros(n_envs, dtype=torch.long, device=device),
            torch.zeros(n_envs, dtype=torch.long, device=device),
        ], dim=-1)
        init_pose = init_idx.float() * env._action_unit + env._action_low
        env._action_history[ids, -1, :] = init_pose
        env._last_action_idx[ids] = init_idx
        return env._build_observation_np()

    fast_env = make_env()
    ref_env = make_env()
    obs_fast = pin_env_to_preprocessing_order(fast_env)
    obs_ref = pin_env_to_preprocessing_order(ref_env)
    reset_obs_diff = float(np.max(np.abs(obs_fast - obs_ref)))
    if reset_obs_diff != 0.0:
        raise AssertionError(f"reset observation mismatch: {reset_obs_diff}")

    action_cycle = np.array([
        [40, 40, 80, 0, 0, 0],
        [80, 40, 40, 0, 0, 0],
        [0, 40, 40, 0, 0, 0],
        [40, 80, 40, 0, 0, 0],
        [40, 0, 40, 0, 0, 0],
    ], dtype=np.int64)

    def compute_empty_pair_rows_same_inputs(env: TensorBatchEnv, action_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compare CUDA and reference builders using one shared render/mask."""
        actions_t = torch.from_numpy(np.asarray(action_np, dtype=np.int64)).to(env.device)
        actions_t = actions_t.clamp(env._idx_low, env._idx_up)
        _, eyes, ats = env._decode_actions(actions_t)
        eye_idx = torch.floor((eyes - env._bbox_min) / env._voxel_size).long()
        in_box = ((eye_idx >= 0) & (eye_idx < env.grid_size)).all(dim=-1)
        eye_idx_clamp = eye_idx.clamp(0, env.grid_size - 1)
        env_arange_t = torch.arange(env.num_envs, device=env.device)
        eye_on_gt_surface = env._grid_gt[
            env_arange_t, eye_idx_clamp[:, 0], eye_idx_clamp[:, 1], eye_idx_clamp[:, 2]
        ] > 0.5
        surface_collision = in_box & eye_on_gt_surface
        mesh_inside = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for mid in torch.unique(env._env_mesh_id).tolist():
            sel = (env._env_mesh_id == mid).nonzero().flatten()
            mesh_inside[sel] = env._renderers[int(mid)].points_inside_mesh(eyes[sel])
        collision = surface_collision | mesh_inside
        active_env = ~collision

        H = W = env.render_size
        alpha = torch.zeros(env.num_envs, H, W, dtype=torch.float32, device=env.device)
        for mid in torch.unique(env._env_mesh_id).tolist():
            sel = (env._env_mesh_id == mid).nonzero().flatten()
            out = env._renderers[int(mid)].render_batch(eyes[sel], ats[sel])
            alpha[sel] = out.alpha
        rays_world = env._world_rays(eyes, ats)
        hit_pixel_mask = (alpha.view(env.num_envs, H * W) > 0.5) & active_env.view(env.num_envs, 1)
        free_hit_mask = hit_pixel_mask | collision.view(env.num_envs, 1)

        cuda_out = empty_ray_pairs_cuda(
            eyes,
            rays_world,
            free_hit_mask,
            env._bbox_min,
            env._voxel_size,
            grid_size=env.grid_size,
            dedupe=bool(getattr(env, "dedupe_empty_ray_pairs", False)),
        )
        if cuda_out is None:
            raise RuntimeError("empty_ray_pairs_cuda returned None in same-input diagnostic")
        cuda_env, cuda_src, cuda_tgt, _, _ = cuda_out
        if cuda_src.numel() == 0:
            cuda_rows = np.empty((0, 7), dtype=np.int64)
        else:
            cuda_rows = torch.cat([cuda_env.view(-1, 1), cuda_src, cuda_tgt], dim=1).detach().cpu().numpy()

        ref_parts = []
        for env_idx in range(env.num_envs):
            src, tgt = env._empty_ray_pairs_for_env(
                env_idx,
                eyes[env_idx],
                rays_world[env_idx],
                ~free_hit_mask[env_idx],
                stats=None,
            )
            if src.numel() == 0:
                continue
            ref_parts.append(torch.cat([
                torch.full((src.shape[0], 1), env_idx, dtype=torch.long, device=env.device),
                src,
                tgt,
            ], dim=1))
        if ref_parts:
            ref_rows = torch.cat(ref_parts, dim=0).detach().cpu().numpy()
        else:
            ref_rows = np.empty((0, 7), dtype=np.int64)
        return (
            cuda_rows.astype(np.int64, copy=False),
            ref_rows.astype(np.int64, copy=False),
        )

    def diagnose_same_input_cuda_examples(
        env: TensorBatchEnv,
        action_np: np.ndarray,
        examples: list[list[int]],
    ) -> list[dict]:
        """Attach ray ids and PyTorch reference numerics to CUDA-only pair rows."""
        if not examples:
            return []
        actions_t = torch.from_numpy(np.asarray(action_np, dtype=np.int64)).to(env.device)
        actions_t = actions_t.clamp(env._idx_low, env._idx_up)
        _, eyes, ats = env._decode_actions(actions_t)
        eye_idx = torch.floor((eyes - env._bbox_min) / env._voxel_size).long()
        in_box = ((eye_idx >= 0) & (eye_idx < env.grid_size)).all(dim=-1)
        eye_idx_clamp = eye_idx.clamp(0, env.grid_size - 1)
        env_arange_t = torch.arange(env.num_envs, device=env.device)
        eye_on_gt_surface = env._grid_gt[
            env_arange_t, eye_idx_clamp[:, 0], eye_idx_clamp[:, 1], eye_idx_clamp[:, 2]
        ] > 0.5
        surface_collision = in_box & eye_on_gt_surface
        mesh_inside = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for mid in torch.unique(env._env_mesh_id).tolist():
            sel = (env._env_mesh_id == mid).nonzero().flatten()
            mesh_inside[sel] = env._renderers[int(mid)].points_inside_mesh(eyes[sel])
        collision = surface_collision | mesh_inside
        active_env = ~collision

        H = W = env.render_size
        alpha = torch.zeros(env.num_envs, H, W, dtype=torch.float32, device=env.device)
        for mid in torch.unique(env._env_mesh_id).tolist():
            sel = (env._env_mesh_id == mid).nonzero().flatten()
            out = env._renderers[int(mid)].render_batch(eyes[sel], ats[sel])
            alpha[sel] = out.alpha
        rays_world = env._world_rays(eyes, ats)
        hit_pixel_mask = (alpha.view(env.num_envs, H * W) > 0.5) & active_env.view(env.num_envs, 1)
        free_hit_mask = hit_pixel_mask | collision.view(env.num_envs, 1)

        cuda_out = empty_ray_pairs_cuda(
            eyes,
            rays_world,
            free_hit_mask,
            env._bbox_min,
            env._voxel_size,
            grid_size=env.grid_size,
            dedupe=False,
            return_ray_indices=True,
        )
        if cuda_out is None:
            raise RuntimeError("empty_ray_pairs_cuda returned None in detailed diagnostic")
        cuda_env, cuda_src, cuda_tgt, _, _, ray_ids = cuda_out
        rows_t = torch.cat([cuda_env.view(-1, 1), cuda_src, cuda_tgt], dim=1)
        details = []
        for ex in examples:
            ex_t = torch.tensor(ex, dtype=torch.long, device=env.device)
            match = torch.nonzero((rows_t == ex_t.view(1, -1)).all(dim=1), as_tuple=False).flatten()
            if match.numel() == 0:
                details.append({"row": ex, "found": False})
                continue
            m = match[0]
            env_idx = int(cuda_env[m].detach().cpu().item())
            ray_id = int(ray_ids[m].detach().cpu().item())
            ray = rays_world[env_idx, ray_id]
            eye = eyes[env_idx]
            bbox_min_t = env._bbox_min[env_idx]
            voxel_size_t = env._voxel_size[env_idx]
            bbox_max_t = bbox_min_t + float(env.grid_size) * voxel_size_t
            eps_dir = 1e-9
            dir_safe = torch.where(
                ray.abs() < eps_dir,
                torch.where(ray >= 0.0, torch.full_like(ray, eps_dir), torch.full_like(ray, -eps_dir)),
                ray,
            )
            t0 = (bbox_min_t - eye) / dir_safe
            t1 = (bbox_max_t - eye) / dir_safe
            t_near = torch.minimum(t0, t1).amax()
            t_far = torch.maximum(t0, t1).amin()
            t_start = t_near.clamp(min=0.0)
            intersects = bool((t_far > (t_start + 1e-6)).detach().cpu().item())
            ref_start = None
            ref_end = None
            if intersects:
                voxel_eps = float(voxel_size_t.min().detach().item()) * 0.25
                t_entry = torch.where(t_near <= 0.0, torch.zeros_like(t_start), t_start + voxel_eps)
                t_exit = torch.maximum(t_far - voxel_eps, t_entry)
                start_pt = eye + t_entry * ray
                end_pt = eye + t_exit * ray
                start_idx = torch.floor((start_pt - bbox_min_t) / voxel_size_t).long().clamp(0, env.grid_size - 1)
                end_idx = torch.floor((end_pt - bbox_min_t) / voxel_size_t).long().clamp(0, env.grid_size - 1)
                ref_start = start_idx.detach().cpu().tolist()
                ref_end = end_idx.detach().cpu().tolist()
            details.append({
                "row": ex,
                "found": True,
                "env": env_idx,
                "ray_id": ray_id,
                "pixel_yx": [int(ray_id // W), int(ray_id % W)],
                "free_hit_mask": bool(free_hit_mask[env_idx, ray_id].detach().cpu().item()),
                "eye": eye.detach().cpu().tolist(),
                "ray": ray.detach().cpu().tolist(),
                "dir_safe": dir_safe.detach().cpu().tolist(),
                "t0": t0.detach().cpu().tolist(),
                "t1": t1.detach().cpu().tolist(),
                "t_near": float(t_near.detach().cpu().item()),
                "t_far": float(t_far.detach().cpu().item()),
                "t_start": float(t_start.detach().cpu().item()),
                "gap": float((t_far - t_start).detach().cpu().item()),
                "intersects_ref": intersects,
                "ref_start": ref_start,
                "ref_end": ref_end,
            })
        return details

    def compute_empty_pair_rows(env: TensorBatchEnv, action_np: np.ndarray, *, use_cuda: bool) -> np.ndarray:
        """Recompute the empty-ray pair rows for diagnostics without mutating grids."""
        actions_t = torch.from_numpy(np.asarray(action_np, dtype=np.int64)).to(env.device)
        actions_t = actions_t.clamp(env._idx_low, env._idx_up)
        _, eyes, ats = env._decode_actions(actions_t)
        eye_idx = torch.floor((eyes - env._bbox_min) / env._voxel_size).long()
        in_box = ((eye_idx >= 0) & (eye_idx < env.grid_size)).all(dim=-1)
        eye_idx_clamp = eye_idx.clamp(0, env.grid_size - 1)
        env_arange_t = torch.arange(env.num_envs, device=env.device)
        eye_on_gt_surface = env._grid_gt[
            env_arange_t, eye_idx_clamp[:, 0], eye_idx_clamp[:, 1], eye_idx_clamp[:, 2]
        ] > 0.5
        surface_collision = in_box & eye_on_gt_surface
        mesh_inside = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for mid in torch.unique(env._env_mesh_id).tolist():
            sel = (env._env_mesh_id == mid).nonzero().flatten()
            mesh_inside[sel] = env._renderers[int(mid)].points_inside_mesh(eyes[sel])
        collision = surface_collision | mesh_inside
        active_env = ~collision

        H = W = env.render_size
        depth = torch.zeros(env.num_envs, H, W, dtype=torch.float32, device=env.device)
        alpha = torch.zeros(env.num_envs, H, W, dtype=torch.float32, device=env.device)
        direct_points = None
        for mid in torch.unique(env._env_mesh_id).tolist():
            sel = (env._env_mesh_id == mid).nonzero().flatten()
            out = env._renderers[int(mid)].render_batch(eyes[sel], ats[sel])
            depth[sel] = out.depth
            alpha[sel] = out.alpha
            if out.points is not None:
                if direct_points is None:
                    direct_points = torch.zeros(
                        env.num_envs, H, W, 3, dtype=torch.float32, device=env.device,
                    )
                direct_points[sel] = out.points.to(device=env.device, dtype=torch.float32)
        rays_world = env._world_rays(eyes, ats)
        hit_pixel_mask = (alpha.view(env.num_envs, H * W) > 0.5) & active_env.view(env.num_envs, 1)
        free_hit_mask = hit_pixel_mask | collision.view(env.num_envs, 1)
        if use_cuda:
            out = empty_ray_pairs_cuda(
                eyes,
                rays_world,
                free_hit_mask,
                env._bbox_min,
                env._voxel_size,
                grid_size=env.grid_size,
                dedupe=bool(getattr(env, "dedupe_empty_ray_pairs", False)),
            )
            if out is None:
                raise RuntimeError("empty_ray_pairs_cuda returned None in diagnostic")
            env_ids, src, tgt, _, _ = out
            if src.numel() == 0:
                return np.empty((0, 7), dtype=np.int64)
            rows_t = torch.cat([env_ids.view(-1, 1), src, tgt], dim=1)
        else:
            parts = []
            for env_idx in range(env.num_envs):
                src, tgt = env._empty_ray_pairs_for_env(
                    env_idx,
                    eyes[env_idx],
                    rays_world[env_idx],
                    ~free_hit_mask[env_idx],
                    stats=None,
                )
                if src.numel() == 0:
                    continue
                parts.append(torch.cat([
                    torch.full((src.shape[0], 1), env_idx, dtype=torch.long, device=env.device),
                    src,
                    tgt,
                ], dim=1))
            if not parts:
                return np.empty((0, 7), dtype=np.int64)
            rows_t = torch.cat(parts, dim=0)
        return rows_t.detach().cpu().numpy().astype(np.int64, copy=False)

    def summarize_pair_set_diff(cuda_rows: np.ndarray, ref_rows: np.ndarray) -> dict:
        cuda_set = set(map(tuple, cuda_rows.tolist()))
        ref_set = set(map(tuple, ref_rows.tolist()))
        cuda_extra = sorted(cuda_set - ref_set)
        ref_extra = sorted(ref_set - cuda_set)
        by_env = {}
        for label, rows in (("cuda_extra", cuda_extra), ("ref_extra", ref_extra)):
            counts = {}
            for row in rows:
                counts[str(row[0])] = counts.get(str(row[0]), 0) + 1
            by_env[label] = counts
        return {
            "cuda_rows": int(cuda_rows.shape[0]),
            "ref_rows": int(ref_rows.shape[0]),
            "cuda_unique": int(len(cuda_set)),
            "ref_unique": int(len(ref_set)),
            "cuda_extra": int(len(cuda_extra)),
            "ref_extra": int(len(ref_extra)),
            "by_env": by_env,
            "cuda_extra_examples": [list(x) for x in cuda_extra[:10]],
            "ref_extra_examples": [list(x) for x in ref_extra[:10]],
        }

    previous_disable = os.environ.get("SHAPENBV_DISABLE_CUDA_EMPTY_PAIRS")
    rows = []
    try:
        for step in range(int(max_steps)):
            action = np.repeat(action_cycle[step % len(action_cycle)][None, :], n_envs, axis=0)
            same_cuda_rows, same_ref_rows = compute_empty_pair_rows_same_inputs(fast_env, action)
            same_input_diff = summarize_pair_set_diff(same_cuda_rows, same_ref_rows)
            print(
                "[empty-pair-same-input-step] "
                + json.dumps({
                    "step": step + 1,
                    "cuda_rows": same_input_diff["cuda_rows"],
                    "ref_rows": same_input_diff["ref_rows"],
                    "cuda_extra": same_input_diff["cuda_extra"],
                    "ref_extra": same_input_diff["ref_extra"],
                    "cuda_extra_examples": same_input_diff["cuda_extra_examples"],
                    "ref_extra_examples": same_input_diff["ref_extra_examples"],
                }, sort_keys=True),
                flush=True,
            )
            if same_input_diff["cuda_extra"] != 0 or same_input_diff["ref_extra"] != 0:
                print(
                    "[empty-pair-same-input-details] "
                    + json.dumps(
                        diagnose_same_input_cuda_examples(
                            fast_env,
                            action,
                            same_input_diff["cuda_extra_examples"],
                        ),
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise AssertionError(
                    f"same-input empty pair mismatch at step {step + 1}: {same_input_diff}"
                )

            os.environ.pop("SHAPENBV_DISABLE_CUDA_EMPTY_PAIRS", None)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fast_env.step_async(action)
            obs_fast, rew_fast, done_fast, info_fast = fast_env.step_wait()
            torch.cuda.synchronize()
            fast_wall = time.perf_counter() - t0

            os.environ["SHAPENBV_DISABLE_CUDA_EMPTY_PAIRS"] = "1"
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ref_env.step_async(action)
            obs_ref, rew_ref, done_ref, info_ref = ref_env.step_wait()
            torch.cuda.synchronize()
            ref_wall = time.perf_counter() - t0

            prob_diff = float((fast_env._prob_grid - ref_env._prob_grid).abs().max().detach().cpu().item())
            scanned_diff = float((fast_env._scanned_gt_grid - ref_env._scanned_gt_grid).abs().max().detach().cpu().item())
            reward_diff = float(np.max(np.abs(np.asarray(rew_fast) - np.asarray(rew_ref))))
            done_match = bool(np.array_equal(np.asarray(done_fast, dtype=bool), np.asarray(done_ref, dtype=bool)))
            obs_diff = float(np.max(np.abs(np.asarray(obs_fast) - np.asarray(obs_ref))))
            cr_fast = fast_env._scanned_gt_grid.sum(dim=(1, 2, 3)) / fast_env._num_valid_gt_per_env
            cr_ref = ref_env._scanned_gt_grid.sum(dim=(1, 2, 3)) / ref_env._num_valid_gt_per_env
            cr_diff = float((cr_fast - cr_ref).abs().max().detach().cpu().item())
            seq_match = [
                fa.get("seq_name", "") == rb.get("seq_name", "")
                for fa, rb in zip(info_fast, info_ref)
            ]
            if prob_diff != 0.0 or scanned_diff != 0.0 or reward_diff != 0.0 or obs_diff != 0.0 or cr_diff != 0.0:
                cuda_rows = compute_empty_pair_rows(fast_env, action, use_cuda=True)
                ref_rows = compute_empty_pair_rows(ref_env, action, use_cuda=False)
                print(
                    "[empty-pair-set-diff] "
                    + json.dumps(summarize_pair_set_diff(cuda_rows, ref_rows), sort_keys=True),
                    flush=True,
                )
                raise AssertionError(
                    f"step {step + 1} mismatch: prob={prob_diff} scanned={scanned_diff} "
                    f"reward={reward_diff} obs={obs_diff} cr={cr_diff}"
                )
            if not done_match or not all(seq_match):
                raise AssertionError(
                    f"step {step + 1} done/seq mismatch: done={done_match} seq={seq_match}"
                )

            prof_fast = getattr(fast_env, "last_step_profile", {}) or {}
            prof_ref = getattr(ref_env, "last_step_profile", {}) or {}
            row = {
                "step": step + 1,
                "fast_wall_ms": float(fast_wall * 1000.0),
                "ref_wall_ms": float(ref_wall * 1000.0),
                "fast_pair_ms": float(prof_fast.get("free/empty_pairs_s", 0.0)) * 1000.0,
                "ref_pair_ms": float(prof_ref.get("free/empty_pairs_s", 0.0)) * 1000.0,
                "fast_free_ms": float(prof_fast.get("free_update_s", 0.0)) * 1000.0,
                "ref_free_ms": float(prof_ref.get("free_update_s", 0.0)) * 1000.0,
                "fast_pair_builder": int(float(prof_fast.get("free/cuda_empty_pair_builder", 0.0))),
                "ref_pair_builder": int(float(prof_ref.get("free/cuda_empty_pair_builder", 0.0))),
                "fast_pairs_before": int(float(prof_fast.get("free/empty_ray_pairs_before_unique", 0.0))),
                "fast_pairs_after": int(float(prof_fast.get("free/empty_ray_pairs_after_unique", 0.0))),
                "ref_pairs_before": int(float(prof_ref.get("free/empty_ray_pairs_before_unique", 0.0))),
                "ref_pairs_after": int(float(prof_ref.get("free/empty_ray_pairs_after_unique", 0.0))),
                "prob_diff": prob_diff,
                "scanned_diff": scanned_diff,
                "reward_diff": reward_diff,
                "obs_diff": obs_diff,
                "cr_diff": cr_diff,
            }
            rows.append(row)
            print("[empty-pair-validate-step] " + json.dumps(row, sort_keys=True), flush=True)
    finally:
        if previous_disable is None:
            os.environ.pop("SHAPENBV_DISABLE_CUDA_EMPTY_PAIRS", None)
        else:
            os.environ["SHAPENBV_DISABLE_CUDA_EMPTY_PAIRS"] = previous_disable

    steady = rows[1:] if len(rows) > 1 else rows
    summary = {
        "samples": names,
        "n_envs": n_envs,
        "max_prob_diff": float(max(r["prob_diff"] for r in rows)),
        "max_scanned_diff": float(max(r["scanned_diff"] for r in rows)),
        "max_reward_diff": float(max(r["reward_diff"] for r in rows)),
        "max_obs_diff": float(max(r["obs_diff"] for r in rows)),
        "max_cr_diff": float(max(r["cr_diff"] for r in rows)),
        "steady_fast_wall_ms_mean": float(np.mean([r["fast_wall_ms"] for r in steady])),
        "steady_ref_wall_ms_mean": float(np.mean([r["ref_wall_ms"] for r in steady])),
        "steady_fast_pair_ms_mean": float(np.mean([r["fast_pair_ms"] for r in steady])),
        "steady_ref_pair_ms_mean": float(np.mean([r["ref_pair_ms"] for r in steady])),
        "steady_fast_free_ms_mean": float(np.mean([r["fast_free_ms"] for r in steady])),
        "steady_ref_free_ms_mean": float(np.mean([r["ref_free_ms"] for r in steady])),
        "rows": rows,
    }
    print("[empty-pair-validate-summary] " + json.dumps(summary, sort_keys=True), flush=True)
    fast_env.close()
    ref_env.close()
    return summary


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data},
    timeout=20 * 60,
    retries=0,
)
def validate_infra_optimizations(
    render_size: int = 400,
    fov_deg: float = 60.0,
    mesh_path: str = "/data/ShapeNetCore.v2/03001627/1006be65e7bc937e9141f9b58470d646/models/model_normalized.obj",
    preproc_path: str = "/data/geoscout_preproc_g128/03001627_1006be65e7bc937e9141f9b58470d646.pt",
    max_faces: int = 5000,
):
    """Assert that infra-only optimizations preserve exact math.

    This intentionally avoids PPO/training and checks the two current
    low-level optimizations directly on the same CUDA image used by the
    smoke probes:
      1. cached camera-frame rays equal the previous per-step formula;
      2. duplicate empty-ray entry/exit dedupe leaves the free grid
         identical after the set-union update.
    """
    import json
    import math
    import time
    from pathlib import Path

    import torch

    from geoscout.mesh_renderer import MeshSequenceRenderer
    from geoscout.preprocess import load_preproc
    from geoscout.tensor_env import TensorBatchEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fake = object.__new__(TensorBatchEnv)
    fake.device = device
    fake.render_size = int(render_size)
    fake.fov_deg = float(fov_deg)

    cached = TensorBatchEnv._build_camera_rays(fake)
    H = W = int(render_size)
    fov_rad = math.radians(float(fov_deg))
    f = 0.5 * H / math.tan(0.5 * fov_rad)
    cx, cy = 0.5 * W, 0.5 * H
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    old = torch.stack([
        (xs - cx) / f,
        (ys - cy) / f,
        torch.ones_like(xs),
    ], dim=-1)
    old = old / old.norm(dim=-1, keepdim=True)
    old = old.reshape(-1, 3)
    max_ray_diff = float((cached - old).abs().max().detach().cpu().item())
    if max_ray_diff != 0.0:
        raise AssertionError(f"cached camera rays changed: max diff {max_ray_diff}")

    fake._rays_cam = cached
    eyes = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    ats = torch.zeros_like(eyes)
    world = TensorBatchEnv._world_rays(fake, eyes, ats)
    if tuple(world.shape) != (2, H * W, 3):
        raise AssertionError(f"unexpected world ray shape {tuple(world.shape)}")
    max_norm_err = float((world.norm(dim=-1) - 1.0).abs().max().detach().cpu().item())
    if max_norm_err > 2e-7:
        raise AssertionError(f"world rays are not unit vectors: {max_norm_err}")

    def make_free_env(
        dedupe: bool,
        triton_free: bool = False,
        free_mask_apply_mode: str = "index",
        free_raycast_backend: str = "triton",
    ):
        env = object.__new__(TensorBatchEnv)
        env.n_free_samples_per_ray = 1
        env.update_empty_rays = True
        env.dedupe_empty_ray_pairs = bool(dedupe)
        env.use_triton_free_raycast = bool(triton_free)
        env.free_raycast_backend = str(free_raycast_backend)
        env.free_mask_apply_mode = str(free_mask_apply_mode)
        env.triton_bresenham_block_rays = 64
        env.profile_timing = True
        env.num_envs = 1
        env.grid_size = 16
        env._bbox_min = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        env._voxel_size = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
        env._prob_grid = torch.zeros(1, 16, 16, 16, dtype=torch.float32, device=device)
        return env

    eyes1 = torch.tensor([[0.5, 0.5, -1.0]], dtype=torch.float32, device=device)
    target_idx = torch.zeros(1, 4, 3, dtype=torch.long, device=device)
    valid_mask = torch.zeros(1, 4, dtype=torch.bool, device=device)
    rays_world = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.05, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    rays_world = rays_world / rays_world.norm(dim=-1, keepdim=True)
    hit_pixel_mask = torch.zeros(1, 4, dtype=torch.bool, device=device)

    env_full = make_free_env(dedupe=False, triton_free=False)
    TensorBatchEnv._update_free_voxels(
        env_full,
        eyes1,
        target_idx,
        valid_mask,
        rays_world=rays_world,
        hit_pixel_mask=hit_pixel_mask,
    )
    grid_full = env_full._prob_grid.clone()
    stats_full = dict(env_full._last_free_update_stats)

    env_dedup = make_free_env(dedupe=True, triton_free=False)
    TensorBatchEnv._update_free_voxels(
        env_dedup,
        eyes1,
        target_idx,
        valid_mask,
        rays_world=rays_world,
        hit_pixel_mask=hit_pixel_mask,
    )
    grid_dedup = env_dedup._prob_grid.clone()
    stats_dedup = dict(env_dedup._last_free_update_stats)

    grid_diff = float((grid_full - grid_dedup).abs().max().detach().cpu().item())
    if grid_diff != 0.0:
        raise AssertionError(f"dedupe changed free grid: max diff {grid_diff}")
    before = int(stats_dedup.get("empty_ray_pairs_before_unique", -1))
    after = int(stats_dedup.get("empty_ray_pairs_after_unique", -1))
    if not (0 < after < before):
        raise AssertionError(f"dedupe did not reduce duplicate pairs: before={before} after={after}")

    env_triton = make_free_env(dedupe=True, triton_free=True, free_mask_apply_mode="index")
    TensorBatchEnv._update_free_voxels(
        env_triton,
        eyes1,
        target_idx,
        valid_mask,
        rays_world=rays_world,
        hit_pixel_mask=hit_pixel_mask,
    )
    triton_grid_diff = float(
        (env_dedup._prob_grid - env_triton._prob_grid).abs().max().detach().cpu().item()
    )
    if triton_grid_diff != 0.0:
        raise AssertionError(f"Triton free raycast changed free grid: max diff {triton_grid_diff}")
    stats_triton = dict(env_triton._last_free_update_stats)

    env_cuda = make_free_env(
        dedupe=True,
        triton_free=True,
        free_mask_apply_mode="index",
        free_raycast_backend="cuda",
    )
    TensorBatchEnv._update_free_voxels(
        env_cuda,
        eyes1,
        target_idx,
        valid_mask,
        rays_world=rays_world,
        hit_pixel_mask=hit_pixel_mask,
    )
    cuda_grid_diff = float(
        (env_dedup._prob_grid - env_cuda._prob_grid).abs().max().detach().cpu().item()
    )
    if cuda_grid_diff != 0.0:
        raise AssertionError(f"CUDA free raycast changed free grid: max diff {cuda_grid_diff}")
    stats_cuda = dict(env_cuda._last_free_update_stats)
    if int(stats_cuda.get("cuda_free_raycast", 0)) != 1:
        raise AssertionError(
            "custom CUDA Bresenham scatter was requested but not used; "
            f"stats={stats_cuda}"
        )

    apply_mode_diffs = {}
    apply_mode_stats = {}
    for mode in ("dense", "triton"):
        env_apply = make_free_env(dedupe=True, triton_free=True, free_mask_apply_mode=mode)
        TensorBatchEnv._update_free_voxels(
            env_apply,
            eyes1,
            target_idx,
            valid_mask,
            rays_world=rays_world,
            hit_pixel_mask=hit_pixel_mask,
        )
        diff = float(
            (env_dedup._prob_grid - env_apply._prob_grid).abs().max().detach().cpu().item()
        )
        if diff != 0.0:
            raise AssertionError(f"free mask apply mode {mode!r} changed grid: max diff {diff}")
        apply_mode_diffs[mode] = diff
        apply_mode_stats[mode] = dict(env_apply._last_free_update_stats)

    def compare_hit_update(
        eye_xyz,
        target_xyz,
        *,
        free_raycast_backend: str,
        expect_triton_hit: int,
        expect_cuda_hit: int = 0,
    ):
        eye_hit = torch.tensor([eye_xyz], dtype=torch.float32, device=device)
        target_hit = torch.tensor([[target_xyz]], dtype=torch.long, device=device)
        valid_hit = torch.tensor([[True]], dtype=torch.bool, device=device)
        rays_hit = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
        hit_mask = torch.tensor([[True]], dtype=torch.bool, device=device)

        env_hit_py = make_free_env(dedupe=True, triton_free=False)
        TensorBatchEnv._update_free_voxels(
            env_hit_py,
            eye_hit,
            target_hit,
            valid_hit,
            rays_world=rays_hit,
            hit_pixel_mask=hit_mask,
        )
        env_hit_fast = make_free_env(
            dedupe=True,
            triton_free=True,
            free_raycast_backend=free_raycast_backend,
        )
        TensorBatchEnv._update_free_voxels(
            env_hit_fast,
            eye_hit,
            target_hit,
            valid_hit,
            rays_world=rays_hit,
            hit_pixel_mask=hit_mask,
        )
        diff = float(
            (env_hit_py._prob_grid - env_hit_fast._prob_grid).abs().max().detach().cpu().item()
        )
        if diff != 0.0:
            raise AssertionError(
                f"{free_raycast_backend} hit raycast changed free grid: max diff {diff}"
            )
        stats_fast = dict(env_hit_fast._last_free_update_stats)
        used_triton = int(stats_fast.get("triton_hit_raycast", 0))
        used_cuda = int(stats_fast.get("cuda_hit_raycast", 0))
        if used_triton != int(expect_triton_hit):
            raise AssertionError(
                f"unexpected triton_hit_raycast={used_triton}, expected {expect_triton_hit}"
            )
        if used_cuda != int(expect_cuda_hit):
            raise AssertionError(
                f"unexpected cuda_hit_raycast={used_cuda}, expected {expect_cuda_hit}"
            )
        return diff, stats_fast

    triton_hit_grid_diff, stats_hit_triton = compare_hit_update(
        [0.5, 0.5, -1.0], [0, 0, 7],
        free_raycast_backend="triton",
        expect_triton_hit=1,
    )
    cuda_hit_grid_diff, stats_hit_cuda = compare_hit_update(
        [0.5, 0.5, -1.0], [0, 0, 7],
        free_raycast_backend="cuda",
        expect_triton_hit=0,
        expect_cuda_hit=1,
    )
    fallback_hit_grid_diff, stats_hit_fallback = compare_hit_update(
        [-100.0, 0.5, 0.5], [15, 0, 0],
        free_raycast_backend="triton",
        expect_triton_hit=0,
    )

    preproc = load_preproc(Path(preproc_path), map_location=device)
    T_canon = preproc.get("T_canon")
    renderer_full = MeshSequenceRenderer(
        mesh_path=Path(mesh_path),
        sequence_name="infra_validate_full",
        device=device,
        render_size=(int(render_size), int(render_size)),
        fov_deg=float(fov_deg),
        T_canon=T_canon,
        max_faces=int(max_faces),
        bbox_ray_cull=False,
    )
    renderer_cull = MeshSequenceRenderer(
        mesh_path=Path(mesh_path),
        sequence_name="infra_validate_cull",
        device=device,
        render_size=(int(render_size), int(render_size)),
        fov_deg=float(fov_deg),
        T_canon=T_canon,
        max_faces=int(max_faces),
        bbox_ray_cull=True,
    )
    positions = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    look_ats = torch.zeros_like(positions)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out_full = renderer_full.render_batch(positions, look_ats)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    full_render_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out_cull = renderer_cull.render_batch(positions, look_ats)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cull_render_s = time.perf_counter() - t0
    alpha_diff = float((out_full.alpha - out_cull.alpha).abs().max().detach().cpu().item())
    depth_diff = float((out_full.depth - out_cull.depth).abs().max().detach().cpu().item())
    if alpha_diff != 0.0 or depth_diff != 0.0:
        raise AssertionError(
            f"bbox ray cull changed render output: alpha_diff={alpha_diff} depth_diff={depth_diff}"
        )
    render_stats = dict(renderer_cull.last_render_stats)

    result = {
        "device": str(device),
        "render_size": int(render_size),
        "fov_deg": float(fov_deg),
        "mesh_path": str(mesh_path),
        "max_cached_ray_diff": max_ray_diff,
        "max_world_ray_norm_error": max_norm_err,
        "max_free_grid_diff": grid_diff,
        "dedupe_pairs_before": before,
        "dedupe_pairs_after": after,
        "full_free_voxels": int(stats_full.get("union_free_voxels", -1)),
        "dedup_free_voxels": int(stats_dedup.get("union_free_voxels", -1)),
        "triton_free_grid_diff": triton_grid_diff,
        "triton_free_voxels": int(stats_triton.get("union_free_voxels", -1)),
        "triton_free_used": int(stats_triton.get("triton_free_raycast", 0)),
        "cuda_free_grid_diff": cuda_grid_diff,
        "cuda_free_voxels": int(stats_cuda.get("union_free_voxels", -1)),
        "cuda_free_used": int(stats_cuda.get("cuda_free_raycast", 0)),
        "dense_apply_grid_diff": float(apply_mode_diffs.get("dense", -1.0)),
        "triton_apply_grid_diff": float(apply_mode_diffs.get("triton", -1.0)),
        "triton_apply_used": int(apply_mode_stats.get("triton", {}).get("free_mask_apply_triton", 0)),
        "triton_hit_grid_diff": triton_hit_grid_diff,
        "triton_hit_used": int(stats_hit_triton.get("triton_hit_raycast", 0)),
        "triton_hit_max_delta": int(stats_hit_triton.get("hit_max_delta", -1)),
        "cuda_hit_grid_diff": cuda_hit_grid_diff,
        "cuda_hit_used": int(stats_hit_cuda.get("cuda_hit_raycast", 0)),
        "cuda_hit_max_delta": int(stats_hit_cuda.get("hit_max_delta", -1)),
        "fallback_hit_grid_diff": fallback_hit_grid_diff,
        "fallback_hit_used": int(stats_hit_fallback.get("triton_hit_raycast", -1)),
        "fallback_hit_max_delta": int(stats_hit_fallback.get("hit_max_delta", -1)),
        "render_alpha_diff": alpha_diff,
        "render_depth_diff": depth_diff,
        "render_full_s": full_render_s,
        "render_cull_s": cull_render_s,
        "render_cull_active_rays": int(render_stats.get("active_rays", -1)),
        "render_cull_total_rays": int(render_stats.get("total_rays", -1)),
        "render_cull_active_ratio": float(render_stats.get("active_ratio", -1.0)),
    }
    print("[infra-validate] " + json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data},
    timeout=2 * 3600,
    retries=0,
)
def benchmark_renderer_backends(
    seq_name: str = "03001627_1006be65e7bc937e9141f9b58470d646",
    synsets: str = "03001627",
    limit_per_synset: int = 1,
    preproc_dir: str = "/data/geoscout_preproc_g128",
    image_size: int = 400,
    max_faces: int = 5000,
    n_cameras: int = 8,
    warmup: int = 1,
    reps: int = 3,
):
    """Fair renderer microbenchmark on identical cameras/mesh.

    Contract:
      - same ShapeNet mesh and preproc T_canon
      - same current camera convention / pixel-center rays
      - same 400x400 render size by default
      - report both speed and output diff against the current renderer
    """
    import json
    import time
    from pathlib import Path

    import numpy as np
    import torch

    from geoscout.data import list_shapenet
    from geoscout.mesh_renderer import MeshSequenceRenderer, _build_camera_basis
    from geoscout.preprocess import load_preproc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synset_list = [s.strip() for s in synsets.split(",") if s.strip()] or None
    entries = list_shapenet(
        Path("/data/ShapeNetCore.v2"),
        synsets=synset_list,
        limit_per_synset=limit_per_synset,
    )
    if seq_name:
        entries = [e for e in entries if e.name == seq_name] or [
            e for e in list_shapenet(Path("/data/ShapeNetCore.v2"))
            if e.name == seq_name
        ]
    if not entries:
        raise RuntimeError(f"No ShapeNet entry found for seq_name={seq_name!r}")
    entry = entries[0]
    pp = Path(preproc_dir) / f"{entry.name}.pt"
    preproc = load_preproc(pp, map_location=device)

    renderer = MeshSequenceRenderer(
        mesh_path=entry.mesh_path,
        sequence_name=entry.name,
        device=device,
        render_size=(int(image_size), int(image_size)),
        fov_deg=60.0,
        T_canon=preproc.get("T_canon"),
        max_faces=int(max_faces),
        bbox_ray_cull=True,
    )
    action_cycle = np.array([
        [40, 40, 80, 0, 0, 0],
        [80, 40, 40, 0, 0, 0],
        [0, 40, 40, 0, 0, 0],
        [40, 80, 40, 0, 0, 0],
        [40, 0, 40, 0, 0, 0],
        [40, 40, 0, 0, 0, 0],
        [80, 80, 40, 0, 0, 0],
        [0, 0, 40, 0, 0, 0],
    ], dtype=np.int64)
    action_low = np.array([-1.0, -1.0, -1.0, 0.0, -np.pi / 2.0, 0.0], dtype=np.float32)
    action_unit = np.array([0.025, 0.025, 0.025, 0.0, np.pi / 12.0, np.pi / 6.0], dtype=np.float32)
    actions = np.vstack([action_cycle[i % len(action_cycle)] for i in range(int(n_cameras))])
    pose6 = actions.astype(np.float32) * action_unit + action_low
    positions = torch.tensor(pose6[:, :3], dtype=torch.float32, device=device)
    look_ats = torch.zeros_like(positions)

    def time_current():
        for _ in range(int(warmup)):
            out = renderer.render_batch(positions, look_ats)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        times = []
        out = None
        for _ in range(int(reps)):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = renderer.render_batch(positions, look_ats)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
        return out, times

    ref_out, current_times = time_current()
    ref_alpha = (ref_out.alpha.detach().cpu().numpy() > 0.5)
    ref_depth = ref_out.depth.detach().cpu().numpy()

    results = {
        "sample": entry.name,
        "image_size": int(image_size),
        "n_cameras": int(n_cameras),
        "max_faces": int(max_faces),
        "faces_after_decimation": int(renderer._faces.shape[0]),
        "current_bbox_cuda_s_mean": float(np.mean(current_times)),
        "current_bbox_cuda_s_min": float(np.min(current_times)),
        "current_render_active_ratio": float(renderer.last_render_stats.get("active_ratio", -1.0)),
    }

    # Open3D CPU BVH raycasting. It is a candidate only if both speed and
    # output parity are acceptable, so we benchmark it without touching the
    # training path.
    try:
        import open3d as o3d

        results["open3d_version"] = str(getattr(o3d, "__version__", "unknown"))
        try:
            results["open3d_cuda_available"] = bool(o3d.core.cuda.is_available())
        except Exception as exc:
            results["open3d_cuda_available_error"] = f"{type(exc).__name__}: {exc}"
        try:
            results["open3d_sycl_available"] = bool(o3d.core.sycl.is_available())
        except Exception as exc:
            results["open3d_sycl_available_error"] = f"{type(exc).__name__}: {exc}"

        verts_np = renderer._verts.detach().cpu().numpy().astype(np.float32)
        faces_np = renderer._faces.detach().cpu().numpy().astype(np.uint32)
        scene_t0 = time.perf_counter()
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(
            o3d.core.Tensor(verts_np, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(faces_np, dtype=o3d.core.Dtype.UInt32),
        )
        scene_build_s = time.perf_counter() - scene_t0

        H = W = int(image_size)
        basis = torch.stack([
            _build_camera_basis(positions[k], look_ats[k])
            for k in range(positions.shape[0])
        ], dim=0)
        rays_world = torch.einsum("kij,hj->khi", basis, renderer._rays_cam)
        origins = positions[:, None, :].expand(positions.shape[0], H * W, 3)
        rays_o3d = torch.cat([origins, rays_world], dim=-1).reshape(-1, 6).detach().cpu().numpy()
        rays_o3d_t = o3d.core.Tensor(rays_o3d, dtype=o3d.core.Dtype.Float32)

        for _ in range(int(warmup)):
            _ = scene.cast_rays(rays_o3d_t)
        o3d_times = []
        ans = None
        for _ in range(int(reps)):
            t0 = time.perf_counter()
            ans = scene.cast_rays(rays_o3d_t)
            o3d_times.append(time.perf_counter() - t0)
        t_hit = ans["t_hit"].numpy().reshape(int(n_cameras), H, W)
        o3d_alpha = np.isfinite(t_hit)
        o3d_depth = np.where(o3d_alpha, t_hit, 0.0).astype(np.float32)

        inter = np.logical_and(ref_alpha, o3d_alpha).sum()
        union = np.logical_or(ref_alpha, o3d_alpha).sum()
        mismatch = np.not_equal(ref_alpha, o3d_alpha).sum()
        common = np.logical_and(ref_alpha, o3d_alpha)
        if common.any():
            depth_abs = np.abs(ref_depth[common] - o3d_depth[common])
            depth_mean = float(depth_abs.mean())
            depth_max = float(depth_abs.max())
        else:
            depth_mean = float("nan")
            depth_max = float("nan")

        results.update({
            "open3d_scene_build_s": float(scene_build_s),
            "open3d_cast_s_mean": float(np.mean(o3d_times)),
            "open3d_cast_s_min": float(np.min(o3d_times)),
            "open3d_alpha_iou": float(inter / max(union, 1)),
            "open3d_alpha_mismatch_pixels": int(mismatch),
            "open3d_depth_abs_mean_common": depth_mean,
            "open3d_depth_abs_max_common": depth_max,
            "open3d_ref_hit_pixels": int(ref_alpha.sum()),
            "open3d_hit_pixels": int(o3d_alpha.sum()),
        })
    except Exception as exc:
        results["open3d_error"] = f"{type(exc).__name__}: {exc}"

    print("[renderer-benchmark] " + json.dumps(results, sort_keys=True), flush=True)
    return results


@app.function(
    image=image,
    gpu="L4",
    timeout=2 * 3600,
    memory=16 * 1024,
    retries=0,
)
def probe_nvdiffrast_install():
    """Check whether NVlabs nvdiffrast can install/run in the Modal image."""
    import json
    import os
    import subprocess
    import sys
    import time

    import torch

    result = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    t0 = time.perf_counter()
    bootstrap_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "setuptools",
        "wheel",
        "ninja",
    ]
    print("[nvdiffrast-probe] running", " ".join(bootstrap_cmd), flush=True)
    subprocess.check_call(bootstrap_cmd)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "git+https://github.com/NVlabs/nvdiffrast.git",
        "--no-build-isolation",
    ]
    print("[nvdiffrast-probe] running", " ".join(cmd), flush=True)
    build_env = dict(os.environ)
    build_env.setdefault("CC", "gcc")
    build_env.setdefault("CXX", "g++")
    subprocess.check_call(cmd, env=build_env)
    result["install_s"] = time.perf_counter() - t0

    import nvdiffrast.torch as dr

    device = torch.device("cuda")
    ctx = dr.RasterizeCudaContext(device=device)
    pos = torch.tensor(
        [[[-0.8, -0.8, 0.0, 1.0],
          [0.8, -0.8, 0.0, 1.0],
          [0.0, 0.8, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=(64, 64), grad_db=False)
    torch.cuda.synchronize()
    result["hit_pixels"] = int((rast[..., 3] > 0).sum().detach().cpu().item())
    print("[nvdiffrast-probe] " + json.dumps(result, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data},
    timeout=2 * 3600,
    memory=32 * 1024,
    retries=0,
)
def validate_voxel_renderer_backend(
    sample_names: str = (
        "03001627_1006be65e7bc937e9141f9b58470d646,"
        "03001627_1007e20d5e811b308351982a6e40cf41,"
        "03001627_100b18376b885f206ae9ad7e32c4139d,"
        "03001627_1013f70851210a618f2e765c4a8ed3d,"
        "04256520_1037fd31d12178d396f164a988ef37cc,"
        "04256520_103b76b2594a1582eaf14273fa406ffc,"
        "04256520_104256e5bb73b0b719fb4103277a6b93,"
        "04256520_1050790962944624febad4f49b26ec52,"
        "04379243_1011e1c9812b84d2a9ed7bb5b55809f8,"
        "04379243_10139657dfa9afe0c3bd24f986301745,"
        "04379243_1028a9cbaa7a333230bbd4cddd04c77b,"
        "04379243_102f0532f9f8bbcdcb503f63ed915ed2"
    ),
    preproc_dir: str = "/data/geoscout_preproc_g128",
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    max_steps: int = 6,
    max_faces: int = 5000,
    candidate_backend: str = "voxel_cuda",
    seed: int = 0,
):
    """A/B test Open3D mesh render against a candidate GPU renderer."""
    import json
    import time
    from pathlib import Path

    import numpy as np
    import torch

    from geoscout.tensor_env import TensorBatchEnv

    names = [s.strip() for s in str(sample_names).split(",") if s.strip()]
    if not names:
        raise ValueError("sample_names must contain at least one sample")
    shapenet_root = Path("/data/ShapeNetCore.v2")
    preproc_root = Path(preproc_dir)
    mesh_paths = []
    preproc_paths = []
    for name in names:
        synset, model_id = name.split("_", 1)
        mesh = shapenet_root / synset / model_id / "models" / "model_normalized.obj"
        pp = preproc_root / f"{name}.pt"
        if not mesh.exists():
            raise FileNotFoundError(f"missing mesh for {name}: {mesh}")
        if not pp.exists():
            raise FileNotFoundError(f"missing preproc for {name}: {pp}")
        mesh_paths.append(mesh)
        preproc_paths.append(pp)

    n_envs = len(names)
    print(
        f"[voxel-render-validate] samples={n_envs} image={image_size} "
        f"grid={grid_size} obs={obs_grid_size} steps={max_steps} "
        f"candidate={candidate_backend}",
        flush=True,
    )

    def make_env(renderer_backend: str) -> TensorBatchEnv:
        return TensorBatchEnv(
            num_envs=n_envs,
            mesh_paths=mesh_paths,
            preproc_paths=preproc_paths,
            device="cuda",
            buffer_size=30,
            grid_size=grid_size,
            obs_grid_size=obs_grid_size,
            episode_len=max_steps + 10,
            render_size=image_size,
            fov_deg=60.0,
            cr_success_threshold=2.0,
            coverage_reward_scale=20.0,
            short_path_grace=30,
            short_path_clip=2.0,
            short_path_scale=0.1,
            only_positive_rewards=True,
            skip_free_raycast=False,
            update_empty_rays=True,
            coverage_hit_dilate_radius=1,
            caption_dim=384,
            auto_lookat_center=True,
            max_faces=max_faces,
            renderer_backend=renderer_backend,
            renderer_bbox_ray_cull=True,
            use_triton_free_raycast=True,
            free_raycast_backend="cuda",
            free_mask_apply_mode="triton",
            coverage_reward_type="linear",
            termination_bonus=1.0,
            collision_penalty=10.0,
            dedupe_empty_ray_pairs=False,
            profile_timing=True,
            seed=seed,
        )

    def pin_env(env: TensorBatchEnv) -> np.ndarray:
        device = env.device
        ids = torch.arange(n_envs, dtype=torch.long, device=device)
        env._prob_grid[ids] = 0.0
        env._scanned_gt_grid[ids] = 0.0
        env._step_idx[ids] = 0
        env._cr_prev[ids] = 0.0
        env._action_history[ids] = 0.0
        env._ep_step_count[ids] = 0
        env._ep_reward_sum[ids] = 0.0
        env._ep_new_gt_sum[ids] = 0.0
        env._ep_visible_gt_sum[ids] = 0.0
        env._ep_redundant_gt_sum[ids] = 0.0
        env._ep_revisit_sum[ids] = 0.0
        env._env_mesh_id[ids] = ids
        selected_grid = env._pool_grid_gt[ids] if env._pool_grid_on_device else env._pool_grid_gt[
            ids.detach().cpu()
        ].to(device, dtype=torch.float32, non_blocking=True)
        env._grid_gt[ids] = selected_grid
        env._bbox_min[ids] = env._pool_bbox_min[ids]
        env._voxel_size[ids] = env._pool_voxel_size[ids]
        env._num_valid_gt_per_env[ids] = env._pool_num_valid[ids].clamp(min=1.0)
        if env._caption_emb is not None:
            env._caption_emb[ids] = env._pool_caption_emb[ids]
        ix0 = env._idx_up[0] // 2
        iy0 = env._idx_up[1] // 2
        iz0 = env._idx_up[2]
        init_idx = torch.stack([
            ix0.expand(n_envs),
            iy0.expand(n_envs),
            iz0.expand(n_envs),
            torch.zeros(n_envs, dtype=torch.long, device=device),
            torch.zeros(n_envs, dtype=torch.long, device=device),
            torch.zeros(n_envs, dtype=torch.long, device=device),
        ], dim=-1)
        init_pose = init_idx.float() * env._action_unit + env._action_low
        env._action_history[ids, -1, :] = init_pose
        env._last_action_idx[ids] = init_idx
        return env._build_observation_np()

    open3d_env = make_env("open3d")
    voxel_env = make_env(candidate_backend)
    obs_open = pin_env(open3d_env)
    obs_voxel = pin_env(voxel_env)
    reset_obs_diff = float(np.max(np.abs(obs_open - obs_voxel)))
    if reset_obs_diff != 0.0:
        raise AssertionError(f"reset obs mismatch before rendering: {reset_obs_diff}")

    action_cycle = np.array([
        [40, 40, 80, 0, 0, 0],
        [80, 40, 40, 0, 0, 0],
        [0, 40, 40, 0, 0, 0],
        [40, 80, 40, 0, 0, 0],
        [40, 0, 40, 0, 0, 0],
        [80, 80, 40, 0, 0, 0],
    ], dtype=np.int64)

    rows = []
    for step in range(int(max_steps)):
        action = np.repeat(action_cycle[step % len(action_cycle)][None, :], n_envs, axis=0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        open3d_env.step_async(action)
        obs_open, rew_open, done_open, info_open = open3d_env.step_wait()
        torch.cuda.synchronize()
        open_wall = time.perf_counter() - t0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        voxel_env.step_async(action)
        obs_voxel, rew_voxel, done_voxel, info_voxel = voxel_env.step_wait()
        torch.cuda.synchronize()
        voxel_wall = time.perf_counter() - t0

        cr_open = (
            open3d_env._scanned_gt_grid.sum(dim=(1, 2, 3)) / open3d_env._num_valid_gt_per_env
        )
        cr_voxel = (
            voxel_env._scanned_gt_grid.sum(dim=(1, 2, 3)) / voxel_env._num_valid_gt_per_env
        )
        prof_open = getattr(open3d_env, "last_step_profile", {}) or {}
        prof_voxel = getattr(voxel_env, "last_step_profile", {}) or {}
        row = {
            "step": step + 1,
            "action": action_cycle[step % len(action_cycle)].tolist(),
            "open3d_wall_ms": float(open_wall * 1000.0),
            "voxel_wall_ms": float(voxel_wall * 1000.0),
            "open3d_env_fps": float(n_envs / max(open_wall, 1e-9)),
            "voxel_env_fps": float(n_envs / max(voxel_wall, 1e-9)),
            "open3d_render_ms": float(prof_open.get("render_s", 0.0)) * 1000.0,
            "voxel_render_ms": float(prof_voxel.get("render_s", 0.0)) * 1000.0,
            "open3d_hit_rays": float(prof_open.get("render/hit_rays", 0.0)),
            "voxel_hit_rays": float(prof_voxel.get("render/hit_rays", 0.0)),
            "open3d_hit_ratio": float(prof_open.get("render/hit_ratio", 0.0)),
            "voxel_hit_ratio": float(prof_voxel.get("render/hit_ratio", 0.0)),
            "voxel_direct_points": float(prof_voxel.get("render/direct_points", 0.0)),
            "open3d_world_rays_ms": float(prof_open.get("world_rays_s", 0.0)) * 1000.0,
            "voxel_world_rays_ms": float(prof_voxel.get("world_rays_s", 0.0)) * 1000.0,
            "open3d_free_ms": float(prof_open.get("free_update_s", 0.0)) * 1000.0,
            "voxel_free_ms": float(prof_voxel.get("free_update_s", 0.0)) * 1000.0,
            "open3d_mean_cr": float(cr_open.mean().detach().cpu().item()),
            "voxel_mean_cr": float(cr_voxel.mean().detach().cpu().item()),
            "mean_cr_abs_diff": float((cr_open - cr_voxel).abs().mean().detach().cpu().item()),
            "max_cr_abs_diff": float((cr_open - cr_voxel).abs().max().detach().cpu().item()),
            "reward_abs_mean": float(np.mean(np.abs(np.asarray(rew_open) - np.asarray(rew_voxel)))),
            "obs_abs_max": float(np.max(np.abs(np.asarray(obs_open) - np.asarray(obs_voxel)))),
            "open3d_done_count": int(np.asarray(done_open, dtype=bool).sum()),
            "voxel_done_count": int(np.asarray(done_voxel, dtype=bool).sum()),
            "open3d_visible_mean": float(np.mean([x.get("visible_gt_voxels", 0.0) for x in info_open])),
            "voxel_visible_mean": float(np.mean([x.get("visible_gt_voxels", 0.0) for x in info_voxel])),
            "open3d_new_mean": float(np.mean([x.get("new_gt_voxels", 0.0) for x in info_open])),
            "voxel_new_mean": float(np.mean([x.get("new_gt_voxels", 0.0) for x in info_voxel])),
        }
        rows.append(row)
        print("[voxel-render-row] " + json.dumps(row, sort_keys=True), flush=True)

    steady = rows[1:] if len(rows) > 1 else rows
    def mean_key(key: str) -> float:
        return float(np.mean([r[key] for r in steady])) if steady else float("nan")

    summary = {
        "samples": names,
        "n_envs": n_envs,
        "image_size": int(image_size),
        "grid_size": int(grid_size),
        "max_steps": int(max_steps),
        "candidate_backend": str(candidate_backend),
        "reset_obs_diff": reset_obs_diff,
        "steady_open3d_render_ms": mean_key("open3d_render_ms"),
        "steady_voxel_render_ms": mean_key("voxel_render_ms"),
        "steady_open3d_wall_ms": mean_key("open3d_wall_ms"),
        "steady_voxel_wall_ms": mean_key("voxel_wall_ms"),
        "steady_open3d_env_fps": mean_key("open3d_env_fps"),
        "steady_voxel_env_fps": mean_key("voxel_env_fps"),
        "steady_open3d_hit_ratio": mean_key("open3d_hit_ratio"),
        "steady_voxel_hit_ratio": mean_key("voxel_hit_ratio"),
        "steady_voxel_direct_points": mean_key("voxel_direct_points"),
        "render_speedup": mean_key("open3d_render_ms") / max(mean_key("voxel_render_ms"), 1e-9),
        "wall_speedup": mean_key("open3d_wall_ms") / max(mean_key("voxel_wall_ms"), 1e-9),
        "final_open3d_mean_cr": rows[-1]["open3d_mean_cr"],
        "final_voxel_mean_cr": rows[-1]["voxel_mean_cr"],
        "final_mean_cr_abs_diff": rows[-1]["mean_cr_abs_diff"],
        "final_max_cr_abs_diff": rows[-1]["max_cr_abs_diff"],
        "steady_reward_abs_mean": mean_key("reward_abs_mean"),
        "steady_obs_abs_max": mean_key("obs_abs_max"),
        "rows": rows,
    }
    print("[voxel-render-summary] " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["steady_voxel_render_ms"] <= 0.0:
        raise AssertionError("voxel renderer did not record positive render time")
    if summary["final_voxel_mean_cr"] <= 0.0:
        raise AssertionError("voxel renderer produced zero coverage on all samples")
    return summary


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    secrets=[wandb_secret],
    timeout=24 * 3600,
    memory=32 * 1024,
    retries=0,
)
def train_shapenet_hard(
    total_timesteps: int = 4_000_000,
    n_envs: int = 32,
    episode_len: int = 20,
    synsets: str = "03001627,04256520,04379243",
    limit_per_synset: int = 200,
    preproc_dir: str = "/data/geoscout_preproc_g32",
    out_subdir: str = "shapenet-train-hard-g32-cube-h20-ig-v2",
    caption_dim: int = 384,
    wandb_project: str = "geoscout",
    wandb_run_name: str = "shapenet-hard-g32-cube-h20-ig-v2",
    wandb_entity: str = "xiaoleichu-university-of-california-berkeley",
    wandb_mode: str = "online",
    max_faces: int = 5000,
    renderer_backend: str = "nvdiffrast",
    free_raycast_backend: str = "auto",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
    image_size: int = 400,
    grid_size: int = 32,
    obs_grid_size: int = 0,
    coverage_hit_dilate_radius: int = 1,
    coverage_reward_type: str = "information_gain",
    coverage_threshold: float = 0.75,
    termination_bonus: float = 1.0,
    novelty_reward_scale: float = 25.0,
    remaining_reward_scale: float = 250.0,
    redundancy_penalty_scale: float = 10.0,
    view_revisit_penalty_scale: float = 5.0,
    view_revisit_angle_deg: float = 12.0,
    collision_penalty: float = 10.0,
    seed: int = 0,
    post_eval_policies: str = "ppo,random,repeat_top,axis6,ring",
    post_eval_episodes: int = 128,
):
    """Harder ShapeNet training path: 32³ coverage + Cube Mode actions.

    Requires `/data/geoscout_preproc_g32` first. Run `preprocess` with
    `--out-dir /data/geoscout_preproc_g32 --grid-size 32` for the same
    synsets/limit before launching this. The default horizon is 20
    because the 50-step hard baseline still lets simple open-loop policies
    reach high CR; at 20 steps they leave more room for adaptation.
    """
    import os, subprocess, sys

    log_dir = f"/runs/{out_subdir}"
    env_pp = {**os.environ, "PYTHONPATH": "/workspace/GeoScout"}
    cmd = [
        sys.executable, "-u", "-m", "scripts.train",
        "--dataset", "shapenet",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--log_dir", log_dir,
        "--total_timesteps", str(total_timesteps),
        "--n_envs", str(n_envs),
        "--seed", str(seed),
        "--device", "cuda",
        "--image_size", str(image_size),
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--coverage_hit_dilate_radius", str(coverage_hit_dilate_radius),
        "--coverage_reward_scale", "20",
        "--short_path_grace_steps", "30",
        "--short_path_max_extra", "2",
        "--short_path_scale", "0.1",
        "--termination_bonus", str(termination_bonus),
        "--coverage_threshold", str(coverage_threshold),
        "--coverage_reward_type", coverage_reward_type,
        "--novelty_reward_scale", str(novelty_reward_scale),
        "--remaining_reward_scale", str(remaining_reward_scale),
        "--redundancy_penalty_scale", str(redundancy_penalty_scale),
        "--view_revisit_penalty_scale", str(view_revisit_penalty_scale),
        "--view_revisit_angle_deg", str(view_revisit_angle_deg),
        "--collision_penalty", str(collision_penalty),
        "--n_steps", "128", "--batch_size", "128", "--n_epochs", "5",
        "--learning_rate", "1e-4", "--clip_range", "0.2",
        "--gamma", "0.99", "--ent_coef", "0.0", "--target_kl", "0.05",
        "--tensor_env", "--tensor_env_n_envs", str(n_envs),
        "--auto_lookat_center",
        "--caption_dim", str(caption_dim),
        "--synsets", synsets,
        "--limit_per_synset", str(limit_per_synset),
        "--max_faces", str(max_faces),
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
        "--wandb_project", wandb_project,
        "--wandb_run_name", wandb_run_name,
        "--wandb_entity", wandb_entity,
        "--wandb_mode", wandb_mode,
    ]
    print("[shapenet-hard-train] running:", " ".join(cmd))
    subprocess.check_call(cmd, env=env_pp)
    vol_runs.commit()

    if post_eval_policies:
        eval_dir = f"/runs/{out_subdir}-tensor-eval"
        eval_cmd = [
            sys.executable, "-u", "-m", "scripts.evaluate_baselines",
            "--dataset", "shapenet",
            "--shapenet_root", "/data/ShapeNetCore.v2",
            "--preproc_dir", preproc_dir,
            "--out_dir", eval_dir,
            "--policies", post_eval_policies,
            "--ckpt", f"{log_dir}/ppo_geoscout.zip",
            "--n_episodes", str(post_eval_episodes),
            "--n_envs", str(n_envs),
            "--seed", str(seed),
            "--device", "cuda",
            "--image_size", str(image_size),
            "--episode_len", str(episode_len),
            "--buffer_size", "30",
            "--grid_size", str(grid_size),
            "--obs_grid_size", str(obs_grid_size),
            "--coverage_hit_dilate_radius", str(coverage_hit_dilate_radius),
            "--auto_lookat_center",
            "--caption_dim", str(caption_dim),
            "--synsets", synsets,
            "--limit_per_synset", str(limit_per_synset),
            "--max_faces", str(max_faces),
            "--renderer_backend", renderer_backend,
            "--free_raycast_backend", free_raycast_backend,
            "--free_mask_apply_mode", free_mask_apply_mode,
            "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
            "--coverage_threshold", str(coverage_threshold),
            "--coverage_reward_type", coverage_reward_type,
            "--novelty_reward_scale", str(novelty_reward_scale),
            "--remaining_reward_scale", str(remaining_reward_scale),
            "--redundancy_penalty_scale", str(redundancy_penalty_scale),
            "--view_revisit_penalty_scale", str(view_revisit_penalty_scale),
            "--view_revisit_angle_deg", str(view_revisit_angle_deg),
            "--termination_bonus", str(termination_bonus),
            "--collision_penalty", str(collision_penalty),
        ]
        print("[shapenet-hard-eval] running:", " ".join(eval_cmd))
        subprocess.check_call(eval_cmd, env=env_pp)
        vol_runs.commit()


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=6 * 3600,
    memory=32 * 1024,
    retries=0,
)
def reward_sandbox_shapenet(
    policies: str = "repeat_top,random,axis6,ring",
    out_subdir: str = "reward-sandbox-shapenet-g128-obs32-ig",
    n_episodes: int = 64,
    n_envs: int = 8,
    episode_len: int = 20,
    synsets: str = "03001627,04256520,04379243",
    limit_per_synset: int = 200,
    preproc_dir: str = "/data/geoscout_preproc_g128",
    caption_dim: int = 384,
    max_faces: int = 5000,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    coverage_hit_dilate_radius: int = 1,
    coverage_reward_type: str = "information_gain",
    coverage_threshold: float = 0.55,
    coverage_reward_scale: float = 20.0,
    novelty_reward_scale: float = 25.0,
    remaining_reward_scale: float = 250.0,
    redundancy_penalty_scale: float = 10.0,
    view_revisit_penalty_scale: float = 5.0,
    view_revisit_angle_deg: float = 12.0,
    collision_penalty: float = 10.0,
    view_radius: float = 0.95,
    oracle_candidates: int = 96,
    oracle_chunk_size: int = 4,
    policy_seed_stride: int = 0,
    seed: int = 0,
    max_meshes: int = 24,
):
    """Small reward-design sandbox.

    Runs fixed policies in the exact tensor env and writes per-policy
    summaries including reward, novelty, redundancy, revisit penalty and
    FPS. Use this before launching expensive PPO when changing reward or
    grid resolution.
    """
    import os, subprocess, sys

    cmd = [
        sys.executable, "-u", "-m", "scripts.evaluate_baselines",
        "--dataset", "shapenet",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--out_dir", f"/runs/{out_subdir}",
        "--policies", policies,
        "--oracle_candidates", str(oracle_candidates),
        "--oracle_chunk_size", str(oracle_chunk_size),
        "--policy_seed_stride", str(policy_seed_stride),
        "--n_episodes", str(n_episodes),
        "--n_envs", str(n_envs),
        "--seed", str(seed),
        "--device", "cuda",
        "--image_size", str(image_size),
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--coverage_hit_dilate_radius", str(coverage_hit_dilate_radius),
        "--auto_lookat_center",
        "--caption_dim", str(caption_dim),
        "--synsets", synsets,
        "--limit_per_synset", str(limit_per_synset),
        "--max_faces", str(max_faces),
        "--coverage_threshold", str(coverage_threshold),
        "--coverage_reward_scale", str(coverage_reward_scale),
        "--coverage_reward_type", coverage_reward_type,
        "--novelty_reward_scale", str(novelty_reward_scale),
        "--remaining_reward_scale", str(remaining_reward_scale),
        "--redundancy_penalty_scale", str(redundancy_penalty_scale),
        "--view_revisit_penalty_scale", str(view_revisit_penalty_scale),
        "--view_revisit_angle_deg", str(view_revisit_angle_deg),
        "--termination_bonus", "0",
        "--collision_penalty", str(collision_penalty),
        "--view_radius", str(view_radius),
    ]
    if max_meshes > 0:
        cmd += ["--max_meshes", str(max_meshes)]
    print("[reward-sandbox] running:", " ".join(cmd))
    subprocess.check_call(cmd, env={**os.environ,
                                    "PYTHONPATH": "/workspace/GeoScout"})
    vol_runs.commit()


# ----------------------------------------------------------------------
# Validation rollout (deterministic-ish, viz dumper)
# ----------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=2 * 3600,
    retries=0,
)
def validate(
    ckpt_subdir: str = "geoscout-firstrun",
    out_subdir: str = "validate-geoscout-firstrun",
    n_episodes: int = 12,
    seq_names: str = "",
    seed: int = 0,
):
    """Rollout the trained ckpt and dump per-episode viz.

    Outputs land in /runs/<out_subdir>/episode_NNNN/.
    """
    import os, subprocess, sys
    ckpt_path = f"/runs/{ckpt_subdir}/ppo_geoscout.zip"
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found at {ckpt_path}")
    out_dir = f"/runs/{out_subdir}"
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "scripts.validate",
        "--ckpt", ckpt_path,
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", "/data/geoscout_preproc_g128",
        "--out_dir", out_dir,
        "--n_episodes", str(n_episodes),
        "--device", "cuda",
        "--seed", str(seed),
        "--grid_size", "128",
        "--obs_grid_size", "32",
    ]
    if seq_names:
        cmd += ["--seq_names", seq_names]
    print("[validate] running:", " ".join(cmd))
    subprocess.check_call(cmd, env={**__import__("os").environ,
                                    "PYTHONPATH": "/workspace/GeoScout"})
    vol_runs.commit()


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=2 * 3600,
    retries=0,
)
def validate_shapenet(
    ckpt_subdir: str = "shapenet-train",
    out_subdir: str = "validate-shapenet",
    n_episodes: int = 12,
    seq_names: str = "",
    seed: int = 0,
    caption_dim: int = 384,
):
    """Mirror of validate_abo but for ShapeNet ckpt."""
    import os, subprocess, sys
    ckpt_path = f"/runs/{ckpt_subdir}/ppo_geoscout.zip"
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found at {ckpt_path}")
    out_dir = f"/runs/{out_subdir}"
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "scripts.validate",
        "--ckpt", ckpt_path,
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", "/data/geoscout_preproc_g128",
        "--out_dir", out_dir,
        "--n_episodes", str(n_episodes),
        "--device", "cuda",
        "--seed", str(seed),
        "--episode_len", "50",
        "--buffer_size", "30",
        "--image_size", "400",
        "--grid_size", "128",
        "--obs_grid_size", "32",
        "--caption_dim", str(caption_dim),
        "--auto_lookat_center",
        "--dataset", "shapenet",
        "--skip_step_dumps",
        "--synsets", "03001627,04256520,04379243",   # chair / sofa / table
    ]
    if seq_names:
        cmd += ["--seq_names", seq_names]
    print("[validate_shapenet] running:", " ".join(cmd))
    subprocess.check_call(cmd, env={**os.environ,
                                    "PYTHONPATH": "/workspace/GeoScout"})
    vol_runs.commit()


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=6 * 3600,
    memory=32 * 1024,
    retries=0,
)
def smoke_validate_8ckpt_visual(
    out_subdir: str = "eval_8ckpt_smoke_validation",
    seq_names: str = (
        "03001627_1006be65e7bc937e9141f9b58470d646,"
        "04256520_1050790962944624febad4f49b26ec52,"
        "04379243_156d606fa86ba19c4eb174a255d0ec5e"
    ),
    action_modes: str = "discrete_det,continuous_det,axis6",
    discrete_ckpt_subdir: str = "train600-24m-discrete-s0-l40s-n128-wandb-0506",
    continuous_ckpt_subdir: str = "train600-24m-continuous-s0-l40s-n128-wandb-0506",
    preproc_dir: str = "/data/geoscout_preproc_g128_attr_v2",
    caption_jsonl: str = (
        "/data/geoscout_captions/"
        "full_attr_600_qwen25_7b_a100_batch64_tok256_array_v2_20260506_corrected.jsonl"
    ),
    episode_len: int = 50,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    caption_dim: int = 384,
    renderer_backend: str = "voxel_cuda",
    free_raycast_backend: str = "cuda",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
    max_faces: int = 5000,
    seed: int = 0,
):
    """Run visual smoke validation on the migrated xiaoleichu Modal data.

    Outputs are written to `/runs/<out_subdir>` and committed at function exit.
    The default keeps `auto_lookat_center=False`, matching the 24M runs.
    """
    import os
    import subprocess
    import sys

    out_dir = f"/runs/{out_subdir}"
    cmd = [
        sys.executable, "-u", "-m", "scripts.smoke_validate_8ckpt_visual",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--caption_jsonl", caption_jsonl,
        "--out_dir", out_dir,
        "--seq_names", seq_names,
        "--action_modes", action_modes,
        "--discrete_ckpt", f"/runs/{discrete_ckpt_subdir}/ppo_geoscout.zip",
        "--continuous_ckpt", f"/runs/{continuous_ckpt_subdir}/ppo_geoscout.zip",
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--image_size", str(image_size),
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--caption_dim", str(caption_dim),
        "--coverage_hit_dilate_radius", "1",
        "--coverage_threshold", "0.99",
        "--coverage_reward_scale", "20",
        "--termination_bonus", "1",
        "--collision_penalty", "10",
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
        "--max_faces", str(max_faces),
        "--seed", str(seed),
        "--device", "cuda",
    ]
    print("[smoke-8ckpt] running:", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": "/workspace/GeoScout"})
    finally:
        print("[smoke-8ckpt] committing /runs volume", flush=True)
        vol_runs.commit()


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=2 * 3600,
    retries=0,
)
def validate_abo(
    ckpt_subdir: str = "abo-train",
    out_subdir: str = "validate-abo",
    n_episodes: int = 12,
    seq_names: str = "",
    seed: int = 0,
    caption_dim: int = 384,
):
    """Rollout the trained ABO ckpt and dump per-episode viz dashboards.

    Mirrors `validate` but points at ABO paths and passes the training-
    time flags (caption_dim, auto_lookat_center) the policy was
    conditioned on.
    """
    import os, subprocess, sys
    ckpt_path = f"/runs/{ckpt_subdir}/ppo_geoscout.zip"
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found at {ckpt_path}")
    out_dir = f"/runs/{out_subdir}"
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "scripts.validate",
        "--ckpt", ckpt_path,
        "--shapenet_root", "/data/ABO",
        "--preproc_dir", "/data/abo_preproc",
        "--out_dir", out_dir,
        "--n_episodes", str(n_episodes),
        "--device", "cuda",
        "--seed", str(seed),
        "--episode_len", "80",
        "--buffer_size", "30",
        "--image_size", "400",
        "--grid_size", "20",
        "--caption_dim", str(caption_dim),
        "--auto_lookat_center",
        "--dataset", "abo",
        "--skip_step_dumps",
    ]
    if seq_names:
        cmd += ["--seq_names", seq_names]
    print("[validate_abo] running:", " ".join(cmd))
    subprocess.check_call(cmd, env={**__import__("os").environ,
                                    "PYTHONPATH": "/workspace/GeoScout"})
    vol_runs.commit()


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=2 * 3600,
    retries=0,
)
def smoke_test_single_obj(
    shape: str = "sphere",
    total_timesteps: int = 50_000,
    n_envs: int = 32,
    episode_len: int = 30,
    out_subdir: str = "smoke-single-obj",
    seed: int = 0,
):
    """End-to-end smoke test on a single synthetic mesh.

    Pipeline:
        1. Generate a sphere/cube/torus mesh (no ShapeNet needed).
        2. Preprocess it into a small smoke-test grid + 100k surface points.
        3. Train PPO for `total_timesteps` steps on this single object.
        4. Validate with deterministic-ish rollouts and dump full viz.

    Goal: confirm PPO converges and cr can reach ~1.0 on a single object,
    and the new viz artefacts (action_history.png, dashboard.png) render
    cleanly. Single-object setting is the easiest learnable task — if
    cr stays low here, the env / reward / encoder has a bug.
    """
    import os, subprocess, sys
    smoke_root = f"/data/geoscout_smoke/{shape}_test"
    smoke_preproc = f"/data/geoscout_smoke_preproc_{shape}"
    smoke_run = f"/runs/{out_subdir}-{shape}"
    os.makedirs(smoke_root, exist_ok=True)
    os.makedirs(smoke_preproc, exist_ok=True)

    env_pp = {**os.environ, "PYTHONPATH": "/workspace/GeoScout"}

    # 1) Generate the mesh.
    print(f"[smoke] step 1/4: generating {shape} mesh...")
    subprocess.check_call(
        [sys.executable, "-u", "-m", "scripts.make_test_mesh",
         "--out", smoke_root, "--shape", shape],
        env=env_pp,
    )
    vol_data.commit()

    # 2) Preprocess just this one object.
    print(f"[smoke] step 2/4: preprocessing...")
    subprocess.check_call(
        [sys.executable, "-u", "-m", "scripts.preprocess",
         "--shapenet_root", smoke_root,
         "--out_dir", smoke_preproc,
         "--grid_size", "20",
         "--n_surface_points", "100000",
         "--n_workers", "1"],
        env=env_pp,
    )
    vol_data.commit()

    # 3) Train PPO. Keep image_size aligned with GenNBV's 400x400 camera.
    #    The renderer chunks rays/triangles to avoid one giant intersection
    #    tensor; this is still compute-heavy until we add a faster backend.
    #    no --subproc — DummyVecEnv lets errors surface in the parent.
    print(f"[smoke] step 3/4: training PPO ({total_timesteps:,} steps, "
          f"n_envs={n_envs}) ...")
    subprocess.check_call(
        [sys.executable, "-u", "-m", "scripts.train",
         "--shapenet_root", smoke_root,
         "--preproc_dir", smoke_preproc,
         "--log_dir", smoke_run,
         "--total_timesteps", str(total_timesteps),
         "--n_envs", str(n_envs),
         "--seed", str(seed),
         "--device", "cuda",
         "--image_size", "400",
         "--episode_len", str(episode_len),
         "--buffer_size", "30",
         "--grid_size", "20",
         "--coverage_reward_scale", "20",
         "--short_path_grace_steps", "30",
         "--short_path_max_extra", "2",
         "--short_path_scale", "0.1",
         "--termination_bonus", "1",
         "--coverage_threshold", "0.99",
         "--n_steps", "128",
         "--batch_size", "128",
         "--n_epochs", "5",
         "--learning_rate", "1e-4",
         "--clip_range", "0.2",
         "--gamma", "0.99",
         "--ent_coef", "0.0",
         "--target_kl", "0.05",
         "--tensor_env",                             # IsaacGym-style: N envs as one [N,...] tensor batch
         "--tensor_env_n_envs", str(n_envs),
         "--auto_lookat_center"],                    # smoke shortcut: skip orientation, look at obj centre
        env=env_pp,
    )
    vol_runs.commit()

    # 4) Validate: 5 episodes on the same single object, dump full viz.
    print(f"[smoke] step 4/4: validating with viz...")
    val_dir = f"{smoke_run}/validate"
    os.makedirs(val_dir, exist_ok=True)
    subprocess.check_call(
        [sys.executable, "-u", "-m", "scripts.validate",
         "--ckpt", f"{smoke_run}/ppo_geoscout.zip",
         "--shapenet_root", smoke_root,
         "--preproc_dir", smoke_preproc,
         "--out_dir", val_dir,
         "--n_episodes", "5",
         "--device", "cuda",
         "--image_size", "400",
         "--buffer_size", "30",
         "--episode_len", str(episode_len),
         "--auto_lookat_center",   # mirror train-time
         "--seed", str(seed)],
        env=env_pp,
    )
    vol_runs.commit()
    print(f"[smoke] DONE. Inspect {smoke_run}/ for tb logs and "
          f"{val_dir}/ for per-episode viz.")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=2 * 3600,
    retries=0,
)
def smoke_test_multi_obj(
    total_timesteps: int = 80_000,
    n_envs: int = 32,
    episode_len: int = 30,
    out_subdir: str = "smoke-multi-obj",
    seed: int = 0,
):
    """End-to-end smoke on a 3-shape synthetic dataset (sphere / cube /
    torus) → 3 ShapeNet-style synsets → 3 caption_embs. Validates:
        1. multi-mesh TensorBatchEnv groups envs by mesh_id correctly
        2. caption_dim=384 pipeline (sentence-transformer encode →
           preproc.pt attach → env obs include → encoder reads)
        3. Phase 1 actually trains across multiple objects in one tensor
           batch — when `n_envs=32` and pool_size=3, each step renders
           3 mesh-groups in parallel.
    Single-process, single GPU; with 3 mesh groups (one render call each)
    we expect fps ~= sphere-only smoke / 3 ≈ 100-130.
    """
    import os, subprocess, sys
    smoke_root = "/data/geoscout_smoke_multi"
    smoke_preproc = "/data/geoscout_smoke_multi_preproc"
    cap_emb_path = f"{smoke_preproc}/category_embeddings.pt"
    smoke_run = f"/runs/{out_subdir}"
    os.makedirs(smoke_root, exist_ok=True)
    os.makedirs(smoke_preproc, exist_ok=True)
    env_pp = {**os.environ, "PYTHONPATH": "/workspace/GeoScout"}

    # 1) Generate 3 meshes (sphere/cube/torus) into ShapeNet-style dirs.
    print("[smoke-multi] step 1/5: generating 3 synthetic meshes...")
    subprocess.check_call(
        [sys.executable, "-u", "-m", "scripts.make_test_mesh",
         "--out", smoke_root, "--shape", "all"],
        env=env_pp,
    )
    vol_data.commit()

    # 2) Encode the 55 ShapeNet category names with sentence-transformer
    # (the smoke meshes use real synsets — bowl/cabinet/jar — so the
    # global category embedding lookup works unchanged).
    print("[smoke-multi] step 2/5: encoding category captions...")
    subprocess.check_call(
        [sys.executable, "-u", "-m", "scripts.precompute_category_embeddings",
         "--out", cap_emb_path,
         "--model", "sentence-transformers/all-MiniLM-L6-v2",
         "--device", "cpu"],
        env=env_pp,
    )

    # 3) Preprocess all 3 meshes (with caption_emb attach).
    print("[smoke-multi] step 3/5: preprocessing 3 meshes...")
    subprocess.check_call(
        [sys.executable, "-u", "-m", "scripts.preprocess",
         "--shapenet_root", smoke_root,
         "--out_dir", smoke_preproc,
         "--grid_size", "20",
         "--n_surface_points", "10000",  # small — keep .pt tiny
         "--caption_emb_path", cap_emb_path,
         "--n_workers", "1"],
        env=env_pp,
    )
    vol_data.commit()

    # 4) Train PPO with caption_dim=384 + multi-mesh tensor batch.
    print(f"[smoke-multi] step 4/5: training PPO ({total_timesteps:,} steps, "
          f"n_envs={n_envs}, pool=3, caption_dim=384)...")
    subprocess.check_call(
        [sys.executable, "-u", "-m", "scripts.train",
         "--shapenet_root", smoke_root,
         "--preproc_dir", smoke_preproc,
         "--log_dir", smoke_run,
         "--total_timesteps", str(total_timesteps),
         "--n_envs", str(n_envs),
         "--seed", str(seed),
         "--device", "cuda",
         "--image_size", "400",
         "--episode_len", str(episode_len),
         "--buffer_size", "30",
         "--grid_size", "20",
         "--coverage_reward_scale", "20",
         "--short_path_grace_steps", "30",
         "--short_path_max_extra", "2",
         "--short_path_scale", "0.1",
         "--termination_bonus", "1",
         "--coverage_threshold", "0.99",
         "--n_steps", "128",
         "--batch_size", "128",
         "--n_epochs", "5",
         "--learning_rate", "1e-4",
         "--clip_range", "0.2",
         "--gamma", "0.99",
         "--ent_coef", "0.0",
         "--target_kl", "0.05",
         "--tensor_env",
         "--tensor_env_n_envs", str(n_envs),
         "--auto_lookat_center",
         "--caption_dim", "384"],
        env=env_pp,
    )
    vol_runs.commit()

    # 5) Validate on each of the 3 shapes (1 episode per shape).
    print("[smoke-multi] step 5/5: validating with viz...")
    val_dir = f"{smoke_run}/validate"
    os.makedirs(val_dir, exist_ok=True)
    subprocess.check_call(
        [sys.executable, "-u", "-m", "scripts.validate",
         "--ckpt", f"{smoke_run}/ppo_geoscout.zip",
         "--shapenet_root", smoke_root,
         "--preproc_dir", smoke_preproc,
         "--out_dir", val_dir,
         "--n_episodes", "9",       # 3 shapes × 3 random rollouts
         "--device", "cuda",
         "--image_size", "400",
         "--buffer_size", "30",
         "--episode_len", str(episode_len),
         "--auto_lookat_center",
         "--caption_dim", "384",
         "--seed", str(seed)],
        env=env_pp,
    )
    vol_runs.commit()
    print(f"[smoke-multi] DONE. Inspect {smoke_run}/ + {val_dir}/.")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    secrets=[wandb_secret],
    timeout=24 * 3600,
    retries=0,
)
def train_abo(
    total_timesteps: int = 4_000_000,    # was 1M — too few; v2 logs showed kl≈0.003 (no learning)
    n_envs: int = 32,
    episode_len: int = 50,                # ABO needs ~50 steps to walk around an object
    categories: str = "CHAIR,LAMP,SOFA,TABLE",
    limit_per_category: int = 200,
    out_subdir: str = "abo-train",
    caption_dim: int = 384,
    wandb_project: str = "geoscout",
    wandb_run_name: str = "abo-firstrun",
    wandb_entity: str = "xiaoleichu-university-of-california-berkeley",
    wandb_mode: str = "online",          # "online" | "offline" | "disabled"
    max_faces: int = 5000,                # quadric-decimate per-mesh tris cap
    coverage_reward_type: str = "log",   # log → steeper gradient at high cr
    coverage_threshold: float = 0.92,    # was 0.99 — physically unreachable on decimated meshes
    termination_bonus: float = 1.0,
    collision_penalty: float = 10.0,
    seed: int = 0,
):
    """Train PPO on the ABO dataset (after `download_abo_subset` +
    `precompute_abo_captions` + `preprocess_abo` have been run).

    Defaults aim for a quick prototype: 4 furniture categories × 200
    objects = 800 meshes, 1M timesteps. With TensorBatchEnv at
    fps≈100-300 (4 mesh groups → 4 GPU launches per step) the run
    finishes in 1-3 hr on L4.
    """
    import os, subprocess, sys
    log_dir = f"/runs/{out_subdir}"
    cmd = [
        sys.executable, "-u", "-m", "scripts.train",
        "--dataset", "abo",
        "--shapenet_root", "/data/ABO",                # ABO root
        "--preproc_dir", "/data/abo_preproc",
        "--log_dir", log_dir,
        "--total_timesteps", str(total_timesteps),
        "--n_envs", str(n_envs),
        "--seed", str(seed),
        "--device", "cuda",
        "--image_size", "400",
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--grid_size", "20",
        "--coverage_reward_scale", "20",
        "--short_path_grace_steps", "30",
        "--short_path_max_extra", "2",
        "--short_path_scale", "0.1",
        "--termination_bonus", str(termination_bonus),
        "--coverage_threshold", str(coverage_threshold),
        "--coverage_reward_type", coverage_reward_type,
        "--collision_penalty", str(collision_penalty),
        "--n_steps", "128", "--batch_size", "128", "--n_epochs", "5",
        "--learning_rate", "1e-4", "--clip_range", "0.2",
        "--gamma", "0.99", "--ent_coef", "0.0", "--target_kl", "0.05",
        "--tensor_env",
        "--tensor_env_n_envs", str(n_envs),
        "--auto_lookat_center",
        "--caption_dim", str(caption_dim),
        "--categories", categories,
        "--limit_per_synset", str(limit_per_category),
        "--max_faces", str(max_faces),
        "--wandb_project", wandb_project,
        "--wandb_run_name", wandb_run_name,
        "--wandb_entity", wandb_entity,
        "--wandb_mode", wandb_mode,
    ]
    print("[abo-train] running:", " ".join(cmd))
    subprocess.check_call(cmd, env={**os.environ,
                                    "PYTHONPATH": "/workspace/GeoScout"})
    vol_runs.commit()


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=6 * 3600,
    memory=32 * 1024,
    retries=0,
)
def evaluate_baselines_shapenet(
    policies: str = "random",
    ckpt_subdir: str = "shapenet-train-v3",
    out_subdir: str = "",
    n_episodes: int = 128,
    n_envs: int = 32,
    episode_len: int = 50,
    synsets: str = "03001627,04256520,04379243",
    categories: str = "",
    seq_names: str = "",
    limit_per_synset: int = 200,
    preproc_dir: str = "/data/geoscout_preproc_g128",
    caption_dim: int = 384,
    max_faces: int = 5000,
    image_size: int = 400,
    grid_size: int = 128,
    obs_grid_size: int = 32,
    coverage_hit_dilate_radius: int = 1,
    coverage_reward_type: str = "linear",
    coverage_threshold: float = 0.99,
    termination_bonus: float = 1.0,
    coverage_reward_scale: float = 20.0,
    novelty_reward_scale: float = 0.0,
    remaining_reward_scale: float = 0.0,
    redundancy_penalty_scale: float = 0.0,
    view_revisit_penalty_scale: float = 0.0,
    view_revisit_angle_deg: float = 12.0,
    collision_penalty: float = 10.0,
    view_radius: float = 0.95,
    oracle_candidates: int = 96,
    oracle_chunk_size: int = 4,
    seed: int = 0,
    deterministic: bool = False,
    max_meshes: int = 0,
    renderer_backend: str = "nvdiffrast",
    free_raycast_backend: str = "auto",
    free_mask_apply_mode: str = "triton",
    triton_bresenham_block_rays: int = 64,
):
    """Evaluate PPO and hand-written baselines in the tensor training env.

    This is the comparison that matters for the current reward-design
    question: same mesh pool, same TensorBatchEnv coverage, same horizon,
    same auto-lookat/caption observation shape as the v3 ShapeNet run.
    """
    import os, subprocess, sys

    policy_tag = policies.replace(",", "_").replace(" ", "")
    out = out_subdir or f"baseline-shapenet-{policy_tag}-s{seed}"
    ckpt_path = f"/runs/{ckpt_subdir}/ppo_geoscout.zip"
    cmd = [
        sys.executable, "-u", "-m", "scripts.evaluate_baselines",
        "--dataset", "shapenet",
        "--shapenet_root", "/data/ShapeNetCore.v2",
        "--preproc_dir", preproc_dir,
        "--out_dir", f"/runs/{out}",
        "--policies", policies,
        "--ckpt", ckpt_path,
        "--oracle_candidates", str(oracle_candidates),
        "--oracle_chunk_size", str(oracle_chunk_size),
        "--n_episodes", str(n_episodes),
        "--n_envs", str(n_envs),
        "--seed", str(seed),
        "--device", "cuda",
        "--image_size", str(image_size),
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--coverage_hit_dilate_radius", str(coverage_hit_dilate_radius),
        "--auto_lookat_center",
        "--caption_dim", str(caption_dim),
        "--synsets", synsets,
        "--limit_per_synset", str(limit_per_synset),
        "--max_faces", str(max_faces),
        "--coverage_threshold", str(coverage_threshold),
        "--coverage_reward_scale", str(coverage_reward_scale),
        "--coverage_reward_type", coverage_reward_type,
        "--termination_bonus", str(termination_bonus),
        "--novelty_reward_scale", str(novelty_reward_scale),
        "--remaining_reward_scale", str(remaining_reward_scale),
        "--redundancy_penalty_scale", str(redundancy_penalty_scale),
        "--view_revisit_penalty_scale", str(view_revisit_penalty_scale),
        "--view_revisit_angle_deg", str(view_revisit_angle_deg),
        "--collision_penalty", str(collision_penalty),
        "--view_radius", str(view_radius),
        "--renderer_backend", renderer_backend,
        "--free_raycast_backend", free_raycast_backend,
        "--free_mask_apply_mode", free_mask_apply_mode,
        "--triton_bresenham_block_rays", str(triton_bresenham_block_rays),
    ]
    if categories:
        cmd += ["--categories", categories]
    if seq_names:
        cmd += ["--seq_names", seq_names]
    if deterministic:
        cmd += ["--deterministic"]
    if max_meshes > 0:
        cmd += ["--max_meshes", str(max_meshes)]
    print("[baseline-shapenet] running:", " ".join(cmd))
    subprocess.check_call(cmd, env={**os.environ,
                                    "PYTHONPATH": "/workspace/GeoScout"})
    vol_runs.commit()


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol_data, "/runs": vol_runs},
    timeout=6 * 3600,
    retries=0,
)
def evaluate_baselines_abo(
    policies: str = "random",
    ckpt_subdir: str = "abo-train-v3",
    out_subdir: str = "",
    n_episodes: int = 128,
    n_envs: int = 32,
    episode_len: int = 50,
    categories: str = "CHAIR,LAMP,SOFA,TABLE",
    limit_per_category: int = 200,
    preproc_dir: str = "/data/abo_preproc",
    caption_dim: int = 384,
    max_faces: int = 5000,
    image_size: int = 400,
    grid_size: int = 20,
    obs_grid_size: int = 0,
    coverage_reward_type: str = "log",
    coverage_threshold: float = 0.92,
    termination_bonus: float = 1.0,
    coverage_reward_scale: float = 20.0,
    novelty_reward_scale: float = 0.0,
    remaining_reward_scale: float = 0.0,
    redundancy_penalty_scale: float = 0.0,
    view_revisit_penalty_scale: float = 0.0,
    view_revisit_angle_deg: float = 12.0,
    collision_penalty: float = 10.0,
    view_radius: float = 0.95,
    seed: int = 0,
    deterministic: bool = False,
    max_meshes: int = 0,
    renderer_backend: str = "nvdiffrast",
):
    """ABO twin of evaluate_baselines_shapenet."""
    import os, subprocess, sys

    policy_tag = policies.replace(",", "_").replace(" ", "")
    out = out_subdir or f"baseline-abo-{policy_tag}-s{seed}"
    ckpt_path = f"/runs/{ckpt_subdir}/ppo_geoscout.zip"
    cmd = [
        sys.executable, "-u", "-m", "scripts.evaluate_baselines",
        "--dataset", "abo",
        "--shapenet_root", "/data/ABO",
        "--preproc_dir", preproc_dir,
        "--out_dir", f"/runs/{out}",
        "--policies", policies,
        "--ckpt", ckpt_path,
        "--n_episodes", str(n_episodes),
        "--n_envs", str(n_envs),
        "--seed", str(seed),
        "--device", "cuda",
        "--image_size", str(image_size),
        "--episode_len", str(episode_len),
        "--buffer_size", "30",
        "--grid_size", str(grid_size),
        "--obs_grid_size", str(obs_grid_size),
        "--auto_lookat_center",
        "--caption_dim", str(caption_dim),
        "--categories", categories,
        "--limit_per_synset", str(limit_per_category),
        "--max_faces", str(max_faces),
        "--coverage_threshold", str(coverage_threshold),
        "--coverage_reward_scale", str(coverage_reward_scale),
        "--coverage_reward_type", coverage_reward_type,
        "--termination_bonus", str(termination_bonus),
        "--novelty_reward_scale", str(novelty_reward_scale),
        "--remaining_reward_scale", str(remaining_reward_scale),
        "--redundancy_penalty_scale", str(redundancy_penalty_scale),
        "--view_revisit_penalty_scale", str(view_revisit_penalty_scale),
        "--view_revisit_angle_deg", str(view_revisit_angle_deg),
        "--collision_penalty", str(collision_penalty),
        "--view_radius", str(view_radius),
        "--renderer_backend", renderer_backend,
    ]
    if deterministic:
        cmd += ["--deterministic"]
    if max_meshes > 0:
        cmd += ["--max_meshes", str(max_meshes)]
    print("[baseline-abo] running:", " ".join(cmd))
    subprocess.check_call(cmd, env={**os.environ,
                                    "PYTHONPATH": "/workspace/GeoScout"})
    vol_runs.commit()


@app.local_entrypoint()
def main():
    print(
        "GeoScout — Modal entrypoints:\n"
        "  Setup:\n"
        "    modal run GeoScout/scripts/modal_app.py::download_shapenet\n"
        "    modal run GeoScout/scripts/modal_app.py::preprocess\n"
        "  Train:\n"
        "    modal run --detach GeoScout/scripts/modal_app.py::train\n"
        "  Validate:\n"
        "    modal run GeoScout/scripts/modal_app.py::validate\n"
        "  Tensor-env baselines:\n"
        "    modal run GeoScout/scripts/modal_app.py::evaluate_baselines_shapenet"
    )
