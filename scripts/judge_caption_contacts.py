#!/usr/bin/env python3
"""Review and apply manual geometry-caption judgements for GeoScout captions.

This script intentionally keeps the human judgement artifact separate from the
caption JSONL. A reviewer edits only a compact JSON file containing corrected
attribute sets; this script then regenerates the text-encoder caption, attrs
array, change report, and progress summary in one deterministic pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ATTRIBUTE_ORDER = [
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

ATTRIBUTE_LABELS = {
    "has_single_main_body": "single main body",
    "has_flat_horizontal_surface": "flat horizontal surface",
    "has_vertical_panel": "vertical panel",
    "has_curved_shell": "curved shell",
    "has_boxy_volume": "boxy volume",
    "has_distinct_seat": "distinct seat",
    "has_backrest": "backrest",
    "has_armrests": "armrests",
    "has_four_or_more_legs": "four or more legs",
    "has_pedestal_or_central_support": "pedestal or central support",
    "has_star_base_or_wheels": "star base or wheels",
    "has_thin_supports": "thin supports",
    "has_cross_braces_or_bars": "cross braces or bars",
    "has_slats": "slats",
    "has_perforations_or_holes": "perforations or holes",
    "has_open_gaps": "open gaps",
    "has_concavity": "concavity",
    "has_cylindrical_parts": "cylindrical parts",
    "has_asymmetry": "asymmetry",
    "has_occluded_or_hidden_supports": "occluded / hidden supports",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def join_phrases(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def make_caption(category: str, attributes: dict[str, int]) -> str:
    phrases = [ATTRIBUTE_LABELS[key] for key in ATTRIBUTE_ORDER if int(attributes.get(key, 0) or 0)]
    if phrases:
        return f"A {category} with {join_phrases(phrases)}."
    return f"A {category}."


def present_keys(caption: dict[str, Any]) -> list[str]:
    attrs = caption.get("attributes")
    if not isinstance(attrs, dict):
        raw = caption.get("attrs")
        if isinstance(raw, list):
            return [key for key, value in zip(ATTRIBUTE_ORDER, raw) if int(value or 0)]
        return []
    order = caption.get("attribute_order")
    if not isinstance(order, list):
        order = ATTRIBUTE_ORDER
    return [key for key in order if int(attrs.get(key, 0) or 0)]


def present_labels(caption: dict[str, Any]) -> list[str]:
    return [ATTRIBUTE_LABELS.get(key, key) for key in present_keys(caption)]


def make_review_atlas(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.jsonl)
    out_dir: Path = args.out_dir
    page_dir = out_dir / "atlas_pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    font_title = load_font(30, bold=True)
    font_small = load_font(20)
    font_caption = load_font(22)
    font_chip = load_font(18)
    font_idx = load_font(26, bold=True)

    page_paths: list[Path] = []
    batch_size = args.batch_size
    for page_idx, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[start : start + batch_size]
        cell_w = 1960
        cell_h = 1160
        cols = 1
        rows_per_page = len(batch)
        page = Image.new("RGB", (cell_w * cols, cell_h * rows_per_page), "#f5f7fb")
        draw = ImageDraw.Draw(page)
        for local_idx, row in enumerate(batch):
            y0 = local_idx * cell_h
            object_id = str(row.get("object_id"))
            caption = row.get("caption") if isinstance(row.get("caption"), dict) else {}
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
            image_path = args.image_dir / f"{object_id}.png"
            draw.rectangle((0, y0, cell_w, y0 + cell_h), fill="#f5f7fb")
            draw.text((24, y0 + 18), f"{start + local_idx + 1:03d}  {object_id}", fill="#111827", font=font_idx)
            draw.text((24, y0 + 58), f"category: {caption.get('category') or source.get('category')}", fill="#475569", font=font_small)
            if image_path.exists():
                im = Image.open(image_path).convert("RGB")
                im.thumbnail((1260, 504), Image.Resampling.LANCZOS)
                page.paste(im, (24, y0 + 96))
            else:
                draw.rectangle((24, y0 + 96, 1284, y0 + 600), outline="#ef4444", width=4)
                draw.text((48, y0 + 260), f"missing image: {image_path.name}", fill="#ef4444", font=font_title)

            x_text = 1320
            text_width = 600
            draw.text((x_text, y0 + 96), "current text-encoder caption", fill="#111827", font=font_title)
            cap_text = str(caption.get("embedding_caption", ""))
            yy = y0 + 142
            for line in wrap_text(draw, cap_text, font_caption, text_width):
                draw.text((x_text, yy), line, fill="#111827", font=font_caption)
                yy += 30
            yy += 16
            draw.text((x_text, yy), "present attributes", fill="#111827", font=font_title)
            yy += 42
            labels = present_labels(caption)
            if not labels:
                labels = ["none"]
            for label in labels[:12]:
                draw.rounded_rectangle((x_text, yy, x_text + text_width, yy + 30), radius=6, fill="#e0f2fe", outline="#bae6fd")
                draw.text((x_text + 10, yy + 5), label, fill="#0f172a", font=font_chip)
                yy += 34
            extra = len(labels) - 12
            if extra > 0:
                draw.text((x_text, yy + 4), f"+ {extra} more", fill="#64748b", font=font_chip)
                yy += 28
            yy += 12
            qtext = (
                f"quality={quality.get('quality_score_auto')}; "
                f"manual={quality.get('needs_manual_review')}; "
                f"attrs={quality.get('num_present_attributes')}"
            )
            draw.text((x_text, yy), qtext, fill="#475569", font=font_small)
            yy += 30
            if row.get("manual_caption_override"):
                draw.text((x_text, yy), "already manually overridden", fill="#166534", font=font_small)

            draw.line((0, y0 + cell_h - 1, cell_w, y0 + cell_h - 1), fill="#cbd5e1", width=2)

        page_path = page_dir / f"caption_review_page_{page_idx:03d}_{start + 1:03d}-{start + len(batch):03d}.png"
        page.save(page_path, optimize=True)
        page_paths.append(page_path)

    index_lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>GeoScout Caption Judgement Atlas</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;background:#f8fafc;color:#111827;margin:24px}"
        ".page{margin:20px 0;padding:16px;background:white;border:1px solid #e5e7eb;border-radius:8px}"
        "img{max-width:100%;height:auto;border:1px solid #cbd5e1}</style></head><body>",
        f"<h1>GeoScout Caption Judgement Atlas</h1><p>{len(rows)} samples, {batch_size} per page.</p>",
    ]
    for path in page_paths:
        index_lines.append(f"<div class='page'><h2>{html.escape(path.name)}</h2><img src='atlas_pages/{html.escape(path.name)}'></div>")
    index_lines.append("</body></html>")
    (out_dir / "caption_review_atlas.html").write_text("\n".join(index_lines), encoding="utf-8")

    progress = {
        "schema_version": "geoscout_caption_judgement_progress_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": str(args.jsonl),
        "image_dir": str(args.image_dir),
        "num_samples": len(rows),
        "batch_size": batch_size,
        "num_pages": len(page_paths),
        "reviewed_count": 0,
        "changed_count": 0,
        "pages": [str(path) for path in page_paths],
    }
    (out_dir / "review_progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(progress, indent=2, sort_keys=True))


def snapshot_originals(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.jsonl)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        object_id = str(row.get("object_id", ""))
        caption = row.get("caption") if isinstance(row.get("caption"), dict) else {}
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        attrs = present_keys(caption)
        item = {
            "index": idx,
            "object_id": object_id,
            "category": caption.get("category") or source.get("category") or source.get("category_hint"),
            "embedding_caption_before_manual_judgement": caption.get("embedding_caption", ""),
            "final_caption_before_manual_judgement": caption.get("final_caption", ""),
            "present_attributes_before_manual_judgement": attrs,
            "attrs_array_before_manual_judgement": caption.get("attrs", []),
            "quality_before_manual_judgement": quality,
            "contact_sheet": str(args.image_dir / f"{object_id}.png") if args.image_dir else "",
        }
        snapshot_rows.append(item)
        csv_rows.append(
            {
                "index": idx,
                "object_id": object_id,
                "category": item["category"],
                "embedding_caption_before_manual_judgement": item["embedding_caption_before_manual_judgement"],
                "present_attributes_before_manual_judgement": "; ".join(attrs),
                "contact_sheet": item["contact_sheet"],
            }
        )

    jsonl_path = out_dir / "captions_before_manual_judgement.jsonl"
    csv_path = out_dir / "captions_before_manual_judgement.csv"
    write_jsonl(jsonl_path, snapshot_rows)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "object_id",
                "category",
                "embedding_caption_before_manual_judgement",
                "present_attributes_before_manual_judgement",
                "contact_sheet",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    manifest = {
        "schema_version": "geoscout_caption_before_manual_judgement_snapshot_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": str(args.jsonl),
        "source_jsonl_sha256": file_sha256(args.jsonl),
        "image_dir": str(args.image_dir) if args.image_dir else "",
        "num_records": len(snapshot_rows),
        "jsonl_snapshot": str(jsonl_path),
        "csv_snapshot": str(csv_path),
    }
    manifest_path = out_dir / "captions_before_manual_judgement_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def load_judgements(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "judgements" in data:
        data = data["judgements"]
    if not isinstance(data, dict):
        raise SystemExit(f"Judgement file must be an object keyed by object_id: {path}")
    return {str(key): value for key, value in data.items()}


def apply_judgements(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.jsonl)
    judgements = load_judgements(args.judgements_json)
    out_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    row_ids = {str(row.get("object_id")) for row in rows}
    missing = sorted(set(judgements) - row_ids)
    if missing:
        raise SystemExit(f"Judgements for unknown object_id(s): {missing[:10]}")

    for row in rows:
        object_id = str(row.get("object_id"))
        caption = row.get("caption") if isinstance(row.get("caption"), dict) else {}
        judgement = judgements.get(object_id)
        previous_text = str(caption.get("embedding_caption", ""))
        previous_attrs = list(present_keys(caption))
        if judgement:
            status = str(judgement.get("status", "reviewed"))
            if status not in {"reviewed", "corrected", "keep"}:
                raise SystemExit(f"{object_id}: unsupported status={status}")
            present = judgement.get("present_attributes", previous_attrs)
            if not isinstance(present, list):
                raise SystemExit(f"{object_id}: present_attributes must be a list")
            unknown = sorted(set(present) - set(ATTRIBUTE_ORDER))
            if unknown:
                raise SystemExit(f"{object_id}: unknown attributes {unknown}")
            category = str(judgement.get("category") or caption.get("category") or row.get("source", {}).get("category_hint") or object_id.split("_")[0])
            present_set = set(present)
            previous_set = set(previous_attrs)
            attributes = {key: int(key in present_set) for key in ATTRIBUTE_ORDER}
            attr_changed = present_set != previous_set
            force_rewrite = status == "corrected"
            text = make_caption(category, attributes) if (attr_changed or force_rewrite) else previous_text
            changed = text != previous_text or attr_changed
            caption["attribute_order"] = ATTRIBUTE_ORDER
            caption["attributes"] = attributes
            caption["attrs"] = [attributes[key] for key in ATTRIBUTE_ORDER]
            caption["category"] = category
            caption["embedding_caption"] = text
            caption["final_caption"] = text
            caption["composer_version"] = "geom_attribute_composer_v2_attrs_only_manual_judged_v1"
            row["caption"] = caption
            quality = row.setdefault("quality", {})
            if isinstance(quality, dict):
                quality["needs_manual_review"] = False
                quality["num_present_attributes"] = int(sum(attributes.values()))
                quality["quality_score_auto"] = max(int(quality.get("quality_score_auto") or 0), 3)
            row["manual_caption_judgement"] = {
                "schema_version": "geoscout_caption_manual_judgement_v1",
                "reviewed_at_utc": now,
                "status": "corrected" if changed else "reviewed",
                "changed": changed,
                "previous_embedding_caption": previous_text,
                "corrected_embedding_caption": text,
                "previous_present_attributes": previous_attrs,
                "corrected_present_attributes": present,
                "reviewer_note": judgement.get("reviewer_note", ""),
            }
            if changed:
                changed_rows.append(row)
            csv_rows.append(
                {
                    "object_id": object_id,
                    "changed": changed,
                    "previous_caption": previous_text,
                    "corrected_caption": text,
                    "previous_attributes": "; ".join(previous_attrs),
                    "corrected_attributes": "; ".join(present),
                    "reviewer_note": judgement.get("reviewer_note", ""),
                }
            )
        out_rows.append(row)

    write_jsonl(args.output_jsonl, out_rows)
    write_jsonl(args.changed_jsonl, changed_rows)
    args.changed_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.changed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "object_id",
                "changed",
                "previous_caption",
                "corrected_caption",
                "previous_attributes",
                "corrected_attributes",
                "reviewer_note",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "schema_version": "geoscout_caption_judgement_apply_summary_v1",
        "source_jsonl": str(args.jsonl),
        "judgements_json": str(args.judgements_json),
        "output_jsonl": str(args.output_jsonl),
        "num_records": len(out_rows),
        "num_reviewed": len(judgements),
        "num_changed": len(changed_rows),
        "changed_object_ids": [str(row.get("object_id")) for row in changed_rows],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_review_html(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.jsonl)
    judgements = load_judgements(args.judgements_json)
    output_html: Path = args.output_html
    output_html.parent.mkdir(parents=True, exist_ok=True)

    def attr_chips(keys: list[str], added: set[str] | None = None, removed: set[str] | None = None) -> str:
        added = added or set()
        removed = removed or set()
        if not keys:
            return "<span class='chip muted'>none</span>"
        chips: list[str] = []
        for key in keys:
            state = " added" if key in added else " removed" if key in removed else ""
            label = ATTRIBUTE_LABELS.get(key, key)
            chips.append(f"<span class='chip{state}'>{html.escape(label)}</span>")
        return "".join(chips)

    def split_view_grid(img_rel: str) -> str:
        view_labels = [f"view {i:02d}" for i in range(1, 11)]
        cells: list[str] = []
        for i, label in enumerate(view_labels):
            col = i % 5
            row = i // 5
            x = col * 25
            y = row * 100
            cells.append(
                "<a class='view-tile zoomable' "
                f"href='{html.escape(img_rel)}' "
                f"style=\"background-image:url('{html.escape(img_rel)}');background-position:{x}% {y}%\" "
                f"data-img='{html.escape(img_rel)}' aria-label='{html.escape(label)}'>"
                f"<span>{html.escape(label)}</span>"
                "</a>"
            )
        return "<div class='views-grid'>" + "".join(cells) + "</div>"

    cards: list[str] = []
    changed_count = 0
    reviewed_count = 0
    for idx, row in enumerate(rows, start=1):
        object_id = str(row.get("object_id", ""))
        caption = row.get("caption") if isinstance(row.get("caption"), dict) else {}
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        before_text = str(caption.get("embedding_caption", ""))
        before_attrs = present_keys(caption)
        judgement = judgements.get(object_id)
        reviewed = judgement is not None
        if reviewed:
            reviewed_count += 1
        after_text = before_text
        after_attrs = before_attrs
        note = ""
        if judgement:
            present = judgement.get("present_attributes", before_attrs)
            category = str(judgement.get("category") or caption.get("category") or source.get("category") or source.get("category_hint") or object_id.split("_")[0])
            attrs_dict = {key: int(key in set(present)) for key in ATTRIBUTE_ORDER}
            force_rewrite = str(judgement.get("status", "")) == "corrected"
            if set(present) != set(before_attrs) or force_rewrite:
                after_text = make_caption(category, attrs_dict)
            after_attrs = list(present)
            note = str(judgement.get("reviewer_note", ""))
        changed = reviewed and (after_text != before_text or set(after_attrs) != set(before_attrs))
        if changed:
            changed_count += 1
        category_label = str(caption.get("category") or source.get("category") or source.get("category_hint") or "")
        img_rel = os.path.relpath(args.image_dir / f"{object_id}.png", output_html.parent).replace(os.sep, "/")
        added_attrs = set(after_attrs) - set(before_attrs)
        removed_attrs = set(before_attrs) - set(after_attrs)
        status_label = "changed" if changed else "reviewed" if reviewed else "pending"
        cards.append(
            "\n".join(
                [
                    f"<article id='sample-{idx:03d}' class='card {status_label}' data-status='{status_label}' data-object='{html.escape(object_id)}' data-category='{html.escape(category_label)}'>",
                    "<div class='card-head'>",
                    f"<div><h2>{idx:03d}. <code>{html.escape(object_id)}</code></h2><p>{html.escape(category_label)} | status: <b>{status_label}</b></p></div>",
                    f"<a href='{html.escape(img_rel)}'>open full contact sheet</a>",
                    "</div>",
                    "<div class='review-layout'>",
                    "<section class='visual-pane'>",
                    "<h3>Ten rendered views seen by the VLM</h3>",
                    split_view_grid(img_rel),
                    "<details>",
                    "<summary>full contact sheet</summary>",
                    f"<img class='contact zoomable' loading='lazy' src='{html.escape(img_rel)}' data-img='{html.escape(img_rel)}' alt='{html.escape(object_id)} contact sheet'>",
                    "</details>",
                    "</section>",
                    "<div class='diff'>",
                    "<section>",
                    "<h3>Before</h3>",
                    f"<p class='caption'>{html.escape(before_text)}</p>",
                    f"<div class='attrs'>{attr_chips(before_attrs, removed=removed_attrs)}</div>",
                    "</section>",
                    "<section>",
                    "<h3>After</h3>",
                    f"<p class='caption'>{html.escape(after_text)}</p>",
                    f"<div class='attrs'>{attr_chips(after_attrs, added=added_attrs)}</div>",
                    "</section>",
                    f"<p class='note'><b>Reviewer note:</b> {html.escape(note) if note else 'not reviewed yet'}</p>",
                    "</div>",
                    "</div>",
                    "</article>",
                ]
            )
        )

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>GeoScout Caption Judgement Review</title>
  <style>
    :root {{ color-scheme: light; --bg:#f6f8fb; --ink:#111827; --muted:#64748b; --line:#d8dee9; --blue:#1d4ed8; --green:#15803d; --red:#b91c1c; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 10; background: rgba(255,255,255,0.96); border-bottom: 1px solid var(--line); padding: 14px 22px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 12px 20px; color: var(--muted); font-size: 14px; }}
    .controls {{ display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }}
    input, select {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; font-size: 14px; background: white; }}
    main {{ padding: 18px 22px 60px; }}
    .card {{ background: white; border: 1px solid var(--line); border-left: 6px solid #94a3b8; border-radius: 8px; padding: 14px; margin: 0 0 18px; box-shadow: 0 1px 2px rgba(15,23,42,0.05); }}
    .card.changed {{ border-left-color: var(--red); }}
    .card.reviewed {{ border-left-color: var(--green); }}
    .card.pending {{ border-left-color: #94a3b8; }}
    .card-head {{ display: flex; justify-content: space-between; align-items: start; gap: 12px; }}
    h2 {{ font-size: 18px; margin: 0; }}
    h3 {{ margin: 0 0 6px; font-size: 15px; }}
    p {{ margin: 4px 0; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .review-layout {{ display: grid; grid-template-columns: minmax(620px, 1.15fr) minmax(440px, .85fr); gap: 14px; align-items: start; }}
    .visual-pane {{ min-width: 0; }}
    .views-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; margin: 10px 0; }}
    .view-tile {{ position: relative; display: block; aspect-ratio: 1 / 1; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background-size: 500% 200%; background-repeat: no-repeat; background-color: #f8fafc; }}
    .view-tile span {{ position: absolute; left: 6px; top: 6px; padding: 2px 6px; border-radius: 999px; background: rgba(15,23,42,0.72); color: white; font-size: 11px; letter-spacing: 0; }}
    .contact {{ width: 100%; border: 1px solid var(--line); background: #f8fafc; display: block; margin: 10px 0; }}
    details {{ margin-top: 6px; color: var(--muted); }}
    summary {{ cursor: pointer; user-select: none; }}
    .diff {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    .diff section {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fbfdff; }}
    .caption {{ font-weight: 650; line-height: 1.35; }}
    .attrs {{ display: flex; flex-wrap: wrap; gap: 6px; line-height: 1.35; margin-top: 8px; }}
    .chip {{ display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border-radius: 999px; border: 1px solid #cbd5e1; background: #f8fafc; color: #334155; font-size: 12px; }}
    .chip.added {{ border-color: #86efac; background: #dcfce7; color: #14532d; }}
    .chip.removed {{ border-color: #fecaca; background: #fee2e2; color: #7f1d1d; text-decoration: line-through; }}
    .chip.muted {{ color: var(--muted); }}
    .note {{ color: #334155; background: #f8fafc; border-radius: 6px; padding: 8px 10px; }}
    .hidden {{ display: none; }}
    a {{ color: var(--blue); text-decoration: none; }}
    .zoomable {{ cursor: zoom-in; }}
    .lightbox {{ position: fixed; inset: 0; z-index: 50; display: none; align-items: center; justify-content: center; background: rgba(15,23,42,0.86); padding: 24px; }}
    .lightbox.open {{ display: flex; }}
    .lightbox img {{ max-width: 96vw; max-height: 92vh; border: 1px solid #475569; background: white; }}
    @media (max-width: 1180px) {{ .review-layout {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 780px) {{ .views-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
<header>
  <h1>GeoScout Caption Judgement Review</h1>
  <div class="summary">
    <span>samples: {len(rows)}</span>
    <span>reviewed: {reviewed_count}</span>
    <span>changed: {changed_count}</span>
    <span>source: <code>{html.escape(str(args.jsonl))}</code></span>
  </div>
  <div class="controls">
    <input id="q" placeholder="search object_id or caption" size="34">
    <select id="status">
      <option value="all">all statuses</option>
      <option value="changed">changed only</option>
      <option value="reviewed">reviewed unchanged</option>
      <option value="pending">pending</option>
    </select>
  </div>
</header>
<main>
{''.join(cards)}
</main>
<div id="lightbox" class="lightbox" aria-hidden="true"><img id="lightbox-img" alt="expanded contact sheet"></div>
<script>
const q = document.getElementById('q');
const status = document.getElementById('status');
const cards = Array.from(document.querySelectorAll('.card'));
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightbox-img');
function applyFilter() {{
  const needle = q.value.toLowerCase();
  const st = status.value;
  for (const card of cards) {{
    const okStatus = st === 'all' || card.dataset.status === st;
    const okText = !needle || card.innerText.toLowerCase().includes(needle);
    card.classList.toggle('hidden', !(okStatus && okText));
  }}
}}
q.addEventListener('input', applyFilter);
status.addEventListener('change', applyFilter);
document.querySelectorAll('.zoomable').forEach((el) => {{
  el.addEventListener('click', (event) => {{
    event.preventDefault();
    const src = el.dataset.img || el.getAttribute('href') || el.getAttribute('src');
    if (!src) return;
    lightboxImg.src = src;
    lightbox.classList.add('open');
    lightbox.setAttribute('aria-hidden', 'false');
  }});
}});
lightbox.addEventListener('click', () => {{
  lightbox.classList.remove('open');
  lightbox.setAttribute('aria-hidden', 'true');
  lightboxImg.removeAttribute('src');
}});
document.addEventListener('keydown', (event) => {{
  if (event.key === 'Escape') {{
    lightbox.classList.remove('open');
    lightbox.setAttribute('aria-hidden', 'true');
    lightboxImg.removeAttribute('src');
  }}
}});
</script>
</body>
</html>
"""
    output_html.write_text(doc, encoding="utf-8")
    summary = {
        "schema_version": "geoscout_caption_judgement_html_summary_v1",
        "output_html": str(output_html),
        "num_records": len(rows),
        "num_reviewed": reviewed_count,
        "num_changed": changed_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_atlas = sub.add_parser("make-atlas")
    p_atlas.add_argument("--jsonl", required=True, type=Path)
    p_atlas.add_argument("--image-dir", required=True, type=Path)
    p_atlas.add_argument("--out-dir", required=True, type=Path)
    p_atlas.add_argument("--batch-size", type=int, default=6)
    p_atlas.set_defaults(func=make_review_atlas)

    p_snapshot = sub.add_parser("snapshot-originals")
    p_snapshot.add_argument("--jsonl", required=True, type=Path)
    p_snapshot.add_argument("--image-dir", type=Path, default=None)
    p_snapshot.add_argument("--out-dir", required=True, type=Path)
    p_snapshot.set_defaults(func=snapshot_originals)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--jsonl", required=True, type=Path)
    p_apply.add_argument("--judgements-json", required=True, type=Path)
    p_apply.add_argument("--output-jsonl", required=True, type=Path)
    p_apply.add_argument("--changed-jsonl", required=True, type=Path)
    p_apply.add_argument("--changed-csv", required=True, type=Path)
    p_apply.add_argument("--summary-json", required=True, type=Path)
    p_apply.set_defaults(func=apply_judgements)

    p_html = sub.add_parser("build-html")
    p_html.add_argument("--jsonl", required=True, type=Path)
    p_html.add_argument("--judgements-json", required=True, type=Path)
    p_html.add_argument("--image-dir", required=True, type=Path)
    p_html.add_argument("--output-html", required=True, type=Path)
    p_html.set_defaults(func=build_review_html)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
