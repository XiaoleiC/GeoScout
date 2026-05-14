#!/usr/bin/env python3
"""Apply hand-reviewed geometry-caption overrides to a GeoScout JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
    "has_occluded_or_hidden_supports": "occluded or hidden supports",
}

APPEARANCE_TERMS = {
    "black", "white", "red", "blue", "green", "yellow", "brown", "gray",
    "grey", "orange", "purple", "pink", "metal", "metallic", "wood",
    "wooden", "plastic", "fabric", "leather", "glass", "texture",
    "textured", "color", "colour", "matte", "glossy",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def join_phrases(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def make_caption(category: str, attrs: dict[str, int], order: list[str]) -> str:
    phrases = [ATTRIBUTE_LABELS[k] for k in order if int(attrs.get(k, 0) or 0)]
    if phrases:
        return f"A {category} with {join_phrases(phrases)}."
    return f"A {category}."


def contains_appearance_term(text: str) -> bool:
    tokens = {
        tok.strip(".,;:!?()[]{}\"'").lower()
        for tok in text.replace("-", " ").split()
    }
    return bool(tokens & APPEARANCE_TERMS)


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    duplicates: list[str] = []
    missing_caption: list[str] = []
    appearance_flags: list[str] = []
    manual_review: list[str] = []
    quality_hist: dict[str, int] = {}
    for row in rows:
        object_id = str(row.get("object_id", ""))
        if object_id in seen:
            duplicates.append(object_id)
        seen.add(object_id)
        caption = row.get("caption") if isinstance(row.get("caption"), dict) else {}
        text = str(caption.get("embedding_caption", ""))
        if not text:
            missing_caption.append(object_id)
        if contains_appearance_term(text):
            appearance_flags.append(object_id)
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        if quality.get("needs_manual_review"):
            manual_review.append(object_id)
        score = str(quality.get("quality_score_auto", "missing"))
        quality_hist[score] = quality_hist.get(score, 0) + 1
    return {
        "schema_version": "geoscout_caption_override_validation_v1",
        "num_records": len(rows),
        "num_unique_object_ids": len(seen),
        "duplicate_object_ids": duplicates,
        "empty_embedding_caption_object_ids": missing_caption,
        "appearance_word_object_ids": appearance_flags,
        "needs_manual_review_object_ids": manual_review,
        "needs_manual_review": len(manual_review),
        "quality_score_histogram": quality_hist,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--overrides-json", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--changed-jsonl", type=Path, default=None)
    parser.add_argument("--validation-json", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    override_doc = json.loads(args.overrides_json.read_text(encoding="utf-8"))
    order = list(override_doc["attribute_order"])
    overrides = dict(override_doc["overrides"])

    rows_by_id = {str(row.get("object_id")): row for row in rows}
    missing_overrides = sorted(set(overrides) - set(rows_by_id))
    if missing_overrides:
        raise SystemExit(f"Override object IDs missing from input JSONL: {missing_overrides}")

    changed: list[dict[str, Any]] = []
    for row in rows:
        object_id = str(row.get("object_id"))
        override = overrides.get(object_id)
        if not override:
            continue

        caption = row.setdefault("caption", {})
        if not isinstance(caption, dict):
            raise SystemExit(f"{object_id}: caption field is not an object")
        category = str(caption.get("category") or row.get("source", {}).get("category_hint") or object_id.split("_")[0])

        present = set(override["present_attributes"])
        unknown_attrs = sorted(present - set(order))
        if unknown_attrs:
            raise SystemExit(f"{object_id}: unknown attributes in override: {unknown_attrs}")

        attributes = {key: int(key in present) for key in order}
        attrs = [attributes[key] for key in order]
        text = make_caption(category, attributes, order)

        previous_caption = caption.get("embedding_caption", "")
        caption["attribute_order"] = order
        caption["attributes"] = attributes
        caption["attrs"] = attrs
        caption["category"] = category
        caption["embedding_caption"] = text
        caption["final_caption"] = text
        caption["composer_version"] = "geom_attribute_composer_v2_attrs_only_manual_review_v1"
        caption["shape_tags"] = list(override.get("shape_tags", []))
        caption["priority_views"] = list(override.get("priority_views", []))
        caption["uncertainties"] = []

        quality = row.setdefault("quality", {})
        if not isinstance(quality, dict):
            raise SystemExit(f"{object_id}: quality field is not an object")
        quality.update({
            "caption_schema_kind": "attribute",
            "has_embedding_caption": True,
            "has_final_caption": True,
            "missing_required_fields": [],
            "needs_manual_review": False,
            "too_generic": False,
            "quality_score_auto": 3,
            "num_present_attributes": int(sum(attrs)),
            "embedding_caption_mentions_appearance_term": contains_appearance_term(text),
            "mentions_appearance_term": contains_appearance_term(text),
            "mentions_color": False,
            "mentions_material": False,
        })

        row["manual_caption_override"] = {
            "schema_version": override_doc["schema_version"],
            "previous_embedding_caption": previous_caption,
            "manual_note": override.get("manual_note", ""),
        }
        changed.append(row)

    write_jsonl(args.output_jsonl, rows)
    if args.changed_jsonl:
        write_jsonl(args.changed_jsonl, changed)
    validation = validate_rows(rows)
    validation.update({
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "overrides_json": str(args.overrides_json),
        "num_overrides_applied": len(changed),
        "override_object_ids": sorted(overrides),
    })
    if args.validation_json:
        args.validation_json.parent.mkdir(parents=True, exist_ok=True)
        args.validation_json.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
