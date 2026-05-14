#!/usr/bin/env python3
"""Build a PDF-ready LaTeX review file for GeoScout attribute captions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def tex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def tex_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("_", r"\_")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_quality(quality: dict) -> str:
    parts = []
    for key in [
        "quality_score_auto",
        "needs_manual_review",
        "json_parse_ok",
        "has_embedding_caption",
        "embedding_caption_mentions_appearance_term",
        "mentions_appearance_term",
        "category_mismatch",
        "too_generic",
        "num_present_attributes",
    ]:
        if key in quality:
            parts.append(f"{key}={quality[key]}")
    missing = quality.get("missing_required_fields")
    if missing:
        parts.append(f"missing={missing}")
    return "; ".join(parts)


def present_attributes(caption: dict) -> list[str]:
    attrs = caption.get("attributes")
    order = caption.get("attribute_order") or list(ATTRIBUTE_LABELS)
    if not isinstance(attrs, dict):
        return []
    labels = []
    for key in order:
        if int(attrs.get(key, 0) or 0):
            labels.append(ATTRIBUTE_LABELS.get(key, key))
    return labels


def format_list(values: object, fallback: str = "None listed.") -> str:
    if not isinstance(values, list) or not values:
        return fallback
    return ", ".join(str(v) for v in values)


def build_tex(
    rows: list[dict],
    image_dir: Path,
    output_tex: Path,
    jsonl_label: str,
    validation: dict | None,
) -> None:
    image_rel_dir = image_dir.relative_to(output_tex.parent)
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.55in,landscape]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{xcolor}",
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\usepackage{enumitem}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{2pt}",
        r"\setlist[itemize]{leftmargin=*,nosep}",
        r"\newcommand{\field}[2]{\textbf{#1:} #2\par}",
        r"\newcommand{\okflag}{\textcolor{green!45!black}{ok}}",
        r"\newcommand{\badflag}{\textcolor{red!70!black}{flagged}}",
        r"\begin{document}",
        r"\begin{center}",
        rf"{{\LARGE GeoScout {len(rows)}-Sample Attribute-Caption Review}}\par",
        r"\vspace{4pt}",
        (
            r"{\small Qwen2.5-VL-7B-Instruct, A100-80GB batch 64, "
            r"10-view 384px contact sheets, deterministic decoding, "
            r"max\_new\_tokens=256, attrs-only caption composer}\par"
        ),
        r"\end{center}",
        r"\vspace{4pt}",
        rf"\field{{Source JSONL}}{{\texttt{{{tex_escape(jsonl_label)}}}}}",
        (
            r"\field{Training Caption Rule}{The field named "
            r"\texttt{Text Encoder Input (exact)} on each sample page is the "
            r"exact string intended for the future text encoder. It is composed "
            r"only from category plus the fixed 20 binary geometry attributes. "
            r"VLM shape tags and priority views are shown for review, but are "
            r"not fed to the text encoder.}"
        ),
    ]
    if validation:
        elapsed = validation.get("elapsed_s")
        elapsed_text = f"{elapsed:.2f}" if isinstance(elapsed, (int, float)) else "n/a"
        summary = (
            f"{validation.get('num_records')}/{validation.get('num_manifest')} records; "
            f"parse_failures={validation.get('json_parse_failures')}; "
            f"manual_review={validation.get('needs_manual_review')}; "
            f"quality_hist={validation.get('quality_score_histogram')}; "
            f"elapsed_s={elapsed_text}"
        )
        lines.append(rf"\field{{Validation Summary}}{{{tex_escape(summary)}}}")
    lines.extend(
        [
            r"\vspace{6pt}",
            (
                r"\textbf{How to read each page.} The image is the contact sheet "
                r"seen by the VLM. The embedding caption is the exact text intended "
                r"for the future text encoder. The remaining fields are diagnostic."
            ),
            r"\newpage",
        ]
    )

    for idx, row in enumerate(rows, start=1):
        object_id = str(row.get("object_id", "unknown"))
        caption = row.get("caption", {}) if isinstance(row.get("caption"), dict) else {}
        quality = row.get("quality", {}) if isinstance(row.get("quality"), dict) else {}
        source = row.get("source", {}) if isinstance(row.get("source"), dict) else {}
        image_path = image_rel_dir / f"{object_id}.png"
        present = present_attributes(caption)
        lines.extend(
            [
                rf"\section*{{{idx:02d}. \texttt{{{tex_escape(object_id)}}}}}",
                rf"\includegraphics[width=\linewidth,height=0.50\textheight,keepaspectratio]{{{tex_path(image_path)}}}",
                r"\vspace{2pt}",
                r"{\small",
                rf"\field{{Category}}{{source={tex_escape(source.get('category'))}; vlm={tex_escape(caption.get('category'))}}}",
                rf"\field{{Text Encoder Input (exact)}}{{\textit{{{tex_escape(caption.get('embedding_caption', ''))}}}}}",
                rf"\field{{Quality}}{{{tex_escape(summarize_quality(quality))}}}",
                rf"\field{{JSON Field}}{{\texttt{{caption.embedding\_caption}}; \texttt{{caption.final\_caption}} is the same compatibility alias.}}",
                rf"\field{{Present Attributes}}{{{tex_escape(format_list(present))}}}",
                rf"\field{{Raw Attrs}}{{{tex_escape(caption.get('attrs', []))}}}",
                rf"\field{{VLM Shape Tags (review only)}}{{{tex_escape(format_list(caption.get('shape_tags')))}}}",
                rf"\field{{VLM Priority Views (review only)}}{{{tex_escape(format_list(caption.get('priority_views')))}}}",
                rf"\field{{Uncertainties}}{{{tex_escape(format_list(caption.get('uncertainties')))}}}",
                r"}",
                r"\newpage",
            ]
        )
    lines.append(r"\end{document}")
    output_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-tex", required=True, type=Path)
    parser.add_argument("--validation-json", type=Path)
    args = parser.parse_args()

    validation = None
    if args.validation_json and args.validation_json.exists():
        validation = json.loads(args.validation_json.read_text(encoding="utf-8"))
    rows = read_jsonl(args.jsonl)
    build_tex(
        rows=rows,
        image_dir=args.image_dir,
        output_tex=args.output_tex,
        jsonl_label=str(args.jsonl),
        validation=validation,
    )


if __name__ == "__main__":
    main()
