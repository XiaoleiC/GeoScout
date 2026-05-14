#!/usr/bin/env python3
"""Build a compact PDF-ready LaTeX audit of text-encoder captions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def quality_summary(row: dict[str, Any]) -> str:
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    fields = [
        f"score={quality.get('quality_score_auto', '')}",
        f"manual={quality.get('needs_manual_review', '')}",
        f"attrs={quality.get('num_present_attributes', '')}",
    ]
    if row.get("manual_caption_override"):
        fields.append("override=yes")
    return "; ".join(fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--output-tex", required=True, type=Path)
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument("--verify-json", type=Path)
    parser.add_argument("--embedding-path-label", default="")
    parser.add_argument("--preproc-dir-label", default="")
    args = parser.parse_args()

    rows = read_jsonl(args.jsonl)
    validation = read_json(args.validation_json)
    verify = read_json(args.verify_json)

    lines = [
        r"\documentclass[9pt]{article}",
        r"\usepackage[margin=0.45in,landscape]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{xcolor}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{2pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\begin{document}",
        r"{\Large GeoScout 600 Caption Audit: Exact Text Encoder Inputs}\par",
        r"\vspace{4pt}",
        rf"\textbf{{Caption JSONL:}} \texttt{{{tex_escape(args.jsonl)}}}\par",
        (
            r"\textbf{Rule:} every row below is exactly "
            r"\texttt{caption.embedding\_caption}, the string encoded by "
            r"\texttt{scripts.precompute\_object\_caption\_embeddings.py}."
            r"\par"
        ),
    ]
    if args.embedding_path_label:
        lines.append(rf"\textbf{{Embedding payload:}} \texttt{{{tex_escape(args.embedding_path_label)}}}\par")
    if args.preproc_dir_label:
        lines.append(rf"\textbf{{Captioned preproc dir:}} \texttt{{{tex_escape(args.preproc_dir_label)}}}\par")
    if validation:
        lines.append(
            r"\textbf{Caption validation:} "
            + tex_escape(
                f"records={validation.get('num_records')}; "
                f"overrides={validation.get('num_overrides_applied')}; "
                f"manual_review={validation.get('needs_manual_review')}; "
                f"quality_hist={validation.get('quality_score_histogram')}; "
                f"appearance_flags={len(validation.get('appearance_word_object_ids', []))}"
            )
            + r"\par"
        )
    if verify:
        lines.append(
            r"\textbf{Embedding/preproc verification:} "
            + tex_escape(
                f"checked={verify.get('num_checked_with_caption_emb')}; "
                f"dim={verify.get('embedding_dim')}; "
                f"missing_files={verify.get('num_missing_files')}; "
                f"missing_caption_emb={verify.get('num_missing_caption_emb')}; "
                f"bad_dim={verify.get('num_bad_dim')}; "
                f"mismatched_emb={verify.get('num_mismatched_emb')}; "
                f"mismatched_text={verify.get('num_mismatched_text')}; "
                f"norm=[{verify.get('caption_emb_norm_min')}, "
                f"{verify.get('caption_emb_norm_mean')}, "
                f"{verify.get('caption_emb_norm_max')}]"
            )
            + r"\par"
        )
    lines.extend([
        r"\vspace{6pt}",
        r"\scriptsize",
        r"\begin{longtable}{r L{0.22\textwidth} L{0.06\textwidth} L{0.55\textwidth} L{0.11\textwidth}}",
        r"\hline",
        r"\textbf{\#} & \textbf{object\_id} & \textbf{cat} & \textbf{caption.embedding\_caption (exact)} & \textbf{audit} \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"\textbf{\#} & \textbf{object\_id} & \textbf{cat} & \textbf{caption.embedding\_caption (exact)} & \textbf{audit} \\",
        r"\hline",
        r"\endhead",
    ])

    for idx, row in enumerate(rows, start=1):
        caption = row.get("caption") if isinstance(row.get("caption"), dict) else {}
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        object_id = str(row.get("object_id", ""))
        category = str(caption.get("category") or source.get("category_hint") or "")
        text = str(caption.get("embedding_caption", ""))
        lines.append(
            f"{idx} & "
            rf"\texttt{{{tex_escape(object_id)}}} & "
            f"{tex_escape(category)} & "
            f"{tex_escape(text)} & "
            f"{tex_escape(quality_summary(row))} \\\\"
        )
        lines.append(r"\hline")

    lines.extend([r"\end{longtable}", r"\end{document}"])
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
