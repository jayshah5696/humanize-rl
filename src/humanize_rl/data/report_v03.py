"""V-Slice 4 -- full v03 report shape.

Generates the three benchmark splits the spec calls for, plus the
full report (per-domain, per-length-band, AUROC).

Inputs:
  - data/benchmark/v03_walking_skeleton.jsonl  (3-class matched, from V-Slice 0+)
  - data/benchmark/test_set_v02.jsonl          (legacy; AI class becomes OOD)
  - data/processed/v03_walking_skeleton_sft.jsonl  (accepted triples)

Outputs:
  - data/benchmark/v03_core.jsonl              (matched triples, 3 classes)
  - data/benchmark/v03_ood_ai.jsonl            (long-form / unmatched AI)
  - data/benchmark/v03_diagnostics.jsonl       (rejected/flagged triples)
  - runs/v03/v03_report.json                   (full numeric report)
  - runs/v03/v03_report.md                     (human-readable summary)

Usage:
    uv run python -m humanize_rl.data.report_v03
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from humanize_rl.scoring.aggregator import score_text


def _auroc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """AUROC where higher score = more 'positive class'. Mann-Whitney U."""
    if not positive_scores or not negative_scores:
        return float("nan")
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)
    rank_sum = 0.0
    paired = sorted(
        [(s, 1) for s in positive_scores] + [(s, 0) for s in negative_scores],
        key=lambda x: x[0],
    )
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if paired[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    u = rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def _length_band(words: int) -> str:
    if words < 100:
        return "short"
    if words < 220:
        return "medium"
    return "long"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _ensure_text(row: dict) -> str:
    text = row.get("text", "")
    if not text or row.get("text_is_preview"):
        return text or ""
    return text


def build_core_split(matched_path: Path, out_path: Path) -> list[dict]:
    """Re-emit the matched 3-class file with a stable v03_core schema."""
    rows = _load_jsonl(matched_path)
    out = []
    for row in rows:
        out.append(
            {
                "split": "v03_core",
                "id": row.get("id"),
                "label": row.get("label"),
                "domain": row.get("domain", "unknown"),
                "text": row.get("text", ""),
                "overall_score": row.get("overall_score"),
                "per_dim": row.get("per_dim", {}),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


def build_ood_ai_split(legacy_path: Path, out_path: Path) -> list[dict]:
    """Extract AI-class rows from the frozen test_set_v02 as the OOD split.

    Per spec section 1.2: long-form ChatGPT/Claude explainers should never
    be averaged with v03_core; they get their own report.
    """
    rows = _load_jsonl(legacy_path)
    out = []
    for row in rows:
        if row.get("label") != "ai":
            continue
        text = _ensure_text(row)
        out.append(
            {
                "split": "v03_ood_ai",
                "id": row.get("id"),
                "label": "ai",
                "domain": row.get("domain", "unknown"),
                "text": text,
                "overall_score": row.get("overall_score"),
                "per_dim": row.get("per_dim", {}),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


def build_diagnostics_split(
    matched_path: Path,
    sft_path: Path,
    out_path: Path,
    max_rows: int = 50,
) -> list[dict]:
    """Triples that did NOT pass the gate go here for failure analysis.

    Per spec section 1.3.
    """
    matched = _load_jsonl(matched_path)
    accepted_ids = {r.get("id") for r in _load_jsonl(sft_path)}

    diagnostics: list[dict] = []
    for row in matched:
        full_id = row.get("id", "")
        triple_id = full_id.rsplit("_", 1)[0] if "_" in full_id else full_id
        if triple_id in accepted_ids:
            continue
        diagnostics.append(
            {
                "split": "v03_diagnostics",
                "id": full_id,
                "triple_id": triple_id,
                "label": row.get("label"),
                "domain": row.get("domain", "unknown"),
                "text": row.get("text", ""),
                "overall_score": row.get("overall_score"),
            }
        )
        if len(diagnostics) >= max_rows:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in diagnostics:
            f.write(json.dumps(r) + "\n")
    return diagnostics


def _per_class_means(rows: list[dict]) -> dict[str, float]:
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = row.get("overall_score")
        if score is not None:
            by_label[row.get("label", "unknown")].append(score)
    return {label: mean(scores) for label, scores in by_label.items() if scores}


def _auroc_pairs(rows: list[dict]) -> dict[str, float]:
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = row.get("overall_score")
        if score is not None:
            by_label[row.get("label", "unknown")].append(score)
    out: dict[str, float] = {}
    pairs = (
        ("human", "ai", "human_vs_ai"),
        ("humanized", "ai", "humanized_vs_ai"),
        ("human", "humanized", "human_vs_humanized"),
    )
    for pos, neg, name in pairs:
        if by_label.get(pos) and by_label.get(neg):
            out[name] = _auroc(by_label[pos], by_label[neg])
    return out


def _per_domain_means(rows: list[dict]) -> dict[str, dict[str, float]]:
    by_domain: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        score = row.get("overall_score")
        if score is None:
            continue
        by_domain[row.get("domain", "unknown")][row.get("label", "unknown")].append(
            score
        )
    out: dict[str, dict[str, float]] = {}
    for domain, labels in by_domain.items():
        out[domain] = {label: mean(scores) for label, scores in labels.items()}
    return out


def _per_length_band(rows: list[dict]) -> dict[str, dict[str, float]]:
    by_band: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        text = row.get("text", "")
        score = row.get("overall_score")
        if score is None or not text:
            continue
        band = _length_band(len(text.split()))
        by_band[band][row.get("label", "unknown")].append(score)
    out: dict[str, dict[str, float]] = {}
    for band, labels in by_band.items():
        out[band] = {label: mean(scores) for label, scores in labels.items()}
    return out


def _per_dim_means(rows: list[dict]) -> dict[str, dict[str, float]]:
    by_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        per_dim = row.get("per_dim") or {}
        label = row.get("label", "unknown")
        for dim, val in per_dim.items():
            if isinstance(val, (int, float)):
                by_label[label][dim].append(val)
    out: dict[str, dict[str, float]] = {}
    for label, dims in by_label.items():
        out[label] = {dim: mean(scores) for dim, scores in dims.items()}
    return out


def _ood_summary(ood_rows: list[dict]) -> dict[str, object]:
    """Mean score + below-threshold rates. Lower scores = AI-like (good)."""
    scores = [
        r.get("overall_score") for r in ood_rows if r.get("overall_score") is not None
    ]
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "mean_score": mean(scores),
        "frac_below_0.5": sum(1 for s in scores if s < 0.5) / len(scores),
        "frac_below_0.7": sum(1 for s in scores if s < 0.7) / len(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V-Slice 4 -- full v03 report.")
    parser.add_argument(
        "--matched",
        type=Path,
        default=Path("data/benchmark/v03_walking_skeleton.jsonl"),
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=Path("data/benchmark/test_set_v02.jsonl"),
    )
    parser.add_argument(
        "--sft",
        type=Path,
        default=Path("data/processed/v03_walking_skeleton_sft.jsonl"),
    )
    parser.add_argument(
        "--core-out",
        type=Path,
        default=Path("data/benchmark/v03_core.jsonl"),
    )
    parser.add_argument(
        "--ood-out",
        type=Path,
        default=Path("data/benchmark/v03_ood_ai.jsonl"),
    )
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=Path("data/benchmark/v03_diagnostics.jsonl"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("runs/v03/v03_report.json"),
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("runs/v03/v03_report.md"),
    )
    args = parser.parse_args()

    print("Building splits ...")
    core = build_core_split(args.matched, args.core_out)
    ood = build_ood_ai_split(args.legacy, args.ood_out)
    diag = build_diagnostics_split(args.matched, args.sft, args.diagnostics_out)
    print(
        f"  core: {len(core)} rows -> {args.core_out}\n"
        f"  ood:  {len(ood)} rows -> {args.ood_out}\n"
        f"  diag: {len(diag)} rows -> {args.diagnostics_out}"
    )

    # OOD legacy file may store previews; re-score whatever text is present.
    for row in ood:
        if row.get("overall_score") is None and row.get("text"):
            row["overall_score"] = score_text(row["text"]).overall

    report: dict[str, object] = {
        "n_core": len(core),
        "n_ood_ai": len(ood),
        "n_diagnostics": len(diag),
        "core": {
            "per_class_mean": _per_class_means(core),
            "auroc": _auroc_pairs(core),
            "per_domain": _per_domain_means(core),
            "per_length_band": _per_length_band(core),
            "per_dim": _per_dim_means(core),
        },
        "ood_ai": _ood_summary(ood),
        "manual_review_queue": [
            {
                "id": d["id"],
                "triple_id": d.get("triple_id"),
                "label": d.get("label"),
                "domain": d.get("domain"),
                "score": d.get("overall_score"),
            }
            for d in diag[:10]
        ],
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport JSON: {args.report_json}")

    lines: list[str] = []
    lines.append("# v03 walking-skeleton report")
    lines.append("")
    lines.append(f"- core rows: **{len(core)}**")
    lines.append(f"- ood-ai rows: **{len(ood)}**")
    lines.append(f"- diagnostics rows: **{len(diag)}**")
    lines.append("")
    lines.append("## Per-class mean (core)")
    core_summary = report["core"]
    assert isinstance(core_summary, dict)
    for label, score in sorted(core_summary["per_class_mean"].items()):
        lines.append(f"- **{label}**: {score:.3f}")
    lines.append("")
    lines.append("## AUROC (core)")
    for name, score in core_summary["auroc"].items():
        lines.append(f"- **{name}**: {score:.3f}")
    lines.append("")
    lines.append("## Per-domain mean score (core)")
    for domain, labels in sorted(core_summary["per_domain"].items()):
        lines.append(f"- **{domain}**")
        for label, score in sorted(labels.items()):
            lines.append(f"  - {label}: {score:.3f}")
    lines.append("")
    lines.append("## Per-length-band (core)")
    for band, labels in sorted(core_summary["per_length_band"].items()):
        lines.append(f"- **{band}**")
        for label, score in sorted(labels.items()):
            lines.append(f"  - {label}: {score:.3f}")
    lines.append("")
    lines.append("## OOD AI summary (legacy long-form, separate)")
    ood_summary = report["ood_ai"]
    assert isinstance(ood_summary, dict)
    for k, v in ood_summary.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.3f}")
        else:
            lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Manual review queue (top 10 failed triples)")
    review_queue = report["manual_review_queue"]
    assert isinstance(review_queue, list)
    for entry in review_queue:
        lines.append(
            f"- `{entry['id']}` ({entry.get('domain', 'unknown')}, "
            f"label={entry.get('label')}, score={entry.get('score')})"
        )

    args.report_md.write_text("\n".join(lines))
    print(f"Report MD:   {args.report_md}")


if __name__ == "__main__":
    main()
