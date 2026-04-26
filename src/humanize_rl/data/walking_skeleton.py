"""V-Slice 0 walking-skeleton driver.

End-to-end on the 10-seed instruction_technical pool:
1. load AIify + humanize outputs (must already exist on disk)
2. score original / aiified / humanized with Layer 1 (free, deterministic)
3. apply v03 pair gate (V-Slice 0 subset)
4. emit a tiny benchmark + sft pairs file
5. print a per-class report so we can see the pipeline shape

Layer 2 is intentionally skipped for V-Slice 0 — keeps the slice free and
the failure surface small. V-Slice 2 will fold L2 back in.

Usage:
    uv run python -m humanize_rl.data.walking_skeleton
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from humanize_rl.data.pair_gate_v03 import GateResult, evaluate_triple
from humanize_rl.pipeline import build_triples, score_triples
from humanize_rl.scoring.aggregator import HumannessResult


def _load_seed_domain_map(seed_path: Path) -> dict[str, str]:
    """Build {original_text -> domain} from the seed JSONL.

    arka only carries `original_text` through to triples; this lets the
    runner re-attach the domain tag for per-domain reporting.
    """
    mapping: dict[str, str] = {}
    if not seed_path.exists():
        return mapping
    with seed_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("response", "").strip()
            domain = row.get("domain", "unknown")
            if text:
                mapping[text] = domain
    return mapping


def _format_per_dim(label: str, result: HumannessResult) -> str:
    dims = ", ".join(f"{k}={v:.2f}" for k, v in sorted(result.per_dim.items()))
    return f"  {label:>10s} overall={result.overall:.3f} | {dims}"


def _gate_to_dict(gate: GateResult) -> dict[str, object]:
    return {
        "accepted": gate.accepted,
        "rejected_reasons": list(gate.rejected_reasons),
        "suspicion_flags": list(gate.suspicion_flags),
        "metrics": {k: round(v, 4) for k, v in gate.metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V-Slice 0 walking-skeleton runner")
    parser.add_argument(
        "--aiify-output",
        type=Path,
        default=Path("output/v03/ws-aiify.jsonl"),
    )
    parser.add_argument(
        "--humanize-output",
        type=Path,
        default=Path("output/v03/ws-humanize.jsonl"),
    )
    parser.add_argument(
        "--benchmark-out",
        type=Path,
        default=Path("data/benchmark/v03_walking_skeleton.jsonl"),
    )
    parser.add_argument(
        "--sft-out",
        type=Path,
        default=Path("data/processed/v03_walking_skeleton_sft.jsonl"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("runs/v03/walking_skeleton_report.json"),
    )
    parser.add_argument(
        "--seeds",
        type=Path,
        default=Path("seeds/v03/walking_skeleton.jsonl"),
        help="Used to recover the domain tag for per-domain reporting.",
    )
    args = parser.parse_args()

    seed_domain_map = _load_seed_domain_map(args.seeds)

    if not args.aiify_output.exists() or not args.humanize_output.exists():
        print("Missing input. Run AIify and humanize for V-Slice 0 first:")
        print("  just v03-ws-aiify")
        print("  just v03-ws-humanize")
        print(f"Looked for:\n  {args.aiify_output}\n  {args.humanize_output}")
        return

    print("Loading triples from:")
    print(f"  AIify:    {args.aiify_output}")
    print(f"  Humanize: {args.humanize_output}")
    triples = build_triples(args.aiify_output, args.humanize_output)
    print(f"  → {len(triples)} triples\n")

    if not triples:
        print("No triples joined. Check that arka outputs preserved 'system' field.")
        return

    print("Scoring with Layer 1 (deterministic) ...")
    scored = score_triples(triples)

    triple_domains: dict[str, str] = {
        s.id: seed_domain_map.get(s.original_text.strip(), "unknown") for s in scored
    }

    # --- gate ---------------------------------------------------------------
    gate_results: list[GateResult] = []
    for s in scored:
        gate_results.append(
            evaluate_triple(
                original_text=s.original_text,
                aiified_text=s.aiified_text,
                humanized_text=s.humanized_text,
                original_score=s.original_score.overall,
                aiified_score=s.aiified_score.overall,
                humanized_score=s.humanized_score.overall,
                aiified_per_dim=s.aiified_score.per_dim,
                humanized_per_dim=s.humanized_score.per_dim,
            )
        )

    # --- per-row dump ------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("PER-TRIPLE BREAKDOWN")
    print("=" * 72)
    for s, g in zip(scored, gate_results, strict=True):
        verdict = "ACCEPT" if g.accepted else "REJECT"
        flags = f" flags={list(g.suspicion_flags)}" if g.suspicion_flags else ""
        domain = triple_domains.get(s.id, "unknown")
        print(f"\n{s.id}  [{domain}]  [{verdict}]{flags}")
        print(_format_per_dim("original", s.original_score))
        print(_format_per_dim("aiified", s.aiified_score))
        print(_format_per_dim("humanized", s.humanized_score))
        print(
            f"  Δ aiify={g.metrics['aiify_delta']:+.3f}  "
            f"Δ humanize={g.metrics['humanize_delta']:+.3f}  "
            f"recovery={g.metrics['recovery_ratio']:+.2f}  "
            f"len(ai/orig)={g.metrics['length_ratio_aiify_over_original']:.2f}  "
            f"len(hum/orig)={g.metrics['length_ratio_humanized_over_original']:.2f}"
        )
        if not g.accepted:
            print(f"  rejected_reasons: {list(g.rejected_reasons)}")

    # --- aggregate report --------------------------------------------------
    n = len(scored)
    n_accept = sum(g.accepted for g in gate_results)
    mean_orig = sum(s.original_score.overall for s in scored) / n
    mean_ai = sum(s.aiified_score.overall for s in scored) / n
    mean_hum = sum(s.humanized_score.overall for s in scored) / n
    mean_aiify_delta = sum(g.metrics["aiify_delta"] for g in gate_results) / n
    mean_hum_delta = sum(g.metrics["humanize_delta"] for g in gate_results) / n

    print(f"\n{'=' * 72}")
    print("AGGREGATE")
    print("=" * 72)
    print(f"  triples:                {n}")
    print(f"  accepted by gate:       {n_accept}/{n} ({n_accept / n * 100:.0f}%)")
    print(f"  mean original score:    {mean_orig:.3f}")
    print(f"  mean aiified score:     {mean_ai:.3f}")
    print(f"  mean humanized score:   {mean_hum:.3f}")
    print(f"  mean aiify delta:       {mean_aiify_delta:+.3f}  (target ≥ +0.25)")
    print(f"  mean humanize delta:    {mean_hum_delta:+.3f}  (target ≥ +0.30)")

    # Per-domain rollup.
    by_domain: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(scored):
        by_domain[triple_domains.get(s.id, "unknown")].append(i)
    if len(by_domain) > 1 or "unknown" not in by_domain:
        print(f"\n{'-' * 72}")
        print("PER-DOMAIN")
        print("-" * 72)
        print(
            f"  {'domain':<24s}  {'n':>3s}  {'acc':>5s}  "
            f"{'orig':>6s}  {'aiify':>6s}  {'human':>6s}  "
            f"{'Δaiify':>7s}  {'Δhuman':>7s}"
        )
        for domain, indices in sorted(by_domain.items()):
            dn = len(indices)
            d_acc = sum(gate_results[i].accepted for i in indices)
            d_orig = sum(scored[i].original_score.overall for i in indices) / dn
            d_ai = sum(scored[i].aiified_score.overall for i in indices) / dn
            d_hum = sum(scored[i].humanized_score.overall for i in indices) / dn
            d_da = sum(gate_results[i].metrics["aiify_delta"] for i in indices) / dn
            d_dh = sum(gate_results[i].metrics["humanize_delta"] for i in indices) / dn
            print(
                f"  {domain:<24s}  {dn:>3d}  {d_acc:>2d}/{dn:<2d}  "
                f"{d_orig:>6.3f}  {d_ai:>6.3f}  {d_hum:>6.3f}  "
                f"{d_da:>+7.3f}  {d_dh:>+7.3f}"
            )

    # --- exports -----------------------------------------------------------
    args.benchmark_out.parent.mkdir(parents=True, exist_ok=True)
    args.sft_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    # 30-row 3-class benchmark
    with args.benchmark_out.open("w") as f:
        for s in scored:
            for label, text, result in (
                ("human", s.original_text, s.original_score),
                ("ai", s.aiified_text, s.aiified_score),
                ("humanized", s.humanized_text, s.humanized_score),
            ):
                f.write(
                    json.dumps(
                        {
                            "id": f"{s.id}_{label}",
                            "label": label,
                            "domain": triple_domains.get(s.id, "unknown"),
                            "split": "v03_walking_skeleton",
                            "overall_score": round(result.overall, 4),
                            "per_dim": {
                                k: round(v, 4) for k, v in result.per_dim.items()
                            },
                            "text": text,
                        }
                    )
                    + "\n"
                )

    # SFT pairs from accepted triples only
    n_sft = 0
    with args.sft_out.open("w") as f:
        for s, g in zip(scored, gate_results, strict=True):
            if not g.accepted:
                continue
            f.write(
                json.dumps(
                    {
                        "id": s.id,
                        "domain": triple_domains.get(s.id, "unknown"),
                        "input": s.aiified_text,
                        "output": s.humanized_text,
                        "humanize_delta": round(g.metrics["humanize_delta"], 4),
                        "aiify_delta": round(g.metrics["aiify_delta"], 4),
                        "suspicion_flags": list(g.suspicion_flags),
                    }
                )
                + "\n"
            )
            n_sft += 1

    # JSON report
    report = {
        "n_triples": n,
        "n_accepted": n_accept,
        "n_sft_exported": n_sft,
        "mean_scores": {
            "original": mean_orig,
            "aiified": mean_ai,
            "humanized": mean_hum,
        },
        "mean_deltas": {
            "aiify": mean_aiify_delta,
            "humanize": mean_hum_delta,
        },
        "per_triple": [
            {
                "id": s.id,
                "scores": {
                    "original": s.original_score.overall,
                    "aiified": s.aiified_score.overall,
                    "humanized": s.humanized_score.overall,
                },
                "gate": _gate_to_dict(g),
            }
            for s, g in zip(scored, gate_results, strict=True)
        ],
    }
    args.report_out.write_text(json.dumps(report, indent=2))

    print("\nWrote:")
    print(f"  benchmark: {args.benchmark_out}  ({n * 3} rows)")
    print(f"  sft pairs: {args.sft_out}  ({n_sft} rows)")
    print(f"  report:    {args.report_out}")


if __name__ == "__main__":
    main()
