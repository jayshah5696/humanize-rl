"""Dump score_response fixtures WITH the ridge scorer loaded.

The TS scoreWithSettings(RIDGE_ENABLED_SETTINGS) must match these within
the float16-quantization tolerance (1e-3 on final reward).

Reuses the same 32 (task, response) cases as the deterministic fixture set,
but scores with track_a_scorer=RidgeScorerAdapter(ridge_pkl). The fixture
records only what we need for parity (reward, raw_reward, components, ridge
dims) — full diagnostics are already validated by the deterministic suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

# Import after path setup so we can run from repo root via `uv run`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from humanize_rl.reward.reward import RIDGE_RUBRIC_DIMS, load_ridge_scorer, score_response
from dump_reward_fixtures import CASES, result_to_jsonable, task_to_jsonable  # noqa: E402


@click.command()
@click.option(
    "--ridge-pkl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("models/track_a_10k/ridge.pkl"),
    show_default=True,
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("app/reward-lab-tests/ridge_reward_fixtures.json"),
    show_default=True,
)
def main(ridge_pkl: Path, out: Path) -> None:
    scorer = load_ridge_scorer(ridge_pkl)
    assert scorer is not None, f"Failed to load ridge scorer from {ridge_pkl}"

    fixtures: list[dict[str, object]] = []
    for case in CASES:
        task = case["task"]
        response = case["response"]
        result = score_response(task, response, track_a_scorer=scorer)
        # Pull the 8 rubric dims separately so the TS test can compare
        # the ridge-sub-score map directly.
        # Always call the scorer — empty string returns intercepts, which is
        # exactly what the TS port also returns (sparse vector is empty, so
        # the dot product is the intercept alone).
        rubric = scorer.predict_rubric([response])[0]
        # RidgeScorerAdapter.predict_proba returns [[p_ai, p_human]] per row;
        # take index 1 to get P(human).
        p_human = float(scorer.predict_proba([response])[0][1])
        fixtures.append(
            {
                "name": case["name"],
                "task": task_to_jsonable(task),
                "response": response,
                "expected": result_to_jsonable(result),
                "ridge_p_human": p_human,
                "ridge_rubric": {
                    name: float(v) for name, v in zip(RIDGE_RUBRIC_DIMS, rubric)
                },
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False) + "\n")
    click.echo(f"Wrote {len(fixtures)} ridge-enabled fixtures to {out}")


if __name__ == "__main__":
    main()
