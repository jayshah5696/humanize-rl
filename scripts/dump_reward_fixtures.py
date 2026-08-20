"""Dump canonical (task, response, RewardResult) cases as JSON parity fixtures.

The TypeScript port in `app/reward-lab/shared/reward/` must reproduce each
expected RewardResult byte-for-byte (within 1e-9 float tolerance).

Run:
    uv run scripts/dump_reward_fixtures.py
Output:
    app/reward-lab-tests/fixtures.json
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click

from humanize_rl.reward.reward import score_response
from humanize_rl.reward.tasks import RLTask


def make_task(**overrides: Any) -> RLTask:
    base: dict[str, Any] = {
        "id": "rl_v01_000001",
        "family": "rewrite_repair",
        "domain": "slack",
        "mode": "rewrite",
        "register": "casual",
        "instruction": "Clean up this Slack update. Return only the message.",
        "input_text": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
        "constraints": {
            "max_words": 20,
            "no_subject_line": True,
            "no_signoff": True,
            "preserve_numbers": True,
            "preserve_entities": True,
        },
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["wrapper_phrase"],
        "split": "train",
        "required_facts": ["Staging recovered", "3 pm", "STRIPE_WEBHOOK_SECRET"],
    }
    constraints = overrides.pop("constraints", None)
    base.update(overrides)
    if constraints is not None:
        merged = dict(base["constraints"])
        merged.update(constraints)
        base["constraints"] = merged
    return RLTask.model_validate(base)


# ---------------------------------------------------------------------------
# Fixture cases
# Cover every code path: passing case, every failing check, every profile,
# placeholder rules, markdown rules, length window, sentence window, refusal,
# AI-tell phrases, invented details, forbidden facts, corporate filler,
# em-dash heavy, list-overuse, hedging heavy, opener/closing AI tells.
# ---------------------------------------------------------------------------

CASES: list[dict[str, Any]] = [
    # ---- clean ----
    {
        "name": "clean_perfect_slack",
        "task": make_task(),
        "response": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
    },
    # ---- option menu + wrapper ----
    {
        "name": "option_menu_and_wrapper",
        "task": make_task(),
        "response": "Here's a cleaner version:\n\nOption 1: Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
    },
    # ---- refusal ----
    {
        "name": "refusal",
        "task": make_task(),
        "response": "I can't help with rewriting that message.",
    },
    # ---- subject line forbidden ----
    {
        "name": "subject_line_forbidden",
        "task": make_task(),
        "response": "Subject: Staging update\n\nStaging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
    },
    # ---- signoff forbidden ----
    {
        "name": "signoff_forbidden",
        "task": make_task(),
        "response": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.\n\nBest,",
    },
    # ---- placeholder when disallowed ----
    {
        "name": "placeholder_disallowed",
        "task": make_task(),
        "response": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix. [Add owner]",
    },
    # ---- missing number ----
    {
        "name": "missing_number",
        "task": make_task(),
        "response": "Staging recovered after the STRIPE_WEBHOOK_SECRET fix.",
    },
    # ---- missing entity ----
    {
        "name": "missing_entity",
        "task": make_task(),
        "response": "Staging recovered at 3 pm after the fix.",
    },
    # ---- invented name (Sarah not in context) ----
    {
        "name": "invented_name",
        "task": make_task(),
        "response": "Sarah fixed staging at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
    },
    # ---- forbidden phrase (AI tell) ----
    {
        "name": "ai_tell_phrase",
        "task": make_task(),
        "response": "Certainly, staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
    },
    # ---- markdown forbidden ----
    {
        "name": "markdown_list_forbidden",
        "task": make_task(),
        "response": "- Staging recovered at 3 pm\n- After the STRIPE_WEBHOOK_SECRET fix",
    },
    # ---- too long ----
    {
        "name": "too_long",
        "task": make_task(constraints={"max_words": 5}),
        "response": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix was deployed by the team.",
    },
    # ---- too short ----
    {
        "name": "too_short",
        "task": make_task(constraints={"min_words": 20}),
        "response": "Staging is back.",
    },
    # ---- exact sentence mismatch ----
    {
        "name": "exact_sentences_off",
        "task": make_task(constraints={"exact_sentences": 1}),
        "response": "Staging recovered. The STRIPE_WEBHOOK_SECRET fix worked. 3 pm.",
    },
    # ---- required fact missing ----
    {
        "name": "missing_required_fact",
        "task": make_task(required_facts=["3 pm", "STRIPE_WEBHOOK_SECRET", "rollback completed"]),
        "response": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
    },
    # ---- forbidden fact present ----
    {
        "name": "forbidden_fact_present",
        "task": make_task(forbidden_facts=["database migration"]),
        "response": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix and database migration.",
    },
    # ---- placeholder required ----
    {
        "name": "placeholder_required_present",
        "task": make_task(
            constraints={
                "allow_placeholders": True,
                "require_placeholders_for_missing_specifics": True,
            }
        ),
        "response": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix. [Owner: TBD]",
    },
    {
        "name": "placeholder_required_missing",
        "task": make_task(
            constraints={
                "allow_placeholders": True,
                "require_placeholders_for_missing_specifics": True,
            }
        ),
        "response": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
    },
    # ---- markdown allowed ----
    {
        "name": "markdown_allowed",
        "task": make_task(constraints={"allow_markdown": True}),
        "response": "- Staging recovered at 3 pm\n- STRIPE_WEBHOOK_SECRET fix shipped",
    },
    # ---- corporate filler ----
    {
        "name": "corporate_filler_heavy",
        "task": make_task(constraints={"max_words": 60}),
        "response": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix; we leveraged synergy to unlock seamless mission-critical robust delivery.",
    },
    # ---- em-dash heavy ----
    {
        "name": "em_dash_heavy",
        "task": make_task(constraints={"max_words": 60}),
        "response": "Staging recovered at 3 pm \u2014 after the STRIPE_WEBHOOK_SECRET fix \u2014 fully \u2014 green \u2014 again.",
    },
    # ---- hedging heavy ----
    {
        "name": "hedging_heavy",
        "task": make_task(constraints={"max_words": 80}),
        "response": "It is worth noting that staging recovered at 3 pm; it is important to note that the STRIPE_WEBHOOK_SECRET fix worked. It is also worth mentioning that one common pitfall was avoided.",
    },
    # ---- list overuse layer1 (markdown allowed so no penalty, just style hit) ----
    {
        "name": "list_overuse_layer1",
        "task": make_task(
            constraints={"allow_markdown": True, "max_words": 100},
            required_facts=[],
        ),
        "response": "- one\n- two\n- three\n- four\n- five\n- six",
    },
    # ---- closing AI tell ----
    {
        "name": "closing_ai_tell",
        "task": make_task(
            constraints={"max_words": 60, "no_signoff": False},
            required_facts=[],
        ),
        "response": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.\n\nLet me know if you have any questions.",
    },
    # ---- opener AI tell ----
    {
        "name": "opener_ai_tell",
        "task": make_task(
            constraints={"max_words": 60},
            required_facts=[],
        ),
        "response": "Great question! Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
    },
    # ---- empty response ----
    {
        "name": "empty_response",
        "task": make_task(),
        "response": "",
    },
    # ---- profile: direct_workplace_message ----
    {
        "name": "profile_direct_workplace_message",
        "task": make_task(
            reward_profile="direct_workplace_message",
            family="direct_email",
            domain="email",
            mode="direct_generation",
            input_text="",
            required_facts=[],
            constraints={"preserve_numbers": False, "preserve_entities": False},
        ),
        "response": "Heads up: deploy is paused while we investigate the timeout.",
    },
    # ---- profile: compression_update ----
    {
        "name": "profile_compression_update",
        "task": make_task(
            reward_profile="compression_update",
            mode="compress",
            constraints={"max_words": 15},
        ),
        "response": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
    },
    # ---- profile: technical_explain_natural ----
    {
        "name": "profile_technical_explain_natural",
        "task": make_task(
            reward_profile="technical_explain_natural",
            family="technical_explain",
            domain="technical",
            mode="direct_generation",
            input_text="",
            required_facts=[],
            constraints={"preserve_numbers": False, "preserve_entities": False, "max_words": 80},
        ),
        "response": "A webhook is a callback URL the server hits when an event happens. You receive a POST request with the event payload and reply 200 to acknowledge.",
    },
    # ---- profile: sensitive_comms ----
    {
        "name": "profile_sensitive_comms",
        "task": make_task(
            reward_profile="sensitive_comms",
            family="candidate_customer_comms",
            domain="hiring",
            mode="direct_generation",
            input_text="",
            required_facts=[],
            constraints={"preserve_numbers": False, "preserve_entities": False, "max_words": 80},
        ),
        "response": "Thanks for interviewing with us. We won't be moving forward, but your experience with payments came through clearly.",
    },
    # ---- penalty stacking ----
    {
        "name": "many_failures_stacked",
        "task": make_task(),
        "response": "Certainly! Subject: Update\n\nHere is option 1:\n- Sarah fixed staging\n- [Add details]\n\nBest,",
    },
    # ---- long passing response in larger window ----
    {
        "name": "long_clean_passing",
        "task": make_task(
            constraints={"max_words": 200},
            required_facts=[],
        ),
        "response": "Staging came back up at 3 pm. The STRIPE_WEBHOOK_SECRET rotation needed a redeploy and a fresh worker pool, and once both shipped the queue drained inside ten minutes. We confirmed inbound webhooks were being acknowledged correctly before handing it back to the on-call team.",
    },
]


def task_to_jsonable(task: RLTask) -> dict[str, Any]:
    return json.loads(task.model_dump_json(by_alias=True, exclude_none=False))


def result_to_jsonable(result: Any) -> dict[str, Any]:
    return {
        "reward": result.reward,
        "raw_reward": result.raw_reward,
        "profile": result.profile,
        "components": dict(result.components),
        "weighted_components": dict(result.weighted_components),
        "penalties": dict(result.penalties),
        "diagnostics": list(result.diagnostics),
    }


@click.command()
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("app/reward-lab-tests/fixtures.json"),
    show_default=True,
)
def main(out: Path) -> None:
    """Generate parity fixtures for the TypeScript reward port."""
    out.parent.mkdir(parents=True, exist_ok=True)

    fixtures: list[dict[str, Any]] = []
    for case in CASES:
        task: RLTask = case["task"]
        response: str = case["response"]
        result = score_response(task, response, track_a_scorer=None)
        fixtures.append(
            {
                "name": case["name"],
                "task": task_to_jsonable(task),
                "response": response,
                "expected": result_to_jsonable(result),
            }
        )

    out.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False) + "\n")
    click.echo(f"Wrote {len(fixtures)} fixtures to {out}")


if __name__ == "__main__":
    main()
