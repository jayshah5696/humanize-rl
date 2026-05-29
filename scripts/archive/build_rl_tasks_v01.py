from __future__ import annotations

import json
from pathlib import Path

import click

from humanize_rl.reward.tasks import (
    RLTask,
    TaskConstraints,
    summarize_tasks,
    write_tasks,
)

TEMPLATE_ROWS: list[dict[str, object]] = [
    {
        "family": "rewrite_repair",
        "domain": "slack",
        "mode": "rewrite",
        "register": "casual",
        "instruction": "Clean up this Slack update. Keep it casual and under 30 words. Return only the message.",
        "input_text": "Please be advised that staging has been restored. The root cause was a missing STRIPE_WEBHOOK_SECRET value, and we will monitor it until 3 pm.",
        "constraints": {
            "max_words": 30,
            "preserve_numbers": True,
            "preserve_entities": True,
        },
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["over_polish", "wrapper_phrase", "verbosity"],
        "required_facts": [
            "staging restored",
            "missing STRIPE_WEBHOOK_SECRET",
            "monitor until 3 pm",
        ],
        "forbidden_facts": ["database migration", "Sarah"],
    },
    {
        "family": "direct_email",
        "domain": "email",
        "mode": "direct_generation",
        "register": "warm_professional",
        "instruction": "Write a short email saying the invoice was paid today and the receipt is attached. No subject line or signoff.",
        "constraints": {"max_words": 55, "no_subject_line": True, "no_signoff": True},
        "reward_profile": "direct_workplace_message",
        "trap_tags": ["email_overformat", "invented_detail", "corporate_filler"],
        "required_facts": ["invoice paid today", "receipt attached"],
        "forbidden_facts": ["wire transfer", "NetSuite", "Sarah"],
    },
    {
        "family": "slack_chat",
        "domain": "slack",
        "mode": "direct_generation",
        "register": "terse",
        "instruction": "Write a Slack message asking the team to hold deploys until the cache issue is fixed. Under 25 words.",
        "constraints": {"max_words": 25, "no_subject_line": True, "no_signoff": True},
        "reward_profile": "direct_workplace_message",
        "trap_tags": ["register_mismatch", "verbosity", "email_overformat"],
        "required_facts": ["hold deploys", "cache issue fixed"],
    },
    {
        "family": "compression",
        "domain": "leadership",
        "mode": "compress",
        "register": "neutral",
        "instruction": "Compress this incident note into two sentences for leadership. Preserve the customer count and ETA.",
        "input_text": "Checkout errors started at 10:10. We traced it to the payment adapter timing out after a vendor deploy. About 180 customers saw retries. The rollback finished at 10:42, and we expect the remaining queue to clear by 11:15.",
        "constraints": {
            "max_words": 45,
            "exact_sentences": 2,
            "preserve_numbers": True,
        },
        "reward_profile": "compression_update",
        "trap_tags": ["dropped_facts", "length_failure", "vague_summary"],
        "required_facts": [
            "180 customers",
            "clear by 11:15",
            "rollback finished at 10:42",
        ],
    },
    {
        "family": "tone_shift",
        "domain": "email",
        "mode": "tone_shift",
        "register": "candid",
        "instruction": "Make this note more candid and less inflated. Keep the same facts. No subject line.",
        "input_text": "We are thrilled to announce a transformative optimization initiative that will unlock unprecedented operational excellence across onboarding by Friday.",
        "constraints": {"max_words": 35, "no_subject_line": True},
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["hype_retention", "over_polish", "ai_tell_phrase"],
        "forbidden_phrases": [
            "thrilled",
            "transformative",
            "unlock",
            "unprecedented",
            "operational excellence",
        ],
        "required_facts": ["onboarding", "Friday"],
    },
    {
        "family": "technical_explain",
        "domain": "technical",
        "mode": "direct_generation",
        "register": "technical",
        "instruction": "Explain to a junior engineer why idempotency keys matter for payment retries. Keep it under 80 words and avoid bullets.",
        "constraints": {"max_words": 80, "allow_markdown": False},
        "reward_profile": "technical_explain_natural",
        "trap_tags": ["over_markdown", "fake_technical_claim", "lecture_tone"],
        "required_facts": ["payment retries", "idempotency keys"],
    },
    {
        "family": "product_copy_cleanup",
        "domain": "product",
        "mode": "rewrite",
        "register": "neutral",
        "instruction": "Rewrite this product copy so it sounds concrete and user-facing. One sentence.",
        "input_text": "Our seamless AI-powered platform empowers teams to unlock mission-critical insights at scale.",
        "constraints": {"max_words": 24, "exact_sentences": 1},
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["buzzword_retention", "vague_benefit", "ai_tell_phrase"],
        "forbidden_phrases": [
            "seamless",
            "empowers",
            "unlock",
            "mission-critical",
            "at scale",
        ],
    },
    {
        "family": "candidate_customer_comms",
        "domain": "hiring",
        "mode": "direct_generation",
        "register": "warm_professional",
        "instruction": "Write a brief candidate rejection after a final interview. Be kind but not over-apologetic. Do not invent feedback.",
        "constraints": {"max_words": 70, "allow_placeholders": False},
        "reward_profile": "sensitive_comms",
        "trap_tags": ["over_apology", "invented_feedback", "corporate_filler"],
        "forbidden_facts": [
            "another candidate",
            "technical assessment",
            "future openings",
        ],
    },
    {
        "family": "adversarial_ai_tell_removal",
        "domain": "email",
        "mode": "repair",
        "register": "warm_professional",
        "instruction": "Remove the AI-sounding phrases while preserving the request and deadline. Return only the revised note.",
        "input_text": "I hope this email finds you well. It is worth noting that we need your budget notes by Thursday. Please do not hesitate to reach out with questions.",
        "constraints": {"max_words": 40, "preserve_entities": True},
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["ai_tell_phrase", "meaning_loss", "wrapper_phrase"],
        "forbidden_phrases": [
            "I hope this email finds you well",
            "It is worth noting",
            "Please do not hesitate",
        ],
        "required_facts": ["budget notes", "Thursday"],
    },
    {
        "family": "placeholder_discipline",
        "domain": "support",
        "mode": "direct_generation",
        "register": "neutral",
        "instruction": "Write a support reply saying we need the customer's account email before investigating. Use a placeholder only for the missing email.",
        "constraints": {
            "max_words": 45,
            "allow_placeholders": True,
            "require_placeholders_for_missing_specifics": True,
        },
        "reward_profile": "direct_workplace_message",
        "trap_tags": ["placeholder_required", "invented_detail", "over_polish"],
        "required_facts": ["need account email", "before investigating"],
    },
]

VARIANTS: list[dict[str, object]] = [
    {"register": "casual", "split": "train", "max_delta": 0},
    {"register": "neutral", "split": "train", "max_delta": 5},
    {"register": "warm_professional", "split": "train", "max_delta": 10},
    {"register": "candid", "split": "train", "max_delta": -3},
    {"register": "terse", "split": "train", "max_delta": -6},
    {"register": "technical", "split": "train", "max_delta": 8},
    {"register": "founder_like", "split": "train", "max_delta": 4},
    {"register": "neutral", "split": "train", "max_delta": 2},
    {"register": "warm_professional", "split": "validation", "max_delta": 0},
    {"register": "candid", "split": "test", "max_delta": 0},
]

FAMILY_REGISTER_OVERRIDES: dict[str, set[str]] = {
    "technical_explain": {"technical", "neutral", "casual"},
    "slack_chat": {"casual", "terse", "candid", "neutral"},
    "candidate_customer_comms": {"warm_professional", "neutral", "candid"},
}


def _constraints(base: dict[str, object], max_delta: int) -> TaskConstraints:
    constraints = dict(base)
    if "max_words" in constraints and isinstance(constraints["max_words"], int):
        constraints["max_words"] = max(5, constraints["max_words"] + max_delta)
    return TaskConstraints.model_validate(constraints)


def build_smoke_tasks() -> list[RLTask]:
    tasks: list[RLTask] = []
    counter = 1
    for template_index, template in enumerate(TEMPLATE_ROWS):
        family = str(template["family"])
        allowed_registers = FAMILY_REGISTER_OVERRIDES.get(family)
        for variant_index, variant in enumerate(VARIANTS):
            register = str(variant["register"])
            if allowed_registers is not None and register not in allowed_registers:
                register = sorted(allowed_registers)[
                    variant_index % len(allowed_registers)
                ]

            instruction = str(template["instruction"])
            if register != template["register"]:
                instruction = (
                    f"{instruction} Use a {register.replace('_', '-')} register."
                )

            source_group = f"template_{family}_{template_index:03d}"
            tasks.append(
                RLTask(
                    id=f"rl_v01_{counter:06d}",
                    family=template["family"],
                    domain=template["domain"],
                    mode=template["mode"],
                    register=register,
                    instruction=instruction,
                    input_text=str(template.get("input_text", "")),
                    constraints=_constraints(
                        dict(template["constraints"]), int(variant["max_delta"])
                    ),
                    reward_profile=template["reward_profile"],
                    trap_tags=list(template["trap_tags"]),
                    split=variant["split"],
                    forbidden_phrases=list(template.get("forbidden_phrases", [])),
                    required_facts=list(template.get("required_facts", [])),
                    forbidden_facts=list(template.get("forbidden_facts", [])),
                    source_group=source_group,
                )
            )
            counter += 1
    return tasks


@click.command()
@click.option(
    "--output-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("data/rl/humanize_tasks_v01_smoke.jsonl"),
    show_default=True,
)
@click.option(
    "--summary-path", type=click.Path(path_type=Path, dir_okay=False), default=None
)
def main(output_path: Path, summary_path: Path | None) -> None:
    """Build the v01 100-row RL smoke task dataset."""
    tasks = build_smoke_tasks()
    write_tasks(output_path, tasks)
    summary = summarize_tasks(tasks)

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    click.echo(f"rows={len(tasks)}")
    click.echo(f"output_path={output_path}")
    click.echo(f"summary={json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
