import argparse
import json
import random
from pathlib import Path

SLACK_TOPICS = [
    "the deployment is delayed", "staging is broken", "API latency is up", "the design links are missing",
    "the billing PRD needs review", "the weekly sync moved", "we need an incident follow-up", "there is a customer escalation",
    "the test suite is flaky", "the release notes need review", "we need a handoff before PTO", "there is a database migration update",
]
EMAIL_TOPICS = [
    "a client integration bug", "a candidate rejection", "a PTO request", "a vendor discount request",
    "a subscription renewal", "a project delay", "a proposal follow-up", "an API access request",
    "a customer billing issue", "a meeting recap", "a launch update", "a security notice",
]
REWRITE_DRAFTS = [
    "I am writing to inform you that {topic}. Please be advised that further updates will be provided in due course.",
    "It is worth noting that {topic}. Furthermore, all relevant stakeholders should remain aligned on next steps.",
    "Please be advised that {topic}. Your prompt attention to this matter would be greatly appreciated.",
    "The purpose of this message is to provide visibility into the fact that {topic}. We will continue monitoring the situation.",
]
PERSONAS = [
    "staff engineer", "product manager", "support lead", "office manager", "recruiter",
    "solutions engineer", "team lead", "customer success manager", "developer advocate", "ops manager",
]


def add(rows, instruction, task_type, domain, mode, persona=None, framing="plain_task"):
    rows.append({
        "instruction": instruction,
        "task_type": task_type,
        "persona": persona,
        "domain": domain,
        "mode": mode,
        "instruction_framing": framing,
        "response": "",
    })


def build(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        kind = len(rows) % 6
        if kind == 0:
            topic = rng.choice(SLACK_TOPICS)
            add(rows, f"Draft a short Slack update for the team saying the {topic}. Keep it natural and direct.", "slack_chat", "chat", "direct_generation", rng.choice(PERSONAS))
        elif kind == 1:
            topic = rng.choice(EMAIL_TOPICS)
            add(rows, f"Write a concise professional email about this situation: {topic}. Use placeholders for missing names or dates.", "email_rewrite", "email", "direct_generation", rng.choice(PERSONAS))
        elif kind == 2:
            topic = rng.choice(SLACK_TOPICS)
            draft = rng.choice(REWRITE_DRAFTS).format(topic=topic)
            add(rows, f"Rewrite this Slack message to be casual and direct:\n\n'{draft}'", "slack_chat", "chat", "rewrite_humanize")
        elif kind == 3:
            topic = rng.choice(EMAIL_TOPICS)
            draft = rng.choice(REWRITE_DRAFTS).format(topic=topic)
            add(rows, f"Rewrite this email to sound like a normal professional person:\n\n'{draft}'", "email_rewrite", "email", "rewrite_humanize")
        elif kind == 4:
            topic = rng.choice(SLACK_TOPICS + EMAIL_TOPICS)
            add(rows, f"Tighten this update to one short paragraph without losing the main point:\n\n'We wanted to provide an update regarding {topic}. There are a few details to consider, and we will provide more information as soon as possible.'", "shorten_compress", "general", "rewrite_humanize")
        else:
            topic = rng.choice(SLACK_TOPICS + EMAIL_TOPICS)
            add(rows, f"Fix the grammar and clarity in this message while keeping it casual:\n\n'hey just checking about {topic}, do we has an update or should i wait till later'", "grammar_clarity", "general", "rewrite_humanize")
    return rows[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--output", default="seeds/v04_synthetic_base500.jsonl")
    args = parser.parse_args()
    rows = build(args.n)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
