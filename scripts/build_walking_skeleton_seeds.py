"""Materialize the walking-skeleton seed pool.

V-Slice 0: instruction_technical only (10 seeds).
V-Slice 1: + email/professional (10 seeds, total 20).
V-Slice 1.5 (next): replace `curated_paste` source with real HF loaders.

Output: seeds/v03/walking_skeleton.jsonl (one row per seed, arka-ready).

Schema is enforced by `humanize_rl.data.seed.Seed`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from humanize_rl.data.seed import Seed, to_arka_seed_row

OUT_PATH = Path("seeds/v03/walking_skeleton.jsonl")


# ---------------------------------------------------------------------------
# Domain: instruction_technical (V-Slice 0)
# ---------------------------------------------------------------------------

INSTRUCTION_TECHNICAL_SEEDS: list[dict[str, object]] = [
    {
        "id": "v03_ws_inst_000",
        "discourse_role": "troubleshooting",
        "text": (
            "Spent half a day chasing a pytest failure that only showed up in CI. "
            "Locally everything passed, in GitHub Actions one parametrize case blew up "
            "with a UnicodeDecodeError on a fixture file. Turned out the runner image "
            "had LANG=C while my Mac defaults to en_US.UTF-8, and our YAML loader was "
            "implicitly using the locale encoding. The fix was three characters: pass "
            "encoding='utf-8' to open(). I added an LC_ALL=C.UTF-8 export to the "
            "workflow as well so this stops biting other tests later."
        ),
    },
    {
        "id": "v03_ws_inst_001",
        "discourse_role": "instructional_explanation",
        "text": (
            "If you want to pin a Python version with uv, don't put the version in "
            "requirements.txt — uv ignores that. Use .python-version at the repo root, "
            "or set requires-python in pyproject.toml. uv will then download a matching "
            "interpreter on first sync. The thing that confused me at first: uv's "
            "managed interpreters live under ~/.local/share/uv/python and won't show up "
            "in `pyenv versions`. That's expected, they're a separate world."
        ),
    },
    {
        "id": "v03_ws_inst_002",
        "discourse_role": "troubleshooting",
        "text": (
            "Our Postgres connection pool started saturating at 80% CPU around lunch "
            "every day. pg_stat_activity showed dozens of idle-in-transaction "
            "sessions, all from the same worker. The bug was a missing finally-block "
            "in a celery task that opened a transaction, raised on a rate-limit error, "
            "and never ran ROLLBACK. autocommit=False + no context manager + a flaky "
            "external API == slow leak. We wrapped the whole task body in a "
            "with conn.transaction(): and the leak vanished within an hour."
        ),
    },
    {
        "id": "v03_ws_inst_003",
        "discourse_role": "instructional_explanation",
        "text": (
            "A trick I keep forgetting and re-learning: `git log -S'foo'` finds commits "
            "that added or removed the literal string 'foo', not commits that just "
            "mention it. So if you're hunting down when a feature flag was first "
            "introduced, -S beats --grep every time. Pair it with --source --all if "
            "the change might live on a long-dead branch you forgot about. I found a "
            "two-year-old bug origin in about 30 seconds this way last week."
        ),
    },
    {
        "id": "v03_ws_inst_004",
        "discourse_role": "troubleshooting",
        "text": (
            "The Docker image was 1.4 GB and I couldn't figure out why. dive helped: "
            "an apt-get install in layer 3 brought in build-essential, and a separate "
            "pip install --no-cache-dir in layer 6 reinstalled gcc anyway because of a "
            "C extension. Splitting the Dockerfile into a builder stage and a slim "
            "runtime stage dropped it to 240 MB. The builder runs apt-get install -y "
            "build-essential, the runtime stage only copies the wheelhouse over."
        ),
    },
    {
        "id": "v03_ws_inst_005",
        "discourse_role": "instructional_explanation",
        "text": (
            "When you write a custom pytest fixture that yields, remember the cleanup "
            "code after `yield` runs even if the test failed. That's usually what you "
            "want, but if your fixture starts a Docker container and the test asserts "
            "on a log line, you'll lose the container before you can `docker logs` it. "
            "Two options: set --pdb so you can poke around at failure time, or have the "
            "fixture write the container id to a tmp_path file so you can grab it after "
            "the run."
        ),
    },
    {
        "id": "v03_ws_inst_006",
        "discourse_role": "troubleshooting",
        "text": (
            "Caught a fun race in our Redis-backed lock. We used SET key value NX PX "
            "30000 to grab the lock, but the unlock path just did DEL key. If the "
            "30-second TTL expired before the worker finished, another worker grabbed "
            "the lock and then the first worker's DEL nuked the second worker's lock. "
            "Switched to the standard Lua-script compare-and-delete (only DEL if value "
            "matches) and the duplicate-processing alert went silent the same day."
        ),
    },
    {
        "id": "v03_ws_inst_007",
        "discourse_role": "instructional_explanation",
        "text": (
            "If you're doing structured JSON output with the OpenAI SDK, prefer "
            "`response_format={'type':'json_schema', 'json_schema': {...}}` over "
            "`json_object`. The schema mode is enforced server-side and you stop having "
            "to wrap every call in pydantic.ValidationError handling. Keep your schema "
            "shallow — deep nested anyOf will silently degrade to slower sampling. We "
            "saw a 2x speedup just by flattening one optional sub-object into a string."
        ),
    },
    {
        "id": "v03_ws_inst_008",
        "discourse_role": "troubleshooting",
        "text": (
            "macOS Sequoia broke our pre-commit hook. The hook called `realpath` and "
            "Sequoia's bundled BSD realpath behaves differently from the GNU one when "
            "given a non-existent path: BSD prints an error and exits 1, GNU prints "
            "the canonicalized path and exits 0. Two engineers on Linux had no idea "
            "the hook was failing on Mac. Fix was to install coreutils via brew and "
            "call grealpath, with a small wrapper that falls back to realpath on Linux."
        ),
    },
    {
        "id": "v03_ws_inst_009",
        "discourse_role": "instructional_explanation",
        "text": (
            "Quick note on numpy broadcasting that bit me: a (1000,) array minus a "
            "(1000, 1) array gives you a (1000, 1000) result, not a (1000,) result. "
            "Numpy is doing what you asked, you just didn't realize you were asking. "
            "Reshape one of them with [:, None] or [None, :] depending on which axis "
            "you actually meant, or use np.subtract.outer if outer subtraction is what "
            "you wanted all along. I lost an afternoon to this once on a notebook that "
            "silently allocated 8 GB."
        ),
    },
]


# ---------------------------------------------------------------------------
# Domain: email / professional (V-Slice 1)
# ---------------------------------------------------------------------------
#
# Hand-paste in the spirit of Enron internal emails: real blocker, named
# person/team, next step, one practical judgment call. Defers HF `datasets`
# dep to V-Slice 1.5.

EMAIL_PROFESSIONAL_SEEDS: list[dict[str, object]] = [
    {
        "id": "v03_ws_email_000",
        "discourse_role": "status_update",
        "text": (
            "Quick update on the Q3 migration. We hit a snag with the Datadog\n"
            "ingestion limits on Tuesday — about 14% of host metrics were\n"
            "silently dropped between 11:30 and 14:00. Sarah from infra has a\n"
            "ticket open with their support but their ETA is ~5 business days.\n"
            "I'm going to backfill from Prometheus for the affected window;\n"
            "it'll take half a day. Will not block the migration unless\n"
            "Datadog comes back with worse news. Talk Friday?"
        ),
    },
    {
        "id": "v03_ws_email_001",
        "discourse_role": "request",
        "text": (
            "Mike — can you take a look at the auth-service rollback policy\n"
            "before EOD? Brian's team wants to ship the new SSO flow Tuesday\n"
            "and I'm not comfortable with the current 'manual rollback only'\n"
            "stance given how much the migration touches session storage.\n"
            "Specifically I want a documented automated rollback path for the\n"
            "Redis schema change. Five minutes on a call works too if easier."
        ),
    },
    {
        "id": "v03_ws_email_002",
        "discourse_role": "decision_rationale",
        "text": (
            "Re: vendor choice — I'm going to push for Snowflake over\n"
            "Databricks for the warehouse rebuild, even though Databricks\n"
            "scored 4 points higher on our matrix. Two reasons: (1) Snowflake's\n"
            "cost is more predictable for our usage shape, which matters for\n"
            "Finance's planning cycle; (2) we already burned six weeks last\n"
            "year on a Databricks PoC and the team has scar tissue. Happy to\n"
            "be argued out of it before the steering meeting."
        ),
    },
    {
        "id": "v03_ws_email_003",
        "discourse_role": "status_update",
        "text": (
            "Hey team — heads up that the Stripe webhook outage from this\n"
            "morning is fully resolved as of 10:42 ET. Root cause was on their\n"
            "side, not ours; ~120 events queued and replayed cleanly. No\n"
            "customer-visible impact, no charges affected. The retry handler\n"
            "we shipped in February did exactly what it was supposed to. I\n"
            "filed a postmortem stub in Notion if anyone wants to add notes."
        ),
    },
    {
        "id": "v03_ws_email_004",
        "discourse_role": "request",
        "text": (
            "Hi Priya — could you forward me the Q2 vendor invoices for\n"
            "Snowflake and Datadog? I'm pulling together the cloud-spend\n"
            "review for next Wednesday and the Finance export only goes back\n"
            "to April. Just the PDFs is fine, I don't need them reconciled.\n"
            "If easier, share the SharePoint folder. Thanks!"
        ),
    },
    {
        "id": "v03_ws_email_005",
        "discourse_role": "decision_rationale",
        "text": (
            "On the contractor question: I'd rather hold off until we close\n"
            "the platform team req. We'd be paying $185/hr to fix problems\n"
            "that an FTE would prevent in the first place, and the runway on\n"
            "the staffing side is only six weeks. If the Tuesday standup\n"
            "shows we're still blocked on the auth migration by mid-month\n"
            "I'll change my mind, but right now contractor velocity won't\n"
            "actually move the date."
        ),
    },
    {
        "id": "v03_ws_email_006",
        "discourse_role": "status_update",
        "text": (
            "FYI — I moved the Thursday architecture review to next week.\n"
            "Two of the three reviewers are out (Lina is at re:Invent, Tom's\n"
            "on PTO), and the design doc is still missing the storage cost\n"
            "section. Better to slip a week than do a half-baked review. New\n"
            "date is Wed Apr 30 at 14:00 ET, calendar updated."
        ),
    },
    {
        "id": "v03_ws_email_007",
        "discourse_role": "request",
        "text": (
            "Quick ask — I need someone from the data team to pair with me\n"
            "for ~2 hours on Friday afternoon to validate the new event\n"
            "schema before we ship. Specifically I want a second pair of\n"
            "eyes on the user_id vs. account_id distinction and how it\n"
            "interacts with the existing GDPR delete pipeline. Reply if you\n"
            "have bandwidth, otherwise I'll grab Anjali."
        ),
    },
    {
        "id": "v03_ws_email_008",
        "discourse_role": "status_update",
        "text": (
            "Late update on the perf regression — turns out it wasn't the new\n"
            "ranking model after all. The p95 latency jump on Tuesday\n"
            "correlates with a Cloudflare config change that landed at the\n"
            "same time. Rolled the ranking change forward this morning, p95\n"
            "is back to 180ms. Apologies for the false alarm earlier in the\n"
            "thread; I should have checked the CDN dashboard sooner."
        ),
    },
    {
        "id": "v03_ws_email_009",
        "discourse_role": "decision_rationale",
        "text": (
            "On promoting Rachel to staff — I'm a strong yes. Three concrete\n"
            "examples from this quarter: (1) she ran the auth migration\n"
            "end-to-end with zero rollbacks despite the SSO complications;\n"
            "(2) she rewrote the on-call runbook and we've seen a measurable\n"
            "drop in escalations to me; (3) she's been mentoring two\n"
            "engineers who are visibly leveling up. The only gap I see is\n"
            "cross-org influence outside engineering, which is normal at\n"
            "this stage. I'll write the formal packet this week."
        ),
    },
]


SEED_BUNDLES: list[tuple[str, list[dict[str, object]]]] = [
    ("instruction_technical", INSTRUCTION_TECHNICAL_SEEDS),
    ("email", EMAIL_PROFESSIONAL_SEEDS),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+\b")
_ANCHOR_RE = re.compile(
    r"""
    \b\d+(?:\.\d+)+\b               # versions: 3.12, 1.2.3
    | \b\d{4}\b                     # years
    | \b[A-Z]{2,}(?:_[A-Z0-9]+)+\b  # SHOUTY_SNAKE constants
    | \b[A-Z]{2,}\b                 # acronyms (ETL, GDPR, SSO)
    | `[^`]+`                       # backtick code
    | [\w./-]+\.\w{1,5}\b           # filenames / paths
    | \b[a-z]+(?:[A-Z][a-z]+)+\b    # camelCase
    | \b\d+\s*(?:ms|s|MB|GB|KB|%|hr)\b  # metrics
    | \$[\d,]+(?:\.\d+)?            # dollar amounts
    """,
    re.VERBOSE,
)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _length_band(words: int) -> str:
    if words < 100:
        return "short"
    if words < 220:
        return "medium"
    return "long"


def _anchor_count(text: str) -> int:
    return len(_ANCHOR_RE.findall(text))


def _instruction_for_domain(domain: str) -> str:
    return {
        "instruction_technical": "Share a short technical note from your own experience.",
        "email": "Write a short professional email from one teammate to another.",
    }.get(domain, f"Write in the style of: {domain}.")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seeds: list[Seed] = []
    for domain, raw_seeds in SEED_BUNDLES:
        for raw in raw_seeds:
            text = str(raw["text"]).strip()
            wc = _word_count(text)
            seed = Seed(
                id=str(raw["id"]),
                text=text,
                domain=domain,  # type: ignore[arg-type]
                discourse_role=raw["discourse_role"],  # type: ignore[arg-type]
                source_dataset="curated_paste",
                length_band=_length_band(wc),  # type: ignore[arg-type]
                word_count=wc,
                anchors_count=_anchor_count(text),
                instruction=_instruction_for_domain(domain),
            )
            seeds.append(seed)

    with OUT_PATH.open("w") as f:
        for seed in seeds:
            f.write(json.dumps(to_arka_seed_row(seed)) + "\n")

    # Per-domain summary so we see the shape we're feeding downstream.
    print(f"Wrote {len(seeds)} seeds to {OUT_PATH}")
    by_domain: dict[str, list[Seed]] = {}
    for s in seeds:
        by_domain.setdefault(s.domain, []).append(s)
    for domain, dseeds in by_domain.items():
        wcs = sorted(s.word_count for s in dseeds)
        anchors = [s.anchors_count for s in dseeds]
        print(
            f"  {domain:<24s} n={len(dseeds):2d}  "
            f"words: min={wcs[0]:3d} median={wcs[len(wcs) // 2]:3d} max={wcs[-1]:3d}  "
            f"anchors: min={min(anchors)} max={max(anchors)}"
        )


if __name__ == "__main__":
    main()
