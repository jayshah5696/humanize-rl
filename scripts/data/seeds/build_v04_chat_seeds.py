import json, random
from pathlib import Path

# All Slack/chat focused, no AI/humanize framing, concrete situations
DIRECT_SLACK = [
    # Status updates
    ("Write a Slack message to the team saying the deploy went out and everything looks stable.", "direct_generation"),
    ("Write a Slack update telling the team the staging bug is fixed and to re-test.", "direct_generation"),
    ("Write a Slack message saying the on-call handoff is done and who's next.", "direct_generation"),
    ("Write a Slack message to the team asking everyone to update their JIRA tickets before standup.", "direct_generation"),
    ("Write a quick Slack message confirming the client call is at 3pm today.", "direct_generation"),
    ("Write a Slack message to the design channel saying the latest mockups are ready for review.", "direct_generation"),
    ("Write a Slack message saying the API keys have been rotated and asking engineers to update their envs.", "direct_generation"),
    ("Write a Slack message to the team saying you're taking a half day on Friday.", "direct_generation"),
    ("Write a Slack message telling the support team that the billing bug is now fixed in prod.", "direct_generation"),
    ("Write a Slack message asking the backend team when the new endpoint will be ready.", "direct_generation"),
    ("Write a Slack message telling the team the sprint review is moved to Thursday at 2pm.", "direct_generation"),
    ("Write a Slack message to #general saying the office will be closed on Friday.", "direct_generation"),
    ("Write a Slack message to the team celebrating hitting 1000 users.", "direct_generation"),
    ("Write a Slack message asking if anyone is free to pair on a tricky bug this afternoon.", "direct_generation"),
    ("Write a Slack message telling the sales team the new pricing page is live.", "direct_generation"),
    ("Write a Slack message saying you're heading out for a doctor's appointment and will be back in 2 hours.", "direct_generation"),
    ("Write a Slack message to the engineering team saying the database migration finished cleanly.", "direct_generation"),
    ("Write a Slack message asking the team to avoid merging to main until the flaky tests are fixed.", "direct_generation"),
    ("Write a quick DM to your manager saying you'll be late to standup by 10 minutes.", "direct_generation"),
    ("Write a Slack message to the product channel sharing a link to the new roadmap doc.", "direct_generation"),
    ("Write a Slack message asking the data team if the pipeline ran successfully last night.", "direct_generation"),
    ("Write a Slack message to a colleague asking if they reviewed your PR yet.", "direct_generation"),
    ("Write a Slack message to the team apologizing for the false alarm alert that fired overnight.", "direct_generation"),
    ("Write a Slack message saying you're blocked on a ticket and need someone to unblock you.", "direct_generation"),
    ("Write a Slack message to the team saying lunch is on the company today in the main conference room.", "direct_generation"),
]

REWRITE_SLACK = [
    # All have stiff source text to rewrite, zero AI/ChatGPT framing
    ("Rewrite this Slack message to be shorter and more direct:\n\n'I wanted to reach out and let everyone know that we have successfully completed the deployment and it is now live in the production environment without any issues.'", "rewrite_humanize"),
    ("Rewrite this Slack message to be casual and direct:\n\n'This is a notification to inform team members that the weekly standup meeting scheduled for tomorrow morning has been postponed to the following Thursday.'", "rewrite_humanize"),
    ("Rewrite this message for Slack so it sounds natural coming from a colleague:\n\n'I am writing to inform you that I will be unavailable for a period of approximately two hours this afternoon due to a personal engagement.'", "rewrite_humanize"),
    ("Tighten this Slack message. It is too wordy:\n\n'I just wanted to drop a quick note to let everyone know that the pull request that was submitted earlier today has now been merged into the main branch successfully.'", "rewrite_humanize"),
    ("Fix this Slack message so it does not sound stiff:\n\n'It has come to my attention that the API rate limits have been exceeded. The team should refrain from making requests until the issue is resolved.'", "rewrite_humanize"),
    ("Rewrite this channel update to sound like a real person:\n\n'Please be advised that maintenance will be performed on the database server this evening between 11pm and 1am. Access may be intermittently unavailable.'", "rewrite_humanize"),
    ("Rewrite this Slack message so it gets to the point faster:\n\n'I wanted to follow up on the conversation we had last week regarding the integration spec. I was hoping to get your feedback on the proposed approach before I move forward.'", "rewrite_humanize"),
    ("Rewrite this as a normal Slack message from a dev:\n\n'I am pleased to report that the performance optimization work has been completed and the response times are now within acceptable parameters.'", "rewrite_humanize"),
    ("Rewrite this message to be warmer and less corporate:\n\n'I would like to request your assistance with the documentation for the new onboarding flow at your earliest convenience.'", "rewrite_humanize"),
    ("Make this Slack message shorter and punchier:\n\n'I wanted to give everyone an update to the effect that the meeting that was originally scheduled for 2pm has been moved and will now take place at 4pm this afternoon instead.'", "rewrite_humanize"),
    ("Rewrite this status update for Slack so it sounds human:\n\n'Work on the feature branch is progressing as expected. The implementation is approximately 70% complete. The remaining work will be completed by end of day tomorrow.'", "rewrite_humanize"),
    ("Rewrite this message to be direct and casual for a team channel:\n\n'I would like to bring to your attention that the error rate on the checkout flow has increased by 12% over the past 24 hours. Investigation is ongoing.'", "rewrite_humanize"),
    ("Tighten this Slack message to under 2 sentences:\n\n'I just wanted to reach out to see if you have had a chance to look at the document I shared with you last Monday. I am hoping to get your thoughts on it before the end of the week.'", "rewrite_humanize"),
    ("Rewrite this Slack ping to sound casual and direct:\n\n'I am writing to inquire as to whether you are available to participate in a brief call this afternoon to discuss the status of the project.'", "rewrite_humanize"),
    ("Rewrite this message so it does not sound like a formal memo:\n\n'Effective immediately, all engineers are required to ensure their local development environments are running the latest version of the software prior to commencing work.'", "rewrite_humanize"),
]

# Tone/edit tasks on Slack content
EDIT_SLACK = [
    ("Cut this Slack message to one sentence:\n\n'Hey everyone, I just wanted to give a quick update to let you know that I have finished reviewing all of the pull requests that were assigned to me and they are all approved and ready to merge.'", "rewrite_humanize"),
    ("Make this Slack message warmer without losing the information:\n\n'The sprint is complete. Velocity was 42 points. Three tickets moved to next sprint.'", "rewrite_humanize"),
    ("Fix the grammar in this Slack message without making it sound formal:\n\n'hey we needs to talk about the pipeline, it been failing since last night and i dont know what happening'", "rewrite_humanize"),
    ("Rewrite this Slack message to be more direct without sounding harsh:\n\n'If it is not too much trouble, could you perhaps find some time this week to review the spec document when you get a chance?'", "rewrite_humanize"),
    ("Make this Slack message sound less passive:\n\n'It might be worth considering whether it would make sense to potentially look at refactoring the authentication module at some point in the future.'", "rewrite_humanize"),
    ("Rewrite this Slack message to sound like a normal status update:\n\n'I would like to take this opportunity to inform relevant stakeholders that the feature scheduled for release in this sprint has encountered a minor setback.'", "rewrite_humanize"),
    ("Rewrite this thread reply to be concise and direct:\n\n'That is a really interesting point that you have raised there. I think that you are absolutely right and I would be inclined to agree with your perspective on this matter.'", "rewrite_humanize"),
    ("Rewrite this Slack message to remove the hedging:\n\n'I was kind of thinking that maybe we could possibly consider looking into whether it might make sense to try a different approach to this problem.'", "rewrite_humanize"),
]

def build() -> list[dict]:
    rng = random.Random(17)
    rows = []
    all_seeds = (
        [(i, t, "slack_chat", "chat", m) for i, (t, m) in enumerate(DIRECT_SLACK)]
        + [(i, t, "slack_chat", "chat", m) for i, (t, m) in enumerate(REWRITE_SLACK)]
        + [(i, t, "grammar_clarity", "chat", m) for i, (t, m) in enumerate(EDIT_SLACK)]
    )
    for i, text, task_type, domain, mode in all_seeds:
        rows.append({
            "instruction": text,
            "task_type": task_type,
            "persona": None,
            "domain": domain,
            "mode": mode,
            "instruction_framing": "robotic_stiff" if "rewrite" in text.lower()[:20] or "Fix" in text or "Make" in text else "plain_task",
            "response": "",
        })
    return rows


if __name__ == "__main__":
    rows = build()
    out = Path("seeds/v04_chat_seeds.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"Wrote {len(rows)} seeds → {out}")
