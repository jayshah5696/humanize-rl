from __future__ import annotations

from humanize_rl.reward.checks import run_deterministic_checks
from humanize_rl.reward.tasks import RLTask


def _task(**overrides: object) -> RLTask:
    base: dict[str, object] = {
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
            "allow_placeholders": False,
        },
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["wrapper_phrase", "verbosity"],
        "split": "train",
        "required_facts": ["Staging recovered", "3 pm", "STRIPE_WEBHOOK_SECRET"],
        "forbidden_facts": ["database migration"],
    }
    base.update(overrides)
    return RLTask.model_validate(base)


def _by_name(response: str, task: RLTask | None = None):
    return run_deterministic_checks(task or _task(), response).by_name()


def _words(count: int) -> str:
    return " ".join(f"word{i}" for i in range(count))


def _target_task(**constraint_overrides: object) -> RLTask:
    constraints: dict[str, object] = {
        "target_words": 100,
        "target_tolerance": 0.20,
        "no_subject_line": True,
        "no_signoff": True,
    }
    constraints.update(constraint_overrides)
    return _task(
        id="rl_v03_000001",
        family="compression",
        domain="creative_general",
        mode="compression",
        register="direct",
        input_text="Source text to compress.",
        constraints=constraints,
        reward_profile="compression_update",
        required_facts=[],
        forbidden_facts=[],
    )


def _romance_recommendation_task(**constraint_overrides: object) -> RLTask:
    constraints: dict[str, object] = {
        "target_words": 29,
        "target_tolerance": 0.15,
        "no_subject_line": True,
        "no_signoff": True,
        "must_not_include_phrases": [
            "in conclusion",
            "furthermore",
            "as a matter of fact",
        ],
        "forbidden_openers": ["Dear followers,", "Greetings,"],
        "must_not_use_em_dash": True,
        "min_contraction_count": 2,
    }
    constraints.update(constraint_overrides)
    return _task(
        id="rl_v03_000392",
        family="tone_shift",
        domain="creative_general",
        mode="tone_shift",
        register="warm",
        instruction=(
            "[ROLE] You are a social media copywriter. [TASK] Rewrite the "
            "source as a post in a new register: warm. [AUDIENCE] Followers "
            "on social media who enjoy uplifting romance films. [MUST] "
            "Include a quotation from a famous romantic movie and add a "
            "postscript with a personal recommendation of one romance film "
            "with an uplifting theme. [MUST_NOT] Do not use an em dash."
        ),
        input_text=(
            "Please draft a social media post encouraging my followers to "
            "watch romance films with uplifting themes. Include a quotation "
            "from a famous romantic movie and make sure to add a postscript "
            "with a personal recommendation of one such film."
        ),
        constraints=constraints,
        reward_profile="rewrite_faithful_concise",
        required_facts=[],
        forbidden_facts=[],
    )


def test_known_good_response_passes_hard_checks() -> None:
    diagnostics = run_deterministic_checks(
        _task(), "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix."
    )

    assert diagnostics.total_penalty == 0
    assert diagnostics.failed == []


def test_option_menu_and_wrapper_trigger_penalties() -> None:
    checks = _by_name(
        "Here's a cleaner version:\n\nOption 1: Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix."
    )

    assert not checks["option_menu"].passed
    assert checks["option_menu"].penalty == -0.40
    assert not checks["wrapper_phrase"].passed
    assert checks["wrapper_phrase"].penalty == -0.20


def test_invented_name_placeholder_subject_and_signoff_trigger() -> None:
    checks = _by_name(
        "Subject: Update\n\nSarah fixed staging at 3 pm after the STRIPE_WEBHOOK_SECRET fix. [Add details]\n\nBest,"
    )

    assert not checks["invented_detail"].passed
    assert "Sarah" in checks["invented_detail"].matches
    assert not checks["placeholder_disallowed"].passed
    assert not checks["subject_line"].passed
    assert not checks["signoff"].passed


def test_length_numbers_entities_and_ai_tells_trigger() -> None:
    response = (
        "It is worth noting that staging recovered after the fix, and furthermore "
        "we will keep carefully monitoring this throughout the afternoon with a detailed "
        "cross-functional operational excellence review for everyone involved."
    )
    checks = _by_name(response)

    assert not checks["too_long"].passed
    assert not checks["missing_number"].passed
    assert "3 pm" in checks["missing_number"].matches
    assert not checks["missing_entity"].passed
    assert "STRIPE_WEBHOOK_SECRET" in checks["missing_entity"].matches
    assert not checks["ai_tell_phrase"].passed


def test_required_placeholder_when_missing_specifics() -> None:
    task = _task(
        id="rl_v01_000002",
        family="placeholder_discipline",
        domain="support",
        mode="direct_generation",
        input_text="",
        constraints={
            "max_words": 30,
            "allow_placeholders": True,
            "require_placeholders_for_missing_specifics": True,
        },
        reward_profile="direct_workplace_message",
        required_facts=[],
    )

    checks = _by_name("Send us the account email and we'll investigate.", task)

    assert not checks["placeholder_required"].passed


def test_forbidden_fact_and_markdown_trigger() -> None:
    checks = _by_name(
        "- Staging recovered at 3 pm after a database migration and the STRIPE_WEBHOOK_SECRET fix."
    )

    assert not checks["forbidden_fact"].passed
    assert not checks["wrong_format_markdown"].passed


def test_title_like_subject_and_inline_signoff_trigger() -> None:
    checks = _by_name(
        "Urgent: Quality Assurance Review and Action Items.\n\n"
        "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix. "
        "Best regards, Patricia Adams."
    )

    assert not checks["subject_line"].passed
    assert (
        "Urgent: Quality Assurance Review and Action Items."
        in checks["subject_line"].matches
    )
    assert not checks["signoff"].passed
    assert "Best regards, Patricia Adams." in checks["signoff"].matches


def test_formal_salutation_and_inline_signature_with_title_trigger() -> None:
    checks = _by_name(
        "Dear William Brown, I am writing to request a meeting. "
        "Best regards, Patricia Adams, Engineer, Cloud Nine Systems."
    )

    assert not checks["salutation"].passed
    assert "Dear William Brown," in checks["salutation"].matches
    assert not checks["signoff"].passed
    assert (
        "Best regards, Patricia Adams, Engineer, Cloud Nine Systems."
        in checks["signoff"].matches
    )


def test_negated_greeting_instruction_does_not_allow_salutation() -> None:
    task = _task(
        id="rl_v03_000022",
        family="rewrite_repair",
        domain="email",
        mode="rewrite_humanize",
        register="direct",
        instruction=(
            "Rewrite this as a direct note. MUST_NOT: Use any greeting or "
            "sign-off formalities such as Dear or Best regards."
        ),
        input_text="Tell Charles staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
        constraints={
            "target_words": 20,
            "target_tolerance": 0.50,
            "no_subject_line": True,
            "no_signoff": True,
        },
        required_facts=["staging recovered", "3 pm", "STRIPE_WEBHOOK_SECRET"],
    )
    checks = _by_name(
        "Dear Charles, staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
        task,
    )

    assert not checks["salutation"].passed
    assert "Dear Charles," in checks["salutation"].matches


def test_multiline_best_regards_signature_triggers() -> None:
    checks = _by_name(
        "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.\n\n"
        "Best regards,\n"
        "Donna Taylor\n"
        "DataStream Analytics, Account Manager"
    )

    assert not checks["signoff"].passed
    assert "Best regards," in checks["signoff"].matches


def test_inline_canned_closing_triggers_signoff_and_ai_tell() -> None:
    checks = _by_name(
        "Production is delayed by two hours because of a validation bug. No testing yet. Thanks for listening."
    )

    assert not checks["signoff"].passed
    assert "Thanks for listening." in checks["signoff"].matches
    assert not checks["ai_tell_phrase"].passed
    assert "thanks for listening" in checks["ai_tell_phrase"].matches


def test_inline_thank_you_closing_triggers_signoff() -> None:
    checks = _by_name(
        "Prepare for the DataStream Analytics audit and schedule the "
        "meeting next week. Thank you for your feedback and collaboration."
    )

    assert not checks["signoff"].passed
    assert (
        "Thank you for your feedback and collaboration."
        in checks["signoff"].matches
    )


def test_instruction_leak_triggers_for_direct_rewrite_tasks() -> None:
    task = _task(
        id="rl_v03_000009",
        family="compression",
        domain="creative_general",
        mode="compression",
        register="formal",
        instruction="Compress this into a plain paragraph. Return only the message.",
        input_text="Prepare for the DataStream Analytics audit.",
        constraints={
            "target_words": 50,
            "target_tolerance": 0.25,
            "no_subject_line": True,
            "no_signoff": True,
        },
        reward_profile="compression_update",
        required_facts=[],
        forbidden_facts=[],
    )
    checks = _by_name(
        "Please review the source message and ensure your draft is easy to reference.",
        task,
    )

    assert not checks["instruction_leak"].passed
    assert "source message" in checks["instruction_leak"].matches
    assert "your draft" in checks["instruction_leak"].matches


def test_instruction_language_allowed_for_multi_constraint_compose() -> None:
    task = _task(
        id="rl_v03_000268",
        family="direct_email",
        domain="creative_general",
        mode="multi_constraint_compose",
        register="formal",
        instruction="Draft a response brief with Requirements and Checks.",
        input_text="Brief context: Arista hesitates near the wave.",
        constraints={
            "target_words": 250,
            "target_tolerance": 0.15,
            "no_subject_line": True,
            "no_signoff": True,
            "required_section_headings": ["Overview", "Requirements", "Checks"],
            "allow_headings": True,
        },
        reward_profile="direct_workplace_message",
        required_facts=[],
        forbidden_facts=[],
    )
    checks = _by_name(
        (
            "Overview\nThe response must test constraint following.\n\n"
            "Requirements\nThe writer must preserve the image.\n\n"
            "Checks\nVerify the output."
        ),
        task,
    )

    assert checks["instruction_leak"].passed


def test_salutation_openers_are_not_treated_as_required_entities() -> None:
    """Slice 4 follow-up audit: words like 'Please', 'Hi', 'Hey', 'Best',
    'Dear', 'Today' were leaking through ENTITY_RE and producing
    missing_entity false positives when paraphrases dropped them.
    The expanded STOP_ENTITIES set must prevent that.
    """
    task = _task(
        id="rl_v01_000010",
        input_text=(
            "Please be advised that staging has been restored. The root cause was "
            "a missing STRIPE_WEBHOOK_SECRET value, and we will monitor it until 3 pm."
        ),
        required_facts=[
            "staging restored",
            "missing STRIPE_WEBHOOK_SECRET",
            "monitor until 3 pm",
        ],
    )
    response = "Staging is back up at 3 pm — STRIPE_WEBHOOK_SECRET was the fix."
    checks = _by_name(response, task)
    # The model preserves the only real entity; "Please" must not be missed.
    assert checks["missing_entity"].passed, checks["missing_entity"].matches
    assert "Please" not in checks["missing_entity"].matches


def test_salutation_openers_in_response_do_not_count_as_invented_entities() -> None:
    """Symmetric: if the response *adds* a 'Hi' that wasn't in the source,
    that should not trigger missing_entity either (it's not an entity)."""
    task = _task(
        input_text="Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
        required_facts=["Staging", "3 pm", "STRIPE_WEBHOOK_SECRET"],
    )
    response = (
        "Hi team — Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix."
    )
    checks = _by_name(response, task)
    assert checks["missing_entity"].passed


def test_prompt_scaffold_required_facts_are_ignored() -> None:
    task = _task(
        id="rl_v01_000020",
        input_text=(
            "Rewrite this message for the #team-engineering Slack channel: "
            "'Please confirm the hotfix is live in production.'"
        ),
        constraints={
            "max_words": 25,
            "no_subject_line": True,
            "no_signoff": True,
            "preserve_entities": True,
        },
        required_facts=["Rewrite", "Slack"],
        forbidden_facts=[],
    )
    checks = _by_name("The hotfix is live in production.", task)

    assert checks["missing_required_fact"].passed
    assert "Rewrite" not in checks["missing_entity"].matches
    assert "Slack" not in checks["missing_entity"].matches


def test_email_subject_and_salutation_scaffold_are_ignored() -> None:
    task = _task(
        id="rl_v01_000016",
        input_text=(
            "Subject: Compliance Review: Key Areas of Focus\n\n"
            "Dear Ms. Young,\n\n"
            "I hope this message finds you well.\n\n"
            "Please review customer insights and product improvements for "
            "Cloud Nine Systems.\n\n"
            "Firstly, gather feedback. Secondly, review recent team updates. "
            "Understanding the status will help prioritize work."
        ),
        constraints={
            "max_words": 25,
            "no_subject_line": True,
            "no_signoff": True,
            "preserve_entities": True,
        },
        required_facts=["Compliance Review", "Key Areas", "Focus\n\nDear", "Young"],
        forbidden_facts=[],
    )
    checks = _by_name(
        "Please review customer insights and product improvements for Cloud Nine Systems.",
        task,
    )

    assert checks["missing_required_fact"].passed
    assert "Focus\n\nDear" not in checks["missing_required_fact"].matches
    assert "Compliance Review" not in checks["missing_entity"].matches
    assert "Review" not in checks["missing_entity"].matches
    assert "Key Areas" not in checks["missing_entity"].matches
    assert "Firstly" not in checks["missing_entity"].matches
    assert "Secondly" not in checks["missing_entity"].matches
    assert "Understanding" not in checks["missing_entity"].matches
    assert "Young" not in checks["missing_entity"].matches


def test_discourse_required_fact_fragments_are_ignored() -> None:
    task = _task(
        id="rl_v01_000117",
        input_text=(
            "Given the finance market, please review the vendor evaluation "
            "with Premier Solutions."
        ),
        constraints={
            "max_words": 25,
            "no_subject_line": True,
            "no_signoff": True,
            "preserve_entities": True,
        },
        required_facts=["Given", "Premier Solutions"],
        forbidden_facts=[],
    )
    checks = _by_name("Review the vendor evaluation with Premier Solutions.", task)

    assert checks["missing_required_fact"].passed
    assert "Given" not in checks["missing_required_fact"].matches
    assert checks["missing_entity"].passed


def test_required_entities_do_not_merge_across_fact_boundaries() -> None:
    task = _task(
        id="rl_v01_000118",
        input_text="Send Jordan the vendor update for Premier Solutions by Tuesday.",
        constraints={
            "max_words": 25,
            "no_subject_line": True,
            "no_signoff": True,
            "preserve_entities": True,
        },
        required_facts=["Premier Solutions", "Tuesday", "Jordan"],
        forbidden_facts=[],
    )
    checks = _by_name(
        "Jordan, the Premier Solutions vendor update is still on for Tuesday.",
        task,
    )

    assert checks["missing_entity"].passed
    assert "Premier Solutions\nTuesday\nJordan" not in checks["missing_entity"].matches


def test_invented_number_and_temporal_details_trigger() -> None:
    task = _task(
        id="rl_v01_000286",
        input_text="The account credentials issue is under review. Support will update the customer.",
        constraints={
            "max_words": 30,
            "no_subject_line": True,
            "no_signoff": True,
            "preserve_numbers": True,
            "preserve_entities": True,
        },
        required_facts=[],
        forbidden_facts=[],
    )
    checks = _by_name(
        "Account credentials were wrong. We'll fix it by Friday at 5pm.",
        task,
    )

    assert not checks["invented_number"].passed
    assert "5pm" in checks["invented_number"].matches
    assert not checks["invented_temporal_detail"].passed
    assert "Friday" in checks["invented_temporal_detail"].matches


def test_unsupported_negation_and_low_source_overlap_trigger() -> None:
    task = _task(
        id="rl_v01_000117",
        input_text=(
            "The vendor evaluation should cover market trends, partnership "
            "performance, risk exposure, implementation timing, benchmark "
            "comparisons, finance-sector constraints, and next-step owners "
            "for Premier Solutions before the Tuesday review."
        ),
        constraints={
            "max_words": 45,
            "no_subject_line": True,
            "no_signoff": True,
            "preserve_numbers": True,
            "preserve_entities": True,
        },
        required_facts=["Premier Solutions"],
        forbidden_facts=[],
    )
    checks = _by_name(
        (
            "I didn't send this because I'm tired of best practices. "
            "If you say no, I'll fire you and pay my cousin $500."
        ),
        task,
    )

    assert not checks["unsupported_negation"].passed
    assert "I didn't" in checks["unsupported_negation"].matches
    assert not checks["low_source_overlap"].passed


def test_fake_casual_low_specificity_and_thanks_padding_trigger() -> None:
    task = _task(
        id="rl_v01_000119",
        input_text=(
            "Project Apollo deliverables are ready in the shared folder. "
            "Feedback is due Friday before the revision closes."
        ),
        constraints={
            "max_words": 35,
            "no_subject_line": True,
            "no_signoff": True,
            "preserve_entities": True,
        },
        required_facts=["Project Apollo", "shared folder", "Friday"],
        forbidden_facts=[],
    )
    checks = _by_name(
        "We got stuff we need for Project too. Thanks for looking out.",
        task,
    )

    assert not checks["fake_casual_phrase"].passed
    assert "we got" in checks["fake_casual_phrase"].matches
    assert "stuff" in checks["fake_casual_phrase"].matches
    assert not checks["low_specificity_substitution"].passed
    assert "stuff" in checks["low_specificity_substitution"].matches
    assert not checks["thanks_padding"].passed
    assert "Thanks for looking out." in checks["thanks_padding"].matches


def test_broken_informal_grammar_and_register_mismatch_trigger() -> None:
    checks = _by_name(
        "We working on it now, we tell you when stuff good again. "
        "We're doing you a solid here, so cut the crap."
    )

    assert not checks["broken_informal_grammar"].passed
    assert "We working" in checks["broken_informal_grammar"].matches
    assert not checks["register_mismatch"].passed
    assert "doing you a solid" in checks["register_mismatch"].matches
    assert "cut the crap" in checks["register_mismatch"].matches


def test_real_casual_slack_control_is_not_penalized() -> None:
    checks = _by_name(
        "Staging is back at 3 pm after the STRIPE_WEBHOOK_SECRET fix. "
        "I'll keep an eye on it through handoff."
    )

    assert checks["fake_casual_phrase"].passed
    assert checks["low_specificity_substitution"].passed
    assert checks["broken_informal_grammar"].passed
    assert checks["register_mismatch"].passed
    assert checks["thanks_padding"].passed


def test_step170_high_reward_survivors_trigger() -> None:
    checks = _by_name(
        "We updated wireframes we posted on Figma. "
        "We wanted feedback before day ends for project we're doing now; "
        "dev people need those tomorrow; thanks for looking."
    )

    assert not checks["broken_informal_grammar"].passed
    assert "before day ends" in checks["broken_informal_grammar"].matches
    assert not checks["thanks_padding"].passed
    assert "thanks for looking." in checks["thanks_padding"].matches


def test_repeated_ngram_triggers_repetition_diagnostic() -> None:
    checks = _by_name("I want to talk " * 12)

    assert not checks["repetition"].passed
    assert checks["repetition"].penalty < 0
    assert "i want to talk" in checks["repetition"].matches


def test_target_words_tolerance_drives_length_diagnostics() -> None:
    task = _target_task()

    on_target = _by_name(_words(100), task)
    runaway = _by_name(_words(200), task)
    too_short = _by_name(_words(40), task)

    assert on_target["too_long"].passed
    assert on_target["too_short"].passed
    assert not runaway["too_long"].passed
    assert not too_short["too_short"].passed


def test_v03_romance_task_flags_unsuitable_recommendations() -> None:
    task = _romance_recommendation_task()
    bad = _by_name(
        (
            '"Love means never having to say you\'re sorry." We could all use '
            "a softer movie night, couldn't we? P.S. Watch 12 Years a Slave."
        ),
        task,
    )
    good = _by_name(
        (
            '"As you wish." Let\'s make tonight gentle, hopeful, and a little '
            "sappy. P.S. I'd pick The Princess Bride when you want something "
            "warm."
        ),
        task,
    )

    assert not bad["unsuitable_recommendation"].passed
    assert "12 Years a Slave" in bad["unsuitable_recommendation"].matches
    assert good["unsuitable_recommendation"].passed


def test_v03_phrase_structure_and_register_constraints_trigger() -> None:
    task = _romance_recommendation_task(
        must_include_phrases=["movie night"],
        allow_bullets=False,
        allow_headings=False,
        max_paragraphs=1,
    )
    checks = _by_name(
        (
            "Dear followers,\n\n"
            "Romance Picks:\n"
            "- In conclusion, this movie night is sweet — watch The Notebook."
        ),
        task,
    )

    assert checks["forbidden_opener"].passed is False
    assert checks["forbidden_phrase"].passed is False
    assert checks["em_dash"].passed is False
    assert checks["wrong_format_bullets"].passed is False
    assert checks["wrong_format_heading"].passed is False
    assert checks["paragraph_count"].passed is False
    assert checks["missing_contraction"].passed is False


def test_surface_decoration_triggers_when_not_requested() -> None:
    task = _romance_recommendation_task()
    star = chr(0x1F31F)
    checks = _by_name(
        (
            "GET DIRECT ROMANCE PICKS NOW.\n\n"
            f"Hey friends! {star} Let's dive into movie night. "
            "P.S. Watch The Princess Bride. #RomanceFilms"
        ),
        task,
    )

    assert not checks["emoji"].passed
    assert star in checks["emoji"].matches
    assert not checks["all_caps"].passed
    assert "GET DIRECT ROMANCE PICKS NOW." in checks["all_caps"].matches
    assert not checks["hashtag"].passed
    assert "#RomanceFilms" in checks["hashtag"].matches


def test_surface_decoration_allowed_when_explicitly_requested() -> None:
    star = chr(0x1F31F)
    task = _task(
        id="rl_v03_000393",
        family="tone_shift",
        domain="creative_general",
        mode="tone_shift",
        register="warm",
        instruction=(
            "Write a social post in ALL CAPS. Include one emoji and one "
            "hashtag. Recommend an uplifting romance film."
        ),
        input_text="Please recommend an uplifting romance film.",
        constraints={
            "target_words": 18,
            "target_tolerance": 0.50,
            "min_contraction_count": 0,
        },
        reward_profile="rewrite_faithful_concise",
        required_facts=[],
        forbidden_facts=[],
    )
    checks = _by_name(
        f"WATCH THE PRINCESS BRIDE TONIGHT {star} #RomanceFilms",
        task,
    )

    assert checks["emoji"].passed
    assert checks["all_caps"].passed
    assert checks["hashtag"].passed


def test_negated_decoration_instructions_do_not_allow_surface_noise() -> None:
    star = chr(0x1F31F)
    task = _task(
        id="rl_v03_000394",
        family="tone_shift",
        domain="creative_general",
        mode="tone_shift",
        register="warm",
        instruction=(
            "Write a direct social post. MUST_NOT: Use emoji, hashtags, or "
            "all caps. Keep the answer plain."
        ),
        input_text="Please recommend an uplifting romance film.",
        constraints={
            "target_words": 18,
            "target_tolerance": 0.50,
            "min_contraction_count": 0,
        },
        reward_profile="rewrite_faithful_concise",
        required_facts=[],
        forbidden_facts=[],
    )
    checks = _by_name(
        f"WATCH THE PRINCESS BRIDE TONIGHT {star} #RomanceFilms",
        task,
    )

    assert not checks["emoji"].passed
    assert not checks["all_caps"].passed
    assert not checks["hashtag"].passed
