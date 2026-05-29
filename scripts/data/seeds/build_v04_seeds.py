import json
import os


def build_seeds():
    os.makedirs("seeds", exist_ok=True)
    
    # 1. 150 task seeds across 9 categories
    # Each seed must have: "instruction", "task_type", "persona", "domain", "mode" (direct_generation or rewrite_humanize)
    task_seeds = []
    
    # --- Category 1: slack_chat (18 seeds) ---
    task_seeds.extend([
        # Rewrite
        {
            "instruction": "Rewrite this Slack message to be casual and direct:\n\n'Please be advised that the weekly sync has been rescheduled to tomorrow at 10 AM. Your presence would be highly appreciated. Let me know if this presents any scheduling conflicts.'",
            "task_type": "slack_chat",
            "persona": None,
            "domain": "chat",
            "mode": "rewrite_humanize"
        },
        {
            "instruction": "Rewrite this robotic update for our team channel:\n\n'I am pleased to announce that we have successfully resolved the authentication bug. The fix is currently live in production. We will continue to monitor performance parameters.'",
            "task_type": "slack_chat",
            "persona": None,
            "domain": "chat",
            "mode": "rewrite_humanize"
        },
        {
            "instruction": "Rewrite this Slack message draft:\n\n'It has come to my attention that the API endpoints are experiencing elevated latency. I am actively investigating the root cause and will keep you informed of further developments.'",
            "task_type": "slack_chat",
            "persona": None,
            "domain": "chat",
            "mode": "rewrite_humanize"
        },
        {
            "instruction": "This sounds too formal. Fix it for Slack:\n\n'Hi team, I would like to request that everyone submit their weekly reports by the close of business today to facilitate project tracking. Thank you in advance for your prompt attention.'",
            "task_type": "slack_chat",
            "persona": None,
            "domain": "chat",
            "mode": "rewrite_humanize"
        },
        {
            "instruction": "Clean up this AI-drafted reply to a colleague's question about the database schemas:\n\n'In response to your query regarding the schema structure, it is worth noting that we have deprecated the legacy user_meta table. Furthermore, all active columns have been migrated to user_profiles.'",
            "task_type": "slack_chat",
            "persona": None,
            "domain": "chat",
            "mode": "rewrite_humanize"
        },
        # Direct
        {
            "instruction": "Draft a quick Slack message to my manager asking if they have 5 minutes to chat about career growth this afternoon.",
            "task_type": "slack_chat",
            "persona": "junior developer",
            "domain": "chat",
            "mode": "direct_generation"
        },
        {
            "instruction": "Write a Slack message to the team apologizing for breaking the staging environment and explaining that I am rolling it back now.",
            "task_type": "slack_chat",
            "persona": "tech lead",
            "domain": "chat",
            "mode": "direct_generation"
        },
        {
            "instruction": "Write a quick ping to a peer PM asking if they've reviewed the PRD for the billing page yet.",
            "task_type": "slack_chat",
            "persona": "product manager",
            "domain": "chat",
            "mode": "direct_generation"
        },
        {
            "instruction": "Ask a colleague in a DM if they want to grab lunch or coffee around 12:30 PM.",
            "task_type": "slack_chat",
            "persona": "coworker",
            "domain": "chat",
            "mode": "direct_generation"
        },
        {
            "instruction": "Write a message for the #general channel letting everyone know there are leftover donuts in the kitchen.",
            "task_type": "slack_chat",
            "persona": "office assistant",
            "domain": "chat",
            "mode": "direct_generation"
        }
    ])
    
    # Fill remaining to make 18 for slack_chat
    slack_extra = [
        ("Rewrite: 'Should you require additional documentation, please do not hesitate to reach out.'", "rewrite_humanize", None),
        ("Rewrite: 'I would like to extend my gratitude for your support during the migration process.'", "rewrite_humanize", None),
        ("Rewrite: 'It is recommended that we verify the results before proceeding with the production release.'", "rewrite_humanize", None),
        ("Draft a Slack update letting the customer support team know that the payment gateway is fully restored.", "direct_generation", "on-call engineer"),
        ("Draft a message to a teammate asking if they can cover my on-call shift for 2 hours on Thursday afternoon.", "direct_generation", "engineer"),
        ("Draft a message to the team celebrating the successful launch of the dashboard.", "direct_generation", "manager"),
        ("Rewrite: 'I am writing to inquire if the marketing assets are prepared for our campaign launch tomorrow.'", "rewrite_humanize", None),
        ("Draft a quick Slack message to the designer asking where the updated Figma links are located.", "direct_generation", "frontend dev")
    ]
    for inst, mode, pers in slack_extra:
        task_seeds.append({
            "instruction": inst,
            "task_type": "slack_chat",
            "persona": pers,
            "domain": "chat",
            "mode": mode
        })

    # --- Category 2: email_rewrite (18 seeds) ---
    task_seeds.extend([
        # Rewrite
        {
            "instruction": "Rewrite this email to sound like a normal professional person:\n\n'Dear team, I hope this email finds you well. I am writing to provide an update on our quarterly milestones. We have made significant progress; however, some challenges remain. Please review the attached deck.'",
            "task_type": "email_rewrite",
            "persona": None,
            "domain": "email",
            "mode": "rewrite_humanize"
        },
        {
            "instruction": "Rewrite this stiff corporate email to a partner:\n\n'Pursuant to our conversation yesterday, please find enclosed the draft proposal for our integration. Please do not hesitate to contact me if you have any questions or require further clarification.'",
            "task_type": "email_rewrite",
            "persona": None,
            "domain": "email",
            "mode": "rewrite_humanize"
        },
        {
            "instruction": "Make this customer email warmer and less robotic:\n\n'We regret to inform you that your request for a billing adjustment has been denied because it falls outside our standard 30-day window. We apologize for any inconvenience this may cause.'",
            "task_type": "email_rewrite",
            "persona": None,
            "domain": "email",
            "mode": "rewrite_humanize"
        },
        # Direct
        {
            "instruction": "Write a professional but warm email to a candidate rejecting them after the final round of interviews.",
            "task_type": "email_rewrite",
            "persona": "recruiter",
            "domain": "email",
            "mode": "direct_generation"
        },
        {
            "instruction": "Write an email to a client explaining that we found a bug in their integration and are deploying a hotfix in the next hour.",
            "task_type": "email_rewrite",
            "persona": "solutions engineer",
            "domain": "email",
            "mode": "direct_generation"
        },
        {
            "instruction": "Draft an email to my manager requesting PTO for the first week of July, outlining who will cover my projects.",
            "task_type": "email_rewrite",
            "persona": "employee",
            "domain": "email",
            "mode": "direct_generation"
        }
    ])
    
    # Fill remaining email_rewrite
    email_extra = [
        ("Rewrite: 'I am writing to express my interest in the Product Manager position at your esteemed firm.'", "rewrite_humanize", None),
        ("Rewrite: 'Please be advised that your subscription is scheduled for renewal on June 1st, 2026.'", "rewrite_humanize", None),
        ("Rewrite: 'I would like to kindly remind all employees that the office will be closed in observance of the upcoming holiday.'", "rewrite_humanize", None),
        ("Rewrite: 'In order to optimize our database throughput, it is necessary that we clean up historical indexes.'", "rewrite_humanize", None),
        ("Rewrite: 'I hope this email finds you well. I wanted to check if you had an opportunity to review my previous email.'", "rewrite_humanize", None),
        ("Draft a cold sales outreach email to a CTO offering a demo of our database scaling tool. Make it brief and direct.", "direct_generation", "sales rep"),
        ("Write a follow-up email to a prospective client who went quiet after a demo. Keep it friendly and concise.", "direct_generation", "account executive"),
        ("Draft an email to a vendor asking for a discount on our software license renewal.", "direct_generation", "procurement lead"),
        ("Draft an email to the company letting them know the AC is broken in the east wing and repair is scheduled.", "direct_generation", "office manager"),
        ("Rewrite: 'Your account has been flagged due to security concerns. Please click the link to authenticate your credentials.'", "rewrite_humanize", None),
        ("Draft an email to an API partner asking for access to their beta endpoints.", "direct_generation", "developer relations"),
        ("Draft an email to my team asking for volunteers to present at the next lunch-and-learn session.", "direct_generation", "team lead")
    ]
    for inst, mode, pers in email_extra:
        task_seeds.append({
            "instruction": inst,
            "task_type": "email_rewrite",
            "persona": pers,
            "domain": "email",
            "mode": mode
        })

    # --- Category 3: style_transfer (16 seeds) ---
    style_extra = [
        ("Rewrite this standard update as a startup founder talking to early team members: 'Our traction has been flat, and we must find a new channel.'", "rewrite_humanize", "startup founder"),
        ("Rewrite this as a senior backend engineer who dislikes corporate jargon: 'We need to align key stakeholders and synergize our efforts.'", "rewrite_humanize", "senior engineer"),
        ("Rewrite this to sound like a tired PM at 6pm: 'The sprint planning is tomorrow. Everyone should update their ticket status.'", "rewrite_humanize", "tired PM"),
        ("Rewrite this email to sound like a supportive team coach: 'You made a mistake, but we can recover.'", "rewrite_humanize", "team leader"),
        ("Write a review of a restaurant in the style of a food critic who hates pretentious descriptions.", "direct_generation", "food critic"),
        ("Explain how database indexes work in the style of a tired programmer explaining it to a non-technical coworker.", "direct_generation", "tired programmer"),
        ("Write a project delay notice in the voice of a direct, transparent CEO who doesn't sugarcoat bad news.", "direct_generation", "transparent CEO"),
        ("Draft a feature release note in the style of a UX writer who is obsessed with simplicity and clarity.", "direct_generation", "UX writer"),
        ("Rewrite: 'We must align on product priorities to ensure maximum customer impact.' as a cynical startup veteran.", "rewrite_humanize", "cynical startup veteran"),
        ("Explain Git rebase to a junior developer in a warm, encouraging, mentor-like voice.", "direct_generation", "mentor lead"),
        ("Draft a LinkedIn post announcing a new job in a humble, down-to-earth voice that avoids standard corporate bragging.", "direct_generation", "humble engineer"),
        ("Rewrite: 'I apologize for the delay in returning your call. Let's schedule a meeting.' in the tone of a busy VC.", "rewrite_humanize", "busy VC"),
        ("Explain why a feature was cut in a transparent, empathetic product manager voice.", "direct_generation", "empathetic PM"),
        ("Draft a Slack update about database migration failure in the style of a calm on-call engineer.", "direct_generation", "calm SRE"),
        ("Rewrite: 'We are pleased to introduce our new analytics dashboard, which offers advanced insights.' to sound like a tech copywriter writing for developers.", "rewrite_humanize", "developer advocate"),
        ("Explain API rate limits like a security engineer talking to a junior frontend engineer.", "direct_generation", "security engineer"),
        ("Explain what a merge conflict is to a designer who has never used git.", "direct_generation", "mentor lead"),
        ("Rewrite: 'We have optimized the codebase resulting in elevated efficiency.' in the style of a pragmatic tech lead.", "rewrite_humanize", "pragmatic tech lead")
    ]
    for inst, mode, pers in style_extra:
        task_seeds.append({
            "instruction": inst,
            "task_type": "style_transfer",
            "persona": pers,
            "domain": "creative",
            "mode": mode
        })

    # --- Category 4: shorten_compress (16 seeds) ---
    compress_extra = [
        ("Shorten this long update to one concise paragraph:\n\n'In order to facilitate the progress of our current initiatives, we decided to perform an audit of our cloud infrastructure cost metrics. It is worth noting that we identified several orphaned storage volumes that are contributing to excess spend. Therefore, we are going to purge these volumes tonight.'", "rewrite_humanize", None),
        ("Cut this Slack message to 2 sentences max:\n\n'Hi team, I wanted to follow up on the discussion we had earlier regarding the API design. I have created a document outlining the core options and would appreciate it if you could take a look and add your comments.'", "rewrite_humanize", None),
        ("Condense this status report to its core blockers and next steps:\n\n'We spent the last two days trying to debug the memory leak. We ran multiple profiles and found that the garbage collector is stalling due to excessive object allocation in the parser. We are planning to refactor the parser module. We cannot proceed with the deployment until this is fixed.'", "rewrite_humanize", None),
        ("Tighten this email draft to be brief and direct:\n\n'Dear client, I am writing to let you know that we have finalized the migration of your account to our new servers. You might notice some speed improvements. Please let us know if you spot any issues.'", "rewrite_humanize", None),
        ("Draft a one-sentence summary of this update: 'We had to roll back the release because the user session token was expiring after 5 minutes due to an incorrect config change. The team is debugging now.'", "direct_generation", None),
        ("Rewrite this to be under 50 words: 'I wanted to check in to see if you have had a chance to review the onboarding designs I sent over on Tuesday. We have a meeting with the client on Friday and it would be great to have your feedback before then.'", "rewrite_humanize", None),
        ("Shorten this meeting invite description to a single bulleted list of 3 key topics.", "direct_generation", None),
        ("Make this release log brief. Cut out all introductory fluff.", "rewrite_humanize", None),
        ("Rewrite: 'It is highly critical that all developers clean up their local branches after merging PRs so that we do not clutter the remote repository.' to be short and clear.", "rewrite_humanize", None),
        ("Draft a 3-bullet-point summary of a long technical incident report.", "direct_generation", None),
        ("Condense this: 'The purpose of this document is to outline the rules for code reviews. Code reviews are important. We must review all PRs before merging.'", "rewrite_humanize", None),
        ("Reduce this status report to just two lines of text.", "rewrite_humanize", None),
        ("Draft a ultra-short Slack ping asking for a design asset. Under 10 words.", "direct_generation", None),
        ("Shorten: 'In order to ensure that we maintain high code standards, we will be holding weekly code review workshops.'", "rewrite_humanize", None),
        ("Compress this customer notification down to a single sentence.", "rewrite_humanize", None),
        ("Draft a short post-mortem summary for a minor outage. Keep it under 60 words.", "direct_generation", None)
    ]
    for inst, mode, pers in compress_extra:
        task_seeds.append({
            "instruction": inst,
            "task_type": "shorten_compress",
            "persona": pers,
            "domain": "general",
            "mode": mode
        })

    # --- Category 5: grammar_clarity (16 seeds) ---
    grammar_extra = [
        ("Fix the grammar and awkward phrasing in this Slack message:\n\n'Me and John done the changes but we seen some error on staging.'", "rewrite_humanize", None),
        ("Clean up the grammar in this draft but keep the casual, friendly tone:\n\n'Just wanted to check if your coming tomorrow because if so we can discuss the schema, what do you think?'", "rewrite_humanize", None),
        ("Rewrite this for clarity. The sentence is too long and confusing:\n\n'The server which was updated yesterday by the ops team that had the new database drivers installed is now showing memory spikes because of the connection pool settings are too high.'", "rewrite_humanize", None),
        ("Fix typos and punctuation in this quick response:\n\n'hey, i think we should use the new API. its much faster and it dont require auth token in header'", "rewrite_humanize", None),
        ("Rewrite this message to make it clear and grammatically correct without sounding formal:\n\n'We was thinking about doing the migration at Friday evening, does that works for you?'", "rewrite_humanize", None),
        ("Improve the clarity of this technical explanation: 'Docker is like a VM but it doesn't have an OS inside, it shares the host OS, making it faster.'", "rewrite_humanize", None),
        ("Fix the punctuation and run-on structure of this bug description.", "rewrite_humanize", None),
        ("Rewrite this customer update to be grammatically correct and easier to read.", "rewrite_humanize", None),
        ("Clean up this stream-of-consciousness draft: 'So I was looking at the logs and there is a lot of timeouts, we probably should increase the database pool size or maybe optimize that query, not sure.'", "rewrite_humanize", None),
        ("Fix grammar and clarity: 'Its important that we dont run migrations when traffic is high because it can lock tables.'", "rewrite_humanize", None),
        ("Rewrite this technical error explanation so a non-technical manager can understand it clearly.", "rewrite_humanize", None),
        ("Correct the grammatical errors in this quick status report: 'Yesterday I have working on the landing page, today I am going to finish it.'", "rewrite_humanize", None),
        ("Rewrite this confusing email sentence to be clear and professional.", "rewrite_humanize", None),
        ("Fix the syntax and typos in this draft reply to a GitHub issue.", "rewrite_humanize", None),
        ("Rewrite this instructions list to make it easier for a new hire to follow.", "rewrite_humanize", None),
        ("Clean up the phrasing: 'We need to make sure that we have all the tests passing before we do any commit to main.'", "rewrite_humanize", None)
    ]
    for inst, mode, pers in grammar_extra:
        task_seeds.append({
            "instruction": inst,
            "task_type": "grammar_clarity",
            "persona": pers,
            "domain": "general",
            "mode": mode
        })

    # --- Category 6: tone_adjustment (16 seeds) ---
    tone_extra = [
        ("Change the tone of this email from cold/formal to warm and professional:\n\n'Your pull request has been reviewed. Several changes are required before it can be merged. Refer to the comments below.'", "rewrite_humanize", None),
        ("Make this message to a client less blunt and more polite:\n\n'We cannot deliver the feature this week. We are behind schedule. It will be ready next Wednesday.'", "rewrite_humanize", None),
        ("Rewrite this casual Slack message to be professional enough for a client email:\n\n'hey, we had a bug with payments but we fixed it. should be all good now. let us know if it fails again.'", "rewrite_humanize", None),
        ("Tone down the defensiveness in this email response to a bug report:\n\n'The bug you reported is not a problem with our code. It is because you configured the API key wrong. Please fix your config.'", "rewrite_humanize", None),
        ("Rewrite this to sound more collaborative and less demanding: 'You need to finish your code review by noon today.'", "rewrite_humanize", None),
        ("Make this rejection email warmer and more encouraging.", "rewrite_humanize", None),
        ("Draft a Slack announcement about a reorganizing of teams. Make it sound exciting rather than corporate and dry.", "direct_generation", None),
        ("Rewrite: 'We regret to inform you that your pull request has been closed because it does not align with our roadmap.' to sound polite but firm.", "rewrite_humanize", None),
        ("Draft a message apologizing for a late response. Make it warm and friendly.", "direct_generation", None),
        ("Make this project update sound more confident and proactive: 'We are still having some trouble with the API, but we hope to fix it soon.'", "rewrite_humanize", None),
        ("Draft a gentle nudge to a coworker who hasn't responded to a blocker ticket.", "direct_generation", None),
        ("Rewrite: 'Do not use this channel for random questions.' to sound polite and redirect users to the correct channel.", "rewrite_humanize", None),
        ("Draft a request for feedback on a document. Make it open and inviting.", "direct_generation", None),
        ("Rewrite this formal out-of-office message to sound more natural and friendly.", "rewrite_humanize", None),
        ("Rewrite: 'We are experiencing server issues and you can't log in.' to sound professional, calm, and reassuring.", "rewrite_humanize", None),
        ("Draft a welcome message for a new hire on Slack. Make it warm and excited.", "direct_generation", None)
    ]
    for inst, mode, pers in tone_extra:
        task_seeds.append({
            "instruction": inst,
            "task_type": "tone_adjustment",
            "persona": pers,
            "domain": "general",
            "mode": mode
        })

    # --- Category 7: meeting_async (16 seeds) ---
    meeting_extra = [
        ("Summarize these raw meeting notes into a brief, scannable Slack update:\n\n'Meeting started 10am. Alice discussed frontend design. Dashboard is ready for review. Bob needs backend endpoints. Connection pool issue is blocking Bob. Charlie will look at connection pool. Goal is launch by Friday.'", "rewrite_humanize", None),
        ("Turn this transcripts section into 3 bullet points for a daily standup update:\n\n'Yesterday I worked on database cleanup. Met with design to talk about dashboard inputs. Today I'm going to start writing the migrations. No blockers except waiting on Alice for assets.'", "rewrite_humanize", None),
        ("Draft a summary of a design sync meeting focusing only on the final decisions and next steps.", "direct_generation", None),
        ("Rewrite this robotic status email into a casual, async team update:\n\n'Project status is green. Milestone 1 has been completed. We are now transitioning to Milestone 2. Alice is lead.'", "rewrite_humanize", None),
        ("Turn this long list of chat messages about a production bug into a clear incident timeline for Slack.", "direct_generation", None),
        ("Draft an async status update for my team outlining my focus for the week and one blocker.", "direct_generation", None),
        ("Rewrite: 'The purpose of this meeting was to align on the database schema. We decided to use Postgres.' to be a quick Slack summary.", "rewrite_humanize", None),
        ("Summarize this video call transcript section about user feedback into a short paragraph.", "rewrite_humanize", None),
        ("Draft a follow-up action item list for a team after a brainstorming session.", "direct_generation", None),
        ("Rewrite: 'We held a sync regarding cloud cost optimization. Alice proposed shutting down idle instances.' as a brief Slack ping.", "rewrite_humanize", None),
        ("Summarize a sprint retro notes list into 'What went well' and 'Action items'.", "direct_generation", None),
        ("Draft a quick Slack summary of a 1-on-1 meeting with a report.", "direct_generation", None),
        ("Rewrite this corporate meeting recap to sound natural and informal.", "rewrite_humanize", None),
        ("Draft a standup update outlining two completed tickets and one active blocker.", "direct_generation", None),
        ("Summarize these release notes into a 2-sentence Slack announcement.", "rewrite_humanize", None),
        ("Draft an email recap of a client kickoff meeting containing decisions and next steps.", "direct_generation", None)
    ]
    for inst, mode, pers in meeting_extra:
        task_seeds.append({
            "instruction": inst,
            "task_type": "meeting_async",
            "persona": pers,
            "domain": "chat",
            "mode": mode
        })

    # --- Category 8: product_copy (16 seeds) ---
    product_extra = [
        ("This landing page hero section copy sounds too generic. Rewrite it to sound authentic and engaging:\n\n'Welcome to the future of team collaboration. Our cutting-edge platform leverages state-of-the-art technology to synergize your workflows and maximize productivity levels.'", "rewrite_humanize", None),
        ("Rewrite this robotic product announcement email to sound like a real person sharing good news:\n\n'We are thrilled to announce the release of our brand-new analytics dashboard. This feature has been meticulously crafted to provide you with actionable insights and deep visibility.'", "rewrite_humanize", None),
        ("Draft a short, punchy app store description for a simple command-line file sharing tool.", "direct_generation", None),
        ("Rewrite this feature description to be simple and user-focused:\n\n'Our platform utilizes an advanced machine learning algorithm to automatically categorize your expenses in real-time, thereby eliminating manual entry.'", "rewrite_humanize", None),
        ("Draft a product launch tweet for a new developer tool. Keep it under 200 characters and avoid buzzwords.", "direct_generation", None),
        ("Write a hero headline and subheadline for a privacy-focused analytics service. Avoid using terms like 'revolutionary' or 'next-gen'.", "direct_generation", None),
        ("Rewrite this corporate marketing bullet list to sound human and practical.", "rewrite_humanize", None),
        ("Draft a changelog entry for a new dark mode feature. Make it lighthearted and brief.", "direct_generation", None),
        ("Rewrite: 'Our software provides unparalleled security and military-grade encryption.' to sound believable and simple.", "rewrite_humanize", None),
        ("Draft a microcopy for a button that confirms a user is deleting their account.", "direct_generation", None),
        ("Rewrite this robotic newsletter intro to sound conversational.", "rewrite_humanize", None),
        ("Draft a short explanation of how our pricing works for a landing page FAQ section.", "direct_generation", None),
        ("Rewrite: 'We empower organizations to unlock their full potential.' to sound concrete and specific.", "rewrite_humanize", None),
        ("Draft a welcome message for a user onboarding screen. Keep it short and friendly.", "direct_generation", None),
        ("Rewrite this landing page copy for a database product: 'Experience blazing-fast query speeds and high availability.'", "rewrite_humanize", None),
        ("Draft a tool tip text explaining what API rate limits are.", "direct_generation", None)
    ]
    for inst, mode, pers in product_extra:
        task_seeds.append({
            "instruction": inst,
            "task_type": "product_copy",
            "persona": pers,
            "domain": "technical",
            "mode": mode
        })

    # --- Category 9: ocr_document_text (16 seeds) ---
    ocr_extra = [
        ("Rewrite the key details from this OCR scan of a meeting agenda into clean, modern prose:\n\n[Screenshot content:]\nDATE: 2026-05-12\nATTENDEES: Alice, Bob, Charlie\nTOPIC 1: Q3 Budget review (10:00 - 10:30)\nTOPIC 2: Frontend architecture (10:30 - 11:30)\nLUNCH BREAK\nTOPIC 3: QA pipeline automation (13:00 - 14:00)", "rewrite_humanize", None),
        ("Rewrite this rough document draft extracted from a scanner:\n\n[Screenshot content:]\nMEMORANDUM\nTO: All Staff Members\nFROM: Management\nSUBJECT: Office policy amendment\nEffective immediately, all conference rooms must be reserved in advance using the scheduling application. Failure to do so will result in reservation forfeiture.", "rewrite_humanize", None),
        ("Rewrite this scanned form text into a natural email confirmation:\n\n[Screenshot content:]\nORDER NUMBER: #94827\nSTATUS: Processing\nITEMS: 2x Wireless keyboard, 1x USB-C cable\nSHIPPING TO: Jane Doe, 123 Main St, Anytown, US\nDELIVERY EST: 3-5 business days", "rewrite_humanize", None),
        ("Rewrite the notes on this digital whiteboard OCR:\n\n[Screenshot content:]\n- migrate db from mysql to postgres\n- alice to write schema file (target wed)\n- bob needs to setup replication replica\n- charlie will run performance benchmarks on staging after migration", "rewrite_humanize", None),
        ("Rewrite this scanned legal disclaimer for a website footer:\n\n[Screenshot content:]\nDISCLAIMER: The information provided on this website is for general informational purposes only. We make no representations or warranties of any kind regarding accuracy or completeness.", "rewrite_humanize", None),
        ("Rewrite this OCR receipt content into a clear expense description for a manager.", "rewrite_humanize", None),
        ("Rewrite this scanned checklist of server migration steps.", "rewrite_humanize", None),
        ("Rewrite this OCR-extracted text from an old handbook explaining the company's dress code policy.", "rewrite_humanize", None),
        ("Rewrite this scanned flyer text welcoming people to a local developer meetup.", "rewrite_humanize", None),
        ("Rewrite this scanned business card content into a quick note to add the contact on LinkedIn.", "rewrite_humanize", None),
        ("Rewrite this OCR text from a company announcement board.", "rewrite_humanize", None),
        ("Rewrite this scanned invoice summary into a direct Slack ping to the accounting team.", "rewrite_humanize", None),
        ("Rewrite this OCR-extracted project scope document section.", "rewrite_humanize", None),
        ("Rewrite this scanned manual page explaining how to reset the office router.", "rewrite_humanize", None),
        ("Rewrite this scanned customer feedback form text.", "rewrite_humanize", None),
        ("Rewrite this OCR-extracted text from a presentation slide summarizing Q1 sales results.", "rewrite_humanize", None)
    ]
    for inst, mode, pers in ocr_extra:
        task_seeds.append({
            "instruction": inst,
            "task_type": "ocr_document_text",
            "persona": pers,
            "domain": "document",
            "mode": mode
        })

    # Assert exactly 150 seeds
    print(f"Total task seeds built: {len(task_seeds)}")
    assert len(task_seeds) == 150, f"Expected 150 seeds, got {len(task_seeds)}"

    # Apply instruction_framing policy. Keep explicit anti-AI framing out of the
    # pilot slice; most rows should look like normal user tasks.
    for seed in task_seeds:
        inst = seed["instruction"]
        lower_inst = inst.lower()
        seed["instruction"] = (
            inst.replace("Make this Slack message sound like a natural instead of over-polished", "Rewrite this Slack message to be casual and direct")
            .replace("This email sounds like standard over-polished output. Rewrite it to sound like a normal professional person", "Rewrite this email to sound like a normal professional person")
            .replace("Clean up this AI-drafted reply", "Clean up this reply")
            .replace("Shorten this long AI-written update", "Shorten this long update")
            .replace("AI-generated", "generic")
            .replace("AI-written", "stiff")
            .replace("ChatGPT", "over-polished")
        )
        if any(w in lower_inst for w in ["robotic", "stiff", "too formal", "corporate", "cliché"]):
            seed["instruction_framing"] = "robotic_stiff"
        else:
            seed["instruction_framing"] = "plain_task"

    # Print distribution
    framing_counts = {"plain_task": 0, "robotic_stiff": 0, "explicit_ai": 0}
    for seed in task_seeds:
        f = seed["instruction_framing"]
        framing_counts[f] += 1
    print(f"Instruction framing distribution: {framing_counts}")
    
    # Save seeds
    with open("seeds/v04_task_seeds.jsonl", "w") as f:
        for seed in task_seeds:
            seed["response"] = ""
            f.write(json.dumps(seed) + "\n")
            
    print("Wrote seeds/v04_task_seeds.jsonl successfully.")

    # Save pilot seeds (first 30 seeds)
    with open("seeds/v04_task_seeds_pilot.jsonl", "w") as f:
        for seed in task_seeds[:30]:
            seed["response"] = ""
            f.write(json.dumps(seed) + "\n")
            
    print("Wrote seeds/v04_task_seeds_pilot.jsonl successfully.")

    # Save split pilot seeds
    with open("seeds/v04_task_seeds_pilot_direct.jsonl", "w") as f_dir, \
         open("seeds/v04_task_seeds_pilot_rewrite.jsonl", "w") as f_rew:
        for seed in task_seeds[:30]:
            seed["response"] = ""
            if seed["mode"] == "direct_generation":
                f_dir.write(json.dumps(seed) + "\n")
            else:
                f_rew.write(json.dumps(seed) + "\n")
                
    print("Wrote split pilot seeds successfully.")

    # Save split full seeds
    with open("seeds/v04_task_seeds_direct.jsonl", "w") as f_dir, \
         open("seeds/v04_task_seeds_rewrite.jsonl", "w") as f_rew:
        for seed in task_seeds:
            seed["response"] = ""
            if seed["mode"] == "direct_generation":
                f_dir.write(json.dumps(seed) + "\n")
            else:
                f_rew.write(json.dumps(seed) + "\n")
                
    print("Wrote split full seeds successfully.")


    # 2. 300 persona synthetic tasks for Stream C
    # Combining 30 persona descriptions with 10 generic prompt/topics
    personas = [
        # 1-8: Neutral workplace voices
        "a staff engineer writing a normal project update — clear, direct, no theatrics",
        "a product manager writing a concise status note for a cross-functional team",
        "a support lead replying to a customer with empathy and practical next steps",
        "an analyst summarizing a decision for a busy manager",
        "a researcher explaining findings to a non-technical stakeholder",
        "a manager writing a clear email that respects the reader's time",
        "a student asking for clarification without sounding stiff",
        "a customer success rep acknowledging a bug and explaining what happens next",
        # 9-16: Stylized voices
        "a startup founder who writes punchy, informal Slack messages with strong opinions",
        "a tired PM at 6pm trying to write a quick status update — direct and honest",
        "a sales rep who is warm, brief, always leaves the ball in their court",
        "a senior engineer who dislikes corporate speak and writes plainly",
        "a product manager at a Series B company who over-explains, but knows it",
        "a VC-backed founder writing to their board — transparent, no spin",
        "a customer success manager who is relentlessly positive but genuine about it",
        "a tech lead who cares about clarity and never uses passive voice",
        # 17-20: Healthcare / Professional services
        "a nurse writing a quick handoff note — clear, factual, fast",
        "a doctor writing to a patient about next steps — warm but precise",
        "a lawyer writing a brief client update — careful with words, professional",
        "a financial advisor writing a quarterly check-in — reassuring, clear",
        # 21-24: Creative / Editorial
        "a journalist writing a punchy lede for a tech story",
        "a UX writer who obsesses over clarity and hates filler words",
        "a startup marketer writing for a jaded, smart audience",
        "a technical blogger who writes for practitioners, not beginners",
        # 25-28: Operations / Logistics
        "an ops manager writing a post-incident summary — direct about what went wrong",
        "a recruiter writing a rejection email that actually feels human",
        "an office manager coordinating a team event — casual and organized",
        "a freelance consultant giving a project update — honest about delays",
        # 29-30: Personal
        "a software engineer texting a colleague a quick question",
        "a remote worker on a Friday afternoon checking in with their manager"
    ]
    
    # 10 topics/instructions
    instructions = [
        {"instruction": "Write a status update about a database migration delay due to query performance issues.", "task_type": "slack_chat", "domain": "chat"},
        {"instruction": "Draft an email requesting a peer project review before Friday.", "task_type": "email_rewrite", "domain": "email"},
        {"instruction": "Write an explanation of why we had to roll back the checkout feature release.", "task_type": "slack_chat", "domain": "chat"},
        {"instruction": "Draft a follow-up note to a client who missed our weekly check-in call.", "task_type": "email_rewrite", "domain": "email"},
        {"instruction": "Write a message asking for volunteers to participate in user testing for our new search page.", "task_type": "slack_chat", "domain": "chat"},
        {"instruction": "Draft an email outlining key action items from our Q3 planning session.", "task_type": "email_rewrite", "domain": "email"},
        {"instruction": "Write a short summary of how the server memory leak was debugged and resolved.", "task_type": "slack_chat", "domain": "chat"},
        {"instruction": "Draft an invitation to a team lunch to celebrate shipping the v2 API.", "task_type": "slack_chat", "domain": "chat"},
        {"instruction": "Write an explanation of the new password policy constraints for the engineering team.", "task_type": "email_rewrite", "domain": "email"},
        {"instruction": "Draft a reply to a client explaining a minor pricing update starting next month.", "task_type": "email_rewrite", "domain": "email"}
    ]
    
    persona_tasks = []
    for p in personas:
        for inst_obj in instructions:
            persona_tasks.append({
                "persona": p,
                "persona_description": p, # for template key compatibility
                "instruction": f"Simulate a response by {p}.\nTask: {inst_obj['instruction']}",
                "task_type": inst_obj["task_type"],
                "domain": inst_obj["domain"],
                "register": "casual" if "chat" in inst_obj["task_type"] else "professional",
                "instruction_framing": "plain_task"
            })
            
    print(f"Total persona tasks built: {len(persona_tasks)}")
    assert len(persona_tasks) == 300, f"Expected 300 persona tasks, got {len(persona_tasks)}"
    
    with open("seeds/v04_persona_tasks.jsonl", "w") as f:
        for pt in persona_tasks:
            pt["response"] = ""
            f.write(json.dumps(pt) + "\n")
            
    print("Wrote seeds/v04_persona_tasks.jsonl successfully.")

if __name__ == "__main__":
    build_seeds()
