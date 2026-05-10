# Submission Notes

Track: Other

Project title: Lead Hunter

One-sentence pitch: Lead Hunter autonomously discovers Australian businesses that are likely high-fit clients for Hourglass, researches them, scores fit, generates evidence-backed outreach angles, and outputs a ranked lead queue that Hourglass could actually use.

## What The Agent Does

Lead Hunter reads a config of public signal sources, discovers candidate companies, researches them from public evidence, scores fit for Hourglass, generates a concise outreach angle, and saves a ranked lead queue.

## What Makes It Autonomous

It runs continuously, chooses sources from config, calls its four tools, maintains SQLite state, dedupes companies, rejects weak leads, renders outputs, logs traces, and resumes from prior runs.

## What Makes It Ambitious

Most challenge submissions are likely to answer messages or summarize documents. Lead Hunter creates pipeline: a concrete list of businesses with evidence, pain signals, recommended agent ideas, and outreach angles.

## What Proves It Works

The output artifact is `leads.csv` plus `leads.html`. Each saved lead includes score, tier, evidence URLs, pain signals, a recommended agent type, risks, confidence, and a trace ID.

## Safety Note

Lead Hunter does not send emails or automate outreach. It uses public company-level information, avoids private or logged-in sources, and produces a human-reviewed lead queue only.

## What I Would Improve Next If Hired

I would add richer source connectors approved by Hourglass, better source-specific parsers for job boards, a review workflow for marking lead quality, and evaluation against real closed-won client patterns.
