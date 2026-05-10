# Lead Hunter

Lead Hunter autonomously discovers Australian businesses that are likely high-fit clients for Hourglass, researches them, scores fit, generates evidence-backed outreach angles, and outputs a ranked lead queue that Hourglass could actually use.

## Why This Matches The Hourglass Challenge

Hourglass sells custom AI agents. Lead Hunter builds pipeline for that exact business: it finds companies with public signals of workflow pain, researches them, scores fit, and produces a qualified lead queue.

## Track

Other.

This is not a chatbot, email sender, scraper dashboard, or RAG demo. The autonomous agent is the product, and the lead list is the receipt.

## What Makes It Autonomous

Lead Hunter runs a loop that selects sources, discovers candidates, dedupes companies, researches public evidence, scores fit, generates outreach angles, saves qualified leads, renders outputs, logs traces, and resumes from SQLite state.

The model-facing tool surface is intentionally small:

1. `discover_candidates`
2. `research_candidate`
3. `score_candidate`
4. `save_qualified_lead`

## Sources It Can Use

Configured YAML sources may be:

- RSS feeds
- public web pages
- public job board URLs
- public ATS job URLs
- public search result URLs treated as normal pages
- company list URLs
- manual company seeds for testing

Lead Hunter does not use browser automation, logged-in pages, private LinkedIn access, inbox access, or paywalled data.

## Outputs

By default, outputs are written to `outputs/`:

- `leads.csv`
- `leads.html`
- `run_log.jsonl`
- `errors.jsonl`
- `lead_hunter.sqlite`

`leads.csv` is the primary artifact. `leads.html` is a static review console.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp config.example.yaml config.yaml
```

Edit `.env`:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=your_model_here
LEAD_HUNTER_ENV=production
```

The model name is read from `OPENAI_MODEL`. The code does not hardcode a production model ID.

## Config

Edit `config.yaml` and replace placeholder source URLs with public sources you are allowed to fetch. Manual seeds are useful for smoke testing but should not be used to fake final leads.

Important run settings:

- `max_hours`
- `max_leads`
- `min_score`
- `allow_c_tier`
- `crawl_delay_seconds`
- `output_dir`

## Commands

Initialize state:

```bash
python agent.py init --config config.yaml
```

Run one short cycle:

```bash
python agent.py once --config config.yaml
```

Run a long lead hunt:

```bash
python agent.py run --config config.yaml --hours 36 --max-leads 50
```

Render dashboard from SQLite:

```bash
python agent.py render --config config.yaml
```

Show progress:

```bash
python agent.py status --config config.yaml
```

Export CSV:

```bash
python agent.py export --config config.yaml
```

Reset local state:

```bash
python agent.py reset --config config.yaml
```

## Short Demo Without API Keys

```bash
python agent.py once --config tests/fixtures/test_config.yaml --test-mode
```

Test mode uses fixture pages and a deterministic fake model client. It does not call OpenAI and does not fetch live websites.

## 24-36 Hour Lead Hunt

1. Put real public sources in `config.yaml`.
2. Set `OPENAI_API_KEY`.
3. Set `OPENAI_MODEL`.
4. Confirm the sources are public and appropriate to fetch.
5. Run:

```bash
python agent.py run --config config.yaml --hours 36 --max-leads 50
```

If fewer than 30 strong leads are found, Lead Hunter keeps the smaller list instead of padding weak leads.

## Reading `leads.csv`

The CSV is ranked by score. Key columns are:

- `total_score`
- `fit_tier`
- `company_name`
- `evidence_urls`
- `pain_signals`
- `recommended_agent_type`
- `why_hourglass`
- `outreach_subject`
- `outreach_pitch`
- `risks_or_uncertainties`
- `agent_trace_id`

Every specific outreach claim should be traceable to evidence.

## Opening `leads.html`

Open the file directly in a browser:

```bash
open outputs/leads.html
```

The dashboard is static HTML. It has summary cards, filters, a ranked table, lead cards, evidence URLs, outreach copy, and trace IDs.

## Safety And Ethics

- No emails are sent.
- No outreach is automated.
- No private, logged-in, or paywalled sources are scraped.
- No CAPTCHA or rate-limit bypass is attempted.
- Company-level information is preferred over personal data.
- Weak or unsupported leads are rejected.
- Evidence URLs are required for saved leads.

## Limitations

Public pages vary in structure, so extraction is best-effort. Real lead quality depends on source quality. The agent does not use LinkedIn, private databases, or CRM enrichment. It is designed for a human-reviewed queue, not automated selling.

## What I Would Improve Next If Hired

I would add source-specific parsers for approved job boards, a human review workflow, richer evaluation against Hourglass ICP examples, scheduled monitoring, and better analytics on why leads are accepted or rejected.
