# Lead Hunter Demo Script

## 0:00-0:15

This is Lead Hunter, an autonomous prospecting agent built for Hourglass. Instead of submitting another demo agent, I wanted to build something useful: an agent that finds real Australian businesses likely to need custom AI agents.

## 0:15-0:35

Show `config.yaml`.

The only input is a config of public signal sources: RSS feeds, public job pages, company list pages, public web pages, and manual seed sources for testing. There is no inbox access, no LinkedIn login, and no private scraping.

## 0:35-0:55

Run:

```bash
python agent.py once --config config.yaml
```

Or show a completed longer run.

The agent discovers candidates, researches them from public pages, extracts evidence, scores fit for Hourglass, and decides whether to save or reject each company.

## 0:55-1:20

Show `outputs/run_log.jsonl` or a trace in the dashboard.

Here is a company it found. It pulled public evidence, identified likely workflow pain, recommended an agent type, scored the fit, and produced a short reasoning summary without exposing private chain-of-thought.

## 1:20-1:40

Show `outputs/leads.html`.

The dashboard is a review console, not a marketing page. It shows ranked leads, evidence URLs, pain signals, recommended agent type, outreach copy, risks, and trace IDs. It also says clearly that the agent is not sending emails.

## 1:40-1:55

Show `outputs/leads.csv`.

This is the artifact Hourglass could actually use: a reviewed queue of leads to call or email on Monday. Weak leads are rejected instead of padding the CSV.

## 1:55-2:00

Lead Hunter is autonomous, grounded in evidence, safe by design, and useful beyond the demo.
