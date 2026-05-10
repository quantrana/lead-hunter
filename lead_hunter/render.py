from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import RunSummary
from .utils import current_env, html_escape, safe_json_dumps


def render_dashboard(
    leads: list[dict[str, Any]],
    summary: RunSummary,
    output_path: str | Path,
    config_name: str,
    output_dir: str,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(leads, key=lambda lead: (-int(lead.get("total_score", 0)), lead.get("company_name", "")))
    industries = sorted({lead.get("industry") for lead in ranked if lead.get("industry")})
    agent_types = sorted({lead.get("recommended_agent_type") for lead in ranked if lead.get("recommended_agent_type")})
    payload = safe_json_dumps(ranked)
    cards = _summary_cards(summary)
    lead_cards = "\n".join(_lead_card(index, lead) for index, lead in enumerate(ranked, start=1))
    rows = "\n".join(_table_row(index, lead) for index, lead in enumerate(ranked, start=1))
    empty_state = ""
    if not ranked:
        empty_state = """
        <section class="rounded border border-slate-200 bg-white p-5">
          <h2 class="text-base font-semibold text-slate-900">No saved leads yet</h2>
          <p class="mt-2 text-sm text-slate-600">The agent has not saved any leads that met the configured threshold. Weak or unsupported candidates are rejected instead of padding the CSV.</p>
        </section>
        """
    elif len(ranked) < 30:
        empty_state = """
        <section class="rounded border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          Fewer than 30 leads are saved because Lead Hunter preserves the quality bar instead of padding the queue with weak companies.
        </section>
        """
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lead Hunter</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-900">
  <header class="border-b border-slate-200 bg-white">
    <div class="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
      <div class="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 class="text-2xl font-semibold tracking-normal">Lead Hunter</h1>
          <p class="mt-1 text-sm text-slate-600">Autonomous prospecting agent for Hourglass</p>
        </div>
        <dl class="grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-slate-500 md:text-right">
          <dt>Last updated</dt><dd class="font-medium text-slate-700">{html_escape(summary.last_trace_at or "Not started")}</dd>
          <dt>Output dir</dt><dd class="font-medium text-slate-700">{html_escape(output_dir)}</dd>
          <dt>Environment</dt><dd class="font-medium text-slate-700">{html_escape(current_env())}</dd>
          <dt>Config</dt><dd class="font-medium text-slate-700">{html_escape(config_name)}</dd>
        </dl>
      </div>
    </div>
  </header>

  <main class="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
    <section class="rounded border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-950">
      Lead Hunter does not send emails. It produces a human-reviewed lead queue grounded in public evidence.
    </section>

    <section class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {cards}
    </section>

    {empty_state}

    <section class="rounded border border-slate-200 bg-white p-4">
      <div class="grid gap-3 md:grid-cols-5">
        <label class="text-xs font-medium uppercase tracking-normal text-slate-500">Tier
          <select id="tierFilter" class="mt-1 w-full rounded border border-slate-300 bg-white px-2 py-2 text-sm">
            <option value="">All</option>
            <option value="A">A</option>
            <option value="B">B</option>
            <option value="C">C</option>
          </select>
        </label>
        <label class="text-xs font-medium uppercase tracking-normal text-slate-500">Industry
          <select id="industryFilter" class="mt-1 w-full rounded border border-slate-300 bg-white px-2 py-2 text-sm">
            <option value="">All</option>
            {"".join(f'<option value="{html_escape(industry)}">{html_escape(industry)}</option>' for industry in industries)}
          </select>
        </label>
        <label class="text-xs font-medium uppercase tracking-normal text-slate-500">Agent type
          <select id="agentFilter" class="mt-1 w-full rounded border border-slate-300 bg-white px-2 py-2 text-sm">
            <option value="">All</option>
            {"".join(f'<option value="{html_escape(agent)}">{html_escape(agent)}</option>' for agent in agent_types)}
          </select>
        </label>
        <label class="text-xs font-medium uppercase tracking-normal text-slate-500">Sort
          <select id="sortFilter" class="mt-1 w-full rounded border border-slate-300 bg-white px-2 py-2 text-sm">
            <option value="score">Score</option>
            <option value="company">Company</option>
            <option value="last_checked">Last checked</option>
          </select>
        </label>
        <label class="text-xs font-medium uppercase tracking-normal text-slate-500">Search
          <input id="searchFilter" class="mt-1 w-full rounded border border-slate-300 bg-white px-2 py-2 text-sm" placeholder="Company, pain, industry">
        </label>
      </div>
    </section>

    <section class="overflow-hidden rounded border border-slate-200 bg-white">
      <div class="border-b border-slate-200 px-4 py-3">
        <h2 class="text-base font-semibold">Ranked lead table</h2>
      </div>
      <div class="overflow-x-auto">
        <table class="min-w-full text-left text-sm">
          <thead class="bg-slate-100 text-xs uppercase tracking-normal text-slate-500">
            <tr>
              <th class="px-3 py-2">Rank</th>
              <th class="px-3 py-2">Score</th>
              <th class="px-3 py-2">Tier</th>
              <th class="px-3 py-2">Company</th>
              <th class="px-3 py-2">Industry</th>
              <th class="px-3 py-2">Pain</th>
              <th class="px-3 py-2">Agent</th>
              <th class="px-3 py-2">Confidence</th>
              <th class="px-3 py-2">Evidence</th>
              <th class="px-3 py-2">Trace</th>
            </tr>
          </thead>
          <tbody id="leadRows" class="divide-y divide-slate-100">{rows}</tbody>
        </table>
      </div>
    </section>

    <section id="leadCards" class="space-y-4">{lead_cards}</section>
  </main>

  <script>
    window.LEADS = {payload};
    function normalize(value) {{ return (value || '').toString().toLowerCase(); }}
    function applyFilters() {{
      const tier = document.getElementById('tierFilter').value;
      const industry = document.getElementById('industryFilter').value;
      const agent = document.getElementById('agentFilter').value;
      const query = normalize(document.getElementById('searchFilter').value);
      const sort = document.getElementById('sortFilter').value;
      const cards = Array.from(document.querySelectorAll('[data-lead-card]'));
      const rows = Array.from(document.querySelectorAll('[data-lead-row]'));
      const visibleIds = new Set();
      let data = [...window.LEADS];
      if (sort === 'company') data.sort((a, b) => normalize(a.company_name).localeCompare(normalize(b.company_name)));
      if (sort === 'last_checked') data.sort((a, b) => normalize(b.last_checked_at).localeCompare(normalize(a.last_checked_at)));
      data.forEach((lead, idx) => lead.__rank = idx + 1);
      for (const lead of data) {{
        const haystack = normalize([lead.company_name, lead.industry, lead.recommended_agent_type, (lead.pain_signals || []).join(' '), lead.why_hourglass].join(' '));
        const ok = (!tier || lead.fit_tier === tier)
          && (!industry || lead.industry === industry)
          && (!agent || lead.recommended_agent_type === agent)
          && (!query || haystack.includes(query));
        if (ok) visibleIds.add(lead.agent_trace_id);
      }}
      for (const item of cards.concat(rows)) {{
        item.style.display = visibleIds.has(item.dataset.traceId) ? '' : 'none';
      }}
    }}
    ['tierFilter','industryFilter','agentFilter','sortFilter','searchFilter'].forEach(id => {{
      document.getElementById(id).addEventListener('input', applyFilters);
    }});
  </script>
</body>
</html>
"""
    target.write_text(html, encoding="utf-8")


def _summary_cards(summary: RunSummary) -> str:
    items = [
        ("Candidates", summary.total_candidates),
        ("Researched", summary.researched_candidates),
        ("Saved leads", summary.saved_leads),
        ("A-tier", summary.a_tier_leads),
        ("B-tier", summary.b_tier_leads),
        ("Avg score", summary.average_score),
        ("Run duration", summary.run_duration),
        ("Rejected", summary.rejected_candidates),
    ]
    return "\n".join(
        f"""
        <article class="rounded border border-slate-200 bg-white p-4">
          <dt class="text-xs font-medium uppercase tracking-normal text-slate-500">{html_escape(label)}</dt>
          <dd class="mt-2 text-2xl font-semibold text-slate-950">{html_escape(value)}</dd>
        </article>
        """
        for label, value in items
    )


def _tier_classes(tier: str) -> str:
    if tier == "A":
        return "border-emerald-200 bg-emerald-50 text-emerald-900"
    if tier == "B":
        return "border-blue-200 bg-blue-50 text-blue-900"
    return "border-amber-200 bg-amber-50 text-amber-900"


def _lead_card(index: int, lead: dict[str, Any]) -> str:
    evidence_items = lead.get("evidence_items") or []
    evidence_html = "\n".join(
        f"""
        <li class="rounded border border-slate-200 bg-slate-50 p-3">
          <a class="break-words text-sm font-medium text-blue-700 underline" href="{html_escape(item.get('url'))}" target="_blank" rel="noopener noreferrer">{html_escape(item.get('title') or item.get('url'))}</a>
          <p class="mt-1 text-sm text-slate-700">{html_escape(item.get('quote_or_summary'))}</p>
          <p class="mt-1 text-xs text-slate-500">{html_escape(item.get('signal_type'))}: {html_escape(item.get('why_it_matters'))}</p>
        </li>
        """
        for item in evidence_items
    )
    website = lead.get("website") or ""
    website_link = f'<a class="text-blue-700 underline" href="{html_escape(website)}" target="_blank" rel="noopener noreferrer">{html_escape(lead.get("domain") or website)}</a>' if website else html_escape(lead.get("domain") or "")
    return f"""
    <article data-lead-card data-trace-id="{html_escape(lead.get('agent_trace_id'))}" class="rounded border border-slate-200 bg-white p-5">
      <div class="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p class="text-xs font-medium uppercase tracking-normal text-slate-500">Rank {index}</p>
          <h2 class="mt-1 text-xl font-semibold">{html_escape(lead.get('company_name'))}</h2>
          <p class="mt-1 text-sm text-slate-600">{website_link}</p>
        </div>
        <div class="flex flex-wrap gap-2">
          <span class="rounded border px-2 py-1 text-sm font-semibold {_tier_classes(str(lead.get('fit_tier')))}">{html_escape(lead.get('fit_tier'))}</span>
          <span class="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-sm font-semibold">{html_escape(lead.get('total_score'))}/100</span>
          <span class="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-sm">{html_escape(lead.get('confidence'))} confidence</span>
          <span class="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-sm">{html_escape(lead.get('evidence_grounding', 'strong'))} grounding</span>
        </div>
      </div>
      <dl class="mt-4 grid gap-3 md:grid-cols-2">
        <div><dt class="text-xs font-medium uppercase tracking-normal text-slate-500">Industry</dt><dd class="text-sm">{html_escape(lead.get('industry'))}</dd></div>
        <div><dt class="text-xs font-medium uppercase tracking-normal text-slate-500">Location</dt><dd class="text-sm">{html_escape(lead.get('location'))}</dd></div>
        <div><dt class="text-xs font-medium uppercase tracking-normal text-slate-500">Recommended agent</dt><dd class="text-sm">{html_escape(lead.get('recommended_agent_type'))}</dd></div>
        <div><dt class="text-xs font-medium uppercase tracking-normal text-slate-500">Trace ID</dt><dd class="break-all text-sm">{html_escape(lead.get('agent_trace_id'))}</dd></div>
      </dl>
      <div class="mt-4 grid gap-4 lg:grid-cols-2">
        <section>
          <h3 class="text-sm font-semibold">Pain signals</h3>
          <p class="mt-1 text-sm text-slate-700">{html_escape('; '.join(lead.get('pain_signals') or []))}</p>
          <h3 class="mt-4 text-sm font-semibold">Why Hourglass</h3>
          <p class="mt-1 text-sm text-slate-700">{html_escape(lead.get('why_hourglass'))}</p>
          <h3 class="mt-4 text-sm font-semibold">Risks or uncertainties</h3>
          <p class="mt-1 text-sm text-slate-700">{html_escape(lead.get('risks_or_uncertainties'))}</p>
        </section>
        <section>
          <h3 class="text-sm font-semibold">Outreach subject</h3>
          <p class="mt-1 text-sm text-slate-700">{html_escape(lead.get('outreach_subject'))}</p>
          <h3 class="mt-4 text-sm font-semibold">Outreach pitch</h3>
          <pre class="mt-1 whitespace-pre-wrap rounded border border-slate-200 bg-slate-50 p-3 text-sm text-slate-800">{html_escape(lead.get('outreach_pitch'))}</pre>
        </section>
      </div>
      <section class="mt-4">
        <h3 class="text-sm font-semibold">Evidence</h3>
        <ul class="mt-2 space-y-2">{evidence_html}</ul>
      </section>
    </article>
    """


def _table_row(index: int, lead: dict[str, Any]) -> str:
    return f"""
    <tr data-lead-row data-trace-id="{html_escape(lead.get('agent_trace_id'))}" class="align-top">
      <td class="px-3 py-2">{index}</td>
      <td class="px-3 py-2 font-semibold">{html_escape(lead.get('total_score'))}</td>
      <td class="px-3 py-2">{html_escape(lead.get('fit_tier'))}</td>
      <td class="px-3 py-2 font-medium">{html_escape(lead.get('company_name'))}</td>
      <td class="px-3 py-2">{html_escape(lead.get('industry'))}</td>
      <td class="max-w-sm px-3 py-2">{html_escape('; '.join(lead.get('pain_signals') or []))}</td>
      <td class="px-3 py-2">{html_escape(lead.get('recommended_agent_type'))}</td>
      <td class="px-3 py-2">{html_escape(lead.get('confidence'))}</td>
      <td class="px-3 py-2">{len(lead.get('evidence_urls') or [])}</td>
      <td class="break-all px-3 py-2 text-xs">{html_escape(lead.get('agent_trace_id'))}</td>
    </tr>
    """
