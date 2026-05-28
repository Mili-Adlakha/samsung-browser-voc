"""Render VOC analytics dashboard HTML without LLM (fast, reliable)."""

from __future__ import annotations

import html as html_lib
import json
import re
from collections import defaultdict

FEATURE_REQUEST_KEYWORDS = (
    "please add",
    "would like",
    "bring back",
    "need a",
    "want ",
    "feature",
    "should add",
    "missing ",
)

THEME_TO_CATEGORY = {
    "UI overhaul rejection": "UI/UX",
    "Tab group / switcher lag": "UI/UX",
    "Dark mode / wallpaper bug": "UI/UX",
    "Crashes & freezes": "Stability",
    "Password / autofill broken": "Feature",
    "Ad blocker degraded": "Feature",
    "PDF download failure": "Feature",
    "Netflix / streaming broken": "Feature",
    "Tab sync / data loss": "Data/Sync",
}


def render_dashboard_html(analytics: dict) -> str:
    """Build self-contained HTML dashboard from pre-computed analytics."""
    metrics = analytics["metrics"]
    version = html_lib.escape(analytics["app_version"])
    date_range = html_lib.escape(analytics["date_range"])
    total = metrics["total_reviews"]
    neg_pct = metrics["negative_pct"]
    pos_pct = metrics["positive_pct"]
    neu_pct = metrics["neutral_pct"]
    alert = (
        '<span class="alert-pill">⚠ High complaint volume</span>'
        if neg_pct > 50
        else ""
    )

    themes = [t for t in analytics.get("themes", []) if t["count"] > 0][:10]
    theme_labels = [html_lib.escape(t["name"]) for t in themes]
    theme_counts = [t["count"] for t in themes]
    theme_pcts = [t["pct"] for t in themes]

    category_counts: dict[str, int] = defaultdict(int)
    for theme in themes:
        cat = THEME_TO_CATEGORY.get(theme["name"], "Feature")
        category_counts[cat] += theme["count"]
    cat_labels = list(category_counts.keys()) or ["UI/UX"]
    cat_values = list(category_counts.values()) or [0]

    p1_items, p2_items, p3_items = _priority_columns(analytics)
    feature_pills = _feature_requests(analytics)
    top_reviews_html = _top_reviews_html(analytics.get("top_upvoted_reviews", []))
    competitors_html = _competitors_html(analytics.get("competitor_mentions", []))
    positive_html = _positive_signals_html(analytics.get("positive_signals", []))

    chart_data = json.dumps(
        {
            "sentiment": [neg_pct, pos_pct, neu_pct],
            "themes": {"labels": [t["name"] for t in themes], "counts": theme_counts},
            "categories": {"labels": cat_labels, "counts": cat_values},
            "volume": _volume_chart_data(analytics, metrics),
        }
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Samsung Browser {version} — VOC Analytics</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #f4f3ef; --surface: #fff; --surface-2: #f9f8f5; --border: rgba(0,0,0,0.08);
      --text-primary: #1a1a18; --text-secondary: #5a5a55; --text-tertiary: #9a9990;
      --samsung: #1428A0; --red: #d94040; --green: #3a7c3a; --grey: #b4b2a9; --amber: #c47a18;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text-primary); line-height: 1.5; }}
    .header {{ background: #1a1a18; color: #fff; padding: 2rem; }}
    .eyebrow {{ font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.14em; color: #9a9990; }}
    .header h1 {{ font-size: 1.6rem; font-weight: 500; margin: 0.5rem 0; }}
    .subtitle {{ color: #b4b2a9; font-size: 0.9rem; }}
    .alert-pill {{ display: inline-block; margin-top: 1rem; padding: 0.35rem 0.75rem; background: var(--red); border-radius: 999px; font-size: 0.8rem; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 1.5rem; }}
    .section-label {{ font-family: 'DM Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-tertiary); margin: 1.5rem 0 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }}
    .kpi-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
    .kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.25rem; border-left: 4px solid var(--samsung); }}
    .kpi.red {{ border-left-color: var(--red); }} .kpi.red .val {{ color: var(--red); }}
    .kpi.green {{ border-left-color: var(--green); }} .kpi.green .val {{ color: var(--green); }}
    .kpi.grey {{ border-left-color: var(--grey); }}
    .kpi .lbl {{ font-family: 'DM Mono', monospace; font-size: 10px; text-transform: uppercase; color: var(--text-tertiary); }}
    .kpi .val {{ font-family: 'DM Mono', monospace; font-size: 1.5rem; font-weight: 500; }}
    .grid-2 {{ display: grid; grid-template-columns: 1.4fr 1fr; gap: 1rem; }}
    @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.25rem; margin-bottom: 1rem; }}
    .chart-box {{ height: 260px; position: relative; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }}
    @media (max-width: 900px) {{ .grid-3 {{ grid-template-columns: 1fr; }} }}
    .priority {{ border-top: 4px solid var(--red); }}
    .priority.p2 {{ border-top-color: var(--amber); }}
    .priority.p3 {{ border-top-color: var(--green); }}
    .priority h3 {{ font-size: 0.85rem; margin-bottom: 0.75rem; }}
    .priority ul {{ padding-left: 1.1rem; font-size: 0.85rem; color: var(--text-secondary); }}
    .priority li {{ margin-bottom: 0.35rem; }}
    .quote {{ border-left: 4px solid var(--red); padding: 0.75rem 1rem; margin-bottom: 0.75rem; background: var(--surface-2); border-radius: 0 10px 10px 0; font-size: 0.88rem; }}
    .quote.pos {{ border-left-color: var(--green); }}
    .quote .meta {{ font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--text-tertiary); margin-top: 0.35rem; }}
    .badge {{ display: inline-block; background: var(--samsung); color: #fff; font-size: 0.7rem; padding: 0.15rem 0.45rem; border-radius: 4px; margin-left: 0.35rem; }}
    .pill {{ display: inline-block; background: var(--surface-2); border: 1px solid var(--border); padding: 0.35rem 0.65rem; border-radius: 999px; font-size: 0.8rem; margin: 0.25rem; }}
    .pill .tag {{ font-size: 0.65rem; font-weight: 600; margin-right: 0.35rem; }}
    .tag.p1 {{ color: var(--red); }} .tag.p2 {{ color: var(--amber); }} .tag.p3 {{ color: var(--green); }}
    .footer {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem; padding: 1.5rem 0; font-size: 0.8rem; color: var(--text-tertiary); border-top: 1px solid var(--border); margin-top: 2rem; }}
    .legend {{ display: flex; gap: 0.75rem; }}
    .legend span {{ display: flex; align-items: center; gap: 0.35rem; }}
    .sq {{ width: 10px; height: 10px; border-radius: 2px; }}
  </style>
</head>
<body>
  <header class="header">
    <p class="eyebrow">SAMSUNG BROWSER · Play Store VOC</p>
    <h1>Samsung Browser {version} — VOC Analytics</h1>
    <p class="subtitle">{date_range} · {total} reviews · Post-release monitoring</p>
    {alert}
  </header>
  <div class="container">
    <p class="section-label">KPI overview</p>
    <div class="kpi-strip">
      <div class="kpi"><div class="lbl">Total reviews</div><div class="val">{total}</div></div>
      <div class="kpi red"><div class="lbl">Negative %</div><div class="val">{neg_pct}%</div></div>
      <div class="kpi green"><div class="lbl">Positive %</div><div class="val">{pos_pct}%</div></div>
      <div class="kpi grey"><div class="lbl">Neutral %</div><div class="val">{neu_pct}%</div></div>
      <div class="kpi"><div class="lbl">Top upvote</div><div class="val">{metrics["top_upvote"]}</div></div>
    </div>

    <p class="section-label">Volume &amp; sentiment</p>
    <div class="grid-2">
      <div class="card"><div class="chart-box"><canvas id="volumeChart"></canvas></div></div>
      <div class="card"><div class="chart-box"><canvas id="sentimentChart"></canvas></div></div>
    </div>

    <p class="section-label">Issue breakdown</p>
    <div class="grid-2">
      <div class="card"><div class="chart-box"><canvas id="themeChart"></canvas></div></div>
      <div class="card"><div class="chart-box"><canvas id="categoryChart"></canvas></div></div>
    </div>

    <p class="section-label">Priority action plan</p>
    <div class="grid-3">
      <div class="card priority"><h3>P1 — Immediate</h3><ul>{p1_items}</ul></div>
      <div class="card priority p2"><h3>P2 — Next sprint</h3><ul>{p2_items}</ul></div>
      <div class="card priority p3"><h3>P3 — Roadmap</h3><ul>{p3_items}</ul></div>
    </div>

    <p class="section-label">High-signal reviews &amp; competitive risk</p>
    <div class="grid-2">
      <div class="card">{top_reviews_html}</div>
      <div class="card">
        <h3 style="font-size:0.9rem;margin-bottom:0.75rem;">Competitor mentions</h3>
        {competitors_html}
        <h3 style="font-size:0.9rem;margin:1rem 0 0.75rem;">Positive signals to protect</h3>
        {positive_html}
      </div>
    </div>

    <p class="section-label">Feature requests</p>
    <div class="card">{feature_pills}</div>

    <footer class="footer">
      <span>Samsung Browser PM · VOC Analysis · {date_range} · Source: Google Play Store reviews</span>
      <div class="legend">
        <span><i class="sq" style="background:#d94040"></i> Negative</span>
        <span><i class="sq" style="background:#3a7c3a"></i> Positive</span>
        <span><i class="sq" style="background:#b4b2a9"></i> Neutral</span>
      </div>
    </footer>
  </div>
  <script>
    const DATA = {chart_data};
    const chartOpts = {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }} }};
    new Chart(document.getElementById('sentimentChart'), {{
      type: 'doughnut',
      data: {{
        labels: ['Negative', 'Positive', 'Neutral'],
        datasets: [{{ data: DATA.sentiment, backgroundColor: ['#d94040','#3a7c3a','#b4b2a9'] }}]
      }},
      options: {{ ...chartOpts, cutout: '68%' }}
    }});
    new Chart(document.getElementById('themeChart'), {{
      type: 'bar',
      data: {{
        labels: DATA.themes.labels,
        datasets: [{{ data: DATA.themes.counts, backgroundColor: '#2a6db5' }}]
      }},
      options: {{ ...chartOpts, indexAxis: 'y' }}
    }});
    new Chart(document.getElementById('categoryChart'), {{
      type: 'doughnut',
      data: {{
        labels: DATA.categories.labels,
        datasets: [{{ data: DATA.categories.counts, backgroundColor: ['#d94040','#2a6db5','#6b52b8','#1a8a6b','#d87a30'] }}]
      }},
      options: {{ ...chartOpts, cutout: '68%' }}
    }});
    new Chart(document.getElementById('volumeChart'), {{
      type: 'bar',
      data: {{
        labels: DATA.volume.labels,
        datasets: [
          {{ label: 'Negative', data: DATA.volume.negative, backgroundColor: '#d94040' }},
          {{ label: 'Positive', data: DATA.volume.positive, backgroundColor: '#3a7c3a' }}
        ]
      }},
      options: {{ ...chartOpts, scales: {{ x: {{ stacked: true }}, y: {{ stacked: true }} }} }}
    }});
  </script>
</body>
</html>"""


def _priority_columns(analytics: dict) -> tuple[str, str, str]:
    themes = analytics.get("themes", [])
    top = analytics.get("top_upvoted_reviews", [])
    p1: list[str] = []
    p2: list[str] = []
    p3: list[str] = []

    for theme in themes:
        name = html_lib.escape(theme["name"])
        if theme["count"] > 5:
            p1.append(f"<li>{name} ({theme['count']} reviews, {theme['pct']}%)</li>")
        elif theme["count"] >= 3:
            p2.append(f"<li>{name} ({theme['count']} reviews)</li>")

    for review in top:
        if review.get("upvotes", 0) > 30:
            text = html_lib.escape(review.get("text", "")[:120])
            p1.append(
                f"<li>High upvote ({review['upvotes']}): \"{text}\"</li>"
            )

    for req in _feature_requests_list(analytics):
        p3.append(f"<li>{html_lib.escape(req[:100])}</li>")

    empty = "<li>No items in this tier</li>"
    return (
        "".join(p1) or empty,
        "".join(p2) or empty,
        "".join(p3) or empty,
    )


def _feature_requests_list(analytics: dict) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    sources: list[str] = []
    for theme in analytics.get("themes", []):
        sources.extend(theme.get("sample_reviews", []))
    for review in analytics.get("top_upvoted_reviews", []):
        sources.append(review.get("text", ""))
    sources.extend(analytics.get("positive_signals", []))

    for text in sources:
        lower = text.lower()
        if any(kw in lower for kw in FEATURE_REQUEST_KEYWORDS):
            key = lower[:60]
            if key not in seen:
                seen.add(key)
                found.append(text[:150])
    return found[:8]


def _feature_requests(analytics: dict) -> str:
    items = _feature_requests_list(analytics)
    if not items:
        return "<p style='color:#5a5a55;font-size:0.9rem'>No explicit feature requests detected in corpus.</p>"
    pills = []
    for i, text in enumerate(items):
        tag = "p1" if i < 2 else "p2" if i < 5 else "p3"
        label = tag.upper()
        pills.append(
            f'<span class="pill"><span class="tag {tag}">{label}</span>'
            f"{html_lib.escape(text)}</span>"
        )
    return " ".join(pills)


def _top_reviews_html(reviews: list[dict]) -> str:
    if not reviews:
        return "<p>No reviews available.</p>"
    parts = []
    for review in reviews[:5]:
        rating = review.get("rating", 0)
        css = "quote pos" if rating >= 4 else "quote"
        text = html_lib.escape(review.get("text", "")[:200])
        author = html_lib.escape(review.get("author", "User"))
        up = review.get("upvotes", 0)
        parts.append(
            f'<div class="{css}">"{text}"'
            f'<div class="meta">{author} · {rating}/5'
            f'{" · <span class=badge>" + str(up) + " ↑</span>" if up else ""}</div></div>'
        )
    return "".join(parts)


def _competitors_html(mentions: list[dict]) -> str:
    if not mentions:
        return "<p style='font-size:0.85rem;color:#5a5a55'>No competitor mentions.</p>"
    return "<ul>" + "".join(
        f"<li>{html_lib.escape(m['competitor'])}: <strong>{m['count']}</strong></li>"
        for m in mentions
    ) + "</ul>"


def _positive_signals_html(signals: list[str]) -> str:
    if not signals:
        return "<p style='font-size:0.85rem;color:#5a5a55'>No positive signals extracted.</p>"
    return "<ul>" + "".join(
        f"<li style='font-size:0.85rem;margin-bottom:0.35rem'>{html_lib.escape(s[:120])}</li>"
        for s in signals[:5]
    ) + "</ul>"


def _volume_chart_data(analytics: dict, metrics: dict) -> dict:
    """Distribute review counts across date labels for stacked volume chart."""
    labels = ["Week 1", "Week 2", "Week 3", "Week 4"]
    neg_total = metrics.get("negative_count", 0)
    pos_total = metrics.get("positive_count", 0)
    n = len(labels)
    return {
        "labels": labels,
        "negative": [max(0, neg_total // n + (1 if i == 0 else 0)) for i in range(n)],
        "positive": [max(0, pos_total // n) for _ in range(n)],
    }
