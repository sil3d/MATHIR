"""
Generate a self-contained HTML report from a bench JSON file.

Usage:
  python render_report.py results_ai_15x10.json report_ai_15x10.html
  python render_report.py results_micro_500.json report_micro_500.html
  python render_report.py results_ai_15x10.json --title "..." --subtitle "..."
"""
import json
import sys
import argparse
from pathlib import Path
from html import escape


# Brand colors (MATHIR)
PURPLE = "#8b5cf6"
PURPLE_DARK = "#6d28d9"
PURPLE_LIGHT = "#a78bfa"
GREEN = "#22c55e"
GREEN_DARK = "#16a34a"
BLUE = "#3b82f6"
AMBER = "#fbbf24"
EMERALD = "#34d399"
SKY = "#60a5fa"
PINK = "#f472b6"
SLATE = "#64748b"
DARK = "#0f172a"
GRAY = "#94a3b8"
LIGHT = "#cbd5e1"
BG = "#0a0e1a"
CARD = "#1e293b"
CARD_HOVER = "#334155"


def fmt_float(x, digits=4):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def fmt_pct(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


def fmt_duration(s):
    try:
        s = float(s)
        if s < 1:
            return f"{s*1000:.1f}ms"
        if s < 60:
            return f"{s:.1f}s"
        return f"{s/60:.1f}min"
    except Exception:
        return str(s)


# -----------------------------------------------------------------------------
# Common HTML scaffolding
# -----------------------------------------------------------------------------

def _html_head(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --purple: {PURPLE};
    --purple-dark: {PURPLE_DARK};
    --purple-light: {PURPLE_LIGHT};
    --green: {GREEN};
    --green-dark: {GREEN_DARK};
    --blue: {BLUE};
    --amber: {AMBER};
    --emerald: {EMERALD};
    --sky: {SKY};
    --pink: {PINK};
    --slate: {SLATE};
    --dark: {DARK};
    --gray: {GRAY};
    --light: {LIGHT};
    --bg: {BG};
    --card: {CARD};
    --card-hover: {CARD_HOVER};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--light);
    line-height: 1.6;
    min-height: 100vh;
  }}
  .container {{ max-width: 1280px; margin: 0 auto; padding: 2rem; }}
  h1 {{ font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, var(--purple-light), var(--green)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.5rem; font-weight: 700; color: var(--light); margin: 2.5rem 0 1rem; border-bottom: 1px solid var(--card); padding-bottom: 0.5rem; }}
  h3 {{ font-size: 1.15rem; font-weight: 600; color: var(--purple-light); margin: 1.5rem 0 0.75rem; }}
  .subtitle {{ color: var(--gray); font-size: 1rem; margin-bottom: 2rem; }}
  .badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; background: var(--card); color: var(--purple-light); border: 1px solid var(--card-hover); margin-right: 0.5rem; }}
  .badge-green {{ background: rgba(34, 197, 94, 0.15); color: var(--green); border-color: rgba(34, 197, 94, 0.3); }}
  .badge-amber {{ background: rgba(251, 191, 36, 0.15); color: var(--amber); border-color: rgba(251, 191, 36, 0.3); }}
  .badge-blue {{ background: rgba(59, 130, 246, 0.15); color: var(--blue); border-color: rgba(59, 130, 246, 0.3); }}
  .badge-pink {{ background: rgba(244, 114, 182, 0.15); color: var(--pink); border-color: rgba(244, 114, 182, 0.3); }}
  .grid {{ display: grid; gap: 1.5rem; margin: 1.5rem 0; }}
  .grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
  .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
  .grid-4 {{ grid-template-columns: repeat(4, 1fr); }}
  .grid-5 {{ grid-template-columns: repeat(5, 1fr); }}
  @media (max-width: 900px) {{ .grid-2, .grid-3, .grid-4, .grid-5 {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--card); border: 1px solid var(--card-hover); border-radius: 0.75rem; padding: 1.5rem; transition: transform 0.15s; }}
  .card:hover {{ transform: translateY(-2px); }}
  .card-title {{ font-size: 0.85rem; color: var(--gray); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }}
  .card-value {{ font-size: 2rem; font-weight: 700; color: var(--light); }}
  .card-sub {{ font-size: 0.85rem; color: var(--gray); margin-top: 0.5rem; }}
  .card-delta {{ display: inline-block; margin-left: 0.5rem; font-size: 0.9rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 0.25rem; }}
  .delta-pos {{ background: rgba(34, 197, 94, 0.2); color: var(--green); }}
  .delta-neg {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; }}
  .delta-zero {{ background: var(--card-hover); color: var(--gray); }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--card-hover); }}
  th {{ font-size: 0.8rem; color: var(--gray); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; background: var(--card); }}
  tr:hover td {{ background: var(--card); }}
  td.num {{ font-family: 'SF Mono', 'Monaco', 'Consolas', monospace; }}
  .chart-container {{ position: relative; height: 320px; margin: 1.5rem 0; }}
  .chart-container.tall {{ height: 420px; }}
  .memory-block {{ background: rgba(139, 92, 246, 0.08); border-left: 3px solid var(--purple); padding: 1rem 1.25rem; border-radius: 0 0.5rem 0.5rem 0; margin: 0.75rem 0; }}
  .memory-content {{ color: var(--light); white-space: pre-wrap; line-height: 1.7; font-size: 0.95rem; }}
  .memory-meta {{ font-size: 0.8rem; color: var(--gray); margin-bottom: 0.5rem; }}
  .qa-block {{ background: var(--card); border: 1px solid var(--card-hover); border-radius: 0.5rem; padding: 1rem 1.25rem; margin: 0.75rem 0; }}
  .qa-q {{ color: var(--sky); font-weight: 600; margin-bottom: 0.5rem; }}
  .qa-a {{ color: var(--light); margin: 0.5rem 0; line-height: 1.6; }}
  .qa-meta {{ display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.8rem; color: var(--gray); margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid var(--card-hover); }}
  .qa-meta span {{ padding: 0.15rem 0.5rem; background: var(--bg); border-radius: 0.25rem; font-family: 'SF Mono', monospace; }}
  .phase-label {{ display: inline-block; padding: 0.4rem 1rem; border-radius: 0.25rem; font-weight: 600; font-size: 0.85rem; margin-right: 0.5rem; }}
  .phase-A {{ background: rgba(139, 92, 246, 0.2); color: var(--purple-light); }}
  .phase-B {{ background: rgba(59, 130, 246, 0.2); color: var(--blue); }}
  .phase-C {{ background: rgba(251, 191, 36, 0.2); color: var(--amber); }}
  .phase-D {{ background: rgba(34, 197, 94, 0.2); color: var(--green); }}
  .footer {{ margin-top: 4rem; padding-top: 2rem; border-top: 1px solid var(--card); color: var(--gray); font-size: 0.85rem; text-align: center; }}
  .delta-arrow {{ font-weight: 800; }}
  .no-data {{ color: var(--gray); font-style: italic; padding: 1rem; text-align: center; }}
  .highlight {{ background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(34, 197, 94, 0.2)); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid var(--purple); margin: 1.5rem 0; }}
  .highlight-num {{ font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, var(--purple-light), var(--green)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  code {{ background: var(--card); padding: 0.1rem 0.4rem; border-radius: 0.25rem; font-family: 'SF Mono', monospace; font-size: 0.9em; color: var(--green); }}
</style>
</head>
<body>
<div class="container">
"""


def _html_foot():
    return """
</div>
<div class="footer">
  Generated by MATHIR v8.5.0 benchmark suite &middot;
  <a href="https://github.com/sil3d/MATHIR" style="color: var(--purple-light);">sil3d/MATHIR</a>
</div>
</body>
</html>
"""


def _header(title: str, subtitle: str, badges: list = None):
    b = ""
    if badges:
        b = "<div style='margin: 1rem 0;'>" + "".join(f"<span class='badge {c}'>{escape(t)}</span>" for t, c in badges) + "</div>"
    return f"""
<h1>{escape(title)}</h1>
<div class="subtitle">{escape(subtitle)}</div>
{b}
"""


def _card(title: str, value: str, sub: str = "", delta: dict = None) -> str:
    delta_html = ""
    if delta:
        v = delta.get("delta", 0)
        cls = "delta-pos" if v > 0 else "delta-neg" if v < 0 else "delta-zero"
        arrow = "+" if v > 0 else ""
        delta_html = f"<span class='card-delta {cls}'>{arrow}{fmt_float(v)}</span>"
    return f"""
<div class="card">
  <div class="card-title">{escape(title)}</div>
  <div class="card-value">{value}{delta_html}</div>
  {f'<div class="card-sub">{escape(sub)}</div>' if sub else ''}
</div>
"""


# -----------------------------------------------------------------------------
# Micro-bench renderer
# -----------------------------------------------------------------------------

def render_micro_bench(data: dict, title: str) -> str:
    cfg = data.get("config", {})
    touch = data.get("touch_recall", {})
    promote = data.get("promote", {})
    decay = data.get("decay", {})
    cons = data.get("consolidate", {})
    links = data.get("link_graph", {})

    html = [_html_head(title)]
    html.append(_header(
        title,
        f"Memory-only micro-benchmark &middot; {cfg.get('count', '?')} memories &middot; "
        f"dim={cfg.get('dim', '?')} &middot; {cfg.get('duplicate_ratio', 0)*100:.0f}% duplicates planted",
        [
            ("micro-bench", "badge-blue"),
            (f"DB: {data.get('db_size_mb', 0):.2f} MB", "badge-amber"),
            (f"Seed: {fmt_duration(data.get('seed_wall_s', 0))}", "badge-pink"),
        ],
    ))

    # Summary cards
    html.append('<h2>Summary</h2><div class="grid grid-4">')
    html.append(_card("Memories stored", fmt_int(cons.get("by_tier", {}).get("episodic", 0)),
                     f"after {cons.get('merged', 0)} merges"))
    html.append(_card("touch_recall", f"{fmt_int(touch.get('ops_per_sec', 0))} ops/s",
                     f"p50={fmt_float(touch.get('latency_ms', {}).get('p50', 0), 2)}ms"))
    html.append(_card("Consolidate", fmt_int(cons.get("merged", 0)),
                     f"{cons.get('candidates', 0)} candidates at >0.95"))
    html.append(_card("BFS get_links", f"{fmt_float(links.get('bfs_get_links', {}).get('latency_ms', {}).get('p95', 0), 3)}ms",
                     f"avg {fmt_float(links.get('bfs_get_links', {}).get('avg_links_per_node', 0), 1)} links/node"))
    html.append('</div>')

    # Charts
    html.append('<h2>Phase timings</h2>')
    html.append('<div class="grid grid-2">')
    html.append('<div class="card"><h3>Throughput</h3>')
    html.append('<div class="chart-container"><canvas id="chartThroughput"></canvas></div></div>')
    html.append('<div class="card"><h3>Latency (touch_recall)</h3>')
    html.append('<div class="chart-container"><canvas id="chartLatency"></canvas></div></div>')
    html.append('</div>')

    html.append('<h2>Maintenance cycle</h2>')
    html.append('<div class="grid grid-2">')
    html.append('<div class="card"><h3>By tier after consolidate</h3>')
    html.append('<div class="chart-container"><canvas id="chartByTier"></canvas></div></div>')
    html.append('<div class="card"><h3>Maintenance op counts</h3>')
    html.append('<div class="chart-container"><canvas id="chartOps"></canvas></div></div>')
    html.append('</div>')

    # Detailed table
    html.append('<h2>Detailed metrics</h2>')
    html.append('<table><thead><tr><th>Phase</th><th>Metric</th><th>Value</th></tr></thead><tbody>')
    rows = [
        ("Seed", "wall time", fmt_duration(data.get("seed_wall_s", 0))),
        ("Seed", "DB size", f"{data.get('db_size_mb', 0):.2f} MB"),
        ("touch_recall", "ops/sec", fmt_int(touch.get("ops_per_sec", 0))),
        ("touch_recall", "p50 latency", f"{touch.get('latency_ms', {}).get('p50', 0):.2f} ms"),
        ("touch_recall", "p95 latency", f"{touch.get('latency_ms', {}).get('p95', 0):.2f} ms"),
        ("touch_recall", "p99 latency", f"{touch.get('latency_ms', {}).get('p99', 0):.2f} ms"),
        ("promote", "scanned", fmt_int(promote.get("scanned", 0))),
        ("promote", "promoted", fmt_int(promote.get("promoted", 0))),
        ("promote", "ms/memory", f"{promote.get('ms_per_mem', 0):.3f}"),
        ("decay", "scanned", fmt_int(decay.get("scanned", 0))),
        ("decay", "decayed", fmt_int(decay.get("decayed", 0))),
        ("decay", "archived", fmt_int(decay.get("archived", 0))),
        ("consolidate", "candidates", fmt_int(cons.get("candidates", 0))),
        ("consolidate", "merged", fmt_int(cons.get("merged", 0))),
        ("consolidate", "wall time", fmt_duration(cons.get("wall_s", 0))),
        ("build_links", "links created", fmt_int(links.get("build", {}).get("links_created", 0))),
        ("build_links", "memories scanned", fmt_int(links.get("build", {}).get("memories_scanned", 0))),
        ("build_links", "wall time", fmt_duration(links.get("build", {}).get("wall_s", 0))),
    ]
    for phase, metric, value in rows:
        html.append(f'<tr><td>{escape(phase)}</td><td>{escape(metric)}</td><td class="num">{escape(value)}</td></tr>')
    html.append('</tbody></table>')

    # Charts JS
    html.append(_micro_charts_js(data))
    html.append(_html_foot())
    return "".join(html)


def _micro_charts_js(data: dict) -> str:
    touch = data.get("touch_recall", {})
    cons = data.get("consolidate", {})
    decay = data.get("decay", {})
    links = data.get("link_graph", {})
    promote = data.get("promote", {})

    # Pre-compute numbers to avoid f-string brace issues
    seed_tp = data.get('count', 0) / max(data.get('seed_wall_s', 1), 0.001)
    promote_tp = promote.get('scanned', 0) / max(promote.get('wall_s', 0.001), 0.001)
    cons_tp = cons.get('merged', 0) / max(cons.get('wall_s', 0.001), 0.001)
    link_tp = links.get('build', {}).get('links_created', 0) / max(links.get('build', {}).get('wall_s', 0.001), 0.001)

    p50 = touch.get('latency_ms', {}).get('p50', 0)
    p95 = touch.get('latency_ms', {}).get('p95', 0)
    p99 = touch.get('latency_ms', {}).get('p99', 0)

    bt = cons.get('by_tier', {})
    bt_labels = list(bt.keys())
    bt_values = list(bt.values())

    return """
<script>
new Chart(document.getElementById('chartThroughput'), {
  type: 'bar',
  data: {
    labels: ['Seed', 'touch_recall', 'auto_promote', 'consolidate', 'build_links'],
    datasets: [{
      label: 'Throughput (ops/s or mem/s)',
      data: [""" + f"{seed_tp:.0f}, {touch.get('ops_per_sec', 0):.0f}, {promote_tp:.0f}, {cons_tp:.0f}, {link_tp:.0f}" + """],
      backgroundColor: ['""" + PURPLE + """', '""" + GREEN + """', '""" + AMBER + """', '""" + BLUE + """', '""" + PINK + """'],
      borderWidth: 1
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '""" + LIGHT + """' } } },
    scales: {
      x: { ticks: { color: '""" + GRAY + """' } },
      y: { ticks: { color: '""" + GRAY + """' }, beginAtZero: true }
    }
  }
});

new Chart(document.getElementById('chartLatency'), {
  type: 'bar',
  data: {
    labels: ['p50', 'p95', 'p99'],
    datasets: [{
      label: 'touch_recall latency (ms)',
      data: [""" + f"{p50:.2f}, {p95:.2f}, {p99:.2f}" + """],
      backgroundColor: ['""" + GREEN + """', '""" + AMBER + """', '""" + PINK + """'],
      borderWidth: 1
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '""" + LIGHT + """' } } },
    scales: {
      x: { ticks: { color: '""" + GRAY + """' } },
      y: { ticks: { color: '""" + GRAY + """' }, beginAtZero: true }
    }
  }
});

new Chart(document.getElementById('chartByTier'), {
  type: 'doughnut',
  data: {
    labels: """ + str(bt_labels) + """,
    datasets: [{
      data: """ + str(bt_values) + """,
      backgroundColor: ['""" + PURPLE + """', '""" + GREEN + """', '""" + AMBER + """', '""" + BLUE + """', '""" + PINK + """'],
      borderColor: '""" + CARD + """',
      borderWidth: 2
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'right', labels: { color: '""" + LIGHT + """' } } }
  }
});

new Chart(document.getElementById('chartOps'), {
  type: 'bar',
  data: {
    labels: ['decay', 'promote', 'consolidate', 'build_links'],
    datasets: [
      { label: 'decayed', data: [""" + f"{decay.get('decayed', 0)}, 0, 0, 0" + """], backgroundColor: '""" + AMBER + """' },
      { label: 'archived', data: [""" + f"{decay.get('archived', 0)}, 0, 0, 0" + """], backgroundColor: '""" + PINK + """' },
      { label: 'promoted', data: [0, """ + f"{promote.get('promoted', 0)}" + """, 0, 0], backgroundColor: '""" + PURPLE + """' },
      { label: 'merged', data: [0, 0, """ + f"{cons.get('merged', 0)}" + """, 0], backgroundColor: '""" + GREEN + """' },
      { label: 'links_created', data: [0, 0, 0, """ + f"{links.get('build', {}).get('links_created', 0)}" + """], backgroundColor: '""" + BLUE + """' }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '""" + LIGHT + """' } } },
    scales: {
      x: { stacked: true, ticks: { color: '""" + GRAY + """' } },
      y: { stacked: true, ticks: { color: '""" + GRAY + """' }, beginAtZero: true }
    }
  }
});
</script>
"""


# -----------------------------------------------------------------------------
# AI cognitive bench renderer
# -----------------------------------------------------------------------------

def render_ai_bench(data: dict, title: str) -> str:
    cfg = data.get("config", {})
    phA = data.get("phase_A", {})
    phB = data.get("phase_B_baseline", {}).get("metrics_summary", {})
    phC = data.get("phase_C_maintenance", {})
    phD = data.get("phase_D_after", {}).get("metrics_summary", {})
    comp = data.get("comparison", {})

    html = [_html_head(title)]
    html.append(_header(
        title,
        f"AI cognitive benchmark &middot; {cfg.get('experiences', '?')} experiences &middot; "
        f"{cfg.get('questions', '?')} questions &middot; "
        f"backend={cfg.get('llm_backend', '?')} &middot; seed={cfg.get('seed', '?')}",
        [
            ("ai-cognitive", "badge-blue"),
            (f"Backend: {cfg.get('llm_backend', '?')}", "badge-pink"),
            (f"Seed: {cfg.get('seed', '?')}", "badge-amber"),
        ],
    ))

    # HIGHLIGHT: the killer result
    if comp.get("recall_at_5", {}).get("delta", 0) > 0:
        delta_pct = (comp["recall_at_5"]["delta"] / max(comp["recall_at_5"]["before"], 0.001)) * 100
        html.append(f'''
<div class="highlight">
  <div class="card-title" style="color: var(--green);">⭐ HEADLINE RESULT</div>
  <div style="display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;">
    <div>
      <div style="font-size: 0.85rem; color: var(--gray);">recall@5 BEFORE</div>
      <div class="highlight-num" style="font-size: 2.5rem;">{fmt_float(comp["recall_at_5"]["before"], 3)}</div>
    </div>
    <div class="delta-arrow" style="font-size: 2rem; color: var(--green);">&rarr;</div>
    <div>
      <div style="font-size: 0.85rem; color: var(--gray);">recall@5 AFTER</div>
      <div class="highlight-num" style="font-size: 2.5rem;">{fmt_float(comp["recall_at_5"]["after"], 3)}</div>
    </div>
    <div style="margin-left: auto;">
      <div style="font-size: 0.85rem; color: var(--gray);">IMPROVEMENT</div>
      <div class="highlight-num" style="font-size: 2.5rem;">+{delta_pct:.1f}%</div>
    </div>
  </div>
  <div style="margin-top: 1rem; color: var(--light);">
    The memory lifecycle (consolidate + build_links) measurably improved retrieval quality.
    After 30 days of simulated aging + maintenance, the system finds <strong>{delta_pct:.0f}% more relevant snippets</strong> in top-5.
  </div>
</div>
''')

    # Summary cards
    html.append('<h2>Phase overview</h2><div class="grid grid-4">')
    html.append(_card("Generated", fmt_int(phA.get("experiences_generated", 0)),
                     f"{phA.get('memories_stored', 0)} memories stored"))
    html.append(_card("Baseline recall@5", fmt_float(phB.get("recall_at_5_mean", 0), 3),
                     f"MRR={fmt_float(phB.get('mrr_mean', 0), 2)}"))
    html.append(_card("Maintenance", fmt_int(phC.get("consolidate", {}).get("result", {}).get("merged", 0)),
                     f"links={phC.get('build_links', {}).get('result', {}).get('links_created', 0)}"))
    html.append(_card("After recall@5", fmt_float(phD.get("recall_at_5_mean", 0), 3),
                     f"MRR={fmt_float(phD.get('mrr_mean', 0), 2)}"))
    html.append('</div>')

    # Comparison table
    html.append('<h2>Before vs after maintenance</h2>')
    html.append('<table><thead><tr><th>Metric</th><th>Before (B)</th><th>After (D)</th><th>Delta</th><th>%</th></tr></thead><tbody>')
    metric_labels = {
        "recall_at_5": "recall@5",
        "precision_at_5": "precision@5",
        "mrr": "MRR",
        "has_answer_rate": "has_answer rate",
    }
    for key, label in metric_labels.items():
        if key not in comp:
            continue
        v = comp[key]
        before = v.get("before", 0)
        after = v.get("after", 0)
        delta = v.get("delta", 0)
        pct = (delta / max(before, 0.001)) * 100
        delta_cls = "delta-pos" if delta > 0 else "delta-neg" if delta < 0 else "delta-zero"
        arrow = "+" if delta > 0 else ""
        before_str = fmt_float(before, 4) if "rate" not in key else fmt_pct(before)
        after_str = fmt_float(after, 4) if "rate" not in key else fmt_pct(after)
        delta_str = f"{arrow}{fmt_float(delta, 4)}" if "rate" not in key else f"{arrow}{fmt_pct(delta)}"
        pct_str = f"{arrow}{pct:.1f}%"
        html.append(f'<tr><td>{label}</td><td class="num">{before_str}</td><td class="num">{after_str}</td>'
                    f'<td class="num"><span class="card-delta {delta_cls}">{delta_str}</span></td>'
                    f'<td class="num">{pct_str}</td></tr>')
    html.append('</tbody></table>')

    # Charts
    html.append('<h2>Recall quality</h2>')
    html.append('<div class="grid grid-2">')
    html.append('<div class="card"><h3>Recall metrics: baseline vs after</h3>')
    html.append('<div class="chart-container"><canvas id="chartRecall"></canvas></div></div>')
    html.append('<div class="card"><h3>Maintenance operation counts</h3>')
    html.append('<div class="chart-container"><canvas id="chartMaint"></canvas></div></div>')
    html.append('</div>')

    # Q&A diff (first 5)
    base_ans = data.get("phase_B_baseline", {}).get("answers", [])
    after_ans = data.get("phase_D_after", {}).get("answers", [])
    if base_ans and after_ans:
        html.append('<h2>Q&amp;A: baseline vs after maintenance</h2>')
        n = min(5, len(base_ans), len(after_ans))
        for i in range(n):
            b = base_ans[i]
            a = after_ans[i]
            html.append('<div class="qa-block">')
            html.append(f'<div class="qa-q">Q{i+1}: {escape(b.get("q", ""))}</div>')
            html.append(f'<div class="qa-meta">'
                        f'<span>target topic: {escape(b.get("topic", ""))}</span></div>')
            html.append('<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.5rem;">')
            # Before
            html.append('<div><div style="color: var(--blue); font-size: 0.8rem; font-weight: 600; margin-bottom: 0.25rem;">BASELINE</div>')
            html.append(f'<div class="qa-a">{escape(b.get("a", "")[:600])}</div>')
            bm = b.get("metrics", {})
            html.append(f'<div class="qa-meta">'
                        f'<span>recall@5={fmt_float(bm.get("recall_at_5", 0), 3)}</span>'
                        f'<span>precision@5={fmt_float(bm.get("precision_at_5", 0), 2)}</span>'
                        f'<span>MRR={fmt_float(bm.get("mrr", 0), 2)}</span>'
                        f'<span>top1={bm.get("top1_topic_match", False)}</span></div>')
            html.append('</div>')
            # After
            html.append('<div><div style="color: var(--green); font-size: 0.8rem; font-weight: 600; margin-bottom: 0.25rem;">AFTER</div>')
            html.append(f'<div class="qa-a">{escape(a.get("a", "")[:600])}</div>')
            am = a.get("metrics", {})
            html.append(f'<div class="qa-meta">'
                        f'<span>recall@5={fmt_float(am.get("recall_at_5", 0), 3)}</span>'
                        f'<span>precision@5={fmt_float(am.get("precision_at_5", 0), 2)}</span>'
                        f'<span>MRR={fmt_float(am.get("mrr", 0), 2)}</span>'
                        f'<span>top1={am.get("top1_topic_match", False)}</span></div>')
            html.append('</div>')
            html.append('</div>')
            html.append('</div>')

    # Footer
    html.append(_html_foot())

    # Charts JS (last, just before </body>)
    return _inject_ai_charts_js("".join(html), data)


def _inject_ai_charts_js(html: str, data: dict) -> str:
    phB = data.get("phase_B_baseline", {}).get("metrics_summary", {})
    phD = data.get("phase_D_after", {}).get("metrics_summary", {})
    phC = data.get("phase_C_maintenance", {})

    b_data = [
        phB.get('recall_at_5_mean', 0),
        phB.get('precision_at_5_mean', 0),
        phB.get('mrr_mean', 0),
        phB.get('has_answer_rate', 0),
        phB.get('top1_topic_match_rate', 0),
    ]
    d_data = [
        phD.get('recall_at_5_mean', 0),
        phD.get('precision_at_5_mean', 0),
        phD.get('mrr_mean', 0),
        phD.get('has_answer_rate', 0),
        phD.get('top1_topic_match_rate', 0),
    ]
    maint_data = [
        phC.get('decay', {}).get('result', {}).get('decayed', 0),
        phC.get('decay', {}).get('result', {}).get('archived', 0),
        phC.get('promote', {}).get('promoted', 0),
        phC.get('consolidate', {}).get('result', {}).get('merged', 0),
        phC.get('build_links', {}).get('result', {}).get('links_created', 0),
    ]

    b_str = ", ".join(f"{v:.4f}" for v in b_data)
    d_str = ", ".join(f"{v:.4f}" for v in d_data)
    m_str = ", ".join(str(v) for v in maint_data)

    charts = """
<script>
new Chart(document.getElementById('chartRecall'), {
  type: 'bar',
  data: {
    labels: ['recall@5', 'precision@5', 'MRR', 'has_answer_rate', 'top1_match'],
    datasets: [
      {
        label: 'Baseline (before maintenance)',
        data: [""" + b_str + """],
        backgroundColor: '""" + BLUE + """',
        borderWidth: 1
      },
      {
        label: 'After maintenance',
        data: [""" + d_str + """],
        backgroundColor: '""" + GREEN + """',
        borderWidth: 1
      }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '""" + LIGHT + """' } } },
    scales: {
      x: { ticks: { color: '""" + GRAY + """' } },
      y: { ticks: { color: '""" + GRAY + """' }, min: 0, max: 1.05 }
    }
  }
});

new Chart(document.getElementById('chartMaint'), {
  type: 'bar',
  data: {
    labels: ['decayed', 'archived', 'promoted', 'merged', 'links'],
    datasets: [{
      label: 'Maintenance operations',
      data: [""" + m_str + """],
      backgroundColor: ['""" + AMBER + """', '""" + PINK + """', '""" + PURPLE + """', '""" + GREEN + """', '""" + BLUE + """'],
      borderWidth: 1
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false, indexAxis: 'y',
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '""" + GRAY + """' }, beginAtZero: true },
      y: { ticks: { color: '""" + GRAY + """' } }
    }
  }
});
</script>
"""
    return html.replace(_html_foot(), charts + _html_foot())


# -----------------------------------------------------------------------------
# Combined index page
# -----------------------------------------------------------------------------

def render_index(reports: list, title: str = "MATHIR v8.5.0 — Benchmark Suite") -> str:
    """reports = list of (filename, label, summary_line)"""
    html = [_html_head(title)]
    html.append(_header(
        title,
        "All v8.5.0 lifecycle benchmarks in one place &middot; click a report to open",
        [("v8.5.0", "badge-blue"), ("living memory", "badge-green")],
    ))

    html.append('<div class="grid grid-3">')
    for fname, label, summary in reports:
        card = (
            '<div class="card">'
            '<h3>' + escape(label) + '</h3>'
            '<p style="color: var(--gray); margin: 0.5rem 0;">' + escape(summary) + '</p>'
            '<a href="' + escape(fname) + '" style="display: inline-block; margin-top: 1rem; padding: 0.5rem 1rem; background: var(--purple); color: white; border-radius: 0.5rem; text-decoration: none; font-weight: 600;">Open report &rarr;</a>'
            '</div>'
        )
        html.append(card)
    html.append('</div>')
    html.append(_html_foot())
    return "".join(html)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="Path to bench JSON file")
    ap.add_argument("out", nargs="?", help="Output HTML file (default: <json>.html)")
    ap.add_argument("--type", choices=["auto", "micro", "ai"], default="auto",
                    help="Report type (default: auto-detect from JSON structure)")
    ap.add_argument("--title", help="Override report title")
    args = ap.parse_args()

    json_path = Path(args.json)
    out_path = Path(args.out) if args.out else json_path.with_suffix(".html")

    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Auto-detect
    report_type = args.type
    if report_type == "auto":
        if "phase_A" in data and "phase_B_baseline" in data:
            report_type = "ai"
        elif "touch_recall" in data and "config" in data and "count" in data.get("config", {}):
            report_type = "micro"
        else:
            raise SystemExit("Cannot auto-detect report type. Pass --type micro or --type ai")

    title = args.title or f"MATHIR v8.5.0 benchmark — {json_path.stem}"

    if report_type == "micro":
        html = render_micro_bench(data, title)
    else:
        html = render_ai_bench(data, title)

    out_path.write_text(html, encoding="utf-8")
    print(f"OK: {out_path} ({len(html)} bytes, {report_type} report)")


if __name__ == "__main__":
    main()
