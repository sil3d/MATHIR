"""Rebuild index.html with all 4 reports."""
import sys
from pathlib import Path
from html import escape

# Brand colors
PURPLE = "#8b5cf6"; PURPLE_LIGHT = "#a78bfa"
GREEN = "#22c55e"; DARK = "#0f172a"; GRAY = "#94a3b8"
LIGHT = "#cbd5e1"; BG = "#0a0e1a"; CARD = "#1e293b"; CARD_HOVER = "#334155"

base = Path(__file__).parent

reports = [
    {
        "file": "report_01_micro_bench_500_memories.html",
        "label": "01 — Micro-benchmark",
        "subtitle": "Memory-only stress test (500 memories, no LLM)",
        "summary": "Tests the 4 lifecycle phases on infra level: touch_recall, promote, decay, consolidate, build_links. Throughput + latency.",
        "color": "#3b82f6",
        "tag": "no-llm",
    },
    {
        "file": "report_02_ai_cognitive_15exp_10q.html",
        "label": "02 — AI cognitive (KILLER)",
        "subtitle": "15 exp × 10 Q &middot; recall@5 +52.3% after maintenance",
        "summary": "The killer test: 15 LLM-generated experiences, 10 blind Q&A, full maintenance cycle, re-test. <strong>recall@5: 0.472 → 0.719</strong>.",
        "color": GREEN,
        "tag": "headline",
    },
    {
        "file": "report_03_openrouter_free_models_verification.html",
        "label": "03 — OpenRouter free models",
        "subtitle": "26 free models tested &middot; 9/26 actually work",
        "summary": "Comprehensive test of all 26 OpenRouter free models. 9 actually respond, 7 are HTTP 429 rate-limited, 8 are multimodal-only. Latency-sorted list of working models.",
        "color": "#fbbf24",
        "tag": "verification",
    },
    {
        "file": "report_04_multi_model_4_models_swap.html",
        "label": "04 — Multi-model swap",
        "subtitle": "4 LLMs head-to-head on same A→B→C→D",
        "summary": "Proves the recall@5 improvement is <strong>LLM-agnostic</strong>. Best result: nemotron-3-nano-30b went from 0.533 to <strong>1.000</strong> (+87.6%).",
        "color": "#f472b6",
        "tag": "comparison",
    },
]

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MATHIR v8.4.0 — Benchmark Suite</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:{BG};color:{LIGHT};line-height:1.6;min-height:100vh}}
  .container{{max-width:1280px;margin:0 auto;padding:2rem}}
  h1{{font-size:2.5rem;font-weight:800;background:linear-gradient(135deg,{PURPLE_LIGHT},{GREEN});-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:.5rem}}
  h2{{font-size:1.5rem;font-weight:700;color:{LIGHT};margin:2.5rem 0 1rem;border-bottom:1px solid {CARD};padding-bottom:.5rem}}
  .subtitle{{color:{GRAY};font-size:1rem;margin-bottom:2rem}}
  .grid{{display:grid;gap:1.5rem;margin:1.5rem 0}}
  .grid-2{{grid-template-columns:repeat(2,1fr)}}
  @media (max-width:900px){{.grid-2{{grid-template-columns:1fr}}}}
  .card{{background:{CARD};border:1px solid {CARD_HOVER};border-radius:.75rem;padding:1.5rem;transition:transform .15s;border-left:4px solid var(--accent)}}
  .card:hover{{transform:translateY(-2px)}}
  h3{{font-size:1.2rem;font-weight:600;color:{PURPLE_LIGHT};margin-bottom:.5rem}}
  .card-subtitle{{font-size:.85rem;color:{GRAY};margin-bottom:.75rem}}
  .card-summary{{font-size:.9rem;color:{LIGHT};margin-bottom:1rem;line-height:1.5}}
  .footer{{margin-top:4rem;padding-top:2rem;border-top:1px solid {CARD};color:{GRAY};font-size:.85rem;text-align:center}}
  .badge{{display:inline-block;padding:.15rem .5rem;border-radius:9999px;font-size:.7rem;font-weight:600;background:{CARD_HOVER};color:{PURPLE_LIGHT};margin-right:.25rem}}
  .tag{{display:inline-block;padding:.25rem .75rem;border-radius:9999px;font-size:.75rem;font-weight:600;background:{CARD};color:{PURPLE_LIGHT};border:1px solid {CARD_HOVER};margin-right:.5rem}}
  .tag-headline{{background:rgba(34,197,94,.15);color:{GREEN};border-color:rgba(34,197,94,.3)}}
  .tag-no-llm{{background:rgba(251,191,36,.15);color:#fbbf24;border-color:rgba(251,191,36,.3)}}
  a{{color:{PURPLE_LIGHT};text-decoration:none}}
  .btn{{display:inline-block;padding:.5rem 1rem;background:{PURPLE};color:white;border-radius:.5rem;font-weight:600;font-size:.9rem;transition:background .15s}}
  .btn:hover{{background:#7c3aed}}
</style></head><body><div class="container">
<h1>MATHIR v8.4.0 — Benchmark Suite</h1>
<div class="subtitle">Four benchmarks that prove the living memory (promote / decay / consolidate / link graph) measurably improves retrieval quality. <a href="README.md">Full reproduction guide &rarr;</a></div>

<div class="subtitle">
  <span class="tag tag-headline">headline result: recall@5 +52.3%</span>
  <span class="tag">9/26 free models work</span>
  <span class="tag">4 LLMs compared</span>
  <span class="tag">147 unit tests pass</span>
</div>

<h2>Benchmarks</h2>
<div class="grid grid-2">
"""

for r in reports:
    accent = r["color"]
    tag_class = f"tag-{r['tag']}" if r['tag'] in ('headline', 'no-llm') else 'tag'
    html += f"""
<div class="card" style="--accent: {accent}">
  <h3>{escape(r['label'])}</h3>
  <div class="card-subtitle"><span class="{tag_class}">{r['tag']}</span> {escape(r['subtitle'])}</div>
  <div class="card-summary">{r['summary']}</div>
  <a class="btn" href="{escape(r['file'])}">Open report &rarr;</a>
</div>
"""

html += f"""
</div>

<h2>How to reproduce</h2>
<div class="card">
  <div class="card-summary">
    <strong>1. Setup:</strong> <code>cd benchmarks/04_lifecycle_bench &amp;&amp; cp .env.example .env</code><br>
    <strong>2. Add OpenRouter key:</strong> edit <code>.env</code>, set <code>MATHIR_API_KEY=sk-or-v1-...</code> (free at <a href="https://openrouter.ai/keys">openrouter.ai/keys</a>)<br>
    <strong>3. Run:</strong> <code>python run_all.py --duration 20</code> &mdash; runs all 4 benches in sequence (~25 min)<br>
    <strong>4. Re-render HTML:</strong> <code>python render_report.py *.json</code>
  </div>
</div>

<h2>Source files</h2>
<div class="grid grid-2">
  <div class="card">
    <h3>Python scripts</h3>
    <div class="card-summary">
      <code>micro_bench.py</code> &mdash; bench 01<br>
      <code>ai_cognitive_bench.py</code> &mdash; bench 02<br>
      <code>multi_model_bench.py</code> &mdash; bench 04<br>
      <code>render_report.py</code> &mdash; JSON &rarr; HTML (micro + AI)<br>
      <code>render_extras.py</code> &mdash; JSON &rarr; HTML (multi + openrouter)<br>
      <code>run_all.py</code> &mdash; orchestrator<br>
      <code>llm_client.py</code> &mdash; env-driven LLM client
    </div>
  </div>
  <div class="card">
    <h3>Result files</h3>
    <div class="card-summary">
      <code>01_micro_bench_500_memories.json</code><br>
      <code>02_ai_cognitive_15exp_10q.json</code><br>
      <code>03_openrouter_free_models_verification.json</code><br>
      <code>04_multi_model_4_models_swap.json</code><br>
      <code>README.md</code> &mdash; full reproduction guide
    </div>
  </div>
</div>

</div>
<div class="footer">Generated by MATHIR v8.4.0 benchmark suite &middot; <a href="https://github.com/sil3d/MATHIR">sil3d/MATHIR</a> &middot; <a href="README.md">README</a></div>
</body></html>
"""

(base / "index.html").write_text(html, encoding="utf-8")
print(f"OK: index.html ({len(html)} bytes)")
