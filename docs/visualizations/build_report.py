"""
MATHIR V7.2 — Build Self-Contained HTML Report
==============================================

Generates `visual_report.html` with all 8 PNG diagrams embedded as base64
so the file is fully self-contained and portable for the master's defense.

Run:
    python visualizations/build_report.py

This script:
  1. Calls `generate_diagrams.main()` if PNGs are missing.
  2. Loads each PNG, base64-encodes it, and injects it into the HTML template.
  3. Writes the result to `visualizations/visual_report.html`.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).resolve().parent
HTML_OUT = OUTPUT_DIR / "visual_report.html"

DIAGRAMS = [
    {
        "id":    "architecture",
        "file":  "01_architecture_main.png",
        "title": "1. Hierarchical Architecture",
        "section": "Architecture",
        "subtitle": "4-tier memory + KL-constrained router",
        "description": (
            "MATHIR is a 4-tier memory system sandwiched between any "
            "modality encoder (text, image, audio, video) and any LLM. "
            "Working memory holds the last N steps; episodic memory stores "
            "past experiences; semantic memory clusters concepts; and "
            "immunological memory detects anomalies. A KL-constrained router "
            "fuses them without collapsing into a single mode."
        ),
    },
    {
        "id":    "tiers",
        "file":  "02_4_memory_tiers.png",
        "title": "2. The 4 Memory Tiers",
        "section": "Architecture",
        "subtitle": "Working · Episodic · Semantic · Immunological",
        "description": (
            "Each tier has its own capacity, update rate, and retrieval "
            "algorithm. Working memory (64 slots, O(1) ring buffer) feeds "
            "short-term attention. Episodic memory (1000 slots, cosine k-NN) "
            "stores unique events. Semantic memory (256 prototypes, online "
            "k-means) compresses recurring patterns. Immunological memory "
            "(100 patterns, Mahalanobis distance) flags out-of-distribution "
            "inputs — proven Neyman-Pearson optimal (Theorem 4)."
        ),
    },
    {
        "id":    "retrieval",
        "file":  "03_retrieval_comparison.png",
        "title": "3. Retrieval Quality Comparison",
        "section": "V7.1 Research",
        "subtitle": "5 systems · 50 domain queries · 200 chunks (White, Fluid Mechanics)",
        "description": (
            "We benchmarked 5 retrieval strategies on a real textbook "
            "corpus. The default V7 retrieval (64-dim projection) "
            "underperforms at 19.7%. Approach D — a hybrid of BM25 sparse "
            "recall, dense embeddings, and a cross-encoder reranker — wins "
            "at 45.7%, a +14.1 percentage point improvement over FAISS."
        ),
        "math": (
            r"\mathrm{quality}(D) = \frac{|\mathrm{top}_5(D) \cap \mathrm{GT}|}{5} = 45.7\%"
        ),
    },
    {
        "id":    "pareto",
        "file":  "04_latency_quality_tradeoff.png",
        "title": "4. Speed–Quality Pareto Frontier",
        "section": "V7.1 Research",
        "subtitle": "Lower-left = fast & poor · Upper-right = slow & rich",
        "description": (
            "There is no free lunch. FAISS raw vectors are 0.05 ms but cap "
            "at 31.6% quality. Approach D reaches 45.7% but takes 494 ms. "
            "Approach A (raw embedding bypass) sits at 31.6% / 1.54 ms — "
            "the recommended online default. For batch or RAG workloads "
            "where every percentage point matters, D is the answer."
        ),
        "math": (
            r"\mathrm{recommend}(\text{default}) = \arg\max_{s} \frac{\mathrm{quality}(s)}{\log(\mathrm{latency}(s))}"
        ),
    },
    {
        "id":    "stress",
        "file":  "05_multi_agent_stress.png",
        "title": "5. Multi-Agent Concurrent Stores",
        "section": "Robustness",
        "subtitle": "20 parallel agents · zero write conflicts",
        "description": (
            "We stress-tested MATHIR with up to 20 concurrent agents calling "
            "store() in parallel. Every scenario reached 100% success and "
            "zero write conflicts. Throughput scales ~17× from 1 to 20 "
            "agents (1.98 → 33.6 QPS). Median store latency drops from "
            "504 ms to 38 ms under load because the LRU cache warms up."
        ),
    },
    {
        "id":    "multimodal",
        "file":  "06_multimodal_fusion.png",
        "title": "6. Multi-Modal Fusion",
        "section": "Capabilities",
        "subtitle": "Text · Image · Audio · Video → shared embedding space",
        "description": (
            "MATHIR is modality-agnostic. It only sees fixed-dim vectors. "
            "BERT encodes text, CLIP encodes images, Whisper encodes audio, "
            "VideoCLIP encodes video. All four land in the same shared "
            "embedding space, which MATHIR then indexes across its 4 tiers."
        ),
    },
    {
        "id":    "theorems",
        "file":  "07_theorem_network.png",
        "title": "7. Theoretical Foundations (6 Theorems)",
        "section": "Theory",
        "subtitle": "Each theorem reduces to a classical result — fully proven",
        "description": (
            "MATHIR's correctness rests on 6 formal theorems, each proven in "
            "<code>docs/PROOFS.md</code> by reduction to a classical result. "
            "Theorem 1 uses Shannon's AWGN capacity; Theorem 2 uses "
            "Hoeffding concentration; Theorem 3 uses Robbins-Monro; "
            "Theorem 4 uses Neyman-Pearson; Theorem 5 uses Olshausen-Field "
            "sparse coding; Theorem 6 uses Sinkhorn-Knopp convergence."
        ),
        "math": (
            r"\Pr(\mathrm{Acc}(K) \geq 1 - CKL\eta/N) \geq 1 - \exp(-N/2) \quad\text{(Theorem 2)}"
        ),
    },
    {
        "id":    "timeline",
        "file":  "08_version_timeline.png",
        "title": "8. Version Evolution (V1 → V7.2)",
        "section": "Project History",
        "subtitle": "5 months · 8 releases · 62+ unit tests",
        "description": (
            "From a 3-tier CNN+MLP prototype (V1, January 2026) to a "
            "doctoral-grade memory system with 8 novel algorithms, "
            "6 formal theorems, and 4 retrieval approaches (V7.2, June 2026). "
            "V7.2 is the current release, with a 5–12× latency speedup via "
            "an LRU result cache and adaptive re-ranking."
        ),
    },
]


def _png_to_data_uri(path: Path) -> str:
    """Encode a PNG file as a base64 data URI for inline embedding."""
    if not path.exists():
        return ""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _ensure_diagrams() -> None:
    """Run the diagram generator if any PNG is missing."""
    missing = [d for d in DIAGRAMS if not (OUTPUT_DIR / d["file"]).exists()]
    if missing:
        print(f"[build_report] {len(missing)} PNG(s) missing — running generator...")
        try:
            import generate_diagrams
            generate_diagrams.main()
        except Exception as e:
            print(f"[build_report] Generator failed: {e}", file=sys.stderr)
            raise


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MATHIR V7.2 — Visual Architecture Report</title>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
  :root {{
    --primary:    #1f4e79;
    --secondary:  #d97706;
    --tertiary:   #059669;
    --danger:     #dc2626;
    --muted:      #64748b;
    --bg:         #f8fafc;
    --paper:      #ffffff;
    --border:     #e2e8f0;
    --ink:        #0f172a;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, Oxygen, Ubuntu, sans-serif;
    color: var(--ink);
    background: var(--bg);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}
  a {{ color: var(--primary); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  /* ============== Layout ============== */
  .layout {{ display: grid; grid-template-columns: 240px 1fr; min-height: 100vh; }}
  @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} }}

  /* ============== Sidebar ============== */
  aside {{
    background: linear-gradient(180deg, #0f172a 0%, #1e3a8a 100%);
    color: #e2e8f0;
    padding: 24px 16px;
    position: sticky; top: 0; height: 100vh; overflow-y: auto;
  }}
  aside h1 {{
    font-size: 18px; margin: 0 0 4px 0; color: #fff;
    letter-spacing: 0.5px;
  }}
  aside .ver {{ font-size: 11px; color: #fbbf24; margin-bottom: 18px;
                font-weight: 600; letter-spacing: 0.8px; }}
  aside nav a {{
    display: block; padding: 8px 10px; color: #cbd5e1; border-radius: 6px;
    font-size: 13px; margin: 2px 0; transition: all 0.15s ease;
  }}
  aside nav a:hover {{ background: rgba(255,255,255,0.08); color: #fff;
                       text-decoration: none; }}
  aside nav a.section {{ color: #fbbf24; font-weight: 600; margin-top: 12px;
                         font-size: 11px; letter-spacing: 0.5px;
                         text-transform: uppercase; }}
  aside nav a.section:hover {{ background: none; }}
  aside .footer {{
    margin-top: 32px; padding-top: 16px; border-top: 1px solid #334155;
    font-size: 10px; color: #94a3b8; line-height: 1.5;
  }}

  /* ============== Main ============== */
  main {{ padding: 0; max-width: 1100px; }}
  .cover {{
    background: linear-gradient(135deg, #1f4e79 0%, #1e3a8a 60%, #d97706 100%);
    color: #fff; padding: 80px 64px; min-height: 60vh;
    display: flex; flex-direction: column; justify-content: center;
  }}
  .cover .tag {{
    display: inline-block; background: rgba(255,255,255,0.15);
    padding: 6px 12px; border-radius: 999px; font-size: 11px;
    letter-spacing: 1.5px; font-weight: 600; margin-bottom: 24px;
    width: fit-content;
  }}
  .cover h1 {{
    font-size: 48px; margin: 0 0 12px 0; line-height: 1.1;
    font-weight: 800; letter-spacing: -1px;
  }}
  .cover .sub {{
    font-size: 20px; color: #fbbf24; font-weight: 500; margin-bottom: 32px;
  }}
  .cover .meta {{
    display: flex; gap: 32px; flex-wrap: wrap;
    font-size: 13px; color: #cbd5e1;
  }}
  .cover .meta strong {{ color: #fff; display: block; font-size: 16px;
                         font-weight: 700; margin-top: 2px; }}
  .toc {{
    background: #fff; border: 1px solid var(--border); border-radius: 12px;
    padding: 32px; margin: 32px 64px;
  }}
  .toc h2 {{ margin-top: 0; color: var(--primary); font-size: 22px; }}
  .toc ol {{ padding-left: 20px; }}
  .toc li {{ margin: 8px 0; }}
  .toc a {{ font-weight: 500; }}

  .section-block {{
    padding: 48px 64px; border-bottom: 1px solid var(--border);
    background: var(--paper);
  }}
  .section-block:nth-of-type(even) {{ background: #fafbfc; }}
  .section-block h2 {{
    font-size: 28px; color: var(--primary); margin: 0 0 6px 0;
    border-left: 4px solid var(--secondary); padding-left: 14px;
  }}
  .section-block .subtitle {{
    font-style: italic; color: var(--muted); margin: 0 0 24px 14px;
    font-size: 14px;
  }}
  .section-block h3 {{
    font-size: 18px; color: var(--ink); margin: 24px 0 8px 0;
  }}
  .section-block p {{ font-size: 15px; line-height: 1.7; color: #1e293b; }}

  .figure-card {{
    background: #fff; border: 1px solid var(--border); border-radius: 12px;
    padding: 16px; margin: 16px 0 24px 0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  }}
  .figure-card img {{
    width: 100%; height: auto; display: block; border-radius: 8px;
    background: #fff;
  }}
  .figure-card .fig-caption {{
    text-align: center; color: var(--muted); font-size: 12px;
    margin-top: 12px; font-style: italic;
  }}

  .math-block {{
    background: #fffbeb; border-left: 4px solid var(--secondary);
    padding: 16px 20px; border-radius: 6px; margin: 16px 0;
    font-size: 16px; overflow-x: auto;
  }}

  .pill {{
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.4px;
    margin-right: 6px;
  }}
  .pill.primary   {{ background: #dbeafe; color: #1e3a8a; }}
  .pill.secondary {{ background: #ffedd5; color: #9a3412; }}
  .pill.tertiary  {{ background: #d1fae5; color: #065f46; }}
  .pill.danger    {{ background: #fee2e2; color: #991b1b; }}
  .pill.gold      {{ background: #fef3c7; color: #92400e; }}

  .summary-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px; margin: 24px 0;
  }}
  .stat {{
    background: #fff; border: 1px solid var(--border); border-radius: 10px;
    padding: 16px;
  }}
  .stat .n {{
    font-size: 28px; font-weight: 800; color: var(--primary);
    line-height: 1.0; margin-bottom: 4px;
  }}
  .stat .l {{ font-size: 12px; color: var(--muted); }}

  footer {{
    text-align: center; padding: 32px 16px; color: var(--muted);
    font-size: 12px; border-top: 1px solid var(--border);
    background: #fff;
  }}
  footer strong {{ color: var(--primary); }}

  /* ============== Print ============== */
  @media print {{
    @page {{ size: A4; margin: 18mm 14mm; }}
    body {{ background: #fff; }}
    aside {{ display: none; }}
    .layout {{ display: block; }}
    .cover {{ background: #1f4e79 !important; -webkit-print-color-adjust: exact;
              print-color-adjust: exact; padding: 32px 24px;
              page-break-after: always; min-height: auto; }}
    .cover h1 {{ font-size: 32px; }}
    .toc {{ page-break-after: always; margin: 16px 0; }}
    .section-block {{ padding: 16px 0; page-break-inside: avoid;
                      border-bottom: 1px solid #e2e8f0; }}
    .section-block h2 {{ font-size: 20px; }}
    .figure-card {{ box-shadow: none; page-break-inside: avoid;
                    border: 1px solid #cbd5e1; }}
    footer {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="layout">

<aside>
  <h1>MATHIR V7.2</h1>
  <div class="ver">VISUAL ARCHITECTURE REPORT</div>
  <nav>
    <a class="section" href="#cover">Cover</a>
    <a href="#toc">Table of contents</a>
    <a class="section" href="#arch">Architecture</a>
    <a href="#architecture">1 · Hierarchical</a>
    <a href="#tiers">2 · 4 Memory Tiers</a>
    <a class="section" href="#research">V7.1 Research</a>
    <a href="#retrieval">3 · Retrieval Quality</a>
    <a href="#pareto">4 · Latency Pareto</a>
    <a class="section" href="#robustness">Robustness</a>
    <a href="#stress">5 · Multi-Agent Stress</a>
    <a class="section" href="#capabilities">Capabilities</a>
    <a href="#multimodal">6 · Multi-Modal</a>
    <a class="section" href="#theory">Theory</a>
    <a href="#theorems">7 · 6 Theorems</a>
    <a class="section" href="#history">Project History</a>
    <a href="#timeline">8 · Version Timeline</a>
  </nav>
  <div class="footer">
    Generated {date}<br>
    MATHIR Research Team<br>
    Confidential — Master's Defense
  </div>
</aside>

<main>

  <div class="cover" id="cover">
    <div class="tag">MASTER'S DEFENSE — JUNE 2026</div>
    <h1>MATHIR V7.2</h1>
    <div class="sub">Visual Architecture Report</div>
    <p style="font-size:16px; color:#e2e8f0; max-width:640px; line-height:1.7;">
      Eight production-quality diagrams covering the 4-tier hierarchical
      memory, the V7.1 retrieval research, multi-modal fusion, robustness
      under 20 concurrent agents, the 6 formal theorems, and the V1→V7.2
      version evolution.
    </p>
    <div class="meta" style="margin-top:32px;">
      <div><strong>4</strong>Tier memory</div>
      <div><strong>6</strong>Formal theorems</div>
      <div><strong>8</strong>Novel algorithms</div>
      <div><strong>62+</strong>Unit tests</div>
      <div><strong>45.7%</strong>Retrieval quality</div>
    </div>
  </div>

  <div class="toc" id="toc">
    <h2>Table of Contents</h2>
    <ol>
      <li><a href="#architecture">Hierarchical Architecture</a> — the 4-tier system and the KL-constrained router</li>
      <li><a href="#tiers">The 4 Memory Tiers</a> — capacity, update rate, slot visualisation per tier</li>
      <li><a href="#retrieval">Retrieval Quality Comparison</a> — 5 systems benchmarked on 200 textbook chunks</li>
      <li><a href="#pareto">Speed–Quality Pareto Frontier</a> — how to pick A vs D vs FAISS</li>
      <li><a href="#stress">Multi-Agent Stress Test</a> — 20 parallel agents, 100% success</li>
      <li><a href="#multimodal">Multi-Modal Fusion</a> — text, image, audio, video in a shared embedding space</li>
      <li><a href="#theorems">Theoretical Foundations</a> — 6 theorems reduced to classical results</li>
      <li><a href="#timeline">Version Evolution</a> — V1 → V7.2 across 5 months</li>
    </ol>
  </div>

  <!-- Sections are generated from the DIAGRAMS list at build time -->
  {sections}

  <footer>
    <strong>MATHIR V7.2</strong> — Visual Architecture Report · Generated {date}<br>
    Author: Prince Gildas Mbama Kombila · MATHIR Research Team<br>
    Source: <a href="../docs/MASTER_RESEARCH_PAPER.md">docs/MASTER_RESEARCH_PAPER.md</a>
    · <a href="../CHANGELOG.md">CHANGELOG.md</a>
  </footer>
</main>
</div>
</body>
</html>
"""


SECTION_TEMPLATE = """
  <div class="section-block" id="{id}">
    <h2>{title}</h2>
    <p class="subtitle">{subtitle}</p>
    {pills}
    <div class="figure-card">
      <img src="{data_uri}" alt="{title}" loading="lazy">
      <div class="fig-caption">Figure {index} — {title}</div>
    </div>
    <p>{description}</p>
    {math_block}
  </div>
"""


def _section_pills(d: dict) -> str:
    """Return a few contextual pills for a diagram section."""
    pills = {
        "architecture": '<span class="pill primary">Architecture</span>'
                        '<span class="pill secondary">MATHIR core</span>',
        "tiers":        '<span class="pill primary">Memory</span>'
                        '<span class="pill tertiary">V7.2 stable</span>',
        "retrieval":    '<span class="pill secondary">V7.1 research</span>'
                        '<span class="pill gold">D wins</span>',
        "pareto":       '<span class="pill secondary">Pareto</span>'
                        '<span class="pill primary">A / D / FAISS</span>',
        "stress":       '<span class="pill tertiary">Robust</span>'
                        '<span class="pill primary">20 agents</span>',
        "multimodal":   '<span class="pill primary">Modality-agnostic</span>'
                        '<span class="pill tertiary">4 encoders</span>',
        "theorems":     '<span class="pill danger">6 theorems</span>'
                        '<span class="pill primary">All proven</span>',
        "timeline":     '<span class="pill gold">V7.2 latest</span>'
                        '<span class="pill danger">V1–V3 legacy</span>',
    }
    return pills.get(d["id"], "")


def build_sections() -> str:
    """Build the HTML for all diagram sections with embedded base64 PNGs."""
    parts = []
    for i, d in enumerate(DIAGRAMS, 1):
        png_path = OUTPUT_DIR / d["file"]
        data_uri = _png_to_data_uri(png_path)
        if not data_uri:
            print(f"[build_report] WARNING: {png_path} missing — diagram will be blank")

        math_block = ""
        if d.get("math"):
            math_block = (
                '<div class="math-block">' + d["math"] + '</div>'
            )

        section_html = SECTION_TEMPLATE.format(
            id=d["id"],
            title=d["title"],
            subtitle=d["subtitle"],
            pills=_section_pills(d),
            data_uri=data_uri,
            index=i,
            description=d["description"],
            math_block=math_block,
        )
        parts.append(section_html)
    return "\n".join(parts)


def main() -> int:
    print("=" * 70)
    print("MATHIR V7.2 — Building Self-Contained HTML Report")
    print("=" * 70)

    # 1) Make sure all PNGs exist
    _ensure_diagrams()

    # 2) Build sections
    sections_html = build_sections()

    # 3) Inject into template
    today = datetime.now().strftime("%B %Y")
    html = HTML_TEMPLATE.replace("{sections}", sections_html)
    html = html.replace("{date}", today)

    # 4) Write
    HTML_OUT.write_text(html, encoding="utf-8")
    size_kb = HTML_OUT.stat().st_size / 1024
    print()
    print(f"  → Wrote {HTML_OUT.name}  ({size_kb:.0f} KB)")
    print(f"  → Open in any browser, or print to PDF for the defense.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
