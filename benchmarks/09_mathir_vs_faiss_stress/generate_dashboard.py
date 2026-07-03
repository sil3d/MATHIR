#!/usr/bin/env python3
"""
Generate one static, self-contained HTML dashboard covering every dataset /
every approach found in the benchmark result JSON files -- FAISS-only,
BM25-only, hybrid RRF, hybrid+CE (from
benchmarks/06_results/archive/multi_dataset_efficient_results.json or the
merged benchmarks/09_mathir_vs_faiss_stress/results/mathir_vs_faiss_results.json),
plus MATHIR_recall / MATHIR_hybrid, plus the stress-test result files
(typo-robustness, long-term decay/consolidate).

"Dynamic even if you change model": nothing here hardcodes a model name or
an approach list. Every table is built by reading whatever `metadata.model`
and whatever approach keys actually exist in the JSON at generation time --
re-run this script any time after a new benchmark run (new model, new
dataset, new approach) and the dashboard regenerates to match, no code edit
needed. It is NOT a live-updating page by itself (opening the same HTML
twice won't show new numbers) -- re-run this script to refresh it. That's a
deliberate simplicity trade-off: a fully live dashboard would need a running
web server; this is a portable single-file HTML you can open straight from
disk or commit to the repo.

Usage:
    python generate_dashboard.py
Output:
    benchmarks/09_mathir_vs_faiss_stress/results/dashboard.html
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARCHIVE_RESULTS = Path(__file__).resolve().parent.parent / "06_results" / "archive" / "multi_dataset_efficient_results.json"
MERGED_RESULTS = RESULTS_DIR / "mathir_vs_faiss_results.json"
OUT_FILE = RESULTS_DIR / "dashboard.html"

METRIC_KEYS = ["nDCG@10", "MRR@10", "Recall@100", "time_s"]

APPROACH_LABELS = {
    "1_FAISS_only": "FAISS (dense-only)",
    "2_BM25_only": "BM25 (sparse-only)",
    "3_hybrid_RRF": "Hybrid RRF (dense+BM25)",
    "4_hybrid_CE": "Hybrid + Cross-Encoder rerank",
    "5_MATHIR_recall": "MATHIR memory_recall",
    "6_MATHIR_hybrid": "MATHIR memory_hybrid_search",
}


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_main_results() -> dict:
    """Prefer the merged file (has MATHIR numbers), fall back to the
    FAISS/BM25-only archive if the merge hasn't been run yet."""
    merged = load_json(MERGED_RESULTS)
    if merged:
        return merged
    archived = load_json(ARCHIVE_RESULTS)
    return archived or {"metadata": {}, "results": {}}


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if v < 100 else f"{v:.2f}"
    return str(v)


def build_main_table(data: dict) -> str:
    model = data.get("metadata", {}).get("model", "(model not recorded in results JSON)")
    device = data.get("metadata", {}).get("device", "?")
    results = data.get("results", {})
    if not results:
        return "<p><em>No results yet.</em></p>"

    # Discover every approach key that appears across ALL datasets, dynamically.
    approach_keys = sorted({k for ds in results.values() for k in ds.keys()})

    html = [f'<p class="meta">Embedding model: <code>{model}</code> &middot; Device: <code>{device}</code></p>']
    for dataset, approaches in results.items():
        html.append(f"<h3>{dataset}</h3>")
        html.append('<table><thead><tr><th>Approach</th>')
        for m in METRIC_KEYS:
            html.append(f"<th>{m}</th>")
        html.append("</tr></thead><tbody>")
        best_ndcg = max((v.get("nDCG@10", -1) for v in approaches.values()), default=-1)
        for key in approach_keys:
            if key not in approaches:
                continue
            row = approaches[key]
            label = APPROACH_LABELS.get(key, key)
            is_best = row.get("nDCG@10") == best_ndcg and best_ndcg >= 0
            css = ' class="best"' if is_best else ""
            html.append(f"<tr{css}><td>{label}</td>")
            for m in METRIC_KEYS:
                html.append(f"<td>{fmt(row.get(m, '-'))}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")
    return "\n".join(html)


def build_robustness_section() -> str:
    files = sorted(RESULTS_DIR.glob("stress_typo_robustness_*.json"))
    if not files:
        return "<p><em>No typo-robustness stress test results yet. Run stress_typo_robustness.py.</em></p>"
    html = []
    for f in files:
        data = load_json(f)
        if not data:
            continue
        dataset = data.get("dataset", f.stem)
        html.append(f"<h3>{dataset} — query-noise robustness ({data.get('n_queries', '?')} queries, "
                    f"2 char-level edits injected per query)</h3>")
        html.append("<table><thead><tr><th>Approach</th><th>Clean nDCG@10</th><th>Noisy nDCG@10</th>"
                     "<th>Absolute drop</th><th>% drop</th></tr></thead><tbody>")
        for name, r in data.get("approaches", {}).items():
            clean = r["clean"]["nDCG@10"]
            noisy = r["noisy"]["nDCG@10"]
            drop = clean - noisy
            pct = (drop / clean * 100) if clean > 0 else 0.0
            html.append(f"<tr><td>{name}</td><td>{fmt(clean)}</td><td>{fmt(noisy)}</td>"
                        f"<td>{fmt(drop)}</td><td>{pct:.1f}%</td></tr>")
        html.append("</tbody></table>")
    return "\n".join(html)


def build_decay_section() -> str:
    files = sorted(RESULTS_DIR.glob("stress_longterm_decay_*.json"))
    if not files:
        return "<p><em>No long-term decay/consolidate stress test results yet. Run stress_longterm_decay.py.</em></p>"
    html = []
    for f in files:
        data = load_json(f)
        if not data:
            continue
        dataset = data.get("dataset", f.stem)
        html.append(f"<h3>{dataset} — accelerated decay / consolidate simulation</h3>")
        html.append("<table><thead><tr><th>Phase</th><th>nDCG@10</th><th>MRR@10</th><th>Recall@100</th></tr></thead><tbody>")
        for phase_key, phase in data.get("phases", {}).items():
            html.append(f"<tr><td>{phase_key}</td><td>{fmt(phase.get('nDCG@10'))}</td>"
                        f"<td>{fmt(phase.get('MRR@10'))}</td><td>{fmt(phase.get('Recall@100'))}</td></tr>")
        html.append("</tbody></table>")
        before = data.get("memory_count_before_consolidate")
        after = data.get("memory_count_after_consolidate")
        if before is not None:
            html.append(f"<p class='meta'>Stored memory count: {before} &rarr; {after} after consolidation.</p>")
    return "\n".join(html)


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MATHIR vs FAISS/BM25 — Retrieval Benchmark Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem; background: #0f1117; color: #e6e6e6; }}
  h1 {{ font-size: 1.6rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 1px solid #333; padding-bottom: .3rem; }}
  h3 {{ margin-top: 1.5rem; color: #9ecbff; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; }}
  th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: left; font-variant-numeric: tabular-nums; }}
  th {{ background: #1b1f2a; }}
  tr.best td {{ background: #163a1e; font-weight: bold; }}
  .meta {{ color: #999; font-size: .9rem; }}
  code {{ background: #1b1f2a; padding: 1px 5px; border-radius: 3px; }}
  .disclosure {{ background: #241a1a; border-left: 4px solid #a33; padding: .8rem 1rem; margin: 1rem 0; }}
</style>
</head>
<body>
<h1>MATHIR vs FAISS / BM25 / Hybrid — Real Retrieval Benchmark</h1>
<p class="meta">Generated by benchmarks/09_mathir_vs_faiss_stress/generate_dashboard.py.
Re-run that script after any new benchmark pass (new dataset, new model, new approach) to refresh this page — nothing here is hand-edited.</p>

<div class="disclosure">
<strong>Honest disclosures:</strong>
<ul>
<li>fluid_mechanics dataset queries are single-gold-passage LLM-generated judgments (not exhaustively annotated like SciFact) — see benchmarks/07_utilities/generate_fluid_mechanics_queries.py docstring.</li>
<li>Long-term/decay results are from ONE accelerated decay+consolidate pass per run, not a literal multi-month field trial — see stress_longterm_decay.py docstring.</li>
<li>All nDCG@10/MRR@10/Recall@100 numbers are computed with the real <code>beir.retrieval.evaluation.EvaluateRetrieval</code> evaluator, not a custom metric implementation.</li>
</ul>
</div>

<h2>Main comparison (all datasets, all approaches)</h2>
{main_table}

<h2>Stress test: query-noise / "auto-correct" robustness</h2>
{robustness}

<h2>Stress test: long-term memory (decay + consolidation)</h2>
{decay}

</body>
</html>
"""


def main():
    main_data = load_main_results()
    html = TEMPLATE.format(
        main_table=build_main_table(main_data),
        robustness=build_robustness_section(),
        decay=build_decay_section(),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote dashboard to {OUT_FILE}")


if __name__ == "__main__":
    main()
