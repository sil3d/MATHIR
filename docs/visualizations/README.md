# 🎨 MATHIR — Visualizations Suite

**8 production-quality diagrams + 1 self-contained HTML report.**

These were previously at the project root (`visualizations/`) before V7.4.
They are now co-located with the rest of the documentation in `docs/`.

---

## 📁 Files in this directory

| File | What it is |
|------|------------|
| `generate_diagrams.py` | matplotlib-based generator. Produces the 8 PNGs below. |
| `build_report.py` | Builds `visual_report.html` with all PNGs embedded as base64. |
| `01_architecture_main.png` | 5-tier MATHIR architecture (high-level system diagram). |
| `02_4_memory_tiers.png` | Memory tier deep-dive with slot visualizations. |
| `03_retrieval_comparison.png` | Quality comparison of 5 retrieval systems (V7.1). |
| `04_latency_quality_tradeoff.png` | Speed-quality Pareto frontier. |
| `05_multi_agent_stress.png` | Concurrent-stores stress test (20 agents). |
| `06_multimodal_fusion.png` | Multi-modal memory (text/image/audio/video). |
| `07_theorem_network.png` | 6 theorems with classical-result dependencies. |
| `08_version_timeline.png` | V1 → V8.4.1 evolution timeline. |
| `visual_report.html` | Self-contained, printable report (1.9 MB, base64 PNGs inline). |

---

## 🔄 Regenerating the diagrams

```bash
# Re-create the 8 PNGs from scratch
python docs/visualizations/generate_diagrams.py

# Re-build the self-contained HTML report
python docs/visualizations/build_report.py

# Open the report
start docs/visualizations/visual_report.html    # Windows
open docs/visualizations/visual_report.html      # macOS
xdg-open docs/visualizations/visual_report.html  # Linux
```

---

## 🖨️ Printing / exporting to PDF

`visual_report.html` is print-optimized. Open it in any modern browser and
`Ctrl+P → Save as PDF` to get a print-ready copy of the defense.

---

## 📐 Style

- Resolution: 150 DPI PNG
- Brand palette: per master's brand guidelines (see `COLORS` in
  `generate_diagrams.py`).
- Watermark: "MATHIR V8.4.1" in every figure.
- All diagrams cite the data source in the footer
  (e.g. `Source: ../../results/compare_all_approaches.json`).

---

*Last reorganized: V7.4 (2026-06-03).*
