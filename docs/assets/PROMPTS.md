# MATHIR — Image Generation Prompts

Two prompts you can paste into Midjourney, DALL-E 3, Stable Diffusion XL, or any image AI.

---

## PROMPT 1 — LOGO

**Use for:** Square logo, 1024x1024 minimum, clean background, professional brand mark.

```
A premium minimalist logo for "MATHIR" (a memory-augmented AI system).
Centered hexagonal core glowing in deep violet (#8b5cf6), representing a
neural memory node. Eight fine neural pathways radiating outward to small
luminous nodes, alternating between violet and electric green (#22c55e)
accents. Inner core contains a stylized "M" monogram in white. Two
concentric orbital rings with dashed patterns surround the core. Soft
bloom and bokeh glow effects. Pure white background. Ultra-clean, modern
tech aesthetic, premium brand mark, 8k render, vector-sharp edges,
suitable for GitHub README and favicon. No text other than the M monogram.
Style: Apple + Linear + Vercel brand mark quality. Color palette strictly:
#8b5cf6, #22c55e, #ffffff, with soft #1a1a2e shadows only.
```

**Midjourney flags:** `--ar 1:1 --v 6.1 --style raw --q 2`
**DALL-E 3:** size 1024x1024, style "vivid"

---

## PROMPT 2 — ARCHITECTURE DIAGRAM

**Use for:** Wide horizontal diagram, 1600x900, technical infographic with 5 stacked layers.

```
A premium technical architecture diagram in clean flat-design style, 5
horizontal layers stacked top to bottom on a soft off-white background
(#fafafa). Subtle drop shadows under each layer for depth.

LAYER 1 (top): A blue gradient rounded rectangle labeled "AI Agent / LLM"
with subtitle "Claude · GPT · Ollama · Any MCP client". Width 300px,
height 60px. Color: #3b82f6 to #1e40af gradient.

LAYER 2: A purple gradient rounded rectangle labeled "MCP Server" with
subtitle "10 tools · 4 prompt resources · tool discovery". Below in small
gray text: "memory_save · memory_recall · memory_smart_search ·
memory_hybrid_search". Color: #8b5cf6 to #6d28d9.

LAYER 3: A violet gradient rounded rectangle labeled "MATHIR Daemon" with
subtitle "Persistent process · embedder pre-loaded · thread-safe · 50
concurrent connections". Color: #a78bfa to #7c3aed.

LAYER 4: Four rounded rectangles side by side, each 180x80px:
  - Left: amber/gold gradient (#fbbf24 to #f59e0b), "Working Memory"
  - Center-left: emerald green (#34d399 to #10b981), "Episodic Memory"
  - Center-right: sky blue (#60a5fa to #3b82f6), "Semantic Memory"
  - Right: pink (#f472b6 to #ec4899), "Procedural Memory"

LAYER 5 (bottom): Three storage icons in a row:
  - Left: green gradient box "Embedder — MiniLM-L12 · 384d · GPU"
  - Center-left: slate gray box "Vector Index — sqlite-vec · cosine"
  - Center-right: slate gray box "Metadata DB — SQLite · thread-safe"
  - Right: slate gray box "BM25 Index — Lexical · hybrid"

Connect layers with thin gray vertical arrows showing data flow direction
top to bottom. Use #6b7280 for arrows, #22c55e for data-embedding flow
on the left side. Use small text annotations like "JSON-RPC over stdio"
between layers. Use a clean sans-serif font (Inter or SF Pro feel). All
text must be crisp and readable. No decorative elements. Pure technical
infographic, like a Stripe architecture page or Vercel docs diagram.
Ultra-clean, 8k, professional, infographic.
```

**Midjourney flags:** `--ar 16:9 --v 6.1 --style raw --q 2`
**DALL-E 3:** size 1792x1024, style "natural"

---

## HOW TO USE

1. **Logo** — Save the result as `docs/assets/logo.png`, then update README line 3:
   ```html
   <img src="docs/assets/logo.png" alt="MATHIR Logo" width="180"/>
   ```

2. **Architecture** — Save as `docs/assets/architecture.png`, then update README line 539:
   ```html
   <img src="docs/assets/architecture.png" alt="MATHIR Architecture" width="900"/>
   ```

3. **Commit + push** — both files will be picked up by the next `git add -A`.
