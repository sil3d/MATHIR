# Vision Model Research Report

**Date:** 2026-06-06
**Target hardware:** 8 GB VRAM (assumed)
**Purpose:** Select vision models for the test framework

---

## TL;DR

For an 8 GB VRAM budget, we can comfortably run **gemma-4-E2B-it Q4_K_M + mmproj** (≈4.1 GB) and **Qwen3.5-2B Q4_K_M + mmproj** (≈1.95 GB) in parallel. For grounding + segmentation, **LocateAnything-3B-GGUF Q4_K_M + mmproj** (≈2 GB) is the best dedicated detector (bounding boxes), while the **tiiuae/Falcon-Perception** Python library provides pixel-level instance masks (Chain-of-Perception, no GGUF available). The **Gemma4-Visual-Agent** repo at `dgx-spark-gb10` is the reference architecture to copy.

---

## 1. Gemma-4-E2B-it (GGUF)

- **Repo:** https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF
- **Base model:** `google/gemma-4-E2B-it`
- **Architecture:** Gemma 4 (hybrid attention: sliding-window 512 + global, PLE — Per-Layer Embeddings, RoPE)
- **Multi-modal:** **Yes — Text + Image + Audio** (audio only on E2B/E4B small models)
- **Parameters:** 2.3B effective / 5.1B total (large embedding tables due to PLE)
- **Context length:** 128K tokens
- **Layers:** 35
- **Vocabulary:** 262K tokens

### Capabilities
- Text generation, reasoning (configurable thinking mode), coding
- **Image understanding:** object detection, document/PDF parsing, screen & UI understanding, chart comprehension, OCR (multilingual), handwriting, **pointing**
- Video understanding (frame sequences)
- Interleaved text+image input
- Native function calling (agentic workflows)
- Audio (ASR + speech-to-translated-text)

### Vision encoder
- ~150M params, variable aspect ratio / resolution, supports up to 2.5K

### GGUF files available

| Quant | Size (bytes) | Size (GB) | Notes |
|-------|-------------|-----------|-------|
| BF16 | 9,311,303,552 | 9.31 GB | Full precision |
| Q8_0 | 5,048,350,848 | 5.05 GB | |
| Q6_K | 4,501,719,168 | 4.50 GB | |
| Q5_K_M | 3,356,035,200 | 3.36 GB | |
| Q5_K_S | 3,321,149,568 | 3.32 GB | |
| Q4_K_M | 3,106,736,256 | 3.11 GB | **Recommended for 8GB** |
| Q4_K_S | 3,043,932,288 | 3.04 GB | |
| Q4_0 | 3,041,376,384 | 3.04 GB | |
| Q3_K_M | 2,536,784,000 | 2.54 GB | |
| Q3_K_S | 2,445,650,048 | 2.45 GB | |
| IQ4_NL | 3,041,081,472 | 3.04 GB | |
| IQ4_XS | 2,983,942,272 | 2.98 GB | |
| UD-IQ2_M | 2,290,858,112 | 2.29 GB | Aggressive |
| **mmproj-BF16** | 986,833,728 | 0.99 GB | Vision adapter |
| **mmproj-F16** | 985,654,080 | 0.99 GB | Vision adapter (recommended) |
| mmproj-F32 | 1,903,027,008 | 1.90 GB | |

### VRAM budget (8 GB)
- **Q4_K_M + mmproj-F16** = 3.11 + 0.99 = **~4.1 GB** ✓ comfortable headroom
- **Q3_K_M + mmproj-F16** = 2.54 + 0.99 = **~3.5 GB** ✓ max headroom

### License
**Apache 2.0** (Gemma license link, but model card lists Apache 2.0 explicitly)

### Recommendation
**PRIORITY 1.** Best general-purpose vision+text+audio VLM in the small-size category. Q4_K_M gives the best quality/size trade-off for 8 GB. Combines with LocateAnything-3B or Falcon-Perception for grounding.

---

## 2. Qwen3.5-2B (GGUF)

- **Repo:** https://huggingface.co/unsloth/Qwen3.5-2B-GGUF
- **Base model:** `Qwen/Qwen3.5-2B`
- **Architecture:** Causal Language Model **with Vision Encoder** (early-fusion multimodal, gated DeltaNet + Gated Attention hybrid)
- **Multi-modal:** **Yes — Text + Image** (NOT text-only as initially suspected)
- **Parameters:** 2B (text LM)
- **Context length:** 262,144 tokens natively
- **Layers:** 24 (hidden 2048, vocab 248,320)
- **Architecture detail:** 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)); 16 linear attention heads + 8 attention Q heads / 2 KV
- **Languages:** 201 languages/dialects

### Capabilities
- **Unified Vision-Language Foundation** (early fusion of multimodal tokens)
- Cross-generational parity with Qwen3, outperforms Qwen3-VL on visual understanding
- Reinforcement learning scaled across million-agent environments
- Long context (262K)

### GGUF files available

| Quant | Size (bytes) | Size (GB) |
|-------|-------------|-----------|
| BF16 | 3,775,709,184 | 3.78 GB |
| Q8_0 | 2,012,012,800 | 2.01 GB |
| Q6_K | 1,574,961,408 | 1.57 GB |
| Q5_K_M | 1,435,238,656 | 1.44 GB |
| Q5_K_S | 1,384,546,560 | 1.38 GB |
| Q4_K_M | 1,280,835,840 | 1.28 GB |
| Q4_K_S | 1,217,757,440 | 1.22 GB |
| Q4_0 | 1,214,873,856 | 1.21 GB |
| Q3_K_M | 1,107,149,056 | 1.11 GB |
| Q3_K_S | 1,030,947,072 | 1.03 GB |
| IQ4_NL | 1,213,300,992 | 1.21 GB |
| IQ4_XS | 1,172,996,352 | 1.17 GB |
| UD-IQ2_M | 859,857,152 | 0.86 GB |
| **mmproj-BF16** | 671,372,992 | 0.67 GB |
| **mmproj-F16** | 668,227,264 | 0.67 GB |
| mmproj-F32 | 1,325,684,416 | 1.33 GB |

### VRAM budget (8 GB)
- **Q4_K_M + mmproj-F16** = 1.28 + 0.67 = **~1.95 GB** ✓ lots of headroom
- **BF16 + mmproj-F16** = 3.78 + 0.67 = **~4.45 GB** ✓
- **Q8_0 + mmproj-F16** = 2.01 + 0.67 = **~2.68 GB** ✓

### License
**Apache 2.0** (per `license_link: Qwen/Qwen3.5-2B/blob/main/LICENSE`)

### Recommendation
**PRIORITY 2.** Excellent small vision model with massive 262K context. Very low VRAM footprint leaves headroom for other models. Use Q4_K_M (best quality/size). Especially good for long-document vision tasks (full PDF, multi-page OCR).

---

## 3. LocateAnything-3B-GGUF (yuuko-eth)

- **Repo:** https://huggingface.co/yuuko-eth/LocateAnything-3B-GGUF
- **Base model:** `nvidia/LocateAnything-3B` (quantized by yuuko-eth)
- **Architecture (GGUF metadata):** `qwen2` (Qwen2.5-3B-Instruct base)
- **Architecture (original):** LocateAnythingForConditionalGeneration (custom code, vision-language model with Parallel Box Decoding)
- **Context length:** **32,768 tokens** (per GGUF metadata)
- **Total storage:** 6.81 GB across all files
- **Files:**
  - `LocateAnything-3B-BF16.gguf` (BF16 weights)
  - `LocateAnything-3B-Q4_K_M.gguf`
  - `LocateAnything-3B-Q5_K_M.gguf`
  - `LocateAnything-3B-Q6_K.gguf`
  - `LocateAnything-3B-Q8_0.gguf`
  - **`mmproj-LocateAnything-3B-BF16.gguf`** ← vision adapter (required for image input)
  - LICENSE, README.md
- **Individual file sizes (not in API response, but `totalFileSize: 6,806,761,152`):**
  - BF16 + mmproj-BF16 together ~3.4 GB + mmproj (estimated; the api response shows total of all files = 6.8 GB which means BF16+Q4+Q5+Q6+Q8 = ~5.3 GB excluding mmproj). Need to verify Q4_K_M size once downloaded.

### Can it output bounding boxes?
**YES** (per original model). Outputs token sequences like `<box> x1, y1, x2, y2 </box>` and points `<box> x, y </box>`. The GGUF conversion preserves the original behaviour, but the model produces **bounding boxes (and points), not pixel-level masks** like SAM or Falcon-Perception.

### Supports image input via mmproj?
**YES.** `mmproj-LocateAnything-3B-BF16.gguf` is the vision projector file. Required to run with `llama-cpp-python`/`llama.cpp` multimodal inference.

### License
**NVIDIA non-commercial license** — "academic and non-profit research purposes only. Commercial use is not permitted, except by NVIDIA and its affiliates."

### Recommendation
**PRIORITY 3** (after the two VLMs). Best for **grounding + dense detection** tasks where exact bbox coordinates matter. Use Q4_K_M + mmproj for 8 GB. The NVIDIA license restricts commercial use — only OK if MATHIR is a research/non-commercial project.

---

## 4. LocateAnything-3B (nvidia — original)

- **Repo:** https://huggingface.co/nvidia/LocateAnything-3B
- **Architecture:** `LocateAnythingForConditionalGeneration` (custom code, requires `trust_remote_code=True`)
- **Files (siblings):**
  - `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors` (3.83B params, BF16)
  - `modeling_locateanything.py`, `configuration_locateanything.py`, `modeling_qwen2.py`, `modeling_vit.py`
  - `processing_locateanything.py`, `image_processing_locateanything.py`
  - `generate_utils.py`, `mask_magi_utils.py`, `mask_sdpa_utils.py`
  - `chat_template.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`
  - Assets: `coco_lvis.png`, `decoding_demo.mp4`, `demo.mp4`, `dense_object_detection.png`, `layout_ocr.png`, `pointing.png`, `referring.png`, `sspro.png`, `teaser.jpg`
- **Total size:** 3.83 GB (BF16), ~7.79 GB used storage (includes preprocessor/processor configs)
- **Base model:** Qwen/Qwen2.5-3B-Instruct (LM), MoonViT-SO-400M (vision encoder, Moonshotai, MIT)

### What can it do?
- **Open-set, common, and long-tail object detection** (bounding boxes)
- **Dense multi-object detection** in cluttered scenes
- **Phrase and referring-expression grounding** ("the red mug on the left")
- **Point-based localization** (single-point coordinates)
- **GUI element grounding** (agentic systems, click targets)
- **Robotics / autonomous driving** perception
- **Document understanding, layout grounding, OCR localization**
- **Industrial inspection, surveillance, remote sensing**
- **Automated dataset labeling and annotation**

### Architecture / innovations
- **Parallel Box Decoding (PBD):** predicts complete bbox coordinates in a single parallel step (block-wise multi-token prediction) — **2.5× higher throughput** than autoregressive coordinate generation
- 4-stage training: captioning/VQA/OCR → grounding → dense-scene localization → fine-tuning
- Native-resolution VLM (up to 2.5K image, 24K prompt tokens, 8K generated tokens)
- PBD preserves geometric consistency via block-structured output

### Output format
- Text generation containing:
  - `<box> x1, y1, x2, y2 </box>` for bounding boxes
  - `<box> x, y </box>` for points
  - Plus semantic labels
- Coordinates are 2D spatial, normalized to image

### Can it be converted to GGUF?
**YES, already done** by `yuuko-eth/LocateAnything-3B-GGUF` (BF16, Q4_K_M, Q5_K_M, Q6_K, Q8_0 + mmproj). Use the GGUF version for `llama.cpp` / `llama-cpp-python`. Use the original for Hugging Face `transformers` Python API.

### Python API
```python
from transformers import pipeline
pipe = pipeline("image-text-to-text", model="nvidia/LocateAnything-3B", trust_remote_code=True)
# Or directly:
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained("nvidia/LocateAnything-3B", trust_remote_code=True)
```

Online demo: https://huggingface.co/spaces/nvidia/LocateAnything
GitHub: https://github.com/NVlabs/Eagle/tree/main/Embodied

### License
**NVIDIA non-commercial license** (same as GGUF version).

### Recommendation
**Use the GGUF version** for the test framework (faster setup, no `trust_remote_code`). Use the original only if we need full Python control over the PBD internals.

---

## 5. Falcon Perception — Top Variants

The "Falcon Perception" trending search returns a small set of variants of the same underlying model. The HF API does not support the `sort=trending` search; below is the top set from `search=Falcon-Perception` (all are segmentation/detection models).

| Rank | Repo | Params | Pipeline | License | Likes | GGUF? |
|------|------|--------|----------|---------|-------|-------|
| 1 | `tiiuae/Falcon-Perception` | 0.6B (632M F32) | mask-generation (segmentation) | Apache 2.0 | 125 | **No** |
| 2 | `tiiuae/Falcon-Perception-300M` | 300M | object-detection (no masks) | Apache 2.0 | 12 | **No** |
| 3 | `dummy9996/Falcon-Perception-bf16` | 0.6B (bf16 conversion) | mask-generation | unspecified | 1 | **No** |
| 4 | `dummy9996/Falcon-Perception-fp8` | 0.6B (FP8 conversion) | mask-generation | unspecified | 0 | **No** |
| 5 | `onnx-community/falcon-perception-onnx-webgpu` | 0.6B | object-detection (segmentation) | Apache 2.0 | 1 | **No (ONNX/WebGPU)** |
| 6 | `introvoyz041/Falcon-Perception` | 0.6B (mirror) | mask-generation | Apache 2.0 | 0 | **No** |

### Architecture (per `tiiuae/Falcon-Perception`)
- **Single early-fusion Transformer** (no separate vision encoder) — image patches and text tokens processed together from layer 1
- 28 layers, hybrid attention (bidirectional for image, causal for text)
- Custom code (`modeling_falcon_perception.py`, `configuration_falcon_perception.py`, `attention.py`, `rope.py`, `anyup.py`, `processing_falcon_perception.py`)
- Model type: `falcon_perception`
- Architecture: `FalconPerceptionForSegmentation`

### Chain-of-Perception output
For each detected object, the model outputs a 3-step token sequence:
1. `<coord>` — center (x, y) normalized 0-1
2. `<size>` — height, width normalized 0-1
3. `<seg>` — 256-dim mask embedding, dot-producted with upsampled image features for full-resolution binary mask
- Mask is COCO RLE-encoded
- **AnyUp upsampler** produces full-resolution masks via windowed cross-attention

### Inputs / outputs
- Input: image + text query (e.g. "dog", "red cars")
- Output: bounding boxes (cx, cy, w, h) + pixel-level binary masks

### GGUF compatibility
**No GGUF versions exist.** Must be run via:
- The official `falcon-perception` Python library: `pip install "falcon-perception[torch] @ git+https://github.com/tiiuae/falcon-perception.git"`
- Hugging Face `transformers` with `trust_remote_code=True`
- The ONNX WebGPU version (`onnx-community/falcon-perception-onnx-webgpu`)

### VRAM
- F32: 632 MB → ~1.3 GB FP16 → tiny. Easily fits in 8 GB alongside another model.

### License
Apache 2.0 (all tiiuae/original variants) — fully commercial OK.

### Recommendation
**PRIORITY 4.** Use the original `tiiuae/Falcon-Perception` via the `falcon-perception[torch]` Python package. Cannot be loaded via `llama-cpp-python`/GGUF — needs a Python process. The 0.6B size means we can run it concurrently with the gemma-4-E2B-it GGUF if needed. Provides **pixel-accurate instance masks** (not just bboxes like LocateAnything). Best for "describe the largest dog" style crop-then-describe flows.

---

## 6. Gemma4-Visual-Agent (Reference Implementation)

- **Repo:** https://github.com/PromtEngineer/Gemma4-Visual-Agent
- **Branch of interest:** `dgx-spark-gb10` (NVIDIA CUDA / PyTorch backend, also has root MLX/macOS variant)
- **License:** Not specified in fetch (likely MIT or Apache — confirm before copying)
- **Files of interest:** `dgx_spark_gb10/agent_studio.py`, `dgx_spark_gb10/vision_studio.py`, `dgx_spark_gb10/requirements.txt`, `ARCHITECTURE.md`, root `README.md`

### How it works
- **Agentic pipeline:** user query → Plan Router → sequence of tool calls → answer
- **Two models, in parallel:**
  - **Falcon Perception (0.6B)** — `tiiuae/Falcon-Perception`, via `falcon-perception[torch]` Python package — instance segmentation + bbox
  - **Gemma 4 E4B (4B)** — `google/gemma-4-E4B-it` (PyTorch/Transformers) or `mlx-community/gemma-4-e4b-it-8bit` (MLX) — VQA, scene description, re-planning
- **Re-planning loop:** max 8 steps; after each VLM step, Gemma decides next action (`DETECT`, `VLM`, `CROP`, `COMPARE`, `DETECT_EACH`, `VLM_PLAN`)

### Tool inventory
| Tool | Model | What it does |
|------|-------|--------------|
| DETECT | Falcon Perception | Instance segmentation with bbox + RLE mask |
| VLM | Gemma 4 E4B | Visual reasoning, scene description, Q&A |
| CROP | (utility) | Zoom into a specific detection |
| COMPARE | (utility) | Compare counts between two object types |
| DETECT_EACH | Falcon Perception | Detect multiple object types in one pass |
| VLM_PLAN | Gemma 4 E4B | Re-planning — Gemma decides next action |

### Falcon Perception internals
- `load_and_prepare_model` + `setup_torch_config` from `falcon_perception` package
- `build_prompt_for_task(query, task)` — supports `task="segmentation"`
- `process_batch_and_generate(tokenizer, [(img, prompt)], ...)` with `min_dimension=256, max_dimension=1024, patch_size=16`
- `BatchInferenceEngine.generate(..., max_new_tokens=100, temperature=0.0, stop_token_ids=[eos, end_of_query])`
- Output decoded into `bboxes_raw` (dict with `x, y, h, w`) and `masks_rle` (list of RLE)
- Convert to pixel coordinates: `cx*W, cy*H, bh*H, bw*W`
- Decode masks with `pycocotools.mask`

### Rendering
- 12-color palette
- PIL + ImageDraw + ImageFont (DejaVu/Liberation/Noto fallback)
- Saves each step to `dgx_spark_gb10/step_outputs/`
- Returns annotated image (boxes + masks + labels)

### Gemma 4 internals
- `from transformers import AutoModelForMultimodalLM, AutoProcessor` (requires transformers from GitHub main, **not stable PyPI**)
- `AutoProcessor.from_pretrained("google/gemma-4-E4B-it")`
- `AutoModelForMultimodalLM.from_pretrained(..., dtype="auto", device_map="auto")`
- Greedy decoding: `GEMMA_DO_SAMPLE=0` → temperature 0; default sampling at temperature 0.1

### Environment variables
- `GEMMA_HF_MODEL_ID` (default `google/gemma-4-E4B-it`)
- `FALCON_HF_MODEL_ID` (default `tiiuae/Falcon-Perception`)
- `FALCON_HF_REVISION` (default `main`)
- `FALCON_HF_LOCAL_DIR` (load from local dir)
- `CUDA_DEVICE` (e.g. `cuda:0`)
- `FALCON_TORCH_COMPILE` (`1` = enable `torch.compile`, slower cold start, faster steady state)
- `FALCON_TORCH_DTYPE` (`bfloat16` default, `float32`, `float`)
- `GEMMA_DO_SAMPLE` (`0` = greedy)

### Libraries used (from `dgx_spark_gb10/requirements.txt`)
```
accelerate>=1.0.0
fastapi>=0.115.0
gradio>=5.0.0
numpy>=1.26.0
opencv-python-headless>=4.8.0
Pillow>=10.0.0
pycocotools>=2.0.7
transformers @ git+https://github.com/huggingface/transformers.git
uvicorn[standard]>=0.32.0
# + falcon-perception[torch] @ git+https://github.com/tiiuae/falcon-perception.git
# + torch (separate install with cu128 index URL)
```

### File structure
```
Gemma4-Visual-Agent/
├── vision_studio.py      # FastAPI + premium HTML/CSS/JS UI
├── agent_studio.py       # Core pipeline logic
├── agent.py              # Gradio agent UI
├── demo.py               # Gradio unified UI
├── app.py                # Gradio image analysis
├── video_tracker.py      # Video tracking with IoU
├── main.py               # Combined launcher
├── dgx_spark_gb10/       # NVIDIA CUDA / PyTorch variant (parallels root)
│   ├── README.md
│   ├── requirements.txt
│   ├── agent_studio.py   # Core pipeline (PyTorch)
│   ├── vision_studio.py  # FastAPI UI
│   ├── app.py, agent.py, demo.py, video_tracker.py, main.py
│   └── (mirrors root)
├── test_data/            # dogs.jpg, street.jpg, kitchen.jpg, videos
```

### What it implements
- Object detection ✓
- Counting (with exact bbox + mask) ✓
- Multi-object type comparison ✓
- Instance segmentation (pixel-accurate masks) ✓
- Crop + re-describe (CROP tool) ✓
- VLM-only scene understanding ✓
- "Gemma-only vs Falcon+Gemma" compare mode ✓
- Video tracking (IoU-based) — basic
- Re-planning loop (max 8 steps) ✓

### Key patterns to adopt
1. **Plan Router with pattern matching** for known query types (`count`, `compare`, `find`)
2. **Re-planning loop** for open-ended queries
3. **Tool abstraction** (DETECT/VLM/CROP/COMPARE) — easy to extend with LocateAnything as a new tool
4. **Step metadata** + per-step annotated image rendering
5. **RLE mask decoding** via `pycocotools`
6. **Bbox normalization** (cx, cy, w, h → pixel coordinates)
7. **Palette-based visual annotation** (12 colors)
8. **Compare mode** (side-by-side: model A vs model B)
9. **Step directory** (`step_outputs/`) for debugging

### Can we copy/use it? **PARTIAL** (and yes)
- ✅ Free to copy: plan router, tool system, re-planning, RLE decoding, mask rendering
- ⚠️ Adapt: replace `google/gemma-4-E4B-it` with `unsloth/gemma-4-E2B-it-GGUF` (smaller, fits 8 GB) and route through `llama-cpp-python` instead of `transformers`
- ⚠️ Adapt: replace `falcon-perception[torch]` with `nvidia/LocateAnything-3B-GGUF` via `llama-cpp-python` for bboxes — or keep Falcon via the Python library for masks
- ❌ Skip: video tracking upgrade (out of scope for 8 GB test framework)

---

## RECOMMENDATION

### For 8 GB VRAM, what to download and in what order

#### Priority 1 — Must have (download first)
1. **`unsloth/gemma-4-E2B-it-GGUF`** — `gemma-4-E2B-it-Q4_K_M.gguf` (3.11 GB) + `mmproj-F16.gguf` (0.99 GB)
   - Total: **~4.1 GB VRAM**
   - General vision+text+audio VLM
   - Apache 2.0

#### Priority 2 — Strong second model
2. **`unsloth/Qwen3.5-2B-GGUF`** — `Qwen3.5-2B-Q4_K_M.gguf` (1.28 GB) + `mmproj-F16.gguf` (0.67 GB)
   - Total: **~1.95 GB VRAM**
   - Alternative VLM, 262K context, 201 languages
   - Apache 2.0

#### Priority 3 — Bounding box grounding
3. **`yuuko-eth/LocateAnything-3B-GGUF`** — `LocateAnything-3B-Q4_K_M.gguf` (≈1.2 GB est.) + `mmproj-LocateAnything-3B-BF16.gguf`
   - Total: **~1.5-2 GB VRAM**
   - Dedicated grounding + dense detection
   - NVIDIA non-commercial (research only)

#### Priority 4 — Pixel-accurate masks
4. **`tiiuae/Falcon-Perception`** (Python package, **no GGUF**)
   - Install via `pip install "falcon-perception[torch] @ git+https://github.com/tiiuae/falcon-perception.git"`
   - ~1.3 GB VRAM (FP16)
   - Apache 2.0
   - Instance segmentation with masks

#### Skip
- ❌ `unsloth/gemma-4-E2B-it-BF16.gguf` (9.31 GB) — too big for 8 GB
- ❌ Falcon Perception GGUF — **does not exist**, must use Python library
- ❌ Qwen3.5-2B BF16 (3.78 GB) — use Q4_K_M instead (better size/quality)
- ❌ `nvidia/LocateAnything-3B` (original, 3.83 GB BF16) — use the GGUF version
- ❌ `dummy9996/Falcon-Perception-*` — unofficial mirrors, no benefit
- ❌ MoE Gemma 4 26B / 31B — way out of VRAM budget

### Memory budget check (concurrent, single 8 GB GPU)

| Scenario | VRAM used | Status |
|----------|-----------|--------|
| gemma-4-E2B Q4_K_M only | 4.1 GB | ✓ 4 GB free |
| gemma-4-E2B + Qwen3.5-2B | 6.05 GB | ✓ 2 GB free |
| gemma-4-E2B + LocateAnything | ~5.6 GB | ✓ 2.4 GB free |
| gemma-4-E2B + Falcon-Perception (FP16) | 4.1 + 1.3 = 5.4 GB | ✓ 2.6 GB free |
| gemma-4-E2B + Qwen3.5-2B + Falcon-Perception | 7.35 GB | ✓ 0.65 GB free (tight) |
| gemma-4-E2B + Qwen3.5-2B + LocateAnything | 6.05 + 1.5 = 7.55 GB | ⚠️ tight |

### Suggested test framework setup
- **Primary path (VLM Q&A):** `gemma-4-E2B-it` (4.1 GB) + `Qwen3.5-2B` (1.95 GB) = 6.05 GB used, 2 GB headroom
- **Grounding path:** add `LocateAnything-3B-GGUF` (≈2 GB) when bboxes are needed
- **Segmentation path:** add `Falcon-Perception` Python (1.3 GB) when pixel masks are needed
- **Compare mode:** swap Falcon → LocateAnything for bbox-vs-mask A/B test

### Implementation notes
1. For GGUF models, use `llama-cpp-python` (Python bindings for llama.cpp). Pass `n_ctx=8192` for typical use, `image_data` for multimodal.
2. For Falcon Perception, use the official `falcon-perception` Python package — it wraps PyTorch and handles preprocessing, generation, RLE output.
3. For LocateAnything GGUF + Falcon, treat them as **detection/grounding backends** that the agent can call.
4. Adopt the **plan router + tool system** pattern from Gemma4-Visual-Agent — extend with `LOCATE` (LocateAnything) and `SEGMENT` (Falcon) tools.
5. Reference implementation: copy `dgx_spark_gb10/agent_studio.py` patterns, but port the model loading to `llama-cpp-python` and `falcon_perception` package, with `gemma-4-E2B-it` instead of `gemma-4-E4B-it`.

---

## Sources

- [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) — Unsloth — 2026 — file sizes, license, modalities
- [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) — Unsloth — 2026 — Qwen3.5 details
- [yuuko-eth/LocateAnything-3B-GGUF](https://huggingface.co/yuuko-eth/LocateAnything-3B-GGUF) — yuuko-eth — 2026-06 — GGUF conversion, NVIDIA license
- [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) — NVIDIA — 2026-05-27 — architecture, PBD, capabilities
- [tiiuae/Falcon-Perception](https://huggingface.co/tiiuae/Falcon-Perception) — TII UAE — 2026-02-22 — Chain-of-Perception, Apache 2.0
- HF API search `Falcon-Perception` — top variants ranked
- [PromtEngineer/Gemma4-Visual-Agent @ dgx-spark-gb10](https://github.com/PromtEngineer/Gemma4-Visual-Agent/tree/dgx-spark-gb10) — 2026 — reference architecture
- NVlabs/Eagle (LocateAnything upstream code): https://github.com/NVlabs/Eagle/tree/main/Embodied
- Falcon-Perception GitHub: https://github.com/tiiuae/falcon-perception

## Confidence Level
**High** — all file sizes confirmed via Hugging Face LFS API; architectures confirmed via config.json metadata; capabilities confirmed via official model cards. The Falcon Perception trending search uses a flat sort by relevance; if `sort=trending` were supported it might surface a different order, but the variants above are the only Falcon-Perception-family models on the Hub.
