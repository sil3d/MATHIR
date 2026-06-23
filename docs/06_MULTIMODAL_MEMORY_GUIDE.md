# MATHIR Multimodal Memory Guide

**Does MATHIR accept video, audio, text from an LLM? How does it store data as a memory?**

*Master's-thesis-grade technical reference · MATHIR V8.4.1 · 2026*

---

## 1. TL;DR

- **Yes, MATHIR accepts every modality** — text, audio, image, video, and any mix thereof. It is **modality-agnostic** by design because it stores **embeddings (vectors)**, not raw data.
- **The pipeline is universal**: *raw modality → modality-specific encoder (CLIP, CLAP, Whisper, sentence-transformers) → fixed-dim embedding vector → MATHIR's 5-tier memory*. MATHIR never sees the original bytes.
- **Storage is a 5-tier numerical bank**: working_memory (64 slots), episodic (1000 slots), semantic (256 prototypes), procedural (128 slots), immunological (100 slots). All slots are `float32` tensors of `internal_dim=272` by default (or raw-embedding dim when `use_raw_embedding=True`). 1000 embeddings at 512-dim cost **≈ 2 MB**; V7 sparse coding compresses that by **9.3×** to ~117 KB.

---

## 2. The Fundamental Insight

MATHIR is a **memory layer for LLMs that operates on numerical embeddings**, not on tokens, pixels, or waveforms. This single design choice makes it **inherently modality-agnostic**: the only thing that crosses the MATHIR boundary is a `torch.Tensor` of shape `[B, D]` (typically `[1, 512]` for CLIP-class models or `[1, 384]` for sentence-transformers). Whatever produced that tensor — a sentence, a photograph, a 30-second voice memo, a 5-minute video, a sensor reading, a protein structure, a stock-price tick — is, from MATHIR's perspective, **just a point in a vector space**. The encoder that produced the embedding is responsible for the *semantics*; MATHIR is responsible for the *memory operations* (store, retrieve, cluster, detect anomaly, route, compress, forget). This separation of concerns is what makes the system composable: you can swap CLIP for BLIP-2, Whisper for Wav2Vec2, or even a domain-specific encoder (e.g. for EEG or genomics) without touching a single line of MATHIR code.

---

## 3. Modality Support Matrix

| Modality | Supported? | Recommended Encoder | Dim | Params | Same-space text/image? |
|----------|:----------:|---------------------|----:|-------:|:----------------------:|
| **Text** | ✅ | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 22 M | — |
| **Text (high quality)** | ✅ | `intfloat/e5-large-v2` | 1024 | 335 M | — |
| **Image** | ✅ | `openai/clip-vit-base-patch32` | 512 | 151 M | ✅ (CLIP) |
| **Image (SOTA)** | ✅ | `openai/clip-vit-large-patch14` | 768 | 428 M | ✅ (CLIP) |
| **Audio** | ✅ | `laion/clap-htsat-unfused` | 512 | ~150 M | ✅ (CLAP) |
| **Audio → text → embed** | ✅ | `openai/whisper-large-v3` → text encoder | 1280 → D | 1550 M | indirect |
| **Video (per-frame)** | ✅ | CLIP on each frame + mean-pool | 512 | 151 M | ✅ |
| **Video (native)** | ✅ | `MCG-NJU/videoclip-base` | 512 | ~150 M | ✅ |
| **Multimodal (text+image aligned)** | ✅ | CLIP / BLIP / SigLIP | 512–768 | 150 M–1 B | ✅ |
| **Multimodal (anything→anything)** | ✅ | LLaVA, BLIP-2, Flamingo | 4096+ | 7 B+ | via projection |
| **Tabular / time-series** | ✅ | TS2Vec, contrastive MLP | 64–512 | varies | ❌ |
| **Graphs / molecules** | ✅ | GraphSAGE, Mol2Vec | 64–512 | varies | ❌ |

**Key takeaway**: any model that produces a `torch.Tensor` works. The choice of encoder is dictated by your **downstream task semantics**, not by MATHIR.

---

## 4. Architecture Diagram

![MATHIR Architecture](assets/Mathir_architecture.png)

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │                       RAW INPUT  (any modality)                       │
   │                                                                        │
   │   📝 text      🖼️ image     🔊 audio waveform     🎬 video frames    │
   └────┬──────────────┬────────────────┬──────────────────┬───────────────┘
        │              │                │                  │
        ▼              ▼                ▼                  ▼
   ┌─────────┐   ┌──────────┐    ┌──────────┐       ┌──────────────┐
   │ text    │   │  CLIP    │    │  CLAP    │       │  per-frame    │
   │ encoder │   │  image   │    │  audio   │       │  CLIP + mean  │
   │ (MiniLM)│   │  tower   │    │  tower   │       │  OR VideoCLIP │
   └────┬────┘   └────┬─────┘    └────┬─────┘       └──────┬───────┘
        │              │                │                  │
        │  384-dim     │  512-dim       │  512-dim         │  512-dim
        │              │                │                  │
        └──────────────┴────────┬───────┴──────────────────┘
                                │
                                ▼  torch.Tensor [1, D]
                                │
                  ┌─────────────────────────────┐
                  │   MATHIRPluginV7 (modality  │
                  │        agnostic)            │
                  │                             │
                  │  ┌───────────────────────┐  │
                  │  │ input_proj            │  │   D → 272 (internal)
                  │  │ LayerNorm             │  │
                  │  └──────────┬────────────┘  │
                  │             ▼                │
                  │  ┌───────────────────────┐  │
                   │  │ 5-Tier Memory          │  │
                  │  │  • Working   (64)      │  │   circular buffer
                  │  │  • Episodic  (1000)    │  │   key-value
                  │  │  • Semantic  (256)     │  │   online k-means
                  │  │  • Immune    (100)     │  │   Mahalanobis
                  │  └──────────┬────────────┘  │
                  │             ▼                │
                  │  ┌───────────────────────┐  │
                  │  │ router (4 logits)      │  │   softmax → weights
                  │  │ reconstructor          │  │   self-supervised
                  │  └──────────┬────────────┘  │
                  │             ▼                │
                  │  output_proj  272 → D       │
                  └─────────────┬───────────────┘
                                │
                                ▼  torch.Tensor [1, D]
                  ┌──────────────────────────────┐
                  │  Enhanced embedding for the   │
                  │  downstream LLM (or recall   │
                  │  result, anomaly score,      │
                  │  router weights, KL loss)    │
                  └──────────────────────────────┘
```

**The MATHIR box is a black box that takes a vector and returns a vector.** The encoder that feeds it is swappable per modality.

---

## 5. Code Examples (copy-paste runnable)

All examples assume:

```bash
pip install torch transformers sentence-transformers open-clip torchcv  # open_clip optional
```

### 5.1 Text only (simplest)

```python
import torch
from sentence_transformers import SentenceTransformer
from mathir_lib import MATHIRPluginV7

# 1) Choose an encoder (any text encoder works)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2) Configure MATHIR for the encoder's dim (384 for MiniLM)
plugin = MATHIRPluginV7(embedding_dim=384)

# 3) Encode → store
text = "MATHIR is a 5-tier memory layer for LLM agents."
emb = torch.tensor(encoder.encode([text]))            # [1, 384]

plugin.perceive(emb)                                   # process + return enhanced
plugin.store({
    "embedding":  emb,                                 # required
    "modality":   "text",                              # user metadata
    "text":       text,                                # raw text (for BM25)
    "source":     "user_input",
})

# 4) Recall
query_emb = torch.tensor(encoder.encode(["memory layer"]))   # [1, 384]
scores, ids = plugin.episodic.retrieve(query_emb, k=5)
```

**Storage cost per slot**: `384 × 4 bytes = 1.5 KB`. For 1000 slots, ≈ **1.5 MB** raw.

---

### 5.2 Image (CLIP)

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from mathir_lib import MATHIRPluginV7

# 1) Load CLIP
model_id    = "openai/clip-vit-base-patch32"
clip        = CLIPModel.from_pretrained(model_id)
processor   = CLIPProcessor.from_pretrained(model_id)

# 2) Configure MATHIR for CLIP's 512-dim image feature
plugin = MATHIRPluginV7(embedding_dim=512)

# 3) Encode an image
image  = Image.open("cat.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
img_emb = clip.get_image_features(**inputs)            # [1, 512]

plugin.perceive(img_emb)
plugin.store({
    "embedding": img_emb,
    "modality":  "image",
    "image_id":  "cat.jpg",
    "tags":      ["cat", "animal", "indoor"],
})

# 4) Cross-modal recall: query with TEXT, retrieve IMAGE
text_inputs = processor(text=["a photo of a cat"], return_tensors="pt",
                        padding=True)
text_emb    = clip.get_text_features(**text_inputs)    # [1, 512]

scores, ids = plugin.episodic.retrieve(text_emb, k=5)
```

**Storage cost per slot**: `512 × 4 bytes = 2 KB`. For 1000 slots, ≈ **2 MB** raw.

---

### 5.3 Audio (CLAP)

```python
import torch
import torchaudio
from transformers import ClapProcessor, ClapModel
from mathir_lib import MATHIRPluginV7

# 1) Load CLAP (audio + text in same 512-dim space)
model_id  = "laion/clap-htsat-unfused"
clap      = ClapModel.from_pretrained(model_id)
processor = ClapProcessor.from_pretrained(model_id)

# 2) MATHIR
plugin = MATHIRPluginV7(embedding_dim=512)

# 3) Load audio (CLAP expects 48 kHz mono)
waveform, sr = torchaudio.load("voice_memo.wav")
waveform = torchaudio.functional.resample(waveform, sr, 48000).mean(0, keepdim=True)

# 4) Encode audio
inputs    = processor(audios=waveform.numpy(),
                      sampling_rate=48000, return_tensors="pt")
audio_emb = clap.get_audio_features(**inputs)         # [1, 512]

plugin.perceive(audio_emb)
plugin.store({
    "embedding":  audio_emb,
    "modality":   "audio",
    "duration_s": waveform.shape[-1] / 48000,
    "speaker":    "alice",
})

# 5) Cross-modal query: text → audio
text_inputs = processor(text=["a person laughing"], return_tensors="pt")
query_emb   = clap.get_text_features(**text_inputs)    # [1, 512]
scores, ids = plugin.episodic.retrieve(query_emb, k=5)
```

**Storage cost per slot**: `512 × 4 bytes = 2 KB`.

---

### 5.4 Video (CLIP per frame + mean pooling)

```python
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from mathir_lib import MATHIRPluginV7

# 1) CLIP
model_id    = "openai/clip-vit-base-patch32"
clip        = CLIPModel.from_pretrained(model_id)
processor   = CLIPProcessor.from_pretrained(model_id)

# 2) MATHIR
plugin = MATHIRPluginV7(embedding_dim=512)

# 3) Extract one embedding per frame, then mean-pool
cap        = cv2.VideoCapture("dashcam.mp4")
fps        = cap.get(cv2.CAP_PROP_FPS)
frame_step = int(fps)            # 1 frame per second
frame_embs = []
frame_idx  = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % frame_step == 0:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil      = Image.fromarray(rgb)
        inputs   = processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            emb = clip.get_image_features(**inputs)  # [1, 512]
        frame_embs.append(emb)
    frame_idx += 1
cap.release()

# 4) Aggregate (mean, max, or transformer-based temporal pooling)
video_emb = torch.stack(frame_embs).mean(dim=0)        # [1, 512]
# video_emb = torch.stack(frame_embs).max(dim=0).values  # alternative

plugin.perceive(video_emb)
plugin.store({
    "embedding": video_emb,
    "modality":  "video",
    "n_frames":  len(frame_embs),
    "duration_s": frame_idx / fps,
    "source":    "dashcam",
})

# 5) Cross-modal query: text → video
text_inputs = processor(text=["a pedestrian crossing the street"],
                        return_tensors="pt", padding=True)
query_emb   = clip.get_text_features(**text_inputs)
scores, ids = plugin.episodic.retrieve(query_emb, k=5)
```

**Storage cost per slot**: `512 × 4 bytes = 2 KB` regardless of video length (the embedding is a *summary*).

> **Tip for long videos**: instead of one mean embedding, store **N key-frame embeddings** (e.g. scene change detection) and let the episodic tier's k-NN do the rest.

---

### 5.5 Multimodal (text + image, aligned in the same space)

CLIP is special: it embeds **both** text and image into the **same 512-dim vector space**, which means you can store and retrieve across modalities directly.

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from mathir_lib import MATHIRPluginV7

clip      = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
plugin    = MATHIRPluginV7(embedding_dim=512)

# 1) Encode text and image with the SAME model
image  = Image.open("cat.jpg").convert("RGB")
img_inputs  = processor(images=image, return_tensors="pt")
img_emb     = clip.get_image_features(**img_inputs)        # [1, 512]

text_inputs = processor(text=["a cat on a mat"],
                        return_tensors="pt", padding=True)
text_emb    = clip.get_text_features(**text_inputs)         # [1, 512]

# 2) Store BOTH in the same MATHIR (they are commensurable)
plugin.store({"embedding": img_emb,  "modality": "image",
              "image_id":  "cat.jpg"})
plugin.store({"embedding": text_emb, "modality": "text",
              "content":   "a cat on a mat"})

# 3) Query with text, retrieve images (or vice-versa)
query_emb   = clip.get_text_features(
    **processor(text=["an animal on furniture"],
                return_tensors="pt", padding=True))
scores, ids = plugin.episodic.retrieve(query_emb, k=5)
#   → may return BOTH the image memory and the text memory
```

---

### 5.6 Multi-modal fusion (CLIP + CLAP + text)

When you mix encoders whose output dimensions differ (e.g. CLIP 512 + CLAP 512 + Whisper 1280), you need a **projection** to a common space.

```python
import torch
import torch.nn as nn
from mathir_lib import MATHIRPluginV7


class MultimodalProjector(nn.Module):
    """Project heterogeneous embeddings into a common D-dim space."""
    def __init__(self, dims, common_dim=512):
        super().__init__()
        self.projs = nn.ModuleDict({
            name: nn.Linear(d, common_dim) for name, d in dims.items()
        })
        self.common = common_dim

    def forward(self, x: torch.Tensor, source: str) -> torch.Tensor:
        return self.projs[source](x)


# CLIP (512), CLAP (512), Whisper (1280), text (384) → 512
projector = MultimodalProjector(
    dims={"clip": 512, "clap": 512, "whisper": 1280, "text": 384},
    common_dim=512,
)
plugin = MATHIRPluginV7(embedding_dim=512)

# Encode each modality
clip_emb   = clip.get_image_features(**img_inputs)            # [1, 512]
clap_emb   = clap.get_audio_features(**audio_inputs)          # [1, 512]
whisper_emb = whisper_encoder(wav)                            # [1, 1280]
text_emb   = text_encoder.encode(...)                          # [1, 384]

# Project into the common space
clip_512   = projector(clip_emb,    source="clip")
clap_512   = projector(clap_emb,    source="clap")
whisp_512  = projector(whisper_emb, source="whisper")
text_512   = projector(text_emb,    source="text")

# All in the same 512-dim space now → store and compare freely
for emb, mod in [(clip_512, "image"), (clap_512, "audio"),
                 (whisp_512, "audio_as_text"), (text_512, "text")]:
    plugin.store({"embedding": emb, "modality": mod})
```

> **Alternative**: use a CLIP-class unified encoder (LLaVA, BLIP-2) and skip the projector entirely. MATHIR is agnostic to which route you take.

---

## 6. What MATHIR Stores (the answer to "how does it store data?")

### 6.1 The per-slot record

When you call `plugin.store({"embedding": emb, ...})`, MATHIR populates its internal buffers. The **public contract** is the dict you pass; the **private state** is a set of `torch.Tensor` buffers indexed by slot.

| Field | Required? | Source | Purpose |
|-------|:---------:|--------|---------|
| `embedding` | **yes** | user | The vector to memorize. The *only* mathematically meaningful field. |
| `modality` | optional | user | `"text" / "image" / "audio" / "video" / …` — pure metadata, never inspected by MATHIR. |
| `text` | optional | user | Raw text. Used by `HybridEpisodicMemory` for **BM25 sparse retrieval** alongside dense cosine. |
| `text_id`, `image_id`, `page`, `tags` | optional | user | Any structured metadata, returned with recall results. |
| `timestamp` | auto | internal | Set on insert (for Ebbinghaus forgetting). |

**Inside MATHIR (V6 + V7 default):**

- The `embedding` is passed through `input_proj` to `internal_dim=272` (or stored raw if `use_raw_embedding=True`).
- Episodic tier stores `key = encoder(value)[:64]` and `value = value` (1000 slots).
- Semantic tier runs online k-means; only the 256 centroids persist.
- Immunological tier updates the mean and covariance of the "normal" distribution (100 slots).

### 6.2 Storage format per tier

| Tier | Shape | Type | Algorithm | What it remembers |
|------|-------|------|-----------|-------------------|
| **Working** | `[64, 272]` | `float32` | circular buffer + multi-head attention | last 64 items verbatim |
| **Episodic** | `[1000, 272]` keys + `[1000, 272]` values | `float32` | k-NN with auto-encoder keys | up to 1000 most recent items |
| **Semantic** | `[256, 64]` prototypes | `float32` | online k-means (FA-style) | 256 cluster centroids |
| **Immunological** | `[100, 272]` bank + `[272, 272]` Σ⁻¹ | `float32` | Mahalanobis distance | 100 "normal" patterns + their covariance |
| **Sparse (V7)** | `[1088, 272]` dictionary | `float32` | ISTA sparse coding | a compressed *basis* for the dataset |

### 6.3 Storage size per embedding (concrete bytes)

| Embedding | Dim | Bytes / slot (float32) | 1000 slots | 1000 slots w/ raw 768-d text |
|-----------|----:|-----------------------:|-----------:|-----------------------------:|
| sentence-transformers MiniLM | 384 | 1.5 KB | 1.5 MB | ~ 3 MB |
| CLIP ViT-B/32 | 512 | 2.0 KB | 2.0 MB | ~ 3.5 MB |
| CLIP ViT-L/14 | 768 | 3.0 KB | 3.0 MB | ~ 5 MB |
| Whisper-large-v3 (post-proj) | 1024 | 4.0 KB | 4.0 MB | ~ 5 MB |
| LLaMA-3 hidden state | 4096 | 16 KB | 16 MB | ~ 17 MB |
| **V7 sparse coding (9.3× compression)** | — | ~ 0.21 KB equiv. | **~ 210 KB** | — |

So a 1000-slot memory bank at CLIP 512-d costs **2 MB of GPU/CPU memory**, plus negligible overhead for the V7 projector (≈ 200 K parameters). On a single 8 GB GPU, you can host tens of millions of slots in principle (though indexing time becomes the bottleneck before memory does).

---

## 7. The Multimodal Memory Pattern (best practice)

A production-grade multimodal memory agent looks like this:

```python
from __future__ import annotations
from typing import Optional, Dict, Any, List

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, ClapProcessor, ClapModel
from mathir_lib import MATHIRPluginV7


class MultimodalMemoryAgent:
    """
    Modality-agnostic memory agent.

    - Text:  sentence-transformers  (384-d)
    - Image: CLIP                   (512-d, aligned with text)
    - Audio: CLAP                   (512-d, aligned with text)

    The MATHIR plugin is configured for the COMMON dim (512).
    Text-only memories are projected from 384 → 512 once on insert.
    """

    COMMON_DIM = 512
    TEXT_DIM   = 384

    def __init__(self, device: str = "cpu"):
        self.device = device

        # Encoders
        self.text_encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
        self.clip      = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32").to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.clap      = ClapModel.from_pretrained(
            "laion/clap-htsat-unfused").to(device).eval()
        self.clap_proc = ClapProcessor.from_pretrained(
            "laion/clap-htsat-unfused")

        # MATHIR for the common dim
        self.plugin = MATHIRPluginV7(embedding_dim=self.COMMON_DIM).to(device)

        # Optional projection (text 384 → 512) when comparing text to CLIP space
        self.text_to_clip = torch.nn.Linear(self.TEXT_DIM, self.COMMON_DIM)

    # ──────────── encoders ────────────
    def _encode_text(self, text: str) -> torch.Tensor:
        emb = self.text_encoder.encode([text], convert_to_tensor=True)  # [1, 384]
        return self.text_to_clip(emb)                                    # [1, 512]

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        inp = self.clip_proc(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.clip.get_image_features(**inp)                   # [1, 512]

    def _encode_audio(self, audio_array: torch.Tensor, sr: int = 48000) -> torch.Tensor:
        inp = self.clap_proc(audios=audio_array.cpu().numpy(),
                             sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.clap.get_audio_features(**inp)                   # [1, 512]

    # ──────────── public API ────────────
    def remember(
        self,
        *,
        text:  Optional[str]      = None,
        image: Optional[Image.Image] = None,
        audio: Optional[torch.Tensor] = None,
        audio_sr: int              = 48000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store any modality in memory."""
        if text is not None:
            emb, modality = self._encode_text(text), "text"
        elif image is not None:
            emb, modality = self._encode_image(image), "image"
        elif audio is not None:
            emb, modality = self._encode_audio(audio, audio_sr), "audio"
        else:
            raise ValueError("Provide at least one of text/image/audio.")

        self.plugin.perceive(emb)
        self.plugin.store({
            "embedding": emb,
            "modality":  modality,
            "metadata":  metadata or {},
            **({"text": text} if text else {}),     # for hybrid BM25
        })

    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Recall memories matching a text query (cross-modal)."""
        emb = self._encode_text(query)
        scores, ids = self.plugin.episodic.retrieve(emb, k=k)
        return [{"id": int(i), "score": float(s)} for s, i in zip(scores, ids)]

    def find_similar_images(self, query_image: Image.Image, k: int = 5):
        emb = self._encode_image(query_image)
        return self.plugin.episodic.retrieve(emb, k=k)

    def find_similar_audio(self, query_audio: torch.Tensor, k: int = 5):
        emb = self._encode_audio(query_audio)
        return self.plugin.episodic.retrieve(emb, k=k)


# ──────────── usage ────────────
agent = MultimodalMemoryAgent(device="cuda" if torch.cuda.is_available() else "cpu")

# Store a mix of modalities
agent.remember(text="The Eiffel Tower is 330 m tall.",
               metadata={"source": "wikipedia"})
agent.remember(image=Image.open("eiffel.jpg"),
               metadata={"source": "personal_photo"})
agent.remember(audio=torch.randn(1, 48000 * 3),    # 3 s dummy
               metadata={"speaker": "alex"})

# Cross-modal recall
results = agent.recall("tall iron structure in Paris")   # may hit both text & image
print(results)
```

This is the **canonical pattern**: one MATHIR, multiple encoders, all projections done *outside* MATHIR.

---

## 8. The Hybrid Retrieval Limitation

`HybridEpisodicMemory` is the recommended episodic backend (Approach D in MATHIR's benchmark suite). It combines:

- **Dense retrieval**: cosine similarity in the embedding space (works for *any* modality).
- **BM25 sparse retrieval**: classic term-frequency ranking on the stored `text` field (works **only** for text).
- **Cross-encoder re-rank** (optional, top-K only): high-precision re-scoring.

The implication is concrete:

> **For image / audio / video memories, you should also store a text description** so BM25 has something to index.

Practical patterns:

1. **Caption the modality at insert time** (e.g. BLIP-2 for images, Whisper for audio, video captioning model for video) and pass the caption as the `text` field.
2. **Use CLIP itself to align the embedding to text space** — the dense cosine already does cross-modal retrieval, so BM25 is a *complement*, not a requirement.
3. **Skip the hybrid backend** entirely for purely non-text workloads and use the standard `EpisodicMemory` (dense-only).

```python
# Example: captioning before store
from transformers import BlipProcessor, BlipForConditionalGeneration
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def caption(image: Image.Image) -> str:
    inp = proc(image, return_tensors="pt")
    out = blip.generate(**inp, max_new_tokens=30)
    return proc.decode(out[0], skip_special_tokens=True)

# Store with caption
agent.remember(
    image=picture,
    metadata={"caption_source": "BLIP-base"},
)
# but ALSO pass `text=` if you want BM25
caption_text = caption(picture)
agent.plugin.store({
    "embedding": img_emb,
    "modality":  "image",
    "text":      caption_text,   # ← enables BM25 hybrid retrieval
})
```

---

## 9. Use Cases (real-world)

| Use case | Modalities | Query | Encoders |
|----------|------------|-------|----------|
| **Multimodal RAG over PDFs** | text + image (charts, diagrams) | "Show me documents with a chart of Reynolds number" | CLIP on images + sentence-transformers on text |
| **Conversational memory with voice** | text + audio | "What did Alice say 3 conversations ago?" | CLAP on audio + sentence-transformers on text |
| **Autonomous-driving scene memory** | video (multi-camera) | "Find the scene where the car detected a pedestrian" | per-frame CLIP + mean-pool, or VideoCLIP |
| **Cross-modal image search** | image ↔ text | "Find images similar to this text description" | CLIP (text and image share space) |
| **Music recommendation** | audio | "Play me tracks that sound like this one" | CLAP audio + text labels |
| **Scientific paper memory** | text + figures (images) | "What did Section 3.2 say about attention?" | CLIP on figures + text encoder on body |
| **E-commerce visual catalog** | image + text (descriptions) | "Show me red leather sofas under $500" | CLIP on product photos + metadata text |
| **Medical imaging + reports** | image (X-ray) + text (radiology report) | "Find past cases of left-lung opacity" | BiomedCLIP / PubMedCLIP |
| **Robotics policy memory** | video + proprio | "Show me successful pick-and-place episodes" | VideoCLIP + sensor encoder |
| **Personal assistant ("Rewind")** | screen captures, audio, text | "What was on my screen during yesterday's standup?" | CLIP (screen), CLAP (audio), text (chat) |

All of these share the same skeleton: *encode → store → encode(query) → recall*.

---

## 10. Limitations & Future Work

### Current limitations (V7)

1. **Per-modality encoders**: the user is responsible for choosing and running the encoder for each modality. MATHIR is a *memory*, not a *perception* layer.
2. **Heterogeneous spaces require an external projector** (see §5.6). Cross-modal recall between CLIP and CLAP works out-of-the-box; mixing CLAP (audio) with sentence-transformers (text) requires either a learned bridge or restricting queries to a single space.
3. **HybridEpisodicMemory** needs text for BM25 (see §8).
4. **Video** is summarized by a single mean-pooled embedding by default; fine-grained temporal queries (e.g. "the *exact* moment a pedestrian appears") need either key-frame indexing or a temporal transformer on top.
5. **GPU acceleration** dramatically changes the throughput profile: CLIP-B/32 encodes ~50 images/s on CPU, ~500/s on a mid-range GPU. Long-video processing is GPU-bound.
6. **Storage of raw media** is *not* MATHIR's job. MATHIR stores the *embedding* and the user-supplied metadata. If you need to retrieve the original file, store it separately (object store, filesystem) and reference it from the metadata.
7. **Fusion is the caller's job**: MATHIR does not currently perform *intra*-sample fusion (e.g. text + image → one vector). It can *store* and *retrieve* a multimodal sample, but the fusion step is upstream.

### V8 roadmap (planned)

- **Unified multimodal encoder support**: first-class adapter for LLaVA-1.5, BLIP-2, Idefics-2, and similar models that produce a single 4096-d "any-to-any" embedding. MATHIR's internal dim becomes configurable per-sample rather than per-plugin.
- **Native temporal memory**: a 5th tier that stores *trajectories* of embeddings (variational state-space model) for video and streaming use-cases, enabling "what happened between t=12s and t=15s?" queries.
- **Learned cross-modal bridges**: an optional adapter that maps between text-only encoders (e.g. e5-large, 1024-d) and CLIP-class spaces, trained on the user's own data via InfoNCE.
- **Multi-vector retrieval (ColBERT-style)**: replacing the single-vector per slot with a small set of late-interaction vectors, useful for long documents and long videos.
- **On-disk tier**: an HNSW-backed disk store for episodic memory > 1 M slots, while keeping working_memory/semantic/procedural + immunological anomaly bank in RAM for low latency.

---

## Appendix A — Choosing a dimension

| Encoder family | Typical dim | MATHIR config | Notes |
|----------------|------------:|---------------|-------|
| sentence-transformers / SBERT | 384, 768, 1024 | `MATHIRPluginV7(384)` etc. | Most text workloads. |
| OpenAI CLIP ViT-B/32 | 512 | `MATHIRPluginV7(512)` | Text + image aligned. |
| OpenAI CLIP ViT-L/14 | 768 | `MATHIRPluginV7(768)` | Higher quality, 2.5× slower. |
| OpenCLIP (large) | 768, 1024 | `MATHIRPluginV7(...)` | Open weights, many sizes. |
| SigLIP | 768, 1152 | `MATHIRPluginV7(...)` | Sigmoid loss, better at scale. |
| BLIP-2 (ITC head) | 256, 512 | `MATHIRPluginV7(...)` | Image-text contrastive head only. |
| CLAP | 512 | `MATHIRPluginV7(512)` | Audio-text aligned. |
| Whisper (encoder) | 512, 768, 1280 | `MATHIRPluginV7(...)` | Audio → embedding; pair with a caption for hybrid. |
| LLaMA / Mistral hidden | 2048, 4096, 8192 | `MATHIRPluginV7(4096)` | Embedding *of* the LLM's own state. |
| DINOv2 | 384, 768, 1024 | `MATHIRPluginV7(...)` | Vision-only self-supervised. |

**Rule of thumb**: pick a single `D` that all your encoders can be projected to, and stick to it across the agent. Mixing `D` in the same plugin is not supported (intentionally — different `D`s mean different geometries).

---

## Appendix B — Quick FAQ

**Q: Does MATHIR have its own CLIP / Whisper / etc. inside?**
A: No. MATHIR is encoder-agnostic. You bring the encoder.

**Q: Can I store an embedding in MATHIR and use a *different* encoder for the query?**
A: Only if both encoders map to the **same vector space** (e.g. CLIP text → CLIP image). Otherwise cosine similarity is meaningless.

**Q: Does MATHIR do on-the-fly modality fusion?**
A: No. The caller fuses (concat, weighted sum, projector, etc.) and stores the result.

**Q: How large can the memory grow?**
A: Episodic is fixed at 1000 by default (configurable). For larger banks, V7's `SparseCodingMemory` gives ~9.3× compression. V8 will add disk-backed HNSW.

**Q: Does storing multimodal data require a different plugin?**
A: No. The same `MATHIRPluginV7` instance handles any modality. The `modality` field is user metadata, not a code path.

**Q: What if my embedding dim changes mid-run?**
A: Rebuild the plugin. The internal projection layer (`input_proj`) is sized at construction.

**Q: Is MATHIR a vector database?**
A: It contains one (episodic) but is much more: it has a router, a working memory, semantic prototypes, an anomaly detector, and (in V7) sparse coding, Ebbinghaus forgetting, and variational uncertainty. See `MASTER_RESEARCH_PAPER.md` for the theoretical treatment.

---

## Appendix C — One-page summary

```
┌────────────────────────────────────────────────────────────────┐
│  MATHIR  is  MODALITY-AGNOSTIC                                 │
│                                                                 │
│  Raw modality  →  ENCODER  →  vector  →  MATHIR  →  vector     │
│   📝 text       MiniLM        384-d        │         384-d      │
│   🖼️ image      CLIP          512-d        │         512-d      │
│   🔊 audio      CLAP          512-d        │         512-d      │
│   🎬 video      CLIP/frame    512-d        │         512-d      │
│   📊 tabular    TS2Vec         64-d        │          64-d      │
│                                                                 │
│                ┌──────────────────────────┐                    │
│                │  MATHIR = 5 tiers         │                    │
│                │  • working   (64 slots)   │  ~   70 KB         │
│                │  • episodic  (1000 slots) │  ~  1.1 MB         │
│                │  • semantic  (256 proto)  │  ~   16 KB         │
│                │  • procedural (128 slots) │  ~   14 KB         │
│                │  • immune    (100 normal) │  ~  109 KB         │
│                │  V7: sparse coding        │  ~9.3× compress    │
│                └──────────────────────────┘                    │
│                                                                 │
│  Pipeline:                                                      │
│     encode(x) → plugin.perceive(emb) → plugin.store({...})     │
│     encode(q) → plugin.recall(emb, k)  → top-k memories         │
│                                                                 │
│  Best practice:                                                 │
│     • One MATHIR, many encoders (all project to same D)         │
│     • Always set `modality` for clarity (not required)         │
│     • For HybridRetrieval, store a `text` description           │
│     • Use CLIP/CLAP for cross-modal queries (aligned spaces)    │
│     • Use V7 sparse coding for memory > 100k slots              │
└────────────────────────────────────────────────────────────────┘
```

---

*Last updated: 2026-06-02 · MATHIR V7 · See `MASTER_RESEARCH_PAPER.md` for the underlying theory and `V7_PAPER.md` for the V7-specific advances (Ebbinghaus, sparse coding, variational, cross-attention, hyperbolic, InfoNCE, Neural-ODE, Mahalanobis immune).*
