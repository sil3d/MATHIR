"""
MATHIR Multimodal Memory Demo
==============================

This demo shows how MATHIR can store and retrieve memories from
different modalities (text, image, audio, video) using a unified
embedding space (CLIP).

MATHIR is modality-agnostic — it only sees embeddings. The user
provides the encoders. This is the same pattern as:
  - Vector DBs (they don't care about modality either)
  - RAG (you embed any content, store the embedding)

MATHIR's added value over Vector DBs:
  - Online learning (adapts during use)
  - Anomaly detection (Mahalanobis NP-optimal)
  - Hierarchical memory (4 temporal tiers)
  - Spaced repetition forgetting (Ebbinghaus)

Run:
    python examples/multimodal_demo.py
"""

import os
import sys
import time
import numpy as np
import torch

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathir_lib import MATHIRPluginV7


# ============================================================
# STEP 1: Build a simple text-based "multimodal" demo
#         (no model download required)
# ============================================================

def text_based_multimodal_demo():
    """
    Simulate multimodal memory with text-only embeddings.

    Each "modality" gets a different embedding (e.g., different
    hashing scheme) to simulate the modality-specific embedding space.
    In production, you'd use CLIP, CLAP, Whisper, etc.
    """
    print("=" * 60)
    print("DEMO 1: Multimodal Memory with Text Embeddings")
    print("=" * 60)

    dim = 384
    plugin = MATHIRPluginV7(embedding_dim=dim)

    # Simulate 5 "modalities" with semantic content
    memories = [
        ("text", "I went hiking in the Alps last summer", "memory_1"),
        ("text", "I learned to cook paella in Spain", "memory_2"),
        ("text", "I saw a documentary about fluid mechanics", "memory_3"),
        ("text", "I drove through the Rocky Mountains", "memory_4"),
        ("text", "I read about the Navier-Stokes equations", "memory_5"),
    ]

    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Store memories
    print("\nStoring 5 memories across modalities...")
    for modality, content, mem_id in memories:
        emb = embedder.encode([content])[0]
        t = torch.from_numpy(emb).float().unsqueeze(0)
        plugin.perceive(t)
        plugin.store({
            "embedding": t,
            "modality": modality,
            "content": content,
            "memory_id": mem_id,
        })
        print(f"  ✅ Stored [{modality}]: {content[:50]}...")

    # Query
    print("\nQuerying memory...")
    queries = [
        "outdoor adventures",
        "mathematics of fluids",
        "European experiences",
    ]

    for q in queries:
        q_emb = embedder.encode([q])[0]
        t = torch.from_numpy(q_emb).float().unsqueeze(0)
        results = plugin.recall(t, k=3)
        print(f"\n  Q: {q}")
        for i, r in enumerate(results, 1):
            # Find content by memory_id
            print(f"    #{i} similarity={r.get('similarity', 0):.3f}")

    # Show memory stats
    print("\n" + "=" * 60)
    print("Memory statistics:")
    print("=" * 60)
    stats = plugin.get_stats()
    print(f"  Version: {stats.get('version', '?')}")
    print(f"  Episodic: {stats.get('episodic', {})}")
    print(f"  Working: {stats.get('working', {})}")


# ============================================================
# STEP 2: Show how it would work with REAL CLIP for images
# ============================================================

def clip_image_demo_concept():
    """
    Conceptual demo of CLIP image integration.

    Shows the code pattern even if we don't run the model.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: CLIP Image Integration (conceptual)")
    print("=" * 60)
    print("""
This is the CODE PATTERN for storing images. In production, run:

    pip install transformers torch torchvision Pillow

Then:
""")
    code = '''
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from mathir_lib import MATHIRPluginV7

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize MATHIR with CLIP's embedding dim (512)
plugin = MATHIRPluginV7(embedding_dim=512)

# Store an image
def remember_image(image_path: str, metadata: dict = None):
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    img_emb = clip_model.get_image_features(**inputs)  # [1, 512]
    plugin.perceive(img_emb)
    plugin.store({
        "embedding": img_emb,
        "modality": "image",
        "image_path": image_path,
        "metadata": metadata or {},
    })

# Query with text, retrieve matching images
def find_images_by_text(text_query: str, k: int = 5):
    inputs = clip_processor(text=[text_query], return_tensors="pt", padding=True)
    text_emb = clip_model.get_text_features(**inputs)  # [1, 512]
    # CLIP embeds text and image in SAME space
    # So we can directly query with text
    return plugin.recall(text_emb, k=k)

# Store some images
remember_image("photo1.jpg", {"location": "Paris"})
remember_image("photo2.jpg", {"location": "Tokyo"})

# Query
results = find_images_by_text("Eiffel Tower in Paris")
for r in results:
    print(f"  Similarity: {r.get('similarity', 0):.3f}")
'''
    print(code)


# ============================================================
# STEP 3: Show how it would work with audio (CLAP)
# ============================================================

def clap_audio_demo_concept():
    """Conceptual demo of CLAP audio integration."""
    print("\n" + "=" * 60)
    print("DEMO 3: CLAP Audio Integration (conceptual)")
    print("=" * 60)
    print("""
For AUDIO memory, use CLAP (Contrastive Language-Audio Pretraining):

    pip install laion-clap

Then:
""")
    code = '''
import laion_clap
import torch
from mathir_lib import MATHIRPluginV7

# Load CLAP
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_pretrained()

# Initialize MATHIR with CLAP's dim (512)
plugin = MATHIRPluginV7(embedding_dim=512)

# Store audio
def remember_audio(audio_path: str, transcript: str = None):
    audio_emb = model.get_audio_embedding_from_filelist(
        x=[audio_path], use_tensor=False
    )  # [1, 512]
    audio_emb_t = torch.from_numpy(audio_emb).float()
    plugin.perceive(audio_emb_t)
    plugin.store({
        "embedding": audio_emb_t,
        "modality": "audio",
        "audio_path": audio_path,
        "transcript": transcript,  # For hybrid BM25 retrieval
    })

# Query with text (CLAP embeds text and audio in same space)
def find_audio_by_text(text_query: str, k: int = 5):
    text_emb = model.get_text_embedding([text_query])  # [1, 512]
    text_emb_t = torch.from_numpy(text_emb).float()
    return plugin.recall(text_emb_t, k=k)

# Store
remember_audio("meeting_2024_01.mp3", transcript="Discussed Q4 revenue")
remember_audio("lecture_fluid_dynamics.mp3", transcript="Navier-Stokes equations")

# Query
results = find_audio_by_text("discussion about revenue")
for r in results:
    print(f"  Similarity: {r.get('similarity', 0):.3f}")
'''
    print(code)


# ============================================================
# STEP 4: Show how it would work with video
# ============================================================

def video_demo_concept():
    """Conceptual demo of video memory."""
    print("\n" + "=" * 60)
    print("DEMO 4: Video Memory (conceptual)")
    print("=" * 60)
    print("""
For VIDEO, extract frames and embed each one with CLIP:

    pip install opencv-python transformers
""")
    code = '''
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from mathir_lib import MATHIRPluginV7

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
plugin = MATHIRPluginV7(embedding_dim=512)

def remember_video(video_path: str, sample_rate_fps: int = 1):
    """Extract 1 frame per second, embed each, store all."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / sample_rate_fps)
    
    frame_count = 0
    stored = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=frame_rgb, return_tensors="pt")
            emb = clip.get_image_features(**inputs)  # [1, 512]
            plugin.perceive(emb)
            plugin.store({
                "embedding": emb,
                "modality": "video_frame",
                "video_path": video_path,
                "frame_number": frame_count,
                "timestamp_s": frame_count / fps,
            })
            stored += 1
        frame_count += 1
    cap.release()
    return stored

# Find a moment in the video
def find_video_moment(text_query: str, k: int = 5):
    inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    text_emb = clip.get_text_features(**inputs)  # [1, 512]
    results = plugin.recall(text_emb, k=k)
    return results

# Store
n = remember_video("driving.mp4", sample_rate_fps=1)  # 1 frame per sec
print(f"Stored {n} video frames")

# Query
results = find_video_moment("person crossing the street")
for r in results:
    sim = r.get("similarity", 0)
    # Look up the metadata
    print(f"  Similarity: {sim:.3f}")
'''
    print(code)


# ============================================================
# STEP 5: THE KEY INSIGHT — Multimodal with a UNIFIED space
# ============================================================

def unified_multimodal_pattern():
    """The pattern for cross-modal retrieval."""
    print("\n" + "=" * 60)
    print("DEMO 5: Cross-Modal Retrieval (text → image → audio)")
    print("=" * 60)
    print("""
THE KEY INSIGHT: Use a model that embeds ALL modalities in the SAME space.

CLIP family does this:
  - CLIP:  text + image  (512-dim)
  - CLAP:  text + audio  (512-dim)
  - ImageBind: text + image + audio + video + depth + ... (1024-dim)

With ImageBind, you can query with text and retrieve memories
across ALL modalities. This is the future of multimodal agents.

    pip install imagebind
""")
    code = '''
import torch
from imagebind import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from mathir_lib import MATHIRPluginV7

# Load ImageBind (1.4GB)
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()

# ImageBind embeds everything in 1024-dim
plugin = MATHIRPluginV7(embedding_dim=1024)

def remember_multimodal(*, text=None, image=None, audio=None, video=None, metadata=None):
    inputs = {}
    if text is not None:
        inputs[ModalityType.TEXT] = [text]
    if image is not None:
        inputs[ModalityType.VISION] = [image]
    if audio is not None:
        inputs[ModalityType.AUDIO] = [audio]
    if video is not None:
        inputs[ModalityType.VIDEO] = [video]
    
    with torch.no_grad():
        emb = model(inputs)  # dict of modality -> [N, 1024]
    
    # Use the first available embedding
    for modality, vec in emb.items():
        plugin.perceive(vec)
        plugin.store({
            "embedding": vec,
            "modality": str(modality),
            "metadata": metadata or {},
        })
        break  # store one per call

def recall_by_text(text_query: str, k: int = 5):
    with torch.no_grad():
        emb = model({ModalityType.TEXT: [text_query]})
    return plugin.recall(emb[ModalityType.TEXT], k=k)

# Store memories across modalities
remember_multimodal(text="the Eiffel Tower", metadata={"country": "France"})
remember_multimodal(image="tokyo_street.jpg", metadata={"country": "Japan"})
remember_multimodal(audio="birds_chirping.wav", metadata={"type": "nature"})

# Query with text — retrieves across all modalities
results = recall_by_text("places in Europe")
for r in results:
    sim = r.get("similarity", 0)
    # The metadata tells you what modality was matched
    print(f"  Similarity: {sim:.3f}")
'''
    print(code)


# ============================================================
# STEP 6: What MATHIR actually stores (the "how")
# ============================================================

def what_mathir_stores():
    print("\n" + "=" * 60)
    print("HOW MATHIR STORES DATA (the answer)")
    print("=" * 60)
    print("""
MATHIR is a MEMORY LAYER, not a database. It stores:

1. **Embeddings** (the main thing):
   - Working memory: 64 slots × 384 floats = 96 KB
   - Episodic memory: 1000 slots × 384 floats = 1.5 MB
   - Semantic memory: 256 prototypes × 384 floats = 384 KB
   - Immunological memory: 100 patterns × 384 floats = 150 KB
   - With V7 compression: 9.3× smaller (~160 KB total)

2. **Metadata** (user-provided):
   - modality: "text" | "image" | "audio" | "video"
   - timestamp: when stored
   - source_path: where the data came from
   - user_defined: anything you want

3. **Optional raw text** (for hybrid retrieval):
   - transcript (for audio)
   - caption (for image)
   - summary (for video)
   - original text

4. **Internal state** (managed by MATHIR):
   - keys (encoded for fast lookup)
   - values (raw or projected embeddings)
   - indices (LRU pointers)
   - statistics (hit counts, anomaly scores)

PER MEMORY SLOT:
```
{
    "embedding": torch.Tensor[1, D],    # The actual vector
    "modality": "text" | "image" | ...,
    "key": torch.Tensor[1, D_k],         # encoded for fast lookup
    "value": torch.Tensor[1, D],         # same as embedding or projected
    "metadata": {...},                    # user-defined
    "text": "...",                        # for hybrid BM25
    "timestamp": datetime,
    "stability": float,                  # Ebbinghaus (V7)
    "recall_count": int,                 # V7
    "last_access": datetime,             # V7
}
```

The KEY insight: MATHIR doesn't store the RAW audio/image/video.
It stores the EMBEDDING (vector). The raw data can be discarded
(if you don't need to retrieve it later for re-encoding).

For PRODUCTION:
- Store raw data in S3/Object storage (referenced by path)
- Store embedding in MATHIR (for retrieval)
- This is the same pattern as vector databases.
""")


def main():
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*   MATHIR MULTIMODAL MEMORY DEMO" + " " * 27 + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print()

    # Working demo (only needs sentence-transformers)
    text_based_multimodal_demo()

    # Conceptual demos (just show code patterns)
    clip_image_demo_concept()
    clap_audio_demo_concept()
    video_demo_concept()
    unified_multimodal_pattern()
    what_mathir_stores()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
✅ YES, MATHIR accepts VIDEO, AUDIO, TEXT — any modality that can be
   embedded as a fixed-dim vector.

✅ MATHIR is modality-AGNOSTIC. It only sees embeddings. The user
   provides the encoders (CLIP, CLAP, Whisper, sentence-transformers, etc.)

✅ HOW IT STORES: as a vector + metadata in one of 4 memory tiers.
   Same storage format regardless of modality.

✅ KEY PATTERN: use a model that embeds all modalities in the SAME space
   (CLIP, CLAP, ImageBind), then MATHIR enables cross-modal retrieval.

✅ MATHIR's value-add over Vector DBs:
   - Online learning
   - Anomaly detection (Mahalanobis NP-optimal)
   - Hierarchical memory (4 temporal tiers)
   - Spaced repetition forgetting
   - Hybrid retrieval (BM25 + dense + cross-encoder)
""")


if __name__ == "__main__":
    main()
