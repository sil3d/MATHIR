# SHARED CONTEXT — Vision Model Accuracy Testing

Started: 2026-06-06

## USER GOAL
Test if our vision models are ACCURATE for real tasks:
- Object recognition
- Color recognition
- Person/face type detection
- Segmentation
- Counting
- Multi-task queries

## MODELS TO DOWNLOAD (from HuggingFace)
- https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF
- https://huggingface.co/unsloth/Qwen3.5-2B-GGUF
- https://huggingface.co/yuuko-eth/LocateAnything-3B-GGUF
- https://huggingface.co/nvidia/LocateAnything-3B
- https://huggingface.co/models?sort=trending&search=Falcon+Perception

## REFERENCE IMPLEMENTATIONS
- https://github.com/PromtEngineer/Gemma4-Visual-Agent/tree/dgx-spark-gb10
- https://github.com/nextlevelbuilder/ui-ux-pro-max-skill (UI skill)

## REQUIREMENTS
- NO OpenAI / Gemini APIs (use turboquant + KV cache ourselves)
- Test each model independently
- Multi-model orchestration for complex queries
- Smart architecture (e.g., use LocateAnything for segmentation, LFM2.5-VL for description)
- Color recognition, person type detection

## EXISTING INTELLIGENCE
- vision_testing/ has working LFM2.5-VL-1.6B (vision) + LFM2.5-Audio-1.5B (audio)
- 8.5GB VRAM (RTX 4060 Laptop) - must be careful with model sizes
- MATHIR memory integration works
- ui_server.py has /api/chat, /api/camera/ask, /api/camera/count endpoints
- turboquant binaries in vision_testing/bin/

## INTELLIGENCE TABLE
| Agent | Finding | Affects | Status |
|-------|---------|---------|--------|

## ACTIVE AGENTS
- @background-researcher: Research all HF models, capabilities, sizes
- @internet_search: Research the Gemma4-Visual-Agent implementation patterns
- @coder: Download all Q4_0 models in parallel
- @make: Build accuracy test framework
- @refactor: Refactor for multi-model orchestration
- @check: Verify no hardcoded questions, no API keys