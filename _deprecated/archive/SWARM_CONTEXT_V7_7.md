# SHARED CONTEXT — Vision Testing UI Cleanup + Context Injection

Started: 2026-06-06

## USER MANDATE (CRITICAL)
1. **STOP previous UX work** — it had hardcoded questions like "What do you see in this image?"
2. **NO hardcoded questions anywhere** — user types all questions
3. **Inject MATHIR context into EVERY model** — system prompt that tells the model:
   - What it is (vision/audio model connected to MATHIR)
   - What MATHIR is (hybrid memory architecture)
   - What we want to achieve (test memory across architectures)
   - How to behave (no lying, structured responses)
4. **Structured responses** — every assistant message should be ## What / ## Why / ## How / ## Confidence
5. **Add Learning graphs** — memory usage, model performance, etc.
6. **Add Tests view** — show all the benchmarks we ran
7. **Multi-modal support** — some models handle text+image+audio
8. **Add test_results.json** — all our test results in one file

## EXISTING INTELLIGENCE
- vision_testing/ has working config.json + ui_config.json (no hardcoded paths)
- Web UI at /ui/ has 5 views (chat, camera, models, memory, settings)
- 2 models downloaded: LFM2.5-VL-1.6B, LFM2.5-Audio-1.5B
- MATHIR memory integrated via universal_recall
- 137/137 tests pass for cross-provider/cross-lingual
- 4/4 Ollama models tested (3 worked, 1 timeout)
- 4/27 OpenRouter free models worked

## ACTIVE AGENTS
- @coder: Write test_results.json, create system_context.json
- @debugger: Find any remaining hardcoded questions/paths
- @refactor: Clean UI code, add structured response rendering
- @make: Build learning graphs + tests view + multi-modal support
- @check: Verify no hardcoded questions

## DELIVERABLES
1. test_results.json — ALL our benchmark results
2. system_context.json — Injected into every model chat
3. NO hardcoded questions in any file
4. UI with Learning + Tests views
5. Structured response rendering (markdown)