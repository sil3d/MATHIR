# SHARED CONTEXT — Universal Cross-Provider Bridge Algorithm

Started: 2026-06-06

## PROBLEM
MATHIR's current cross-provider bridge has 3 critical limitations:
1. FTS5 text search fails on conversational queries ("What do you know about python closures?" → 0 results)
2. Different embedding spaces are incompatible (OpenAI 1536d ≠ Anthropic ≠ Ollama 1024d)
3. No language-agnostic matching (English queries can't find French content)

## GOAL
Design an algorithm that:
- Works with ANY word (not just vocabulary in training data)
- Works with ANY language (cross-lingual retrieval)
- Bridges between different embedding spaces WITHOUT retraining
- Is mathematically grounded (provable bounds)
- Integrates with MATHIR's `recall()` and `recall_text()` APIs

## INTELLIGENCE (findings from agents)

| Agent | Finding | Affects | Status |
|-------|---------|---------|--------|
| @coder | FTS5 with 'porter unicode61' tokenizer does work for some queries | Test A | FOUND |
| @coder | recall_text takes 'query_text' not 'query' parameter | Test A | FOUND |
| @coder | MATHIRMemory.episodic doesn't have all_ids() - use _store.all_ids() | Test B | FOUND |
| @coder | MiniMax API returns empty for some prompts | Test A4 | FOUND |
| @coder | recall_count is tracked but NOT used in recall scoring | Test B3 | FOUND |
| @coder | Cross-provider recall returns 0 - no fallback logic | Test A3 | FOUND |

## ACTIVE AGENTS
- @math: Design universal algorithm (cross-lingual, cross-space, no-retraining)
- @coder: Implement algorithm in mathir_dropin
- @debugger: Find edge cases in cross-provider logic
- @refactor: Clean up existing code while integrating
- @make: Build feature with tests
- @test: Run comprehensive test suite

## CONSTRAINTS
- Must work with existing MATHIRMemory (don't break current API)
- No external API dependencies (must work offline)
- Must handle embedding space dimensions mismatch
- Mathematically grounded with complexity analysis

## BENCHMARKS
- Cross-provider recall should achieve >70% precision@5
- Cross-lingual recall should achieve >60% precision@5
- Query expansion should improve FTS5 recall by >40%
- Algorithm should be O(n log n) for n memories