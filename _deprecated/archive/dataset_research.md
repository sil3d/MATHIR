# SWARM CONTEXT — MATHIR Dataset Research
_@scout — 2026-06-05 — READ ONLY, do not edit_

## Tech Stack Detected
- Not applicable (this is a research task)

## Verification Commands
```bash
# Download NAB dataset
git clone https://github.com/numenta/NAB.git

# Download CoQA dataset
wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json
wget https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json

# Download QuAC dataset
wget https://s3.amazonaws.com/my89public/quac/train_v0.2.json
wget https://s3.amazonaws.com/my89public/quac/val_v0.2.json
```

## Files Related to Task
| File | Role | Suggested Owner |
|------|------|----------------|
| dataset_research.md | This file - identified datasets | @scout |

## Datasets Identified

### 1. Anomaly Detection Dataset

#### NAB (Numenta Anomaly Benchmark)
- **URL**: https://github.com/numenta/NAB
- **Direct data**: https://github.com/numenta/NAB/tree/master/data
- **Description**: Contains labeled anomalies in time series and text logs. Standard benchmark for anomaly detection. Includes artificial times series with labeled anomalies, and real-world log data with labeled anomalies.
- **Format**: CSV, JSON
- **Size**: ~1MB dataset
- **What it tests**: Does immunological memory flag novel/out-of-distribution inputs?

### 2. Online Learning / Episodic Memory Dataset

#### QuAC (Question Answering in Context)
- **URL**: https://quac.ai/
- **Training Set**: https://s3.amazonaws.com/my89public/quac/train_v0.2.json
- **Val Set**: https://s3.amazonaws.com/my89public/quac/val_v0.2.json
- **Description**: Dialog between student (questioner) and teacher (Wikipedia text). Follow-up questions depend on prior context - perfect for testing if storing relevant docs improves future recall.
- **Format**: JSON
- **Size**: Train ~15MB, Val ~3MB
- **What it tests**: Does storing relevant docs improve future recall? Simulate storing Wikipedia passages, then query follow-up questions.
- **Key feature**: Questions are context-dependent across dialog turns

#### CoQA (Conversational Question Answering)
- **URL**: https://stanfordnlp.github.io/coqa/
- **Training Set**: https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json
- **Dev Set**: https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json
- **Description**: 127,000+ questions from 8000+ conversations. Each answer requires understanding prior conversation context. Has coreference and pragmatic reasoning challenges.
- **Format**: JSON
- **Size**: Train ~47MB, Dev ~9MB
- **What it tests**: Episodic memory online learning - storing context improves subsequent question answering

### 3. Working Memory / Context-Dependent Dataset

#### QuAC (reused for working memory)
- **URL**: https://quac.ai/
- **Description**: Dialog questions like "What did he say?" where "he" refers to recent context. Perfect for testing if recent context influences query results.
- **Format**: JSON
- **What it tests**: Does recent context influence query results?

#### CoQA (reused for working memory)
- **URL**: https://stanfordnlp.github.io/coqa/
- **Description**: Coreference-heavy questions depend on immediately prior context. Tests working memory effects.
- **Format**: JSON
- **What it tests**: Working memory - does immediately preceding context affect answers?

### 4. Router Accuracy Dataset

No specific benchmark exists for testing KL router accuracy. For MATHIR, router accuracy would be tested by:
- Injecting synthetic queries with known distribution characteristics
- Measuring if KL divergence routing correctly classifies queries to tiers
- Could use a subset of QuAC/CoQA with manually labeled query types

**Suggested approach for Router Testing**:
- Use QuAC/CoQA with queries manually labeled by expected tier (semantic search, exact match, etc.)
- Compare MATHIR's routing decisions against ground truth

## Existing Tests
- None identified - this is a new project

## Code Patterns Found
- N/A (research task)

## DO NOT TOUCH
- All source code files
- Build artifacts

## Risk Areas
- NAB dataset is time-series focused - may need adaptation for embedding-based anomaly detection
- Router accuracy testing requires custom ground truth labeling