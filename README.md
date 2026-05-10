# Alternative RAG Evaluation Framework
### Gates Foundation AI Fellowship — India 2026 | Technical Assignment: Option B

This repository contains a critique of the [CeRAI AIEvaluationTool](https://github.com/cerai-iitm/AIEvaluationTool) and a working alternative evaluation framework built using [DeepEval](https://github.com/confident-ai/deepeval).

The system under evaluation is **DIGIT Studio Assistant** — a RAG chatbot built on the DIGIT platform. Source: [srujana-egov/EGOV_RAG_V5](https://github.com/srujana-egov/EGOV_RAG_V5).

**Live report:** https://evaluationreportpy-ecqdeuaymaoycteqpxhs9v.streamlit.app/

---

## Why Option B

Attempting to install CeRAI via `docker compose build` on Mac Apple Silicon (M2) failed with `OSError: [Errno 5] Input/output error` before the image could start. The build pulls ~3.5 GB of NVIDIA CUDA packages (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `triton`, etc.) that are architecturally incompatible with aarch64. This is not a disk or configuration issue — the packages target Linux x86_64 with NVIDIA drivers and cannot run on Apple Silicon.

Reading the source code after the installation failure revealed deeper problems than portability. See [cerai_github_issues.md](cerai_github_issues.md) for the full critique and the ten issues filed on the CeRAI repository.

---

## What is broken or insufficient in CeRAI

Ten issues were filed at [github.com/cerai-iitm/AIEvaluationTool/issues](https://github.com/cerai-iitm/AIEvaluationTool/issues). Summary:

| # | Issue | Impact |
|---|---|---|
| 1 | RAG metrics run against SQuAD/CODAH/HotPotQA academic benchmarks, not the bot's actual knowledge base | Scores measure general QA, not domain faithfulness |
| 2 | No adversarial/injection safety testing | Prompt injection and jailbreak resistance are invisible |
| 3 | No limitation-awareness evaluation category | Capability hallucinations go undetected |
| 4 | GPU/RAM requirements exclude the tool's primary audience | Unusable on standard laptops and commodity cloud |
| 5 | Docker build fails on Apple Silicon | Blocks any aarch64 developer before evaluation starts |
| 6 | Silent exception handling makes failures indistinguishable from low scores | Evaluation runs that crash partially look identical to completed runs |
| 7 | GPU service calls have no timeout | Evaluation hangs indefinitely if GPU service is slow or unreachable |
| 8 | BiasDetection scores surface language, not AI-generated bias | False positives on factual reporting of demographic research |
| 9 | ComputeErrorRate returns a raw count, not a rate — with false positives and missed FATAL/CRITICAL | Metric name and implementation disagree; healthy sessions can score non-zero |
| 10 | Compute_MTBF has no failure deduplication | Three error lines from one incident produce sub-second MTBF; crashes on healthy logs |

The most significant finding: CeRAI's "RAG evaluation" (`truth_internal.py`, `hallucination.py`) tests responses against academic benchmark datasets — SQuAD, CODAH, HotPotQA, HaluQA, HaluSumm. For a bot built over a domain-specific knowledge base, this produces scores that are meaningless. The evaluator checks whether the bot knows Wikipedia passage comprehension, not whether it correctly represents the domain it was built for.

---

## Design decisions in this alternative

| Decision | Rationale |
|---|---|
| DeepEval over RAGAS | DeepEval's GEval supports custom evaluation criteria per category, enabling `limitation_awareness` and `adversarial` categories that RAGAS cannot express |
| OpenAI gpt-4o as judge | No GPU required; ~$0.05/run; works on any laptop — matches the environments where most evaluation actually happens |
| HTTP API evaluation | Calls the bot over `POST /chat` rather than importing internal modules, making it portable across any compatible conversational endpoint |
| `limitation_awareness` category | RAG hallucination risk is highest for questions about capabilities that don't exist. No existing framework has this category. One confirmed hallucination was found this way that a generic faithfulness metric missed. |
| `adversarial` category | Government-facing bots are high-value targets for prompt injection. CeRAI has no adversarial coverage. |
| Keyword check alongside LLM judge | Provides a free, deterministic baseline that catches obvious retrieval failures without any API calls |

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/srujana-egov/GatesAlternateAIEvaluatorTool
cd GatesAlternateAIEvaluatorTool
```

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# edit .env and add your key
```

Or export directly:
```bash
export OPENAI_API_KEY=sk-...
```

### 3. Start a conversational endpoint

This evaluation framework calls any bot that exposes `POST /chat` with a JSON body `{"message": "..."}` and returns `{"response": "..."}`.

The endpoint used in this demonstration is the DIGIT Studio Assistant from [EGOV_RAG_V5](https://github.com/srujana-egov/EGOV_RAG_V5):

```bash
# In EGOV_RAG_V5:
uvicorn api:app --port 8001
```

Verify:
```bash
curl http://localhost:8001/health
# → {"status":"ok"}
```

---

## Running the evaluation

```bash
# Keyword checks only (free, no OpenAI calls)
python deepeval_eval.py --no-judge

# Full run with LLM judge (~$0.05, ~5 min)
python deepeval_eval.py

# Single category
python deepeval_eval.py --use-case digit_studio

# Verbose (shows answer previews per question)
python deepeval_eval.py --verbose

# Point at a different endpoint
python deepeval_eval.py --url http://your-server:8001
```

Results are saved to `results/deepeval_results_YYYYMMDD_HHMMSS.json`.

---

## What the evaluation found

Results from the full run are in [`results/deepeval_results_latest.json`](results/deepeval_results_latest.json).

| Category | N | Keyword pass | LLM judge pass | Notes |
|---|---|---|---|---|
| in_domain | 27 | 22% | 77% | 2 questions dropped — 500 server error |
| limitation_awareness | 7 | 0% | 28% | |
| out_of_domain | 5 | 100% | 80% | digit_studio use case only; hcm OOD not run |
| edge_case | 5 | 0% | 57% | |
| adversarial | 5 | 100% | 100% | |

**Key findings:**
- **Confirmed hallucination** (`ds_lim_004`): bot stated that published configurations can be edited. They cannot. LLM judge score: 0.0. A generic faithfulness metric missed this; the `limitation_awareness` GEval caught it.
- **False rejections**: MDMS, inbox search, service/module distinction, and workflow state questions were refused as out-of-domain despite being core in-domain topics. Retrieval gaps in the knowledge base.
- **OOD evasion** (`ds_ood_001`): weather question answered as "DIGIT Studio does not do weather" rather than a clean refusal. OOD classifier missed it; GEval scored it 0.23.
- **Adversarial safety**: 5/5 (100%). The bot correctly resisted all injection and jailbreak attempts.
- **Keyword gap vs LLM judge**: in_domain keyword pass was 22% vs 77% LLM judge pass, demonstrating that keyword-only evaluation significantly underestimates quality.

---

## What this alternative does not do well

- **No multi-turn evaluation** — each question is scored independently; conversational coherence across turns is not tested
- **No WhatsApp or browser-channel evaluation** — only REST API endpoints are supported
- **No load/performance testing** — latency is measured but not stressed at scale
- **No dataset management UI** — golden set is a JSON file; no dashboard for managing test suites across runs
- **GEval is non-deterministic** — scores can vary ±0.05 between runs; average 3 runs for publishable results
- **Faithfulness requires contexts** — the bot endpoint must return retrieved chunks (`include_contexts: true`); if contexts are empty, Faithfulness is skipped
- **OpenAI dependency for LLM judge** — `--no-judge` mode is free but limited to keyword checks only

---

## Viewing the report

Pre-saved results are included in `results/` — the report runs without a live bot or OpenAI key.

```bash
streamlit run evaluation_report.py
```

Opens at `http://localhost:8501`.

---

## Golden set

[`golden_set.json`](golden_set.json) contains 56 test cases across 5 categories:

| Category | N | Description |
|---|---|---|
| `in_domain` | 29 | Questions the bot should answer from its knowledge base |
| `limitation_awareness` | 7 | Capabilities the system does NOT have — highest hallucination risk |
| `out_of_domain` | 10 | Questions the bot should decline |
| `edge_case` | 5 | Ambiguous, broad, or cross-domain questions |
| `adversarial` | 5 | Prompt injection, jailbreaks, domain-wrapped harmful requests |

The `limitation_awareness` category is the novel contribution of this framework. No existing public evaluation framework (CeRAI, RAGAS, DeepEval defaults, TruLens) has an equivalent category.
