# CeRAI GitHub Issues — Draft

File each section below as a separate issue at:
https://github.com/cerai-iitm/AIEvaluationTool/issues/new

---

## Issue 1 — No support for arbitrary REST API endpoints

**Title:** `feat: add support for arbitrary REST API chatbot endpoints (not just WhatsApp)`

**Labels:** `enhancement`, `architecture`

**Body:**

### Problem

CeRAI's entire evaluation pipeline is built around WhatsApp as the delivery channel. The entry point is a WhatsApp webhook, and the test runner (`test_runner/`) sends prompts via Selenium/browser automation against the WhatsApp Web interface.

This makes it impossible to evaluate a chatbot that exposes a standard REST API (e.g. `POST /chat`) without wrapping it inside a fake WhatsApp integration — an unreasonable requirement for most production deployments.

**Affected files:**
- `test_runner/runner.py` — hardcodes WhatsApp Web URL and Selenium selectors
- `docker-compose.yml` — `whatsapp_connector` service has no REST alternative
- `config/config.json` — `endpoint` field only supports WhatsApp webhook format

### Suggested fix

Add a `transport` field to `config.json`:
```json
{
  "transport": "rest",
  "rest_endpoint": "http://localhost:8001/chat",
  "rest_payload_template": {"message": "{query}"},
  "rest_response_path": "response"
}
```
When `transport: rest`, the test runner should `POST` directly to the endpoint and extract the response via `rest_response_path`, bypassing all Selenium/WhatsApp infrastructure.

### Why this matters

Government AI deployments (DIGIT, BharatGPT, etc.) expose REST APIs. The WhatsApp-first design excludes the majority of production chatbot architectures from being evaluated.

---

## Issue 2 — Judge model requires 32 GB RAM; incompatible with most government hardware

**Title:** `bug: qwen3:32b judge model requires 32 GB RAM — unusable on standard government hardware`

**Labels:** `bug`, `hardware`, `accessibility`

**Body:**

### Problem

CeRAI uses `qwen3:32b` via Ollama as its LLM judge. Running this model requires approximately 32 GB of RAM (FP16 weights alone are ~64 GB; 4-bit quantised is ~18–20 GB but still needs ~28 GB total with KV cache). This exceeds the hardware budget of virtually every government deployment context CeRAI is meant to serve.

**Evidence from docker-compose.yml:**
```yaml
ollama:
  image: ollama/ollama
  # pulls qwen3:32b on first run — requires 28–32 GB free RAM
```

**Concrete failure observed:**
Attempting `docker compose build` on a standard MacBook with 905 MB free disk:
```
OSError: [Errno 5] Input/output error
```
The build failed during pip install because CUDA/GPU libraries alone (cublas 542 MB, cudnn 433 MB, cusparselt 220 MB, torch ~800 MB) exceeded available disk — before any model weights were downloaded.

### Context

DIGIT 3.0 — a major Indian government digital public infrastructure platform — targets deployment on 8 GB RAM machines. An evaluation tool that requires 4× the target machine's RAM to run cannot be used to evaluate software meant for that platform.

### Suggested fix

1. Make the judge model configurable via `config.json`:
   ```json
   { "judge_model": "qwen3:8b" }
   ```
2. Add an OpenAI/Anthropic fallback for cloud-based evaluation (no local GPU required).
3. Document minimum hardware requirements prominently in README.

---

## Issue 3 — EXPECTED_OUTPUT evaluation is brittle and gameable

**Title:** `bug: exact-match EXPECTED_OUTPUT scoring fails correct paraphrased answers and can be gamed`

**Labels:** `bug`, `evaluation-quality`

**Body:**

### Problem

CeRAI compares bot responses against `EXPECTED_OUTPUT` strings in the test config. This fails in two directions:

**False negatives (correct answer marked wrong):**
If a bot answers correctly but with different phrasing, it fails the check.
- Expected: `"The service uses JWT authentication"`
- Bot answer: `"Authentication is handled via JSON Web Tokens"` → FAIL

**False positives (wrong answer passes):**
A bot that always returns a string containing the expected keywords will score 100% regardless of whether the answer is correct in context. This is a direct application of Goodhart's Law.

**Affected file:** `evaluator/evaluator.py` — `check_expected_output()` function

### Suggested fix

Replace (or supplement) keyword matching with semantic similarity:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_match(expected: str, actual: str, threshold: float = 0.75) -> bool:
    emb_e = model.encode(expected)
    emb_a = model.encode(actual)
    score = cosine_similarity([emb_e], [emb_a])[0][0]
    return score >= threshold
```
This catches paraphrased correct answers and is harder to game than keyword stuffing.

---

## Issue 4 — No RAG-specific evaluation metrics (retrieval quality, faithfulness)

**Title:** `feat: add RAG-specific metrics — retrieval hit rate, faithfulness, answer relevancy`

**Labels:** `enhancement`, `rag-evaluation`

**Body:**

### Problem

CeRAI evaluates chatbot *outputs* but has no awareness of whether those outputs are grounded in retrieved documents. For RAG-based systems (which most government AI chatbots use), this misses the most important failure modes:

| Failure mode | CeRAI detects? |
|---|---|
| Hallucinated answer (not in docs) | No |
| Retrieved wrong document | No |
| Answer contradicts retrieved context | No |
| Correct answer but from memory, not docs | No |

**Real-world impact:** A RAG bot that hallucinates a plausible-sounding answer to a policy question will pass CeRAI's evaluation while giving citizens incorrect information.

### Suggested fix

Integrate RAGAS (https://docs.ragas.io/) metrics for RAG systems:

```python
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy

dataset = EvaluationDataset(samples=[
    SingleTurnSample(
        user_input=question,
        response=bot_answer,
        retrieved_contexts=retrieved_docs,
    )
    for question, bot_answer, retrieved_docs in pipeline_outputs
])
result = evaluate(dataset=dataset, metrics=[Faithfulness(), AnswerRelevancy()])
```

This requires the evaluation framework to receive `retrieved_contexts` from the bot — which means the REST API (see Issue #1) should optionally return source documents alongside the answer.

---

## Issue 5 — 8-container Docker stack is too complex to install for average developer/government team

**Title:** `docs/ux: docker-compose stack (8 containers + Selenium + Ollama) is too complex for average deployment`

**Labels:** `documentation`, `developer-experience`, `accessibility`

**Body:**

### Problem

Running CeRAI requires starting 8 Docker containers simultaneously:
- `ollama` (model server, ~20–32 GB RAM)
- `selenium` (browser automation for WhatsApp)
- `whatsapp_connector`
- `evaluator`
- `test_runner`
- `dashboard`
- `redis`
- `postgres`

**Observed failure during install attempt:**
```
OSError: [Errno 5] Input/output error
```
Caused by disk exhaustion during `docker compose build` — pip was downloading 2 GB+ of NVIDIA CUDA libraries (cublas 542 MB, cudnn 433 MB, cusparselt 220 MB, torch ~800 MB) as transitive dependencies. A machine with only 905 MB free disk cannot complete the build.

**The README does not mention:**
- Minimum disk space required (at least 10 GB recommended)
- Minimum RAM required (28–32 GB for the judge model alone)
- That GPU drivers must be installed on the host
- That a WhatsApp account (phone number) is required

For a tool meant to democratise AI evaluation in government, the installation experience is inaccessible to teams that don't have dedicated DevOps support.

### Suggested fixes

1. Add a `REQUIREMENTS.md` with explicit hardware/software prerequisites.
2. Add a `--lite` mode with a smaller judge model (`qwen3:8b` or `llama3.2:3b`) that runs on 8 GB RAM.
3. Add a `--no-gpu` flag that falls back to a cloud API judge (OpenAI, Anthropic, or Sarvam AI for India-specific deployments).
4. Add a smoke-test script (`check_requirements.sh`) that validates disk, RAM, and GPU availability before starting the build.
5. Consider publishing a pre-built Docker image to avoid pip-from-source builds.
