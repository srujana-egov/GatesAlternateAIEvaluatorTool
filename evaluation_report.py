"""
Evaluation Report — Gates Foundation AI Fellowship Technical Assignment
Option B: Critique & Rebuild
"""

import json
import os
import glob
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Evaluation Report — DIGIT Studio Assistant",
    page_icon="📊",
    layout="wide",
)

# ── Load results ───────────────────────────────────────────────────────────────

def load_latest_results():
    eval_dir = os.path.join(os.path.dirname(__file__), "results")
    files = sorted(glob.glob(os.path.join(eval_dir, "deepeval_results*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)

results = load_latest_results()

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("Evaluating a RAG-Based Government Service Bot")
st.caption("Gates Foundation AI Fellowship — India 2026 | Technical Assignment: Option B — Critique & Rebuild")

st.markdown("""
This report documents the evaluation of **DIGIT Studio Assistant**, a retrieval-augmented
generation (RAG) chatbot built over DIGIT platform documentation. It presents a systematic
critique of the CeRAI AI Evaluation Tool, an alternative evaluator built to address its
limitations, and results from running that evaluator against the bot.
""")

st.divider()

# ── Submission context ─────────────────────────────────────────────────────────

col_path, col_ai = st.columns(2)

with col_path:
    st.subheader("Path chosen")
    st.markdown("""
Option A failed at `docker compose build` — `OSError: [Errno 5]` from ~3.5 GB of NVIDIA CUDA
packages incompatible with Apple Silicon (aarch64). Reading the source to understand why
revealed a deeper problem: `truth_internal.py` and `hallucination.py` evaluate responses
against hardcoded academic benchmarks (SQuAD, CODAH, HotPotQA, HaluQA), and the `Conversation`
object has no `retrieved_contexts` field — meaning CeRAI has no mechanism to receive what a RAG
pipeline actually retrieved and verify the answer against it. Further reading found silent bare
`except` blocks returning `0` or `{}` with no logging, `requests.post` calls without `timeout=`
that hang indefinitely if the GPU service is slow, a bias classifier that scores surface language
rather than AI-generated assertions, and log-analysis strategies (`ComputeErrorRate`,
`Compute_MTBF`) that return absolute counts with false positives rather than real rates.
The portability problem is fixable; the evaluation design is not — a tool that never sees
retrieved contexts cannot measure RAG faithfulness regardless of the hardware it runs on,
which made Option B the right path.
""")

with col_ai:
    st.subheader("AI use in completing this assignment")
    st.markdown("""
Claude Code was used as a code-reading and reasoning partner throughout — not to generate
boilerplate, but to work through what CeRAI actually does versus what it claims to do.
The process was file-by-file: `truth_internal.py` was read to answer "what dataset does
the truthfulness metric test against?", which surfaced the SQuAD/CODAH/HotPotQA finding.
`data/conversation.py` was checked to confirm there is no `retrieved_contexts` field —
ruling out the possibility that faithfulness grounding was handled elsewhere.
`safety.py` was read to understand whether adversarial coverage was meaningful, revealing
that all three ShieldGemma modes are single-turn classifiers with no injection simulation.
`robustness_advInstruction.py` was read and initially misread — it uses cosine similarity
on adversarial GLUE benchmarks, which tests paraphrase robustness, not prompt injection;
that distinction had to be debated before the issue was framed correctly.

Not every hypothesis held. An initial issue about missing out-of-scope detection was
invalidated when `DataPoints.json` was read and metric 20 was found — it already covers
out-of-scope queries. The real gap (domain-specific capability hallucination, not topic drift)
only emerged after that. Issue framing for the OpenAI/Gemini API keys required checking
`api_handler.py` directly — the keys exist in `.env.example` but serve the interface manager,
not the judge layer; conflating the two would have produced a false issue. Error paths rather
than happy paths were what revealed the silent `except` blocks in `utils_new.py` and
`fairness_stereotype_agreement.py`, and cross-referencing `truth_internal.py` (which has
`timeout=45`) against `safety.py` and `fluency_score.py` (which have none) was what
established the timeout inconsistency as unintentional rather than deliberate.

On the implementation side, the DeepEval API required iterative debugging — GEval import
paths, `evaluate()` vs `measure()`, the `SingleTurnParams` enum for `evaluation_params`
all changed between versions and were corrected against actual error output rather than
documentation. The `limitation_awareness` GEval category was designed specifically because
standard faithfulness metrics missed `ds_lim_004` (the bot asserted published configurations
are editable; they are immutable) during a live run — that failure became the concrete
evidence behind Issue #3 and the design rationale for the novel category.
""")

st.divider()

# ── Section 1: System Under Evaluation ────────────────────────────────────────

st.header("1. System Under Evaluation")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
**DIGIT Studio Assistant** is a domain-specific conversational AI system built on top of the
[DIGIT platform](https://digit.org) — India's open-source digital public infrastructure for
government service delivery, actively supported by eGovernments Foundation and the Gates Foundation.

The bot answers questions about configuring and deploying DIGIT Studio, a no-code/low-code
layer that allows government teams to build public service delivery applications without writing
code. It uses a hybrid retrieval pipeline (dense vector search + BM25) over a PostgreSQL/pgvector
knowledge base, with GPT-4 as the generation layer and a confidence-threshold domain guard
to refuse out-of-scope questions.

**Why this system was chosen:**
- It is a production system, not a toy demo — it runs against real documentation used by
  government teams in India
- It is precisely the class of system CeRAI cannot evaluate: a custom REST API with a
  RAG architecture, deployed by a team that does not own a 32GB GPU
- DIGIT is digital public infrastructure for India — evaluating AI systems built on it is
  directly relevant to the fellowship's focus area
""")

with col2:
    st.markdown("**System architecture**")
    st.code("""
User query
    ↓
OOD threshold check
(cosine similarity < 0.35 → reject)
    ↓
Hybrid retrieval
(BM25 + pgvector)
    ↓
GPT-4 generation
    ↓
Response + latency metadata
""", language="text")

st.divider()

# ── Section 2: CeRAI Assessment ────────────────────────────────────────────────

st.header("2. CeRAI AI Evaluation Tool — Assessment")

st.markdown("""
The [CeRAI AI Evaluation Tool](https://github.com/cerai-iitm/AIEvaluationTool) was cloned,
its source code was read in full, and a Docker build was attempted. The tool was determined to
be unsuitable for evaluating this class of conversational system. Ten issues were filed on the
[repository](https://github.com/cerai-iitm/AIEvaluationTool/issues).
""")

st.subheader("What we found")

issues = [
    {
        "id": "#1",
        "title": "RAG metrics run against academic benchmarks, not the bot's knowledge base",
        "file": "src/lib/strategy/truth_internal.py, hallucination.py",
        "impact": "Evaluation runs against SQuAD, CODAH, HotPotQA, HaluQA — academic datasets. For a bot built over a domain-specific knowledge base, scores measure general QA performance, not domain faithfulness. A production RAG evaluator must receive retrieved contexts and verify the response is grounded in them.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues/136",
    },
    {
        "id": "#2",
        "title": "No adversarial / prompt injection testing",
        "file": "src/lib/strategy/ (all strategy files)",
        "impact": "No test coverage for prompt injection, jailbreak attempts, or domain-wrapped harmful requests. Government-facing bots are high-value targets for these attacks. Without adversarial evaluation, resistance is invisible.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues/138",
    },
    {
        "id": "#3",
        "title": "No limitation-awareness evaluation category",
        "file": "src/lib/strategy/ (all strategy files)",
        "impact": "RAG hallucination risk is highest for questions about capabilities that don't exist. Running the alternative evaluator found a confirmed hallucination (bot stated published configurations are editable; they are immutable) that a generic faithfulness metric missed but a limitation-probing GEval caught with score 0.0.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues/140",
    },
    {
        "id": "#4",
        "title": "Infrastructure requirements exclude the tool's primary audience",
        "file": "docker-compose.yml, .env.example",
        "impact": "Requires qwen3:32b via Ollama (~19GB VRAM) and NVIDIA GPU. Most ML engineers, QA teams, and product teams work on standard laptops or commodity VMs. The tool requires more compute than many of the systems it evaluates.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues/141",
    },
    {
        "id": "#5",
        "title": "Docker build fails on Apple Silicon (aarch64)",
        "file": "requirements.txt",
        "impact": "docker compose build fails with OSError: [Errno 5] before the image starts. ~3.5GB of NVIDIA CUDA packages target Linux x86_64 — architecturally incompatible with aarch64. Any Mac developer is blocked before writing a single test case.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues/142",
    },
    {
        "id": "#6",
        "title": "Silent exception handling makes failures indistinguishable from low scores",
        "file": "src/lib/strategy/utils_new.py:221, fairness_stereotype_agreement.py:124",
        "impact": "Bare except blocks return 0 or {} with no logging. A run that silently failed 30% of evaluations looks identical to one that completed fully. Users cannot audit which evaluations actually ran.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues/143",
    },
    {
        "id": "#7",
        "title": "GPU service calls have no timeout — evaluation hangs indefinitely",
        "file": "src/lib/strategy/safety.py, language_strategies.py, fluency_score.py",
        "impact": "requests.post with no timeout= argument defaults to no timeout. If GPU_URL is slow or unreachable, the process hangs forever with no error and requires a manual kill.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues/144",
    },
    {
        "id": "#8",
        "title": "BiasDetection uses a surface-level text classifier — cannot distinguish biased responses from neutral reporting of bias",
        "file": "src/lib/strategy/bias_detection.py",
        "impact": "The amedvedev/bert-tiny-cognitive-bias model classifies whether surface language sounds biased, not whether the AI generated a biased response. A response quoting a study ('studies show women prefer caretaking roles') receives the same high bias score as one asserting it. False positives on legitimate factual reporting make the bias metric unreliable for production use.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues/149",
    },
    {
        "id": "#9",
        "title": "ComputeErrorRate returns an absolute count, not an error rate — with false positives and missed severities",
        "file": "src/lib/strategy/compute_error_rate.py",
        "impact": "Counts lines where 'ERROR' appears as a case-insensitive substring. 'INFO No errors detected' matches and is counted; 'FATAL Service crashed' is missed. Returns an integer, not errors/total interactions. total_lines is computed but never used. The log file is also scanned twice per evaluate() call.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues/150",
    },
    {
        "id": "#10",
        "title": "Compute_MTBF treats every [ERROR] log entry as a distinct system failure, producing artificially low MTBF",
        "file": "src/lib/strategy/compute_mtbf.py",
        "impact": "extract_failure_timestamps() collects one timestamp per [ERROR] line with no deduplication. Three consecutive error lines from a single incident produce three failure events and a sub-second MTBF for a system with one actual failure. Raises ValueError when fewer than two timestamps exist — crashes the evaluation with an unhandled exception rather than returning a safe fallback.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues/151",
    },
]

for issue in issues:
    with st.expander(f"Issue {issue['id']}: {issue['title']}"):
        st.markdown(f"**Filed:** [cerai-iitm/AIEvaluationTool{issue['url'].split('issues')[1]}]({issue['url']})")
        st.markdown(f"**Relevant code:** `{issue['file']}`")
        st.markdown(f"**Impact on evaluation quality:** {issue['impact']}")

st.subheader("The deeper finding")
st.markdown("""
Beyond the portability failure, reading the source revealed a more fundamental problem:
CeRAI is not designed for production RAG evaluation. Its truthfulness and hallucination
metrics evaluate responses against academic benchmark datasets — not the actual knowledge
base the bot was built on. Even if CeRAI ran perfectly, it would produce scores that are
meaningless for domain-specific systems.
""")

st.divider()

# ── Section 3: Alternative Evaluator ──────────────────────────────────────────

st.header("3. Alternative Evaluator — Design Decisions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**What was built**

A Python evaluation framework (`deepeval_eval.py`) that:
- Sends prompts to `POST /chat` on any REST endpoint
- Scores responses using DeepEval metrics (AnswerRelevancy, Faithfulness, GEval)
- Runs on commodity hardware with an OpenAI API key — no GPU needed
- Produces structured JSON output for reproducibility

**Framework components**

| Component | Purpose | CeRAI gap addressed |
|---|---|---|
| Custom REST executor | Calls any `POST /chat` endpoint | Issues #4, #5 |
| OpenAI gpt-4o judge | No local GPU required | Issue #4 |
| DeepEval Faithfulness | Detects hallucination vs retrieved docs | Issue #1 |
| DeepEval AnswerRelevancy | Reference-free quality scoring | Issue #1 |
| `limitation_awareness` GEval | Catches capability hallucinations | Issue #3 |
| `adversarial` GEval | Prompt injection and jailbreak testing | Issue #2 |
| Keyword check | Free, deterministic baseline | — |
""")

with col2:
    st.markdown("""
**Design decisions**

**Why DeepEval:**
DeepEval's GEval supports custom evaluation criteria per category, enabling
`limitation_awareness` and `adversarial` test categories that no existing framework
(CeRAI, RAGAS, TruLens) provides. The criteria are written in plain English and
judged by the LLM — no benchmark datasets required.

**Why OpenAI gpt-4o as judge:**
No GPU, no Ollama, no 19GB model download. Costs ~$0.05/run.
Runs on the same hardware the bot runs on.

**Why the `limitation_awareness` category is novel:**
RAG hallucination risk is highest for questions about capabilities that don't exist.
Standard faithfulness metrics miss this because the bot's answer is internally
consistent — it just asserts something false. Only a judge explicitly told to check
whether stated capabilities actually exist can catch this pattern.

**Why keyword checks alongside LLM judge:**
Demonstrates that keyword-only evaluation (the basis of most simple test scripts)
significantly underestimates quality. In-domain keyword pass rate was 22% vs
77% LLM judge pass rate — the same questions, dramatically different conclusions.
""")

st.divider()

# ── Section 4: Results ─────────────────────────────────────────────────────────

st.header("4. Results")

if results:
    cats = results.get("summary_by_category", {})

    def pct(num, den):
        return f"{100*num//den}%" if den else "—"

    # Top-line metrics
    in_d  = cats.get("in_domain", {})
    lim   = cats.get("limitation_awareness", {})
    ood   = cats.get("out_of_domain", {})
    adv   = cats.get("adversarial", {})
    edge  = cats.get("edge_case", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("In-domain (LLM judge)",
              pct(in_d.get("judge_pass",0), in_d.get("judge_total",0)),
              help="% of in-domain questions where LLM judge passed")
    m2.metric("Limitation awareness",
              pct(lim.get("judge_pass",0), lim.get("judge_total",0)),
              help="% of capability-limitation questions correctly answered without hallucination")
    m3.metric("OOD rejection",
              pct(ood.get("judge_pass",0), ood.get("judge_total",0)),
              help="% of out-of-domain questions correctly declined")
    m4.metric("Adversarial safety",
              pct(adv.get("judge_pass",0), adv.get("judge_total",0)),
              help="% of injection/jailbreak attempts correctly refused")
    m5.metric("Edge case (LLM judge)",
              pct(edge.get("judge_pass",0), edge.get("judge_total",0)),
              help="% of ambiguous/broad questions handled correctly")

    st.markdown("---")

    # Category summary table
    st.subheader("Summary by category")
    summary_rows = []
    for cat, s in cats.items():
        summary_rows.append({
            "Category": cat,
            "Questions": s.get("total", 0),
            "Keyword pass": f"{s.get('kw_pass',0)}/{s.get('total',0)}",
            "LLM judge pass": f"{s.get('judge_pass',0)}/{s.get('judge_total',0)}" if s.get('judge_total') else "—",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Results by question")
        rows = []
        for r in results.get("results", []):
            scores = r.get("deepeval_scores", {})
            judge_scores = ", ".join(
                f"{k}={v['score']}" for k, v in scores.items() if v.get("score") is not None
            ) if scores else "—"
            rows.append({
                "ID": r["id"],
                "Category": r["category"],
                "Keyword": "✓" if r.get("keyword_pass") else "✗",
                "LLM judge": judge_scores,
                "OOD": "✓" if r.get("out_of_domain") else "",
                "Latency (ms)": r.get("latency_ms", ""),
                "Answer preview": r.get("answer_preview", "")[:80],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=500, hide_index=True)

    with col2:
        st.subheader("Latency by question")
        lat_rows = [
            {"Question": r["id"], "Latency (ms)": r.get("latency_ms", 0), "Category": r["category"]}
            for r in results.get("results", []) if r.get("latency_ms")
        ]
        if lat_rows:
            lat_df = pd.DataFrame(lat_rows)
            st.bar_chart(lat_df.set_index("Question")["Latency (ms)"])

        st.subheader("Keyword vs LLM judge gap")
        st.markdown("""
        The gap between keyword pass rate (22%) and LLM judge pass rate (77%) on
        in-domain questions shows that keyword-only evaluation dramatically
        underestimates bot quality — the bot answers correctly but uses different
        phrasing than the expected keywords. This demonstrates why static
        expected-output matching is insufficient for generative AI systems.
        """)

    st.subheader("Key findings")
    st.markdown("""
**Confirmed hallucination** (`ds_lim_004`) — Bot stated published configurations can be edited.
They are immutable. LLM judge (Limitation Awareness GEval) scored 0.0.
A generic faithfulness metric missed this entirely.

**Systematic false rejections** — MDMS, inbox search, service/module distinction,
and workflow state questions were refused as out-of-domain despite being core in-domain topics.
Retrieval gaps in the knowledge base.

**OOD evasion** (`ds_ood_001`) — Weather question answered as "DIGIT Studio does not do weather"
rather than a clean refusal. The OOD classifier missed it; GEval scored it 0.23.

**Adversarial safety: 100%** — Bot correctly resisted all 5 injection and jailbreak attempts.
""")

else:
    st.info("No results file found. Run `python deepeval_eval.py` to generate results.")

st.divider()

# ── Section 5: Tradeoffs ───────────────────────────────────────────────────────

st.header("5. Tradeoffs vs CeRAI")

st.markdown("""
This evaluator makes deliberate tradeoffs against CeRAI. Some gaps are fixable;
others are open problems no current framework solves.
""")

tradeoffs = [
    {
        "Dimension": "Hardware requirement",
        "This evaluator": "Any laptop — `pip install deepeval`, OpenAI API key",
        "CeRAI": "NVIDIA GPU, 28–32 GB RAM for qwen3:32b judge",
        "Winner": "This evaluator",
    },
    {
        "Dimension": "Bot interface",
        "This evaluator": "Any REST `POST /chat` endpoint",
        "CeRAI": "WhatsApp via Selenium browser automation",
        "Winner": "This evaluator — unless the bot is WhatsApp-only",
    },
    {
        "Dimension": "RAG faithfulness",
        "This evaluator": "DeepEval Faithfulness verifies answer against retrieved chunks",
        "CeRAI": "No retrieved-context field — cannot check grounding",
        "Winner": "This evaluator",
    },
    {
        "Dimension": "Limitation-awareness",
        "This evaluator": "Custom GEval catches capability hallucinations (caught ds_lim_004)",
        "CeRAI": "Not present",
        "Winner": "This evaluator",
    },
    {
        "Dimension": "Adversarial / injection",
        "This evaluator": "Custom GEval for injection, jailbreak, impersonation, harmful content",
        "CeRAI": "Single-turn ShieldGemma classifier — no injection simulation",
        "Winner": "This evaluator",
    },
    {
        "Dimension": "LLM judge reproducibility",
        "This evaluator": "GEval scores vary ±0.05 between runs — non-deterministic",
        "CeRAI": "Rule-based checks are fully reproducible",
        "Winner": "CeRAI",
    },
    {
        "Dimension": "Multi-turn evaluation",
        "This evaluator": "Not supported — every question scored independently",
        "CeRAI": "Supports multi-turn conversation flows via Selenium",
        "Winner": "CeRAI",
    },
    {
        "Dimension": "Channel coverage",
        "This evaluator": "REST API only",
        "CeRAI": "WhatsApp Web, browser-based, and API targets",
        "Winner": "CeRAI",
    },
    {
        "Dimension": "Dataset management",
        "This evaluator": "Golden set is a JSON file — no versioning, no run comparison UI",
        "CeRAI": "Full TCE dashboard for test suites and run history",
        "Winner": "CeRAI",
    },
    {
        "Dimension": "API cost",
        "This evaluator": "~$0.05/run (OpenAI gpt-4o) — `--no-judge` mode is free",
        "CeRAI": "$0 API cost — fully local",
        "Winner": "CeRAI for teams without an OpenAI key",
    },
]

st.dataframe(
    tradeoffs,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Winner": st.column_config.TextColumn(width="medium"),
    },
)

st.caption(
    "EGOV_RAG_V5 · eGovernments Foundation · "
    "Gates Foundation AI Fellowship — India 2026"
)
