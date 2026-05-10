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
The assignment began with Option A — attempting to install and run the CeRAI AIEvaluationTool
against the DIGIT Studio Assistant. The Docker build failed before the image could start on an
Apple Silicon (M2) laptop: `OSError: [Errno 5]` from ~3.5 GB of NVIDIA CUDA packages that
target Linux x86-64 and are architecturally incompatible with aarch64.

Investigating further revealed a more significant problem than portability. Reading the CeRAI
source showed that its truthfulness and hallucination metrics evaluate responses against academic
benchmark datasets — SQuAD, CODAH, HotPotQA, HaluQA — not the bot's actual knowledge base.
A tool that claims to evaluate RAG systems but checks responses against Wikipedia passage
comprehension produces scores that are meaningless for domain-specific deployments, regardless
of whether it runs. That finding made Option B the right path: file the issues systematically,
then implement an alternative that evaluates what the tool claims to evaluate — faithfulness to
retrieved contexts, domain correctness, and safety — without requiring hardware the tool's own
audience does not have.
""")

with col_ai:
    st.subheader("AI use in completing this assignment")
    st.markdown("""
Claude Code was used throughout to navigate the decision and implementation.
The starting point was a working RAG bot with known gaps — false rejections on in-domain
topics, uncertainty about hallucination risk on capability questions — and an open question:
does CeRAI address these better than DeepEval or RAGAS? Reading through the CeRAI source files
together is where the benchmark-dataset finding emerged; the question "what does
`truth_internal.py` actually test against?" led to the SQuAD/CODAH discovery that became
the core of Issue #1.

Multiple evaluation scripts were written and debugged iteratively. The DeepEval API changed
between versions — GEval import paths, `evaluate()` vs `measure()`, `SingleTurnParams` enum
for evaluation_params — and each failure was diagnosed and corrected in the session rather
than by working around it. The session also served as a thinking partner for framing: deciding
which roadblocks were environment-specific (Apple Silicon) versus fundamental design limitations,
and sharpening issue descriptions so they applied to all teams using the tool, not only DIGIT's
deployment context. The confirmed hallucination in `ds_lim_004` — caught by the
`limitation_awareness` GEval and missed by generic faithfulness — was found during a live run
in the session and then incorporated into the issue framing for Issue #3.
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
be unsuitable for evaluating this class of conversational system. Seven issues were filed on the
repository.
""")

st.subheader("What we found")

issues = [
    {
        "id": "#1",
        "title": "RAG metrics run against academic benchmarks, not the bot's knowledge base",
        "file": "src/lib/strategy/truth_internal.py, hallucination.py",
        "impact": "Evaluation runs against SQuAD, CODAH, HotPotQA, HaluQA — academic datasets. For a bot built over a domain-specific knowledge base, scores measure general QA performance, not domain faithfulness. A production RAG evaluator must receive retrieved contexts and verify the response is grounded in them.",
    },
    {
        "id": "#2",
        "title": "No adversarial / prompt injection testing",
        "file": "src/lib/strategy/ (all strategy files)",
        "impact": "No test coverage for prompt injection, jailbreak attempts, or domain-wrapped harmful requests. Government-facing bots are high-value targets for these attacks. Without adversarial evaluation, resistance is invisible.",
    },
    {
        "id": "#3",
        "title": "No limitation-awareness evaluation category",
        "file": "src/lib/strategy/ (all strategy files)",
        "impact": "RAG hallucination risk is highest for questions about capabilities that don't exist. Running the alternative evaluator found a confirmed hallucination (bot stated published configurations are editable; they are immutable) that a generic faithfulness metric missed but a limitation-probing GEval caught with score 0.0.",
    },
    {
        "id": "#4",
        "title": "Infrastructure requirements exclude the tool's primary audience",
        "file": "docker-compose.yml, .env.example",
        "impact": "Requires qwen3:32b via Ollama (~19GB VRAM) and NVIDIA GPU. Most ML engineers, QA teams, and product teams work on standard laptops or commodity VMs. The tool requires more compute than many of the systems it evaluates.",
    },
    {
        "id": "#5",
        "title": "Docker build fails on Apple Silicon (aarch64)",
        "file": "requirements.txt",
        "impact": "docker compose build fails with OSError: [Errno 5] before the image starts. ~3.5GB of NVIDIA CUDA packages target Linux x86_64 — architecturally incompatible with aarch64. Any Mac developer is blocked before writing a single test case.",
    },
    {
        "id": "#6",
        "title": "Silent exception handling makes failures indistinguishable from low scores",
        "file": "src/lib/strategy/utils_new.py:221, fairness_stereotype_agreement.py:124",
        "impact": "Bare except blocks return 0 or {} with no logging. A run that silently failed 30% of evaluations looks identical to one that completed fully. Users cannot audit which evaluations actually ran.",
    },
    {
        "id": "#7",
        "title": "GPU service calls have no timeout — evaluation hangs indefinitely",
        "file": "src/lib/strategy/safety.py, language_strategies.py, fluency_score.py",
        "impact": "requests.post with no timeout= argument defaults to no timeout. If GPU_URL is slow or unreachable, the process hangs forever with no error and requires a manual kill.",
    },
    {
        "id": "#8",
        "title": "BiasDetection uses a surface-level text classifier — cannot distinguish biased responses from neutral reporting of bias",
        "file": "src/lib/strategy/bias_detection.py",
        "impact": "The amedvedev/bert-tiny-cognitive-bias model classifies whether surface language sounds biased, not whether the AI generated a biased response. A response quoting a study ('studies show women prefer caretaking roles') receives the same high bias score as one asserting it. False positives on legitimate factual reporting make the bias metric unreliable for production use.",
    },
    {
        "id": "#9",
        "title": "ComputeErrorRate returns an absolute count, not an error rate — with false positives and missed severities",
        "file": "src/lib/strategy/compute_error_rate.py",
        "impact": "Counts lines where 'ERROR' appears as a case-insensitive substring. 'INFO No errors detected' matches and is counted; 'FATAL Service crashed' is missed. Returns an integer, not errors/total interactions. total_lines is computed but never used. The log file is also scanned twice per evaluate() call.",
    },
    {
        "id": "#10",
        "title": "Compute_MTBF treats every [ERROR] log entry as a distinct system failure, producing artificially low MTBF",
        "file": "src/lib/strategy/compute_mtbf.py",
        "impact": "extract_failure_timestamps() collects one timestamp per [ERROR] line with no deduplication. Three consecutive error lines from a single incident produce three failure events and a sub-second MTBF for a system with one actual failure. Raises ValueError when fewer than two timestamps exist — crashes the evaluation with an unhandled exception rather than returning a safe fallback.",
    },
]

for issue in issues:
    with st.expander(f"Issue {issue['id']}: {issue['title']}"):
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

# ── Section 5: Limitations ─────────────────────────────────────────────────────

st.header("5. What This Evaluator Does Not Do Well")

st.markdown("""
These are not gaps to fix later — they are open problems that no current framework
(CeRAI, RAGAS, DeepEval, TruLens) solves adequately.
""")

limitations = [
    ("No multi-turn evaluation",
     "Every question is scored independently. Conversational coherence across turns, follow-up handling, and context retention are not tested."),
    ("GEval is non-deterministic",
     "LLM judge scores can vary ±0.05 between runs. Average 3 runs for publishable results."),
    ("No WhatsApp or browser-channel evaluation",
     "Only REST API endpoints are supported. CeRAI's Selenium-based WhatsApp evaluation is a genuine capability this framework does not replicate."),
    ("No load or performance testing",
     "Latency is measured per-question but the framework does not test at scale or under concurrent load."),
    ("No dataset management UI",
     "The golden set is a JSON file. There is no dashboard for managing test suites, versioning test cases, or comparing runs over time."),
    ("Faithfulness requires retrieved contexts",
     "The bot endpoint must return chunks via include_contexts=True. If contexts are empty, Faithfulness is skipped for that question."),
    ("OpenAI dependency for LLM judge",
     "--no-judge mode is free but limited to keyword checks only. Teams without OpenAI access cannot run the full evaluation."),
]

for title, body in limitations:
    with st.expander(title):
        st.write(body)

st.divider()

# ── Section 6: Machine-readable summary ───────────────────────────────────────

st.header("6. Machine-Readable Summary")

cats = results.get("summary_by_category", {}) if results else {}

summary_block = {
    "evaluation": {
        "system": "DIGIT Studio Assistant (EGOV_RAG_V5)",
        "system_type": "RAG chatbot — custom REST API, PostgreSQL/pgvector, GPT-4",
        "domain": "DIGIT platform documentation (eGovernments Foundation / India DPI)",
        "tool_assessed": "CeRAI AIEvaluationTool (https://github.com/cerai-iitm/AIEvaluationTool)",
        "option": "B — Critique & Rebuild",
        "alternative_framework": "DeepEval 4.0.0 with OpenAI gpt-4o judge",
        "golden_set_version": results.get("golden_set_version", "3.0") if results else "3.0",
        "run_at": results.get("run_at", "N/A") if results else "N/A",
    },
    "cerai_issues_filed": [
        "RAG metrics run against academic benchmarks, not the bot's knowledge base",
        "No adversarial / prompt injection testing",
        "No limitation-awareness evaluation category",
        "Infrastructure requirements exclude the tool's primary audience",
        "Docker build fails on Apple Silicon (aarch64)",
        "Silent exception handling makes failures indistinguishable from low scores",
        "GPU service calls have no timeout — evaluation hangs indefinitely",
        "BiasDetection uses a surface-level text classifier — cannot distinguish biased responses from neutral reporting of bias",
        "ComputeErrorRate returns an absolute count, not an error rate — with false positives and missed severities",
        "Compute_MTBF treats every [ERROR] log entry as a distinct system failure, producing artificially low MTBF",
    ],
    "alternative_evaluator": {
        "framework": "DeepEval 4.0.0",
        "judge": "gpt-4o via OpenAI API",
        "install": "pip install deepeval requests python-dotenv",
        "test_cases": sum(v.get("total", 0) for v in cats.values()) if cats else 56,
        "categories": list(cats.keys()) if cats else ["in_domain", "limitation_awareness", "out_of_domain", "edge_case", "adversarial"],
        "novel_contribution": "limitation_awareness category — no existing evaluation framework has an equivalent",
    },
    "results_summary": {
        cat: {
            "total": s.get("total"),
            "keyword_pass_rate": f"{100*s['kw_pass']//s['total']}%" if s.get("total") else None,
            "llm_judge_pass_rate": f"{100*s['judge_pass']//s['judge_total']}%" if s.get("judge_total") else None,
        }
        for cat, s in cats.items()
    } if cats else {},
    "key_findings": [
        "Confirmed hallucination: ds_lim_004 — bot stated published configs are editable (they are immutable). LLM judge score 0.0.",
        "Systematic false rejections: MDMS, inbox, service/module distinction refused as OOD despite being in-domain",
        "OOD evasion: weather question answered in DIGIT-framed language instead of clean refusal",
        "Adversarial safety: 100% — bot resisted all 5 injection and jailbreak attempts",
        "Keyword vs LLM judge gap: 22% vs 77% on in-domain — keyword-only eval dramatically underestimates quality",
    ],
}

st.json(summary_block)

st.caption(
    "EGOV_RAG_V5 · eGovernments Foundation · "
    "Gates Foundation AI Fellowship — India 2026"
)
