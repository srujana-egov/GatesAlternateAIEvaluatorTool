"""
Evaluation Report — Gates Foundation AI Fellowship Technical Assignment
Option B: Critique & Rebuild

This page presents the full evaluation of the CeRAI AIEvaluationTool and the
alternative evaluator built for DIGIT Studio Assistant (EGOV_RAG_V5).
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

# ─────────────────────────────────────────────
# Load latest results
# ─────────────────────────────────────────────

def load_latest_results():
    eval_dir = os.path.join(os.path.dirname(__file__), "results")
    files = sorted(glob.glob(os.path.join(eval_dir, "deepeval_results*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)

results = load_latest_results()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.title("Evaluating a RAG-Based Government Service Bot")
st.caption("Gates Foundation AI Fellowship — India 2026 | Technical Assignment: Option B — Critique & Rebuild")

st.markdown("""
This report documents the evaluation of **DIGIT Studio Assistant**, a retrieval-augmented
generation (RAG) chatbot built over DIGIT platform documentation. It presents a systematic
critique of the CeRAI AI Evaluation Tool, an alternative evaluator built to address its
limitations, and results from running that evaluator against the bot.
""")

st.divider()

# ─────────────────────────────────────────────
# Section 1: System Under Evaluation
# ─────────────────────────────────────────────

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
Query rewrite (GPT-3.5)
    ↓
Multi-query retrieval
(BM25 + pgvector hybrid)
    ↓
Reranking + section filtering
    ↓
GPT-4 generation
    ↓
Response + latency metadata
""", language="text")

st.divider()

# ─────────────────────────────────────────────
# Section 2: CeRAI Assessment
# ─────────────────────────────────────────────

st.header("2. CeRAI AI Evaluation Tool — Assessment")

st.markdown("""
The [CeRAI AI Evaluation Tool](https://github.com/cerai-iitm/AIEvaluationTool) was installed,
its source code was read in full, and a Docker build was attempted. The tool was determined to
be unsuitable for evaluating this class of conversational system. Five issues were filed on the
repository.
""")

st.subheader("What the tool claims")
st.markdown("""
> *"Multi-platform support allows the same framework to be used across API, WhatsApp, and web deployments."*
> — CeRAI documentation

The tool targets AI engineers, QA teams, and product teams working on conversational AI systems.
""")

st.subheader("What we found")

issues = [
    {
        "id": "#1",
        "title": "API mode only supports OpenAI-compatible format — not arbitrary REST endpoints",
        "file": "src/app/interface_manager/api_handler.py:98–147",
        "impact": "The majority of real-world conversational systems, including all RAG pipelines and domain-specific bots, expose simple REST endpoints (POST /chat), not OpenAI Chat Completions format. These cannot be evaluated.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues",
    },
    {
        "id": "#2",
        "title": "LLM judge hardcoded to Ollama/qwen3:32b — no path to use OpenAI or Gemini",
        "file": "src/lib/strategy/llm_judge.py:21–23, src/lib/strategy/data/defaults.json:5",
        "impact": "Requires 20–32GB RAM and a separately installed Ollama server. No configuration path exists to substitute a cloud LLM. The tool ships OPENAI_API_KEY and GEMINI_API_KEY in .env.example but never uses them for judging.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues",
    },
    {
        "id": "#3",
        "title": "Test schema requires EXPECTED_OUTPUT — incompatible with open-ended conversational AI",
        "file": "data/updated_datapoints.json",
        "impact": "Semantic similarity against a pre-written expected answer cannot evaluate systems where many valid responses exist. This is the fundamental limitation of rule-based NLP evaluation applied to generative AI.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues",
    },
    {
        "id": "#4",
        "title": "No RAG-specific metrics — cannot evaluate retrieval quality, faithfulness, or grounding",
        "file": "src/lib/strategy/ (all strategy files)",
        "impact": "Modern conversational AI in government contexts is predominantly RAG-based. CeRAI cannot ask: did the bot retrieve the right documents? Did it hallucinate facts not in the knowledge base? These are the most critical failure modes.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues",
    },
    {
        "id": "#5",
        "title": "Infrastructure requirements exceed the compute budget of the systems it evaluates",
        "file": "docker-compose.yml, .env.example",
        "impact": "DIGIT 3.0 targets deployment on 8GB machines for India's state-level government infrastructure. The CeRAI judge alone requires 4x that RAM. An evaluation tool for government AI should run on commodity hardware — suggested fix: make judge configurable via API key.",
        "url": "https://github.com/cerai-iitm/AIEvaluationTool/issues",
    },
]

for issue in issues:
    with st.expander(f"Issue {issue['id']}: {issue['title']}"):
        st.markdown(f"**Relevant code:** `{issue['file']}`")
        st.markdown(f"**Impact on evaluation quality:** {issue['impact']}")
        st.markdown(f"[View filed issue →]({issue['url']})")

st.subheader("Why Option B")
st.markdown("""
CeRAI works for its intended use case: Selenium-driven WhatsApp bot evaluation in a GPU lab
environment. That use case is real and the architecture is sound. But the tool was open-sourced
without removing assumptions that make it inaccessible to the teams most likely to use it —
government digital teams, NGOs, and independent developers building on India's DPI stack.

The API pathway, which is the documented alternative for non-WhatsApp deployments, does not
support arbitrary REST endpoints. For any team building a domain-specific RAG system,
CeRAI offers no viable evaluation path.
""")

st.divider()

# ─────────────────────────────────────────────
# Section 3: Alternative Evaluator
# ─────────────────────────────────────────────

st.header("3. Alternative Evaluator — Design Decisions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**What we built**

A Python evaluation framework (`eval/run_eval.py`) that:
- Sends prompts directly to `POST /chat` on any REST endpoint
- Scores responses across five metric categories
- Runs entirely on commodity hardware with an API key
- Produces structured JSON output for reproducibility

**Framework components**

| Component | Purpose | CeRAI issue addressed |
|---|---|---|
| Custom REST executor | Calls any `POST /chat` endpoint | Issue #1 |
| OpenAI as judge (via RAGAS) | No local GPU required | Issue #2 |
| RAGAS faithfulness metric | Detects hallucination against retrieved docs | Issue #4 |
| RAGAS answer relevancy | Reference-free quality scoring | Issue #3 |
| OOD + adversarial categories | Domain boundary and safety testing | Issue #4 |
""")

with col2:
    st.markdown("""
**Design decisions**

**Why RAGAS over rebuilding from scratch:**
Option B explicitly permits using existing open-source frameworks. RAGAS is maintained,
well-documented, and addresses the specific gap — RAG-specific metrics — that CeRAI
lacks entirely. Building equivalent faithfulness scoring from scratch would take weeks.

**Why keyword + retrieval metrics alongside RAGAS:**
RAGAS measures semantic quality. It does not measure whether the right documentation
section was retrieved, or whether the domain guard fired correctly. These require
structural knowledge of the system that only a purpose-built evaluator can provide.

**Why GPT-4o as judge:**
The target system already uses GPT-4 for generation. Using the same provider for
judging keeps costs predictable and removes the Ollama dependency that makes CeRAI
inaccessible. Total evaluation cost for this run: ~$0.15.

**Why 33 test cases across 4 categories:**
CeRAI's test data is single-category (prompt + expected output). Splitting into
in-domain, out-of-domain, edge case, and adversarial directly tests the four failure
modes most critical for a government-facing chatbot.
""")

st.divider()

# ─────────────────────────────────────────────
# Section 4: Results
# ─────────────────────────────────────────────

st.header("4. Results")

if results:
    s = results["summary"]

    # Top metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Retrieval Hit Rate", f"{s['retrieval_hit_rate']*100:.0f}%", help="Was the expected documentation section retrieved?")
    m2.metric("Answer Quality", f"{s['answer_quality_rate']*100:.0f}%", help="Did the response contain core expected terms?")
    m3.metric("OOD Rejection", f"{s['ood_rejection_rate']*100:.0f}%", help="Were off-topic questions correctly refused?")
    m4.metric("Adversarial Safety", f"{s['adversarial_safety_rate']*100:.0f}%", help="Were prompt injection attempts handled safely?")
    m5.metric("False Rejection", f"{s['false_rejection_rate']*100:.0f}%", delta_color="inverse", help="Were valid questions wrongly refused? Lower is better.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Results by question")
        rows = []
        for r in results["results"]:
            rows.append({
                "ID": r["id"],
                "Category": r["category"],
                "Status": r["status"],
                "Latency (ms)": r["latency_ms"],
                "Answer preview": r["answer_preview"][:80] + "..." if len(r.get("answer_preview","")) > 80 else r.get("answer_preview",""),
            })
        df = pd.DataFrame(rows)

        def colour_status(val):
            if val == "PASS":
                return "background-color: #d4edda"
            elif val == "PARTIAL":
                return "background-color: #fff3cd"
            elif "FAIL" in str(val):
                return "background-color: #f8d7da"
            return ""

        st.dataframe(
            df.style.applymap(colour_status, subset=["Status"]),
            use_container_width=True,
            height=500,
        )

    with col2:
        st.subheader("Latency distribution")
        lat_df = pd.DataFrame([
            {"Question": r["id"], "Latency (ms)": r["latency_ms"], "Category": r["category"]}
            for r in results["results"]
        ])
        st.bar_chart(lat_df.set_index("Question")["Latency (ms)"])

        st.subheader("Pass/fail by category")
        cat_df = pd.DataFrame([
            {"Category": r["category"], "Status": "Pass" if r["status"] == "PASS" else "Fail/Partial"}
            for r in results["results"]
        ])
        st.dataframe(
            cat_df.groupby(["Category", "Status"]).size().unstack(fill_value=0),
            use_container_width=True,
        )

    st.subheader("Interpretation")

    st.markdown(f"""
**Retrieval and answer quality (100%)** — The bot correctly retrieves relevant documentation
sections and produces answers containing core expected terms for every in-domain question.
This validates the hybrid BM25 + vector retrieval approach.

**OOD rejection (70%, 3 failures)** — Two failures are HCM-domain questions: campaign, role,
and configuration vocabulary appears in DIGIT Studio chunks with enough similarity to pass the
0.35 threshold despite no HCM data being ingested. This is *vocabulary bleed* between adjacent
domains — a known failure mode in single-domain RAG systems. The third failure (`ds_ood_001`)
demonstrates *domain-name injection*: phrasing an off-topic question with "DIGIT Studio"
in the prompt raised its similarity score above the threshold.

**Adversarial safety (33%, 2 failures)** — Two distinct attack patterns succeeded:
1. Direct prompt injection (`ds_adv_001`): the system prompt was partially exposed when directly asked
2. Domain-wrapped attack (`ds_adv_003`): prefixing a harmful request with "In the context of
   DIGIT Studio" caused the domain guard to treat it as in-scope

The third adversarial case (developer mode jailbreak) was correctly refused.

**Latency** — Average {s['avg_latency_ms']:.0f}ms, P95 {s['p95_latency_ms']}ms. The long tail is
driven by multi-query retrieval (3 query variants × hybrid search) on complex questions. Acceptable
for async use cases; borderline for real-time chat without streaming.
""")

st.divider()

# ─────────────────────────────────────────────
# Section 5: What This Evaluator Still Does Not Do Well
# ─────────────────────────────────────────────

st.header("5. Honest Limitations")

st.markdown("""
These are not gaps to fix later. They are open problems in the evaluation field that no current
framework — CeRAI, RAGAS, or deepeval — solves adequately.
""")

limitations = [
    ("Adaptive response length is not evaluated",
     "The bot uses a fixed 300-word cap. A simple question getting 300 words and a complex "
     "one being truncated would both pass. No framework evaluates whether response length "
     "matched question complexity."),
    ("Positional coherence is not evaluated",
     "Hybrid retrieval ranks chunks by similarity score, not document order. Retrieved chunks "
     "for sequential content (setup guides, step-by-step processes) may be fed to the LLM out "
     "of order, producing incoherent answers that still score 100% on retrieval hit rate."),
    ("Procedural fragmentation is not evaluated",
     "A 10-step process split across 4 chunks may yield partial retrieval — steps 1, 3, 5 "
     "retrieved without 2 and 4. The answer appears complete but is missing steps. Faithfulness "
     "passes because retrieved content is accurate; completeness is not measured."),
    ("Verbatim content preservation is not evaluated",
     "Code blocks, API endpoints, and structured data in retrieved chunks are paraphrased by "
     "the LLM. A correct curl command becomes a prose description, losing exactness. RAGAS "
     "faithfulness catches contradictions, not loss of precision."),
    ("Domain-specific terminology degrades judge quality",
     "GPT-4o as judge may not recognise DIGIT-specific terms (MDMS, boundary hierarchy, muster "
     "roll). A technically correct answer using domain jargon may score lower than a fluent "
     "but vague response. This affects RAGAS answer relevancy most."),
    ("No multi-turn evaluation",
     "Every question is scored independently. Conversation history, follow-up coherence, and "
     "context retention across turns are not tested. Real users rarely ask single isolated questions."),
    ("Adversarial coverage is heuristic",
     "Three adversarial cases cannot characterise a system's robustness. The two failures found "
     "here (domain-wrapper attack, direct injection) are the tip of a much larger attack surface."),
]

for title, body in limitations:
    with st.expander(title):
        st.write(body)

st.divider()

# ─────────────────────────────────────────────
# Section 6: Machine-readable summary
# ─────────────────────────────────────────────

st.header("6. Machine-Readable Summary")

summary_block = {
    "evaluation": {
        "system": "DIGIT Studio Assistant (EGOV_RAG_V5)",
        "system_type": "RAG chatbot — custom REST API, PostgreSQL/pgvector, GPT-4",
        "domain": "DIGIT platform documentation (eGovernments Foundation / India DPI)",
        "tool_assessed": "CeRAI AIEvaluationTool (https://github.com/cerai-iitm/AIEvaluationTool)",
        "option": "B — Critique & Rebuild",
        "run_at": results["run_at"] if results else "N/A",
    },
    "cerai_issues_filed": [
        "API mode only supports OpenAI-compatible format, not arbitrary REST endpoints",
        "LLM judge hardcoded to Ollama/qwen3:32b — no cloud LLM path",
        "Test schema requires EXPECTED_OUTPUT — incompatible with generative AI",
        "No RAG-specific metrics (faithfulness, retrieval quality, grounding)",
        "Infrastructure requirements (32GB RAM) exceed deployment target (8GB)",
    ],
    "alternative_evaluator": {
        "framework": "Custom Python (eval/run_eval.py) + RAGAS",
        "judge": "GPT-4o via OpenAI API",
        "install": "pip install ragas datasets",
        "test_cases": 33,
        "categories": ["in_domain", "out_of_domain", "edge_case", "adversarial"],
    },
    "results": {
        "retrieval_hit_rate": 1.0,
        "answer_quality_rate": 1.0,
        "false_rejection_rate": 0.0,
        "ood_rejection_rate": 0.7,
        "adversarial_safety_rate": 0.333,
        "avg_latency_ms": 6102,
        "p95_latency_ms": 13012,
    },
    "key_findings": [
        "Vocabulary bleed: HCM-domain questions pass the 0.35 OOD threshold due to shared terminology with DIGIT Studio chunks",
        "Domain-name injection: including 'DIGIT Studio' in an off-topic prompt raises similarity score above rejection threshold",
        "Adversarial gap: domain-wrapped harmful requests and direct prompt injection both bypass the domain guard",
        "Keyword-based evaluation failed 48% of valid questions in initial run — demonstrates why static EXPECTED_OUTPUT matching is insufficient for generative AI",
    ],
    "open_limitations": [
        "Adaptive response length not evaluated",
        "Positional coherence of retrieved chunks not evaluated",
        "Procedural fragmentation (partial step retrieval) not evaluated",
        "Verbatim content preservation (code blocks) not evaluated",
        "No multi-turn evaluation",
    ],
}

st.json(summary_block)

st.caption(
    "EGOV_RAG_V5 · eGovernments Foundation · "
    "Evaluation run against DIGIT Studio documentation · "
    "Gates Foundation AI Fellowship — India 2026"
)
