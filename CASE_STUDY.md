# Case Study: Automated Company Vetting Summaries

## 1. What a good summary is for this team, and why

### Business Context
The end users are a vetting team deciding whether to onboard a company as a **supplier or project partner**. They do not need a full accounting report—they need a **fast, trustworthy first-pass screening brief** (readable in under 30 seconds) to answer one core question: *"Is this company safe to proceed with, or are there red flags requiring a closer look?"*

### Data Exploration Findings (`explore.py`)
Before designing the summarization system, I profiled all **50 synthetic companies** in `data/companies.jsonl` (records as of 31 Dec 2025) to understand the data distribution and edge cases:

- **Company Status (Deal-Breakers):** 48 companies are `Active`, while **2 companies are `Winding up`** (`SYN-000545` and `SYN-001212`, both with 8 court cases). For a vetting team, a winding-up status is an immediate deal-breaker that must be surfaced at the very top of the summary.
- **Financial Data Completeness (Edge Cases):**
  - 38 companies have 3 years of financial statements.
  - 4 companies have 2 years, and 5 companies have only 1 year.
  - **3 companies have 0 years of financial data** (`SYN-000302`, `SYN-000333`, `SYN-001203`), with `SYN-001203` also carrying an active court case.
  - *Implication:* A rigid prompt asking for a "3-year trend" risks causing hallucinations on 0- or 1-year companies. The system must explicitly flag missing financials as an information gap.
- **Litigation Exposure:** **18 out of 50 companies (36%)** have court cases (ranging from 1 to 8 cases). Missing a defendant litigation record is far more costly than omitting minor operational details.
- **Industry Diversity:** Covers 11 industries (led by Hospitality, Food & Beverage, and Retail), with raw records averaging ~2,000+ tokens due to 24 statement line items and 4 nested ratio groups per year.

### Definition of a "Good" Summary
Based on the business goal and the dataset realities above, a good summary for this team must satisfy **5 criteria**:
1. **Risk-First Triage:** Surface deal-breakers (`Winding up`, defendant lawsuits, negative equity, qualified audits, missing financials) immediately—never bury them at the bottom.
2. **Strict Factual Grounding (Zero Hallucination):** Every RM figure, percentage, date, and case detail must match the source JSON exactly. When data is missing, state it explicitly.
3. **Decision-Ready Structure:** Consistent Markdown sections so analysts can scan 20+ companies quickly without hunting for information.
4. **Conciseness & Signal-to-Noise:** Distill 2,000+ tokens of raw filings into a high-density brief (~150–250 words) without dumping irrelevant ratios.
5. **Actionable Context:** Distinguish *plaintiff* vs. *defendant* in court cases, and highlight *direction* (improving vs. deteriorating financials) rather than just listing raw numbers.

---

## 2. What you built

### Sampling Strategy (Stratified 20-Company Sample)
Rather than processing all 50 records blindly or taking an arbitrary first-N slice, I built a **stratified sample of 20 companies** designed to stay safely within free-tier API token limits while stress-testing every risk profile discovered in EDA:
- **2 / 2 `Winding up` companies** (`SYN-000545`, `SYN-001212`) — deal-breaker triage test.
- **3 / 3 `0-year financial` companies** (`SYN-000302`, `SYN-000333`, `SYN-001203`) — missing-data / anti-hallucination test.
- **2 single-year financial companies** (`SYN-000802`, `SYN-001158`) — non-trend handling test.
- **6 active litigation companies** (`SYN-000165`, `SYN-000267`, `SYN-000286`, `SYN-000397`, `SYN-000640`, `SYN-001389`) — spanning 1 to 8 court cases across Plaintiff and Defendant roles.
- **7 active, 0-case companies** (`SYN-000029`, `SYN-000067`, `SYN-000148`, `SYN-000476`, `SYN-000546`, `SYN-001455`, `SYN-001569`) — spanning both healthy and financially distressed firms across 7 industries to test false-positive control.

### Model & Infrastructure Choices
- **Initial API Exploration:** I initially tested Google AI Studio (`gemini-flash-latest`), which revealed two practical constraints: internal reasoning tokens consumed `maxOutputTokens` (causing mid-sentence truncation until raised to 4096), and the free daily request cap (RPD) was quickly exhausted during retries.
- **Primary Generator:** Switched to **Groq (`openai/gpt-oss-120b`)** via its OpenAI-compatible REST endpoint (`temperature=0.2` for factual grounding). Measuring live response headers showed a generous **1,000 RPD** cap alongside a tight **8,000 TPM** (tokens per minute) ceiling.
- **Pipeline Reliability (`generate_summary.py`):** Implemented automatic `retry-after` backoff for HTTP `429`/`503`, empty-response guards (preventing silent failures from being cached), and idempotent resume so completed records are never regenerated.

### Version 1 (`summaries/v1.jsonl`) — Naive Baseline & Error Analysis
For `v1`, I passed the **raw, unprocessed company JSON** (~2,500–3,000 tokens per record) with a simple, unstructured procurement prompt. Manually auditing all 20 `v1` outputs revealed **4 concrete failure modes**:

1. **Arithmetic & Ratio Hallucinations:**
   - In `SYN-001212`, `v1` misread decimal net profit margins by 10×–100× (reporting `0.58%`, `0.03%`, and `0.01%` instead of the true `5.81%`, `2.65%`, and `0.85%`).
   - In `SYN-000545`, `v1` mis-summed pending court claims as `~RM 2.9M` (true sum: `RM 2.50M`) and misstated the leverage formula as `Liabilities / Assets` instead of `Liabilities / Equity`.
   - In `SYN-001389`, `v1` claimed two winding-up petitions were still pending when only one (`CASE-2024-5611`) was pending and the 2025 petition was withdrawn.
2. **Plaintiff vs. Defendant Role Confusion:**
   - In `SYN-000640`, 5 of 6 court cases list the company as **Plaintiff** (recovering debt owed *to* them), with only 1 Defendant judgment (`RM 310.6k`). `v1` lumped all debt-recovery judgments together (`> RM 2.8M`) and mischaracterized them as "material contingent liabilities."
3. **Temporal Grounding Hallucination:**
   - Because `v1` did not inject the dataset's as-of date (`31 December 2025`), it falsely flagged legitimate 2024–2025 director appointments in `SYN-000067` and `SYN-001158` as *"future dates / possible data entry errors."*
4. **Poor Scanability, Buried Risks & Token Bottlenecks:**
   - Summaries averaged **500–650 words** with inconsistent layouts and buried critical `Winding up` statuses inside profile tables instead of surfacing them first.
   - Raw JSON payloads saturated Groq's 8,000 TPM cap every 2–3 companies.

### Version 2 (`summaries/v2.jsonl`) — Feature-Engineered & Risk-First System
To fix all four `v1` failure modes systematically, `v2` separates **deterministic computation (Python)** from **narrative synthesis (LLM)**:

1. **Deterministic Preprocessing (`preprocess_company` in `generate_summary.py`):**
   - Extracts only high-signal vetting fields, compressing input payload size by **63.6%** (eliminating the 8,000 TPM bottleneck).
   - Pre-computes exact net profit margins (`%`), multi-year revenue deltas, and non-unqualified audit warnings in Python so the LLM never performs arithmetic.
   - Strictly separates litigation into `as_defendant` (true liability exposure: pending count/RM, judgment count/RM, winding-up petitions) vs. `as_plaintiff` (informational only).
   - *QA Catch during Iteration:* Inspecting the first `v2` draft caught a subtle schema-key bug—litigation type was stored under `caseType` rather than `type`, causing winding-up petitions to be missed. Fixing this single key immediately surfaced the pending RM 355,900 winding-up petition in `SYN-000286` and upgraded it to `🔴 HIGH RISK`.
2. **Temporal Grounding:** Explicitly injects `data_as_of_date: "2025-12-31"` into both the payload and system prompt.
3. **Risk-First, UI-Aware Markdown Template:** Enforces a concise 4-section bullet layout (~180–250 words) starting with a `🔴 HIGH RISK / 🟡 MEDIUM RISK / 🟢 LOW RISK` verdict. Replacing `v1`'s wide Markdown tables with concise bullets also fixed severe table-wrapping issues observed side-by-side in the local Viewer (`http://localhost:8000`).

---

## 3. How you know whether it's good

I evaluated `v1` vs. `v2` on the 20 matched companies using a **three-pronged evaluation strategy** (`evaluate.py`, full results saved in `eval_results.json`):

### 1. Qualitative Side-by-Side Inspection (Local Viewer)
Using the built-in Viewer (`http://localhost:8000`), I inspected `v1` and `v2` side-by-side across edge cases (`SYN-000545`, `SYN-001212`, `SYN-000286`, `SYN-000640`, `SYN-000302`). In the narrow UI card, `v1`'s multi-column tables broke across lines and buried `Winding up` statuses, whereas `v2` fit cleanly above the fold with the verdict on line 1.

### 2. Layer 1 — Deterministic Code-Based Checks (100% Objective)

| Metric | `v1` (Raw Baseline) | `v2` (Preprocessed + Risk-First) |
|---|---:|---:|
| **Input Payload Size** | Baseline (100%) | **-63.6% smaller** |
| **Average Word Count** | 606.2 words | **217.4 words** |
| **Concise (≤ 300 words)** | 5.0% (1/20) | **100.0% (20/20)** |
| **Structured Format Compliance** | 0.0% (0/20) | **100.0% (20/20)** |
| **`Winding up` Surfaced in First 35 Words** | 0/2 | **2/2 (100%)** |
| **Winding-Up Petition Recall** | 4/4 | **4/4 (100%)** |
| **Audit Warning Recall (Going Concern / Disclaimer)** | 2/2 | **2/2 (100%)** |
| **0-Year Financials Honesty (No Hallucination)** | 3/3 | **3/3 (100%)** |
| **Temporal Hallucination False Alarms** | 2 (`SYN-000067`, `SYN-001158`) | **0** |

### 3. Layer 2 — Blind Cross-Model LLM-as-a-Judge (`qwen/qwen3.8-27b`)
To avoid self-enhancement bias, I used a **completely different model family** (`qwen/qwen3.8-27b` judging `openai/gpt-oss-120b` at `temperature=0.0`) and **alternated Summary A / Summary B ordering** across odd/even companies to eliminate position bias:

| Rubric Dimension (1–5 Scale) | `v1` Avg | `v2` Avg | Gain |
|---|---:|---:|---:|
| **Factual Accuracy & No Hallucination** | 2.00 | **5.00** | +3.00 |
| **Risk Triage & Red-Flag Recall** | 3.40 | **5.00** | +1.60 |
| **Conciseness & 30s Scanability** | 2.10 | **4.95** | +2.85 |
| **Decision Utility** | 3.05 | **4.85** | +1.80 |
| **Blind Pairwise Win Rate** | 0 / 20 (0%) | **20 / 20 (100%)** | — |

### Honest Limitations of This Evaluation
- **Judge Alignment Bias:** Because the LLM judge verified summaries against the same preprocessed ground-truth JSON used by `v2`, `v2` aligns very naturally with the judge's rubric. A true production eval requires a **human-annotated golden set** scored by credit analysts to validate judge-to-human agreement.
- **Conservative Risk Calibration:** Inspecting `v2` verdicts showed **4 `🔴 HIGH RISK`**, **16 `🟡 MEDIUM RISK`**, and **0 `🟢 LOW RISK`** ratings. Even relatively stable companies (`SYN-000029`, `SYN-000546`) were rated Medium Risk because the prompt lacked explicit quantitative thresholds for `LOW RISK`.

---

## 4. What you'd do next, and how you'd keep improving it

If given more time or moving this prototype into production at OneCredit, I would prioritize:

1. **Deterministic Risk-Tier Calibration (Fixing the 0% `LOW RISK` Skew):**
   - Work with the vetting team to define explicit, auditable business rules for `HIGH` / `MEDIUM` / `LOW` risk tiers (e.g., `Winding up` or pending winding-up petition = automatic `HIGH`; clean litigation + positive FCF + quick ratio > 1.2 + leverage < 1.0 = `LOW`) computed in Python rather than leaving tiering solely to LLM discretion.
2. **Scale to Full Dataset & Production Observability:**
   - Run `v2` across all 50 companies (and future SSM/court feeds) using async batching, prompt versioning, and latency/token/cost tracking via **MLflow** or **Langfuse**.
3. **Automated Numeric Fact-Checking Regression Suite:**
   - Build a regex/NER post-processor that extracts every RM figure and percentage from generated Markdown and asserts exact membership in the source JSON before serving to users (hard guardrail against any numeric hallucination).
4. **Human-in-the-Loop Feedback Loop:**
   - Add lightweight thumbs-up/down + reason tags (*"missed risk"*, *"wrong figure"*, *"too verbose"*) inside the analyst UI, feeding weekly error clusters back into a golden evaluation set before shipping any prompt or model upgrade.

---

### Disclosure of AI Tools & Models Used
- **LLM APIs:**
  - **Groq (`openai/gpt-oss-120b`)** — Primary model used to generate `summaries/v1.jsonl` and `summaries/v2.jsonl`.
  - **Groq (`qwen/qwen3.8-27b`)** — Independent cross-model judge used in `evaluate.py`.
  - **Google AI Studio (`gemini-flash-latest`)** — Tested during initial prototyping before switching to Groq due to free-tier daily quota and thinking-token truncation constraints.
- **AI Coding Assistant:** Used an AI coding assistant (Antigravity / Gemini) as a pair-programming guide for brainstorming evaluation metrics, debugging API rate-limit and schema bugs, and refining documentation structure. All code, data exploration, error analysis, and verification runs were executed and reviewed directly by me.