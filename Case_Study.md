# Case Study: Automated Company Vetting Summaries

## 1. What a good summary is for this team, and why

### Business Context
The end users are a vetting team deciding whether to onboard a company as a **supplier or project partner**. They do not need a full accounting report—they need a **fast, trustworthy first-pass screening brief** (readable in under 30 seconds) to answer one core question: *"Is this company safe to proceed with, or are there red flags requiring a closer look?"*

### Step 1: Data Exploration Findings (`explore.py`)
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
*(TODO: We will fill this in after running v1 vs v2 with Groq + Data Preprocessing)*

---

## 3. How you know whether it's good
*(TODO: We will fill this in after running our automated evaluation + cross-model judge)*

---

## 4. What you'd do next, and how you'd keep improving it
*(TODO: We will fill this in at the end)*