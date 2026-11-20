import json
import os
import requests
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# 1. Load Groq API Key from local .env
load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("GROQ_API_KEY")

# Groq OpenAI-compatible endpoint (fast, free tier: 1,000 req/day)
MODEL_NAME = "openai/gpt-oss-120b"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# 2. Resilient API caller for Groq (OpenAI-compatible format)
def call_llm(prompt: str, max_retries: int = 5) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,   # Low temperature minimizes hallucination
        "max_tokens": 2048,   # Plenty of room for a concise summary (no hidden thinking tokens)
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif resp.status_code in (429, 503):
                # Read exact wait time from Groq's header if available
                wait_time = int(resp.headers.get("retry-after", (attempt + 1) * 10))
                print(f"  [Rate Limit {resp.status_code}] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"  [HTTP Error {resp.status_code}] {resp.text[:200]}")
                if attempt == max_retries - 1:
                    resp.raise_for_status()
                time.sleep(3)
        except Exception as e:
            wait_time = (attempt + 1) * 5
            print(f"  [Network/Timeout: {e}] Waiting {wait_time}s before retrying...")
            if attempt == max_retries - 1:
                raise e
            time.sleep(wait_time)
    return ""

# 3. Version 1 Prompt: Baseline unstructured summary
def get_v1_prompt(company_record: dict) -> str:
    return f"""You are an assistant helping a procurement team evaluate potential business partners.
Write a concise summary of the following Malaysian company in Markdown format.
Highlight who they are, their financial standing, and any red flags or litigation.
Keep it strictly grounded in the provided data. Do not make up any facts or numbers.

Company Data:
{json.dumps(company_record, indent=2)}
"""

# Data preprocessing steps for v2 after runing v1 where I found out v1 may cause a bit less accurate output
# 3.5 Data Preprocessing & Deterministic Feature Engineering (for v2)
def preprocess_company(comp: dict) -> dict:
    """Extracts high-signal vetting features and pre-computes exact metrics in Python
    to prevent LLM arithmetic hallucinations and Plaintiff/Defendant confusion."""
    p = comp.get("profile", {})
    r = comp.get("relationships", {})
    years = sorted(comp.get("financials", {}).get("years", []), key=lambda x: x.get("financialYear", 0))
    cases = comp.get("litigation", {}).get("cases", [])

    # 1. Pre-compute clean financial summary + audit warnings
    fin_years = []
    audit_flags = []
    for y in years:
        yr = y.get("financialYear")
        stmts = y.get("statements") or {}
        ratios = y.get("ratiosAndMetrics") or {}
        opinion = (y.get("audit") or {}).get("opinion", "Unknown")
        if opinion and opinion.lower() != "unqualified":
            audit_flags.append(f"FY{yr}: {opinion}")

        npm = (ratios.get("profitability") or {}).get("netProfitMargin")
        fin_years.append({
            "year": yr,
            "audit_opinion": opinion,
            "revenue_rm": stmts.get("revenue"),
            "net_profit_rm": stmts.get("profitAfterTax"),
            "net_profit_margin_pct": f"{npm * 100:.2f}%" if npm is not None else "N/A",
            "cash_rm": stmts.get("cash"),
            "borrowings_rm": stmts.get("borrowings"),
            "equity_rm": stmts.get("totalEquity"),
            "free_cash_flow_rm": (ratios.get("liquidityAndCashFlow") or {}).get("freeCashFlow"),
            "quick_ratio": (ratios.get("liquidityAndCashFlow") or {}).get("quickRatio"),
            "leverage_debt_to_equity": (ratios.get("capitalStructure") or {}).get("leverageRatio"),
            "interest_coverage": (ratios.get("capitalStructure") or {}).get("interestCoverageRatio"),
        })

    rev_trend = "N/A (fewer than 2 years of financials)"
    if len(fin_years) >= 2 and fin_years[0]["revenue_rm"] and fin_years[-1]["revenue_rm"]:
        start_rev = fin_years[0]["revenue_rm"]
        end_rev = fin_years[-1]["revenue_rm"]
        pct_change = ((end_rev - start_rev) / abs(start_rev)) * 100
        rev_trend = f"{pct_change:+.1f}% from FY{fin_years[0]['year']} to FY{fin_years[-1]['year']}"

    # 2. Strictly separate Defendant vs Plaintiff litigation in Python
    def_cases = [c for c in cases if c.get("role", "").lower() == "defendant"]
    pla_cases = [c for c in cases if c.get("role", "").lower() == "plaintiff"]

    def_pending = [c for c in def_cases if c.get("status", "").lower() == "pending"]
    def_judgments = [c for c in def_cases if "judgment" in c.get("status", "").lower()]
    winding_up_cases = [c for c in def_cases if "winding" in c.get("caseType", "").lower()]

    litigation_summary = {
        "total_cases": len(cases),
        "as_defendant_count": len(def_cases),
        "as_defendant_pending_count": len(def_pending),
        "as_defendant_pending_claim_rm": sum((c.get("claimAmount") or 0) for c in def_pending),
        "as_defendant_judgment_count": len(def_judgments),
        "as_defendant_judgment_claim_rm": sum((c.get("claimAmount") or 0) for c in def_judgments),
        "as_defendant_winding_up_petitions": [
            {"case": c.get("caseNumber"), "status": c.get("status"), "claim_rm": c.get("claimAmount")}
            for c in winding_up_cases
        ],
        "as_plaintiff_count": len(pla_cases),
        "as_plaintiff_pending_count": sum(1 for c in pla_cases if c.get("status", "").lower() == "pending"),
    }

    return {
        "data_as_of_date": "2025-12-31",
        "registrationNumber": comp.get("registrationNumber"),
        "profile": {
            "companyName": p.get("companyName"),
            "status": p.get("status"),
            "incorporationDate": p.get("incorporationDate"),
            "industry": p.get("industry"),
            "state": p.get("state"),
            "natureOfBusiness": p.get("natureOfBusiness"),
            "shareCapital_rm": p.get("shareCapital"),
            "formerNames": p.get("formerNames", []),
        },
        "governance": {
            "directors_count": len(r.get("directors", [])),
            "recent_director_appointments_2024_2025": [
                {"name": d.get("name"), "role": d.get("designation"), "appointed": d.get("appointedOn")}
                for d in r.get("directors", []) if str(d.get("appointedOn", "")) >= "2024-01-01"
            ],
            "major_shareholders": [
                {"name": s.get("name"), "type": s.get("type"), "pct": s.get("percentage")}
                for s in r.get("shareholders", [])
            ],
        },
        "financial_summary": {
            "years_available": len(fin_years),
            "revenue_trend": rev_trend,
            "audit_warnings": audit_flags,
            "yearly_metrics": fin_years,
        },
        "litigation_summary": litigation_summary,
    }
    
# 4. Version 2 Prompt: Structured, risk-first credit & supplier vetting summary
# 4. Version 2 Prompt: Risk-first, concise, pre-computed vetting brief
def get_v2_prompt(preprocessed_data: dict) -> str:
    return f"""You are a Senior Credit & Supplier Vetting Analyst at OneCredit (Malaysia).
Write a concise, decision-ready vetting summary in Markdown (~180-250 words) so an analyst can triage this company in 30 seconds.

IMPORTANT CONTEXT:
- All records are as of **31 December 2025**. Dates in 2024 or 2025 are recent historical events, NOT future dates or data errors.
- Use ONLY the exact pre-computed figures provided. Do NOT recalculate ratios or confuse Plaintiff vs. Defendant cases.

REQUIRED STRUCTURE (use these exact headings):
### Risk Verdict
- Start with ONE tag: **🔴 HIGH RISK**, **🟡 MEDIUM RISK**, or **🟢 LOW RISK** (if status is "Winding up", it MUST be 🔴 HIGH RISK).
- One crisp sentence explaining the primary reason for the verdict.

### Key Red Flags & Litigation
- Explicitly state Defendant litigation exposure (pending count, pending RM claim, judgments, and any winding-up petitions). Mention Plaintiff cases separately only as minor context (not as company liabilities). If 0 cases, write: "No litigation records found."
- Note any non-unqualified audit warnings, missing financial statements, recent name changes, or severe financial distress.

### Financial Health
- If `years_available == 0`, explicitly state: "No financial statements filed (major transparency gap)."
- Otherwise, summarize revenue trend, profitability (net profit & exact margin %), liquidity (quick ratio / cash / FCF), and leverage (Debt/Equity) in 3-4 concise bullet points.

### Profile & Governance
- 2-3 brief bullets: Industry/business, incorporation year, state, controlling shareholder(s), and any recent 2024–2025 director changes.

Preprocessed Company Data:
{json.dumps(preprocessed_data, indent=2)}
"""

# 5. Core execution pipeline
def run_pipeline(version: str = "v1", limit: int = None):
    data_file = Path(__file__).parent / "data" / "companies.jsonl"
    out_dir = Path(__file__).parent / "summaries"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{version}.jsonl"

    companies = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                companies.append(json.loads(line))
                
        # Stratified 20-company sample covering all key edge cases
    SAMPLE_IDS = {
        "SYN-000545", "SYN-001212",                                      # Winding up (2/2)
        "SYN-000302", "SYN-000333", "SYN-001203",                        # 0 financial years (3/3)
        "SYN-000802", "SYN-001158",                                      # 1 financial year (2)
        "SYN-000165", "SYN-000267", "SYN-000286", "SYN-000397", "SYN-000640", "SYN-001389",  # Litigation (6)
        "SYN-000029", "SYN-000067", "SYN-000148", "SYN-000476", "SYN-000546", "SYN-001455", "SYN-001569"  # Active/0-case (7)
    }
    companies = [c for c in companies if c.get("registrationNumber") in SAMPLE_IDS]

    if limit:
        companies = companies[:limit]

    print(f"\n=======================================================")
    print(f"Running [{version}] pipeline for {len(companies)} company records...")
    print(f"Output target: {out_file}")
    print(f"=======================================================\n")

    # Check already processed records to enable caching and resume
    existing_regs = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_regs.add(item.get("registrationNumber"))

    with open(out_file, "a", encoding="utf-8") as f_out:
        for i, comp in enumerate(companies):
            reg_num = comp.get("registrationNumber")
            comp_name = comp.get("profile", {}).get("companyName", "Unknown")
            
            if reg_num in existing_regs:
                print(f"[{i+1}/{len(companies)}] Skipping {reg_num} (already generated)")
                continue

            print(f"[{i+1}/{len(companies)}] Generating summary for {reg_num}: {comp_name}...")
            
            prompt = get_v1_prompt(comp) if version == "v1" else get_v2_prompt(preprocess_company(comp))
            summary = call_llm(prompt)
            
            # Handle empty summaries 
            if not summary:
                print(f"  [Skipped] Empty summary for {reg_num}, will retry next run.")
                continue

            # Persist summary in the exact schema specified by OneCredit
            record = {
                "registrationNumber": reg_num,
                "summary": summary
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            time.sleep(2)  # Respect free-tier rate limits smoothly

    print(f"\nCompleted! [{version}] summaries saved successfully to: {out_file}\n")

if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_pipeline(ver, lim)