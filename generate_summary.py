import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

# 1. Load Gemini API Key from local .env
load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")

# Official Google AI Studio model endpoint
MODEL_NAME = "gemini-flash-latest"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# 2. Resilient API caller with retry logic for 429 rate limits
def call_gemini(prompt: str, max_retries: int = 5) -> str:
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY,
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,  # Low temperature strictly minimizes hallucination
            "maxOutputTokens": 800,
        }
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(API_URL, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                candidates = res_json.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
                return ""
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):  # Handle free-tier rate limits and server busy errors
                wait_time = (attempt + 1) * 10
                print(f"  [Rate Limit 429] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"  [HTTP Error {e.code}] {e.reason}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(4)
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

# 4. Version 2 Prompt: Structured, risk-first credit & supplier vetting summary
def get_v2_prompt(company_record: dict) -> str:
    return f"""You are a Senior Risk & Credit Analyst at OneCredit, Malaysia's credit bureau.
A vetting team is deciding whether to approve this company as a supplier or project partner.
Review the company's full record below and write a high-density, structured summary in Markdown.

Format Requirements:
### Executive Verdict & Profile
- **Company & Status**: Name, Registration Number, State, Industry, and Active status.
- **Nature of Business**: Key operational activities and years since incorporation.

### Ownership & Governance
- Controlling shareholder(s) and equity percentage.
- Key directors and any notable governance structure.

### Financial Health (3-Year Trend)
- Revenue and Profit trend (direction, margin trajectory).
- Liquidity & Solvency: Cash levels, borrowings, leverage/coverage ratios. Note audit opinions.

### Risk Flags & Litigation (CRITICAL)
- Court cases: Detail whether plaintiff or defendant, claim amounts, and case status. If NONE, explicitly state: "No litigation records found."
- Solvency/Operational risks: Highlight negative equity, high debt burdens, or declining profit margins.

Constraints:
- ZERO hallucination: Rely ONLY on the numbers and names provided.
- Be concise, objective, and action-oriented for credit risk decision makers.

Company Data:
{json.dumps(company_record, indent=2)}
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
            
            prompt = get_v1_prompt(comp) if version == "v1" else get_v2_prompt(comp)
            summary = call_gemini(prompt)

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