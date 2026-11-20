import json
import os
import re
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from generate_summary import preprocess_company

# Load Groq API Key
load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("GROQ_API_KEY")

# Cross-model Judge: Qwen family judging OpenAI-OSS family (eliminates self-enhancement bias)
JUDGE_MODEL = "qwen/qwen3.8-27b"
API_URL = "https://api.groq.com/openai/v1/chat/completions"


def load_jsonl_map(path: Path) -> dict:
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                records[item["registrationNumber"]] = item
    return records


def run_deterministic_checks(companies: dict, summaries: dict, version: str) -> dict:
    """Layer 1: 100% objective, code-based metrics."""
    word_counts = []
    within_300_words = 0
    temporal_false_alarms = 0
    winding_up_top_hits = 0
    winding_up_total = 0
    missing_fin_honest = 0
    missing_fin_total = 0
    audit_flag_hits = 0
    audit_flag_total = 0
    wu_petition_hits = 0
    wu_petition_total = 0
    format_compliant = 0

    required_headings = ["### risk verdict", "### key red flags", "### financial health", "### profile & governance"]

    for reg, item in summaries.items():
        text = item["summary"]
        lower = text.lower()
        words = text.split()
        word_counts.append(len(words))
        if len(words) <= 300:
            within_300_words += 1

        if all(h in lower for h in required_headings):
            format_compliant += 1

        if "future date" in lower or "data entry error" in lower:
            temporal_false_alarms += 1

        comp = companies[reg]
        pre = preprocess_company(comp)

        # Check Winding-up companies surfaced within first 35 words
        if comp["profile"].get("status", "").lower() == "winding up":
            winding_up_total += 1
            first_35 = " ".join(words[:35]).lower()
            if "winding" in first_35 or "high risk" in first_35:
                winding_up_top_hits += 1

        # Check 0-year financial companies explicitly state no financial statements
        if pre["financial_summary"]["years_available"] == 0:
            missing_fin_total += 1
            if "no financial" in lower or "no filed financial" in lower:
                missing_fin_honest += 1

        # Check non-unqualified audit warnings recalled
        if pre["financial_summary"]["audit_warnings"]:
            audit_flag_total += 1
            if "disclaimer" in lower or "going concern" in lower or "qualified" in lower:
                audit_flag_hits += 1

        # Check winding-up petition recall (for companies with defendant winding-up petitions)
        if pre["litigation_summary"]["as_defendant_winding_up_petitions"]:
            wu_petition_total += 1
            if "winding" in lower and "petition" in lower:
                wu_petition_hits += 1

    avg_words = round(sum(word_counts) / len(word_counts), 1) if word_counts else 0
    return {
        "version": version,
        "count": len(summaries),
        "avg_word_count": avg_words,
        "concise_under_300_words_pct": round(within_300_words / len(summaries) * 100, 1),
        "structured_format_compliance_pct": round(format_compliant / len(summaries) * 100, 1),
        "winding_up_surfaced_top_35w": f"{winding_up_top_hits}/{winding_up_total}",
        "winding_up_petition_recall": f"{wu_petition_hits}/{wu_petition_total}",
        "audit_warning_recall": f"{audit_flag_hits}/{audit_flag_total}",
        "missing_financials_honesty": f"{missing_fin_honest}/{missing_fin_total}",
        "temporal_hallucination_count": temporal_false_alarms,
    }


def call_judge(ground_truth: dict, summary_a: str, summary_b: str, max_retries: int = 5) -> dict:
    """Layer 2: Blind cross-model pairwise evaluation using Qwen 3.8 27B."""
    prompt = f"""You are an impartial Senior Credit Risk Reviewer evaluating two AI-generated company vetting summaries (Summary A and Summary B) against verified ground-truth company data (as of 31 December 2025).

Score BOTH Summary A and Summary B from 1 (poor) to 5 (excellent) on 4 criteria:
1. factual_accuracy: Exact match to ground-truth numbers, margins, Plaintiff vs. Defendant roles, and no temporal hallucinations.
2. risk_triage: Surfaces critical red flags (winding-up, defendant claims, audit warnings, missing financials) immediately.
3. conciseness: Readable in ~30 seconds (~180-280 words) without bloated tables.
4. decision_utility: Actionable for a supplier vetting team deciding whether to proceed or escalate.

Ground-Truth Facts (verified):
{json.dumps(ground_truth, indent=2)}

=== SUMMARY A ===
{summary_a}

=== SUMMARY B ===
{summary_b}

Respond ONLY with valid JSON in this exact format (no markdown fences):
{{
  "A": {{"factual_accuracy": 0, "risk_triage": 0, "conciseness": 0, "decision_utility": 0}},
  "B": {{"factual_accuracy": 0, "risk_triage": 0, "conciseness": 0, "decision_utility": 0}},
  "winner": "A" or "B" or "Tie",
  "reason": "One concise sentence explaining why."
}}"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                raw_text = resp.json()["choices"][0]["message"]["content"]
                # Strip <think>...</think> blocks if present in Qwen reasoning output
                clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                match = re.search(r"\{.*\}", clean_text, flags=re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                print("  [Parse Warning] Could not parse JSON, retrying...")
            elif resp.status_code in (429, 503):
                wait_time = int(resp.headers.get("retry-after", (attempt + 1) * 10))
                print(f"  [Judge Rate Limit {resp.status_code}] Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  [Judge HTTP {resp.status_code}] {resp.text[:200]}")
                time.sleep(3)
        except Exception as e:
            print(f"  [Judge Error] {e}")
            time.sleep(5)
    return {}


def main():
    companies = load_jsonl_map(Path("data/companies.jsonl"))
    v1_map = load_jsonl_map(Path("summaries/v1.jsonl"))
    v2_map = load_jsonl_map(Path("summaries/v2.jsonl"))

    common_ids = sorted(set(v1_map.keys()) & set(v2_map.keys()))
    print(f"\nEvaluating {len(common_ids)} matched companies (v1 vs v2)...\n")

    # Token payload reduction measurement (character proxy: ~4 chars/token)
    raw_chars = [len(json.dumps(companies[r])) for r in common_ids]
    pre_chars = [len(json.dumps(preprocess_company(companies[r]))) for r in common_ids]
    reduction_pct = round((1 - sum(pre_chars) / sum(raw_chars)) * 100, 1)

    # Layer 1: Deterministic Checks
    det_v1 = run_deterministic_checks(companies, v1_map, "v1")
    det_v2 = run_deterministic_checks(companies, v2_map, "v2")

    print("=== LAYER 1: DETERMINISTIC CHECKS ===")
    print(f"Input Payload Compression (v1 -> v2): -{reduction_pct}% characters/tokens")
    print(json.dumps({"v1": det_v1, "v2": det_v2}, indent=2))

    # Layer 2: Blind Cross-Model Judge (Qwen 3.8 27B)
    print(f"\n=== LAYER 2: BLIND CROSS-MODEL JUDGE ({JUDGE_MODEL}) ===")
    per_company_eval = []
    v1_totals = {"factual_accuracy": 0, "risk_triage": 0, "conciseness": 0, "decision_utility": 0}
    v2_totals = {"factual_accuracy": 0, "risk_triage": 0, "conciseness": 0, "decision_utility": 0}
    wins = {"v1": 0, "v2": 0, "Tie": 0}

    for idx, reg in enumerate(common_ids):
        gt = companies[reg]
        s1 = v1_map[reg]["summary"]
        s2 = v2_map[reg]["summary"]

        # Alternate A/B ordering to eliminate position bias
        swapped = (idx % 2 == 1)
        sum_a, sum_b = (s2, s1) if swapped else (s1, s2)

        print(f"[{idx+1}/{len(common_ids)}] Judging {reg} (A={'v2' if swapped else 'v1'})...")
        res = call_judge(gt, sum_a, sum_b)
        if not res or "A" not in res or "B" not in res:
            print(f"  [Warning] Skipping {reg} due to judge error.")
            continue

        scores_v1 = res["B"] if swapped else res["A"]
        scores_v2 = res["A"] if swapped else res["B"]
        raw_winner = res.get("winner", "Tie")
        if raw_winner == "A":
            winner = "v2" if swapped else "v1"
        elif raw_winner == "B":
            winner = "v1" if swapped else "v2"
        else:
            winner = "Tie"

        wins[winner] = wins.get(winner, 0) + 1
        for k in v1_totals:
            v1_totals[k] += scores_v1.get(k, 0)
            v2_totals[k] += scores_v2.get(k, 0)

        per_company_eval.append({
            "registrationNumber": reg,
            "swapped_order": swapped,
            "v1_scores": scores_v1,
            "v2_scores": scores_v2,
            "winner": winner,
            "reason": res.get("reason", ""),
        })
        time.sleep(3)

    n = len(per_company_eval)
    v1_avg = {k: round(v / n, 2) for k, v in v1_totals.items()} if n else {}
    v2_avg = {k: round(v / n, 2) for k, v in v2_totals.items()} if n else {}

    final_report = {
        "evaluated_companies": n,
        "judge_model": JUDGE_MODEL,
        "payload_reduction_pct": reduction_pct,
        "deterministic_checks": {"v1": det_v1, "v2": det_v2},
        "llm_judge_averages": {"v1": v1_avg, "v2": v2_avg},
        "pairwise_wins": wins,
        "details": per_company_eval,
    }

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\n=== FINAL EVALUATION SUMMARY ===")
    print("v1 Avg Scores:", v1_avg)
    print("v2 Avg Scores:", v2_avg)
    print("Pairwise Wins:", wins)
    print("\nSaved complete evaluation report to eval_results.json!")


if __name__ == "__main__":
    main()