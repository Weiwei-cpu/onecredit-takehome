import json
import re
from pathlib import Path


def load_jsonl_map(path: Path) -> dict:
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                records[item["registrationNumber"]] = item
    return records


def build_valid_percentages(comp: dict) -> set:
    """Collect all mathematically valid percentages derivable from the raw JSON:
    shareholder stakes (and sums), financial ratios (decimal * 100), and statement % changes."""
    valid = {100.0}

    # 1. Shareholder percentages (individual + pairwise/total sums)
    sh_pcts = [s.get("percentage") for s in comp.get("relationships", {}).get("shareholders", []) if s.get("percentage") is not None]
    for p in sh_pcts:
        valid.add(round(float(p), 2))
    # Common aggregations (e.g., top-2 or all-directors sum)
    if sh_pcts:
        sorted_sh = sorted(sh_pcts, reverse=True)
        running = 0.0
        for p in sorted_sh:
            running += float(p)
            valid.add(round(running, 2))

    # 2. All financial ratios (stored as decimals, e.g., 0.0581 -> 5.81%)
    years = sorted(comp.get("financials", {}).get("years", []), key=lambda x: x.get("financialYear", 0))
    for y in years:
        ratios = y.get("ratiosAndMetrics") or {}
        for group in ratios.values():
            if isinstance(group, dict):
                for val in group.values():
                    if isinstance(val, (int, float)):
                        valid.add(round(abs(val) * 100, 2))
                        valid.add(round(val * 100, 2))
        # Also common balance-sheet ratios (e.g. equity/assets, borrowings/assets)
        stmts = y.get("statements") or {}
        assets = stmts.get("totalAssets")
        if assets and assets != 0:
            for k in ("totalEquity", "borrowings", "totalLiabilities", "cash"):
                v = stmts.get(k)
                if isinstance(v, (int, float)):
                    valid.add(round(abs(v / assets) * 100, 2))

    # 3. Pairwise & multi-year % changes across key statement metrics
    if len(years) >= 2:
        keys = ("revenue", "grossProfit", "operatingProfit", "profitAfterTax", "totalAssets", "totalEquity", "cash", "borrowings")
        for i in range(len(years)):
            for j in range(i + 1, len(years)):
                s1 = years[i].get("statements") or {}
                s2 = years[j].get("statements") or {}
                for k in keys:
                    v1, v2 = s1.get(k), s2.get(k)
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1 != 0:
                        pct = ((v2 - v1) / abs(v1)) * 100
                        valid.add(round(pct, 2))
                        valid.add(round(abs(pct), 2))
                        # Approximate annualised rate if 2-year span
                        if j - i == 2:
                            valid.add(round(abs(pct) / 2, 2))
                            # CAGR approximation
                            if v1 > 0 and v2 > 0:
                                cagr = ((v2 / v1) ** 0.5 - 1) * 100
                                valid.add(round(abs(cagr), 2))
    return valid


def extract_percentages(text: str) -> list:
    """Extract all percentage numbers (e.g. 5.81%, -12.5%) from summary text."""
    # Match numbers followed by optional whitespace/narrow-nbsp and '%'
    matches = re.findall(r"(?<![A-Za-z0-9])([+-]?\d+(?:\.\d+)?)\s*%", text)
    return [float(m) for m in matches]


def audit_version(companies: dict, summaries: dict, version: str, tol: float = 0.15) -> dict:
    total_pcts = 0
    unmatched_pcts = 0
    flagged_examples = []

    for reg in sorted(summaries.keys()):
        comp = companies[reg]
        text = summaries[reg]["summary"]
        valid_set = build_valid_percentages(comp)
        found = extract_percentages(text)

        company_unmatched = []
        for val in found:
            total_pcts += 1
            # Match against absolute or signed value within rounding tolerance
            if not any(abs(abs(val) - abs(v)) <= tol for v in valid_set):
                unmatched_pcts += 1
                company_unmatched.append(val)

        if company_unmatched:
            flagged_examples.append({"registrationNumber": reg, "unmatched_pcts": company_unmatched})

    return {
        "version": version,
        "companies_checked": len(summaries),
        "total_percentages_cited": total_pcts,
        "unmatched_percentages": unmatched_pcts,
        "grounding_accuracy_pct": round((1 - unmatched_pcts / total_pcts) * 100, 2) if total_pcts else 100.0,
        "companies_with_unmatched": len(flagged_examples),
        "flagged_details": flagged_examples,
    }


def main():
    companies = load_jsonl_map(Path("data/companies.jsonl"))
    results = {}
    for ver in ("v1", "v1_preprocessed", "v2"):
        path = Path(f"summaries/{ver}.jsonl")
        if path.exists():
            summaries = load_jsonl_map(path)
            results[ver] = audit_version(companies, summaries, ver)

    print(json.dumps(results, indent=2))
    with open("numeric_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved programmatic numeric audit to numeric_audit_results.json")


if __name__ == "__main__":
    main()