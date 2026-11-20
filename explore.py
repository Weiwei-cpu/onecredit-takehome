import json
from collections import Counter
from pathlib import Path

data_file = Path("data/companies.jsonl")

companies = []
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            companies.append(json.loads(line))

print(f"Total companies: {len(companies)}\n")

# 1. Print overview table of all 50 companies
header = f"{'ID':<12} {'Status':<10} {'Industry':<22} {'Yrs':<4} {'Dir':<4} {'SH':<4} {'Cases':<5} {'Company Name'}"
print(header)
print("-" * len(header))

statuses = Counter()
industries = Counter()
years_dist = Counter()
has_litigation = 0
company_rows = []

for c in companies:
    p = c.get("profile", {})
    r = c.get("relationships", {})
    yrs = c.get("financials", {}).get("years", [])
    cases = c.get("litigation", {}).get("cases", [])

    status = p.get("status", "N/A")
    industry = p.get("industry", "N/A")
    statuses[status] += 1
    industries[industry] += 1
    years_dist[len(yrs)] += 1
    if len(cases) > 0:
        has_litigation += 1

    print(
        f"{c['registrationNumber']:<12} "
        f"{status:<10} "
        f"{industry[:21]:<22} "
        f"{len(yrs):<4} "
        f"{len(r.get('directors', [])):<4} "
        f"{len(r.get('shareholders', [])):<4} "
        f"{len(cases):<5} "
        f"{p.get('companyName', 'N/A')}"
    )
    
    company_rows.append({
        "registrationNumber": c["registrationNumber"],
        "companyName": p.get("companyName", "N/A"),
        "status": status,
        "industry": industry,
        "financialYears": len(yrs),
        "directorsCount": len(r.get("directors", [])),
        "shareholdersCount": len(r.get("shareholders", [])),
        "courtCasesCount": len(cases),
    })

# 2. Summary statistics
print("\n=== DATASET SUMMARY ===")
print("Status counts      :", dict(statuses))
print("Financial years    :", dict(years_dist))
print(f"Companies w/ cases : {has_litigation} / {len(companies)}")
print("Top industries     :", dict(industries.most_common()))

# 3. Inspect exact attributes (keys) of the first company
sample = companies[0]
print("\n=== ATTRIBUTES (SCHEMA) IN EACH RECORD ===")
print("profile       :", list(sample["profile"].keys()))
print("relationships :", list(sample["relationships"].keys()))
if sample["financials"]["years"]:
    y0 = sample["financials"]["years"][0]
    print("financials.yr :", list(y0.keys()))
    print("  statements  :", list(y0["statements"].keys()))
    print("  ratios      :", list(y0["ratiosAndMetrics"].keys()))
print("litigation    :", list(sample["litigation"].keys()))

# 4. Save EDA summary to JSON file
eda_output = {
    "total_companies": len(companies),
    "status_counts": dict(statuses),
    "financial_years_distribution": dict(years_dist),
    "companies_with_litigation": has_litigation,
    "industry_distribution": dict(industries.most_common()),
    "companies": company_rows,
}

out_path = Path("eda_summary.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(eda_output, f, indent=2, ensure_ascii=False)

print(f"\nSaved EDA profile to {out_path}")