import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Dynamically load both evaluation JSON artifacts (zero hard-coding)
num_audit = json.loads(Path("numeric_audit_results.json").read_text(encoding="utf-8"))
llm_eval = json.loads(Path("eval_results.json").read_text(encoding="utf-8"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("OneCredit Company Summaries: Programmatic vs. Cross-Model Evaluation (n=20)", fontsize=13, fontweight="bold")

# --- Left Panel: Programmatic % Grounding Accuracy (v1 vs v1_preprocessed vs v2) ---
versions = [v for v in ("v1", "v1_preprocessed", "v2") if v in num_audit]
labels_left = {"v1": "v1\n(Raw + Simple)", "v1_preprocessed": "v1_preprocessed\n(Ablation)", "v2": "v2\n(Final)"}
accs = [num_audit[v]["grounding_accuracy_pct"] for v in versions]
colors_left = ["#d9534f", "#f0ad4e", "#2e8b57"][:len(versions)]

bars = ax1.bar([labels_left[v] for v in versions], accs, color=colors_left, width=0.55)
ax1.set_ylim(0, 120)
ax1.set_ylabel("Percentage Grounding Accuracy (%)")
ax1.set_title("Layer 1: Deterministic Python Audit (±0.15pp)")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.6, f"{val:.2f}%", ha="center", fontweight="bold")

# --- Right Panel: Blind Cross-Model Judge Scores (v1 vs v2) ---
dims = ["factual_accuracy", "risk_triage", "conciseness", "decision_utility"]
dim_labels = ["Factual\nAccuracy", "Risk\nTriage", "Conciseness", "Decision\nUtility"]
v1_scores = [llm_eval["llm_judge_averages"]["v1"][d] for d in dims]
v2_scores = [llm_eval["llm_judge_averages"]["v2"][d] for d in dims]

x = np.arange(len(dims))
w = 0.35
b1 = ax2.bar(x - w / 2, v1_scores, width=w, label="v1 (Baseline)", color="#d9534f")
b2 = ax2.bar(x + w / 2, v2_scores, width=w, label="v2 (Final)", color="#2e8b57")
ax2.set_ylim(0, 6.0)
ax2.set_xticks(x)
ax2.set_xticklabels(dim_labels)
ax2.set_ylabel("Average Score (1–5 Scale)")
ax2.set_title(f"Layer 2: Blind Judge ({llm_eval['judge_model']})")
ax2.legend(loc="upper center", ncol=2)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
for bar in list(b1) + list(b2):
    val = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
out_img = Path("eval_comparison.png")
plt.savefig(out_img, dpi=180)
print(f"Saved chart to {out_img}")