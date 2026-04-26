"""
generate_plots.py — Comprehensive visualization suite for GitPRTriage evaluation.

Generates:
  1. Model Comparison (Baseline vs V1 vs V2) — overall + per-difficulty
  2. Individual model capability radars
  3. Baseline reward distribution by difficulty
  4. Training progression (V1 → V2 improvement)
  5. Reward component breakdown heatmaps
  6. Rolling-window reward curve from baseline episodes
  7. Score distribution violin plots

Run:
    python evaluation/generate_plots.py
"""

import json, os, math, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE_RESULTS = os.path.join(BASE_DIR, "pre_training",  "baseline_results.json")
BASELINE_SUMMARY = os.path.join(BASE_DIR, "pre_training",  "baseline_summary.json")
V1_SUMMARY       = os.path.join(BASE_DIR, "post_V1_training", "trained_summary.json")
V2_SUMMARY       = os.path.join(BASE_DIR, "post_V2_training", "trained_summary.json")

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "baseline": "#5E81F4",   # indigo-blue
    "v1":       "#F4A25E",   # amber
    "v2":       "#4FD1A5",   # teal-green
    "easy":     "#56CFE1",
    "medium":   "#FF9F1C",
    "hard":     "#E84855",
    "bg":       "#0F1117",
    "card":     "#1A1D27",
    "grid":     "#252836",
    "text":     "#E2E8F0",
    "subtext":  "#8892A4",
}

def apply_dark_style(fig, axes_list):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in (axes_list if hasattr(axes_list, "__iter__") else [axes_list]):
        ax.set_facecolor(PALETTE["card"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
        ax.grid(True, color=PALETTE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK]  {path}")

# ── Load data ──────────────────────────────────────────────────────────────────
with open(BASELINE_RESULTS) as f:
    raw_episodes = json.load(f)

with open(BASELINE_SUMMARY) as f:
    b_sum = json.load(f)

with open(V1_SUMMARY) as f:
    v1_sum = json.load(f)

with open(V2_SUMMARY) as f:
    v2_sum = json.load(f)

episodes_easy   = [e for e in raw_episodes if e["task_level"] == "easy"]
episodes_medium = [e for e in raw_episodes if e["task_level"] == "medium"]
episodes_hard   = [e for e in raw_episodes if e["task_level"] == "hard"]

LEVELS = ["easy", "medium", "hard"]
MODELS = ["Baseline (Qwen-2.5 1.5B)", "GRPO V1 (400 steps)", "GRPO V2 (600 steps)"]
MODEL_KEYS = ["baseline", "v1", "v2"]

def get_means(summary):
    return {lvl: summary["by_difficulty"][lvl]["mean"] for lvl in LEVELS}

b_means  = get_means(b_sum)
v1_means = get_means(v1_sum)
v2_means = get_means(v2_sum)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Model Comparison Bar Chart (Overall + Per-Difficulty)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Model Comparison Bar Chart …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
apply_dark_style(fig, axes)

# Overall bar
overall_vals = [b_sum["overall"]["mean"], v1_sum["overall"]["mean"], v2_sum["overall"]["mean"]]
colors_bar   = [PALETTE[k] for k in MODEL_KEYS]
bars = axes[0].bar(MODELS, overall_vals, color=colors_bar, width=0.55, zorder=3, edgecolor=PALETTE["bg"], linewidth=1.2)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Overall Mean Reward", fontweight="bold", fontsize=13)
axes[0].set_ylabel("Mean Reward")
for bar, val in zip(bars, overall_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.018, f"{val:.4f}",
                 ha="center", va="bottom", color=PALETTE["text"], fontsize=10, fontweight="bold")
axes[0].tick_params(axis="x", rotation=12)

# Per-difficulty grouped bar
x = np.arange(len(LEVELS))
w = 0.26
for i, (mk, col) in enumerate(zip(MODEL_KEYS, colors_bar)):
    vals = [b_means, v1_means, v2_means][i]
    yvals = [vals[lvl] for lvl in LEVELS]
    bars2 = axes[1].bar(x + (i-1)*w, yvals, w, label=MODELS[i], color=col,
                         zorder=3, edgecolor=PALETTE["bg"], linewidth=1.0)
axes[1].set_xticks(x)
axes[1].set_xticklabels([l.capitalize() for l in LEVELS])
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Per-Difficulty Mean Reward", fontweight="bold", fontsize=13)
axes[1].set_ylabel("Mean Reward")
axes[1].legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=8)

fig.suptitle("GitPRTriage — Model Performance Comparison", color=PALETTE["text"],
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "01_model_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Individual Model Capability Radar Charts
# ══════════════════════════════════════════════════════════════════════════════
print("[2/7] Individual Radar Charts …")

def radar_chart(ax, values, labels, color, title):
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values_c = values + values[:1]
    angles_c  = angles + angles[:1]
    ax.set_facecolor(PALETTE["card"])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color=PALETTE["text"], fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.50","0.75","1.00"], color=PALETTE["subtext"], fontsize=7)
    ax.grid(color=PALETTE["grid"], linewidth=0.8)
    ax.spines["polar"].set_color(PALETTE["grid"])
    ax.plot(angles_c, values_c, color=color, linewidth=2.0, linestyle="solid")
    ax.fill(angles_c, values_c, color=color, alpha=0.25)
    ax.set_title(title, color=PALETTE["text"], fontweight="bold", pad=18, fontsize=11)

fig2, ax2s = plt.subplots(1, 3, figsize=(16, 5), subplot_kw=dict(polar=True))
fig2.patch.set_facecolor(PALETTE["bg"])

radar_labels = ["Easy", "Medium", "Hard", "Overall"]
for ax, (mk, col, sm, title) in zip(ax2s, [
    ("baseline", PALETTE["baseline"], b_sum,  "Baseline (Qwen-2.5 1.5B)"),
    ("v1",       PALETTE["v1"],       v1_sum, "GRPO V1  (400 steps)"),
    ("v2",       PALETTE["v2"],       v2_sum, "GRPO V2  (600 steps)"),
]):
    vals = [sm["by_difficulty"][l]["mean"] for l in LEVELS] + [sm["overall"]["mean"]]
    radar_chart(ax, vals, radar_labels, col, title)

fig2.suptitle("Individual Model Capability Profiles", color=PALETTE["text"],
              fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
save(fig2, "02_capability_radars.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Baseline Reward Distribution by Difficulty
# ══════════════════════════════════════════════════════════════════════════════
print("[3/7] Baseline Reward Distribution …")
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
apply_dark_style(fig3, axes3)

for ax, (lvl, eps) in zip(axes3, [("easy", episodes_easy), ("medium", episodes_medium), ("hard", episodes_hard)]):
    rewards = [e["reward"] for e in eps]
    col = PALETTE[lvl]
    ax.hist(rewards, bins=20, color=col, edgecolor=PALETTE["bg"], linewidth=0.8, zorder=3)
    ax.axvline(np.mean(rewards), color="white", linestyle="--", linewidth=1.5, label=f"μ={np.mean(rewards):.3f}")
    ax.set_title(f"{lvl.capitalize()} Tasks  (n={len(eps)})", fontweight="bold", fontsize=12)
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=9)

fig3.suptitle("Baseline Reward Distributions per Difficulty Level", color=PALETTE["text"],
              fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig3, "03_baseline_distributions.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Training Progression: V1 vs V2 improvement arrows
# ══════════════════════════════════════════════════════════════════════════════
print("[4/7] Training Progression …")
fig4, ax4 = plt.subplots(figsize=(10, 6))
apply_dark_style(fig4, ax4)

stages   = ["Baseline", "Post V1\n(400 steps)", "Post V2\n(600 steps)"]
all_means_easy   = [b_means["easy"],   v1_means["easy"],   v2_means["easy"]]
all_means_medium = [b_means["medium"], v1_means["medium"], v2_means["medium"]]
all_means_hard   = [b_means["hard"],   v1_means["hard"],   v2_means["hard"]]

x4 = np.arange(3)
for vals, col, lvl in [
    (all_means_easy,   PALETTE["easy"],   "Easy"),
    (all_means_medium, PALETTE["medium"], "Medium"),
    (all_means_hard,   PALETTE["hard"],   "Hard"),
]:
    ax4.plot(x4, vals, "o-", color=col, linewidth=2.5, markersize=9, label=lvl, zorder=3)
    for xi, yi in zip(x4, vals):
        ax4.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 10), ha="center", color=col, fontsize=9, fontweight="bold")

ax4.set_xticks(x4)
ax4.set_xticklabels(stages)
ax4.set_ylim(0, 1.0)
ax4.set_title("Training Progression — Mean Reward per Difficulty", fontweight="bold", fontsize=13,
              color=PALETTE["text"])
ax4.set_ylabel("Mean Reward")
ax4.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

# Annotation: % improvement V1 → V2 overall
imp = (v2_sum["overall"]["mean"] - b_sum["overall"]["mean"]) / max(b_sum["overall"]["mean"], 1e-9) * 100
ax4.text(0.98, 0.05, f"Overall improvement\nBaseline → V2: +{imp:.1f}%",
         transform=ax4.transAxes, ha="right", va="bottom",
         color=PALETTE["v2"], fontsize=10, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["card"], edgecolor=PALETTE["v2"], alpha=0.9))

plt.tight_layout()
save(fig4, "04_training_progression.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Reward Component Breakdown Heatmap (baseline)
# ══════════════════════════════════════════════════════════════════════════════
print("[5/7] Reward Component Heatmap …")

def avg_component(eps, key):
    vals = [e["reward_breakdown"].get(key, 0.0) for e in eps]
    return np.mean(vals)

easy_components   = ["review_decision", "blocker_type"]
medium_components = ["review_decision", "defect_category", "faulty_line"]
hard_components   = ["review_decision", "defect_category", "faulty_line", "reviewer_team", "suggested_change"]
all_components    = list(dict.fromkeys(easy_components + medium_components + hard_components))

heatmap_data = np.full((3, len(all_components)), np.nan)
for ci, comp in enumerate(all_components):
    heatmap_data[0, ci] = avg_component(episodes_easy,   comp)
    heatmap_data[1, ci] = avg_component(episodes_medium, comp)
    heatmap_data[2, ci] = avg_component(episodes_hard,   comp)

fig5, ax5 = plt.subplots(figsize=(12, 4))
fig5.patch.set_facecolor(PALETTE["bg"])
ax5.set_facecolor(PALETTE["card"])

im = ax5.imshow(heatmap_data, cmap="YlGn", aspect="auto", vmin=0, vmax=0.55)
cbar = plt.colorbar(im, ax=ax5)
cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
cbar.set_label("Avg Component Score", color=PALETTE["text"])
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

comp_labels = [c.replace("_", "\n") for c in all_components]
ax5.set_xticks(range(len(all_components)))
ax5.set_xticklabels(comp_labels, color=PALETTE["text"], fontsize=9)
ax5.set_yticks(range(3))
ax5.set_yticklabels([l.capitalize() for l in LEVELS], color=PALETTE["text"], fontsize=10)

for ri in range(3):
    for ci in range(len(all_components)):
        val = heatmap_data[ri, ci]
        if not np.isnan(val):
            ax5.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                     color="white" if val > 0.25 else PALETTE["subtext"], fontsize=8, fontweight="bold")
        else:
            ax5.text(ci, ri, "N/A", ha="center", va="center", color=PALETTE["grid"], fontsize=7)

ax5.set_title("Baseline — Average Reward Component Scores by Difficulty", fontweight="bold",
              fontsize=13, color=PALETTE["text"], pad=12)
for spine in ax5.spines.values():
    spine.set_edgecolor(PALETTE["grid"])

plt.tight_layout()
save(fig5, "05_component_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Rolling-Window Reward Curve (Baseline Episodes)
# ══════════════════════════════════════════════════════════════════════════════
print("[6/7] Rolling-Window Reward Curve …")
fig6, ax6 = plt.subplots(figsize=(13, 5))
apply_dark_style(fig6, ax6)

all_rewards = [e["reward"] for e in raw_episodes]
episodes_x  = np.arange(1, len(all_rewards)+1)
window = 15
rolling = np.convolve(all_rewards, np.ones(window)/window, mode="valid")
roll_x  = np.arange(window, len(all_rewards)+1)

ax6.scatter(episodes_x, all_rewards, s=6, alpha=0.4, color=PALETTE["baseline"], zorder=2, label="Episode Reward")
ax6.plot(roll_x, rolling, color=PALETTE["v2"], linewidth=2.5, zorder=3, label=f"Rolling Mean (w={window})")

# colour-coded difficulty bands
for e in raw_episodes:
    col = PALETTE[e["task_level"]]
    ax6.axvline(e["episode_number"], color=col, alpha=0.04, linewidth=0.8)

legend_patches = [
    mpatches.Patch(color=PALETTE["easy"],   label="Easy"),
    mpatches.Patch(color=PALETTE["medium"], label="Medium"),
    mpatches.Patch(color=PALETTE["hard"],   label="Hard"),
]
ax6.legend(handles=[mpatches.Patch(color=PALETTE["baseline"], label="Episode Reward"),
                     mpatches.Patch(color=PALETTE["v2"], label=f"Rolling Mean (w={window})")] + legend_patches,
           facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=8)

ax6.set_xlim(1, len(all_rewards))
ax6.set_ylim(-0.05, 1.1)
ax6.set_xlabel("Episode Number")
ax6.set_ylabel("Reward")
ax6.set_title("Baseline Agent — Episode Rewards Over Time", fontweight="bold", fontsize=13, color=PALETTE["text"])
plt.tight_layout()
save(fig6, "06_episode_reward_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Score Distribution Violin / Box Plot (Baseline per difficulty)
# ══════════════════════════════════════════════════════════════════════════════
print("[7/7] Violin / Box Plot …")
fig7, ax7 = plt.subplots(figsize=(10, 6))
apply_dark_style(fig7, ax7)

data_by_level = [
    [e["reward"] for e in episodes_easy],
    [e["reward"] for e in episodes_medium],
    [e["reward"] for e in episodes_hard],
]
colors_v = [PALETTE["easy"], PALETTE["medium"], PALETTE["hard"]]

parts = ax7.violinplot(data_by_level, positions=[1,2,3], showmeans=True, showextrema=True)
for i, (body, col) in enumerate(zip(parts["bodies"], colors_v)):
    body.set_facecolor(col)
    body.set_alpha(0.65)
    body.set_edgecolor("white")
for part in ("cmeans", "cmaxes", "cmins", "cbars"):
    parts[part].set_color("white")
    parts[part].set_linewidth(1.4)

ax7.set_xticks([1, 2, 3])
ax7.set_xticklabels(["Easy", "Medium", "Hard"], fontsize=12)
ax7.set_ylim(-0.05, 1.1)
ax7.set_ylabel("Reward Score")
ax7.set_title("Baseline Reward Score Distributions (Violin Plot)", fontweight="bold",
              fontsize=13, color=PALETTE["text"])

for i, (col, lvl_data) in enumerate(zip(colors_v, data_by_level)):
    mu = np.mean(lvl_data)
    ax7.text(i+1, mu + 0.04, f"μ={mu:.3f}", ha="center", color=col, fontsize=9, fontweight="bold")

plt.tight_layout()
save(fig7, "07_violin_distributions.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — Summary Dashboard (composite)
# ══════════════════════════════════════════════════════════════════════════════
print("[BONUS] Summary Dashboard …")
fig8 = plt.figure(figsize=(16, 9))
fig8.patch.set_facecolor(PALETTE["bg"])
gs = GridSpec(2, 3, figure=fig8, hspace=0.45, wspace=0.38)

# -- Top-left: overall comparison bar --
ax_ov = fig8.add_subplot(gs[0, 0])
apply_dark_style(fig8, [ax_ov])
bars_ov = ax_ov.bar(["Baseline", "V1", "V2"], overall_vals, color=colors_bar, width=0.55, zorder=3,
                    edgecolor=PALETTE["bg"], linewidth=1.0)
ax_ov.set_ylim(0, 1.0)
ax_ov.set_title("Overall Mean Reward", fontweight="bold", fontsize=11)
for bar, val in zip(bars_ov, overall_vals):
    ax_ov.text(bar.get_x()+bar.get_width()/2, val+0.015, f"{val:.3f}",
               ha="center", color=PALETTE["text"], fontsize=9, fontweight="bold")

# -- Top-center: per-difficulty grouped bar --
ax_diff = fig8.add_subplot(gs[0, 1])
apply_dark_style(fig8, [ax_diff])
xd = np.arange(3)
wd = 0.26
for i, (mk, col) in enumerate(zip(MODEL_KEYS, colors_bar)):
    yv = [[b_means, v1_means, v2_means][i][lvl] for lvl in LEVELS]
    ax_diff.bar(xd+(i-1)*wd, yv, wd, label=["B","V1","V2"][i], color=col, zorder=3, edgecolor=PALETTE["bg"])
ax_diff.set_xticks(xd)
ax_diff.set_xticklabels(["Easy","Medium","Hard"])
ax_diff.set_ylim(0, 1.0)
ax_diff.set_title("Per-Difficulty Comparison", fontweight="bold", fontsize=11)
ax_diff.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=7)

# -- Top-right: progression lines --
ax_prog = fig8.add_subplot(gs[0, 2])
apply_dark_style(fig8, [ax_prog])
for vals_p, col_p, lvl_p in [(all_means_easy, PALETTE["easy"], "Easy"),
                               (all_means_medium, PALETTE["medium"], "Medium"),
                               (all_means_hard, PALETTE["hard"], "Hard")]:
    ax_prog.plot([0,1,2], vals_p, "o-", color=col_p, linewidth=2, markersize=7, label=lvl_p, zorder=3)
ax_prog.set_xticks([0,1,2])
ax_prog.set_xticklabels(["Baseline","V1","V2"])
ax_prog.set_ylim(0, 1.0)
ax_prog.set_title("Training Progression", fontweight="bold", fontsize=11)
ax_prog.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=7)

# -- Bottom-left: rolling reward --
ax_roll = fig8.add_subplot(gs[1, 0:2])
apply_dark_style(fig8, [ax_roll])
ax_roll.scatter(episodes_x, all_rewards, s=4, alpha=0.35, color=PALETTE["baseline"], zorder=2)
ax_roll.plot(roll_x, rolling, color=PALETTE["v2"], linewidth=2, zorder=3, label=f"Rolling Mean (w={window})")
ax_roll.set_xlim(1, len(all_rewards))
ax_roll.set_ylim(-0.05, 1.1)
ax_roll.set_xlabel("Episode")
ax_roll.set_ylabel("Reward")
ax_roll.set_title("Baseline Episode Rewards (Rolling Avg)", fontweight="bold", fontsize=11)
ax_roll.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=8)

# -- Bottom-right: violin --
ax_vio = fig8.add_subplot(gs[1, 2])
apply_dark_style(fig8, [ax_vio])
parts8 = ax_vio.violinplot(data_by_level, positions=[1,2,3], showmeans=True, showextrema=True)
for body, col in zip(parts8["bodies"], colors_v):
    body.set_facecolor(col); body.set_alpha(0.6); body.set_edgecolor("white")
for part in ("cmeans", "cmaxes", "cmins", "cbars"):
    parts8[part].set_color("white"); parts8[part].set_linewidth(1.2)
ax_vio.set_xticks([1,2,3])
ax_vio.set_xticklabels(["Easy","Med","Hard"])
ax_vio.set_ylim(-0.05, 1.1)
ax_vio.set_title("Score Distributions", fontweight="bold", fontsize=11)

fig8.suptitle("GitPRTriage — Full Evaluation Dashboard", color=PALETTE["text"],
              fontsize=17, fontweight="bold", y=1.01)
save(fig8, "00_dashboard.png")

print(f"\n[DONE] All 8 plots saved to: {OUT_DIR}")
