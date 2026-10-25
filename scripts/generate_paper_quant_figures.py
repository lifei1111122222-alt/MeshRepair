"""Rebuild final quantitative PDF figures from frozen experiment summaries.

Figure contracts
----------------
results_correct_safe.pdf: endpoint performance on the separate 50/60-case formal
    evaluations. Exact rate CIs and paired-bootstrap differences come from source JSON.
results_development_exposed_supplement.pdf: the 570-case historical replication,
    kept separate because its cost-target-locked policy differs from the final method.
results_geometry_tradeoff.pdf: paired geometric gains and their quality/locality
    costs on common correct-safe same-target outputs in the 50/60-case cohorts.
results_ablation.pdf: endpoint decomposition for four cross-category mechanism
    variants and stored paired-bootstrap differences from the same 60 inputs.
results_scan_robustness.pdf: seven within-object scan perturbations; uncertainty
    comes from the stored source-object cluster bootstrap, never 168 independent n.
results_risk_coverage.pdf: empirical holdout-only risk--coverage step curve.
results_runtime.pdf: 50-case fresh input-to-verified-output runtimes with one mesh
    as the timing unit. Device arithmetic differs by method, as stated in protocol.
results_collar_policy.pdf: every paired source-level seam change between fixed and
    movable collar in the full predicted-target policy, plus stored mean intervals.
results_fairing_tradeoff.pdf: paired seam, motion and lower-tail quality changes
    after the same safe fallback pool, with prescribed targets and all outliers.

Every image is a vector PDF with editable Helvetica text. No data are invented,
smoothed, pooled across cohorts, or selected according to the final outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("TCMR_FIGURE_DIR", str(ROOT / "figures")))
FONTS = Path(os.environ.get("TCMR_HELVETICA_DIR", str(Path.home() / "AppData" / "Local" / "HelveticaFonts")))
COLOR = {
    "coral": "#E38067",
    "indigo": "#271A92",
    "violet": "#724BB6",
    "purple": "#6F15AC",
    "magenta": "#A730A2",
}
METHOD_COLOR = {
    "tcmr": COLOR["indigo"],
    "cgal_pool": COLOR["violet"],
    "semigcn": COLOR["magenta"],
    "meshfix": COLOR["coral"],
}
METHOD_LABEL = {
    "tcmr": "TCMR",
    "cgal_pool": "CGAL",
    "semigcn": "SeMIGCN",
    "meshfix": "MeshFix",
}


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def setup_style() -> None:
    for filename in ("H-Helvetica-6__02__Helvetica.ttf", "H-Helvetica-6__03__Helvetica-Bold.ttf"):
        path = FONTS / filename
        if not path.exists():
            raise FileNotFoundError(f"Required installed Helvetica font missing: {path}")
        font_manager.fontManager.addfont(str(path))
    # Verify there is no silent fallback in any exported figure.
    resolved = font_manager.findfont(font_manager.FontProperties(family="Helvetica"), fallback_to_default=False)
    if "Helvetica" not in Path(resolved).name:
        raise RuntimeError(f"Matplotlib did not resolve Helvetica: {resolved}")
    mpl.rcParams.update({
        "font.family": "Helvetica",
        "font.size": 11.0,
        "axes.labelsize": 11.2,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.85,
        "xtick.major.width": 0.85,
        "ytick.major.width": 0.85,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
        "text.usetex": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format="pdf", bbox_inches="tight", pad_inches=0.11)
    preview_dir = os.environ.get("PAPER_FIGURE_PREVIEW_DIR")
    if preview_dir:
        preview = Path(preview_dir)
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / name.replace(".pdf", ".png"), format="png",
                    dpi=175, bbox_inches="tight", pad_inches=0.11)
    plt.close(fig)


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.13) -> None:
    ax.text(x, 1.035, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12.0, weight="bold", color="black", clip_on=False)


def ci_bar(ax: plt.Axes, x: float, y: float, lo: float, hi: float,
           color: str, marker: str = "o", ms: float = 6.8) -> None:
    ax.plot([lo, hi], [y, y], color=color, lw=1.8, solid_capstyle="round", zorder=2)
    ax.plot([lo, lo], [y - 0.065, y + 0.065], color=color, lw=1.5, zorder=2)
    ax.plot([hi, hi], [y - 0.065, y + 0.065], color=color, lw=1.5, zorder=2)
    ax.plot([x], [y], marker=marker, color=color, mec="white", mew=0.75,
            ms=ms, linestyle="None", zorder=3)


def endpoint_rate(d: dict, method: str) -> tuple[float, float, float]:
    s = d["summary"][method]
    return s["correct_safe_rate"], *s["correct_safe_rate_ci95"]


def figure_correct_safe() -> None:
    HOLDOUT = load_json("experiments/formal_coseg_external_holdout/unified_evaluation.json")
    CROSS = load_json("experiments/formal_coseg_cross_category/unified_evaluation.json")
    assert HOLDOUT["n"] == 50 and HOLDOUT["complete"]
    assert CROSS["n"] == 60 and CROSS["complete"]
    fig = plt.figure(figsize=(6.88, 5.65))
    a = fig.add_axes([0.12, 0.57, 0.83, 0.32])
    b = fig.add_axes([0.36, 0.12, 0.59, 0.28])
    cohort_x = [0, 1.45]
    cohorts = [
        (HOLDOUT, "Independent holdout\n(n = 50)"),
        (CROSS, "Cross-category\n(n = 60)"),
    ]
    offset = {"tcmr": -0.315, "cgal_pool": -0.105, "semigcn": 0.105, "meshfix": 0.315}
    for i, (d, _) in enumerate(cohorts):
        for method in offset:
            rate, lo, hi = endpoint_rate(d, method)
            x = cohort_x[i] + offset[method]
            rate, lo, hi = np.array([rate, lo, hi]) * 100
            a.bar(x, rate, width=0.175, color=METHOD_COLOR[method],
                  edgecolor=METHOD_COLOR[method], linewidth=0.8, zorder=2)
            a.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]],
                       fmt="none", ecolor="black", elinewidth=1.0,
                       capsize=3.0, capthick=1.0, zorder=3)
            label = f"{rate:.1f}".removesuffix(".0")
            a.text(x, hi + 3.2, label, ha="center", va="bottom",
                   fontsize=10.5, color="black")
    a.set_xticks(cohort_x, [c[1] for c in cohorts])
    a.set_xlim(-0.52, 1.97)
    a.set_ylim(0, 114)
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("Correct-safe rate (%)")
    a.tick_params(axis="x", length=0, pad=8)
    panel_letter(a, "a", -0.105)

    paired_rows = [
        (3, HOLDOUT, "tcmr_vs_cgal_pool", COLOR["violet"]),
        (2, HOLDOUT, "tcmr_vs_semigcn", COLOR["magenta"]),
        (1, CROSS, "tcmr_vs_cgal_pool", COLOR["violet"]),
        (0, CROSS, "tcmr_vs_semigcn", COLOR["magenta"]),
    ]
    for y, d, comparison, color in paired_rows:
        q = d["paired_comparisons"][comparison]["correct_safe_endpoint"]
        center, (lo, hi) = q["difference_pp"], q["bootstrap_ci95_pp"]
        b.barh(y, center, height=0.47, color=color, edgecolor=color,
               linewidth=0.8, zorder=2)
        b.errorbar(center, y, xerr=[[center - lo], [hi - center]],
                   fmt="none", ecolor="black", elinewidth=1.0,
                   capsize=3.0, capthick=1.0, zorder=3)
        if center == 0:
            b.plot([0, 0], [y - 0.235, y + 0.235], color=color, lw=2, zorder=4)
        label = "0.0" if center == 0 else f"{center:+.1f}"
        b.text(hi + 1.3, y, label.replace("-", "−"),
               ha="left", va="center", color="black", fontsize=10.5)
    b.axvline(0, color="black", lw=0.85, zorder=0)
    b.set_yticks([3, 2, 1, 0],
                 ["Holdout − CGAL", "Holdout − SeMIGCN",
                  "Cross-category − CGAL", "Cross-category − SeMIGCN"])
    b.set_ylim(-0.50, 3.48)
    b.set_xlim(-9, 39)
    b.set_xticks([-5, 0, 10, 20, 30])
    b.set_xlabel("Paired TCMR − comparator difference\n(percentage points)")
    b.tick_params(axis="y", length=0, pad=7)
    panel_letter(b, "b", -0.555)
    handles = [Patch(facecolor=METHOD_COLOR[m], edgecolor="none", label=METHOD_LABEL[m])
               for m in offset]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.015),
               ncol=4, frameon=False, handlelength=1.15, handletextpad=0.45,
               columnspacing=1.15)
    save(fig, "results_correct_safe.pdf")


def figure_development_supplement() -> None:
    LARGE = load_json("experiments/large_scale_replication/unified_metrics.json")
    assert LARGE["n_source_clean"] == 570 and LARGE["historical_exposure"]
    tcmr_policy = {r["policy"] for r in LARGE["rows"] if r["method"] == "TCMR"}
    assert tcmr_policy == {"cost_target_locked_pool_with_joint"}
    fig = plt.figure(figsize=(6.86, 4.75))
    a = fig.add_axes([0.32, 0.55, 0.62, 0.32])
    b = fig.add_axes([0.32, 0.18, 0.62, 0.20])
    for y, key, color, name in [
        (1, "TCMR", COLOR["indigo"], "Earlier policy"),
        (0, "CGAL_shared_prior", COLOR["violet"], "Shared-prior CGAL"),
    ]:
        q = LARGE["summary"][key]
        ci_bar(a, q["correct_safe_rate"] * 100, y,
               q["correct_safe_rate_ci95"]["low"] * 100,
               q["correct_safe_rate_ci95"]["high"] * 100, color)
    a.set_yticks([1, 0], ["Earlier policy", "Shared-prior CGAL"])
    a.set_ylim(-0.5, 1.5)
    a.set_xlim(38, 67)
    a.set_xticks([40, 50, 60])
    a.tick_params(axis="y", length=0, pad=6)
    a.set_xlabel("Correct-safe rate (%)")
    panel_letter(a, "a", -0.48)
    q = LARGE["paired_tcmr_vs_cgal"]["correct_safe_rate_delta_tcmr_minus_cgal_bootstrap95"]
    ci_bar(b, q["estimate"] * 100, 0, q["low"] * 100, q["high"] * 100, COLOR["indigo"])
    b.axvline(0, color="black", lw=0.85, zorder=0)
    b.set_yticks([0], ["Earlier policy − CGAL"])
    b.set_ylim(-0.5, 0.5)
    b.set_xlim(-2, 15)
    b.set_xticks([0, 5, 10, 15])
    b.tick_params(axis="y", length=0, pad=6)
    b.set_xlabel("Paired difference (percentage points)")
    panel_letter(b, "b", -0.48)
    fig.text(0.51, 0.055, "Development-exposed, cost-target-locked policy; n = 570",
             ha="center", va="center", fontsize=10.5, color="black")
    save(fig, "results_development_exposed_supplement.pdf")


def figure_geometry_tradeoff() -> None:
    HOLDOUT = load_json("experiments/formal_coseg_external_holdout/unified_evaluation.json")
    CROSS = load_json("experiments/formal_coseg_cross_category/unified_evaluation.json")
    fig = plt.figure(figsize=(6.88, 5.15))
    # The same paired cohorts appear in every panel, so cohort labels are shared
    # by column.  This leaves room for four independent horizontal scales.
    axs = [fig.add_axes(bounds) for bounds in [
        [0.22, 0.64, 0.31, 0.23], [0.62, 0.64, 0.32, 0.23],
        [0.22, 0.21, 0.31, 0.23], [0.62, 0.21, 0.32, 0.23],
    ]]
    metrics = [
        ("seam_normal_absolute_mean", "Mean absolute seam\nangle (rad)", (-0.49, 0.08), [-0.4, -0.2, 0]),
        ("triangle_quality_mean", "Mean triangle quality", (-0.025, 0.11), [0, .05, .1]),
        ("retained_displacement_rms_over_edge", "Retained RMS / input edge", (-0.0006, 0.006), [0, .002, .004, .006]),
        ("changed_surface_area_ratio", "Changed area / input area", (-0.02, 0.26), [0, .1, .2]),
    ]
    for i, (ax, (key, xlabel, lim, ticks)) in enumerate(zip(axs, metrics)):
        ax.axvline(0, color="black", lw=0.85, zorder=0)
        for y, d, color, marker in [
            (1, HOLDOUT, COLOR["indigo"], "o"),
            (0, CROSS, COLOR["coral"], "D"),
        ]:
            q = d["paired_comparisons"]["tcmr_vs_cgal_pool"][key]
            expected = 49 if d is HOLDOUT else 51
            assert q["n"] == expected
            ci_bar(ax, q["mean_difference"], y, *q["bootstrap_ci95"], color, marker)
        ax.set_xlim(*lim)
        ax.set_xticks(ticks)
        ax.set_ylim(-0.55, 1.55)
        if i % 2 == 0:
            ax.set_yticks([1, 0], ["Holdout\n(49 pairs)", "Cross-category\n(51 pairs)"])
            ax.tick_params(axis="y", length=0, pad=6)
            panel_letter(ax, "abcd"[i], -0.44)
        else:
            ax.set_yticks([])
            panel_letter(ax, "abcd"[i], -0.13)
        ax.set_xlabel(xlabel, labelpad=7)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=COLOR["indigo"],
               markersize=8, label="Holdout (49 paired cases)"),
        Line2D([0], [0], marker="D", linestyle="None", color=COLOR["coral"],
               markersize=7, label="Cross-category (51 paired cases)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.995),
               ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.15)
    fig.text(0.58, 0.045, "Paired difference (TCMR − CGAL pool)",
             ha="center", va="center", fontsize=11.0, color="black")
    save(fig, "results_geometry_tradeoff.pdf")


def figure_ablation() -> None:
    CROSS = load_json("experiments/formal_coseg_cross_category/unified_evaluation.json")
    assert CROSS["n"] == 60 and CROSS["complete"]
    variants = [
        ("tcmr", "Full method"),
        ("ablation_cue_only", "Cue only"),
        ("ablation_prior_only", "Prior only"),
        ("ablation_no_safety_gate", "No safety gate"),
    ]
    fig = plt.figure(figsize=(6.86, 6.7))
    a = fig.add_axes([0.24, 0.57, 0.70, 0.27])
    b = fig.add_axes([0.34, 0.11, 0.60, 0.29])
    ys = [3, 2, 1, 0]
    for y, (method, label) in zip(ys, variants):
        q = CROSS["summary"][method]
        n_safe, n_wrong = q["correct_safe_n"], q["wrong_accepted_n"]
        n_other = 60 - n_safe - n_wrong
        assert q["accepted_n"] == n_safe + n_wrong
        left = 0
        for value, color in [(n_safe, COLOR["indigo"]),
                             (n_wrong, COLOR["coral"]),
                             (n_other, COLOR["violet"])]:
            if value:
                a.barh(y, value, left=left, height=0.58, color=color,
                       edgecolor="white", linewidth=0.9)
                if value >= 4:
                    a.text(left + value / 2, y, str(value), color="white",
                           fontsize=10.7, ha="center", va="center", weight="bold")
            left += value
    a.set_yticks(ys, [label for _, label in variants])
    a.set_xlim(0, 62)
    a.set_xticks([0, 15, 30, 45, 60])
    a.set_ylim(-0.55, 3.55)
    a.tick_params(axis="y", length=0, pad=8)
    a.set_xlabel("Inputs (n = 60)")
    panel_letter(a, "a", -0.30)

    compared = [
        (2, "ablation_cue_only", "Cue only"),
        (1, "ablation_prior_only", "Prior only"),
        (0, "ablation_no_safety_gate", "No safety gate"),
    ]
    b.axvline(0, color="black", lw=0.85, zorder=0)
    for y, method, label in compared:
        q = CROSS["paired_comparisons"][f"tcmr_vs_{method}"]["correct_safe_endpoint"]
        ci_bar(b, q["difference_pp"], y, *q["bootstrap_ci95_pp"],
               COLOR["purple"], marker="D")
    b.set_yticks([2, 1, 0], [label for _, _, label in compared])
    b.set_xlim(-8, 20)
    b.set_ylim(-0.55, 2.55)
    b.set_xticks([-5, 0, 5, 10, 15, 20])
    b.tick_params(axis="y", length=0, pad=8)
    b.set_xlabel("Full − ablation: paired correct-safe\ndifference (percentage points)")
    panel_letter(b, "b", -0.47)
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=c,
                      markeredgecolor="white", markersize=9, label=label)
               for c, label in [(COLOR["indigo"], "Correct-safe"),
                                (COLOR["coral"], "Wrong accepted"),
                                (COLOR["violet"], "No output accepted")]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               ncol=3, frameon=False, handletextpad=0.25, columnspacing=0.8)
    save(fig, "results_ablation.pdf")


def figure_scan_robustness() -> None:
    SCAN = load_json("experiments/scan_robustness/report.json")
    assert SCAN["n_sources"] == 24 and SCAN["n_planned_cases"] == 168
    conditions = [
        ("clean", "Clean"),
        ("noise_low", "Low noise"),
        ("noise_high", "High noise"),
        ("resolution_half", "Half resolution"),
        ("resolution_quarter", "Quarter resolution"),
        ("hole_small", "Small hole"),
        ("hole_large", "Large hole"),
    ]
    fig = plt.figure(figsize=(6.88, 8.2))
    a = fig.add_axes([0.33, 0.58, 0.62, 0.34])
    b = fig.add_axes([0.33, 0.10, 0.62, 0.34])
    y = np.arange(len(conditions))[::-1]
    seam_label = []
    success_label = []
    for pos, (key, label) in zip(y, conditions):
        d = SCAN["by_condition"][key]
        q = d["paired"]["geometry"]["seam_normal_absolute_mean"]
        assert d["n_planned"] == 24 and q["n_sources"] == q["n"]
        ci_bar(a, q["mean_difference"], pos, *q["source_cluster_ci95"], COLOR["indigo"], ms=6.2)
        seam_label.append(f"{label} ({q['n']} pairs)")
        s = d["summary"]["tcmr"]
        t = d["summary"]["cgal_library"]
        assert s["correct_safe_n"] == t["correct_safe_n"]
        ci_bar(b, s["correct_safe_rate"] * 100, pos,
               *(100 * v for v in s["correct_safe_rate_ci95"]), COLOR["coral"], marker="D", ms=6.2)
        success_label.append(label)
    a.axvline(0, color="black", lw=0.85, zorder=0)
    a.set_xlim(-0.079, 0.012)
    a.set_xticks([-.075, -.05, -.025, 0])
    a.set_yticks(y, seam_label)
    a.set_ylim(-0.60, 6.60)
    a.tick_params(axis="y", length=0, pad=7)
    a.set_xlabel("TCMR − CGAL pool: absolute seam (rad)")
    panel_letter(a, "a", -0.48)
    b.set_xlim(25, 107)
    b.set_xticks([25, 50, 75, 100])
    b.set_yticks(y, success_label)
    b.set_ylim(-0.60, 6.60)
    b.tick_params(axis="y", length=0, pad=7)
    b.set_xlabel("Correct-safe rate, both methods (%)")
    panel_letter(b, "b", -0.48)
    save(fig, "results_scan_robustness.pdf")


def figure_risk_coverage() -> None:
    path = ROOT / "experiments/risk_coverage/risk_coverage.csv"
    with path.open(encoding="utf-8", newline="") as f:
        data = [r for r in csv.DictReader(f) if r["dataset"] == "50 holdout"]
    assert len(data) >= 50 and all(int(r["n"]) == 50 for r in data)
    # Each threshold is a discrete operating point; duplicate coverage values are
    # collapsed without smoothing or changing the measured empirical risk.
    points = {}
    for r in data:
        if int(r["accepted_n"]) > 0:
            points[int(r["accepted_n"])] = (float(r["coverage"]), float(r["risk"]))
    assert points[50] == (1.0, .02) and points[49] == (.98, 0.0)
    accepted = sorted(points)
    x = np.array([points[k][0] * 100 for k in accepted])
    z = np.array([points[k][1] * 100 for k in accepted])
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.20, top=0.94)
    ax.step(x, z, where="post", lw=2.15, color=COLOR["purple"], zorder=2)
    ax.scatter([98, 100], [0, 2], s=[72, 74], marker="o",
               facecolor=[COLOR["indigo"], COLOR["coral"]],
               edgecolor="white", linewidth=0.9, zorder=3)
    ax.annotate("49 accepted; 0 wrong", xy=(98, 0), xytext=(84, 0.42),
                ha="center", va="bottom", fontsize=10.6,
                arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.75})
    ax.annotate("50 accepted; 1 wrong", xy=(100, 2), xytext=(79, 2.35),
                ha="center", va="bottom", fontsize=10.6,
                arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.75})
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.17, 3.0)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("Coverage of independent holdout (%)")
    ax.set_ylabel("Empirical risk among accepted (%)")
    save(fig, "results_risk_coverage.pdf")


def figure_runtime() -> None:
    HOLDOUT = load_json("experiments/formal_coseg_external_holdout/unified_evaluation.json")
    d = HOLDOUT["unified_runtime"]
    assert d["complete"] and len(d["rows"]) == 50
    methods = ["meshfix", "cgal_pool", "tcmr", "semigcn"]
    values = {}
    for method in methods:
        values[method] = np.array([
            r["methods"][method]["input_to_verified_output_wall_seconds"]
            for r in d["rows"]
        ], dtype=float)
        assert len(values[method]) == 50 and np.all(np.isfinite(values[method]))
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.175, top=0.94)
    rng = np.random.default_rng(20261024)
    for i, method in enumerate(methods):
        v = values[method]
        color = METHOD_COLOR[method]
        x = np.full(len(v), i, dtype=float) + rng.uniform(-0.15, 0.15, size=len(v))
        ax.scatter(x, v, s=11, c=color, edgecolors="none", zorder=1)
        p05, med, p95 = np.quantile(v, [0.05, 0.5, 0.95])
        ax.plot([i, i], [p05, p95], color=color, lw=2.3, zorder=3)
        ax.plot([i - 0.20, i + 0.20], [med, med], color=color, lw=3.0, zorder=4)
        ax.plot([i - 0.08, i + 0.08], [p05, p05], color=color, lw=1.4, zorder=3)
        ax.plot([i - 0.08, i + 0.08], [p95, p95], color=color, lw=1.4, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 10000)
    ax.set_yticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"])
    ax.set_xticks(range(len(methods)), [METHOD_LABEL[m] for m in methods])
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylabel("Input-to-verified-output time (s, log scale)")
    save(fig, "results_runtime.pdf")


def figure_optimizer_budget() -> None:
    """Finite budgets with a shared safe fallback pool and all-source coverage."""
    data = load_json("experiments/formal_collar_mechanism/budget_fallback_results.json")
    if not data.get("complete"):
        raise ValueError("The complete safe-policy replay is required")
    summary = data["summary"]
    common_n = summary["common_correct_safe_all_six_n"]
    steps = np.array([10, 30, 100])
    fig, axes = plt.subplots(1, 2, figsize=(7.14, 3.32))
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.22, top=0.78, wspace=0.44)
    paired = [summary["paired_by_budget"][str(step)]["all_six_common"]["seam_normal_absolute_mean"] for step in steps]
    means = np.array([x["mean_first_minus_second"] for x in paired])
    bounds = np.array([x["bootstrap_95ci_first_minus_second"] for x in paired])
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].errorbar(steps, means, yerr=np.stack((means-bounds[:,0],bounds[:,1]-means)),
                     color=COLOR["indigo"], marker="o", markersize=5.5, linewidth=1.6,
                     capsize=3.3, elinewidth=1, markeredgecolor="white", markeredgewidth=0.5)
    axes[0].set_xscale("log")
    axes[0].set_xlim(7,140)
    axes[0].set_xticks(steps,["10","30","100"])
    axes[0].set_ylim(-.001,.031)
    axes[0].set_yticks([0,.01,.02,.03],["0","0.01","0.02","0.03"])
    axes[0].set_ylabel("Fixed − joint mean seam (rad)")
    axes[0].text(.5,1.04,f"Same {common_n} accepted sources",transform=axes[0].transAxes,
                 ha="center",va="bottom",fontsize=10.8,color="black",clip_on=False)
    from matplotlib.patches import Patch
    handles=[]
    for i,(mode,label,color) in enumerate((
        ("patch_only","Fixed collar",COLOR["coral"]),
        ("joint","Joint collar",COLOR["indigo"]),
    )):
        entries=[summary["methods"][f"{mode}_{step}"] for step in steps]
        fallback=np.array([e["fallback_selected_n"] for e in entries])
        refined=np.array([e["accepted_n"] for e in entries])-fallback
        x=np.arange(3)+(-.18 if i==0 else .18)
        axes[1].bar(x,refined,width=.32,color=color,linewidth=0)
        axes[1].bar(x,fallback,bottom=refined,width=.32,facecolor="white",edgecolor=color,hatch="///",linewidth=.85)
        handles.append(Patch(facecolor=color,edgecolor="none",label=label))
    handles.append(Patch(facecolor="white",edgecolor="black",hatch="///",linewidth=.7,label="Fallback"))
    axes[1].set_xlim(-.55,2.55)
    axes[1].set_xticks(np.arange(3),["10","30","100"])
    axes[1].set_ylim(0,30)
    axes[1].set_yticks([0,10,20,30])
    axes[1].set_ylabel("Accepted outputs (of 30)")
    for ax,letter in zip(axes,"ab"):
        ax.minorticks_off()
        ax.set_xlabel("Adam steps")
        panel_letter(ax,letter,-.25)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,1.0),
               ncol=3,frameon=False,handlelength=1.5,columnspacing=1.3)
    save(fig,"results_optimizer_budget.pdf")


def _source_strip(ax: plt.Axes, values: np.ndarray, y: float, color: str,
                  x_limits: tuple[float, float]) -> None:
    """Deterministic vertical packing; horizontal coordinates are unmodified."""
    values = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(values))
    threshold = (x_limits[1] - x_limits[0]) * 0.025
    placed: list[tuple[float, float]] = []
    offsets = np.array([0, .045, -.045, .090, -.090, .135, -.135])
    for index in np.argsort(values, kind="stable"):
        x = values[index]
        scores = [sum(abs(x - px) < threshold and abs(offset - py) < .043
                      for px, py in placed) for offset in offsets]
        offset = offsets[int(np.argmin(scores))]
        placed.append((float(x), float(offset)))
    ax.scatter([x for x, _ in placed], [y + dy for _, dy in placed],
               s=18, c=color, edgecolors="white", linewidths=.35,
               alpha=.88, zorder=3, clip_on=False)


def _paired_interval(ax: plt.Axes, entry: dict, y: float, color: str,
                     scale: float = 1.0) -> None:
    mean = entry["mean_first_minus_second"] * scale
    lo, hi = np.asarray(entry["bootstrap_95ci_first_minus_second"]) * scale
    ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="D",
                color=color, markersize=5.5, elinewidth=2, capsize=3,
                capthick=1.25, markeredgecolor="white", markeredgewidth=.6,
                zorder=4)


def _paired_legend(fig: plt.Figure, top: float = .985) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="none", color="black",
               markersize=4, label="Individual source"),
        Line2D([], [], marker="D", color="black", linewidth=1.5,
               markersize=5, label="Mean and 95% CI"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, top),
               frameon=False, ncol=2, columnspacing=1.5, handlelength=1.5,
               handletextpad=.5, fontsize=11.5)


def figure_collar_policy() -> None:
    """Full-policy source effects, preserving target prediction and fallback."""
    data = load_json("experiments/formal_collar_mechanism/predicted_fallback_results.json")
    assert data["complete"] and data["predictions_complete_before_scoring"]
    with mpl.rc_context({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5}):
        fig, ax = plt.subplots(figsize=(7.14, 3.35))
        fig.subplots_adjust(left=.23, right=.98, bottom=.22, top=.77)
        limits = (-.28, .56)
        counts = []
        for dataset, y, color in (("holdout50", 1.0, COLOR["coral"]),
                                  ("cross_category60", 0.0, COLOR["indigo"])):
            rows = [r for r in data["rows"] if r["dataset"] == dataset
                    and all(r["scored"][m]["correct_safe"]
                            for m in ("fixed_collar", "movable_collar"))
                    and r["scored"]["fixed_collar"]["selected_loop_id"]
                    == r["scored"]["movable_collar"]["selected_loop_id"]]
            delta = np.array([
                r["scored"]["fixed_collar"]["geometry"]["seam_normal_absolute_mean"]
                - r["scored"]["movable_collar"]["geometry"]["seam_normal_absolute_mean"]
                for r in rows])
            q = data["summary"][dataset]["paired_same_target_fixed_minus_movable"]["seam_normal_absolute_mean"]
            assert len(delta) == q["n"] and len({r["source_model_id"] for r in rows}) == q["n"]
            assert np.isclose(delta.mean(), q["mean_first_minus_second"], atol=1e-12)
            assert delta.min() > limits[0] and delta.max() < limits[1]
            _source_strip(ax, delta, y + .1, color, limits)
            _paired_interval(ax, q, y - .2, color)
            count = [q["first_greater_n"], q["equal_n"], q["first_lower_n"]]
            counts.append(count)
            ax.text(limits[1], y + .38,
                    f"{count[0]} improved / {count[1]} tied / {count[2]} worse",
                    ha="right", va="center", fontsize=11.5, color="black")
        assert np.array(counts).sum(axis=0).tolist() == [93, 4, 3]
        ax.axvline(0, color="black", linewidth=.8, zorder=1)
        ax.set_xlim(limits)
        ax.set_ylim(-.42, 1.6)
        ax.set_xticks([-.2, 0, .2, .4], ["−0.2", "0", "0.2", "0.4"])
        ax.set_yticks([.94, -.06], ["Holdout\n(n = 49)", "Cross-category\n(n = 51)"])
        ax.tick_params(axis="y", length=0, pad=10)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Fixed − movable collar mean seam angle (rad)", labelpad=9)
        _paired_legend(fig)
        save(fig, "results_collar_policy.pdf")


def figure_fairing_tradeoff() -> None:
    """Saved paired metrics after shared fallback, including adverse outliers."""
    data = load_json("experiments/formal_collar_mechanism/fairing_fallback_results.json")
    assert data["complete"] and data["protocol"]["prescribed_target"]
    metrics = [
        ("seam_normal_absolute_mean", "Mean seam (rad)", 1., (-.46, 2.17),
         [-.4, 0, 1, 2], ["−0.4", "0", "1", "2"]),
        ("retained_displacement_rms_over_edge", "Retained RMS\n(10⁻³ input edges)", 1000., (-3.8, 4.55),
         [-3, 0, 3], ["−3", "0", "3"]),
        ("triangle_quality.p05", "5th-percentile\ntriangle quality", 1., (-.11, .5),
         [0, .2, .4], ["0", "0.2", "0.4"]),
    ]
    with mpl.rc_context({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5}):
        fig, axes = plt.subplots(1, 3, figsize=(7.14, 3.55))
        fig.subplots_adjust(left=.18, right=.98, bottom=.29, top=.77, wspace=.30)
        for ax, (key, xlabel, scale, limits, ticks, ticklabels), letter in zip(axes, metrics, "abc"):
            for dataset, y, color in (("holdout50", 1., COLOR["coral"]),
                                      ("cross_category60", 0., COLOR["indigo"])):
                rows = [r for r in data["rows"] if r["dataset"] == dataset
                        and all(r["policies"][m]["valid"]
                                for m in ("joint_collar", "bounded_fair"))]
                def value(r: dict, method: str) -> float:
                    v = r["policies"][method]["geometry"]
                    for part in key.split("."):
                        v = v[part]
                    return v
                delta = np.array([value(r, "joint_collar") - value(r, "bounded_fair")
                                  for r in rows])
                q = data["summary"][dataset]["paired"]["joint_collar_minus_bounded_fair"]["common_valid_metrics"][key]
                assert len(delta) == q["n"] and len({r["source_model_id"] for r in rows}) == q["n"]
                assert np.isclose(delta.mean(), q["mean_first_minus_second"], atol=1e-12)
                assert (delta * scale).min() > limits[0] and (delta * scale).max() < limits[1]
                _source_strip(ax, delta * scale, y + .1, color, limits)
                _paired_interval(ax, q, y - .2, color, scale)
            ax.axvline(0, color="black", linewidth=.8, zorder=1)
            ax.set_xlim(limits)
            ax.set_ylim(-.44, 1.5)
            ax.set_xticks(ticks, ticklabels)
            ax.set_xlabel(xlabel, labelpad=8)
            ax.set_yticks([.94, -.06])
            ax.tick_params(axis="y", length=0, pad=8)
            ax.spines["left"].set_visible(False)
            ax.text(0, 1.035, letter, transform=ax.transAxes, weight="bold",
                    fontsize=12, color="black")
        axes[0].set_yticklabels(["Holdout\n(n = 50)", "Cross-category\n(n = 57)"])
        for ax in axes[1:]:
            ax.set_yticklabels([])
        _paired_legend(fig)
        fig.text(.58, .045, "Joint refinement − bounded fairing", ha="center",
                 fontsize=12, color="black")
        save(fig, "results_fairing_tradeoff.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("all", "budget", "collar-policy", "fairing-tradeoff", "submission-additions"), default="all")
    args = parser.parse_args()
    setup_style()
    if args.only == "budget":
        figure_optimizer_budget()
        return
    if args.only in ("collar-policy", "submission-additions"):
        figure_collar_policy()
    if args.only in ("fairing-tradeoff", "submission-additions"):
        figure_fairing_tradeoff()
    if args.only != "all":
        return
    figure_correct_safe()
    figure_development_supplement()
    figure_geometry_tradeoff()
    figure_ablation()
    figure_scan_robustness()
    figure_risk_coverage()
    figure_runtime()
    for p in sorted(OUT.glob("results_*.pdf")):
        print(f"{p.name}\t{p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
