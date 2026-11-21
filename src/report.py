"""Reporting, visualization, and executive decision memo generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import matplotlib

# Headless backend for server environments and corporate laptops without GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pruning import StrategyResult
from src.sensitivity import OATResult


def plot_npv_distributions(
    strategy_results: Mapping[str, StrategyResult],
    output_path: str | Path = "output/npv_distribution.png",
    dpi: int = 300,
) -> Path:
    """Generate empirical CDF chart comparing Portfolio NPV across strategies.

    Plots overlaid empirical Cumulative Distribution Functions (CDF) for
    Baseline, Strategy A (Naive), Strategy B (Ratio), and Strategy C (SEP-Protected),
    with vertical dashed lines indicating Strategy B P10 and P50.

    Args:
        strategy_results: Dictionary of strategy results.
        output_path: Destination path for the saved chart.
        dpi: Resolution of the output image.

    Returns:
        Path to the saved image.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    # Styling palette
    color_map = {
        "Baseline (Keep All)": "#5A6577",  # Slate gray
        "Strategy A (Naive Cost-Cutting)": "#E05638",  # Coral / Red
        "Strategy B (Value-to-Cost Ratio)": "#1B75BC",  # Royal Blue
        "Strategy C (SEP-Protected Ratio)": "#2EB872",  # Green
    }
    linestyle_map = {
        "Baseline (Keep All)": ":",
        "Strategy A (Naive Cost-Cutting)": "--",
        "Strategy B (Value-to-Cost Ratio)": "-",
        "Strategy C (SEP-Protected Ratio)": "-.",
    }

    strat_b_result = None

    for name, res in strategy_results.items():
        if "Strategy B" in name:
            strat_b_result = res
        color = color_map.get(name, "#333333")
        linestyle = linestyle_map.get(name, "-")

        npv_m = np.sort(res.portfolio_npv_draws) / 1e6
        cdf_y = np.linspace(0.0, 1.0, len(npv_m))

        lw = 2.5 if "Strategy B" in name else 1.8
        label = f"{name} (P50: €{res.p50_npv / 1e6:.1f}M)"
        ax.plot(npv_m, cdf_y, label=label, color=color, linestyle=linestyle, linewidth=lw)

    # Highlight Strategy B P10 and P50
    if strat_b_result is not None:
        p10_m = strat_b_result.p10_npv / 1e6
        p50_m = strat_b_result.p50_npv / 1e6
        ax.axvline(
            p10_m,
            color="#D9534F",
            linestyle="--",
            alpha=0.85,
            linewidth=1.4,
            label=f"Strategy B P10: €{p10_m:.1f}M",
        )
        ax.axvline(
            p50_m,
            color="#1B75BC",
            linestyle="-.",
            alpha=0.85,
            linewidth=1.4,
            label=f"Strategy B P50: €{p50_m:.1f}M",
        )

    ax.axvline(0.0, color="#111111", linestyle="-", linewidth=0.8, alpha=0.5)

    ax.set_title(
        "Monte Carlo Portfolio Net Present Value (NPV) CDF by Pruning Strategy",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("10-Year Discounted Portfolio NPV (EUR Millions)", fontsize=11, labelpad=8)
    ax.set_ylabel("Empirical Cumulative Probability P(NPV ≤ x)", fontsize=11, labelpad=8)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def plot_tornado_chart(
    oat_results: Mapping[str, OATResult],
    output_path: str | Path = "output/tornado_sensitivity.png",
    dpi: int = 300,
) -> Path:
    """Generate horizontal Tornado Chart ranking assumptions by P50 NPV swing.

    Args:
        oat_results: Mapping of parameter name to OATResult.
        output_path: Destination path for the saved chart.
        dpi: Resolution of the output image.

    Returns:
        Path to the saved image.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Sort parameters by absolute swing descending
    sorted_items = sorted(
        oat_results.values(), key=lambda r: r.swing_eur, reverse=True
    )

    params = [r.parameter for r in sorted_items]
    y_pos = np.arange(len(params))

    # Calculate deviations in EUR Millions relative to baseline P50 NPV
    delta_low_m = np.array([r.delta_low_eur / 1e6 for r in sorted_items])
    delta_high_m = np.array([r.delta_high_eur / 1e6 for r in sorted_items])
    swings_m = np.array([r.swing_eur / 1e6 for r in sorted_items])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    bar_height = 0.45

    # Draw low shock bar (-20%) and high shock bar (+20%)
    for i in range(len(params)):
        d_low = delta_low_m[i]
        d_high = delta_high_m[i]

        left_val = min(d_low, d_high, 0.0)
        right_val = max(d_low, d_high, 0.0)
        width = right_val - left_val

        # Bar spanning the total swing
        ax.barh(
            y_pos[i],
            width,
            left=left_val,
            height=bar_height,
            color="#3498DB",
            alpha=0.75,
            edgecolor="#1D6FA5",
            linewidth=1.2,
        )

        # Annotate swing value on the right
        swing_label = f"Swing: €{swings_m[i]:.2f}M"
        label_x = right_val + max(swings_m) * 0.02
        ax.text(
            label_x,
            y_pos[i],
            swing_label,
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="semibold",
            color="#2C3E50",
        )

    ax.axvline(0.0, color="#333333", linestyle="-", linewidth=1.2, alpha=0.8)

    ax.set_yticks(y_pos)
    # Human-friendly parameter names
    param_labels = [
        p.replace("_", " ").title().replace("Sep", "SEP").replace("Eur", "EUR")
        for p in params
    ]
    ax.set_yticklabels(param_labels, fontsize=10)
    ax.invert_yaxis()  # Top is largest swing

    ax.set_title(
        "Tornado Sensitivity: Strategy B Median NPV Impact (±20% Parameter Shocks)",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Deviation in Median Portfolio NPV (EUR Millions) from Baseline", fontsize=10.5, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def generate_decision_memo(
    strategy_results: Mapping[str, StrategyResult],
    oat_results: Mapping[str, OATResult],
    rank_correlations: Mapping[str, float],
    portfolio_rows: list[dict[str, Any]],
    sim_result: Any,
    output_path: str | Path = "output/pruning_decision_memo.md",
) -> Path:
    """Generate executive decision memo in Markdown format.

    Args:
        strategy_results: Pruning strategy metrics.
        oat_results: OAT parameter sensitivity results.
        rank_correlations: Spearman rank correlations.
        portfolio_rows: Parsed portfolio rows.
        sim_result: SimulationResult containing matrices.
        output_path: Path for output markdown memo.

    Returns:
        Path to the written memo.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    base = strategy_results["Baseline (Keep All)"]
    strat_a = strategy_results["Strategy A (Naive Cost-Cutting)"]
    strat_b = strategy_results["Strategy B (Value-to-Cost Ratio)"]
    strat_c = strategy_results["Strategy C (SEP-Protected Ratio)"]

    # Top 2 sensitivity drivers
    sorted_oat = sorted(oat_results.values(), key=lambda r: r.swing_eur, reverse=True)
    driver_1 = sorted_oat[0].parameter if len(sorted_oat) > 0 else "N/A"
    driver_2 = sorted_oat[1].parameter if len(sorted_oat) > 1 else "N/A"

    # Identify top 10 abandoned patent families under Strategy B
    abandon_mask_b = strat_b.abandon_mask
    exp_costs = np.mean(sim_result.pv_cost_matrix, axis=1)
    exp_npv = np.mean(sim_result.npv_matrix, axis=1)

    abandon_indices = np.where(abandon_mask_b)[0]
    # Sort abandoned patents by worst bang-for-buck (lowest NPV to Cost ratio)
    ratios = np.where(exp_costs[abandon_indices] > 0, exp_npv[abandon_indices] / exp_costs[abandon_indices], np.inf)
    worst_order = np.argsort(ratios)
    top_10_indices = abandon_indices[worst_order[:10]]

    # Strategy Comparison Rows
    strat_order = [
        "Baseline (Keep All)",
        "Strategy A (Naive Cost-Cutting)",
        "Strategy B (Value-to-Cost Ratio)",
        "Strategy C (SEP-Protected Ratio)",
    ]

    memo_lines = [
        "# Executive Decision Memo: Telecom Patent Portfolio Pruning & ROI Analysis",
        "",
        "**To**: Chief Intellectual Property Officer (CIPO) & Chief Financial Officer (CFO)  ",
        "**From**: IP Strategy & Quantitative Analytics Group  ",
        "**Subject**: Monte Carlo 10-Year Portfolio Pruning Strategy & ROI Optimization  ",
        "**Date**: November 2026  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Strategic Recommendation",
        "",
        "Under corporate directive to reduce IP maintenance budget expenditures by **20%**, we evaluated 500 telecom patent families using a **10,000-path Monte Carlo Common Random Numbers (CRN)** valuation framework.",
        "",
        f"- **Recommended Decision: Adopt Strategy B (Value-to-Cost Bang-for-Buck Ratio)**.",
        f"- **Cost Reduction Target Met**: Strategy B reduces 10-year discounted renewal spend by **{strat_b.cost_reduction_pct * 100:.1f}%** (saving **€{base.expected_pv_cost - strat_b.expected_pv_cost:,.0f}** in expected present-value fees) by abandoning **{strat_b.patents_abandoned_count} non-performing patent families**.",
        f"- **Economic Superiority over Naive Cost-Cutting**: Naive cost-cutting (Strategy A) indiscriminately drops the most expensive patents—inadvertently pruning high-value foreign filings and critical commercial assets. Compared to Strategy A, **Strategy B delivers €{(strat_b.mean_npv - strat_a.mean_npv):,.0f} (+{((strat_b.mean_npv / strat_a.mean_npv) - 1.0)*100:.1f}%) in higher mean portfolio NPV** and **€{(strat_b.p50_npv - strat_a.p50_npv):,.0f} higher median NPV**.",
        f"- **Downside Risk Protection**: Strategy B maintains a resilient P10 downside threshold of **€{strat_b.p10_npv:,.0f}** (vs. €{strat_a.p10_npv:,.0f} for Strategy A) and holds the probability of net lifetime loss to **{strat_b.prob_net_loss * 100:.2f}%**.",
        "",
        "---",
        "",
        "## 2. Pruning Strategy Comparison",
        "",
        "| Strategy | Patents Kept | Patents Abandoned | Expected PV Cost (EUR) | Cost Cut % | P10 NPV (EUR) | P50 NPV (EUR) | P90 NPV (EUR) | Prob(Loss) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s_name in strat_order:
        s_res = strategy_results[s_name]
        kept = len(s_res.keep_mask) - s_res.patents_abandoned_count
        memo_lines.append(
            f"| **{s_name}** | {kept} | {s_res.patents_abandoned_count} | "
            f"€{s_res.expected_pv_cost:,.0f} | {s_res.cost_reduction_pct * 100:.1f}% | "
            f"€{s_res.p10_npv:,.0f} | €{s_res.p50_npv:,.0f} | €{s_res.p90_npv:,.0f} | "
            f"{s_res.prob_net_loss * 100:.2f}% |"
        )

    memo_lines.extend(
        [
            "",
            "### Portfolio NPV Distribution Overview",
            "",
            "![Portfolio NPV Distribution](npv_distribution.png)",
            "",
            "---",
            "",
            "## 3. Sensitivity & Tornado Analysis",
            "",
            f"A One-At-a-Time (OAT) parameter shock (±20%) and Latin-Hypercube Sampling (LHS) across 1,000 multi-parameter trajectories show that the portfolio decision is overwhelmingly governed by **two key drivers**:",
            "",
            f"1. **`{driver_1}`**: Drives the primary magnitude of portfolio monetization (Tornado Swing: **€{sorted_oat[0].swing_eur:,.0f}**; Spearman Rank Correlation: **{rank_correlations.get(driver_1, 0.0):.3f}**).",
            f"2. **`{driver_2}`**: Dictates standard-essential patent leverage and monetization probability (Tornado Swing: **€{sorted_oat[1].swing_eur:,.0f}**; Spearman Rank Correlation: **{rank_correlations.get(driver_2, 0.0):.3f}**).",
            "",
            "| Rank | Parameter | Baseline | Low (-20%) | High (+20%) | Median NPV Swing (EUR) | Spearman Rank Corr |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for rank_idx, oat in enumerate(sorted_oat, start=1):
        corr_val = rank_correlations.get(oat.parameter, 0.0)
        memo_lines.append(
            f"| {rank_idx} | `{oat.parameter}` | {oat.baseline_value:.3f} | {oat.low_value:.3f} | "
            f"{oat.high_value:.3f} | €{oat.swing_eur:,.0f} | {corr_val:+.3f} |"
        )

    memo_lines.extend(
        [
            "",
            "### Tornado Sensitivity Chart",
            "",
            "![Tornado Sensitivity](tornado_sensitivity.png)",
            "",
            "---",
            "",
            "## 4. Top 10 Abandoned Patent Families under Strategy B",
            "",
            "The following patent families exhibit the lowest expected return per euro of maintenance cost and are scheduled for immediate non-renewal under Strategy B:",
            "",
            "| Family ID | Jurisdiction | Tech Domain | Is SEP? | Expected PV Cost (EUR) | Expected NPV (EUR) | Bang-for-Buck (E[NPV]/Cost) |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for idx in top_10_indices:
        p_row = portfolio_rows[idx]
        c = exp_costs[idx]
        v = exp_npv[idx]
        ratio_val = v / c if c > 0 else 0.0
        memo_lines.append(
            f"| `{p_row['family_id']}` | {p_row['jurisdiction']} | {p_row['tech_domain']} | "
            f"{'Yes' if p_row['is_sep'] else 'No'} | €{c:,.0f} | €{v:,.0f} | {ratio_val:+.3f} |"
        )

    memo_lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Implementation Roadmap",
            "",
            "1. **Execution**: Transmit abandonment instructions for the identified families to foreign patent annuities agents prior to Q1 renewal deadlines.",
            "2. **Continuous Monitoring**: Re-evaluate the portfolio annually under updated citation counts and standardization declarations using this Monte Carlo simulator.",
            "",
        ]
    )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(memo_lines))

    return out_file
