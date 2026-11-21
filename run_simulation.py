"""Main simulation orchestration entrypoint for Monte Carlo IP Pruning & ROI Analysis."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src.generate_portfolio import generate_portfolio
from src.pruning import evaluate_pruning_strategies
from src.report import (
    generate_decision_memo,
    plot_npv_distributions,
    plot_tornado_chart,
)
from src.sensitivity import compute_rank_correlations, run_oat_sensitivity
from src.simulator import (
    load_assumptions,
    load_portfolio,
    simulate_portfolio_npv,
)


def run_full_analysis() -> int:
    """Run full Monte Carlo portfolio simulation, strategy evaluation, and reporting."""
    print("=" * 80)
    print(" MONTE CARLO PATENT PORTFOLIO PRUNING & ROI SIMULATOR")
    print("=" * 80)

    # 1. Load configuration and portfolio
    config_path = Path("config/assumptions.yaml")
    data_path = Path("data/portfolio_snapshot.csv")

    if not config_path.exists():
        print(f"Error: {config_path} not found.", file=sys.stderr)
        return 1

    assumptions = load_assumptions(config_path)

    if not data_path.exists():
        print(f"Portfolio snapshot missing. Generating synthetic portfolio at {data_path}...")
        generate_portfolio(data_path, n_families=500, seed=assumptions.get("random_seed", 42))

    portfolio = load_portfolio(data_path)
    n_patents = len(portfolio)
    n_sims = assumptions["n_simulations"]
    horizon = assumptions["horizon_years"]
    target_cut_pct = assumptions["target_cost_cut_pct"]

    print(f"Loaded {n_patents} patent families.")
    print(f"Parameters: {n_sims:,} simulations | {horizon} years | Target cost cut: {target_cut_pct * 100:.0f}%")
    print("-" * 80)

    # 2. Run Monte Carlo simulation
    print("Running vectorized Monte Carlo simulation with Common Random Numbers (CRN)...")
    t0 = time.time()
    sim_result = simulate_portfolio_npv(portfolio, assumptions)
    t_sim = time.time() - t0
    print(f"Simulation completed in {t_sim:.2f} seconds.")

    # 3. Evaluate Pruning Strategies
    print("Evaluating Baseline and Pruning Strategies (A: Naive, B: Ratio, C: SEP-Protected)...")
    strategy_results = evaluate_pruning_strategies(sim_result, portfolio, assumptions)

    # Print summary table
    print("\n" + "=" * 80)
    print(" PRUNING STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    header = f"{'Strategy':<33} | {'Kept':<4} | {'Aband':<5} | {'PV Cost (EUR)':<13} | {'Cut %':<6} | {'Median NPV (EUR)':<16} | {'P10 NPV (EUR)':<14} | {'Loss Prob':<9}"
    print(header)
    print("-" * len(header))

    for name, res in strategy_results.items():
        kept = n_patents - res.patents_abandoned_count
        print(
            f"{name:<33} | {kept:>4} | {res.patents_abandoned_count:>5} | "
            f"€{res.expected_pv_cost:>11,.0f} | {res.cost_reduction_pct * 100:>5.1f}% | "
            f"€{res.p50_npv:>14,.0f} | €{res.p10_npv:>12,.0f} | "
            f"{res.prob_net_loss * 100:>8.2f}%"
        )
    print("=" * 80 + "\n")

    # 4. Sensitivity Diagnostics on Strategy B
    strat_b_mask = strategy_results["Strategy B (Value-to-Cost Ratio)"].keep_mask
    print("Running One-At-a-Time (OAT) parameter shock analysis (±20%)...")
    oat_results = run_oat_sensitivity(
        portfolio,
        assumptions,
        strategy_mask=strat_b_mask,
    )

    print("Computing Spearman rank correlations across 1,000 Latin-Hypercube draws...")
    correlations = compute_rank_correlations(
        portfolio,
        assumptions,
        strategy_mask=strat_b_mask,
        n_samples=1000,
        n_sims_per_sample=150,
        seed=assumptions.get("random_seed", 42),
    )

    # 5. Generate Visualizations and Executive Decision Memo
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Rendering high-resolution distribution chart: output/npv_distribution.png...")
    plot_npv_distributions(strategy_results, output_path=output_dir / "npv_distribution.png")

    print("Rendering Tornado sensitivity chart: output/tornado_sensitivity.png...")
    plot_tornado_chart(oat_results, output_path=output_dir / "tornado_sensitivity.png")

    print("Compiling Executive Decision Memo: output/pruning_decision_memo.md...")
    memo_path = generate_decision_memo(
        strategy_results,
        oat_results,
        correlations,
        portfolio,
        sim_result,
        output_path=output_dir / "pruning_decision_memo.md",
    )

    print(f"\nAnalysis complete! Deliverables generated in {output_dir.resolve()}:")
    print("  - output/npv_distribution.png")
    print("  - output/tornado_sensitivity.png")
    print(f"  - {memo_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_full_analysis())
