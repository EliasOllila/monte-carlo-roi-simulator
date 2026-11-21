"""Unit and integration tests for patent portfolio pruning heuristics."""

from pathlib import Path
import numpy as np
import pytest
import warnings

from src.pruning import (
    apply_abandonment_invariant,
    apply_strategy_a,
    apply_strategy_b,
    apply_strategy_c,
    compute_portfolio_metrics,
    evaluate_pruning_strategies,
)
from src.simulator import load_assumptions, load_portfolio, simulate_portfolio_npv


def test_abandonment_invariant():
    """Verify that abandoned patents (w_i = 0) produce exact 0.0 in all cash flows."""
    # 3 patents, 4 simulations
    matrix = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    )
    # Abandon patent 1 (index 1)
    keep_mask = np.array([True, False, True])
    pruned = apply_abandonment_invariant(matrix, keep_mask)

    assert np.all(pruned[1, :] == 0.0)
    assert np.array_equal(pruned[0, :], matrix[0, :])
    assert np.array_equal(pruned[2, :], matrix[2, :])


def test_strategy_a_cuts_at_least_target():
    """Verify Strategy A abandons highest cost patents first and cuts >= target cost."""
    costs = np.array([100.0, 500.0, 200.0, 50.0, 300.0])
    total_cost = np.sum(costs)  # 1150.0
    target_cut = 0.40 * total_cost  # 460.0

    # Highest costs: 500.0 (index 1), then 300.0 (index 4)
    # Abandoning index 1 gives 500.0 >= 460.0
    keep_mask = apply_strategy_a(costs, target_cut)

    assert not keep_mask[1]  # Most expensive abandoned
    assert keep_mask[0] and keep_mask[2] and keep_mask[3] and keep_mask[4]
    saved_cost = np.sum(costs[~keep_mask])
    assert saved_cost >= target_cut


def test_strategy_b_value_to_cost_ratio():
    """Verify Strategy B abandons lowest bang-for-buck (E[NPV] / E[Cost]) patents first."""
    costs = np.array([100.0, 100.0, 100.0])
    npvs = np.array([500.0, -50.0, 200.0])
    # Ratios: patent 0 = 5.0, patent 1 = -0.5, patent 2 = 2.0
    target_cut = 90.0  # Abandon 1 patent (100.0 >= 90.0)

    keep_mask = apply_strategy_b(npvs, costs, target_cut)

    # Lowest ratio is patent 1 (-0.5), it should be abandoned first
    assert not keep_mask[1]
    assert keep_mask[0]
    assert keep_mask[2]


def test_strategy_c_protects_seps():
    """Verify Strategy C protects all Standard-Essential Patents from being pruned."""
    costs = np.array([100.0, 200.0, 150.0, 300.0])
    npvs = np.array([-100.0, -200.0, 50.0, 100.0])
    # Patents 0 and 1 have worst ratios, but patent 1 is SEP!
    is_sep = np.array([False, True, False, False])
    target_cut = 250.0

    keep_mask = apply_strategy_c(npvs, costs, is_sep, target_cut)

    # SEP must be protected regardless of its bad ratio
    assert keep_mask[1], "SEP must be protected under Strategy C"

    # Saved cost among non-SEPs must be >= target_cut
    saved = np.sum(costs[~keep_mask])
    assert saved >= target_cut


def test_strategy_c_warning_when_insufficient_non_seps():
    """Verify Strategy C warns when non-SEPs cannot reach the target cut."""
    costs = np.array([50.0, 500.0])
    npvs = np.array([10.0, 1000.0])
    is_sep = np.array([False, True])  # non-SEP cost is only 50.0
    target_cut = 100.0

    with pytest.warns(UserWarning, match="insufficient to reach the target cut"):
        keep_mask = apply_strategy_c(npvs, costs, is_sep, target_cut)

    # Non-SEP is pruned, SEP is kept
    assert not keep_mask[0]
    assert keep_mask[1]


def test_evaluate_pruning_strategies_full_portfolio():
    """Verify evaluation of Baseline, Strategy A, Strategy B, Strategy C on portfolio."""
    assumptions = load_assumptions()
    portfolio = load_portfolio()
    # Fast evaluation with 250 simulations
    fast_assump = dict(assumptions)
    fast_assump["n_simulations"] = 250

    sim_res = simulate_portfolio_npv(portfolio, fast_assump, seed_override=42)
    results = evaluate_pruning_strategies(sim_res, portfolio, fast_assump)

    assert "Baseline (Keep All)" in results
    assert "Strategy A (Naive Cost-Cutting)" in results
    assert "Strategy B (Value-to-Cost Ratio)" in results
    assert "Strategy C (SEP-Protected Ratio)" in results

    base = results["Baseline (Keep All)"]
    strat_a = results["Strategy A (Naive Cost-Cutting)"]
    strat_b = results["Strategy B (Value-to-Cost Ratio)"]
    strat_c = results["Strategy C (SEP-Protected Ratio)"]

    assert base.patents_abandoned_count == 0
    assert base.cost_reduction_pct == 0.0

    # Pruned strategies must cut ~20% of costs (>= 0.20)
    assert strat_a.cost_reduction_pct >= 0.20
    assert strat_b.cost_reduction_pct >= 0.20
    assert strat_c.cost_reduction_pct >= 0.20

    # Strategy B (value-to-cost) should deliver higher mean and median NPV than Strategy A (naive cost)
    assert strat_b.mean_npv > strat_a.mean_npv
    assert strat_b.p50_npv > strat_a.p50_npv

    # Verify all expected metric fields are present and valid
    for name, res in results.items():
        assert np.isfinite(res.mean_npv)
        assert np.isfinite(res.p10_npv)
        assert np.isfinite(res.p50_npv)
        assert np.isfinite(res.p90_npv)
        assert 0.0 <= res.prob_net_loss <= 1.0
        assert res.expected_pv_cost >= 0.0
        assert 0 <= res.patents_abandoned_count <= len(portfolio)
        assert len(res.portfolio_npv_draws) == 250
