"""Comprehensive tests for the Monte Carlo patent portfolio simulator."""

from pathlib import Path
import csv
import numpy as np
import pytest
import yaml

from src.pruning import (
    apply_abandonment_invariant,
    apply_strategy_a,
    apply_strategy_b,
    apply_strategy_c,
    evaluate_pruning_strategies,
)
from src.simulator import load_assumptions, load_portfolio, simulate_portfolio_npv


def test_assumptions_config_loads():
    """Verify assumptions.yaml exists and contains all required parameters."""
    config_path = Path("config/assumptions.yaml")
    assert config_path.exists(), "config/assumptions.yaml should exist"

    config = load_assumptions(config_path)

    expected_keys = {
        "n_simulations",
        "horizon_years",
        "random_seed",
        "discount_rate",
        "fee_inflation_mean",
        "fee_inflation_std",
        "target_cost_cut_pct",
        "prob_monetized_non_sep",
        "prob_monetized_sep",
        "lognormal_mu",
        "lognormal_sigma",
        "sep_value_multiplier",
        "annual_obsolescence_rate",
        "oat_shock_pct",
    }
    assert expected_keys.issubset(config.keys())
    assert config["n_simulations"] == 10000
    assert config["horizon_years"] == 10
    assert config["random_seed"] == 42


def test_portfolio_snapshot_format():
    """Verify generated portfolio snapshot matches schema requirements."""
    data_path = Path("data/portfolio_snapshot.csv")
    assert data_path.exists(), "data/portfolio_snapshot.csv should exist"

    rows = load_portfolio(data_path)
    assert len(rows) == 500

    first = rows[0]
    last = rows[-1]
    assert first["family_id"] == "FAM-0001"
    assert last["family_id"] == "FAM-0500"

    for row in rows:
        assert row["jurisdiction"] in {"US", "EP", "DE", "GB", "JP", "CN"}
        age = int(row["patent_age"])
        assert 1 <= age <= 15
        assert isinstance(row["is_sep"], bool)
        cites = int(row["forward_citations"])
        assert 0 <= cites <= 45
        assert row["tech_domain"] in {
            "5G_RAN",
            "CORE_NETWORK",
            "VIDEO_CODEC",
            "IOT_CONNECTIVITY",
            "LEGACY_3G_4G",
        }
        cost = float(row["annual_maintenance_eur"])
        assert 800.0 <= cost <= 6500.0


def test_crn_determinism():
    """Verify Common Random Numbers (CRN) determinism: identical seed produces bit-identical npv_matrix."""
    assumptions = load_assumptions()
    fast_assump = dict(assumptions)
    fast_assump["n_simulations"] = 200

    portfolio = load_portfolio()[:25]

    run1 = simulate_portfolio_npv(portfolio, fast_assump, seed_override=42)
    run2 = simulate_portfolio_npv(portfolio, fast_assump, seed_override=42)
    run3 = simulate_portfolio_npv(portfolio, fast_assump, seed_override=43)

    # Identical seed -> bit-identical matrix
    np.testing.assert_array_equal(run1.npv_matrix, run2.npv_matrix)
    np.testing.assert_array_equal(run1.pv_cost_matrix, run2.pv_cost_matrix)
    np.testing.assert_array_equal(run1.pv_rev_matrix, run2.pv_rev_matrix)

    # Different seed -> distinct random trajectories
    assert not np.array_equal(run1.npv_matrix, run3.npv_matrix)


def test_abandonment_zero_out_invariant():
    """Verify Abandonment zero-out invariant: w_i = 0 sets pv_cost and pv_rev to exact 0.0."""
    assumptions = load_assumptions()
    fast_assump = dict(assumptions)
    fast_assump["n_simulations"] = 50

    portfolio = load_portfolio()[:10]
    sim = simulate_portfolio_npv(portfolio, fast_assump, seed_override=42)

    # Abandon patents at indices 1, 3, 7
    keep_mask = np.ones(10, dtype=bool)
    abandoned_indices = [1, 3, 7]
    keep_mask[abandoned_indices] = False

    cost_pruned = apply_abandonment_invariant(sim.pv_cost_matrix, keep_mask)
    rev_pruned = apply_abandonment_invariant(sim.pv_rev_matrix, keep_mask)
    npv_pruned = apply_abandonment_invariant(sim.npv_matrix, keep_mask)

    # Abandoned patents must contribute exactly 0.0
    for idx in abandoned_indices:
        assert np.all(cost_pruned[idx, :] == 0.0)
        assert np.all(rev_pruned[idx, :] == 0.0)
        assert np.all(npv_pruned[idx, :] == 0.0)

    # Kept patents must remain completely unperturbed
    for idx in range(10):
        if idx not in abandoned_indices:
            np.testing.assert_array_equal(cost_pruned[idx, :], sim.pv_cost_matrix[idx, :])
            np.testing.assert_array_equal(rev_pruned[idx, :], sim.pv_rev_matrix[idx, :])
            np.testing.assert_array_equal(npv_pruned[idx, :], sim.npv_matrix[idx, :])


def test_expiration_invariant():
    """Verify that patents with effective_age > 20 incur zero cost and zero revenue."""
    assumptions = load_assumptions()
    fast_assump = dict(assumptions)
    fast_assump["n_simulations"] = 50
    fast_assump["horizon_years"] = 10

    test_patents = [
        {
            "family_id": "FAM-EXP-1",
            "jurisdiction": "EP",
            "patent_age": 16,
            "is_sep": True,
            "forward_citations": 30,
            "tech_domain": "5G_RAN",
            "annual_maintenance_eur": 2500.0,
        },
        {
            "family_id": "FAM-EXP-2",
            "jurisdiction": "US",
            "patent_age": 20,
            "is_sep": True,
            "forward_citations": 25,
            "tech_domain": "CORE_NETWORK",
            "annual_maintenance_eur": 3000.0,
        },
    ]

    res = simulate_portfolio_npv(test_patents, fast_assump, seed_override=42)

    # Patent 0: age 16 (expires at year 5..10 when effective_age > 20)
    assert np.all(res.cash_flows_c[0, 4:, :] == 0.0)
    assert np.all(res.cash_flows_r[0, 4:, :] == 0.0)
    assert np.any(res.cash_flows_c[0, :4, :] > 0.0)

    # Patent 1: age 20 (expired across all 10 projection years)
    assert np.all(res.cash_flows_c[1, :, :] == 0.0)
    assert np.all(res.cash_flows_r[1, :, :] == 0.0)
    assert np.all(res.pv_cost_matrix[1, :] == 0.0)
    assert np.all(res.pv_rev_matrix[1, :] == 0.0)
    assert np.all(res.npv_matrix[1, :] == 0.0)


def test_budget_constraint_compliance():
    """Verify Strategies A and B reduce expected PV maintenance cost by at least target_cost_cut_pct."""
    assumptions = load_assumptions()
    portfolio = load_portfolio()
    fast_assump = dict(assumptions)
    fast_assump["n_simulations"] = 200

    sim_res = simulate_portfolio_npv(portfolio, fast_assump, seed_override=42)
    results = evaluate_pruning_strategies(sim_res, portfolio, fast_assump)

    strat_a = results["Strategy A (Naive Cost-Cutting)"]
    strat_b = results["Strategy B (Value-to-Cost Ratio)"]

    target_pct = assumptions["target_cost_cut_pct"]  # 0.20
    assert strat_a.cost_reduction_pct >= target_pct
    assert strat_b.cost_reduction_pct >= target_pct


def test_economic_dominance():
    """Verify Strategy B achieves higher expected NPV than Strategy A at same cost reduction target."""
    assumptions = load_assumptions()
    portfolio = load_portfolio()
    fast_assump = dict(assumptions)
    fast_assump["n_simulations"] = 300

    sim_res = simulate_portfolio_npv(portfolio, fast_assump, seed_override=42)
    results = evaluate_pruning_strategies(sim_res, portfolio, fast_assump)

    strat_a = results["Strategy A (Naive Cost-Cutting)"]
    strat_b = results["Strategy B (Value-to-Cost Ratio)"]

    # Strategy B ranks by value-to-cost ratio and preserves high-value patents,
    # strictly dominating Naive Cost-Cutting in expected NPV and median NPV.
    assert strat_b.mean_npv > strat_a.mean_npv
    assert strat_b.p50_npv > strat_a.p50_npv
    assert strat_b.p10_npv >= strat_a.p10_npv
