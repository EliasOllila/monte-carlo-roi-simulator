"""Unit and integration tests for sensitivity analysis module."""

import numpy as np
import pytest

from src.sensitivity import (
    compute_rank_correlations,
    compute_spearman_correlation,
    run_oat_sensitivity,
)
from src.simulator import load_assumptions, load_portfolio


def test_compute_spearman_correlation_known_values():
    """Verify compute_spearman_correlation on analytical reference arrays."""
    # Perfectly monotonically increasing -> correlation = 1.0
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 25.0, 30.0, 45.0, 100.0])
    assert pytest.approx(compute_spearman_correlation(x, y), abs=1e-7) == 1.0

    # Perfectly monotonically decreasing -> correlation = -1.0
    y_rev = np.array([100.0, 45.0, 30.0, 25.0, 10.0])
    assert pytest.approx(compute_spearman_correlation(x, y_rev), abs=1e-7) == -1.0

    # Constant array -> correlation = 0.0
    y_const = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    assert compute_spearman_correlation(x, y_const) == 0.0


def test_run_oat_sensitivity():
    """Verify One-At-a-Time (OAT) parameter sensitivity calculations."""
    assumptions = load_assumptions()
    portfolio = load_portfolio()[:50]  # Subset for fast testing

    oat_results = run_oat_sensitivity(
        portfolio,
        assumptions,
        n_simulations=100,
        oat_shock_pct=0.20,
        seed_override=42,
    )

    expected_params = {
        "discount_rate",
        "prob_monetized_sep",
        "prob_monetized_non_sep",
        "lognormal_mu",
        "sep_value_multiplier",
        "fee_inflation_mean",
        "annual_obsolescence_rate",
    }
    assert expected_params.issubset(oat_results.keys())

    for param, res in oat_results.items():
        assert res.parameter == param
        assert res.low_value < res.high_value or (res.low_value == res.high_value == 0.0)
        assert res.swing_eur >= 0.0
        assert np.isfinite(res.p50_npv_baseline)
        assert np.isfinite(res.p50_npv_low)
        assert np.isfinite(res.p50_npv_high)

    # lognormal_mu has an exponential effect and should have the largest swing
    swings = {p: res.swing_eur for p, res in oat_results.items()}
    assert max(swings, key=swings.get) == "lognormal_mu"


def test_compute_rank_correlations_lhs():
    """Verify Latin-Hypercube sampling rank correlations across 1,000 samples."""
    assumptions = load_assumptions()
    portfolio = load_portfolio()[:40]

    # Run 1000 LHS samples with fast CRN simulation
    correlations = compute_rank_correlations(
        portfolio,
        assumptions,
        n_samples=1000,
        shock_pct=0.20,
        n_sims_per_sample=50,
        seed=42,
        method="lhs",
    )

    assert len(correlations) == 7
    for param, corr in correlations.items():
        assert -1.0 <= corr <= 1.0

    # Lognormal mu strongly dominates portfolio licensing revenue
    assert correlations["lognormal_mu"] > 0.80
