"""Sensitivity analysis module: One-At-a-Time (OAT) and rank correlations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np

from src.pruning import apply_strategy_b
from src.simulator import (
    load_assumptions,
    load_portfolio,
    simulate_portfolio_npv,
)

DEFAULT_KEY_PARAMETERS = [
    "discount_rate",
    "prob_monetized_sep",
    "prob_monetized_non_sep",
    "lognormal_mu",
    "sep_value_multiplier",
    "fee_inflation_mean",
    "annual_obsolescence_rate",
]


@dataclass
class OATResult:
    """Result of One-At-a-Time (OAT) parameter shock analysis.

    Attributes:
        parameter: Parameter name.
        baseline_value: Baseline parameter value.
        low_value: Lower shocked parameter value (-oat_shock_pct).
        high_value: Higher shocked parameter value (+oat_shock_pct).
        p50_npv_baseline: Median portfolio NPV at baseline (EUR).
        p50_npv_low: Median portfolio NPV at low shocked parameter (EUR).
        p50_npv_high: Median portfolio NPV at high shocked parameter (EUR).
        swing_eur: Absolute median NPV swing abs(p50_high - p50_low) in EUR.
        delta_low_eur: Change in median NPV from baseline at low shock (EUR).
        delta_high_eur: Change in median NPV from baseline at high shock (EUR).
    """

    parameter: str
    baseline_value: float
    low_value: float
    high_value: float
    p50_npv_baseline: float
    p50_npv_low: float
    p50_npv_high: float
    swing_eur: float
    delta_low_eur: float
    delta_high_eur: float

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> Sequence[str]:
        return (
            "parameter",
            "baseline_value",
            "low_value",
            "high_value",
            "p50_npv_baseline",
            "p50_npv_low",
            "p50_npv_high",
            "swing_eur",
            "delta_low_eur",
            "delta_high_eur",
        )


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Compute fractional average ranks in pure NumPy handling ties."""
    a_arr = np.asarray(a)
    n = len(a_arr)
    if n == 0 or np.all(a_arr == a_arr[0]):
        return np.full(n, (n + 1) / 2.0, dtype=float)

    sorter = np.argsort(a_arr, kind="stable")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(n)
    a_sorted = a_arr[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.bincount(dense)
    cumsum = np.cumsum(count)
    return (cumsum[dense] - (count[dense] - 1) / 2.0).astype(float)


def compute_spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation using pure NumPy argsort.

    No external dependencies required (e.g. scipy).

    Args:
        x: First 1D array.
        y: Second 1D array.

    Returns:
        Spearman rank correlation coefficient in [-1.0, 1.0].
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) != len(y_arr):
        raise ValueError("Arrays x and y must have identical length.")
    if len(x_arr) < 2:
        return 0.0

    rx = _rankdata(x_arr)
    ry = _rankdata(y_arr)

    rx_diff = rx - np.mean(rx)
    ry_diff = ry - np.mean(ry)

    denom = np.sqrt(np.sum(rx_diff**2) * np.sum(ry_diff**2))
    if denom == 0.0:
        return 0.0

    return float(np.sum(rx_diff * ry_diff) / denom)


def run_oat_sensitivity(
    portfolio_df_or_rows: Any,
    assumptions: dict[str, Any] | None = None,
    strategy_mask: np.ndarray | None = None,
    key_parameters: Sequence[str] | None = None,
    oat_shock_pct: float | None = None,
    n_simulations: int | None = None,
    seed_override: int | None = None,
) -> dict[str, OATResult]:
    """Perform One-At-a-Time (OAT) sensitivity analysis on Strategy B.

    Shifts each key parameter individually by -oat_shock_pct and +oat_shock_pct
    while holding all other parameters at baseline, recording the resulting
    median portfolio NPV (p50_npv) swing in EUR.

    Args:
        portfolio_df_or_rows: Portfolio data.
        assumptions: Model assumptions dictionary.
        strategy_mask: Boolean keep mask for Strategy B. If None, derived on baseline.
        key_parameters: List of parameter names to shock.
        oat_shock_pct: Shock percentage (e.g. 0.20 for +/- 20%).
        n_simulations: Optional simulation count override.
        seed_override: Seed for deterministic Common Random Numbers (CRN).

    Returns:
        Dictionary mapping parameter names to OATResult objects.
    """
    if assumptions is None:
        assumptions = load_assumptions()
    if key_parameters is None:
        key_parameters = DEFAULT_KEY_PARAMETERS
    if oat_shock_pct is None:
        oat_shock_pct = float(assumptions.get("oat_shock_pct", 0.20))

    base_assumptions = dict(assumptions)
    if n_simulations is not None:
        base_assumptions["n_simulations"] = int(n_simulations)

    seed = seed_override if seed_override is not None else int(base_assumptions.get("random_seed", 42))

    # Run baseline simulation
    baseline_sim = simulate_portfolio_npv(
        portfolio_df_or_rows, base_assumptions, seed_override=seed
    )

    # Derive Strategy B keep mask if not provided
    if strategy_mask is None:
        exp_cost = np.mean(baseline_sim.pv_cost_matrix, axis=1)
        exp_npv = np.mean(baseline_sim.npv_matrix, axis=1)
        target_cut_pct = float(base_assumptions.get("target_cost_cut_pct", 0.20))
        target_cut = target_cut_pct * float(np.sum(exp_cost))
        strategy_mask = apply_strategy_b(exp_npv, exp_cost, target_cut)

    w = np.asarray(strategy_mask, dtype=float)[:, None]
    base_portfolio_npvs = np.sum(baseline_sim.npv_matrix * w, axis=0)
    p50_baseline = float(np.percentile(base_portfolio_npvs, 50))

    results: dict[str, OATResult] = {}

    for param in key_parameters:
        if param not in base_assumptions:
            continue

        base_val = float(base_assumptions[param])
        low_val = base_val * (1.0 - oat_shock_pct)
        high_val = base_val * (1.0 + oat_shock_pct)

        # Ensure valid domain bounds for probability parameters
        if "prob_" in param:
            low_val = max(0.0, min(1.0, low_val))
            high_val = max(0.0, min(1.0, high_val))

        # 1. Low shock run (CRN preserves randomness)
        assump_low = dict(base_assumptions)
        assump_low[param] = low_val
        sim_low = simulate_portfolio_npv(portfolio_df_or_rows, assump_low, seed_override=seed)
        npv_low_draws = np.sum(sim_low.npv_matrix * w, axis=0)
        p50_low = float(np.percentile(npv_low_draws, 50))

        # 2. High shock run
        assump_high = dict(base_assumptions)
        assump_high[param] = high_val
        sim_high = simulate_portfolio_npv(portfolio_df_or_rows, assump_high, seed_override=seed)
        npv_high_draws = np.sum(sim_high.npv_matrix * w, axis=0)
        p50_high = float(np.percentile(npv_high_draws, 50))

        swing_eur = abs(p50_high - p50_low)

        results[param] = OATResult(
            parameter=param,
            baseline_value=base_val,
            low_value=low_val,
            high_value=high_val,
            p50_npv_baseline=p50_baseline,
            p50_npv_low=p50_low,
            p50_npv_high=p50_high,
            swing_eur=swing_eur,
            delta_low_eur=p50_low - p50_baseline,
            delta_high_eur=p50_high - p50_baseline,
        )

    return results


def compute_rank_correlations(
    portfolio_df_or_rows: Any,
    assumptions: dict[str, Any] | None = None,
    strategy_mask: np.ndarray | None = None,
    key_parameters: Sequence[str] | None = None,
    n_samples: int = 1000,
    shock_pct: float | None = None,
    n_sims_per_sample: int = 150,
    seed: int = 42,
    method: str = "lhs",
) -> dict[str, float]:
    """Compute Spearman rank correlations between key model parameters and median portfolio NPV.

    Perturbs parameters across Latin-Hypercube (or uniform) random draws around
    baseline (+/- shock_pct) and computes pure NumPy rank correlation with Strategy B median NPV.

    Args:
        portfolio_df_or_rows: Portfolio data.
        assumptions: Model assumptions dictionary.
        strategy_mask: Boolean keep mask for Strategy B. If None, derived on baseline.
        key_parameters: Sequence of parameter names to evaluate.
        n_samples: Number of sample evaluations (default: 1000).
        shock_pct: Perturbation range around baseline (default from oat_shock_pct or 0.20).
        n_sims_per_sample: Simulations per sample evaluation for fast CRN vectorization.
        seed: Random seed for sampling and simulations.
        method: Sampling method ('lhs' for Latin-Hypercube or 'uniform').

    Returns:
        Dictionary mapping each parameter name to its Spearman rank correlation.
    """
    if assumptions is None:
        assumptions = load_assumptions()
    if key_parameters is None:
        key_parameters = DEFAULT_KEY_PARAMETERS
    if shock_pct is None:
        shock_pct = float(assumptions.get("oat_shock_pct", 0.20))

    rng = np.random.default_rng(seed)
    n_params = len(key_parameters)

    # Derive Strategy B keep mask if needed
    if strategy_mask is None:
        base_sim = simulate_portfolio_npv(portfolio_df_or_rows, assumptions, seed_override=seed)
        exp_cost = np.mean(base_sim.pv_cost_matrix, axis=1)
        exp_npv = np.mean(base_sim.npv_matrix, axis=1)
        target_cut_pct = float(assumptions.get("target_cost_cut_pct", 0.20))
        target_cut = target_cut_pct * float(np.sum(exp_cost))
        strategy_mask = apply_strategy_b(exp_npv, exp_cost, target_cut)

    w = np.asarray(strategy_mask, dtype=float)[:, None]

    # Generate parameter sample matrix: shape (n_samples, n_params)
    param_samples = np.zeros((n_samples, n_params), dtype=float)

    for j, param in enumerate(key_parameters):
        base_val = float(assumptions[param])
        low_val = base_val * (1.0 - shock_pct)
        high_val = base_val * (1.0 + shock_pct)

        if method.lower() == "lhs":
            # Latin Hypercube Sampling: stratified intervals permuted randomly
            strata = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
            rng.shuffle(strata)
            vals = low_val + strata * (high_val - low_val)
        else:
            # Uniform random sampling
            vals = rng.uniform(low=low_val, high=high_val, size=n_samples)

        if "prob_" in param:
            vals = np.clip(vals, 0.0, 1.0)

        param_samples[:, j] = vals

    # Evaluate median portfolio NPV for each sampled parameter set
    median_npv_draws = np.zeros(n_samples, dtype=float)

    for m in range(n_samples):
        sample_assumptions = dict(assumptions)
        sample_assumptions["n_simulations"] = n_sims_per_sample
        for j, param in enumerate(key_parameters):
            sample_assumptions[param] = float(param_samples[m, j])

        # Run simulation with deterministic Common Random Numbers
        sim_res = simulate_portfolio_npv(
            portfolio_df_or_rows, sample_assumptions, seed_override=seed
        )
        port_npv = np.sum(sim_res.npv_matrix * w, axis=0)
        median_npv_draws[m] = float(np.percentile(port_npv, 50))

    # Compute Spearman rank correlation for each parameter
    correlations: dict[str, float] = {}
    for j, param in enumerate(key_parameters):
        corr = compute_spearman_correlation(param_samples[:, j], median_npv_draws)
        correlations[param] = corr

    return correlations
