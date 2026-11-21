"""Patent portfolio pruning heuristic strategies and distribution metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import logging
import warnings
import numpy as np

from src.simulator import (
    SimulationResult,
    _parse_portfolio_data,
    load_assumptions,
    load_portfolio,
    simulate_portfolio_npv,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Evaluation metrics for a pruning strategy across Monte Carlo simulations.

    Attributes:
        strategy_name: Name of the pruning strategy.
        keep_mask: Boolean array of shape (N,) indicating kept patents (w_i = 1).
        abandon_mask: Boolean array of shape (N,) indicating abandoned patents (w_i = 0).
        mean_npv: Mean portfolio NPV across all simulations (EUR).
        p10_npv: 10th percentile portfolio NPV (EUR).
        p50_npv: 50th percentile (median) portfolio NPV (EUR).
        p90_npv: 90th percentile portfolio NPV (EUR).
        prob_net_loss: Probability of portfolio Net Present Value < 0 (P(Portfolio NPV < 0)).
        expected_pv_cost: Expected discounted maintenance cost of the portfolio (EUR).
        cost_reduction_pct: Percentage reduction in expected maintenance cost relative to baseline.
        patents_abandoned_count: Number of abandoned patents.
        portfolio_npv_draws: Vector of portfolio NPV draws across simulations (shape: (S,)).
        portfolio_cost_draws: Vector of portfolio cost draws across simulations (shape: (S,)).
        portfolio_rev_draws: Vector of portfolio revenue draws across simulations (shape: (S,)).
    """

    strategy_name: str
    keep_mask: np.ndarray
    abandon_mask: np.ndarray
    mean_npv: float
    p10_npv: float
    p50_npv: float
    p90_npv: float
    prob_net_loss: float
    expected_pv_cost: float
    cost_reduction_pct: float
    patents_abandoned_count: int
    portfolio_npv_draws: np.ndarray
    portfolio_cost_draws: np.ndarray
    portfolio_rev_draws: np.ndarray

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> Sequence[str]:
        return (
            "strategy_name",
            "keep_mask",
            "abandon_mask",
            "mean_npv",
            "p10_npv",
            "p50_npv",
            "p90_npv",
            "prob_net_loss",
            "expected_pv_cost",
            "cost_reduction_pct",
            "patents_abandoned_count",
            "portfolio_npv_draws",
            "portfolio_cost_draws",
            "portfolio_rev_draws",
        )


def apply_abandonment_invariant(
    matrix: np.ndarray, keep_mask: np.ndarray
) -> np.ndarray:
    """Enforce the Abandonment Invariant.

    When a patent has binary mask w_i = 0, its entire row in npv_matrix,
    pv_cost_matrix, or pv_rev_matrix is multiplied by 0.0 so abandoned
    patents generate zero future cost and zero future revenue.

    Args:
        matrix: 2D array of shape (N_patents, n_simulations).
        keep_mask: Boolean or binary array of shape (N_patents,).

    Returns:
        Matrix of shape (N_patents, n_simulations) with abandoned rows zeroed out.
    """
    w = np.asarray(keep_mask, dtype=float)[:, None]
    return matrix * w


def apply_strategy_a(
    expected_costs: np.ndarray, target_cost_cut: float
) -> np.ndarray:
    """Strategy A (Naive Cost-Cutting).

    Sort patents descending by expected discounted maintenance cost E[PV(Cost)]
    and abandon the most expensive patents first until cumulative expected
    cost savings >= target cut.

    Args:
        expected_costs: Expected discounted maintenance cost per patent (shape: (N,)).
        target_cost_cut: Target expected cost savings in EUR.

    Returns:
        keep_mask: Boolean array of shape (N,) where True = kept, False = abandoned.
    """
    n_patents = len(expected_costs)
    keep_mask = np.ones(n_patents, dtype=bool)

    if target_cost_cut <= 0:
        return keep_mask

    # Sort descending by expected cost
    order = np.argsort(-expected_costs, kind="stable")
    cum_savings = np.cumsum(expected_costs[order])

    # Find the cut-off index where cumulative savings meet or exceed target
    cut_idx = int(np.searchsorted(cum_savings, target_cost_cut, side="left"))
    if cut_idx < n_patents:
        abandon_indices = order[: cut_idx + 1]
    else:
        abandon_indices = order

    keep_mask[abandon_indices] = False
    return keep_mask


def apply_strategy_b(
    expected_npv: np.ndarray,
    expected_costs: np.ndarray,
    target_cost_cut: float,
) -> np.ndarray:
    """Strategy B (Value-to-Cost Ratio).

    Rank patents ascending by expected bang-for-buck ratio E[NPV] / E[PV(Cost)]
    and abandon the lowest-ratio patents first until cumulative expected cost
    savings >= target cut.

    Args:
        expected_npv: Expected Net Present Value per patent (shape: (N,)).
        expected_costs: Expected discounted maintenance cost per patent (shape: (N,)).
        target_cost_cut: Target expected cost savings in EUR.

    Returns:
        keep_mask: Boolean array of shape (N,) where True = kept, False = abandoned.
    """
    n_patents = len(expected_costs)
    keep_mask = np.ones(n_patents, dtype=bool)

    if target_cost_cut <= 0:
        return keep_mask

    # Compute ratio: if cost is zero, assigning np.inf ensures it does not consume target cost cut
    ratio = np.where(expected_costs > 0, expected_npv / expected_costs, np.inf)

    # Sort ascending by ratio (lowest value per cost abandoned first)
    order = np.argsort(ratio, kind="stable")
    costs_in_order = expected_costs[order]
    cum_savings = np.cumsum(costs_in_order)

    cut_idx = int(np.searchsorted(cum_savings, target_cost_cut, side="left"))
    if cut_idx < n_patents:
        abandon_indices = order[: cut_idx + 1]
    else:
        abandon_indices = order

    keep_mask[abandon_indices] = False
    return keep_mask


def apply_strategy_c(
    expected_npv: np.ndarray,
    expected_costs: np.ndarray,
    is_sep: np.ndarray,
    target_cost_cut: float,
) -> np.ndarray:
    """Strategy C (SEP-Protected Ratio).

    Protect all Standard-Essential Patents (is_sep == True) from pruning, and
    rank only non-SEP patents ascending by E[NPV] / E[PV(Cost)] to hit the
    target cost cut. If non-SEPs alone cannot reach the target cut, log a
    warning and cut the maximum available non-SEP spend.

    Args:
        expected_npv: Expected Net Present Value per patent (shape: (N,)).
        expected_costs: Expected discounted maintenance cost per patent (shape: (N,)).
        is_sep: Boolean array of shape (N,) indicating Standard-Essential Patents.
        target_cost_cut: Target expected cost savings in EUR.

    Returns:
        keep_mask: Boolean array of shape (N,) where True = kept, False = abandoned.
    """
    n_patents = len(expected_costs)
    keep_mask = np.ones(n_patents, dtype=bool)

    if target_cost_cut <= 0:
        return keep_mask

    non_sep_indices = np.where(~is_sep)[0]
    if len(non_sep_indices) == 0:
        msg = "No non-SEP patents available to prune under Strategy C."
        logger.warning(msg)
        warnings.warn(msg, UserWarning)
        return keep_mask

    non_sep_costs = expected_costs[non_sep_indices]
    non_sep_npv = expected_npv[non_sep_indices]
    total_non_sep_cost = float(np.sum(non_sep_costs))

    if total_non_sep_cost < target_cost_cut:
        msg = (
            f"Strategy C warning: Total non-SEP expected maintenance cost "
            f"(EUR {total_non_sep_cost:,.2f}) is insufficient to reach the target cut "
            f"(EUR {target_cost_cut:,.2f}). Pruning all available non-SEP spend."
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning)
        keep_mask[non_sep_indices] = False
        return keep_mask

    ratio_non_sep = np.where(
        non_sep_costs > 0, non_sep_npv / non_sep_costs, np.inf
    )
    order_local = np.argsort(ratio_non_sep, kind="stable")
    cum_savings = np.cumsum(non_sep_costs[order_local])

    cut_idx = int(np.searchsorted(cum_savings, target_cost_cut, side="left"))
    if cut_idx < len(non_sep_indices):
        abandon_local = order_local[: cut_idx + 1]
    else:
        abandon_local = order_local

    abandon_indices = non_sep_indices[abandon_local]
    keep_mask[abandon_indices] = False
    return keep_mask


def compute_portfolio_metrics(
    strategy_name: str,
    npv_matrix: np.ndarray,
    pv_cost_matrix: np.ndarray,
    pv_rev_matrix: np.ndarray,
    keep_mask: np.ndarray,
    baseline_expected_cost: float,
) -> StrategyResult:
    """Compute distribution metrics across Monte Carlo simulations for a given strategy.

    Enforces the Abandonment Invariant on all matrices prior to aggregation.

    Args:
        strategy_name: Name of the strategy.
        npv_matrix: NPV matrix of shape (N, S).
        pv_cost_matrix: Present value cost matrix of shape (N, S).
        pv_rev_matrix: Present value revenue matrix of shape (N, S).
        keep_mask: Boolean keep mask of shape (N,).
        baseline_expected_cost: Total expected cost of the unpruned baseline portfolio.

    Returns:
        StrategyResult containing summary metrics and simulation distribution vectors.
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    abandon_mask = ~keep_mask

    # Enforce Abandonment Invariant: zero out abandoned patents
    npv_pruned = apply_abandonment_invariant(npv_matrix, keep_mask)
    pv_cost_pruned = apply_abandonment_invariant(pv_cost_matrix, keep_mask)
    pv_rev_pruned = apply_abandonment_invariant(pv_rev_matrix, keep_mask)

    # Aggregate cash flows across kept patents for each simulation path s in [1..S]
    portfolio_npv_draws = np.sum(npv_pruned, axis=0)  # shape: (S,)
    portfolio_cost_draws = np.sum(pv_cost_pruned, axis=0)  # shape: (S,)
    portfolio_rev_draws = np.sum(pv_rev_pruned, axis=0)  # shape: (S,)

    mean_npv = float(np.mean(portfolio_npv_draws))
    p10_npv = float(np.percentile(portfolio_npv_draws, 10))
    p50_npv = float(np.percentile(portfolio_npv_draws, 50))
    p90_npv = float(np.percentile(portfolio_npv_draws, 90))
    prob_net_loss = float(np.mean(portfolio_npv_draws < 0.0))

    expected_pv_cost = float(np.mean(portfolio_cost_draws))
    patents_abandoned_count = int(np.sum(abandon_mask))

    if patents_abandoned_count == 0 or np.all(keep_mask):
        cost_reduction_pct = 0.0
    elif baseline_expected_cost > 0:
        cost_reduction_pct = float(1.0 - (expected_pv_cost / baseline_expected_cost))
    else:
        cost_reduction_pct = 0.0

    return StrategyResult(
        strategy_name=strategy_name,
        keep_mask=keep_mask,
        abandon_mask=abandon_mask,
        mean_npv=mean_npv,
        p10_npv=p10_npv,
        p50_npv=p50_npv,
        p90_npv=p90_npv,
        prob_net_loss=prob_net_loss,
        expected_pv_cost=expected_pv_cost,
        cost_reduction_pct=cost_reduction_pct,
        patents_abandoned_count=patents_abandoned_count,
        portfolio_npv_draws=portfolio_npv_draws,
        portfolio_cost_draws=portfolio_cost_draws,
        portfolio_rev_draws=portfolio_rev_draws,
    )


def evaluate_pruning_strategies(
    simulation_result: SimulationResult | Mapping[str, Any],
    portfolio_df_or_rows: Any,
    assumptions: dict[str, Any] | None = None,
    target_cost_cut_pct: float | None = None,
) -> dict[str, StrategyResult]:
    """Evaluate Baseline and three pruning strategies against the target cost cut.

    Strategies evaluated:
        - Baseline (Keep All): w_i = 1 for all patents.
        - Strategy A (Naive Cost-Cutting): Abandon most expensive patents first.
        - Strategy B (Value-to-Cost Ratio): Abandon lowest E[NPV]/E[Cost] ratio first.
        - Strategy C (SEP-Protected Ratio): Protect SEPs, rank non-SEPs by ratio.

    Args:
        simulation_result: SimulationResult or dict containing npv_matrix,
            pv_cost_matrix, and pv_rev_matrix.
        portfolio_df_or_rows: Raw portfolio data to extract metadata (is_sep, etc.).
        assumptions: Model assumptions dictionary (optional if target_cost_cut_pct given).
        target_cost_cut_pct: Target cost reduction percentage (e.g. 0.20).

    Returns:
        Dictionary mapping strategy names to StrategyResult objects.
    """
    if target_cost_cut_pct is None:
        if assumptions is None:
            assumptions = load_assumptions()
        target_cost_cut_pct = float(assumptions.get("target_cost_cut_pct", 0.20))

    npv_matrix = simulation_result["npv_matrix"]
    pv_cost_matrix = simulation_result["pv_cost_matrix"]
    pv_rev_matrix = simulation_result["pv_rev_matrix"]

    _, _, is_sep, _, _ = _parse_portfolio_data(portfolio_df_or_rows)

    n_patents = npv_matrix.shape[0]

    # Expected values per patent across all simulations
    expected_costs = np.mean(pv_cost_matrix, axis=1)  # shape: (N,)
    expected_npv = np.mean(npv_matrix, axis=1)  # shape: (N,)

    total_expected_cost = float(np.sum(expected_costs))
    target_cost_cut = target_cost_cut_pct * total_expected_cost

    # 1. Baseline: Keep all patents
    keep_baseline = np.ones(n_patents, dtype=bool)
    baseline_result = compute_portfolio_metrics(
        strategy_name="Baseline (Keep All)",
        npv_matrix=npv_matrix,
        pv_cost_matrix=pv_cost_matrix,
        pv_rev_matrix=pv_rev_matrix,
        keep_mask=keep_baseline,
        baseline_expected_cost=total_expected_cost,
    )

    # 2. Strategy A (Naive Cost-Cutting)
    keep_strat_a = apply_strategy_a(expected_costs, target_cost_cut)
    strat_a_result = compute_portfolio_metrics(
        strategy_name="Strategy A (Naive Cost-Cutting)",
        npv_matrix=npv_matrix,
        pv_cost_matrix=pv_cost_matrix,
        pv_rev_matrix=pv_rev_matrix,
        keep_mask=keep_strat_a,
        baseline_expected_cost=total_expected_cost,
    )

    # 3. Strategy B (Value-to-Cost Ratio)
    keep_strat_b = apply_strategy_b(expected_npv, expected_costs, target_cost_cut)
    strat_b_result = compute_portfolio_metrics(
        strategy_name="Strategy B (Value-to-Cost Ratio)",
        npv_matrix=npv_matrix,
        pv_cost_matrix=pv_cost_matrix,
        pv_rev_matrix=pv_rev_matrix,
        keep_mask=keep_strat_b,
        baseline_expected_cost=total_expected_cost,
    )

    # 4. Strategy C (SEP-Protected Ratio)
    keep_strat_c = apply_strategy_c(
        expected_npv, expected_costs, is_sep, target_cost_cut
    )
    strat_c_result = compute_portfolio_metrics(
        strategy_name="Strategy C (SEP-Protected Ratio)",
        npv_matrix=npv_matrix,
        pv_cost_matrix=pv_cost_matrix,
        pv_rev_matrix=pv_rev_matrix,
        keep_mask=keep_strat_c,
        baseline_expected_cost=total_expected_cost,
    )

    return {
        "Baseline (Keep All)": baseline_result,
        "Strategy A (Naive Cost-Cutting)": strat_a_result,
        "Strategy B (Value-to-Cost Ratio)": strat_b_result,
        "Strategy C (SEP-Protected Ratio)": strat_c_result,
    }
