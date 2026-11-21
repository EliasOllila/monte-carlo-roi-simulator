"""Monte Carlo ROI simulator engine for patent portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import numpy as np
import yaml


@dataclass
class SimulationResult:
    """Container for portfolio simulation matrices.

    Attributes:
        npv_matrix: Net Present Value matrix of shape (N_patents, n_simulations).
        pv_cost_matrix: Present value of maintenance costs of shape (N_patents, n_simulations).
        pv_rev_matrix: Present value of licensing revenues of shape (N_patents, n_simulations).
        cash_flows_c: Undiscounted cost array of shape (N_patents, horizon_years, n_simulations).
        cash_flows_r: Undiscounted revenue array of shape (N_patents, horizon_years, n_simulations).
        family_ids: List of patent family IDs of length N_patents.
    """

    npv_matrix: np.ndarray
    pv_cost_matrix: np.ndarray
    pv_rev_matrix: np.ndarray
    cash_flows_c: np.ndarray
    cash_flows_r: np.ndarray
    family_ids: list[str]

    def __getitem__(self, item: Any) -> Any:
        """Allow dict-like string key access as well as index / slice access."""
        if isinstance(item, str):
            try:
                return getattr(self, item)
            except AttributeError:
                raise KeyError(item)
        fields = (
            self.npv_matrix,
            self.pv_cost_matrix,
            self.pv_rev_matrix,
            self.cash_flows_c,
            self.cash_flows_r,
            self.family_ids,
        )
        return fields[item]

    def __iter__(self):
        """Unpack directly into primary simulation matrices (npv, pv_cost, pv_rev)."""
        return iter((self.npv_matrix, self.pv_cost_matrix, self.pv_rev_matrix))

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute by string key with optional default."""
        return getattr(self, key, default)

    def keys(self) -> Sequence[str]:
        """Return field names for mapping compatibility."""
        return (
            "npv_matrix",
            "pv_cost_matrix",
            "pv_rev_matrix",
            "cash_flows_c",
            "cash_flows_r",
            "family_ids",
        )


def load_assumptions(config_path: str | Path = "config/assumptions.yaml") -> dict[str, Any]:
    """Load model assumptions from a YAML file.

    Args:
        config_path: Path to the assumptions YAML file.

    Returns:
        Dictionary of assumptions.
    """
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        assumptions = yaml.safe_load(f)
    return assumptions


def load_portfolio(data_path: str | Path = "data/portfolio_snapshot.csv") -> list[dict[str, Any]]:
    """Load patent portfolio snapshot from a CSV file.

    Args:
        data_path: Path to the CSV file.

    Returns:
        List of dictionary rows representing patent families.
    """
    path = Path(data_path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "family_id": row["family_id"],
                    "jurisdiction": row["jurisdiction"],
                    "patent_age": int(row["patent_age"]),
                    "is_sep": row["is_sep"].strip().lower() in {"true", "1", "yes"},
                    "forward_citations": int(row["forward_citations"]),
                    "tech_domain": row["tech_domain"],
                    "annual_maintenance_eur": float(row["annual_maintenance_eur"]),
                }
            )
    return rows


def _parse_portfolio_data(portfolio_df_or_rows: Any) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract required numerical and boolean arrays from input data.

    Supports:
        - List of dicts (from csv.DictReader or load_portfolio)
        - pandas.DataFrame or duck-typed tabular objects
        - Dict of sequences
        - Path/str to a CSV file

    Returns:
        Tuple of (family_ids, patent_age, is_sep, forward_citations, annual_maintenance_eur).
    """
    if isinstance(portfolio_df_or_rows, (str, Path)):
        rows = load_portfolio(portfolio_df_or_rows)
        return _parse_portfolio_data(rows)

    # Duck-typing for pandas DataFrame or mapping of columns
    if hasattr(portfolio_df_or_rows, "to_dict") and hasattr(portfolio_df_or_rows, "columns"):
        family_ids = [str(x) for x in portfolio_df_or_rows["family_id"].tolist()]
        patent_age = np.asarray(portfolio_df_or_rows["patent_age"], dtype=int)
        is_sep_raw = portfolio_df_or_rows["is_sep"]
        is_sep = np.asarray([bool(x) if not isinstance(x, str) else x.strip().lower() in {"true", "1"} for x in is_sep_raw], dtype=bool)
        forward_citations = np.asarray(portfolio_df_or_rows["forward_citations"], dtype=float)
        annual_maintenance_eur = np.asarray(portfolio_df_or_rows["annual_maintenance_eur"], dtype=float)
        return family_ids, patent_age, is_sep, forward_citations, annual_maintenance_eur

    if isinstance(portfolio_df_or_rows, Mapping):
        family_ids = [str(x) for x in portfolio_df_or_rows["family_id"]]
        patent_age = np.asarray(portfolio_df_or_rows["patent_age"], dtype=int)
        is_sep_raw = portfolio_df_or_rows["is_sep"]
        is_sep = np.asarray([bool(x) if not isinstance(x, str) else str(x).strip().lower() in {"true", "1"} for x in is_sep_raw], dtype=bool)
        forward_citations = np.asarray(portfolio_df_or_rows["forward_citations"], dtype=float)
        annual_maintenance_eur = np.asarray(portfolio_df_or_rows["annual_maintenance_eur"], dtype=float)
        return family_ids, patent_age, is_sep, forward_citations, annual_maintenance_eur

    # Sequence of row dicts
    family_ids: list[str] = []
    patent_age_list: list[int] = []
    is_sep_list: list[bool] = []
    citations_list: list[float] = []
    maintenance_list: list[float] = []

    for row in portfolio_df_or_rows:
        family_ids.append(str(row["family_id"]))
        patent_age_list.append(int(row["patent_age"]))
        sep_val = row["is_sep"]
        if isinstance(sep_val, str):
            is_sep_list.append(sep_val.strip().lower() in {"true", "1", "yes"})
        else:
            is_sep_list.append(bool(sep_val))
        citations_list.append(float(row["forward_citations"]))
        maintenance_list.append(float(row["annual_maintenance_eur"]))

    return (
        family_ids,
        np.asarray(patent_age_list, dtype=int),
        np.asarray(is_sep_list, dtype=bool),
        np.asarray(citations_list, dtype=float),
        np.asarray(maintenance_list, dtype=float),
    )


def simulate_portfolio_npv(
    portfolio_df_or_rows: Any,
    assumptions: dict[str, Any],
    seed_override: int | None = None,
) -> SimulationResult:
    """Simulate patent portfolio cash flows and Net Present Value via Monte Carlo.

    Uses Common Random Numbers (CRN) across the portfolio with pure, deterministic
    evaluation given the random seed.

    Cash flows and expiration invariants:
        - Horizon: projection years t = 1..horizon_years.
        - Expiration: effective_age = patent_age + t. If effective_age > 20, both
          maintenance cost C and licensing revenue R are set to exact 0.0.
        - Obsolescence: exponential technology decay (1 - annual_obsolescence_rate) ** t.
        - Costs C: compounded annual inflation draws Normal(fee_inflation_mean, fee_inflation_std).
        - Revenues R: zero-inflated log-normal IP payoffs:
            * Bernoulli monetization mask per patent-simulation.
            * Log-normal cash flow magnitude scaled by sep_value_multiplier and
              mild citation factor (1 + 0.02 * forward_citations).
        - Discounting: (R - C) / (1 + discount_rate) ** t.

    Args:
        portfolio_df_or_rows: Portfolio data (list of row dicts, DataFrame, or CSV path).
        assumptions: Dictionary containing model parameters.
        seed_override: Optional seed override for random generator reproducibility.

    Returns:
        SimulationResult containing npv_matrix, pv_cost_matrix, pv_rev_matrix,
        cash_flows_c, cash_flows_r, and family_ids.
    """
    seed = seed_override if seed_override is not None else int(assumptions.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    (
        family_ids,
        patent_age,
        is_sep,
        forward_citations,
        annual_maintenance_eur,
    ) = _parse_portfolio_data(portfolio_df_or_rows)

    n_patents = len(family_ids)
    horizon_years = int(assumptions["horizon_years"])
    n_simulations = int(assumptions["n_simulations"])
    discount_rate = float(assumptions["discount_rate"])
    fee_inflation_mean = float(assumptions["fee_inflation_mean"])
    fee_inflation_std = float(assumptions["fee_inflation_std"])
    prob_monetized_non_sep = float(assumptions["prob_monetized_non_sep"])
    prob_monetized_sep = float(assumptions["prob_monetized_sep"])
    lognormal_mu = float(assumptions["lognormal_mu"])
    lognormal_sigma = float(assumptions["lognormal_sigma"])
    sep_value_multiplier = float(assumptions["sep_value_multiplier"])
    annual_obsolescence_rate = float(assumptions["annual_obsolescence_rate"])

    # Projection year grid t = 1 .. horizon_years
    t_grid = np.arange(1, horizon_years + 1)  # shape: (horizon_years,)

    # Expiration mask: effective_age = patent_age + t; active if <= 20
    # Shape: (n_patents, horizon_years)
    effective_age = patent_age[:, None] + t_grid[None, :]
    active_mask = (effective_age <= 20)

    # ---------------------------------------------------------
    # 1. Stochastic Maintenance Costs C: shape (N, H, S)
    # ---------------------------------------------------------
    # Annual inflation rate draws per simulation year: shape (horizon_years, n_simulations)
    inflation_draws = rng.normal(
        loc=fee_inflation_mean,
        scale=fee_inflation_std,
        size=(horizon_years, n_simulations),
    )
    # Cumulative compound inflation from t=1 to horizon_years: shape (horizon_years, n_simulations)
    cum_inflation = np.cumprod(1.0 + inflation_draws, axis=0)

    # Base cost compounded by inflation: (N, 1, 1) * (1, H, S) -> (N, H, S)
    cash_flows_c = annual_maintenance_eur[:, None, None] * cum_inflation[None, :, :]
    # Apply expiration invariant (cost is exact 0.0 if effective_age > 20)
    cash_flows_c = np.where(active_mask[:, :, None], cash_flows_c, 0.0)

    # ---------------------------------------------------------
    # 2. Stochastic Licensing Revenue R: shape (N, H, S)
    # ---------------------------------------------------------
    # Bernoulli monetization mask per patent-simulation: shape (N, S)
    prob_monetized = np.where(is_sep, prob_monetized_sep, prob_monetized_non_sep)  # (N,)
    u_draws = rng.random(size=(n_patents, n_simulations))
    monetization_mask = (u_draws < prob_monetized[:, None])  # (N, S)

    # Lognormal cash flow magnitude: shape (N, S)
    base_magnitude = rng.lognormal(
        mean=lognormal_mu,
        sigma=lognormal_sigma,
        size=(n_patents, n_simulations),
    )
    sep_multiplier = np.where(is_sep, sep_value_multiplier, 1.0)  # (N,)
    citation_factor = 1.0 + 0.02 * forward_citations  # (N,)
    scale = (sep_multiplier * citation_factor)[:, None]  # (N, 1)

    cash_flow_magnitude = base_magnitude * scale  # (N, S)
    monetized_cash_flow = np.where(monetization_mask, cash_flow_magnitude, 0.0)  # (N, S)

    # Exponential technology decay across projection years: shape (horizon_years,)
    tech_decay = (1.0 - annual_obsolescence_rate) ** t_grid

    # Revenue across years: (N, 1, S) * (1, H, 1) -> (N, H, S)
    cash_flows_r = monetized_cash_flow[:, None, :] * tech_decay[None, :, None]
    # Apply expiration invariant (revenue is exact 0.0 if effective_age > 20)
    cash_flows_r = np.where(active_mask[:, :, None], cash_flows_r, 0.0)

    # ---------------------------------------------------------
    # 3. Discounting: t = 1 .. horizon_years
    # ---------------------------------------------------------
    discount_factors = 1.0 / ((1.0 + discount_rate) ** t_grid)  # shape: (horizon_years,)

    # Discounted cash flow matrices: shape (N, H, S)
    pv_c_by_year = cash_flows_c * discount_factors[None, :, None]
    pv_r_by_year = cash_flows_r * discount_factors[None, :, None]

    # Sum across horizon years (axis=1) -> shape (N, S)
    pv_cost_matrix = np.sum(pv_c_by_year, axis=1)
    pv_rev_matrix = np.sum(pv_r_by_year, axis=1)
    npv_matrix = pv_rev_matrix - pv_cost_matrix

    return SimulationResult(
        npv_matrix=npv_matrix,
        pv_cost_matrix=pv_cost_matrix,
        pv_rev_matrix=pv_rev_matrix,
        cash_flows_c=cash_flows_c,
        cash_flows_r=cash_flows_r,
        family_ids=family_ids,
    )
