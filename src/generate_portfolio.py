"""Generate synthetic telecom patent portfolio snapshot."""

import csv
from pathlib import Path
import numpy as np


def generate_portfolio(
    output_path: str | Path = "data/portfolio_snapshot.csv",
    n_families: int = 500,
    seed: int = 42,
) -> Path:
    """Generate a realistic synthetic telecom patent portfolio snapshot.

    Args:
        output_path: Path where the CSV file should be saved.
        n_families: Number of patent families to generate (default: 500).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Path to the saved CSV file.
    """
    rng = np.random.default_rng(seed)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    family_ids = [f"FAM-{i:04d}" for i in range(1, n_families + 1)]

    # Jurisdiction distribution: US, EP, DE, GB, JP, CN
    jurisdictions = ["US", "EP", "DE", "GB", "JP", "CN"]
    jurisdiction_weights = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
    jurisdiction_choices = rng.choice(
        jurisdictions, size=n_families, p=jurisdiction_weights
    )

    # Patent age: integers 1 to 15
    patent_ages = rng.integers(1, 16, size=n_families)

    # Standard-Essential Patents (~25% True)
    is_sep = rng.random(size=n_families) < 0.25

    # Tech domains
    domains = [
        "5G_RAN",
        "CORE_NETWORK",
        "VIDEO_CODEC",
        "IOT_CONNECTIVITY",
        "LEGACY_3G_4G",
    ]
    domain_weights = [0.30, 0.25, 0.15, 0.15, 0.15]
    domain_choices = rng.choice(domains, size=n_families, p=domain_weights)

    # Forward citations: skewed distribution 0 to 45
    # Older patents and SEPs naturally accumulate more citations on average
    base_citations = rng.negative_binomial(n=2, p=0.25, size=n_families)
    sep_citation_boost = np.where(is_sep, rng.integers(3, 12, size=n_families), 0)
    age_citation_boost = (patent_ages * 0.8).astype(int)
    forward_citations = np.clip(
        base_citations + sep_citation_boost + age_citation_boost, 0, 45
    )

    # Realistic maintenance cost in EUR scaling with age (1-15) and jurisdiction
    # Costs range between 800 and 6,500 EUR.
    # Older patents have escalating renewal fees, and major registries (EP, US) carry higher fees.
    jurisdiction_multipliers = {
        "EP": 1.30,
        "US": 1.25,
        "DE": 1.00,
        "GB": 0.95,
        "JP": 1.05,
        "CN": 0.85,
    }

    maintenance_costs = []
    for age, jur in zip(patent_ages, jurisdiction_choices):
        # Base fee progression with age: steepens at higher ages
        # Age 1: ~900-1100 EUR, Age 15: ~4500-5200 EUR before jurisdiction multiplier
        norm_age = (age - 1) / 14.0  # 0.0 to 1.0
        base_cost = 950.0 + 3800.0 * (norm_age**1.35)
        jur_mult = jurisdiction_multipliers[jur]
        noise = rng.uniform(-150.0, 150.0)
        cost = (base_cost * jur_mult) + noise
        clamped_cost = round(float(np.clip(cost, 800.0, 6500.0)), 2)
        maintenance_costs.append(clamped_cost)

    fieldnames = [
        "family_id",
        "jurisdiction",
        "patent_age",
        "is_sep",
        "forward_citations",
        "tech_domain",
        "annual_maintenance_eur",
    ]

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_families):
            writer.writerow(
                {
                    "family_id": family_ids[i],
                    "jurisdiction": jurisdiction_choices[i],
                    "patent_age": int(patent_ages[i]),
                    "is_sep": bool(is_sep[i]),
                    "forward_citations": int(forward_citations[i]),
                    "tech_domain": domain_choices[i],
                    "annual_maintenance_eur": maintenance_costs[i],
                }
            )

    return output_file


if __name__ == "__main__":
    generated_path = generate_portfolio()
    print(f"Generated portfolio snapshot at {generated_path}")
