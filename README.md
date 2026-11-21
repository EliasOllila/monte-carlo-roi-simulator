# Monte Carlo Patent Portfolio Pruning & ROI Simulator

An enterprise-grade, fully vectorized **Monte Carlo Patent Portfolio Pruning & ROI Simulator** built in Python 3.10+ using **NumPy** and **Common Random Numbers (CRN)**.

---

## 1. Executive Overview & Business Problem

Corporate IP departments frequently face executive mandates to **cut patent maintenance spend by 20% or more**. However, traditional patent pruning methods rely on naive heuristics—such as indiscriminately abandoning the most expensive foreign filings or elderly patents—which inadvertently discard high-yield Standard-Essential Patents (SEPs) and destroy massive future licensing revenue.

This simulator models the stochastic commercial reality of patent portfolios:
- **Skewed Commercial Value**: Intellectual property returns follow a zero-inflated, right-skewed lognormal distribution where <15% of non-SEPs and ~45% of SEPs ever achieve commercial monetization.
- **Inflation & Expiration Dynamics**: Annual patent maintenance fees compound over time under stochastic renewal inflation until exact expiration at age 20.
- **Technology Obsolescence**: Active patents face exponential technological obsolescence as next-generation cellular and networking standards emerge.
- **Optimal Capital Allocation**: By evaluating portfolios under **Common Random Numbers (CRN)**, decision-makers can compare pruning strategies on identical stochastic trajectories to identify the **Value-to-Cost Bang-for-Buck** frontier.

---

## 2. 3-Step Quickstart

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: `numpy>=1.26`, `PyYAML>=6.0`, `matplotlib>=3.8`, `pytest>=8.0`)*

### Step 2: Run Full Simulation & Decision Analysis
```bash
python run_simulation.py
```
This executes 10,000 Monte Carlo paths across 500 patent families, evaluates pruning strategies, runs One-At-a-Time (OAT) and Latin-Hypercube sensitivity diagnostics, prints an executive summary table, and generates the decision memo and visual charts in `output/`.

### Step 3: Run the Test Suite
```bash
pytest -v
```

---

## 3. Mathematical Formulation

### A. Zero-Inflated Lognormal Commercial Payoff Model
For each patent family $i \in \{1, \dots, N\}$ and simulation trajectory $s \in \{1, \dots, S\}$:
1. **Bernoulli Monetization Event**:
   $$\mathbb{P}(\text{Monetized}_{i, s} = 1) = \begin{cases} p_{\text{sep}} & \text{if } \text{is\_sep}_i = \text{True} \\ p_{\text{non\_sep}} & \text{otherwise} \end{cases}$$
2. **Commercial Value Magnitude**:
   $$\text{Magnitude}_{i, s} = \exp\left(\mu + \sigma Z_{i, s}\right) \times \text{Multiplier}_i \times (1 + 0.02 \times \text{citations}_i)$$
   where $Z_{i, s} \sim \mathcal{N}(0, 1)$, and $\text{Multiplier}_i = \text{sep\_value\_multiplier}$ if SEP else $1.0$.
3. **Obsolescence & Expiration Across Horizon $t \in \{1, \dots, H\}$**:
   $$R_{i, t, s} = \begin{cases} \text{Monetized}_{i, s} \cdot \text{Magnitude}_{i, s} \cdot (1 - \delta)^t & \text{if } \text{age}_i + t \le 20 \\ 0.0 & \text{if } \text{age}_i + t > 20 \text{ (Expiration Invariant)} \end{cases}$$
   where $\delta$ is `annual_obsolescence_rate`.

### B. Stochastic Maintenance Costs
Annual renewal annuities compound with stochastic inflation rates $I_{t, s} \sim \mathcal{N}(\mu_{\text{infl}}, \sigma_{\text{infl}})$:
$$C_{i, t, s} = \begin{cases} \text{BaseCost}_i \cdot \prod_{k=1}^t (1 + I_{k, s}) & \text{if } \text{age}_i + t \le 20 \\ 0.0 & \text{if } \text{age}_i + t > 20 \text{ (Expiration Invariant)} \end{cases}$$

### C. Discounting & Net Present Value (NPV)
Discounted at corporate hurdle rate $r$ (`discount_rate`):
$$PV(R_{i, s}) = \sum_{t=1}^H \frac{R_{i, t, s}}{(1 + r)^t}, \quad PV(C_{i, s}) = \sum_{t=1}^H \frac{C_{i, t, s}}{(1 + r)^t}$$
$$NPV_{i, s} = PV(R_{i, s}) - PV(C_{i, s})$$

### D. Abandonment Invariant & Strategy Aggregation
When a strategy decides binary abandonment decision $w_i \in \{0, 1\}$ ($w_i = 0$ for abandoned patents):
$$\text{Portfolio NPV}_s = \sum_{i=1}^N w_i \cdot NPV_{i, s}$$
Because $w_i = 0$ sets all future cash flows to exact $0.0$, abandoned patents generate zero future maintenance liabilities and zero licensing revenue.

---

## 4. Pruning Strategies Evaluated

| Strategy | Description | Selection Logic |
| :--- | :--- | :--- |
| **Baseline (Keep All)** | Status-quo unpruned portfolio. | $w_i = 1$ for all patents. |
| **Strategy A (Naive Cost-Cutting)** | Drop most expensive patents. | Sort descending by $\mathbb{E}[PV(C_i)]$ until cost savings $\ge 20\%$. |
| **Strategy B (Value-to-Cost Ratio)** | **Recommended: Maximize bang-for-buck.** | Sort ascending by $\frac{\mathbb{E}[NPV_i]}{\mathbb{E}[PV(C_i)]}$ until cost savings $\ge 20\%$. |
| **Strategy C (SEP-Protected Ratio)** | Protect all Standard-Essential Patents. | Fix $w_i = 1$ for SEPs; prune lowest ratio non-SEPs until cost savings $\ge 20\%$. |

---

## 5. Modifying Assumptions (`config/assumptions.yaml`)

All macroeconomic and patent portfolio assumptions can be edited in [config/assumptions.yaml](config/assumptions.yaml):

```yaml
n_simulations: 10000              # Number of Monte Carlo simulation trajectories
horizon_years: 10                 # Projection horizon in years
random_seed: 42                   # Common Random Numbers (CRN) master seed
discount_rate: 0.08               # WACC / Hurdle discount rate (8%)
fee_inflation_mean: 0.03          # Mean annual renewal fee inflation (3%)
fee_inflation_std: 0.01           # Std dev of annual renewal fee inflation (1%)
target_cost_cut_pct: 0.20         # Target PV maintenance cost reduction (20%)
prob_monetized_non_sep: 0.12      # Commercial licensing probability for non-SEPs (12%)
prob_monetized_sep: 0.45          # Commercial licensing probability for SEPs (45%)
lognormal_mu: 9.8                 # Lognormal payoff location parameter
lognormal_sigma: 1.35             # Lognormal payoff dispersion parameter
sep_value_multiplier: 3.2         # Value uplift factor for Standard-Essential Patents
annual_obsolescence_rate: 0.05    # Exponential technology decay rate per annum (5%)
oat_shock_pct: 0.20               # Sensitivity analysis shock range (±20%)
```

---

## 6. Generated Output Deliverables

Running `python run_simulation.py` produces the following artifacts in `output/`:
- **`output/pruning_decision_memo.md`**: Executive recommendation memo ready for presentation to the CIPO and CFO.
- **`output/npv_distribution.png`**: High-resolution empirical CDF comparing Portfolio NPV distributions across Baseline, Strategy A, Strategy B, and Strategy C.
- **`output/tornado_sensitivity.png`**: Tornado chart ranking parameters by their swing on Strategy B median NPV under $\pm 20\%$ shocks.
