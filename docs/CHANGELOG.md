# Phase log

A running record of what's been built and what remains. Phases match
[`ROADMAP.md`](ROADMAP.md). See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the
target shape of the system.

---

## Phase 0 — Scaffold & DuckDB swap

**Goal:** drop Postgres / SQLAlchemy; introduce Parquet + DuckDB; restructure
the repo into proper Python packages.

**Done**

- `pyproject.toml` as the single source of truth (`basel-alm-pipeline 0.2.0`).
- Flat package layout at the repo root: `basel_common/`, `basel_ingestion/`,
  `basel_risk_engine/`, `basel_dagster/`, `src/`.
- `basel_common/` — pydantic v2 row models, enums (HQLAType, Direction, …),
  DuckDB connection helper with `[duckdb] path = …` secrets fallback chain.
- `basel_ingestion/generate.py` — synthetic Parquet generator (`--small`
  flag for seeds).
- `basel_ingestion/load.py` — idempotent Parquet → DuckDB loader.
- `src/queries.py` rewritten to DuckDB. Public signatures preserved so the
  dashboard pages and `compute.py` didn't move yet.
- `.streamlit/secrets.toml`, `.gitignore`, `scripts/run.cmd`.
- Postgres-era files archived under `legacy/` with a README pointer.
- All five Streamlit pages execute clean via `streamlit.testing.v1.AppTest`.

---

## Phase 1 — Dagster + dbt foundation

**Goal:** every transformation either a dbt model or a Dagster asset; full
DAG visible in the Dagster UI; marts drive the Streamlit dashboard.

**Done**

- **dbt project** (`dbt_project/`) with `dbt-duckdb` adapter and
  `dbt-labs/dbt_utils` package.
- **6 staging views** (`stg_*`) — 1:1 typed views over `source('raw', ...)`.
- **3 intermediate views** — `int_params_wide`, `int_cashflows_enriched`
  (HQLA post-haircut, day/year maturity buckets, signed_amount,
  asf_contribution, rsf_contribution), `int_balance_sheet_pivoted`.
- **11 mart tables** — `mart_lcr`, `mart_lcr_daily`, `mart_nsfr`,
  `mart_nsfr_components`, `mart_nsfr_daily`, `mart_capital_ratios`,
  `mart_capital_ratios_daily`, `mart_rwa_breakdown`, `mart_pv01_profile`,
  `mart_repricing_gap`, `mart_cashflow_gap`.
- **Tests:** unique, not_null, accepted_values, relationships, dbt_utils
  range checks; custom singular tests for LCR inflow cap and capital-stack
  ordering. All green except the ordering one (severity `warn` — synthetic
  data limitation, flipped to `error` in Phase 2 once realistic balances arrive).
- **Dagster code location** (`basel_dagster/`): `parquet_files` →
  multi-asset `raw_tables` (6 outputs keyed `["raw", <t>]`) →
  `@dbt_assets` covering all dbt models. Full DAG visible at
  `http://localhost:3000`.
- `full_refresh` job + `daily_full_refresh` schedule (06:00 Europe/Lisbon).
- `src/compute.py` refactored into a thin marts client. Numerical parity
  with Phase 0 verified to 10+ sig figs.
- Helper scripts: `scripts/dbt.cmd`, `scripts/dagster_dev.cmd`.

---

## Phase 2 — Quant risk engine

**Goal:** real ALM substance — calibrated short-rate model, MC valuation,
BCBS 368 scenarios, supervisory outlier test, behavioural overlay, NII paths.

**Done**

- **Rate models** (`basel_risk_engine/rate_models/`)
  - Vasicek 1F with pydantic-validated parameters.
  - Exact-discretisation MC with antithetic variates.
  - Analytical zero-coupon bond price and zero yield (Brigo–Mercurio A/B).
  - Closed-form OLS calibration on AR(1) representation, with
    half-life and log-likelihood.
  - `MCPathSet` with bulk discount-factor helpers.
- **Behavioural overlay** (`basel_risk_engine/behavioral/nmd.py`)
  - Parametric NMD model: `stable_core_pct`, `core_behavioral_maturity_yrs`,
    `deposit_beta`. Stretches short-maturity deposits' effective repricing
    horizon, leaves other rows untouched.
- **EVE engine** (`basel_risk_engine/valuation/eve.py`)
  - `YieldCurve` with per-EBA-bucket and parallel shifts.
  - BCBS 368 §132 six prescribed scenarios — deterministic curve
    revaluation, not a PV01-linearisation.
  - MC ∆EVE distribution at 1y forward horizon (Vasicek-implied curve per
    path).
  - Supervisory outlier test (`|∆EVE_worst| / Tier1 ≤ 15%`) reporting
    both the deterministic worst and a distributional |∆EVE|₉₉.
- **NII engine** (`basel_risk_engine/valuation/nii.py`)
  - MC ∆NII paths at 12 / 24 / 36-month horizons.
  - Behavioural-aware repricing gap (deposit β reduces NMD sensitivity).
- **Synthetic market data** added to ingestion:
  `short_rate_history.parquet` (5y monthly Vasicek-simulated) and
  `yield_curve.parquet` (initial zero curve).
- **End-to-end runner** (`basel_risk_engine/run.py`) writes 6 Parquet
  outputs to `data/risk_outputs/`.
- **Dagster wiring:** new `risk_engine_run` asset (depends on
  `int_cashflows_enriched`, `mart_capital_ratios`, `raw/short_rate_history`,
  `raw/yield_curve`) and multi-asset `risk_outputs_tables` loading 6 Parquet
  files into DuckDB. Total asset count: **46**.
- **dbt marts** for risk outputs: `stg_risk_*` (5), `mart_eve_bcbs368`,
  `mart_eve_supervisory`, `mart_eve_distribution_stats` (with
  `QUANTILE_CONT` percentiles), `mart_nii_horizon_stats`, `mart_model_metadata`.
  **108 tests, all passing** (1 expected warn).
- **Streamlit IRRBB page upgraded:**
  - Model-lineage expander (κ / θ / σ, half-life, NMD config).
  - Supervisory outlier test panel (worst scenario, ∆EVE, Tier1, ratio,
    breach flag, distributional p99).
  - BCBS 368 bar chart with proper curve revaluation.
  - MC ∆EVE distribution histogram with p1 / p99 / mean reference lines.
  - MC ∆NII violin per horizon.
  - Legacy PV01 × parallel-shock slider retained for intuition.
- **Tests:** 12 hypothesis-/property-based tests in
  `tests/risk_engine/` covering MC terminal mean → θ convergence, bond
  monotonicity in τ, calibration roundtrip, non-mean-reverting rejection,
  yield-curve shift semantics, EVE monotonicity under parallel shock,
  BCBS 368 scenario completeness, supervisory threshold logic, NMD overlay
  selectivity, and NII path shape.

---

## Phase 2.1a — Hull-White 1F

**Goal:** close the "model is not arbitrage-free against today's curve" gap.
Under Vasicek with parameters fitted to history, P_model(0, τ; r0) does not
in general equal P_market(0, τ) — so the MC ΔEVE distribution is biased
relative to the actual curve. HW1F fixes this by construction.

**Done**

- **`basel_risk_engine/rate_models/hull_white.py`** — `HullWhiteParams`,
  `HullWhiteCalibration`, `HullWhiteModel`. Exact-discretisation MC with
  curve-bootstrapped α(t) = f^M(0,t) + σ²/(2a²)(1−e^{−at})². Analytical
  bond price A(0,τ)·exp(−B(0,τ)·r) with A(0,τ) = P^M(0,τ)·exp(B(0,τ)·r0)
  — reproduces the input curve to machine precision when r0 = f^M(0,0).
- **`YieldCurve.forward_rate(τ)`** — instantaneous forward derived
  analytically from the linear-yield interpolation (y(τ) + τ·dy/dτ).
- **`ShortRateModel` Protocol** in `rate_models/paths.py`. `MCPathSet`
  and `EVEEngine.rate_model` now accept any model satisfying the protocol;
  the `vasicek_model` kwarg has been renamed to `rate_model`.
- **`run.py`** — new `--model {hull_white,vasicek}` CLI flag, defaulting
  to `hull_white`. Calibration dispatches per model. r0 is pinned to
  f^M(0, 0) for HW1F so the curve-fit residual is 0 by construction.
- **Metadata schema** (`risk_model_metadata`) refactored from a fixed
  Vasicek-shaped row to a model-agnostic row: `model_family`,
  `model_version`, `params_json`, `calibration_*`, `curve_fit_max_residual`,
  `n_mc_paths`, `mc_horizon_years`, `mc_dt`, `nmd_params_json`. dbt staging
  + mart updated with `accepted_values` test on `model_family`.
- **`dashboard/pages/2_IRRBB.py`** — lineage panel dispatches per family,
  shows curve-fit residual prominently (≈0 for HW1F, non-trivial for
  Vasicek under a sloped curve).
- **Tests** — `tests/risk_engine/test_hull_white.py`: 8 property tests
  covering grid and off-grid curve-fit, monotonicity in r, MC expected
  short rate vs forward+convexity, calibration recovery of σ,
  non-mean-reverting rejection, σ→0 ΔEVE variance collapse, and
  cross-model API compatibility (`EVEEngine.mc_distribution` accepts
  either Vasicek or HW1F). **20 hypothesis tests total, all passing.**

## Phase 2.1b — FTP engine + NII attribution

**Goal:** decompose book NII into the slices an ALM/treasury committee
actually monitors — customer margin (commercial), funding margin (treasury),
and the NMD *behavioural value* that the deposit business earns from being
priced at long behavioural maturity rather than contractual O/N.

**Done**

- **`basel_risk_engine/ftp/curve.py`** — `LiquidityPremiumSchedule` and
  `FTPCurve`. Internal FTP curve = wholesale base + per-tenor LP add-on
  in bps; linear interp on LP, flat-tail extrapolation. `overnight_funding_rate()`
  returns the τ→0 reference used as 'cost of funding' in the funding-margin leg.
- **`basel_risk_engine/ftp/attribution.py`** — `compute_attribution(book,
  ftp_curve)` returns per-row and book-level `customer_margin`,
  `funding_margin`, `behavioral_value`, `nii_total`. Total NII is invariant
  to the FTP choice — only the split moves.
- **Ingestion extensions:** synthetic `customer_rate` column on cashflows
  (matched-tenor base + per-product commercial spread + ±10bps jitter;
  loans +200bps, bonds +50bps, deposits −150bps). New
  `liquidity_premium` raw table with a realistic 2bps→80bps LP curve
  bootstrapped per valuation date.
- **`run.py`** — reads `liquidity_premium`, builds the FTP curve, runs
  attribution per scenario, writes 3 new Parquets:
  `risk_ftp_curve`, `risk_nii_attribution`, `risk_nii_attribution_rows`.
- **Dagster** — `RAW_TABLES` gains `liquidity_premium`; `RISK_TABLES`
  gains the 3 new risk outputs; `risk_engine_run` depends on
  `raw/liquidity_premium`.
- **dbt** — `stg_liquidity_premium`, `stg_risk_ftp_curve`,
  `stg_risk_nii_attribution`, `stg_risk_nii_attribution_rows` (4 views).
  Marts: `mart_ftp_curve`, `mart_nii_attribution`,
  `mart_nii_attribution_by_product` (3 tables). Tests on `tenor_years`
  uniqueness, `scenario_id` uniqueness, product accepted values.
- **IRRBB dashboard** — new section: FTP yield curve (base vs FTP),
  NII attribution waterfall, and per-product stacked bar.
- **Tests** — `tests/risk_engine/test_ftp.py` (10 property tests):
  LP=0 ⇒ FTP = base; LP > 0 ⇒ FTP > base; customer + funding = nii_total;
  behavioural value = 0 for non-NMDs; behavioural value > 0 for NMDs
  under upward curve; LP interpolates linearly; missing-column refusal.
  **30 hypothesis tests in total, all passing.**

## Phase 2.1c — Mortgage CPR + Black-76 callable bonds

**Goal:** introduce the two main optionality blocks in an ALM book —
prepayment-sensitive level-payment mortgages and embedded-call premium bonds —
both priced under the curve-calibrated HW1F model.

**Done**

- **`basel_risk_engine/behavioral/mortgage_cpr.py`** — level-payment
  amortisation schedule, refinancing-incentive CPR curve
  `CPR(r) = clip(cpr_base + β · max(0, c − r), 0, cpr_cap)`, monthly SMM,
  CPR-adjusted recursion that conserves notional exactly. `value_mortgage`
  reports CPR-adjusted PV, scheduled (no-prepay) PV, weighted-average life,
  and average CPR; `value_mortgage_book` vectorises across the book. Refi
  rates per scheduled month sourced from the curve at the remaining-term
  tenor, floored at 12 months.
- **`basel_risk_engine/valuation/black76.py`** — HW1F closed-form European
  bond option pricer (Brigo–Mercurio §3.3): integrated lognormal vol
  σ_P = σ · B(T, S) · √((1 − e^{−2aT}) / (2a)) and the standard
  P(0,S)·Φ(h) − K·P(0,T)·Φ(h − σ_P) ZBC formula. `value_callable_bond` returns
  straight PV, call value, and callable PV = straight − call. Degenerate
  cases (T_call ≥ T_mat, σ → 0) return the no-time-value intrinsic.
- **Cashflow schema** (`cashflows` raw + dbt staging + intermediate)
  extended with: `amortization_type ∈ {bullet, level}`, `term_months`,
  `is_callable`, `call_date`, `call_strike_pct`. Loan tenors split into
  short loans (30–365 days) and mortgages (5–30y, ~30% of loans); bonds
  span 1–30y with ~40% of long bonds (> 5y) flagged callable at half-life
  with strikes in [75, 88]% of par. The below-par strike approximates a
  premium coupon bond under the no-coupon synthetic model so the European
  call is meaningfully in the money.
- **EVE engine type-aware pricing** — `EVEEngine.value` now branches on
  cashflow type: bullets (existing fast vectorised path), mortgages (full
  CPR-adjusted schedule discounted on the input curve), callables (straight
  PV − Black-76 call value, HW1F only; falls back to straight pricing for
  Vasicek). BCBS 368 deterministic ΔEVE is fully optionality-aware. MC ΔEVE
  uses a flat per-payment table built once at the base curve (CPR schedule
  frozen, callable option drag held at base) so the distribution measures
  the duration component; the deterministic BCBS 368 path captures the full
  convexity from optionality.
- **`run.py`** — wires `CPRParams` end-to-end, emits two new Parquet
  outputs: `risk_mortgage_cashflows` (per-mortgage notional / contract rate /
  term / WAL / avg CPR / pv_cpr / pv_scheduled) and `risk_callable_bonds`
  (per-bond t_call / t_mat / strike_unit / integrated_vol / straight_pv /
  call_value / callable_pv). Metadata schema gains `cpr_params_json`;
  `model_version` bumped to 0.3.0.
- **Dagster** — `RISK_TABLES` gains the two new risk outputs (load via the
  existing `risk_outputs_tables` multi-asset).
- **dbt** — 2 new staging views (`stg_risk_mortgage_cashflows`,
  `stg_risk_callable_bonds`) and 3 new marts (`mart_mortgage_cf` per-mortgage,
  `mart_callable_bonds` per-bond, `mart_optionality_summary` per-scenario
  book-level rollup). Sources YAML gains `accepted_values` on
  `amortization_type` and `not_null` on `is_callable`. Schema tests added on
  `avg_cpr`, `call_value`, `call_value_pct_of_straight`.
- **IRRBB dashboard** — new "Embedded optionality" section with four KPI
  cards (mortgage count + avg CPR + WAL, mortgage CPR-PV impact, callable
  count + integrated vol, total call value as option drag), a per-mortgage
  scatter (avg CPR vs WAL, sized by notional, coloured by contract rate),
  and a top-10 callable-bond stacked bar showing the Black-76 decomposition.
- **`scripts/risk_engine.cmd`** — now also reloads risk_outputs Parquets
  into DuckDB after the engine run, eliminating the Phase 2.1b workflow
  gotcha where dbt build saw stale schemas.
- **Tests** — `tests/risk_engine/test_mortgage_cpr.py` (14 tests) covers
  level-payment formula, schedule conservation under CPR, refi-incentive
  monotonicity, CPR cap, WAL shortening with higher CPR, par-curve PV ≈ par,
  below-market-coupon PV < par, refi-rate projection geometry.
  `tests/risk_engine/test_black76.py` (14 tests) covers the HW1F integrated-vol
  formula's edge cases, ZBC degeneracy, ZBC monotone in σ and K, no-arbitrage
  bound by P(0,S), callable PV < straight PV when option in the money,
  convergence to straight PV as σ → 0 and as strike → ∞, linear scaling in
  notional. **58 hypothesis tests in total, all passing.**

**Not yet done — scope frozen 2026-05-27**

Next in queue: ALMM-style liquidity survival horizon (independent of rate
model). Explicitly dropped from the Phase 2.1 backlog: Sobol low-discrepancy
sampling, forward valuation at intermediate MC steps, MC NII attribution.
Phase 4 (FastAPI / Docker / CI) deferred indefinitely. Rationale: ship a
polished portfolio piece, not a never-finished platform.

---

## Phase 3 — Streamlit polish (planned)

- Consolidated "ALM / NII" page: NII fan chart, FTP attribution waterfall,
  behavioural-toggle comparison.
- Capital projection under MC rate paths.
- Liquidity page: survival horizon chart, multi-scenario stress comparison
  (requires Phase 2.1 survival-horizon engine).
- Model-lineage panel on every page (currently only on IRRBB).
- Replace deprecated `use_container_width` with `width=` to silence the
  Streamlit ≥ 1.57 warnings.

---

## Phase 4 — Production polish (deferred)

- FastAPI read-only service in front of DuckDB.
- Docker Compose: warehouse + Dagster + dbt-docs + Streamlit + FastAPI.
- GitHub Actions CI: ruff, mypy, pytest, dbt build against seed,
  Dagster asset-graph diff.
- Observability: Dagster run logs + a minimal Prometheus scrape.
- Optional deploy: Streamlit Cloud (UI) + Render / Fly (Dagster + DuckDB).

---

## Notes / known limitations

- Synthetic balance-sheet items are drawn independently per row, so
  `CET1 ≤ Tier1 ≤ Total Capital` doesn't always hold for the synthetic
  generator (singular test downgraded to `warn`; should be re-flipped to
  `error` once Phase 2.1 introduces realistic balance generation).
- The cashflow shape is one row = one bullet payment at `maturity_date`; a
  real ALM stack would carry an amortisation schedule per instrument.
- BCBS 368's continuous scenario function is approximated by 5-bucket
  step shifts.
- `dbt_executable` is auto-located via a Windows-aware probe in
  `basel_dagster/resources.py`; if you switch Python environments it may
  need to be re-detected.
