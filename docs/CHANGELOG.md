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

## Phase 2.1 — Risk-engine follow-ups (planned, not started)

- **Hull-White 1F** (extended Vasicek) calibrated to today's curve —
  arbitrage-free MC paths.
- **FTP engine** — matched-funded transfer-pricing curve; NII attribution
  into customer margin / treasury margin / FTP residual.
- **ALMM-style liquidity survival horizon** — cashflow projection under
  idiosyncratic / market-wide / combined stresses, counterbalancing-capacity
  timeline, days-to-exhaustion metric.
- Mortgage CPR (rate-sensitive prepayment) and embedded options
  (Black-76 closed form).
- Sobol low-discrepancy sampling for MC.
- Forward valuation at intermediate MC steps (currently t=0 and t=1y only).

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
