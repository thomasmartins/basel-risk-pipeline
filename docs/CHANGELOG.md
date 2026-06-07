# Phase log

A running record of what's been built and what remains. Phases match
[`ROADMAP.md`](ROADMAP.md). See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the
target shape of the system.

---

## Phase 0: Scaffold & DuckDB swap

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

## Phase 1: Dagster + dbt foundation

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

## Phase 2: Quant risk engine

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

## Phase 2.1a: Hull-White 1F

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

## Phase 2.1b: FTP engine + NII attribution

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

## Phase 2.1c: Mortgage CPR + Black-76 callable bonds

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

## Phase 2.1d: ALMM-style liquidity survival horizon

**Goal:** the second Basel pillar in this portfolio piece. Under a liquidity
stress scenario, how many days can the bank survive cumulative net outflows
before its counterbalancing capacity (CBC) — HQLA stock plus stressed net
cashflows — is exhausted? This is the EBA ALMM maturity-ladder metric
(template C66) and is independent of the rate model.

**Done**

- **`basel_risk_engine/liquidity/survival.py`** — `LiquidityStressParams`
  pydantic config (retail stable / unstable runoff, wholesale runoff, asset
  inflow haircut, Level1 / Level2A / Level2B HQLA haircuts).
  `LIQUIDITY_STRESS_SCENARIOS` exposes three preset scenarios aligned with
  EBA LCR conventions: `idiosyncratic` (bank-specific shock — 5% / 10% retail,
  40% wholesale runoff, baseline HQLA haircuts), `market_wide` (100%
  wholesale runoff, widened HQLA haircuts), `combined` (most aggressive
  retail + market shocks layered).
- **`compute_survival_horizon`** walks the maturity ladder day-by-day:
  initial CBC = HQLA inventory with stressed haircuts; per-day net cashflow
  = stressed inflows (with LCR-style 75% cap) − stressed outflows; running
  CBC = initial + cumulative net. Survival horizon = first day running CBC
  < 0, capped at `max_horizon_days` (365 in the runner). Cashflows past the
  horizon are excluded (they have no effect on near-term survival) rather
  than clipped to the horizon edge.
- **HQLA-vs-ladder split** — Asset-side HQLA inflows are counted in the
  t=0 CBC stock (we'd realise them by selling under stress) and *removed*
  from the maturity ladder to avoid double-counting. Non-HQLA inflows
  (loans, non-HQLA bond redemptions) stay in the ladder with a credit
  haircut.
- **`run.py`** — for each (scenario × stress), emit a row to
  `risk_survival_horizon` and the full 0..365 day ladder to
  `risk_cbc_ladder`. Metadata gains `liquidity_stress_params_json`;
  `model_version` bumped to 0.4.0. The cashflow SELECT now also pulls
  `counterparty`, `direction`, `hqla_type` (required by the survival engine).
- **Dagster** — `RISK_TABLES` gains the two new risk outputs.
- **dbt** — two new staging views (`stg_risk_survival_horizon`,
  `stg_risk_cbc_ladder`) and two new marts (`mart_survival_horizon` with a
  `severity_bucket` classification, `mart_cbc_ladder`). Tests on
  `stress_name` accepted_values, `survival_horizon_days` in [0, 365],
  `day_offset` in [0, 365].
- **Synthetic-data calibration** — `hqlatype` is now weighted (15 / 5 / 5 /
  75% for Level1 / Level2A / Level2B / None) instead of uniform, so HQLA
  stock sits at ~30M EUR per scenario rather than ~90M, putting the
  survival horizon in the realistic ALMM-stress range (7-9 months under
  combined stress).
- **Liquidity Streamlit page** — new section at the top: three KPI cards
  (one per stress) with horizon, severity bucket, initial CBC, peak deficit;
  Plotly survival-curve chart overlaying the three running-CBC trajectories
  with a zero-line and breach-point markers.
- **Tests** — `tests/risk_engine/test_survival.py` (16 hypothesis tests):
  no-outflow + any CBC → no breach; zero CBC + outflow → breach at day 1;
  CBC exactly covers outflow → no breach; higher runoff shortens survival;
  larger HQLA stock extends survival; Level1 CBC > Level2B CBC; combined
  stress ≤ each component; NMD-flagged retail deposits last longer than
  unstable retail; LCR cap binding monotonicity; deterministic; refuses on
  missing columns. **74 hypothesis tests in total, all passing.**

Sample headline (scenario 1):
    idiosyncratic   365d (no breach) · CBC 37M
    market_wide     289d breach · peak deficit -13M
    combined        266d breach · peak deficit -17M

**Not yet done — scope frozen 2026-05-27**

The Phase 2.1 substance backlog is complete (HW1F, FTP, CPR + Black-76,
ALMM survival). Explicitly dropped: Sobol low-discrepancy sampling,
forward valuation at intermediate MC steps, MC NII attribution. Phase 4
(FastAPI / Docker / CI) deferred indefinitely. Next move is Phase 3 polish
+ README / methodology write-up — and stop.

---

## Phase 3: data-quality follow-up

A re-audit of the dashboard figures found four implausibilities, all rooted
in Phase 0 / Phase 1 synthetic-data placeholders that survived into the
finished build. All four are fixed; the risk-engine logic is untouched
(it was already correct).

**Fixed**

- **NSFR** was ~0.73 in every scenario because `asf_factor` and
  `rsf_factor` were drawn uniformly from `[0.0, 0.5, 0.9]` and
  `[0.05, 0.85, 1.0]` respectively, giving E[ASF]/E[RSF] ≈ 0.74 by
  construction. Rebalanced to weighted distributions
  (ASF p=[0.10, 0.20, 0.60, 0.10] over [0.0, 0.5, 0.95, 1.0];
  RSF p=[0.20, 0.40, 0.25, 0.15] over [0.05, 0.65, 0.85, 1.0])
  → NSFR ≈ 1.18–1.24 across scenarios. Now consistent with a CRR-II
  compliant bank.
- **Capital ratios** were 159–278 % with CET1 > Tier1 in two scenarios.
  Root cause: `balance_sheet` rows were generated per (date × item)
  with a *random* `scenario_id`, each item drawn independently from
  `rng.integers(1M, 10M)`; `mart_capital_ratios` then SUM'd across
  dates, inflating both numerator and denominator but with different
  random scenario coverage and breaking the stack invariant.
  Rewrote `generate_balance_sheet` to emit one row per
  (scenario × date × item) with the stack constructed coherently
  (Tier1 = CET1 + AT1, Total = Tier1 + Tier2 with positive
  increments). Sized capital against expected per-day RWA so ratios
  land at CET1 ≈ 12 %, Tier1 ≈ 14 %, Total ≈ 17 %.
- **n_rwa** raised from 1 000 to 5 000 so each (date × scenario) bucket
  has ~14 exposures instead of ~3, taming daily-ratio noise.
- **PV01 by tenor** was random N(0, 1) noise from `irrbb.pv01`. Replaced
  `mart_pv01_profile` with a curve-aware aggregation from
  `int_cashflows_enriched`:
      `PV01_i = sign · amount · τ · DF(τ) · 0.0001`
  with DF using a flat 3 % proxy (the average level of the synthetic
  base curve). Result is monotone in tenor for the asset-heavy book
  (≈ +100 k EUR at 10y+ vs ≈ −2 k EUR at 0-1y).
- **Removed the dead `irrbb` table** entirely — `stg_irrbb`,
  `generate_irrbb`, `n_irrbb`, the source declaration, the dbt schema
  test, the `get_irrbb` query, and the Home-page "Show Raw IRRBB Data"
  expander. The risk engine never used it; nothing references it now.
- **`assert_capital_ratios_ordered` flipped from `warn` to `error`** —
  the new generator enforces the stack invariant by construction so the
  test should be hard-failing going forward.
- **Supervisory outlier test** now flags BREACH (ratio ~ 30 %) under
  Parallel-down across all scenarios. This is a genuine finding, not a
  bug: with Tier1 at a realistic ~ 43M EUR (not the inflated 127M EUR
  from the previous SUM-across-dates artefact), the worst-case ΔEVE of
  ~ 15M EUR exceeds the EBA 15 % threshold. The IRRBB page already
  surfaces this via the colored ratio metric.

**Cleanup**

- Untracked `.claude/settings.local.json` and `dbt_project/.user.yml`
  (personal / per-user state; added to `.gitignore`).
- Removed `.devcontainer/` (Phase-0-era Codespaces stub, unused).
- Removed `data/seed/*.parquet` (8 files, stale schema, no consumer).
- Two hypothesis edge cases tightened: `test_scheduled_principal_sums_to_notional`
  tolerates fp drift at sub-1bp rates; `test_bond_price_monotone_in_tau`
  `assume`s θ − σ²/(2κ²) > 0 to avoid Vasicek's negative-long-run-yield
  pathology where bond prices are U-shaped.

74 hypothesis tests still passing. **dbt now 157 PASS / 0 WARN / 0 ERROR**
(was 163/1/0 — the 6 missing tests are the now-deleted `stg_irrbb` source
+ staging tests).

---

## Phase 3: Streamlit polish (planned)

- Consolidated "ALM / NII" page: NII fan chart, FTP attribution waterfall,
  behavioural-toggle comparison.
- Capital projection under MC rate paths.
- Liquidity page: survival horizon chart, multi-scenario stress comparison
  (requires Phase 2.1 survival-horizon engine).
- Model-lineage panel on every page (currently only on IRRBB).
- Replace deprecated `use_container_width` with `width=` to silence the
  Streamlit ≥ 1.57 warnings.

---

## Phase 4: Production polish (deferred)

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
