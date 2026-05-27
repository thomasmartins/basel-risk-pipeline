# Basel ALM Risk Pipeline

A production-shaped IRRBB + liquidity + capital pipeline for a bank's
Asset-Liability Management function. Synthetic feeds → DuckDB warehouse →
dbt transformations → quant Python risk engine → orchestrated Dagster
DAG → Streamlit dashboard. Curve-calibrated short-rate model, prepayment
and callable-bond optionality, FTP attribution, and an ALMM-style
liquidity survival horizon.

> **Scope.** ALM under BCBS 368 (IRRBB), EBA Guidelines, LCR Delegated
> Act, NSFR Regulation, ALMM template C66. **Market-risk capital
> (FRTB) is intentionally out of scope** to keep clear separation from
> my current professional engagements.

For the modelling write-up — what each engine actually does, the
formulas, the calibration choices, and the honest limitations — see
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
For the build log see [`docs/CHANGELOG.md`](docs/CHANGELOG.md).
For the architecture diagram see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Stack

| Layer            | Tool                          | Why                                                    |
|------------------|-------------------------------|--------------------------------------------------------|
| Storage          | Parquet (`data/raw/`)         | Columnar, dbt-native, zero-setup                       |
| Warehouse        | DuckDB                        | Single-file analytical store; embedded — no server     |
| Transformations  | dbt-duckdb + dbt-utils        | Lineage, tests, conformed marts                        |
| Risk engine      | Python (numpy, polars, scipy) | Rate models, behavioural overlays, EVE/NII, FTP, ALMM  |
| Orchestration    | Dagster                       | Asset-centric DAG, partitioned, schedules              |
| UI               | Streamlit                     | Interactive dashboard over the marts                   |

## What's in the box

**Rate model — Hull-White 1F (curve-calibrated; Vasicek selectable)**
- `dr_t = (θ(t) − a·r_t) dt + σ dW_t` with θ(t) bootstrapped from the
  initial forward curve so `P_model(0,τ; r0) = P_market(0,τ)` to machine
  precision (curve-fit residual ~1e-16).
- Exact-discretisation MC, antithetic variates.
- (a, σ) calibrated by closed-form OLS on the AR(1) representation;
  half-life and log-likelihood reported.
- `--model vasicek` flag still works for comparison.

**EVE engine — BCBS 368 + MC distribution**
- Type-aware pricing: bullets vectorised, mortgages via CPR-adjusted
  schedule, callable bonds via Black-76 closed form.
- Six BCBS 368 §132 prescribed shocks (parallel ±, short ±, steepener,
  flattener) computed deterministically.
- MC ΔEVE distribution at the 1-year forward horizon under HW1F paths.
- EBA supervisory outlier test (`|ΔEVE_worst| / Tier1 ≤ 15%`) reporting
  both the deterministic worst and a distributional `|ΔEVE|₉₉`.

**Behavioural overlays**
- **NMD repricing** — parametric `stable_core_pct` × `core_behavioral_maturity_yrs`
  blend, applied to short-tenor deposits.
- **Mortgage CPR** — refinancing-incentive
  `CPR(r) = clip(cpr_base + β·max(0, c − r), 0, cpr_cap)`. CPR-adjusted
  level-payment schedule conserves notional exactly.
- **Black-76 callable bonds** — Brigo–Mercurio HW1F closed form with
  integrated lognormal vol `σ_P = σ·B(T,S)·√((1−e^{−2aT})/(2a))`.

**NII engine — MC paths + behavioural repricing**
- 12 / 24 / 36-month horizons.
- Behavioural-aware repricing gap (deposit β reduces NMD sensitivity).
- Static FTP attribution decomposes book NII into customer margin,
  funding margin, and NMD behavioural value.

**FTP engine — matched-funded transfer pricing**
- Internal FTP curve = wholesale base + tenor-dependent liquidity
  premium (linear interp on bps add-ons).
- Per-row attribution: `customer_margin = sign·N·(c − FTP_b)`,
  `funding_margin = sign·N·(FTP_b − r_f)`, behavioural value =
  `sign·N·(FTP_c − FTP_b)` on NMDs.

**Liquidity — LCR / NSFR + ALMM survival horizon**
- LCR with HQLA tiering, post-haircut composition, EBA 75% inflow cap.
- NSFR with ASF / RSF decomposition by product.
- Cashflow gap heatmap (signed, capped) across maturity buckets.
- **ALMM-style survival horizon** under three preset stresses
  (idiosyncratic, market-wide, combined) — daily maturity ladder with
  stressed runoff factors and HQLA haircuts, walked forward until
  counterbalancing capacity goes negative.

**Capital**
- CET1 / Tier1 / Total Capital ratios per scenario, daily.
- RWA breakdown by approach (STD / IRB) and asset class.
- IRB output floor flag (72.5% of STD RWA).

## Architecture

```
Parquet feeds (data/raw/)
         │
         ▼
    DuckDB warehouse
         │
         ├──▶ dbt staging ─▶ intermediate ─▶ marts ─┐
         │                                          │
         ▼                                          ▼
    Python risk engine                          Streamlit
    (curve calibration, MC,                     (reads only marts;
     EVE / NII / CPR / Black-76 / ALMM)         no engine work)
         │
         ▼
    Parquet outputs (data/risk_outputs/)
         │
         ▼
    Reloaded into DuckDB ──▶ dbt staging ──▶ marts ──▶ Streamlit
```

Every node is a Dagster asset; the full DAG renders at
`http://localhost:3000` via `scripts/dagster_dev.cmd`.

## Quick start (Windows + Anaconda)

```
scripts\ingest.cmd          # generate synthetic Parquet + load into DuckDB
scripts\risk_engine.cmd     # calibrate HW1F + run all engines + reload outputs
scripts\dbt.cmd build       # transform + 163 dbt tests
scripts\run.cmd             # Streamlit at :8501
scripts\dagster_dev.cmd     # Dagster UI at :3000 (optional)
```

The risk engine accepts `--model {hull_white,vasicek}` (default
`hull_white`) and `--n-paths N` (default 2000).

## Layout

```
basel_common/         Shared pydantic types, enums, DuckDB helper
basel_ingestion/      Synthetic Parquet generator + DuckDB loader
basel_risk_engine/
  rate_models/        Vasicek 1F + Hull-White 1F + ShortRateModel protocol
  behavioral/         NMD overlay + mortgage CPR
  valuation/          YieldCurve, EVE, NII, Black-76
  ftp/                Internal FTP curve + NII attribution
  liquidity/          ALMM survival horizon
  run.py              End-to-end runner — calibrate, MC, write 13 Parquets
basel_dagster/        Code location: ingestion + risk + dbt assets
dbt_project/          staging / intermediate / marts; dbt_utils tests
dashboard/            5 Streamlit pages, all reading marts only
src/                  Marts client + scenario sidebar + lineage panel
data/                 raw/ Parquet, seed/ small sample, risk_outputs/, warehouse.duckdb
docs/                 ARCHITECTURE.md, ROADMAP.md, CHANGELOG.md, METHODOLOGY.md
scripts/              One .cmd per workflow entrypoint
legacy/               Postgres / SQLAlchemy v1 artefacts (archived)
```

## Tests

- **dbt:** 163 tests covering unique / not-null / accepted-values /
  relationships / range constraints on every staging model, plus custom
  singular tests for the LCR inflow cap and capital-stack ordering.
- **Risk engine:** 74 hypothesis-based property tests in
  `tests/risk_engine/` covering rate-model MC convergence, calibration
  roundtrip, BCBS 368 scenario completeness, supervisory threshold logic,
  CPR schedule conservation, Black-76 monotonicity, ALMM survival
  stress ordering, and more.

## Regulatory references

- BCBS 368 — *Interest rate risk in the banking book* (April 2016)
- EBA/GL/2022/14 — *Guidelines on IRRBB and CSRBB management*
- Commission Delegated Regulation (EU) 2015/61 — LCR
- Commission Delegated Regulation (EU) 2017/208 — LCR inflow cap
- Regulation (EU) 2019/876 (CRR II) — NSFR
- EBA ALMM Reporting Framework (Annex III, template C66) — maturity ladder

## Author

[Thomas Martins](https://thomasmartins.github.io). Built as a public ALM
portfolio piece. **Market risk / FRTB is intentionally out of scope.**

## Licence

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
