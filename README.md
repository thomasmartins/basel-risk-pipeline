# Basel ALM Risk Pipeline

End-to-end ALM / IRRBB / Liquidity risk pipeline with a Streamlit dashboard,
modelled on how a treasury function actually ships regulatory metrics:
synthetic data feeds → DuckDB warehouse → dbt transformations → quant risk
engine → orchestrated, tested, lineage-tracked.

> **Scope.** ALM/treasury risk under BCBS 368 (IRRBB), EBA Guidelines, LCR
> Delegated Act, NSFR Regulation. Market risk capital (FRTB) is deliberately
> out of scope.

## Stack

| Layer            | Tool                          | Role                                                   |
|------------------|-------------------------------|--------------------------------------------------------|
| Storage          | Parquet (`data/raw/`)         | Synthetic market & balance-sheet feeds                 |
| Warehouse        | DuckDB                        | Single-file analytical store; read-only from Streamlit |
| Transformations  | dbt-duckdb                    | Lineage, tests, conformed marts                        |
| Risk engine      | Python (numpy, polars, scipy) | Rate models, behavioral overlays, EVE/NII, FTP         |
| Orchestration    | Dagster                       | Asset-centric DAG with daily schedule                  |
| UI               | Streamlit                     | Interactive dashboard over the marts                   |

## Architecture

```
Parquet feeds
     │
     ▼
DuckDB ─► dbt staging ─► dbt intermediate ─► dbt marts ─► Streamlit
                                       │
                                       ▼
                              Risk engine (Python)
                                       │
                                       ▼
                              Parquet outputs ─► dbt marts ─► Streamlit
```

Every step is a Dagster asset; the full DAG renders in the Dagster UI.
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the detailed view
and [`docs/ROADMAP.md`](docs/ROADMAP.md) for the phased plan.

## What's in the box

**Liquidity**

- LCR with HQLA tiering, post-haircut composition, EBA inflow cap (75%)
- NSFR with ASF / RSF decomposition by product
- Cashflow gap heatmap (signed, capped) across maturity buckets
- Daily LCR / NSFR timeseries

**IRRBB** *(Phase 2 — quant content)*

- Calibrated Vasicek 1F short-rate model
- Monte Carlo curve paths → EVE distribution (not just a point estimate)
- BCBS 368 six prescribed shock scenarios (parallel ±, short ±, steepener, flattener)
- EBA supervisory outlier test: `|∆EVE| / Tier1 ≤ 15%` — point and distributional
- Behavioral overlay: NMD repricing decay (parametric stable %, runoff half-life)

**Capital**

- CET1 / Tier1 / Total Capital ratios per scenario, daily
- RWA breakdown by approach (STD / IRB) and asset class
- IRB output floor (72.5% of STD RWA) flag
- Capital ratios under RWA stress

**ALM / NII**

- Forward NII paths under MC curves + behavioral repricing
- ∆NII per EBA scenario
- (FTP engine + liquidity survival horizon: planned Phase 2 follow-up)

## Quick start (Windows + Anaconda)

1. Install deps into the active Python env:

   ```
   python -m pip install -e .
   ```

   (Or skip the install — the `scripts/run.cmd` helper sets `PYTHONPATH`
   directly and only needs the packaged third-party deps from `pyproject.toml`.)

2. Generate synthetic data and load it into DuckDB:

   ```
   python -m basel_ingestion.generate
   python -m basel_ingestion.load
   ```

3. Build the dbt models:

   ```
   scripts\dbt.cmd build
   ```

4. Launch the dashboard:

   ```
   scripts\run.cmd
   ```

5. (Optional) Inspect the asset graph in Dagster:

   ```
   scripts\dagster_dev.cmd
   ```

   Browse to `http://localhost:3000`.

## Layout

```
basel_common/         Shared pydantic types, enums, DuckDB helper
basel_ingestion/      Synthetic Parquet generator + DuckDB loader
basel_risk_engine/    Rate models, behavioral overlays, EVE/NII, FTP
basel_dagster/        Dagster code location: assets, jobs, schedules
dbt_project/          dbt-duckdb models: staging / intermediate / marts
dashboard/            Streamlit app, reads marts via DuckDB
src/                  Compute glue between Streamlit and marts
data/                 raw/ Parquet, seed/ small sample, warehouse.duckdb
docs/                 Architecture, roadmap
scripts/              run.cmd, dbt.cmd, dagster_dev.cmd
legacy/               Postgres / SQLAlchemy v1 artefacts (archived)
```

## Tests

- **dbt:** unique, not-null, accepted-values, relationships on every staging
  model; range constraints on factors; custom singular tests for the LCR
  inflow cap and capital-stack ordering. Currently 89 tests.
- **Risk engine:** pydantic input/output validation + hypothesis-based
  property tests (monotonicity of EVE in parallel shock, NII decomposition
  conservation) — see `tests/risk_engine/`.

## Domain & regulatory references

- BCBS 368 — *Interest rate risk in the banking book* (April 2016)
- EBA/GL/2022/14 — *Guidelines on IRRBB and CSRBB management*
- Commission Delegated Regulation (EU) 2015/61 — LCR
- Commission Delegated Regulation (EU) 2017/208 — Inflow cap
- Regulation (EU) 2019/876 (CRR II) — NSFR

## Author

[Thomas Martins](https://thomasmartins.github.io) — built as a public ALM
risk-engine portfolio piece. **Market risk / FRTB is intentionally out of
scope** to keep clear separation from my current professional engagements.

## Licence

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
