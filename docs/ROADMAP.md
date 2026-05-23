# Roadmap — Basel ALM Risk Engine

Five phases. Phase 4 is deferred and can be skipped if time-constrained. Effort estimates assume ~half-day blocks (so "5 days" ≈ 5 focused half-days).

## Phase 0 — Scaffold & DuckDB swap (1–2 days)

**Goal:** repo restructured, Postgres dropped, DuckDB + Parquet running, existing Streamlit pages still work.

Deliverables:
- `pyproject.toml` migration (uv or poetry; recommend uv for speed)
- New folder layout per ARCHITECTURE.md
- `packages/ingestion/`: Parquet generators (port `src/generate_data.py` to write Parquet, not insert into Postgres)
- `packages/basel_common/`: shared pydantic models, date conventions
- `data/seed/` checked-in tiny sample (a few hundred rows total) for smoke tests
- `dashboard/` and `src/compute.py` repointed at DuckDB via a thin `connection.py` helper (drop SQLAlchemy from runtime path; keep models.py around or convert to pydantic)
- `.streamlit/secrets.toml` becomes a simple `[duckdb] path = "data/warehouse.duckdb"`
- Old Postgres bits archived in a `legacy/` folder or removed in a single commit

Exit criteria: `streamlit run dashboard/Home.py` works against DuckDB with no Postgres dependency. All five pages render.

## Phase 1 — Dagster + dbt foundation (3–5 days)

**Goal:** every transformation is either a dbt model or a Dagster asset, all visible in the Dagster UI.

Deliverables:
- `dbt_project/` with `dbt-duckdb` adapter
- Staging models: 1:1 typed views over raw tables (positions, cashflows, market curves, RWA, IRRBB inputs, balance sheet, scenarios, params)
- Intermediate models: joined views, derived fields (e.g. `behavioral_maturity`, `repricing_bucket`, `hqla_tier_post_haircut`)
- Mart models:
  - `mart_lcr`, `mart_nsfr`, `mart_almm_buckets`
  - `mart_pv01`, `mart_repricing_gap`
  - `mart_rwa`, `mart_capital_ratios`
- dbt tests: unique/not-null on PKs, accepted_values on enums (HQLA tier, asset class, scenario), Basel-specific custom tests (ASF/RSF factors in [0,1], LCR inflow cap ≤ 75% outflows when materialized)
- Dagster code location with: ingestion assets, dbt assets via `@dbt_assets`, schedules (daily for ingestion, monthly for full rebuild), one job per mart group
- Existing `src/compute.py` pure-SQL aggregations migrate into dbt; the Python module shrinks to just engine glue

Exit criteria: `dagster dev` shows the asset graph end-to-end; `dbt build` runs clean with passing tests; Streamlit reads from marts.

## Phase 2 — Risk engine substance (5–7 days, the differentiator)

**Goal:** real quant content. ALM/treasury reviewer should see model depth, not dashboards on hardcoded numbers.

Deliverables (in `packages/risk_engine/`):

**Rate models**
- Vasicek 1F: simulation + MLE/OLS calibration from synthetic short-rate history
- Hull-White 1F (extended Vasicek): fitted to today's curve, used for arbitrage-free MC paths
- Antithetic + Sobol sampling, configurable horizon and path count

**Behavioral overlays**
- NMD repricing decay curve (parametric, params live in `dbt seed`): core stable %, runoff half-life, beta to policy rate
- Mortgage prepayment (CPR-style, rate-sensitive)
- Embedded floor/cap optionality on loans/deposits (Black-76 closed-form for first cut)

**EVE engine**
- Reprices full balance sheet under MC curve paths → EVE distribution
- BCBS 368 six prescribed shock scenarios (parallel ±, short ±, steepener, flattener) computed deterministically
- EBA supervisory outlier test: |∆EVE| / Tier1 vs 15% — but computed from the **distribution** (e.g. 99% VaR of ∆EVE), not just point estimate. Both reported.

**NII engine**
- Forward NII paths under MC curves + behavioral repricing
- 12m, 24m, 36m horizons
- Decomposition into rate, volume, margin contributions

**FTP engine**
- Tenor-bucketed cost-of-funds curve (matched-funded transfer pricing)
- NII attribution: customer margin vs treasury margin vs FTP residual

**Liquidity stress**
- ALMM-style survival horizon: cashflow projection under idiosyncratic, market-wide, combined stresses
- Outputs: days-to-counterbalancing-exhaustion, peak intraday deficit, counterbalancing capacity timeline

**Plumbing**
- All engines: pydantic input/output models, hypothesis-based property tests (e.g. EVE monotone in parallel shock for a non-optionality book, NII decomposition sums to total)
- All outputs written as Parquet to `data/risk_outputs/`, then read by dbt models in `models/marts/irrbb/`, `marts/alm/`, `marts/liquidity/`
- Each engine is a Dagster asset, partitioned by `valuation_date`, with `AssetCheck` for sanity (EVE finite, ratios in plausible range)

Exit criteria: engines run end-to-end as Dagster jobs; outputs flow back through dbt; property tests pass; one full backfill of 24 months synthetic history completes.

## Phase 3 — Streamlit upgrade (2–3 days)

**Goal:** UI matches the new substance. Each page now tells a quant story, not a metric story.

Per page:
- **Liquidity:** add survival horizon chart, multi-scenario stress comparison, counterbalancing capacity timeline. Keep LCR/NSFR waterfalls.
- **IRRBB:** EVE distribution histogram (MC), not just a point. Six BCBS 368 scenarios as a bar chart. Behavioral-on/off toggle showing impact of NMD assumptions. Supervisory outlier test panel.
- **ALM / NII (new):** NII path fan chart (MC), forward NII decomposition (rate/volume/margin), FTP attribution waterfall.
- **Capital:** keep current; add capital projection under MC rate paths (CET1 trajectory distribution).
- **Stress (consolidated):** combined stress scenarios (rate + liquidity + RWA), ICAAP-flavored capital adequacy view.

Cross-cutting:
- **Model lineage panel** on each page: model name, version, calibration date, key assumptions surfaced from `risk_outputs.model_metadata`.
- **Scenario selector** retains the session_state pattern from current code.
- Replace any remaining `@st.cache_data` on engine results with materialized DuckDB reads — the engine outputs are pre-computed by Dagster, not per-request.

Exit criteria: all pages render; lineage panel shows real model_version strings; toggling behavioral assumptions visibly moves EVE/NII.

## Phase 4 — Production polish (deferred)

Only when phases 0–3 are solid:
- FastAPI service in front of DuckDB (read-only) — Streamlit becomes one client of an API
- Docker Compose: warehouse + Dagster + dbt-docs + Streamlit + FastAPI
- GitHub Actions CI: ruff, mypy, pytest, dbt build (against seed), Dagster asset graph diff
- Optional deploy: Streamlit Cloud (frontend) + Render/Fly (Dagster + API + DuckDB volume), or a Hetzner box
- Observability: Dagster's built-in run logs + a minimal Prometheus scrape

## Sequencing dependencies

- Phase 0 unblocks everything.
- Phase 1 can technically proceed without 0 but the path-of-least-pain is 0 → 1.
- Phase 2 depends on Phase 1's mart layer (engines read from `mart_*`).
- Phase 3 depends on Phase 2 outputs.
- Phase 4 can start in parallel with 3 but is best last.

## Open questions / decisions to revisit

- **Synthetic data realism.** Do we generate a small synthetic market history (e.g. 5y of daily curves) and a corresponding balance sheet evolution, or stay with a single snapshot? MC engines need history for calibration → recommend a generated history.
- **Calibration cadence.** Daily recalibration of Vasicek/HW1F is realistic but adds compute. Probably weekly recalibration + daily MC re-run is the right tradeoff for a demo.
- **Polars vs pandas in the engine.** Polars is faster on Parquet and aligns better with DuckDB. Recommend polars in `risk_engine`, leave pandas for the dashboard layer where Streamlit examples assume it.
- **NMD integration timing.** Keep the NMD project standalone for now; revisit in a Phase 5 once both have shipped.
- **GPU.** Skip. CPU MC is fine at this scale; bringing in JAX/CuPy is a distraction.

## Effort total

~12–17 focused half-days for phases 0–3. Phase 4 adds ~5 more. Realistic calendar: 4–6 weeks part-time alongside other work.
