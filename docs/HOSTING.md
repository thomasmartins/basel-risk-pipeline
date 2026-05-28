# Hosting

The project is designed to be hosted as three independent web surfaces.
Tier 1 (described here) is the **zero-cost portfolio setup**: Streamlit
Community Cloud for the dashboard, GitHub Pages for the dbt docs, and a static
screenshot + Loom for the Dagster asset graph.

| Surface | Hosted on | Cost | URL pattern |
|---|---|---|---|
| Streamlit dashboard (`dashboard/Home.py`) | Streamlit Community Cloud | Free | `https://<app>.streamlit.app` |
| dbt docs (`dbt_project/target/`) | GitHub Pages (`gh-pages` branch) | Free | `https://<user>.github.io/<repo>/` |
| Dagster asset graph | Static screenshot embedded in dashboard + Loom | Free | n/a |

The DuckDB warehouse (`data/warehouse.duckdb`) and the raw / risk-output
parquets are **committed to the repo** (~30 MB total). Both hosts therefore
boot against a real, pre-built database — no synthetic-data regeneration on
cold start, sub-second first paint.

---

## Streamlit Community Cloud — the dashboard

### One-time setup

1. Push the repo to GitHub as a **public** repository.
2. Sign in to <https://streamlit.io/cloud> with the same GitHub account.
3. Click **New app** and fill in:
   - **Repository:** `<user>/basel-risk-pipeline`
   - **Branch:** `main`
   - **Main file path:** `dashboard/Home.py`
   - **Python version:** 3.11 (matches `runtime.txt`)
4. Click **Deploy**. First build takes ~3 minutes; subsequent pushes redeploy in
   ~30 seconds.

### What the cloud reads

- `requirements.txt` (repo root) — pip-installable runtime deps. Intentionally
  slim (no dbt, no Dagster, no Polars, no SciPy) so cold start is fast.
- `runtime.txt` — pins Python to 3.11.
- `data/warehouse.duckdb` — single-file DuckDB warehouse, opened in read-only
  mode by `basel_common.connection.duckdb_connect`. No env vars needed —
  `warehouse_path()` defaults to `<repo_root>/data/warehouse.duckdb`.
- `.streamlit/config.toml` (optional) — theme / server config. Not required.

### Refreshing the warehouse

The hosted dashboard reflects whatever warehouse is committed on `main`. To
refresh:

```cmd
scripts/ingest.cmd            REM regenerate synthetic parquets + load
scripts/risk_engine.cmd        REM (re)compute risk outputs
scripts/dbt.cmd build          REM rebuild marts
git add data/ && git commit -m "Refresh warehouse" && git push
```

Streamlit Cloud picks up the new warehouse on the next deploy
(~30 s after push).

### Secrets

`.streamlit/secrets.toml` is gitignored. The dashboard reads no secrets in the
default configuration. If you later need to override the warehouse path (e.g.
to point at an S3-backed DuckDB), add to Streamlit Cloud's **Secrets** UI:

```toml
[duckdb]
path = "s3://my-bucket/warehouse.duckdb"
```

`basel_common.connection.warehouse_path()` reads this before falling back to
the env var or default.

---

## GitHub Pages — dbt docs

A GitHub Actions workflow (`.github/workflows/dbt-docs.yml`) rebuilds the dbt
docs static site on every push to `main` that touches `dbt_project/**` or
`data/warehouse.duckdb`, and publishes the result to the `gh-pages` branch.

### One-time setup

1. After the workflow runs once (push any change to `dbt_project/**`), go to
   the repo's **Settings → Pages**.
2. Set **Source** to `Deploy from a branch`.
3. Set **Branch** to `gh-pages`, folder `/ (root)`.
4. Save. The docs go live at `https://<user>.github.io/<repo>/` within a
   minute.

### How the workflow runs

- Checks out the repo (including the committed warehouse).
- Installs `dbt-core` + `dbt-duckdb`.
- Sets `BASEL_WAREHOUSE_PATH` so dbt finds the committed warehouse.
- Runs `dbt deps` + `dbt docs generate` from `dbt_project/`.
- Publishes `dbt_project/target/` to `gh-pages` via `peaceiris/actions-gh-pages`.

The workflow uses `force_orphan: true` so the `gh-pages` branch is rewritten
on each publish (no accumulating history). Total runtime ~90 seconds.

### Linking from the dashboard

The Liquidity / IRRBB / RWA pages already render a "Model lineage" expander
that summarises the model graph. To deep-link to the live dbt docs, add a
caption in `src/lineage.py`:

```python
st.caption(
    "Full model graph + column-level lineage: "
    "[dbt docs](https://<user>.github.io/<repo>/)"
)
```

---

## Dagster asset graph — static showcase

Hosting Dagster live (Fly.io / Railway) is **Tier 2** and adds ~$5/month. For
the portfolio piece, a static screenshot + a 30-second Loom is enough:

1. Run `scripts/dagster_dev.cmd` locally; open <http://localhost:3000>.
2. Screenshot the asset graph at the **Assets** tab (light mode for readability).
3. Record a Loom showing a `dagster job execute` run materialising the
   `risk_outputs_tables` multi-asset.
4. Drop both into `docs/screenshots/` and reference them from `README.md` and
   the upcoming blog post.

The orchestration story is the asset graph, not the live trigger button. A
screenshot conveys this and avoids the cost of a long-running daemon.

---

## Tier 2 — live Dagster (when you want it)

Add a `Dockerfile` that bundles the warehouse + Dagster code location, then
deploy to Fly.io or Railway:

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[dev]"
ENV BASEL_WAREHOUSE_PATH=/app/data/warehouse.duckdb
ENV DAGSTER_HOME=/app/.dagster_home
EXPOSE 3000
CMD ["dagster-webserver", "-h", "0.0.0.0", "-p", "3000", "-w", "basel_dagster/workspace.yaml"]
```

Read-only mode (`--read-only`) prevents random visitors from triggering MC
runs. Link from the dashboard's lineage panel.

---

## Tier 3 — production-grade (overkill for portfolio)

- Cloud Run for each service (Streamlit, Dagster UI, nginx-served dbt docs).
- DuckDB warehouse in GCS, hydrated into the container on cold start, or
  replaced with [MotherDuck](https://motherduck.com) for hosted DuckDB.
- Cloud Build rebuilds containers on `git push`.
- Cost: $10-30/month depending on traffic and request volume.

Not recommended unless you're using this as a learning exercise for serverless
deployment patterns.
