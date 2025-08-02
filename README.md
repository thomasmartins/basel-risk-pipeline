# Basel III Risk Data Pipeline & Dashboard

## 📊 Project Overview
This project is a fully operational **Basel III risk reporting pipeline and interactive dashboard**, designed to simulate how financial institutions monitor and report regulatory metrics under the Basel III framework.

It covers liquidity risk, capital adequacy, and interest rate risk (IRRBB), with full integration between **data ingestion**, **SQL-backed storage**, **risk metric computation**, and a **Streamlit UI** for scenario-driven exploration.

---

## 🔧 Features

- 💧 **Liquidity Risk**
  - LCR & NSFR calculations
  - HQLA tiering, inflow/outflow caps, ASF/RSF weightings
- 🧮 **Capital Adequacy**
  - CET1 and Total Capital Ratios
  - RWA decomposition and visualization
- 📈 **Interest Rate Risk (IRRBB)**
  - PV01 profile by tenor
  - ∆EVE and ∆NII simulation under parallel rate shocks
- 🔍 **Scenario Filtering**
  - Toggle between baseline and stress scenarios
  - Scenario-specific balance sheet snapshots
- 🔁 **ETL Pipelines**
  - SQL storage → Python integration (SQLAlchemy) → real-time dashboard aggregation
- 🗄️ **Database-Backed**
  - PostgreSQL schema aligned with ECB/EBA Basel III templates

---

## 🛠️ Tech Stack

| Layer          | Tools                     |
|----------------|----------------------------|
| Backend DB     | PostgreSQL, SQLAlchemy     |
| Data Processing| pandas, numpy              |
| Visualization  | Streamlit (interactive UI) |
| Structure      | Modular Python (ETL + Dashboard) |

---

## ▶️ How to Run

1. **Start your PostgreSQL instance**
   - Use a local DB or a remotely-hosted one
2. **Create schema**
   - Execute SQL files in the `/sql/` directory
3. **Add credentials**
   - In local use: configure `.streamlit/secrets.toml` with DB info
4. **Launch app**
   - streamlit run dashboard/Home.py

## 📁 Project Structure

├── sql/                 # SQL schema (tables + sample data)
├── src/                 # ETL and transformation scripts
├── dashboard/           # Streamlit app and UI logic
├── basel_pipeline/      # Custom local Python package for risk logic
├── .streamlit/          # Local secrets file (excluded from repo)
├── requirements.txt     # Dependencies
├── setup.py             # Local module install config
├── README.md            # You're here!

## 👤 Author

Thomas Martins
thomasmartins.github.io

## 📝 License

This project is open-source and released under the GNU General Public License.