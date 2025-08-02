import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import streamlit as st

db_config = st.secrets["postgres"]

engine = create_engine(
    f"postgresql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
)

# ===================================================
# ✅ Params Table Fetcher
# ===================================================
def get_params():
    """
    Returns params table as a dictionary {key: value}
    """
    df = pd.read_sql("SELECT * FROM params", con=engine)
    params = pd.Series(df.value.values, index=df.key).to_dict()
    return params


# ===================================================
# ✅ Cashflows Query
# ===================================================
def get_cashflows(start_date=None, end_date=None, scenario_id=None):
    """
    Fetch cashflows with optional date range and scenario filter.
    Returns a pandas DataFrame.
    """
    query = """
    SELECT * FROM cashflows
    WHERE (:start IS NULL OR date >= :start)
    AND (:end IS NULL OR date <= :end)
    AND (:scenario IS NULL OR scenario_id = :scenario)
    """
    df = pd.read_sql(
        text(query),
        con=engine,
        params={'start': start_date, 'end': end_date, 'scenario': scenario_id}
    )
    return df


# ===================================================
# ✅ RWA Query
# ===================================================
def get_rwa(start_date=None, end_date=None, scenario_id=None):
    """
    Fetch RWA exposures with optional date and scenario filters.
    """
    query = """
    SELECT * FROM rwa
    WHERE (:start IS NULL OR date >= :start)
    AND (:end IS NULL OR date <= :end)
    AND (:scenario IS NULL OR scenario_id = :scenario)
    """
    df = pd.read_sql(
        text(query),
        con=engine,
        params={'start': start_date, 'end': end_date, 'scenario': scenario_id}
    )
    return df


# ===================================================
# ✅ IRRBB Query
# ===================================================
def get_irrbb(scenario_id=None):
    """
    Fetch IRRBB instruments with optional scenario filter.
    """
    query = """
    SELECT * FROM irrbb
    WHERE (:scenario IS NULL OR scenario_id = :scenario)
    """
    df = pd.read_sql(
        text(query),
        con=engine,
        params={'scenario': scenario_id}
    )
    return df


# ===================================================
# ✅ Balance Sheet Query
# ===================================================
def get_balance_sheet(scenario_id=None):
    """
    Fetch balance sheet items with optional scenario filter.
    """
    query = """
    SELECT * FROM balance_sheet
    WHERE (:scenario IS NULL OR scenario_id = :scenario)
    """
    df = pd.read_sql(
        text(query),
        con=engine,
        params={'scenario': scenario_id}
    )
    return df


# ===================================================
# ✅ Scenarios Query
# ===================================================
def get_scenarios():
    """
    Fetch all scenarios.
    """
    df = pd.read_sql("SELECT * FROM scenarios", con=engine)
    return df


# ===================================================
# ✅ Example Run
# ===================================================
if __name__ == "__main__":
    # Test cashflows
    cashflows = get_cashflows(start_date='2024-01-01', end_date='2024-03-31')
    print(cashflows.head())

    # Test params
    params = get_params()
    print(params)

    # Test scenarios
    scenarios = get_scenarios()
    print(scenarios)
