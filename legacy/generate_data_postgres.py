import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Create connection string
connection_string = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Create engine
engine = create_engine(connection_string)

# =======================================================
# ✅ Common Date Range
# =======================================================
dates = pd.date_range(start='2024-01-01', periods=90, freq='D')

# =======================================================
# ✅ Generate Scenarios
# =======================================================
scenarios = pd.DataFrame({
    'name': ['Baseline', 'ECB Stress', 'Liquidity Shock', 'Interest Rate Shock'],
    'description': [
        'Normal conditions',
        'Comprehensive ECB stress scenario',
        '30% wholesale funding withdrawal',
        'Parallel +200bps rate shift'
    ],
    'liquidity_shock': [0, 20, 30, 0],
    'ir_shift': [0, 100, 0, 200],
    'credit_shock': [0, 50, 0, 0]
})

scenarios.to_sql('scenarios', con=engine, if_exists='append', index=False)
print("✅ Scenarios table populated.")

# Fetch scenario IDs to use as foreign keys
scenario_ids = pd.read_sql('SELECT id FROM scenarios', con=engine)['id'].tolist()

# =======================================================
# ✅ Generate Cashflows (LCR + NSFR)
# =======================================================
base_dates = np.random.choice(dates, 5000)
maturity_offsets = pd.to_timedelta(np.random.randint(30, 365, 5000), unit='D')
maturity_dates = base_dates + maturity_offsets

cashflows = pd.DataFrame({
    'date': base_dates,
    'product': np.random.choice(['loan', 'deposit', 'bond'], 5000),
    'counterparty': np.random.choice(['retail', 'wholesale'], 5000),
    'maturity_date': maturity_dates,
    'bucket': np.random.choice(['7d', '30d', '90d', '180d'], 5000),
    'amount': np.random.randint(10000, 500000, 5000),
    'direction': np.random.choice(['inflow', 'outflow'], 5000),
    'hqlatype': np.random.choice(['Level1', 'Level2A', 'Level2B', 'None'], 5000),
    'asf_factor': np.random.choice([0, 0.5, 0.9], 5000),
    'rsf_factor': np.random.choice([0.05, 0.85, 1.0], 5000),
    'scenario_id': np.random.choice(scenario_ids, 5000)
})

cashflows.to_sql('cashflows', con=engine, if_exists='append', index=False)
print("✅ Cashflows table populated.")

# =======================================================
# ✅ Generate RWA (Capital)
# =======================================================
rwa = pd.DataFrame({
    'date': np.random.choice(dates, 1000),
    'exposure_id': [f'EXP{i:04d}' for i in range(1000)],
    'asset_class': np.random.choice(['mortgage', 'corporate', 'sovereign', 'retail'], 1000),
    'approach': np.random.choice(['STD', 'IRB'], 1000),
    'amount': np.random.randint(50000, 1000000, 1000),
    'risk_weight': np.random.choice([0.0, 0.35, 0.5, 1.0], 1000),
    'scenario_id': np.random.choice(scenario_ids, 1000)
})

rwa['rwa_amount'] = rwa['amount'] * rwa['risk_weight']
rwa['capital_requirement'] = rwa['rwa_amount'] * 0.08

rwa.to_sql('rwa', con=engine, if_exists='append', index=False)
print("✅ RWA table populated.")

# =======================================================
# ✅ Generate IRRBB
# =======================================================
base_dates = np.random.choice(dates, 500)
maturity_offsets = pd.to_timedelta(np.random.randint(30, 3650, 500), unit='D')
maturity_dates = base_dates + maturity_offsets

irrbb = pd.DataFrame({
    'date': base_dates,
    'instrument': [f'INST{i:04d}' for i in range(500)],
    'cashflow': np.random.randint(-100000, 100000, 500),
    'maturity_date': maturity_dates,
    'tenor_bucket': np.random.choice(['0-1y', '1-3y', '3-5y', '5-10y', '10y+'], 500),
    'pv01': np.random.normal(0, 1, 500).round(6),
    'rate_sensitivity': np.random.normal(0, 1, 500).round(6),
    'scenario_id': np.random.choice(scenario_ids, 500)
})

irrbb.to_sql('irrbb', con=engine, if_exists='append', index=False)
print("✅ IRRBB table populated.")

# =======================================================
# ✅ Generate Balance Sheet
# =======================================================
balance_items = ['CET1', 'Tier1', 'Total Capital', 'Total Assets', 'Total Liabilities']
balance_data = []

for date in dates:
    for item in balance_items:
        balance_data.append({
            'date': date,
            'item': item,
            'amount': np.random.randint(1000000, 10000000),
            'scenario_id': np.random.choice(scenario_ids)
        })

balance_sheet = pd.DataFrame(balance_data)

balance_sheet.to_sql('balance_sheet', con=engine, if_exists='append', index=False)
print("✅ Balance Sheet table populated.")

# =======================================================
# ✅ Populate Params Table
# =======================================================
params = pd.DataFrame([
    # NSFR ASF Factors
    {'key': 'asf_factor_retail_stable', 'value': '0.95'},
    {'key': 'asf_factor_retail_less_stable', 'value': '0.90'},
    {'key': 'asf_factor_wholesale_lt1y', 'value': '0.0'},
    {'key': 'asf_factor_wholesale_gt1y', 'value': '0.5'},

    # NSFR RSF Factors
    {'key': 'rsf_factor_loans_gt1y', 'value': '1.0'},
    {'key': 'rsf_factor_loans_lt1y', 'value': '0.85'},
    {'key': 'rsf_factor_hqla_level1', 'value': '0.05'},
    {'key': 'rsf_factor_other_assets', 'value': '1.0'},

    # LCR Inflow/Outflow Caps
    {'key': 'lcr_inflow_cap', 'value': '0.75'},
    {'key': 'lcr_outflow_cap', 'value': '1.00'},

    # HQLA Haircuts
    {'key': 'haircut_level2a', 'value': '0.15'},
    {'key': 'haircut_level2b', 'value': '0.50'},

    # IRRBB Regulatory Threshold
    {'key': 'eve_tier1_breach_ratio', 'value': '0.15'},

    # Other Constants
    {'key': 'capital_requirement_ratio', 'value': '0.08'}
])

params.to_sql('params', con=engine, if_exists='append', index=False)
print("✅ Params table populated.")

print("🎉 ✅ All data generated successfully.")
