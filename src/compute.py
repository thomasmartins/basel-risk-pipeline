import pandas as pd
import numpy as np
from src import queries
import streamlit as st


# ==========================================================
# ✅ Liquidity Coverage Ratio (LCR)
# ==========================================================
def calculate_lcr(scenario_id=None):
    """
    Calculates LCR = HQLA / Net 30-day Outflows
    """
    cashflows = queries.get_cashflows(scenario_id=scenario_id)
    params = queries.get_params()

    # HQLA calculation
    hqla = cashflows[cashflows['hqlatype'].isin(['Level1', 'Level2A', 'Level2B'])]
    haircut_map = {
        'Level1': 0.0,
        'Level2A': float(params.get('haircut_level2a', 0.15)),
        'Level2B': float(params.get('haircut_level2b', 0.5)),
        'None': 1.0
    }
    hqla['adjusted_hqla'] = hqla.apply(
        lambda x: x['amount'] * (1 - haircut_map.get(x['hqlatype'], 1)), axis=1
    )
    total_hqla = hqla['adjusted_hqla'].sum()

    # Outflows and inflows
    outflows = cashflows[cashflows['direction'] == 'outflow']['amount'].sum()
    inflows = cashflows[cashflows['direction'] == 'inflow']['amount'].sum()

    inflow_cap = float(params.get('lcr_inflow_cap', 0.75))
    capped_inflows = min(inflows, outflows * inflow_cap)

    net_outflows = outflows - capped_inflows

    lcr = total_hqla / net_outflows if net_outflows > 0 else np.inf

    return {
        'HQLA': total_hqla,
        'Outflows': outflows,
        'Inflows': inflows,
        'NetOutflows': net_outflows,
        'LCR': lcr
    }


# ==========================================================
# ✅ Net Stable Funding Ratio (NSFR)
# ==========================================================
def calculate_nsfr(scenario_id=None):
    """
    Calculates NSFR = ASF / RSF + breakdowns for stacked bar chart
    """
    cashflows = queries.get_cashflows(scenario_id=scenario_id)
    params = queries.get_params()

    # Compute ASF and RSF contributions
    cashflows['asf'] = cashflows['amount'] * cashflows['asf_factor']
    cashflows['rsf'] = cashflows['amount'] * cashflows['rsf_factor']

    # Filter inflows/outflows
    asf_df = cashflows[cashflows['direction'] == 'inflow']
    rsf_df = cashflows[cashflows['direction'] == 'outflow']

    # Total ASF and RSF
    asf = asf_df['asf'].sum()
    rsf = rsf_df['rsf'].sum()
    nsfr = asf / rsf if rsf > 0 else np.inf

    # Breakdown by product
    asf_components = asf_df.groupby('product')['asf'].sum().to_dict()
    rsf_components = rsf_df.groupby('product')['rsf'].sum().to_dict()

    return {
        'ASF': asf,
        'RSF': rsf,
        'NSFR': nsfr,
        'ASF_components': asf_components,
        'RSF_components': rsf_components
    }

# ==========================================================
# ✅ Cashflow Gap Heatmap
# ==========================================================

def calculate_cashflow_gap_heatmap(scenario_id=None):
    cashflows = queries.get_cashflows(scenario_id=scenario_id)

    # Ensure date columns are datetime
    cashflows['date'] = pd.to_datetime(cashflows['date'])
    cashflows['maturity_date'] = pd.to_datetime(cashflows['maturity_date'])

    # Compute time-to-maturity
    cashflows['maturity_days'] = (cashflows['maturity_date'] - cashflows['date']).dt.days

    # Define maturity buckets
    buckets = [
        (0, 7, '0-7d'),
        (8, 30, '8-30d'),
        (31, 90, '31-90d'),
        (91, 180, '91-180d'),
        (181, 365, '181-365d'),
        (366, 9999, '>1y')
    ]

    def assign_bucket(days):
        for low, high, label in buckets:
            if low <= days <= high:
                return label
        return '>1y'

    cashflows['bucket'] = cashflows['maturity_days'].apply(assign_bucket)

    # Split into inflows and outflows
    inflows = cashflows[cashflows['direction'] == 'inflow'].copy()
    outflows = cashflows[cashflows['direction'] == 'outflow'].copy()

    # Daily totals
    daily_inflows = inflows.groupby('date')['amount'].sum()
    daily_outflows = outflows.groupby('date')['amount'].sum()

    # EBA 75% cap on inflows
    inflow_cap = 0.75
    inflow_limits = pd.concat([
        daily_inflows.rename("inflows"),
        daily_outflows.rename("outflows")
    ], axis=1).fillna(0)
    inflow_limits["capped_inflows"] = inflow_limits["inflows"].clip(upper=inflow_limits["outflows"] * inflow_cap)

    # Apply the cap proportionally across each day’s inflows
    inflows['capped_amount'] = inflows.groupby('date')['amount'].transform(
        lambda x: x * (inflow_limits.loc[x.name, "capped_inflows"] / x.sum())
        if x.name in inflow_limits.index and x.sum() > 0 else 0
    )
    inflows['signed_amount'] = inflows['capped_amount']
    outflows['signed_amount'] = -outflows['amount']

    # Recombine capped inflows + outflows
    cashflows_capped = pd.concat([inflows, outflows])

    # Group and pivot
    grouped = cashflows_capped.groupby(['date', 'bucket'])['signed_amount'].sum().reset_index()
    pivot = grouped.pivot(index='bucket', columns='date', values='signed_amount').fillna(0)
    pivot = pivot.sort_index()

    return pivot


# ==========================================================
# ✅ LCR and NSFR Time Series
# ==========================================================

def calculate_lcr_timeseries(scenario_id=None):
    cashflows = queries.get_cashflows(scenario_id=scenario_id)
    params = queries.get_params()
    print("Params for scenario", scenario_id, params)

    # Prepare inflows/outflows by date
    inflows = cashflows[cashflows['direction'] == 'inflow'].groupby('date')['amount'].sum()
    outflows = cashflows[cashflows['direction'] == 'outflow'].groupby('date')['amount'].sum()

    # Combine inflows and outflows into a single DataFrame
    capped_inflows = pd.concat(
        [inflows.rename("inflows"), outflows.rename("outflows")],
        axis=1
    ).fillna(0)

    # EBA inflow cap: inflows cannot exceed 75% of outflows
    capped_inflows['capped_inflows'] = capped_inflows['inflows'].clip(upper=capped_inflows['outflows'] * 0.75)
    capped_inflows['net_outflows'] = capped_inflows['outflows'] - capped_inflows['capped_inflows']

    # HQLA: assume constant or derive from params
    # Dummy HQLA values per scenario
    scenario_hqla_map = {
        None: 1e9,   # Baseline (no scenario_id)
        1: 8e7,      # ECB Stress (Outflow Spike)
        2: 6e7,      # Liquidity Shock (Wholesale Freeze)
        3: 7e7       # Interest Rate Shock (if applicable)
    }
    hqla = scenario_hqla_map.get(scenario_id, 1e8)  # Fallback to baseline

    # Calculate daily LCR
    capped_inflows['lcr'] = hqla / capped_inflows['net_outflows']
    capped_inflows['net_cashflow'] = inflows.sub(outflows, fill_value=0)

    capped_inflows.index = pd.to_datetime(capped_inflows.index)
    return capped_inflows.reset_index()
    
def calculate_nsfr_timeseries(scenario_id=None):
    cashflows = queries.get_cashflows(scenario_id=scenario_id)
    params = queries.get_params()

    cashflows['date'] = pd.to_datetime(cashflows['date'])

    # ASF
    cashflows['asf'] = cashflows['amount'] * cashflows['asf_factor']
    daily_asf = cashflows[cashflows['direction'] == 'inflow'].groupby('date')['asf'].sum()

    # RSF
    cashflows['rsf'] = cashflows['amount'] * cashflows['rsf_factor']
    daily_rsf = cashflows[cashflows['direction'] == 'outflow'].groupby('date')['rsf'].sum()

    df = pd.concat([daily_asf.rename("ASF"), daily_rsf.rename("RSF")], axis=1).fillna(0)
    df['NSFR'] = df['ASF'] / df['RSF'].replace(0, np.nan)
    return df.reset_index()

# ==========================================================
# ✅ Capital Adequacy (CET1, Tier1, Total Capital)
# ==========================================================
def calculate_capital_ratios(scenario_id=None):
    """
    Calculates CET1, Tier1, Total Capital ratios against RWA
    """
    rwa = queries.get_rwa(scenario_id=scenario_id)
    balance = queries.get_balance_sheet(scenario_id=scenario_id)

    total_rwa = rwa['rwa_amount'].sum()

    def get_capital(item):
        return balance[balance['item'] == item]['amount'].sum()

    cet1 = get_capital('CET1')
    tier1 = get_capital('Tier1')
    total_capital = get_capital('Total Capital')

    ratios = {
        'CET1 Ratio': cet1 / total_rwa if total_rwa > 0 else np.inf,
        'Tier1 Ratio': tier1 / total_rwa if total_rwa > 0 else np.inf,
        'Total Capital Ratio': total_capital / total_rwa if total_rwa > 0 else np.inf,
        'RWA': total_rwa
    }

    return ratios
    
    
def calculate_rwa_by_approach_and_asset_class(scenario_id=None):
    rwa = queries.get_rwa(scenario_id=scenario_id)
    grouped = rwa.groupby(['approach', 'asset_class'])['rwa_amount'].sum().reset_index()
    return grouped.sort_values('rwa_amount', ascending=False)
    
def calculate_rwa_by_approach(scenario_id=None):
    rwa = queries.get_rwa(scenario_id=scenario_id)
    grouped = rwa.groupby('approach')['rwa_amount'].sum().reset_index()
    return grouped.sort_values('rwa_amount', ascending=False)

    
    
def calculate_capital_timeseries(scenario_id=None):
    rwa = queries.get_rwa(scenario_id=scenario_id)
    balance = queries.get_balance_sheet(scenario_id=scenario_id)

    rwa_by_date = rwa.groupby('date')['rwa_amount'].sum()
    capital_by_date = balance.pivot_table(index='date', columns='item', values='amount', aggfunc='sum')

    df = pd.concat([rwa_by_date, capital_by_date], axis=1).fillna(0)

    df['CET1 Ratio'] = df['CET1'] / df['rwa_amount']
    df['Tier1 Ratio'] = df['Tier1'] / df['rwa_amount']
    df['Total Capital Ratio'] = df['Total Capital'] / df['rwa_amount']

    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])

    return df


def calculate_capital_ratios_under_rwa_shock(rwa_shock_pct=0.0, scenario_id=None):
    """
    Simulates capital ratios under an RWA increase (e.g. downgrade).
    rwa_shock_pct: e.g. 0.25 for +25% RWA
    """
    rwa = queries.get_rwa(scenario_id=scenario_id)
    balance = queries.get_balance_sheet(scenario_id=scenario_id)

    total_rwa = rwa['rwa_amount'].sum() * (1 + rwa_shock_pct)

    def get_capital(item):
        return balance[balance['item'] == item]['amount'].sum()

    cet1 = get_capital('CET1')
    tier1 = get_capital('Tier1')
    total_capital = get_capital('Total Capital')

    return {
        'RWA (shocked)': total_rwa,
        'CET1 Ratio': cet1 / total_rwa if total_rwa > 0 else np.inf,
        'Tier1 Ratio': tier1 / total_rwa if total_rwa > 0 else np.inf,
        'Total Capital Ratio': total_capital / total_rwa if total_rwa > 0 else np.inf
    }



# ==========================================================
# ✅ IRRBB - PV01 Profile
# ==========================================================
def calculate_pv01_profile(scenario_id=None):
    """
    Calculates PV01 by tenor bucket
    """
    irrbb = queries.get_irrbb(scenario_id=scenario_id)

    pv01_by_bucket = irrbb.groupby('tenor_bucket')['pv01'].sum().reset_index()

    return pv01_by_bucket


# ==========================================================
# ✅ IRRBB - ∆EVE Approximation (Simple Shock)
# ==========================================================
def calculate_eve_sensitivity(shock_bps=200, scenario_id=None):
    """
    Simple EVE sensitivity → sum(PV01) * shock in bps
    """
    irrbb = queries.get_irrbb(scenario_id=scenario_id)

    total_pv01 = irrbb['pv01'].sum()

    delta_eve = total_pv01 * (shock_bps / 10000)

    return {
        'Total PV01': total_pv01,
        'Shock (bps)': shock_bps,
        'Delta EVE': delta_eve
    }
    
def calculate_nii_sensitivity(shock_bps=200, scenario_id=None):
    """
    Calculates ∆NII under a parallel shock using repricing gap from cashflows
    """
    cashflows = queries.get_cashflows(scenario_id=scenario_id)

    # If not already present, assign buckets by maturity gap
    if 'bucket' not in cashflows.columns:
        cashflows['date'] = pd.to_datetime(cashflows['date'])
        cashflows['maturity_date'] = pd.to_datetime(cashflows['maturity_date'])
        cashflows['maturity_days'] = (cashflows['maturity_date'] - cashflows['date']).dt.days

        def assign_bucket(days):
            if days <= 7:
                return '0-7d'
            elif days <= 30:
                return '8-30d'
            elif days <= 90:
                return '31-90d'
            elif days <= 180:
                return '91-180d'
            elif days <= 365:
                return '181-365d'
            else:
                return '>1y'

        cashflows['bucket'] = cashflows['maturity_days'].apply(assign_bucket)

    # Sum signed cashflows (inflow - outflow) per bucket
    cashflows['signed_amount'] = cashflows.apply(
        lambda row: row['amount'] if row['direction'] == 'inflow' else -row['amount'], axis=1
    )

    gap_by_bucket = cashflows.groupby('bucket')['signed_amount'].sum()

    # Apply interest rate shock
    delta_nii = (gap_by_bucket * (shock_bps / 10_000)).sum()

    return {
        'Total Repricing Gap': gap_by_bucket.sum(),
        'Shock (bps)': shock_bps,
        'Delta NII': delta_nii
    }
    
# ==========================================================
# Calculate EBA-Defined IRRBB Shocks
# ==========================================================
def calculate_eve_eba_scenarios(scenario_id=None):
    irrbb = queries.get_irrbb(scenario_id=scenario_id)

    # PV01 by tenor bucket
    pv01_by_bucket = irrbb.groupby('tenor_bucket')['pv01'].sum()

    # Sort buckets in expected order
    buckets = ['0-1y', '1-3y', '3-5y', '5-10y', '10y+']
    pv01_by_bucket = pv01_by_bucket.reindex(buckets).fillna(0)

    # EBA shocks per bucket (bps)
    eba_shocks = {
        'Parallel Up':      [200, 200, 200, 200, 200],
        'Parallel Down':    [-200, -200, -200, -200, -200],
        'Steepener':        [-50, 0, 100, 150, 200],
        'Flattener':        [250, 200, 150, 100, 50],
        'Short Rate Up':    [300, 200, 100, 0, 0],
        'Short Rate Down':  [-300, -200, -100, 0, 0]
    }

    results = []
    for scenario, shifts in eba_shocks.items():
        delta_eve = sum([
            pv01 * (shock / 10_000)
            for pv01, shock in zip(pv01_by_bucket.values, shifts)
        ])
        results.append({'Scenario': scenario, 'Delta EVE': delta_eve})

    df = pd.DataFrame(results)
    df.columns = ['Scenario', 'Delta EVE']
    return df
    
    
def calculate_nii_eba_scenarios(scenario_id=None):
    irrbb = queries.get_irrbb(scenario_id=scenario_id)

    # Only use short-term buckets for NII — typically '0-1y'
    short_buckets = ['0-1y']  # Optionally add '1-3y' if justified
    irrbb_short = irrbb[irrbb['tenor_bucket'].isin(short_buckets)]

    # Group PV01 by bucket
    pv01_by_bucket = irrbb_short.groupby('tenor_bucket')['pv01'].sum()

    # Ensure we align with the EBA shocks
    eba_shocks = {
        'Parallel Up':      [200],
        'Parallel Down':    [-200],
        'Steepener':        [-50],
        'Flattener':        [250],
        'Short Rate Up':    [300],
        'Short Rate Down':  [-300]
    }

    results = []
    for scenario, shocks in eba_shocks.items():
        delta_nii = sum([
            pv01_by_bucket.get(bucket, 0) * (shock / 10_000)
            for bucket, shock in zip(short_buckets, shocks)
        ])
        results.append({'Scenario': scenario, 'Delta NII': delta_nii})

    return pd.DataFrame(results)
    
def calculate_custom_shock_effects(shock_dict, scenario_id=None):
    """
    Applies user-defined yield curve shifts and computes ∆EVE and ∆NII.
    """
    irrbb = queries.get_irrbb(scenario_id=scenario_id)
    pv01_by_bucket = irrbb.groupby('tenor_bucket')['pv01'].sum()
    buckets = ['0-1y', '1-3y', '3-5y', '5-10y', '10y+']
    pv01_by_bucket = pv01_by_bucket.reindex(buckets).fillna(0)

    delta_eve = sum([
        pv01_by_bucket[bucket] * (shock_dict[bucket] / 10_000)
        for bucket in buckets
    ])
    delta_nii = delta_eve  # Same logic unless you have a different cashflow engine

    return delta_eve, delta_nii
    
def calculate_irrbb_risk_summary(shock_bps_list=None, scenario_id=None):
    """
    Computes key IRRBB KPIs: Total PV01, Max ∆EVE (as % Tier 1), Max ∆NII, Breach flags
    """
    irrbb = queries.get_irrbb(scenario_id=scenario_id)
    tier1 = queries.get_balance_sheet(scenario_id=scenario_id)
    tier1_cap = tier1[tier1['item'] == 'Tier1']['amount'].sum()

    # Total PV01
    total_pv01 = irrbb['pv01'].sum()

    # Max ∆EVE
    max_eve = max([total_pv01 * (bps / 10_000) for bps in shock_bps_list])
    
    #tier1_cap_eur = tier1_cap * 1_000_000  # Convert from millions to EUR
    eve_ratio = max_eve / tier1_cap

    # Max ∆EVE as % Tier 1 Capital
    eve_pct_tier1 = max_eve / tier1_cap if tier1_cap > 0 else 0
    eve_breach = eve_pct_tier1 > 0.15

    # Max ∆NII
    cashflows = queries.get_cashflows(scenario_id=scenario_id)
    cashflows['signed_amount'] = cashflows.apply(
        lambda row: row['amount'] if row['direction'] == 'inflow' else -row['amount'], axis=1
    )
    gap_by_bucket = cashflows.groupby('bucket')['signed_amount'].sum()
    max_nii = max([ (gap_by_bucket * (bps / 10_000)).sum() for bps in shock_bps_list ])

    return {
        'Total PV01': total_pv01,
        'Max ∆EVE': max_eve,
        'Max ∆EVE (%)': eve_pct_tier1,
        '∆EVE Breach': eve_breach,
        'Max ∆NII': max_nii,
        '∆EVE Ratio': eve_ratio
    }


# ==========================================================
# ✅ Example Run
# ==========================================================
if __name__ == "__main__":
    print("LCR:", calculate_lcr())
    print("NSFR:", calculate_nsfr())
    print("Capital Ratios:", calculate_capital_ratios())
    print("PV01 Profile:")
    print(calculate_pv01_profile())
    print("∆EVE Sensitivity:", calculate_eve_sensitivity(200))


def run_stress_test(
    shock_bps=200,
    retail_withdrawal_pct=0.2,
    wholesale_withdrawal_pct=0.4,
    rwa_stress_pct=0.1,
    scenario_id=None
):
    # --- IRRBB Effects ---
    eve_result = calculate_eve_sensitivity(shock_bps=shock_bps, scenario_id=scenario_id)
    nii_result = calculate_nii_sensitivity(shock_bps=shock_bps, scenario_id=scenario_id)

    delta_eve = eve_result['Delta EVE']
    delta_nii = nii_result['Delta NII']

    # --- Liquidity Ratios ---
    lcr_df = calculate_lcr_timeseries(scenario_id)
    latest_lcr = lcr_df.iloc[-1]  # assume latest date
    base_lcr = latest_lcr['lcr']

    nsfr_result = calculate_nsfr(scenario_id)
    base_nsfr = nsfr_result['NSFR']

    # Stressed liquidity assumptions (simple proportional deterioration)
    stressed_lcr = base_lcr * (1 - retail_withdrawal_pct - wholesale_withdrawal_pct / 2)
    stressed_nsfr = base_nsfr * (1 - wholesale_withdrawal_pct)

    # --- Capital Ratios ---
    capital_result = calculate_capital_ratios(scenario_id)
    base_cet1 = capital_result['CET1 Ratio']
    base_tier1 = capital_result['Tier1 Ratio']

    stressed_cet1 = base_cet1 / (1 + rwa_stress_pct)
    stressed_tier1 = base_tier1 / (1 + rwa_stress_pct)

    return {
        "LCR (Base)": base_lcr,
        "LCR (Stressed)": stressed_lcr,
        "NSFR (Base)": base_nsfr,
        "NSFR (Stressed)": stressed_nsfr,
        "CET1 Ratio (Base)": base_cet1,
        "CET1 Ratio (Stressed)": stressed_cet1,
        "Tier1 Ratio (Base)": base_tier1,
        "Tier1 Ratio (Stressed)": stressed_tier1,
        "∆EVE (Base)": delta_eve,
        "∆EVE (Stressed)": delta_eve,  # assumed same
        "∆NII (Base)": delta_nii,
        "∆NII (Stressed)": delta_nii,  # assumed same
    }