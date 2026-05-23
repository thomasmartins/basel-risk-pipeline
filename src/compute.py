"""Basel III risk metric helpers.

Most aggregations now live in dbt marts under `dbt_project/models/marts/`.
This module is a thin client that reads those marts and a small number of
parameterised functions (rate shocks, stress slider math) that depend on
runtime UI inputs and stay in Python.
"""

import numpy as np
import pandas as pd

from src import queries


def _pooled_or_filtered(mart: str, scenario_id):
    """Read a per-scenario mart. If scenario_id is None, pool additive
    columns across scenarios (recompute ratios in the caller).

    Returns a single-row DataFrame.
    """
    df = queries.get_mart(mart, scenario_id=scenario_id)
    if scenario_id is not None:
        return df.iloc[[0]] if not df.empty else df
    # Pool: sum numeric columns, drop scenario_id
    numeric = df.select_dtypes(include="number").drop(columns=["scenario_id"], errors="ignore")
    return numeric.sum(numeric_only=True).to_frame().T


# ==========================================================
# Liquidity Coverage Ratio (LCR)
# ==========================================================
def calculate_lcr(scenario_id=None):
    """LCR = HQLA (post-haircut) / Net 30-day Outflows. Reads mart_lcr."""
    row = _pooled_or_filtered("mart_lcr", scenario_id)
    if row.empty:
        return {"HQLA": 0.0, "Outflows": 0.0, "Inflows": 0.0, "NetOutflows": 0.0, "LCR": np.inf}
    r = row.iloc[0]
    net_out = r["outflows"] - min(r["inflows"], r["outflows"] * 0.75) if scenario_id is None else r["net_outflows"]
    lcr = r["hqla"] / net_out if net_out > 0 else np.inf
    return {
        "HQLA": float(r["hqla"]),
        "Outflows": float(r["outflows"]),
        "Inflows": float(r["inflows"]),
        "NetOutflows": float(net_out),
        "LCR": float(lcr),
    }


def calculate_lcr_timeseries(scenario_id=None):
    """Daily LCR + net cashflow. Reads mart_lcr_daily."""
    df = queries.get_mart("mart_lcr_daily", scenario_id=scenario_id)
    df = df.rename(columns={"as_of_date": "date"}).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return df


# ==========================================================
# Net Stable Funding Ratio (NSFR)
# ==========================================================
def calculate_nsfr(scenario_id=None):
    """NSFR = ASF / RSF. Reads mart_nsfr + mart_nsfr_components."""
    row = _pooled_or_filtered("mart_nsfr", scenario_id)
    components = queries.get_mart("mart_nsfr_components", scenario_id=scenario_id)
    if row.empty:
        return {"ASF": 0.0, "RSF": 0.0, "NSFR": np.inf,
                "ASF_components": {}, "RSF_components": {}}
    r = row.iloc[0]
    asf = float(r["asf"])
    rsf = float(r["rsf"])
    nsfr = asf / rsf if rsf > 0 else np.inf
    asf_components = (
        components.loc[components["side"] == "ASF"]
        .groupby("product")["contribution"].sum().to_dict()
    )
    rsf_components = (
        components.loc[components["side"] == "RSF"]
        .groupby("product")["contribution"].sum().to_dict()
    )
    return {
        "ASF": asf, "RSF": rsf, "NSFR": nsfr,
        "ASF_components": asf_components, "RSF_components": rsf_components,
    }


def calculate_nsfr_timeseries(scenario_id=None):
    df = queries.get_mart("mart_nsfr_daily", scenario_id=scenario_id)
    df = df.rename(columns={"as_of_date": "date", "nsfr": "NSFR"}).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return df


# ==========================================================
# Cashflow Gap Heatmap
# ==========================================================
def calculate_cashflow_gap_heatmap(scenario_id=None):
    """Pivot mart_cashflow_gap to (bucket x date) matrix expected by the heatmap."""
    df = queries.get_mart("mart_cashflow_gap", scenario_id=scenario_id)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return (
        df.pivot_table(
            index="maturity_day_bucket", columns="as_of_date",
            values="signed_amount", aggfunc="sum",
        ).fillna(0).sort_index()
    )


# ==========================================================
# Capital Adequacy
# ==========================================================
def calculate_capital_ratios(scenario_id=None):
    row = _pooled_or_filtered("mart_capital_ratios", scenario_id)
    if row.empty:
        return {"CET1 Ratio": np.inf, "Tier1 Ratio": np.inf,
                "Total Capital Ratio": np.inf, "RWA": 0.0}
    r = row.iloc[0]
    rwa = float(r["rwa"])
    return {
        "CET1 Ratio": float(r["cet1"]) / rwa if rwa > 0 else np.inf,
        "Tier1 Ratio": float(r["tier1"]) / rwa if rwa > 0 else np.inf,
        "Total Capital Ratio": float(r["total_capital"]) / rwa if rwa > 0 else np.inf,
        "RWA": rwa,
    }


def calculate_rwa_by_approach_and_asset_class(scenario_id=None):
    df = queries.get_mart("mart_rwa_breakdown", scenario_id=scenario_id)
    return (
        df.groupby(["approach", "asset_class"], as_index=False)["rwa_amount"]
        .sum()
        .sort_values("rwa_amount", ascending=False)
    )


def calculate_rwa_by_approach(scenario_id=None):
    df = queries.get_mart("mart_rwa_breakdown", scenario_id=scenario_id)
    return (
        df.groupby("approach", as_index=False)["rwa_amount"]
        .sum()
        .sort_values("rwa_amount", ascending=False)
    )


def calculate_capital_timeseries(scenario_id=None):
    df = queries.get_mart("mart_capital_ratios_daily", scenario_id=scenario_id)
    df = df.rename(
        columns={
            "as_of_date": "date", "rwa": "rwa_amount",
            "cet1": "CET1", "tier1": "Tier1", "total_capital": "Total Capital",
            "cet1_ratio": "CET1 Ratio", "tier1_ratio": "Tier1 Ratio",
            "total_capital_ratio": "Total Capital Ratio",
        }
    ).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return df


def calculate_capital_ratios_under_rwa_shock(rwa_shock_pct=0.0, scenario_id=None):
    """Shock RWA by a slider amount; recompute ratios in Python."""
    base = calculate_capital_ratios(scenario_id=scenario_id)
    row = _pooled_or_filtered("mart_capital_ratios", scenario_id)
    r = row.iloc[0]
    shocked_rwa = float(r["rwa"]) * (1 + rwa_shock_pct)
    return {
        "RWA (shocked)": shocked_rwa,
        "CET1 Ratio": float(r["cet1"]) / shocked_rwa if shocked_rwa > 0 else np.inf,
        "Tier1 Ratio": float(r["tier1"]) / shocked_rwa if shocked_rwa > 0 else np.inf,
        "Total Capital Ratio": float(r["total_capital"]) / shocked_rwa if shocked_rwa > 0 else np.inf,
    }


# ==========================================================
# IRRBB
# ==========================================================
EBA_BUCKETS = ["0-1y", "1-3y", "3-5y", "5-10y", "10y+"]

EBA_SHOCKS_BPS = {
    "Parallel Up":     [200, 200, 200, 200, 200],
    "Parallel Down":   [-200, -200, -200, -200, -200],
    "Steepener":       [-50, 0, 100, 150, 200],
    "Flattener":       [250, 200, 150, 100, 50],
    "Short Rate Up":   [300, 200, 100, 0, 0],
    "Short Rate Down": [-300, -200, -100, 0, 0],
}


def calculate_pv01_profile(scenario_id=None):
    df = queries.get_mart("mart_pv01_profile", scenario_id=scenario_id)
    return df.groupby("tenor_bucket", as_index=False)["pv01"].sum()


def _pv01_by_bucket(scenario_id):
    df = calculate_pv01_profile(scenario_id=scenario_id).set_index("tenor_bucket")["pv01"]
    return df.reindex(EBA_BUCKETS).fillna(0)


def _gap_by_bucket(scenario_id):
    df = queries.get_mart("mart_repricing_gap", scenario_id=scenario_id)
    return df.set_index("tenor_bucket")["gap"].reindex(EBA_BUCKETS).fillna(0)


def calculate_eve_sensitivity(shock_bps=200, scenario_id=None):
    pv01 = _pv01_by_bucket(scenario_id)
    total_pv01 = float(pv01.sum())
    return {
        "Total PV01": total_pv01,
        "Shock (bps)": shock_bps,
        "Delta EVE": total_pv01 * (shock_bps / 10_000),
    }


def calculate_nii_sensitivity(shock_bps=200, scenario_id=None):
    """∆NII over a 1y horizon: short-term (≤1y) repricing gap × shock."""
    gap = _gap_by_bucket(scenario_id)
    delta_nii = float(gap.loc["0-1y"]) * (shock_bps / 10_000)
    return {
        "Total Repricing Gap": float(gap.sum()),
        "Short-Term Gap (0-1y)": float(gap.loc["0-1y"]),
        "Shock (bps)": shock_bps,
        "Delta NII": delta_nii,
    }


def calculate_eve_eba_scenarios(scenario_id=None):
    pv01 = _pv01_by_bucket(scenario_id)
    rows = []
    for name, shifts in EBA_SHOCKS_BPS.items():
        delta_eve = sum(p * (s / 10_000) for p, s in zip(pv01.values, shifts))
        rows.append({"Scenario": name, "Delta EVE": delta_eve})
    return pd.DataFrame(rows)


def calculate_nii_eba_scenarios(scenario_id=None):
    """∆NII per EBA scenario using the short-end (0-1y) shift × short-term gap."""
    short_gap = float(_gap_by_bucket(scenario_id).loc["0-1y"])
    rows = []
    for name, shifts in EBA_SHOCKS_BPS.items():
        rows.append({"Scenario": name, "Delta NII": short_gap * (shifts[0] / 10_000)})
    return pd.DataFrame(rows)


def calculate_custom_shock_effects(shock_dict, scenario_id=None):
    """Apply a per-bucket bps shift; return (∆EVE, ∆NII)."""
    pv01 = _pv01_by_bucket(scenario_id)
    delta_eve = sum(pv01[b] * (shock_dict[b] / 10_000) for b in EBA_BUCKETS)
    short_gap = float(_gap_by_bucket(scenario_id).loc["0-1y"])
    delta_nii = short_gap * (shock_dict["0-1y"] / 10_000)
    return delta_eve, delta_nii


def calculate_irrbb_risk_summary(shock_bps_list=None, scenario_id=None):
    """Max ∆EVE / ∆NII across a list of parallel shocks, plus Tier1 breach flag."""
    shock_bps_list = shock_bps_list or [-200, 200]

    pv01 = _pv01_by_bucket(scenario_id)
    total_pv01 = float(pv01.sum())
    gap = _gap_by_bucket(scenario_id)
    short_gap = float(gap.loc["0-1y"])

    cap = _pooled_or_filtered("mart_capital_ratios", scenario_id)
    tier1_cap = float(cap.iloc[0]["tier1"]) if not cap.empty else 0.0

    eve_values = [total_pv01 * (bps / 10_000) for bps in shock_bps_list]
    nii_values = [short_gap * (bps / 10_000) for bps in shock_bps_list]
    max_eve = max(eve_values, key=abs)
    max_nii = max(nii_values, key=abs)

    eve_pct_tier1 = abs(max_eve) / tier1_cap if tier1_cap > 0 else 0.0
    return {
        "Total PV01": total_pv01,
        "Max ∆EVE": max_eve,
        "Max ∆EVE (%)": eve_pct_tier1,
        "∆EVE Breach": eve_pct_tier1 > 0.15,
        "Max ∆NII": max_nii,
        "∆EVE Ratio": eve_pct_tier1,
    }


# ==========================================================
# Stress Test
# ==========================================================
def run_stress_test(
    shock_bps=200,
    retail_withdrawal_pct=0.2,
    wholesale_withdrawal_pct=0.4,
    rwa_stress_pct=0.1,
    scenario_id=None,
):
    """Compose base + stressed views from marts + parameterised shocks."""
    base_lcr = calculate_lcr(scenario_id=scenario_id)["LCR"]
    base_nsfr = calculate_nsfr(scenario_id=scenario_id)["NSFR"]
    base_cap = calculate_capital_ratios(scenario_id=scenario_id)
    base_eve = calculate_eve_sensitivity(shock_bps=0, scenario_id=scenario_id)["Delta EVE"]
    base_nii = calculate_nii_sensitivity(shock_bps=0, scenario_id=scenario_id)["Delta NII"]

    liquidity_deterioration = retail_withdrawal_pct + wholesale_withdrawal_pct / 2
    stressed_lcr = (
        base_lcr / (1 + liquidity_deterioration) if base_lcr != np.inf else np.inf
    )
    stressed_nsfr = base_nsfr * (1 - wholesale_withdrawal_pct)
    stressed_cet1 = base_cap["CET1 Ratio"] / (1 + rwa_stress_pct)
    stressed_tier1 = base_cap["Tier1 Ratio"] / (1 + rwa_stress_pct)
    stressed_eve = calculate_eve_sensitivity(shock_bps=shock_bps, scenario_id=scenario_id)["Delta EVE"]
    stressed_nii = calculate_nii_sensitivity(shock_bps=shock_bps, scenario_id=scenario_id)["Delta NII"]

    return {
        "LCR (Base)": base_lcr,
        "LCR (Stressed)": stressed_lcr,
        "NSFR (Base)": base_nsfr,
        "NSFR (Stressed)": stressed_nsfr,
        "CET1 Ratio (Base)": base_cap["CET1 Ratio"],
        "CET1 Ratio (Stressed)": stressed_cet1,
        "Tier1 Ratio (Base)": base_cap["Tier1 Ratio"],
        "Tier1 Ratio (Stressed)": stressed_tier1,
        "∆EVE (Base)": base_eve,
        "∆EVE (Stressed)": stressed_eve,
        "∆NII (Base)": base_nii,
        "∆NII (Stressed)": stressed_nii,
    }


if __name__ == "__main__":
    print("LCR:", calculate_lcr(scenario_id=1))
    print("NSFR:", calculate_nsfr(scenario_id=1))
    print("Capital Ratios:", calculate_capital_ratios(scenario_id=1))
    print("PV01 Profile:\n", calculate_pv01_profile(scenario_id=1))
    print("EVE Sensitivity:", calculate_eve_sensitivity(200, scenario_id=1))
