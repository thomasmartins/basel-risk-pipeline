"""ETL utilities for ECB SDW and local CSVs."""
import requests, io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

BASE = "https://data-api.ecb.europa.eu/service/data"
START = "2000-01"

def fetch_csv(dataset, key, start=START):
    # key should NOT include the dataset id prefix
    url = f"{BASE}/{dataset}/{key}?startPeriod={start}&format=csvdata"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

series_map = {
    # Prices
    "hicp_index_monthly.csv": ("ICP", "M.U2.N.000000.4.INX"),
    # Real activity
    "gdp_clv_q_sa_levels.csv": ("MNA", "Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N"),
    # Policy rate
    "policy_rate_mro.csv": ("FM", "B.U2.EUR.4F.KR.MRR_FR.LEV"),    
    # Fiscal
    "gov_deficit_pct_gdp_q.csv": ("GFS", "Q.N.I9.W0.S13.S1._Z.B.B9._Z._Z._Z.XDC_R_B1GQ_CY._Z.S.V.CY._T"),
    "gov_debt_pct_gdp_q.csv": ("GFS", "Q.N.I9.W0.S13.S1.C.L.LE.GD.T._Z.XDC_R_B1GQ_CY._T.F.V.N._T"),
    # Yields (daily)
    "yc_spot_1y_daily.csv": ("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y"),
    "yc_spot_5y_daily.csv": ("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y"),
    "yc_spot_10y_daily.csv": ("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"),
}

# optional: quick self-test to see exact URL + status
for fname, (ds, key) in series_map.items():
    url = f"{BASE}/{ds}/{key}?startPeriod={START}&format=csvdata"
    try:
        df = fetch_csv(ds, key)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.to_csv(f"data_raw/{fname}", index=False)
        print("OK ", url)
    except Exception as e:
        print("ERR", url, "->", e)

os.makedirs("data_raw", exist_ok=True)

for fname, (ds, key) in series_map.items():
    df = fetch_csv(ds, key)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.to_csv(f"data_raw/{fname}", index=False)
    print("Saved", f"data_raw/{fname}")

# --- Optional transformations for modeling ---

# 1) Quarterly inflation from HICP index (log-diff of quarterly averages, annualized)
hicp = pd.read_csv("data_raw/hicp_index_monthly.csv")
time_col = "time_period" if "time_period" in hicp.columns else "TIME_PERIOD"
val_col  = "obs_value"   if "obs_value"   in hicp.columns else "OBS_VALUE"
hicp[time_col] = pd.to_datetime(hicp[time_col])
hicp_q = (hicp.set_index(time_col)[val_col]
              .resample("QE").mean()
              .to_frame("hicp_index_q"))

# Make sure hicp_q is sorted and 'hicp_index_q' is numeric
hicp_q = hicp_q.sort_values("time_period").copy()
hicp_q["hicp_index_q"] = pd.to_numeric(hicp_q["hicp_index_q"], errors="coerce")

# Annualized quarterly log difference (400 = 4 quarters * 100 for %)
# safer: compute log diff directly
hicp_q["pi_q_ann"] = 400 * np.log(hicp_q["hicp_index_q"] / hicp_q["hicp_index_q"].shift(1))
hicp_q.reset_index().to_csv("data_raw/hicp_quarterly_inflation_from_index.csv", index=False)
print("Saved data_raw/hicp_quarterly_inflation_from_index.csv")

# 2) Quarterly-average the yields (1Y/5Y/10Y)
for tenor in ("1y","5y","10y"):
    y = pd.read_csv(f"data_raw/yc_spot_{tenor}_daily.csv")
    tcol = "time_period" if "time_period" in y.columns else "TIME_PERIOD"
    vcol = "obs_value"   if "obs_value"   in y.columns else "OBS_VALUE"
    y[tcol] = pd.to_datetime(y[tcol])
    y_q = y.set_index(tcol)[vcol].resample("QE").mean().to_frame(f"yc_spot_{tenor}")
    y_q.reset_index().to_csv(f"data_raw/yc_spot_{tenor}_quarterly_mean.csv", index=False)
    print(f"Saved data_raw/yc_spot_{tenor}_quarterly_mean.csv")
    
# 3) Quarterly policy rate 
pol = pd.read_csv("data_raw/policy_rate_mro.csv",  # cols: date, rate
                  parse_dates=["time_period"]).sort_values("time_period")
pol = pol.rename(columns={"time_period":"event_date", "obs_value":"policy_rate"})

start_q = pol["event_date"].min().to_period("Q").end_time.normalize()
end_q   = pol["event_date"].max().to_period("Q").end_time.normalize()
qe = pd.date_range(start=start_q, end=end_q, freq="QE")  # quarter ends

target = pd.DataFrame({"date": qe})

# Map each quarter end to the latest event at or before that date
pol_q = pd.merge_asof(
    target.sort_values("date"),
    pol.sort_values("event_date"),
    left_on="date", right_on="event_date",
    direction="backward"
)[["date", "policy_rate"]]

# Optional: if you want to fill *before* the first event with its first value:
# q_series["policy_rate"] = q_series["policy_rate"].fillna(pol["policy_rate"].iloc[0])

# Use in your panel
pol_q = pol_q.set_index("date")
pol_q.reset_index().to_csv("data_raw/policy_rate_mro.csv", index=False)
print("Saved data_raw/policy_rate_mro.csv")

# 4) Alternative: Compute GDP growth from CLV levels (quarterly, annualized)
gdp = pd.read_csv("data_raw/gdp_clv_q_sa_levels.csv")
tcol = "time_period" if "time_period" in gdp.columns else "TIME_PERIOD"
vcol = "obs_value"   if "obs_value"   in gdp.columns else "OBS_VALUE"
gdp[tcol] = pd.PeriodIndex(gdp[tcol], freq='Q')
gdp = gdp.set_index(tcol).sort_index()
gdp["gdp_q_ann"] = 400 * np.log(gdp[vcol] / gdp[vcol].shift(1))
gdp.reset_index()[[tcol, vcol, "gdp_q_ann"]].to_csv("data_raw/gdp_growth_from_levels_q.csv", index=False)
print("Saved data_raw/gdp_growth_from_levels_q.csv")

RAW = "data_raw"
OUTDIR = "data"
os.makedirs(OUTDIR, exist_ok=True)

TRIM_START = "2000-01-01"  # harmonization trim

def _read_q_csv(path, date_col_guess=("time_period","TIME_PERIOD","date","Date"), value_cols=None):
    df = pd.read_csv(path)
    dcol = next(c for c in date_col_guess if c in df.columns)
    df[dcol] = pd.PeriodIndex(df[dcol], freq='Q')
    df = df.sort_values(dcol)
    if value_cols is not None:
        keep = [dcol] + [c for c in value_cols if c in df.columns]
        df = df[keep]
    df = df.rename(columns={dcol:"date"})
    # 🔧 Normalize to QUARTER END here (the key change)
    df["date"] = df["date"].dt.to_timestamp(how="end")
    return df

# --- Load (unchanged) ---
hicp_q = _read_q_csv(os.path.join(RAW, "hicp_quarterly_inflation_from_index.csv"),
                     value_cols=["hicp_index_q","pi_q_ann"]).rename(columns={"pi_q_ann":"infl_q_ann"})
gdp_q = _read_q_csv(os.path.join(RAW, "gdp_growth_from_levels_q.csv"),
                    value_cols=["gdp_q_ann"]).rename(columns={"gdp_q_ann":"gdp_q_ann"})
pol_q = _read_q_csv(os.path.join(RAW, "policy_rate_mro.csv"),
                    value_cols=["policy_rate"]).rename(columns={"policy_rate":"policy_rate"})
deficit_q = _read_q_csv(os.path.join(RAW, "gov_deficit_pct_gdp_q.csv"),
                        value_cols=["obs_value"]).rename(columns={"obs_value":"gg_deficit_pct_gdp"})
debt_q = _read_q_csv(os.path.join(RAW, "gov_debt_pct_gdp_q.csv"),
                     value_cols=["obs_value"]).rename(columns={"obs_value":"gg_debt_pct_gdp"})
yc1 = _read_q_csv(os.path.join(RAW, "yc_spot_1y_quarterly_mean.csv"),
                  value_cols=["yc_spot_1y"])
yc5 = _read_q_csv(os.path.join(RAW, "yc_spot_5y_quarterly_mean.csv"),
                  value_cols=["yc_spot_5y"])
yc10 = _read_q_csv(os.path.join(RAW, "yc_spot_10y_quarterly_mean.csv"),
                   value_cols=["yc_spot_10y"])

# (Optional) sanity check after normalization
for name, df in [("inflation", hicp_q), ("gdp", gdp_q), ("policy_rate", pol_q), ("deficit", deficit_q),
                 ("debt", debt_q), ("yc1y", yc1), ("yc5y", yc5), ("yc10y", yc10)]:
    print(name, ":", df["date"].min().date(), "→", df["date"].max().date(), "(n=", len(df), ")")

# --- Merge ---
dfs = [hicp_q, gdp_q, pol_q, deficit_q, debt_q, yc1, yc5, yc10]
panel = dfs[0]
for df in dfs[1:]:
    panel = panel.merge(df, on="date", how="inner")

panel = panel.sort_values("date").reset_index(drop=True)

# Term structure factors
panel["level"] = panel[["yc_spot_1y","yc_spot_5y","yc_spot_10y"]].mean(axis=1)
panel["slope_10y_1y"] = panel["yc_spot_10y"] - panel["yc_spot_1y"]
panel["curvature_ns_like"] = 2*panel["yc_spot_5y"] - panel["yc_spot_1y"] - panel["yc_spot_10y"]

# Numeric coercion & drop NA
num_cols = [c for c in panel.columns if c != "date"]
panel[num_cols] = panel[num_cols].apply(pd.to_numeric, errors="coerce")
panel = panel.dropna().copy()

# Harmonization trim (works now that dates are quarter-end)
panel = panel.loc[panel["date"] >= pd.to_datetime(TRIM_START)].copy()

### WINSORIZE THE GDP SERIES AT 15/-15 INSTEAD OF 40/-40 AT COVID SHOCK

gdp_bound = 15.0  # +/- 15 percentage points, annualized

panel['gdp_q_ann_original'] = panel['gdp_q_ann']
panel['gdp_q_ann'] = panel['gdp_q_ann'].clip(-gdp_bound, gdp_bound)

# Which quarters got clipped at the winsorization?
#clipped = panel['gdp_q_ann'] != panel['gdp_q_ann_original']
#print(panel.loc[clipped, ['gdp_q_ann']])

# ✅ Set date as datetime index
panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
panel = panel.set_index("date")

# Save
panel.to_csv(os.path.join(OUTDIR, "quarterly_panel.csv"), index_label="date")
cols_model = [
    "infl_q_ann","gdp_q_ann","policy_rate","gg_deficit_pct_gdp","gg_debt_pct_gdp",
    "yc_spot_1y","yc_spot_5y","yc_spot_10y",
    "level","slope_10y_1y","curvature_ns_like",
]
panel[cols_model].to_csv(os.path.join(OUTDIR, "quarterly_panel_modelvars.csv"), index_label="date")

# Plot functions:
# These will be used directly on the notebook

#print("\nMerged quarterly panel:")
#print(panel["date"].min().date(), "→", panel["date"].max().date(), " (n=", len(panel), ")")
#print(panel.tail(3))

#panel.plot(subplots=True, layout=(4,3), figsize=(12, 8), legend=True)

#panel["gdp_q_ann"].plot(ylim=(-5,5))

# panel["infl_q_ann"].plot()
# panel["gdp_q_ann"].plot()
# panel["gg_debt_pct_gdp"].plot()
# panel["gg_deficit_pct_gdp"].plot()
#panel["yc_spot_1y"].plot()
#panel["yc_spot_5y"].plot()
#panel["yc_spot_10y"].plot()
#panel['policy_rate'].plot()
#plt.show()