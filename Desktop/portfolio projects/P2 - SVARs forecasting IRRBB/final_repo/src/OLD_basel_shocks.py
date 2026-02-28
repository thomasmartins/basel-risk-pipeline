"""Basel III IRRBB scenarios and scaling."""
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
from irrbb_svar.svar import apply_sign_restrictions, make_candidates_from_idata, summarize_irfs, plot_irf_filtered
import arviz as az

@dataclass
class BaselScenario:
    name: str
    delta_bps: Dict[str, float]
    alpha: float = 1.0

panel = pd.read_csv("data/quarterly_panel_modelvars.csv", parse_dates=["date"], index_col="date")

idata = az.from_netcdf("results/bvar_results.nc")

var_order = [
    "infl_q_ann",          # 0  π
    "gdp_q_ann",           # 1  Δy
    "policy_rate",         # 3  policy rate
    "gg_deficit_pct_gdp",  # 4  deficit
    "gg_debt_pct_gdp",     # 5  debt
    "level",               # 6  curve level
    "slope_10y_1y",        # 7  slope 10y - 1y
    "curvature_ns_like"    # 8  curvature
]
var2idx = {v:i for i,v in enumerate(var_order)}
i_pi,i_gdp,i_pol,i_def,i_debt,i_lev,i_slp,i_curv = [var2idx[v] for v in var_order]

cands = make_candidates_from_idata(
    idata=idata,
    p=idata.attrs.get("lags", 2),
    H=20,
    max_draws=5000,           # cap for speed; increase later
    rotations_per_draw=50,    # 5–20 is typical; increase if acceptance is low later
    require_stable=True,
    seed=123
)

# ===== Extracted from basel_scenarios.ipynb =====
def fit_factor_to_tenor_map(panel, tenor_cols):
    # factors must exist in panel with these exact names:
    fac_cols = ["level", "slope_10y_1y", "curvature_ns_like"]
    df = panel[fac_cols + tenor_cols].dropna()
    F  = df[fac_cols].to_numpy()
    Y  = df[tenor_cols].to_numpy()
    F -= F.mean(axis=0); Y -= Y.mean(axis=0)
    B = np.linalg.lstsq(F, Y, rcond=None)[0]  # (3 x n_tenors)
    M = B.T                                   # (n_tenors x 3)
    return M

# Example:
tenor_cols   = ["yc_spot_1y","yc_spot_5y","yc_spot_10y"]   # adjust to your data
tenors_years = np.array([1, 5, 10], float)
M = fit_factor_to_tenor_map(panel, tenor_cols)
M.round(2)

def factor_orientation_signs(M, tenors_years):
    """
    Returns sL,sS,sC ∈ {+1,-1} to align factor signs with intuitive yield effects:
      - sL: +1 if +level → yields up across tenors (else -1)
      - sS: +1 if +slope → 10y rises more than 1y (steepener) (else -1)
      - sC: +1 if +curv → mid-tenor (e.g., 5y) rises relative to ends (else -1)
    """
    t = np.asarray(tenors_years, float)
    j_short = int(np.argmin(np.abs(t - 1.0)))   # tweak if you prefer 0.25y
    j_long  = int(np.argmin(np.abs(t - 10.0)))
    j_mid   = int(np.argmin(np.abs(t - 5.0)))

    mL = M[:, 0]            # level column
    mS = M[:, 1]            # slope column
    mC = M[:, 2]            # curvature column

    sL = 1 if np.mean(mL) >= 0 else -1
    sS = 1 if (mS[j_long] - mS[j_short]) >= 0 else -1
    sC = 1 if (mC[j_mid] - 0.5*(mC[j_short]+mC[j_long])) >= 0 else -1
    return sL, sS, sC

def build_basel_specs_oriented(var_order, M, tenors_years, strict=False, purity=False):
    i_lev  = var_order.index("level")
    i_slp  = var_order.index("slope_10y_1y")
    i_curv = var_order.index("curvature_ns_like")
    sL, sS, sC = factor_orientation_signs(M, tenors_years)

    W = (0,0) if strict else (0,1)   # impact only vs impact+1q window
    zero = (W[0], W[1], 0)

    specs = {
        # parallel: force LEVEL sign after orientation
        "parallel_up":   { i_lev: [(W[0], W[1], +1 * sL)] },
        "parallel_down": { i_lev: [(W[0], W[1], -1 * sL)] },

        # twists: force SLOPE sign after orientation
        "steepener":     { i_slp: [(W[0], W[1], +1 * sS)] },
        "flattener":     { i_slp: [(W[0], W[1], -1 * sS)] },

        # short-end: slope opposes level (front moves more)
        "short_up":      { i_slp: [(W[0], W[1], -1 * sS)], i_lev: [(W[0], W[1], +1 * sL)] },
        "short_down":    { i_slp: [(W[0], W[1], +1 * sS)],  i_lev: [(W[0], W[1], -1 * sL)] },
    }

    if purity:
        # keep parallels near-flat at impact in factor space
        for k in ("parallel_up","parallel_down"):
            specs[k][i_slp]  = [zero]
            specs[k][i_curv] = [zero]
        # keep twists “pure” on impact (optional curvature zero)
        for k in ("steepener","flattener"):
            specs[k][i_lev]  = [zero]
            if purity == "clean":
                specs[k][i_curv] = [zero]
        if purity == "clean":
            for k in ("short_up","short_down"):
                specs[k].setdefault(i_curv, [zero])
    return specs


def basel_target_curve(scenario, tenors_years, R_parallel, R_short, R_long, x=4.0):
    t = np.asarray(tenors_years, float)
    short_shape = np.exp(-t / x)
    long_shape  = 1.0 - np.exp(-t / x)
    if scenario == "parallel_up":   return  +R_parallel * np.ones_like(t)
    if scenario == "parallel_down": return  -R_parallel * np.ones_like(t)
    if scenario == "short_up":      return  +R_short    * short_shape
    if scenario == "short_down":    return  -R_short    * short_shape
    if scenario == "steepener":     return  (-R_short * short_shape) + ( +R_long * long_shape)
    if scenario == "flattener":     return  (+R_short * short_shape) + ( -R_long * long_shape)
    raise ValueError(scenario)

# 👇 Replace with the official magnitudes for your currency (bps)
basel_mag = {
    "parallel_up":   dict(R_parallel=200.0, R_short=0.0,   R_long=0.0,   x=4.0),
    "parallel_down": dict(R_parallel=200.0, R_short=0.0,   R_long=0.0,   x=4.0),
    "short_up":      dict(R_parallel=0.0,   R_short=250.0, R_long=0.0,   x=4.0),
    "short_down":    dict(R_parallel=0.0,   R_short=250.0, R_long=0.0,   x=4.0),
    "steepener":     dict(R_parallel=0.0,   R_short=150.0, R_long=100.0, x=4.0),
    "flattener":     dict(R_parallel=0.0,   R_short=150.0, R_long=100.0, x=4.0),
}


i_lev  = var_order.index("level")
i_slp  = var_order.index("slope_10y_1y")
i_crv  = var_order.index("curvature_ns_like")
specs_basel = build_basel_specs_oriented(var_order, M, tenors_years=(1,5,10), strict=True, purity=False)

accepted_basel = {
    scen: apply_sign_restrictions(cands, spec, scen, tol_zero=1e-4, max_accept=3000)
    for scen, spec in specs_basel.items()
}

for scen in list(specs_basel.keys()):
    print(scen, ": accepted", accepted_basel[scen]["accepted"], "of", accepted_basel[scen]["tried"],
    f"(rate={accepted_basel[scen]['accept_rate']:.2%})")

def compute_alphas_for_scenario(accepted_scen, var_order, tenors_years, M, target_params, scenario):
    i_lev  = var_order.index("level")
    i_slp  = var_order.index("slope_10y_1y")
    i_curv = var_order.index("curvature_ns_like")

    IRFs = np.asarray(accepted_scen["IRFs"])     # (Nacc, H+1, K, K) with shock col 0
    impact = IRFs[:, 0, :, 0]                    # (Nacc, K)
    F = np.column_stack([impact[:, i_lev], impact[:, i_slp], impact[:, i_curv]])  # (Nacc, 3)

    # unit-impact curve per draw at the selected tenors
    Y_unit = F @ M.T                              # (Nacc, n_tenors)

    y_star_bps = basel_target_curve(scenario, tenors_years, **target_params)  # (n_tenors,)
    y_star = y_star_bps * 1e-2
    num   = (Y_unit * y_star[None,:]).sum(axis=1)
    denom = (Y_unit**2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alphas = np.where(denom > 0, num / denom, 0.0)

    # Optional diagnostics (R² and RMSE of the fitted impact curve)
    fitted = (alphas[:,None] * Y_unit)
    ss_res = ((fitted - y_star[None,:])**2).sum(axis=1)
    ss_tot = ((y_star - y_star.mean())**2).sum()
    r2     = 1 - (ss_res / (ss_tot + 1e-12))
    rmse   = np.sqrt(np.mean((fitted - y_star[None,:])**2, axis=1))
    return alphas, y_star, r2, rmse


# alphas = compute_alphas_for_scenario(accepted_basel['parallel_down'], var_order, np.array([1,5,10]), M, basel_mag['parallel_down'], 'parallel_down')[0].round(1)
# # Are you plotting *scaled* impact curves?
# fitted = alphas[:, None] * Y_unit_bps     # NOT Y_unit_bps alone

# # For parallel_down, is alpha mostly negative?
# print("sign(alphas) median:", np.median(np.sign(alphas)))
# # Expect < 0 for parallel_down, > 0 for parallel_up

# # Does the Basel target itself have the right sign?
# print(basel_target_curve("parallel_down", tenors_years, **basel_mag["parallel_down"])[:5])


# compute_alphas_for_scenario(accepted_basel['parallel_up'], var_order, np.array([1,5,10]), M, basel_mag['parallel_up'], 'parallel_up')[4].round(1)

def srsvar_conditional_fanchart_scaled(
    accepted_all, panel, vars_order,
    scenario, alphas,                      # aligned with accepted_all[scenario]['B']
    p=None, h=12, shock_horizon=1, n_paths=200, seed=123
):
    import numpy as np, pandas as pd
    params = accepted_all[scenario]
    B_all = np.asarray(params["B"])                 # (Nacc,K,Kp)
    c_all = np.asarray(params.get("c", np.zeros((B_all.shape[0], B_all.shape[1]))))
    IRFs  = np.asarray(params["IRFs"])              # (Nacc,H+1,K,K)
    Nacc, K, Kp = B_all.shape

    rng = np.random.default_rng(seed)
    sel = rng.choice(Nacc, size=min(n_paths, Nacc), replace=False)

    if Kp % K != 0: raise ValueError("B shape mismatch.")
    p_eff = (Kp // K) if p is None else p

    hist_mat = panel[vars_order].dropna().to_numpy()
    if hist_mat.shape[0] < p_eff: raise ValueError("Not enough history.")
    hist0 = hist_mat[-p_eff:, :][::-1].T

    last_idx = panel.index[-1]
    last_q = last_idx.asfreq("Q") if hasattr(panel.index, "asfreq") else pd.Period(pd.Timestamp(last_idx), freq="Q")
    dates = pd.period_range(last_q, periods=h+1, freq="Q")[1:].to_timestamp("Q")

    paths = np.zeros((len(sel), h, K))
    for d, idx in enumerate(sel):
        B = B_all[idx]
        c = c_all[idx] if c_all.ndim == 2 else c_all
        Cimp = IRFs[idx, 0]                 # (K,K) impact
        alpha = float(alphas[idx])          # Basel scaling for this draw

        As = [B[:, j*K:(j+1)*K] for j in range(p_eff)]
        hist_buf = hist0.copy()
        for t in range(h):
            yhat = c.copy()
            for j, Aj in enumerate(As):
                yhat += Aj @ hist_buf[:, j]
            if t < shock_horizon:
                eps = np.zeros(K); eps[0] = alpha
                yhat = yhat + Cimp @ eps
            paths[d, t, :] = yhat
            hist_buf = np.column_stack([yhat, hist_buf[:, :-1]])

    med = np.median(paths, axis=0)
    lo  = np.percentile(paths, 10, axis=0)
    hi  = np.percentile(paths, 90, axis=0)

    fan_df = pd.concat([
        pd.DataFrame({
            "date": dates, "var": name, "h": np.arange(1, h+1),
            "yhat": med[:, k], "yhat_lo": lo[:, k], "yhat_hi": hi[:, k],
            "model": f"SR-SVAR_{scenario}"
        })
        for k, name in enumerate(vars_order)
    ], ignore_index=True)
    return fan_df, paths

# def srsvar_conditional_fanchart_scaled(
#     accepted_all, panel, vars_order,
#     scenario, alphas,                      # alphas aligned to accepted draws for *this* scenario
#     p=None, h=12, shock_horizon=1, n_paths=200, seed=123
# ):
#     """
#     Conditional SR-SVAR forecast with per-draw scaling (alphas).
#     Robust to accepted_all being either:
#       - a single accepted dict with keys {'B','C','IRFs',...}, or
#       - a dict of scenarios -> accepted dicts (we pick accepted_all[scenario]).
#     """
#     import numpy as np, pandas as pd

#     # ---------- Resolve params dict ----------
#     def _is_accepted_dict(d):
#         return isinstance(d, dict) and isinstance(d.get("IRFs", None), np.ndarray)

#     if _is_accepted_dict(accepted_all) and "B" in accepted_all:
#         params = accepted_all
#         resolved_scen = "<single>"
#     elif isinstance(accepted_all, dict) and scenario in accepted_all and _is_accepted_dict(accepted_all[scenario]):
#         params = accepted_all[scenario]
#         resolved_scen = scenario
#     else:
#         keys = list(accepted_all.keys()) if isinstance(accepted_all, dict) else str(type(accepted_all))
#         raise KeyError(
#             f"srsvar_conditional_fanchart_scaled: could not resolve scenario.\n"
#             f"  scenario={scenario!r}\n"
#             f"  accepted_all keys/type={keys}\n"
#             f"Pass either a single accepted dict, or {{scenario_name: accepted_dict}} with matching 'scenario'."
#         )

#     # ---------- Basic shapes ----------
#     B_all = np.asarray(params["B"])                      # (Nacc, K, Kp)
#     C_all = np.asarray(params["C"])                      # (Nacc, K, K) impact matrices (shock is col 0)
#     IRFs  = np.asarray(params["IRFs"])                   # (Nacc, H+1, K, K)
#     Nacc, K, Kp = B_all.shape
#     if p is None:
#         p = Kp // K
#     assert Kp == K * p, f"Kp ({Kp}) != K*p ({K}*{p})"

#     # ---------- Panel columns check ----------
#     missing = [c for c in vars_order if c not in panel.columns]
#     if missing:
#         raise KeyError(
#             f"Panel is missing required columns from vars_order: {missing}\n"
#             f"Available columns start: {list(panel.columns)[:10]}"
#         )

#     # ---------- Alphas alignment ----------
#     alphas = np.asarray(alphas).reshape(-1)
#     if alphas.size == 1 and Nacc > 1:
#         alphas = np.repeat(alphas, Nacc)
#     if alphas.size != Nacc:
#         raise ValueError(
#             f"alphas length {alphas.size} does not match number of accepted draws {Nacc} "
#             f"for scenario {resolved_scen}."
#         )

#     # ---------- History buffer ----------
#     hist = panel[vars_order].dropna().to_numpy()[-p:, :]    # (p, K)
#     hist0 = hist[::-1].T                                    # (K, p)

#     # ---------- Simulate ----------
#     rng = np.random.default_rng(seed)
#     draw_idx = rng.choice(Nacc, size=min(n_paths, Nacc), replace=False)
#     paths = np.zeros((draw_idx.size, h, K))
#     dates = pd.period_range(panel.index[-1], periods=h+1, freq="Q")[1:].to_timestamp("Q")

#     for dpos, di in enumerate(draw_idx):
#         B = B_all[di]              # (K, Kp)
#         C = C_all[di]              # (K, K)
#         a = float(alphas[di])      # scalar scale for this draw

#         # Unpack companion blocks
#         As = [B[:, j*K:(j+1)*K] for j in range(p)]   # list of (K,K)
#         c_vec = params.get("c", None)
#         if c_vec is None:
#             c_vec = np.zeros(K, dtype=float)
#         else:
#             c_vec = np.asarray(c_vec[di]) if np.ndim(params["c"])==2 else np.asarray(c_vec)

#         buf = hist0.copy()         # (K, p)
#         for t in range(h):
#             yhat = c_vec.copy()
#             for j, Aj in enumerate(As):
#                 yhat += Aj @ buf[:, j]
#             if t < shock_horizon:
#                 eps = np.zeros(K); eps[0] = a       # identified shock is column 0
#                 yhat = yhat + C @ eps
#             paths[dpos, t, :] = yhat
#             buf = np.column_stack([yhat, buf[:, :-1]])

#     # ---------- Summarize ----------
#     med = np.median(paths, axis=0)
#     lo  = np.percentile(paths, 10, axis=0)
#     hi  = np.percentile(paths, 90, axis=0)

#     rows = []
#     for k, name in enumerate(vars_order):
#         rows.append(pd.DataFrame({
#             "date": dates,
#             "var": name,
#             "h": np.arange(1, h+1),
#             "yhat": med[:, k],
#             "yhat_lo": lo[:, k],
#             "yhat_hi": hi[:, k],
#             "model": f"SR-SVAR_{resolved_scen}"
#         }))
#     fan_df = pd.concat(rows, ignore_index=True)
#     return fan_df, paths


def run_all_basel_scenarios(
    accepted_basel, panel, var_order, tenors_years, M, basel_mag,
    r2_min=None, rmse_max=None,
    h=12, shock_horizon=1, n_paths=200, seed=123
):
    results = {}
    for scen in ["parallel_up","parallel_down","steepener","flattener","short_up","short_down"]:
        acc = accepted_basel[scen]
        alphas, y_star, r2, rmse = compute_alphas_for_scenario(acc, var_order, tenors_years, M, basel_mag[scen], scen)

        # Optional: keep only well-fitted draws for simulation
        keep = np.ones_like(alphas, dtype=bool)
        if r2_min  is not None: keep &= (r2 >= r2_min)
        if rmse_max is not None: keep &= (rmse <= rmse_max)

        if keep.sum() == 0:
            print(f"[{scen}] No draws pass fit filter; falling back to all accepted draws.")
            keep = np.ones_like(alphas, dtype=bool)

        # Restrict accepted dict to kept draws for clean simulation
        sub = {k: (v[keep] if hasattr(v, "shape") and len(getattr(v, "shape", ()))>0 and v.shape[0]==alphas.shape[0] else v)
               for k, v in acc.items()}
        accepted_tmp = {**accepted_basel, scen: sub}

        fan_df, paths = srsvar_conditional_fanchart_scaled(
            accepted_tmp, panel, var_order,
            scenario=scen, alphas=alphas[keep],
            h=h, shock_horizon=shock_horizon, n_paths=n_paths, seed=seed
        )
        results[scen] = dict(fan=fan_df, paths=paths, alphas=alphas, r2=r2, rmse=rmse, target=y_star)
    return results

results = run_all_basel_scenarios(
    accepted_basel, panel, var_order, tenors_years, M, basel_mag,
    r2_min=0.90,        # tighten this to trim acceptance
    rmse_max=15.0,      # bps
    h=12, shock_horizon=1, n_paths=200, seed=123
)

results['parallel_up']['fan'][results['parallel_up']['fan']['var'] == 'infl_q_ann']['yhat'].plot()

# OLD

# import numpy as np

# def _unit_curve_from_accepted(accepted_scen, var_order, M, bps=True):
#     """Return unit impact curves in tenor space for all accepted draws."""
#     i_lev  = var_order.index("level")
#     i_slp  = var_order.index("slope_10y_1y")
#     i_curv = var_order.index("curvature_ns_like")
#     IRFs = np.asarray(accepted_scen["IRFs"])           # (Nacc, H+1, K, K)
#     imp  = IRFs[:, 0, :, 0]                            # (Nacc, K)
#     F    = np.column_stack([imp[:, i_lev], imp[:, i_slp], imp[:, i_curv]])  # (Nacc, 3)
#     Y    = F @ M.T                                     # (Nacc, n_tenors), decimals
#     return (1e4 * Y) if bps else Y

# def _cos2_and_rmse(Y_unit, y_star_bps):
#     """cos² and (post-α) RMSE per draw, both in bps space."""
#     # α per draw (through-origin LS)
#     num   = (Y_unit * y_star_bps[None, :]).sum(axis=1)
#     denom = (Y_unit**2).sum(axis=1)
#     alphas = np.divide(num, denom, out=np.zeros_like(num), where=denom>0)

#     fitted = alphas[:, None] * Y_unit
#     # cos² (shape), robust and bounded in [0,1]
#     dot   = (fitted * y_star_bps[None, :]).sum(axis=1)
#     normf = np.sqrt((fitted**2).sum(axis=1)) + 1e-12
#     normt = float(np.linalg.norm(y_star_bps) + 1e-12)
#     cos2  = (dot / (normf * normt))**2
#     # RMSE (size), in bps
#     rmse  = np.sqrt(np.mean((fitted - y_star_bps[None, :])**2, axis=1))
#     return alphas, cos2, rmse

# def filter_by_curve_shape(
#     accepted_scen, var_order, tenors_years, M, basel_params, scenario,
#     cos2_min=0.9, rmse_max=None
# ):
#     """
#     Keep only accepted draws whose unit impact curve matches Basel shape (cos²) and, optionally, size (RMSE).
#     Returns: filtered dict (same keys as accepted_scen), mask, and diagnostics arrays.
#     """
#     # Build Basel target (bps)
#     def basel_target_curve(scenario, tenors_years, R_parallel, R_short, R_long, x=4.0):
#         t = np.asarray(tenors_years, float)
#         short_shape = np.exp(-t / x)
#         long_shape  = 1.0 - np.exp(-t / x)
#         if scenario == "parallel_up":   return  +basel_params["R_parallel"] * np.ones_like(t)
#         if scenario == "parallel_down": return  -basel_params["R_parallel"] * np.ones_like(t)
#         if scenario == "short_up":      return  +basel_params["R_short"]    * short_shape
#         if scenario == "short_down":    return  -basel_params["R_short"]    * short_shape
#         if scenario == "steepener":     return  (-basel_params["R_short"] * short_shape) + ( +basel_params["R_long"] * long_shape)
#         if scenario == "flattener":     return  (+basel_params["R_short"] * short_shape) + ( -basel_params["R_long"] * long_shape)
#         raise ValueError(scenario)

#     Y_unit_bps = _unit_curve_from_accepted(accepted_scen, var_order, M, bps=True)
#     y_star_bps = basel_target_curve(scenario, tenors_years, **basel_params)

#     alphas, cos2, rmse = _cos2_and_rmse(Y_unit_bps, y_star_bps)

#     keep = (cos2 >= cos2_min)
#     if rmse_max is not None:
#         keep &= (rmse <= rmse_max)

#     # slice arrays in the accepted dict by mask
#     filtered = {}
#     for k, v in accepted_scen.items():
#         if isinstance(v, np.ndarray) and v.shape[:1] == (Y_unit_bps.shape[0],):
#             filtered[k] = v[keep]
#         else:
#             filtered[k] = v
#     # stash diagnostics & alphas (aligned to pre-filter length for reference)
#     filtered["_shape_cos2"] = cos2
#     filtered["_shape_rmse"] = rmse
#     filtered["_alphas_unitfit"] = alphas
#     filtered["_keep_mask"] = keep
#     # update counts
#     filtered["accepted"]   = int(keep.sum())
#     filtered["accept_rate"] = float(keep.mean())
#     return filtered

def _unit_curve_from_accepted(accepted_scen, var_order, M, bps=True):
    """Return unit impact curves in tenor space for all accepted draws."""
    i_lev  = var_order.index("level")
    i_slp  = var_order.index("slope_10y_1y")
    i_curv = var_order.index("curvature_ns_like")
    IRFs = np.asarray(accepted_scen["IRFs"])           # (Nacc, H+1, K, K)
    imp  = IRFs[:, 0, :, 0]                            # (Nacc, K)
    F    = np.column_stack([imp[:, i_lev], imp[:, i_slp], imp[:, i_curv]])  # (Nacc, 3)
    Y    = F @ M.T                                     # (Nacc, n_tenors), decimals
    return (1e4 * Y) if bps else Y

def _cos2_and_rmse(Y_unit, y_star_bps):
    """cos² and (post-α) RMSE per draw, both in bps space."""
    # α per draw (through-origin LS)
    num   = (Y_unit * y_star_bps[None, :]).sum(axis=1)
    denom = (Y_unit**2).sum(axis=1)
    alphas = np.divide(num, denom, out=np.zeros_like(num), where=denom>0)

    fitted = alphas[:, None] * Y_unit
    # cos² (shape), robust and bounded in [0,1]
    dot   = (fitted * y_star_bps[None, :]).sum(axis=1)
    normf = np.sqrt((fitted**2).sum(axis=1)) + 1e-12
    normt = float(np.linalg.norm(y_star_bps) + 1e-12)
    cos2  = (dot / (normf * normt))**2
    # RMSE (size), in bps
    rmse  = np.sqrt(np.mean((fitted - y_star_bps[None, :])**2, axis=1))
    return alphas, cos2, rmse


def filter_by_curve_shape(
    accepted_scen, var_order, tenors_years, M, basel_params, scenario,
    cos2_min=0.90, rmse_max=None, enforce_direction=True,
    min_den=1e-10, alpha_max=None
):
    """
    Filter accepted draws by curve shape (cos²) and optional size (RMSE after scaling).
    Returns: filtered dict (same keys), plus diagnostics.
    Assumes _unit_curve_from_accepted returns array (n_draws, K) of impact in *bps* for the chosen shock.
    """

    import numpy as np

    # --- 1) Basel target builder (all in bps) ---
    def basel_target_curve(scenario, tenors_years, R_parallel, R_short, R_long, x=4.0):
        t = np.asarray(tenors_years, float)
        short_shape = np.exp(-t / x)             # decays with maturity
        long_shape  = 1.0 - np.exp(-t / x)       # grows with maturity
        if scenario == "parallel_up":   return  +R_parallel * np.ones_like(t)
        if scenario == "parallel_down": return  -R_parallel * np.ones_like(t)
        if scenario == "short_up":      return  +R_short    * short_shape
        if scenario == "short_down":    return  -R_short    * short_shape
        if scenario == "steepener":     return  (-R_short * short_shape) + (+R_long * long_shape)
        if scenario == "flattener":     return  (+R_short * short_shape) + (-R_long * long_shape)
        raise ValueError(f"Unknown scenario: {scenario}")

    # --- 2) Pull unit-shock impact (bps) and target (bps) ---
    Y_unit_bps = _unit_curve_from_accepted(accepted_scen, var_order, M, bps=True)  # shape (D, K)
    y_star_bps = basel_target_curve(scenario, tenors_years,
                                    R_parallel=basel_params["R_parallel"],
                                    R_short=basel_params["R_short"],
                                    R_long=basel_params["R_long"],
                                    x=basel_params.get("x", 4.0))

    # --- 3) Cos², α (LS), RMSE (after scaling) per draw ---
    D, K = Y_unit_bps.shape
    s2   = np.einsum("dk,dk->d", Y_unit_bps, Y_unit_bps)            # ||s||^2
    st   = np.einsum("dk,k->d",  Y_unit_bps, y_star_bps)            # s·t
    t2   = float(np.dot(y_star_bps, y_star_bps))                    # ||t||^2

    cos2 = np.where((s2 > min_den) & (t2 > 0.0), (st * st) / (s2 * t2), 0.0)
    alpha = np.where(s2 > min_den, st / s2, np.nan)

    # Optional: enforce direction using first tenor’s sign (if target has sign)
    if enforce_direction and y_star_bps[0] != 0.0:
        sign_mismatch = np.sign(alpha * Y_unit_bps[:, 0]) != np.sign(y_star_bps[0])
        alpha = np.where(sign_mismatch, -alpha, alpha)

    # Guard against tiny denominators and optional alpha bounds
    bad_den = s2 <= min_den
    if alpha_max is not None:
        bad_alpha = np.abs(alpha) > alpha_max
    else:
        bad_alpha = np.zeros_like(alpha, dtype=bool)

    # RMSE after scaling: sqrt(mean((t - α s)^2))
    resid = y_star_bps[None, :] - alpha[:, None] * Y_unit_bps
    rmse  = np.sqrt(np.mean(resid**2, axis=1))

    # --- 4) Keep mask ---
    keep = (~bad_den) & (~bad_alpha) & (cos2 >= cos2_min)
    if rmse_max is not None:
        keep &= (rmse <= rmse_max)

    # --- 5) Slice arrays in the accepted dict by first dimension when it matches D ---
    def slice_like(v):
        if isinstance(v, np.ndarray) and v.shape and v.shape[0] == D:
            return v[keep]
        return v

    filtered = {k: slice_like(v) for k, v in accepted_scen.items()}

    # --- 6) Stash diagnostics (full length, pre-filter) for auditability ---
    filtered["_shape_cos2"]      = cos2
    filtered["_shape_rmse"]      = rmse
    filtered["_alphas_unitfit"]  = alpha
    filtered["_s2_unit"]         = s2
    filtered["_keep_mask"]       = keep
    filtered["_kept_idx"]        = np.nonzero(keep)[0]
    filtered["_target_curve_bps"]= y_star_bps
    filtered["_cos2_min"]        = float(cos2_min)
    filtered["_rmse_max"]        = None if rmse_max is None else float(rmse_max)
    filtered["_alpha_max"]       = None if alpha_max is None else float(alpha_max)

    # Summary counts
    filtered["accepted"]     = int(np.sum(keep))
    filtered["accept_rate"]  = float(np.mean(keep)) if D > 0 else 0.0

    return filtered


# After your SR acceptance step:
# specs = build_sign_specs_basel_8var(var_order, strict=True, purity="parallel")  # e.g., stricter SR

# i_lev  = var_order.index("level")
# i_slp  = var_order.index("slope_10y_1y")
# specs_basel = build_sign_specs_basel_8var(strict=True, i_lev=i_lev, i_slp=i_slp)
accepted_basel = {sc: apply_sign_restrictions(cands, sp, sc, tol_zero=1e-4, max_accept=3000)
                  for sc, sp in specs_basel.items()}

for scen in list(specs_basel.keys()):
    print(scen, ": accepted", accepted_basel[scen]["accepted"], "of", accepted_basel[scen]["tried"],
    f"(rate={accepted_basel[scen]['accept_rate']:.2%})")

# Recommended thresholds (tune to taste)
thresholds = {
    "parallel_up":   dict(cos2_min=0.9, rmse_max=25.0),
    "parallel_down": dict(cos2_min=0.9, rmse_max=25.0),
    "steepener":     dict(cos2_min=0.8, rmse_max=35.0),
    "flattener":     dict(cos2_min=0.8, rmse_max=35.0),
    "short_up":      dict(cos2_min=0.9, rmse_max=20.0),
    "short_down":    dict(cos2_min=0.9, rmse_max=20.0),
}

# Basel magnitudes (bps) for your currency:
# basel_mag = {...}  # as you set earlier

accepted_basel_tight = {}
for scen, acc in accepted_basel.items():
    th = thresholds[scen]
    accepted_basel_tight[scen] = filter_by_curve_shape(
        acc, var_order, tenors_years, M, basel_mag[scen], scen,
        cos2_min=th["cos2_min"], rmse_max=th["rmse_max"]
    )
    
for scen in list(specs_basel.keys()):
    print(scen, ": accepted", accepted_basel_tight[scen]["accepted"], "of", accepted_basel_tight[scen]["tried"],
    f"(rate={accepted_basel_tight[scen]['accept_rate']:.2%})")

accepted_basel_tight['short_up']['IRFs'].shape

import numpy as np

def check_M_parallel_flatness(M, tenors_years, accepted_basel, var_order, scenario="parallel_up"):
    """
    Prints quick diagnostics:
      1) How flat is M's 'level' column across tenors?
      2) How flat are the *unit* impact curves implied by accepted draws (before scaling)?
    Interpreting: higher cos² (~≥0.95) and small RMSE (≤10–20 bps) = good parallel span.
    """
    t = np.asarray(tenors_years, float)
    ones = np.ones_like(t, float)

    # --- 1) M's level column vs flat ---
    i_lev  = var_order.index("level")
    i_slp  = var_order.index("slope_10y_1y")
    i_curv = var_order.index("curvature_ns_like")

    mL = np.asarray(M[:, 0], float)                   # shape (n_tenors,)
    # cos²(level column, all-ones)
    cos2_level = (mL @ ones)**2 / ((mL @ mL) * (ones @ ones) + 1e-12)
    # max relative deviation from its mean (how "bowed" the column is)
    mean_mL = np.mean(mL)
    rel_bow = np.max(np.abs(mL - mean_mL)) / (abs(mean_mL) + 1e-12)

    print(f"[M level]  cos² vs flat = {cos2_level:.3f}  |  max|dev|/|mean| = {100*rel_bow:.1f}%")

    # --- 2) Unit impact curves from accepted draws vs flat (in bps) ---
    acc = accepted_basel[scenario]
    if "IRFs" not in acc or len(acc["IRFs"]) == 0:
        print(f"[{scenario}] No accepted draws to check."); return

    IRFs = np.asarray(acc["IRFs"])            # (Nacc, H+1, K, K)
    imp  = IRFs[:, 0, :, 0]                   # (Nacc, K) unit impacts at t=0
    F    = np.column_stack([imp[:, i_lev], imp[:, i_slp], imp[:, i_curv]])  # (Nacc,3)
    Y_unit_bps = (F @ M.T) * 1e2             # (Nacc, n_tenors), in bps

    # cos² of each draw vs flat vector
    num   = Y_unit_bps @ ones
    denom = np.sqrt((Y_unit_bps**2).sum(axis=1)) * np.sqrt(ones @ ones) + 1e-12
    cos2_draws = (num / denom)**2

    # RMSE vs best constant (i.e., remove the mean across tenors)
    resid = Y_unit_bps - Y_unit_bps.mean(axis=1, keepdims=True)
    rmse_const = np.sqrt(np.mean(resid**2, axis=1))  # bps

    def p(x, q): return float(np.percentile(x, q))
    print(f"[{scenario} unit curves]  cos² vs flat  median={p(cos2_draws,50):.3f}  "
          f"(p10={p(cos2_draws,10):.3f}, p90={p(cos2_draws,90):.3f})")
    print(f"[{scenario} unit curves]  RMSE vs const (bps)  median={p(rmse_const,50):.1f}  "
          f"(p10={p(rmse_const,10):.1f}, p90={p(rmse_const,90):.1f})")

    # Quick rule-of-thumb verdicts
    ok_level = (cos2_level >= 0.95) and (rel_bow <= 0.10)     # ≤10% bow
    ok_unit  = (p(cos2_draws,50) >= 0.95) and (p(rmse_const,50) <= 15.0)
    print(f"Verdict: M level column {'OK' if ok_level else 'needs work'}; "
          f"parallel unit curves {'OK' if ok_unit else 'need work'}.")

check_M_parallel_flatness(M, tenors_years, accepted_basel_tight, var_order, scenario="parallel_up")

import numpy as np

# std devs (decimals) from your panel, matching var_order
std_by_var = panel[var_order].std(ddof=1).to_numpy()

i_lev  = var_order.index("level")
i_slp  = var_order.index("slope_10y_1y")
i_curv = var_order.index("curvature_ns_like")

def unit_curves_bps_rescaled(accepted_scen, var_order, M, std_by_var):
    """Unit impact curves in tenor space (bps) after unstandardizing factors."""
    IRFs = np.asarray(accepted_scen["IRFs"])         # (Nacc, H+1, K, K)
    imp  = IRFs[:, 0, :, 0]                          # (Nacc, K) unit impacts at t=0 (σ-units if standardized)
    # unstandardize ONLY the factors used in the curve map:
    F_sigma = np.c_[imp[:, i_lev], imp[:, i_slp], imp[:, i_curv]]   # (Nacc, 3)
    F_dec   = F_sigma * std_by_var[[i_lev, i_slp, i_curv]]          # (Nacc, 3) → decimals
    Y_dec   = F_dec @ M.T                                           # decimals
    return Y_dec * 1e4                                             # bps


# unit_curves_bps_rescaled(accepted_basel_tight['parallel_up'], var_order, M, std_by_var)

_, _, _, df_irf_basel  = summarize_irfs(accepted_basel_tight["parallel_up"],  var_order, H=20)

plot_irf_filtered(df_irf_basel[df_irf_basel['shock'] == "parallel_up"], "parallel_up", "infl_q_ann", band=(10,90))

# # var_order used in your VAR:
# var_order = [
#     "infl_q_ann","gdp_q_ann","gg_deficit_pct_gdp","gg_debt_pct_gdp",
#     "policy_rate","level","slope_10y_1y","curvature_ns_like"
# ]

# # Example: build tables
# table_parallel_up = diag_table(accepted_basel_tight["parallel_up"],  var_order, "Parallel up")
# table_parallel_down = diag_table(accepted_basel_tight["parallel_down"],  var_order, "Parallel down")
# table_steepener = diag_table(accepted_basel_tight["steepener"],  var_order, "Steepener")
# table_flattener = diag_table(accepted_basel_tight["flattener"],  var_order, "Flattener")
# table_short_up = diag_table(accepted_basel_tight["short_up"],  var_order, "Short up")
# table_short_down = diag_table(accepted_basel_tight["short_down"],  var_order, "Short down")

# # Combine into one table for display
# diag_tables_basel = pd.concat([table_parallel_up, table_parallel_down, table_steepener, table_flattener, table_short_up, table_short_down], ignore_index=True)
# diag_tables_basel

# # Compose heatmap input and plot
# impact_M, shocks, vars_ = build_impact_matrix(
#     {"Parallel up": table_parallel_up, "Parallel down": table_parallel_down},
#     var_order
# )
# plot_impact_heatmap(impact_M, shocks, vars_, title="Median (h=0) responses by shock")


# accepted_basel_tight["parallel_up"]
# results['parallel_up']

import numpy as np
import matplotlib.pyplot as plt

# --- helper: build Basel target (bps) ---
def _basel_target_curve(scenario, tenors_years, R_parallel=200.0, R_short=250.0, R_long=100.0, x=4.0):
    t = np.asarray(tenors_years, float)
    short_shape = np.exp(-t / x)
    long_shape  = 1.0 - np.exp(-t / x)
    if scenario == "parallel_up":   return  +R_parallel * np.ones_like(t)
    if scenario == "parallel_down": return  -R_parallel * np.ones_like(t)
    if scenario == "short_up":      return  +R_short    * short_shape
    if scenario == "short_down":    return  -R_short    * short_shape
    if scenario == "steepener":     return  (-R_short * short_shape) + ( +R_long * long_shape)
    if scenario == "flattener":     return  (+R_short * short_shape) + ( -R_long * long_shape)
    raise ValueError(scenario)

# --- helper: unit impact curves in tenor space (bps) from accepted dict ---
def _unit_curves_bps(accepted_scen, var_order, M):
    i_lev  = var_order.index("level")
    i_slp  = var_order.index("slope_10y_1y")
    i_curv = var_order.index("curvature_ns_like")
    IRFs   = np.asarray(accepted_scen["IRFs"])       # (Nacc, H+1, K, K) (identified shock in col 0)
    imp    = IRFs[:, 0, :, 0]                        # (Nacc, K) at t=0
    F      = np.column_stack([imp[:, i_lev], imp[:, i_slp], imp[:, i_curv]])  # (Nacc, 3)
    Y_dec  = F @ M.T                                 # decimals
    return 1e4 * Y_dec                               # → bps

# --- main: 4 diagnostics for one scenario ---
def plot_basel_curve_fit_diagnostics(
    scenario, accepted_basel_dict, var_order, tenors_years, M, basel_mag,
    th_cos2=None, th_rmse=None, n_examples=8, seed=0
):
    # Pull accepted set (already filtered, if you passed accepted_basel_tight)
    acc = accepted_basel_dict[scenario]
    if "IRFs" not in acc or len(acc["IRFs"]) == 0:
        print(f"[{scenario}] No accepted draws.")
        return

    rng = np.random.default_rng(seed)
    Y_unit = _unit_curves_bps(acc, var_order, M)     # (Nacc, n_tenors)
    y_star = _basel_target_curve(scenario, tenors_years, **basel_mag[scenario])  # (n_tenors,)

    # α (through-origin LS in bps), fitted curves, cos², RMSE
    num    = (Y_unit * y_star[None, :]).sum(axis=1)
    denom  = (Y_unit**2).sum(axis=1)
    alphas = np.divide(num, denom, out=np.zeros_like(num), where=denom>0)

    fitted = alphas[:, None] * Y_unit
    dot    = (fitted * y_star[None, :]).sum(axis=1)
    normf  = np.sqrt((fitted**2).sum(axis=1)) + 1e-12
    normt  = float(np.linalg.norm(y_star) + 1e-12)
    cos2   = (dot / (normf * normt))**2
    rmse   = np.sqrt(np.mean((fitted - y_star[None, :])**2, axis=1))

    # 1) Target vs fitted (few draws)
    plt.figure(figsize=(6,4))
    idx = rng.choice(len(alphas), size=min(n_examples, len(alphas)), replace=False)
    for i in idx:
        plt.plot(tenors_years, fitted[i], lw=1, alpha=0.8)
    plt.plot(tenors_years, y_star, lw=3, label="Basel target")
    plt.plot(tenors_years, np.median(fitted, axis=0), lw=3, label="Fitted median")
    plt.title(f"{scenario}: target vs fitted impact curves")
    plt.xlabel("Tenor (years)"); plt.ylabel("bps"); plt.legend(); plt.tight_layout(); plt.show()

    # 2) Residual-by-tenor “butterfly”
    resid = fitted - y_star[None, :]
    plt.figure(figsize=(6,4))
    plt.errorbar(tenors_years, resid.mean(0), yerr=resid.std(0), fmt="o-")
    plt.axhline(0, color="k", lw=1)
    plt.title(f"{scenario}: impact residuals (mean ±1σ)")
    plt.xlabel("Tenor (years)"); plt.ylabel("bps"); plt.tight_layout(); plt.show()

    # 3) cos² vs RMSE scatter (optionally draw your cut lines)
    plt.figure(figsize=(6,4))
    plt.scatter(cos2, rmse, s=14, alpha=0.6)
    if th_cos2 is not None: plt.axvline(th_cos2, color="r", ls="--", lw=1)
    if th_rmse is not None: plt.axhline(th_rmse, color="r", ls="--", lw=1)
    plt.xlabel("cos² (shape match)"); plt.ylabel("RMSE (bps)")
    plt.title(f"{scenario}: fit quality"); plt.tight_layout(); plt.show()

    # 4) α histogram
    plt.figure(figsize=(6,4))
    plt.hist(alphas, bins=30, edgecolor="k")
    plt.title(f"{scenario}: α distribution")
    plt.xlabel("α (unit → Basel scale)"); plt.ylabel("count")
    plt.tight_layout(); plt.show()

    # (Optional) quick text summary
    med = lambda a: float(np.median(a))
    p10 = lambda a: float(np.percentile(a,10))
    p90 = lambda a: float(np.percentile(a,90))
    print(f"[{scenario}] kept={len(alphas)} | α med/p10/p90 = {med(alphas):.2f}/{p10(alphas):.2f}/{p90(alphas):.2f} | "
          f"cos² med={med(cos2):.3f} | RMSE med={med(rmse):.1f} bps")


# unit curve magnitude sanity check
Y_unit_bps = _unit_curves_bps(accepted_basel["parallel_up"], var_order, M)  # as you computed
j10 = int(np.argmin(np.abs(np.asarray(tenors_years) - 10.0)))
print("median |unit| move @10y (bps):", float(np.median(np.abs(Y_unit_bps[:, j10]))))


scen = "steepener"  # or any of: parallel_up, parallel_down, steepener, flattener, short_up, short_down
plot_basel_curve_fit_diagnostics(
    scen,
    accepted_basel_tight,          # or accepted_basel if you didn’t apply the shape gate yet
    var_order, tenors_years, M, basel_mag,
    th_cos2=thresholds[scen]["cos2_min"],
    th_rmse=thresholds[scen]["rmse_max"],
    n_examples=80, seed=0
)


import numpy as np, pandas as pd, matplotlib.pyplot as plt

def plot_acceptance_funnel(accepted_pre, accepted_post=None, scenarios=None):
    """
    accepted_pre : dict[scenario] -> accepted dict from SR (before shape-gate)
    accepted_post: dict[scenario] -> filtered dict (after shape-gate). If None, uses pre.
    """
    if scenarios is None:
        scenarios = list(accepted_pre.keys())

    rows = []
    for scen in scenarios:
        pre = accepted_pre[scen]
        tried = int(pre.get("tried", pre["IRFs"].shape[0]))
        sr_ok = pre["IRFs"].shape[0]
        kept  = (accepted_post.get(scen, pre)["IRFs"].shape[0]
                 if accepted_post is not None else sr_ok)
        rows.append(dict(scenario=scen, tried=tried, sr_ok=sr_ok, kept=kept))
    df = pd.DataFrame(rows)

    # grouped bars
    x = np.arange(len(df)); w = 0.28
    plt.figure(figsize=(8,4))
    plt.bar(x- w, df["tried"], width=w, label="candidates tried")
    plt.bar(x+0, df["sr_ok"],  width=w, label="SR accepted")
    plt.bar(x+ w, df["kept"],  width=w, label="shape-gate kept")
    plt.xticks(x, df["scenario"], rotation=15)
    plt.ylabel("count"); plt.title("Acceptance funnel by scenario")
    plt.legend(); plt.tight_layout(); plt.show()
    return df

funnel_df = plot_acceptance_funnel(accepted_basel, accepted_basel_tight)

# import numpy as np, pandas as pd

# def bottleneck_report(cands, spec_one_shock, var_order, tol_zero=1e-4):
#     """
#     For each constraint (var, (h0,h1,sgn)) in spec_one_shock, compute the share of candidates
#     for which that constraint is *impossible* to satisfy (no column/orientation meets it).
#     Returns a DataFrame with fail_rate per constraint and by variable.
#     """
#     IRF_all = np.asarray(cands["IRFs"])  # (N, H+1, K, K)
#     N, H1, K, _ = IRF_all.shape

#     # list constraints as tuples: (var_idx, h0, h1, sgn)
#     cons = []
#     for i_var, spans in spec_one_shock.items():
#         for (h0,h1,sgn) in spans:
#             cons.append((i_var, h0, h1, sgn))

#     def ok_seg(seg, sgn):
#         if sgn == +1: return np.all(seg >= -tol_zero)
#         if sgn == -1: return np.all(seg <=  tol_zero)
#         if sgn ==  0: return np.all(np.abs(seg) <= tol_zero)
#         return False

#     fail_counts = {c: 0 for c in cons}
#     for n in range(N):
#         irf = IRF_all[n]  # (H+1,K,K)
#         for c in cons:
#             i_var, h0, h1, sgn = c
#             # Is there ANY column/orientation that satisfies just this constraint?
#             feasible = False
#             for j in range(K):
#                 for flip in (+1, -1):
#                     seg = flip * irf[h0:h1+1, i_var, j]
#                     if ok_seg(seg, sgn):
#                         feasible = True; break
#                 if feasible: break
#             if not feasible:
#                 fail_counts[c] += 1

#     rows = []
#     for (i_var, h0, h1, sgn), cnt in fail_counts.items():
#         rows.append(dict(
#             var=var_order[i_var], h0=h0, h1=h1, sign=sgn,
#             fail_count=cnt, tried=N, fail_rate=cnt/max(1,N)
#         ))
#     df = pd.DataFrame(rows).sort_values(["fail_rate","var","h0","h1"], ascending=[False,True,True,True])

#     # variable-level summary
#     var_sum = (df.groupby("var")["fail_rate"]
#                  .mean().rename("avg_fail_rate")
#                  .reset_index().sort_values("avg_fail_rate", ascending=False))
#     return df, var_sum

# # Example, per scenario:
# spec = specs_basel["steepener"]  # the SR spec you used for that scenario
# cons_df, var_summary = bottleneck_report(cands, spec, var_order, tol_zero=1e-4)
# display(cons_df) ; display(var_summary)

# # Optional heatmap:
# import seaborn as sns;  # if allowed in your env, else pivot + imshow
# piv = cons_df.pivot_table(index="var", columns=["h0","h1","sign"], values="fail_rate", aggfunc="mean")
# sns.heatmap(piv, cmap="magma", vmin=0, vmax=1)

import numpy as np, matplotlib.pyplot as plt

def _basel_target_curve(scenario, tenors_years, R_parallel, R_short, R_long, x=4.0):
    t = np.asarray(tenors_years, float)
    short = np.exp(-t/x); long = 1.0 - np.exp(-t/x)
    if scenario=="parallel_up":   return +R_parallel*np.ones_like(t)
    if scenario=="parallel_down": return -R_parallel*np.ones_like(t)
    if scenario=="short_up":      return +R_short*short
    if scenario=="short_down":    return -R_short*short
    if scenario=="steepener":     return (-R_short*short) + (+R_long*long)
    if scenario=="flattener":     return (+R_short*short) + (-R_long*long)
    raise ValueError(scenario)

def impact_neutrality_plot(accepted_scen, scenario, var_order, tenors_years, M, basel_params):
    """
    Recompute α per kept draw; show distributions of inflation & GDP impact (h=0), Basel-scaled.
    Units: macro plotted in pp (annualized).
    """
    i_pi  = var_order.index("infl_q_ann")
    i_gdp = var_order.index("gdp_q_ann")
    i_lev = var_order.index("level"); i_slp = var_order.index("slope_10y_1y"); i_curv = var_order.index("curvature_ns_like")

    IRFs = np.asarray(accepted_scen["IRFs"])         # (Nkeep,H+1,K,K)
    imp  = IRFs[:,0,:,0]                             # (Nkeep,K) unit impacts at t=0

    # factor → tenor (bps) for unit shock
    F = np.column_stack([imp[:, i_lev], imp[:, i_slp], imp[:, i_curv]])  # (Nkeep,3)
    Y_unit_bps = F @ M.T * 1e4
    y_star_bps = _basel_target_curve(scenario, tenors_years, **basel_params)

    # α per draw (through-origin LS, in bps space)
    num   = (Y_unit_bps * y_star_bps[None,:]).sum(axis=1)
    denom = (Y_unit_bps**2).sum(axis=1)
    alphas = np.divide(num, denom, out=np.zeros_like(num), where=denom>0)

    # Basel-scaled macro impacts at h=0 (pp ann.)
    imp_macro_scaled = (alphas[:,None] * imp)[:, [i_pi, i_gdp]]
    names = ["inflation (pp, ann.)", "GDP growth (pp, ann.)"]

    plt.figure(figsize=(6,4))
    for k,name in enumerate(names):
        x = imp_macro_scaled[:,k]
        med = np.median(x); p10, p90 = np.percentile(x,[10,90])
        plt.errorbar([k], [med], yerr=[[med-p10],[p90-med]], fmt="o", capsize=4, label=name)
    plt.axhline(0,color="k",lw=1)
    plt.xticks([0,1], names, rotation=15)
    plt.ylabel("Impact at h=0 (pp, annualized)")
    plt.title(f"{scenario}: impact neutrality (π & y at t=0)")
    plt.tight_layout(); plt.show()

# Example:
impact_neutrality_plot(accepted_basel_tight["parallel_up"], "parallel_up",
                       var_order, tenors_years, M, basel_mag["parallel_up"])

import numpy as np
import matplotlib.pyplot as plt

def _basel_target_curve(scenario, tenors_years, R_parallel, R_short, R_long, x=4.0):
    t = np.asarray(tenors_years, float)
    short = np.exp(-t/x); long = 1.0 - np.exp(-t/x)
    if scenario=="parallel_up":   return +R_parallel*np.ones_like(t)
    if scenario=="parallel_down": return -R_parallel*np.ones_like(t)
    if scenario=="short_up":      return +R_short*short
    if scenario=="short_down":    return -R_short*short
    if scenario=="steepener":     return (-R_short*short) + (+R_long*long)
    if scenario=="flattener":     return (+R_short*short) + (-R_long*long)
    raise ValueError(scenario)

def plot_factor_contrib_bars_at_impact(
    scenario, accepted_scen, var_order, tenors_years, M, basel_params,
    tenors_to_show=(0.25, 1, 5, 10), seed=0
):
    """
    accepted_scen: one scenario dict (post shape-gate), with IRFs reordered so identified shock is col 0
    M: (n_tenors, 3) factor→tenor map
    tenors_to_show: tuple of tenors (years) to plot
    """
    rng = np.random.default_rng(seed)
    i_lev  = var_order.index("level")
    i_slp  = var_order.index("slope_10y_1y")
    i_curv = var_order.index("curvature_ns_like")

    IRFs = np.asarray(accepted_scen["IRFs"])        # (Nkeep,H+1,K,K)
    if IRFs.size == 0:
        print(f"[{scenario}] No kept draws."); return
    impK = IRFs[:,0,:,0]                            # (Nkeep,K) unit impact at t=0

    # factors (unit shock) and unit impact curves in bps
    F_unit = np.c_[impK[:,i_lev], impK[:,i_slp], impK[:,i_curv]]   # (Nkeep,3)
    Y_unit_bps = F_unit @ M.T * 1e4                                 # (Nkeep, n_tenors)

    # Basel target (bps) + per-draw alpha (through-origin LS)
    y_star_bps = _basel_target_curve(scenario, tenors_years, **basel_params)
    num   = (Y_unit_bps * y_star_bps[None,:]).sum(axis=1)
    denom = (Y_unit_bps**2).sum(axis=1)
    alphas = np.divide(num, denom, out=np.zeros_like(num), where=denom>0)   # (Nkeep,)

    # factor contributions at each tenor = alpha * factor_component * loading_row
    # For tenor row r, contribution of factor f is: alpha * F_unit[:,f] * M[r,f] * 1e4 (bps)
    tenors_years = np.asarray(tenors_years, float)
    idxs = [int(np.argmin(np.abs(tenors_years - x))) for x in tenors_to_show]
    labels = [f"{tenors_years[j]:g}y" for j in idxs]

    contrib_level = []
    contrib_slope = []
    contrib_curv  = []
    target        = []
    for j in idxs:
        cL = alphas * (F_unit[:,0] * M[j,0] * 1e4)
        cS = alphas * (F_unit[:,1] * M[j,1] * 1e4)
        cC = alphas * (F_unit[:,2] * M[j,2] * 1e4)
        contrib_level.append(np.percentile(cL, [10,50,90]))
        contrib_slope.append(np.percentile(cS, [10,50,90]))
        contrib_curv.append(np.percentile(cC, [10,50,90]))
        target.append(y_star_bps[j])

    contrib_level = np.array(contrib_level)  # (T,3) p10/50/90
    contrib_slope = np.array(contrib_slope)
    contrib_curv  = np.array(contrib_curv)
    target        = np.array(target)

    # stacked bars: median contributions; overlay target as markers; whiskers for totals
    med_total = contrib_level[:,1] + contrib_slope[:,1] + contrib_curv[:,1]
    p10_total = (contrib_level[:,0] + contrib_slope[:,0] + contrib_curv[:,0])
    p90_total = (contrib_level[:,2] + contrib_slope[:,2] + contrib_curv[:,2])

    x = np.arange(len(labels))
    plt.figure(figsize=(7,4))
    plt.bar(x, contrib_level[:,1], label="level (median)")
    plt.bar(x, contrib_slope[:,1], bottom=contrib_level[:,1], label="slope (median)")
    plt.bar(x, contrib_curv[:,1], bottom=contrib_level[:,1]+contrib_slope[:,1], label="curvature (median)")
    # total whiskers
    plt.errorbar(x, med_total, yerr=[med_total-p10_total, p90_total-med_total], fmt="none", ecolor="k", capsize=4, lw=1)
    # target dots
    plt.plot(x, target, "ko", label="Basel target (bps)")
    plt.xticks(x, labels)
    plt.ylabel("Impact at t=0 (bps)")
    plt.title(f"{scenario}: factor contributions at impact")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout(); plt.show()


def tenor_path_fan(
    scenario, paths, var_order, M, tenors_years, base_curve,
    tenors_to_show=(0.25, 1, 5, 10), quantiles=(10,50,90), units="pp"
):
    """
    paths: (Nkeep, H, K) scaled scenario paths in VAR units (pp if units='pp')
    base_curve: (n_tenors,) current curve in *decimals*
    Plots median + 10–90% for each selected tenor, in decimals.
    """
    import numpy as np, matplotlib.pyplot as plt
    i_lev  = var_order.index("level")
    i_slp  = var_order.index("slope_10y_1y")
    i_curv = var_order.index("curvature_ns_like")

    N, H, K = paths.shape
    # factor paths → tenor deviations (same units as VAR)
    F = np.stack([paths[:,:,i_lev], paths[:,:,i_slp], paths[:,:,i_curv]], axis=-1)   # (N,H,3)
    Y_dev_units = F @ M.T                                                             # (N,H,n_tenors)

    # convert to decimals before adding base
    if units == "pp":
        Y_dev_dec = Y_dev_units
    else:  # units == "dec"
        Y_dev_dec = Y_dev_units / 100.0

    Y = Y_dev_dec + base_curve[None, None, :]  # levels (decimals)

    tenors_years = np.asarray(tenors_years, float)
    idxs = [int(np.argmin(np.abs(tenors_years - x))) for x in tenors_to_show]
    labels = [f"{tenors_years[j]:g}y" for j in idxs]

    t = np.arange(1, H+1)
    qlo, qmed, qhi = quantiles
    plt.figure(figsize=(8,5))
    for j, lab in zip(idxs, labels):
        series = Y[:,:,j]  # (N,H)
        lo  = np.percentile(series, qlo, axis=0)
        med = np.percentile(series, qmed, axis=0)
        hi  = np.percentile(series, qhi, axis=0)
        plt.fill_between(t, lo, hi, alpha=0.15)
        plt.plot(t, med, lw=2, label=lab)

    plt.xlabel("h (quarters)")
    plt.ylabel("Yield (level, decimal)")
    plt.title(f"{scenario}: tenor paths (median & {qlo}–{qhi}%)")
    plt.legend()
    plt.tight_layout(); plt.show()


# A) factor contributions at impact (use your post shape-gate accepted set)
scenario = "steepener"

plot_factor_contrib_bars_at_impact(
    scenario,
    accepted_basel_tight[scenario],
    var_order, tenors_years, M, basel_mag[scenario],
    tenors_to_show=(1, 5, 10)
)

# B) tenor path fans (use your *scaled* paths for that scenario)
# e.g., from results = run_all_basel_scenarios(...): results[scen]["paths"]
tenor_cols = ["yc_spot_1y","yc_spot_5y","yc_spot_10y"]  # decimals
current_curve_levels = panel[tenor_cols].iloc[-1].to_numpy(dtype=float)

tenor_path_fan(
    scenario,
    results[scenario]["paths"],   # (Nkeep, H, K) scaled
    var_order, M, tenors_years,
    base_curve=current_curve_levels,   # (n_tenors,), decimal
    tenors_to_show=(1, 5, 10)
)

import numpy as np
import pandas as pd

def elasticity_table_from_irfs(
    scenario, accepted_scen, var_order,
    alphas, y_star_bps, tenors_years,
    anchor_tenor=None, peak_window_4=4, peak_window_8=8,
    vars_to_report=("infl_q_ann","gdp_q_ann","policy_rate","gg_deficit_pct_gdp","gg_debt_pct_gdp")
):
    """
    Returns a tidy table of elasticities (per 100 bps at the anchor tenor) for:
      - impact (h=0),
      - peak within 4 quarters (h=1..4),
      - peak within 8 quarters (h=1..8).
    Uses scaled IRFs: scaled_irf = alpha * IRF_unit.
    Units: macro vars are in their native units (e.g., pp annualized).
    """

    # Choose sensible default anchor by scenario if not given
    if anchor_tenor is None:
        if scenario in ("parallel_up","parallel_down","steepener","flattener"):
            anchor_tenor = 10.0     # 10y
        elif scenario in ("short_up","short_down"):
            anchor_tenor = 0.25     # 3m
        else:
            anchor_tenor = 10.0

    # Denominator: target move at the anchor tenor (bps)
    t = np.asarray(tenors_years, float)
    j_anchor = int(np.argmin(np.abs(t - anchor_tenor)))
    anchor_bps = float(abs(y_star_bps[j_anchor]))
    if anchor_bps < 1e-9:
        raise ValueError("Anchor Basel move is ~0 bps; pick a different anchor tenor.")

    # Pull IRFs and scale by alpha per draw (shock column must be col 0 after your reorder)
    IRFs = np.asarray(accepted_scen["IRFs"])   # (Nacc, H+1, K, K)
    Nacc, H1, K, _ = IRFs.shape

    # Helper to compute signed peak over window {1..W} (use argmax(|.|))
    def signed_peak(vec, W):
        if W >= len(vec): W = len(vec)-1
        seg = vec[1:W+1]
        if seg.size == 0: return 0.0
        idx = int(np.argmax(np.abs(seg)))
        return float(seg[idx])

    rows = []
    for var in vars_to_report:
        i = var_order.index(var)
        impacts = []
        peaks4  = []
        peaks8  = []
        for d in range(Nacc):
            irf_unit = IRFs[d, :, i, 0]           # (H+1,)
            scaled   = alphas[d] * irf_unit       # per-draw α
            impacts.append(scaled[0])
            peaks4.append(signed_peak(scaled, peak_window_4))
            peaks8.append(signed_peak(scaled, peak_window_8))

        impacts = np.array(impacts)
        peaks4  = np.array(peaks4)
        peaks8  = np.array(peaks8)

        # Elasticities per 100 bps at anchor tenor
        per100 = 100.0 / anchor_bps
        e_imp  = per100 * impacts
        e_p4   = per100 * peaks4
        e_p8   = per100 * peaks8

        def qstats(x):
            return np.percentile(x, [10,50,90])

        r_imp = qstats(e_imp); r_p4 = qstats(e_p4); r_p8 = qstats(e_p8)
        rows.append(dict(
            variable=var,
            impact_med=r_imp[1], impact_p10=r_imp[0], impact_p90=r_imp[2],
            peak4_med=r_p4[1],  peak4_p10=r_p4[0],  peak4_p90=r_p4[2],
            peak8_med=r_p8[1],  peak8_p10=r_p8[0],  peak8_p90=r_p8[2],
            anchor_tenor_years=anchor_tenor, anchor_move_bps=anchor_bps
        ))

    cols_order = ["variable",
                  "impact_med","impact_p10","impact_p90",
                  "peak4_med","peak4_p10","peak4_p90",
                  "peak8_med","peak8_p10","peak8_p90",
                  "anchor_tenor_years","anchor_move_bps"]
    return pd.DataFrame(rows)[cols_order]


scen = "parallel_up"
acc  = accepted_basel_tight[scen]             # post shape-gate set
alph = acc["_alphas_unitfit"]                 # or recompute per your α function
ystar = basel_target_curve(scen, tenors_years, **basel_mag[scen])  # bps

etable = elasticity_table_from_irfs(
    scen, acc, var_order,
    alphas=alph, y_star_bps=ystar, tenors_years=tenors_years,
    anchor_tenor=None,   # or 10.0 / 0.25 explicitly
    vars_to_report=("infl_q_ann","gdp_q_ann","policy_rate","gg_deficit_pct_gdp","gg_debt_pct_gdp")
)
etable
