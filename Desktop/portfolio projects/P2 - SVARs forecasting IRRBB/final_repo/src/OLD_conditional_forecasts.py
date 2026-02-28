import pandas as pd, numpy as np
import matplotlib.pyplot as plt

### UNUSED CONDITIONAL FORECASTS FUNCTIONS

def basel_scenario_to_forecasts(df_all: pd.DataFrame, scenario: str, var_order: list, H: int):
    """
    Faster: extract ONE scenario and return (N_draws, H, K) forecasts.
    df_all is the long Basel DF from make_ecb_basel6()["all"].
    """
    df_s = df_all[df_all["scenario"] == scenario].copy()
    if df_s.empty:
        raise ValueError(f"Scenario {scenario!r} not found in df_all.")

    # create a compact draw id: (post_draw, draw) -> 0..N-1
    df_s["global_draw"], _ = pd.factorize(list(zip(df_s["post_draw"], df_s["draw"])))
    K = len(var_order)
    N = df_s["global_draw"].max() + 1

    # we want to end up with (N, H, K)
    # we'll fill a 3D array directly
    forecasts = np.full((N, H, K), np.nan, dtype=float)

    # get numpy views to avoid pandas-in-the-loop
    gdraw = df_s["global_draw"].to_numpy()
    h     = df_s["h"].to_numpy() - 1   # 0-based
    v     = df_s["var"].to_numpy()
    y     = df_s["yhat"].to_numpy()

    # map var names -> indices once
    v2i = {vname: i for i, vname in enumerate(var_order)}

    for i in range(df_s.shape[0]):
        vi = v2i.get(v[i])
        if vi is None:
            continue
        hi = h[i]
        di = gdraw[i]
        if 0 <= hi < H:
            forecasts[di, hi, vi] = y[i]

    return forecasts

def basel_to_accepted_dict_fast(df_all, var_order, H):
    accepted = {}
    for scen in df_all["scenario"].unique():
        forecasts = basel_scenario_to_forecasts(df_all, scen, var_order, H)
        accepted[scen] = {
            "B": None,
            "c": None,
            "forecasts": forecasts,
        }
    return accepted

def make_basel_fan_df_from_paths(paths, var_order, last_date, q_lo=10, q_hi=90):
    """
    paths: np.ndarray (N, H, K)
    var_order: list of variable names, length K
    last_date: pandas Timestamp (last obs in your panel)
    returns: DataFrame with columns
        ["date","var","h","yhat","yhat_lo","yhat_hi"]
    compatible with plot_conditional_forecast(...)
    """
    N, H, K = paths.shape

    # quantiles over draws
    med = np.nanmedian(paths, axis=0)                   # (H, K)
    lo  = np.nanpercentile(paths, q_lo, axis=0)         # (H, K)
    hi  = np.nanpercentile(paths, q_hi, axis=0)         # (H, K)

    dates = pd.date_range(last_date, periods=H+1, freq="QE")[1:]

    rows = []
    for k, name in enumerate(var_order):
        rows.append(pd.DataFrame({
            "date": dates,
            "var": name,
            "h": np.arange(1, H+1),
            "yhat": med[:, k],
            "yhat_lo": lo[:, k],
            "yhat_hi": hi[:, k],
        }))
    fan_df = pd.concat(rows, ignore_index=True)
    return fan_df




### OLD FUNCTIONS

# Delta vs baseline

def delta_fan(fan_cond, fan_base, var):
    a = fan_cond[fan_cond["var"]==var].set_index("date")[["yhat","yhat_lo","yhat_hi"]]
    b = fan_base[fan_base["var"]==var].set_index("date")[["yhat","yhat_lo","yhat_hi"]]
    out = a.subtract(b); out["var"]=var; return out.reset_index()


# Reduced form conditional (min-var) vs SVAR

import numpy as np

def rf_minvar_impact(As, c, Sigma, y_hist, H, target):
    """
    One-step minimal-variance RF conditioning:
    y1_pred = c + sum_j A_j y_{0-j}; choose u0 to minimize u0'Σ^{-1}u0 s.t. H(y1_pred+u0)=target.
    Returns u0 and y1 = y1_pred + u0.
    - As: list [A1,..,Ap] each (K,K)
    - c: (K,)
    - Sigma: (K,K)
    - y_hist: (p,K) history with y0 on row 0 (most recent), then y_-1, ...
    - H: (m,K) maps state to constrained quantities (e.g., 10y yield)
    - target: (m,)
    """
    K = c.shape[0]
    y1_pred = c.copy()
    for j, Aj in enumerate(As):
        y1_pred += Aj @ y_hist[j]                 # note: y_hist[j] is y_-j (row j)

    d = target - (H @ y1_pred)                    # required adjustment in constrained space
    # u0* = Σ H' (H Σ H')^{-1} d
    HSigma = H @ Sigma
    G = HSigma @ H.T
    u0 = Sigma @ H.T @ np.linalg.solve(G, d)
    y1 = y1_pred + u0
    return u0, y1


def H_for_anchor_tenor(var_order, tenors_years, M, anchor=10.0):
    iL = var_order.index("level"); iS = var_order.index("slope_10y_1y"); iC = var_order.index("curvature_ns_like")
    S = np.zeros((3, len(var_order))); S[0,iL]=S[1,iS]=S[2,iC]=1.0
    j = int(np.argmin(np.abs(np.asarray(tenors_years)-anchor)))
    e = np.zeros((1, M.shape[0])); e[0, j] = 1.0
    H = e @ M @ S                              # shape (1,K); in VAR units (pp)
    return H


# Pick one accepted Basel draw to build RF-vs-SVAR comparison on the same history
acc = accepted_basel_tight["parallel_up"]   # example
B = acc["B"][0]; Sigma = acc["Sigma"][0]; c = acc.get("c", np.zeros(B.shape[0]))
K = B.shape[0]; p = B.shape[1] // K
As = [B[:, j*K:(j+1)*K] for j in range(p)]

# history (y0, y-1, ... in rows 0..p-1), using your panel (pp units)
hist = panel[var_order].dropna().to_numpy()[-p:, :][::-1]  # rows: y0,y-1,...

# RF minimal-variance impact to hit +200 bps at 10y (pp: +2.00)
H = H_for_anchor_tenor(var_order, tenors_years, M, anchor=10.0)
target_pp = np.array([+2.00])   # +200 bps = +2.00 pp in your VAR units
u0, y1_rf = rf_minvar_impact(As, c if c.ndim==1 else c[0], Sigma, hist, H, target_pp)

# Build RF path: impact step uses u0, then zero shocks onward (same as your SVAR with shock_horizon=1)
def simulate_path(As, c, Cshock=None, eps0=None, h=12, y_hist=None):
    K = As[0].shape[0]; buf = y_hist.copy().T    # (K,p), columns = y0,y-1,...
    out = np.zeros((h, K))
    for t in range(h):
        yhat = (c if c.ndim==1 else c[0]).copy()
        for j, Aj in enumerate(As): yhat += Aj @ buf[:, j]
        if t==0 and eps0 is not None:
            yhat += eps0                          # reduced-form innovation at impact
        if t==0 and Cshock is not None:           # alternative: structural impact
            e = np.zeros(K); e[0]=1.0; yhat += Cshock @ e
        out[t] = yhat; buf = np.column_stack([yhat, buf[:, :-1]])
    return out

rf_path = simulate_path(As, c, eps0=u0, h=12, y_hist=hist)
# For SVAR (single structural shock of size α at impact): use the same draw’s C and your α
alpha = results["parallel_up"]["alphas"][0]
C = acc["C"][0]
svar_path = simulate_path(As, c, Cshock=alpha*C, h=12, y_hist=hist)

# Now compare Δ vs baseline (baseline is simulate_path(..., eps0=None, Cshock=None))
base_path = simulate_path(As, c, eps0=None, h=12, y_hist=hist)
delta_rf   = rf_path   - base_path
delta_svar = svar_path - base_path


# Shock attribution bars

def structural_attribution(u0, C):
    # Solve C @ eps = u0  → eps = C^{-1} u0
    eps = np.linalg.solve(C, u0)
    w = np.abs(eps) / (np.sum(np.abs(eps))+1e-12)       # share by structural shock
    return eps, w

eps_rf, w_rf = structural_attribution(u0, C)            # RF solution: messy mix
eps_sv, w_sv = np.zeros_like(eps_rf), np.zeros_like(w_rf); eps_sv[0]=alpha; w_sv[0]=1.0
# Plot w_rf vs w_sv as stacked bars (RF vs SVAR)


# Policy frontier

def policy_frontier(paths_svar, var_order):
    i_pi  = var_order.index("infl_q_ann")
    i_gdp = var_order.index("gdp_q_ann")
    pi_4q = paths_svar[:, 3, i_pi]             # Δπ at 4Q (pp)
    y_min = paths_svar[:, :4, i_gdp].min(axis=1)  # min Δy over 1–4Q
    return pi_4q, y_min
# Then scatter(pi_4q, y_min) with median crosshairs.


# forecasts

# Step 1: toolbox

import numpy as np
import pandas as pd

UNITS = "pp"                    # your VAR is in percentage points
to_bps = lambda x: (100.0 * x) if UNITS=="pp" else (1e4 * x)
to_dec = lambda x: (x/100.0) if UNITS=="pp" else x

def has_irfs(x) -> bool:
    return isinstance(x, dict) and isinstance(x.get("IRFs", None), np.ndarray) and x["IRFs"].shape[0] > 0

def basel_target_curve(scenario, tenors_years, R_parallel, R_short, R_long, x=4.0):
    t = np.asarray(tenors_years, float)
    short = np.exp(-t/x); long = 1.0 - np.exp(-t/x)
    if scenario=="parallel_up":   return +R_parallel*np.ones_like(t)
    if scenario=="parallel_down": return -R_parallel*np.ones_like(t)
    if scenario=="short_up":      return +R_short*short
    if scenario=="short_down":    return -R_short*short
    if scenario=="steepener":     return (-R_short*short) + (+R_long*long)
    if scenario=="flattener":     return (+R_short*short) + (-R_long*long)
    raise ValueError(scenario)

def compute_alphas_pp(accepted_scen, var_order, tenors_years, M, basel_params, scenario):
    """α per draw, consistent with pp VAR and bps target."""
    iL = var_order.index("level"); iS = var_order.index("slope_10y_1y"); iC = var_order.index("curvature_ns_like")
    IRFs = np.asarray(accepted_scen["IRFs"])              # (Nacc,H+1,K,K)
    if IRFs.size == 0:  return np.array([]), np.array([]), np.array([]), np.array([])
    impK = IRFs[:,0,:,0]                                  # (Nacc,K) in pp
    F    = np.c_[impK[:,iL], impK[:,iS], impK[:,iC]]      # (Nacc,3) pp
    Ybps = to_bps(F @ M.T)                                # (Nacc,n_tenors) bps
    y_tgt   = basel_target_curve(scenario, tenors_years, **basel_params)
    num  = (Ybps * y_tgt[None,:]).sum(axis=1); denom = (Ybps**2).sum(axis=1)
    alphas = np.divide(num, denom, out=np.zeros_like(num), where=denom>0)

    fit = alphas[:,None]*Ybps
    dot = (fit*y_tgt[None,:]).sum(axis=1)
    cos2 = (dot / ((np.linalg.norm(fit,axis=1)+1e-12)*(np.linalg.norm(y_tgt)+1e-12)))**2
    rmse = np.sqrt(np.mean((fit - y_tgt[None,:])**2, axis=1))
    return alphas, y_tgt, cos2, rmse

def curves_from_paths_levels(paths, var_order, M, base_curve_levels):
    """Factor paths (pp) -> tenor curve levels (decimals)."""
    iL = var_order.index("level"); iS = var_order.index("slope_10y_1y"); iC = var_order.index("curvature_ns_like")
    F = np.stack([paths[:,:,iL], paths[:,:,iS], paths[:,:,iC]], axis=-1)   # (N,H,3) in pp
    Y_dev_dec = to_dec(F @ M.T)                                            # (N,H,n_ten) decimals
    return Y_dev_dec + base_curve_levels[None,None,:]

def delta_fan(fan_cond, fan_base, var):
    a = fan_cond[fan_cond['var']==var].set_index("date")[["yhat","yhat_lo","yhat_hi"]]
    b = fan_base[fan_base['var']==var].set_index("date")[["yhat","yhat_lo","yhat_hi"]]
    out = a.subtract(b); out["var"]=var; return out.reset_index()


def srsvar_conditional_fanchart_scaled(
    accepted_all, panel, vars_order, scenario, alphas,
    p=None, h=12, shock_horizon=1, n_paths=200, seed=123
):
    import numpy as np, pandas as pd
    def _is_acc(d): return isinstance(d, dict) and isinstance(d.get("IRFs",None), np.ndarray)

    # resolve params (accept single-scenario dict or {name: dict})
    if _is_acc(accepted_all) and "B" in accepted_all:
        params = accepted_all; resolved = "<single>"
    elif isinstance(accepted_all, dict) and scenario in accepted_all and _is_acc(accepted_all[scenario]):
        params = accepted_all[scenario]; resolved = scenario
    else:
        keys = list(accepted_all.keys()) if isinstance(accepted_all, dict) else str(type(accepted_all))
        raise KeyError(f"could not resolve scenario={scenario!r}; accepted_all keys/type={keys}")

    B_all = np.asarray(params["B"]); C_all = np.asarray(params["C"]); IRFs = np.asarray(params["IRFs"])
    Nacc, K, Kp = B_all.shape; p = (Kp//K) if p is None else p
    assert Kp == K*p, f"Kp ({Kp}) != K*p ({K}*{p})"

    # panel columns check
    missing = [c for c in vars_order if c not in panel.columns]
    if missing: raise KeyError(f"panel missing columns: {missing}")

    # align alphas
    alphas = np.asarray(alphas).reshape(-1)
    if alphas.size == 1 and Nacc>1: alphas = np.repeat(alphas, Nacc)
    if alphas.size != Nacc:
        raise ValueError(f"alphas len {alphas.size} != accepted draws {Nacc} for {resolved}")

    # history buffer
    hist = panel[vars_order].dropna().to_numpy()[-p:, :]
    hist0 = hist[::-1].T  # (K,p)

    rng = np.random.default_rng(seed)
    sel = rng.choice(Nacc, size=min(n_paths, Nacc), replace=False)
    paths = np.zeros((sel.size, h, K))
    dates = pd.period_range(panel.index[-1], periods=h+1, freq="Q")[1:].to_timestamp("Q")

    for dpos, di in enumerate(sel):
        B = B_all[di]; C = C_all[di]; a = float(alphas[di])
        As = [B[:, j*K:(j+1)*K] for j in range(p)]
        c_vec = params.get("c", None)
        if c_vec is None: c_use = np.zeros(K)
        else: c_use = np.asarray(c_vec[di]) if np.ndim(c_vec)==2 else np.asarray(c_vec)

        buf = hist0.copy()
        for t in range(h):
            yhat = c_use.copy()
            for j, Aj in enumerate(As): yhat += Aj @ buf[:, j]
            if t < shock_horizon:
                eps = np.zeros(K); eps[0] = a
                yhat = yhat + C @ eps
            paths[dpos, t, :] = yhat
            buf = np.column_stack([yhat, buf[:, :-1]])

    # summarize
    med = np.median(paths, axis=0); lo = np.percentile(paths, 10, axis=0); hi = np.percentile(paths, 90, axis=0)
    rows = []
    for k, name in enumerate(vars_order):
        rows.append(pd.DataFrame({
            "date": dates, "var": name, "h": np.arange(1,h+1),
            "yhat": med[:,k], "yhat_lo": lo[:,k], "yhat_hi": hi[:,k],
            "model": f"SR-SVAR_{resolved}"
        }))
    fan_df = pd.concat(rows, ignore_index=True)
    return fan_df, paths


# Step 2: Basel and macro forecasts

scen = "short_down"
acc  = accepted_basel_tight[scen]                 # post shape/size gate
# α aligned to kept draws (use stored filtered α if you saved it; else compute now)
alphas = acc.get("_alphas_unitfit")
if (alphas is None) or (len(alphas) != acc["IRFs"].shape[0]):
    alphas, y_star_bps, cos2, rmse = compute_alphas_pp(acc, var_order, tenors_years, M, basel_mag[scen], scen)

# conditional forecast
fan_s, paths_s = srsvar_conditional_fanchart_scaled(
    accepted_all={scen: acc}, panel=panel, vars_order=var_order,
    scenario=scen, alphas=alphas, p=None, h=12, shock_horizon=1, n_paths=200, seed=123
)

# baseline (no shock; same draw pool)
alphas0 = np.zeros(acc["IRFs"].shape[0])
fan_0, paths_0 = srsvar_conditional_fanchart_scaled(
    accepted_all={scen: acc}, panel=panel, vars_order=var_order,
    scenario=scen, alphas=alphas0, p=None, h=12, shock_horizon=0, n_paths=200, seed=123
)

# tenor curves in levels (decimals)
curves_s = curves_from_paths_levels(paths_s, var_order, M, current_curve_levels)
curves_0 = curves_from_paths_levels(paths_0, var_order, M, current_curve_levels)



# SCALING THE ALPHAS

i_pol = var_order.index("policy_rate")
imp_pol_unit_pp = accepted_macro["monetary_tightening"]["IRFs"][:, 0, i_pol, 0]  # (Nacc,)
print("Unit policy-rate impact (pp): median=",
      float(np.median(imp_pol_unit_pp)), " | p10/p90=",
      float(np.percentile(imp_pol_unit_pp,10)),
      float(np.percentile(imp_pol_unit_pp,90)))
# If the median here is, say, 3.2 pp, your "unit" shock = 320 bps hike (!)

### OR ALTERNATIVELY
# i_pol = var_order.index("policy_rate")
# IRFs_m = accepted_macro["monetary_tightening"]["IRFs"]   # shape (Nacc, H+1, K, K)
# imp_pol_unit_pp = IRFs_m[:, 0, i_pol, 0]      

target_hike_pp = 1.00                                # +100 bps
eps = 1e-12
alphas_monet = target_hike_pp / (imp_pol_unit_pp + eps)   # (Nacc,)

accm   = accepted_macro["monetary_tightening"]
IRFs_m = accm["IRFs"]                 # (Nacc, H+1, K, K)
i_pol  = var_order.index("policy_rate")

# a) is the identified shock actually column 0 for every draw?
col0_ok = np.allclose(IRFs_m[:, 0, i_pol, 0] != 0, True)  # crude: should be nonzero if it's the policy shock
print("Shock is in column 0 for all draws? ->", bool(col0_ok))

# b) did the sign restriction hold on impact?
imp_pol_unit_pp = IRFs_m[:, 0, i_pol, 0]   # policy impact from unit structural shock
print("Policy impact @h=0 (pp): median=", float(np.median(imp_pol_unit_pp)),
      " p10=", float(np.percentile(imp_pol_unit_pp,10)),
      " p90=", float(np.percentile(imp_pol_unit_pp,90)))

neg_share = float(np.mean(imp_pol_unit_pp < 0))
print("Share of draws with *negative* impact on policy @h=0:", neg_share)


name = "monetary_tightening"
accm = accepted_macro[name]
Nacc = accm["IRFs"].shape[0]
fan_m, paths_m = srsvar_conditional_fanchart_scaled(
    accepted_all={name: accm}, panel=panel, vars_order=var_order,
    scenario=name, alphas=np.ones(Nacc), p=None, h=12, shock_horizon=1, n_paths=200, seed=123
)
# baseline for macro:
fan_m0, paths_m0 = srsvar_conditional_fanchart_scaled(
    accepted_all={name: accm}, panel=panel, vars_order=var_order,
    scenario=name, alphas=alphas_monet, p=None, h=12, shock_horizon=0, n_paths=200, seed=123
)


# Step 3: views

# Example: plot inflation level forecast under monetary_tightening
df = fan_m[fan_m["var"]=="infl_q_ann"]
import matplotlib.pyplot as plt
plt.fill_between(df["h"], df["yhat_lo"], df["yhat_hi"], alpha=0.15)
plt.plot(df["h"], df["yhat"], lw=2)
plt.title("Monetary tightening: inflation forecast (pp)"); plt.xlabel("h (quarters)"); plt.ylabel("pp"); plt.show()


# Δ inflation under parallel_down
fan_d = delta_fan(fan_s, fan_0, "infl_q_ann")
fan_d['h'] = list(map(lambda x: x + 1, fan_d.index.to_list()))
plt.fill_between(fan_d["h"], fan_d["yhat_lo"], fan_d["yhat_hi"], alpha=0.15)
plt.plot(fan_d["h"], fan_d["yhat"], lw=2)
plt.axhline(0, color="k", lw=1)
plt.title("Δ inflation vs baseline (pp) for scenario: " + scen); plt.xlabel("h"); plt.ylabel("pp"); plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_delta_yields(
    curves_s,              # (N, H, K) stressed yields
    curves_0,              # (N, H, K) baseline yields
    tenors_years,          # length-K maturities (years)
    tenors_to_plot=(1.0, 5.0, 10.0),
    percentiles=(10, 50, 90),
    units="decimal",       # "decimal" or "bps"
    title="Scenario: Δ yields vs baseline",
    xlabel="h (quarters)",
    ylabel=None,
    show_band=True,        # draw percentile band
    show=True,
    ax=None,
):
    # --- checks ---
    curves_s = np.asarray(curves_s); curves_0 = np.asarray(curves_0)
    if curves_s.shape != curves_0.shape or curves_s.ndim != 3:
        raise ValueError("curves_s and curves_0 must be (N, H, K) arrays with same shape.")
    N, H, K = curves_s.shape

    tenors = np.asarray(tenors_years, float)
    if tenors.shape != (K,):
        raise ValueError("tenors_years must have length K (matches last dim).")

    lo_p, md_p, hi_p = percentiles
    if not (0 <= lo_p < md_p < hi_p <= 100):
        raise ValueError("percentiles must satisfy 0 <= lo < median < hi <= 100.")

    # delta curves
    dcurves = curves_s - curves_0

    # units
    if units == "decimal":
        factor = 1.0
        default_ylabel = "Δ yield (decimal)"
    elif units == "bps":
        factor = 10000.0
        default_ylabel = "Δ yield (bps)"
    else:
        raise ValueError("units must be 'decimal' or 'bps'.")

    # axes
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots()
        created_ax = True
    x = np.arange(1, H + 1)

    # collect line handles & labels for legend
    line_handles, line_labels = [], []

    for tenor_req in tenors_to_plot:
        j = int(np.argmin(np.abs(tenors - float(tenor_req))))
        tenor_actual = tenors[j]

        lo = np.percentile(dcurves[:, :, j], lo_p, axis=0) * factor
        md = np.percentile(dcurves[:, :, j], md_p, axis=0) * factor
        hi = np.percentile(dcurves[:, :, j], hi_p, axis=0) * factor

        # median line first -> we can reuse its color for the band if desired
        ln, = ax.plot(x, md, linewidth=2)  # label goes to legend via handles below

        if show_band:
            # keep bands out of legend
            ax.fill_between(x, lo, hi, alpha=0.15, label="_nolegend_")
            # (Optional) to match band color to line:
            # ax.fill_between(x, lo, hi, alpha=0.15, facecolor=ln.get_color(), label="_nolegend_")

        # legend label text
        if np.isclose(tenor_req, tenor_actual):
            lab = f"{tenor_actual:.0f}y" if tenor_actual >= 1 else f"{tenor_actual*12:.0f}m"
        else:
            lab = f"{tenor_req:g}y→{tenor_actual:.0f}y" if tenor_actual >= 1 else f"{tenor_req:g}y→{tenor_actual*12:.0f}m"

        line_handles.append(ln)
        line_labels.append(lab)

    ax.axhline(0.0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or default_ylabel)

    # legend for the *lines only*
    ax.legend(line_handles, line_labels, title="Tenor")

    if show and created_ax:
        plt.show()
    return ax


# Example: plot parallel-down deltas for 1y, 5y, 10y in bps
plot_delta_yields(
    curves_s=curves_s,
    curves_0=curves_0,
    tenors_years=tenors_years,        # e.g., [0.25, 1, 2, 5, 10]
    tenors_to_plot=(1, 5, 10),
    percentiles=(10, 50, 90),
    units="bps",
    title="Δ yields vs baseline for scenario:" + scen
)


accepted_macro = {'fiscal_expansion' : accepted_fiscal, "monetary_tightening" : accepted_monet}

# =========================
# Forecasting runner (pp-safe, Basel-ready)
# =========================
import numpy as np
import pandas as pd

# --- units: your VAR is in percentage points (pp) ---
UNITS = "pp"                   # change to "dec" if you ever switch
BPS_PER_UNIT = 100.0 if UNITS=="pp" else 1e4
def to_bps(x): return BPS_PER_UNIT * x
def to_dec(x): return (x/100.0) if UNITS=="pp" else x

# --- Basel target (bps) ---
def basel_target_curve(scenario, tenors_years, R_parallel, R_short, R_long, x=4.0):
    t = np.asarray(tenors_years, float)
    short = np.exp(-t/x); long = 1.0 - np.exp(-t/x)
    if scenario=="parallel_up":   return +R_parallel*np.ones_like(t)
    if scenario=="parallel_down": return -R_parallel*np.ones_like(t)
    if scenario=="short_up":      return +R_short*short
    if scenario=="short_down":    return -R_short*short
    if scenario=="steepener":     return (-R_short*short) + (+R_long*long)
    if scenario=="flattener":     return (+R_short*short) + (-R_long*long)
    raise ValueError(scenario)

# --- α per draw (through-origin LS) with pp→bps handled ---
def compute_alphas_pp(accepted_scen, var_order, tenors_years, M, basel_params, scenario):
    iL = var_order.index("level"); iS = var_order.index("slope_10y_1y"); iC = var_order.index("curvature_ns_like")
    IRFs = np.asarray(accepted_scen["IRFs"])              # (Nacc,H+1,K,K)
    if IRFs.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    impK = IRFs[:,0,:,0]                                  # (Nacc,K) impact (pp)
    F    = np.c_[impK[:,iL], impK[:,iS], impK[:,iC]]      # (Nacc,3) pp
    Ybps = to_bps(F @ M.T)                                # (Nacc,n_tenors) bps
    y_star = basel_target_curve(scenario, tenors_years, **basel_params)  # (n_tenors,) bps
    num   = (Ybps * y_star[None,:]).sum(axis=1)
    denom = (Ybps**2).sum(axis=1)
    alphas = np.divide(num, denom, out=np.zeros_like(num), where=denom>0)

    # diagnostics (optional): cos² & RMSE in bps
    fit   = alphas[:,None] * Ybps
    sse   = ((fit - y_star[None,:])**2).sum(axis=1)
    sst0  = float((y_star**2).sum()) + 1e-12
    r2unc = 1.0 - sse/sst0
    dot   = (fit * y_star[None,:]).sum(axis=1)
    cos2  = (dot / ((np.linalg.norm(fit,axis=1)+1e-12)* (np.linalg.norm(y_star)+1e-12)))**2
    rmse  = np.sqrt(np.mean((fit - y_star[None,:])**2, axis=1))
    return alphas, y_star, cos2, rmse

# --- factor paths (pp) -> tenor curve levels (decimals) ---
def curves_from_paths_levels(paths, var_order, M, base_curve_levels):
    iL = var_order.index("level"); iS = var_order.index("slope_10y_1y"); iC = var_order.index("curvature_ns_like")
    F = np.stack([paths[:,:,iL], paths[:,:,iS], paths[:,:,iC]], axis=-1)   # (N,H,3) pp or dec
    Y_dev_units = F @ M.T                                                  # (N,H,n_ten) in VAR units
    Y_dev_dec   = to_dec(Y_dev_units)                                      # -> decimals
    return Y_dev_dec + base_curve_levels[None,None,:]                      # absolute levels

# --- thin wrapper: baseline (no shock) forecast by setting alphas=0 ---
def run_baseline_forecast(accepted_any, panel, var_order, scenario_name, p=None, h=12, n_paths=200, seed=123):
    # zeros alphas: simulator should ignore the shock
    Nacc = accepted_any[scenario_name]["IRFs"].shape[0]
    alphas0 = np.zeros(Nacc)
    fan0, paths0 = srsvar_conditional_fanchart_scaled(
        accepted_all=accepted_any, panel=panel, vars_order=var_order,
        scenario=scenario_name, alphas=alphas0, p=p, h=h, shock_horizon=0,
        n_paths=n_paths, seed=seed
    )
    return fan0, paths0

# --- main runner ---
import numpy as np

def run_all_forecasts(
    panel, var_order, M, tenors_years, basel_mag, current_curve_levels,
    accepted_macro, accepted_basel_tight,
    p=None, h=12, shock_horizon=1, n_paths=200, seed=123
):
    """
    Returns:
      results_basel[scen] = {'fan','paths','alphas','curve_levels','target_bps'}
      results_macro[scen] = {'fan','paths','curve_levels'}
      baseline[scen]      = {'fan','paths','curve_levels'}
    """
    results_basel, results_macro, baseline = {}, {}, {}

    # -------- helper: resolve alphas for Basel on the *accepted* set --------
    def _alphas_for_basel(acc, scen):
        """
        Prefer stored _alphas_unitfit only if its length == Nacc.
        If a mapping of kept indices exists (keep_idx / idx / sel / accept_idx),
        use it to subset; else recompute alphas on the accepted set.
        """
        Nacc = acc["IRFs"].shape[0]
        target_bps = basel_target_curve(scen, tenors_years, **basel_mag[scen])

        stored = acc.get("_alphas_unitfit", None)
        if stored is not None:
            stored = np.asarray(stored).reshape(-1)
            if stored.size == Nacc:
                return stored, target_bps
            # try to subset if accepted kept original indices
            for key in ("keep_idx", "idx", "sel", "accept_idx"):
                if key in acc:
                    idx = np.asarray(acc[key]).reshape(-1)
                    # if idx maps into stored (candidates) and produces right length, subset:
                    if idx.size == Nacc and idx.max() < stored.size:
                        return stored[idx], target_bps
            # length mismatch with no mapping → fall through to recompute

        # recompute α on the accepted set (safe default)
        alphas, _, _, _ = compute_alphas_pp(
            acc, var_order, tenors_years, M, basel_mag[scen], scen
        )
        # (optional) cache back into acc so you don't recompute next time
        acc["_alphas_unitfit"] = alphas
        return alphas, target_bps

    # ---------------- Basel scenarios ----------------
    for scen in ["parallel_up","parallel_down","steepener","flattener","short_up","short_down"]:
        acc = accepted_basel_tight.get(scen, {})
        if not acc or "IRFs" not in acc or acc["IRFs"].size == 0:
            continue

        alphas, y_star_bps = _alphas_for_basel(acc, scen)

        fan_df, paths = srsvar_conditional_fanchart_scaled(
            accepted_all=accepted_basel_tight, panel=panel, vars_order=var_order,
            scenario=scen, alphas=alphas, p=p, h=h, shock_horizon=shock_horizon,
            n_paths=n_paths, seed=seed
        )
        curves = curves_from_paths_levels(paths, var_order, M, current_curve_levels)
        results_basel[scen] = dict(
            fan=fan_df, paths=paths, alphas=alphas,
            curve_levels=dict(
                p10=np.percentile(curves,10,axis=0),
                median=np.percentile(curves,50,axis=0),
                p90=np.percentile(curves,90,axis=0),
            ),
            target_bps=y_star_bps
        )

        # baseline (no shock), same draw pool
        fan0, paths0 = run_baseline_forecast(
            accepted_basel_tight, panel, var_order, scen,
            p=p, h=h, n_paths=n_paths, seed=seed
        )
        curves0 = curves_from_paths_levels(paths0, var_order, M, current_curve_levels)
        baseline[scen] = dict(
            fan=fan0, paths=paths0,
            curve_levels=dict(
                p10=np.percentile(curves0,10,axis=0),
                median=np.percentile(curves0,50,axis=0),
                p90=np.percentile(curves0,90,axis=0),
            )
        )

    # ---------------- Macro scenarios ----------------
    for scen in ["fiscal_expansion","monetary_tightening"]:
        acc = accepted_macro.get(scen, {})
        if not acc or "IRFs" not in acc or acc["IRFs"].size == 0:
            continue

        Nacc = acc["IRFs"].shape[0]
        alphas_ones = np.ones(Nacc)  # unit structural shock (or swap in your 1σ-policy α's)

        fan_df, paths = srsvar_conditional_fanchart_scaled(
            accepted_all=accepted_macro, panel=panel, vars_order=var_order,
            scenario=scen, alphas=alphas_ones, p=p, h=h, shock_horizon=shock_horizon,
            n_paths=n_paths, seed=seed
        )
        curves = curves_from_paths_levels(paths, var_order, M, current_curve_levels)
        results_macro[scen] = dict(
            fan=fan_df, paths=paths,
            curve_levels=dict(
                p10=np.percentile(curves,10,axis=0),
                median=np.percentile(curves,50,axis=0),
                p90=np.percentile(curves,90,axis=0),
            )
        )

        # baseline (no shock), same draw pool
        fan0, paths0 = run_baseline_forecast(
            accepted_macro, panel, var_order, scen,
            p=p, h=h, n_paths=n_paths, seed=seed
        )
        curves0 = curves_from_paths_levels(paths0, var_order, M, current_curve_levels)
        baseline[scen] = dict(
            fan=fan0, paths=paths0,
            curve_levels=dict(
                p10=np.percentile(curves0,10,axis=0),
                median=np.percentile(curves0,50,axis=0),
                p90=np.percentile(curves0,90,axis=0),
            )
        )

    return results_basel, results_macro, baseline


# accepted_basel_tight["parallel_up"]['IRFs'].shape[0]
accepted_basel_tight.get('parallel_up', {}).get("_alphas_unitfit")

# Example call (fill these from your workspace)
# accepted_macro = {"fiscal": accepted_fiscal, "monet": accepted_monet}  # post SR
# accepted_basel_tight = {...}  # your 6 scenarios after shape/size gate
# current_curve_levels = panel[tenor_cols].iloc[-1].to_numpy(float)  # decimals

results_basel, results_macro, baseline = run_all_forecasts(
    panel, var_order, M, tenors_years, basel_mag, current_curve_levels,
    accepted_macro, accepted_basel_tight,
    p=None, h=12, shock_horizon=1, n_paths=200, seed=123
)

# # Quick peek: macro fan for inflation (median path)
# fan_pi = results_macro["monetary_tightening"]["fan"].query("var=='infl_q_ann'")[["date","yhat","yhat_lo","yhat_hi"]].head()
# fan_pi


results_basel

results_macro

import numpy as np
import pandas as pd
import warnings

def srsvar_conditional_fanchart(
    accepted_or_scen,
    panel,
    vars_order,
    p=None,
    h=8,
    shock_size=1.0,
    shock_horizon=1,
    shock_index=0,
    n_paths=200,
    seed=123,
    scenario=None,          # <-- NEW: which scenario to use (e.g., "steepener")
):
    rng = np.random.default_rng(seed)
    K = len(vars_order)

    # --- descend into scenario if the top level is a mapping of scenarios ---
    params = accepted_or_scen
    if isinstance(accepted_or_scen, dict) and "B" not in accepted_or_scen:
        # looks like a dict of scenarios; pick the requested one or default to the first
        scen_keys = [k for k, v in accepted_or_scen.items() if isinstance(v, dict)]
        if scenario is None:
            # deterministic default (sorted) to avoid surprises
            scenario = sorted(scen_keys)[0]
            warnings.warn(f"No scenario specified; using '{scenario}'. "
                          f"Available: {sorted(scen_keys)}", RuntimeWarning)
        if scenario not in accepted_or_scen:
            raise KeyError(f"Scenario '{scenario}' not found. Available: {sorted(scen_keys)}")
        params = accepted_or_scen[scenario]

    # --- required pieces
    if "B" not in params or "c" not in params:
        missing = [k for k in ("B", "c") if k not in params]
        raise KeyError(f"Missing required key(s) in scenario params: {missing}. "
                       f"Ensure your accepted['{scenario}'] includes 'B' and 'c'.")

    B_all = np.asarray(params["B"])      # (N, K, Kp)
    c_all = np.asarray(params["c"])      # (N, K) or (K,)
    Nacc, K_B, Kp = B_all.shape
    if K_B != K:
        raise ValueError(f"K mismatch: vars_order has {K}, but B has {K_B}")

    # choose draws
    n_sims = min(n_paths, Nacc)
    sel = rng.choice(Nacc, size=n_sims, replace=False)

    # infer/validate p
    if Kp % K != 0:
        raise ValueError(f"B has incompatible shape (K, Kp)=({K},{Kp}); Kp must be multiple of K.")
    p_eff = (Kp // K) if p is None else p
    if p is not None and (K * p != Kp):
        raise ValueError(f"Provided p={p} does not match B (Kp={Kp} ⇒ implied p={Kp//K}).")

    # history buffer
    hist_mat = panel[vars_order].dropna().to_numpy()
    if hist_mat.shape[0] < p_eff:
        raise ValueError(f"Not enough history: need {p_eff} rows after dropna(), have {hist_mat.shape[0]}")
    hist0 = hist_mat[-p_eff:, :][::-1].T   # K x p_eff

    # forecast dates
    last_idx = panel.index[-1]
    last_q = last_idx.asfreq("Q") if isinstance(panel.index, pd.PeriodIndex) else pd.Period(pd.Timestamp(last_idx), freq="Q")
    dates = pd.period_range(last_q, periods=h+1, freq="Q")[1:].to_timestamp("Q")

    # impact matrix getter (prefer IRFs; fallback to C)
    def get_impact(idx):
        if "IRFs" in params:
            IRF = np.asarray(params["IRFs"][idx])
            # Accept shapes: (K,K), (K,K,H), (H+1,K,K)
            if IRF.ndim == 2 and IRF.shape == (K, K):
                return IRF
            if IRF.ndim == 3:
                if IRF.shape[0] == K and IRF.shape[1] == K:   # (K,K,H+)
                    return IRF[:, :, 0]
                if IRF.shape[-2] == K and IRF.shape[-1] == K: # (H+1,K,K)
                    return IRF[0, :, :]
        if "C" in params:
            C = np.asarray(params["C"][idx])
            if C.ndim == 2 and C.shape == (K, K):
                return C
        raise KeyError("No impact matrix found for scenario: expected 'IRFs' (…KxK…) or 'C' (KxK).")

    # simulate
    paths = np.zeros((n_sims, h, K))
    for d, idx in enumerate(sel):
        B = B_all[idx]
        c = c_all[idx] if c_all.ndim == 2 else c_all
        Cimp = get_impact(idx)

        As = [B[:, j*K:(j+1)*K] for j in range(p_eff)]
        hist_buf = hist0.copy()

        for t in range(h):
            yhat = c.copy()
            for j, Aj in enumerate(As):
                yhat += Aj @ hist_buf[:, j]
            if t < shock_horizon:
                eps = np.zeros(K); eps[shock_index] = shock_size
                yhat = yhat + Cimp @ eps
            paths[d, t, :] = yhat
            hist_buf = np.column_stack([yhat, hist_buf[:, :-1]])

    # summarize (10–90%)
    med = np.median(paths, axis=0)
    lo  = np.percentile(paths, 10, axis=0)
    hi  = np.percentile(paths, 90, axis=0)

    fan_df = pd.concat([
        pd.DataFrame({
            "date": dates,
            "var": name,
            "h": np.arange(1, h+1),
            "yhat": med[:, k],
            "yhat_lo": lo[:, k],
            "yhat_hi": hi[:, k],
            "model": f"SR-SVAR_{scenario or 'scenario'}"
        })
        for k, name in enumerate(vars_order)
    ], ignore_index=True)

    return fan_df, paths

# Choose origin: latest available
train = panel

# BVAR unconditional from this origin (for overlay)
bvar_fc = forecast_bvar(idata, train, p=idata.attrs.get("lags",2),
                        h=8, draws=500, model_name="BVAR")

# SR-SVAR conditional, e.g. fiscal (1σ shock at t=0 for one quarter)
fan_fiscal, paths_fiscal = srsvar_conditional_fanchart(
    accepted_fiscal, panel=train, vars_order=var_order,
    p=idata.attrs.get("lags",2), h=8,
    shock_size=1.0, shock_horizon=1, n_paths=200,
    )

# Monetary scenario too (e.g., 2σ and 2-quarter shock)
fan_monet, paths_monet = srsvar_conditional_fanchart(
    accepted_monet, panel=train, vars_order=var_order,
    p=idata.attrs.get("lags",2), h=8,
    shock_size=2.0, shock_horizon=2, n_paths=200,
    )

# Optional: RW line for this origin (split≈1.0 to forecast only from final point)
rw_fc = bvar_fc.copy()
for i in bvar_fc['var'].unique():
    for j in range(1,9):
        rw_fc.loc[(rw_fc['var'] == i) & (rw_fc['h'] == j), 'yhat'] = panel[-1:][i].values

import matplotlib.pyplot as plt

def plot_conditional_overlay(fan_df, bvar_fc, rw_fc, var, title_suffix=""):
    # SR-SVAR fan
    sub_s = fan_df[fan_df["var"]==var]
    # BVAR (mean path)
    sub_b = bvar_fc[bvar_fc["var"]==var]
    # RW at same origin (take horizon up to max h)
    sub_r = rw_fc[rw_fc["var"]==var]

    plt.figure(figsize=(7,4))
    # SR-SVAR shaded band + median
    plt.fill_between(sub_s["date"], sub_s["yhat_lo"], sub_s["yhat_hi"], alpha=0.25, label="SR-SVAR band")
    plt.plot(sub_s["date"], sub_s["yhat"], label="SR-SVAR median", linewidth=2)
    # BVAR mean path
    plt.plot(sub_b["date"], sub_b["yhat"], linestyle="--", label="BVAR (uncond)")
    # RW line (may have fewer points depending on horizons)
    # Align by date
    sub_r_agg = (sub_r.groupby("date")["yhat"].mean().reset_index())
    plt.plot(sub_r_agg["date"], sub_r_agg["yhat"], linestyle=":", label="RW")

    plt.axhline(0, color="black", linewidth=0.6)  # useful for rates/changes
    plt.title(f"{var}: SR-SVAR scenario vs BVAR/RW {title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example: inflation
plot_conditional_overlay(fan_fiscal, bvar_fc, rw_fc, var="infl_q_ann", title_suffix="(fiscal 1σ)")
# Example: policy rate
plot_conditional_overlay(fan_monet, bvar_fc, rw_fc, var="policy_rate", title_suffix="(monet 2σ x 2q)")


def forecast_macro(
    scenario,                                # "fiscal_expansion" | "monetary_tightening"
    accepted_macro, panel, var_order,
    M, tenors_years, current_curve_levels,
    p=None, h=12, shock_horizon=1, n_paths=200, seed=123,
    alphas=None                             # optional scaling; defaults to ones
):
    """
    Forecasts macro variables and yield curve given a macro scenario shock.

    Parameters
    ----------
    scenario : str
        Macro scenario key ("fiscal_expansion" or "monetary_tightening").
    accepted_macro : dict
        Dict of accepted SR-SVAR draws per macro scenario.
    panel : DataFrame
        Time series data.
    var_order : list of str
        Variable ordering in the VAR.
    M, tenors_years : array-like
        Nelson-Siegel factor-to-tenor map and grid.
    current_curve_levels : array
        Today's yield curve levels (decimals).
    p, h, shock_horizon, n_paths, seed : VAR forecast args.
    alphas : array-like or None
        Optional scaling factors (per draw). Defaults to ones (unit shock).

    Returns
    -------
    dict with:
        fan      - tidy DataFrame (macro forecasts)
        paths    - (N,H,K) macro paths
        curves   - dict with tenor-level forecasts (levels & delta)
        baseline - forecasts with no shock
    """
    acc = accepted_macro[scenario]
    Nacc = acc["IRFs"].shape[0]

    # default: unit shock
    if alphas is None:
        alphas = np.ones(Nacc)

    # shocked forecast
    fan_s, paths_s = srsvar_conditional_fanchart_scaled(
        accepted_all={scenario: acc}, panel=panel, vars_order=var_order,
        scenario=scenario, alphas=alphas, p=p, h=h,
        shock_horizon=shock_horizon, n_paths=n_paths, seed=seed
    )

    # baseline forecast (no shock)
    fan_0, paths_0 = srsvar_conditional_fanchart_scaled(
        accepted_all={scenario: acc}, panel=panel, vars_order=var_order,
        scenario=scenario, alphas=np.zeros_like(alphas), p=p, h=h,
        shock_horizon=0, n_paths=n_paths, seed=seed
    )

    # map factor paths → tenor curve
    curves_s = curves_from_paths_levels(paths_s, var_order, M, current_curve_levels)
    curves_0 = curves_from_paths_levels(paths_0, var_order, M, current_curve_levels)
    dcurves  = curves_s - curves_0

    # summarize (10/50/90)
    q = lambda x: dict(p10=np.percentile(x,10,axis=0),
                       median=np.percentile(x,50,axis=0),
                       p90=np.percentile(x,90,axis=0))

    return dict(
        fan=fan_s, paths=paths_s,
        curves=dict(levels=q(curves_s), delta=q(dcurves)),
        baseline=dict(fan=fan_0, paths=paths_0, curves=dict(levels=q(curves_0)))
    )


acc   = accepted_macro["monetary_tightening"]
IR    = np.asarray(acc["IRFs"])  # (Nacc,H+1,K,K)
i_pol = var_order.index("policy_rate")

# What policy move did you actually apply in this run?
# If you used alphas=None/ones, this is your effective shock size in pp.
unit_imp = IR[:,0,i_pol,0]                # unit-shock impact on policy (pp)
alpha_used = np.ones_like(unit_imp)       # or whatever you actually passed
eff_policy_move_pp = np.median(alpha_used * unit_imp)

print("Effective policy move (median, pp):", float(eff_policy_move_pp))


alphas_100 = 1.00 / (unit_imp + 1e-12)

res_monet = forecast_macro(
    scenario="monetary_tightening",
    accepted_macro=accepted_macro, panel=panel, var_order=var_order,
    M=M, tenors_years=tenors_years, current_curve_levels=current_curve_levels,
    h=12, shock_horizon=1, n_paths=300, seed=123, alphas=None
)

# Δ yield curve (median, decimals) after monetary tightening
deltas = res_monet["curves"]["delta"]["median"]


res_monet['fan'][res_monet['fan']['var'] == 'infl_q_ann']

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_macro_forecast(res, var_order,
                        vars_delta=("infl_q_ann","gdp_q_ann"),
                        tenors_years=None, tenor_picks=(0.25, 10.0),
                        title_prefix="Macro scenario"):
    """
    res: dict returned by forecast_macro(...)
    Shows:
      - Δ fans for selected macro variables (vs baseline)
      - Δ yield paths at selected tenors (e.g., 3m and 10y, decimals)
      - Curve snapshots (Δ) at h = 1, 4, 8, 12 (if available)
    """
    # --- Δ macro fans
    fan_s   = res["fan"]
    fan_0   = res["baseline"]["fan"]
    def delta_fan(var):
        a = fan_s[fan_s['var']==var].set_index("date")[["yhat","yhat_lo","yhat_hi"]]
        b = fan_0[fan_0['var']==var].set_index("date")[["yhat","yhat_lo","yhat_hi"]]
        out = a.subtract(b); out["h"]=np.arange(1, out.shape[0]+1); return out

    nvars = len(vars_delta)
    plt.figure(figsize=(12, 3.5*nvars))
    for r, v in enumerate(vars_delta, start=1):
        df = delta_fan(v)
        ax = plt.subplot(nvars,1,r)
        ax.fill_between(df["h"], df["yhat_lo"], df["yhat_hi"], alpha=0.15)
        ax.plot(df["h"], df["yhat"], lw=2)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(f"{title_prefix}: Δ {v} vs baseline (pp)")
        ax.set_xlabel("h (quarters)")
        
        plt.figtext(
        0.5, 0.01,
        "Note: Responses correspond to a one-unit structural shock. "
        "Scaling to +100 bps policy move ≈ ×10 in magnitude.",
        ha="center", fontsize=9, style="italic"
        )

    plt.tight_layout(); plt.show()

    # --- Δ yield tenors (decimals)
    dmed = res["curves"]["delta"]["median"]   # (H, n_tenors)
    if tenors_years is not None:
        H, nT = dmed.shape
        tgrid = np.asarray(tenors_years, float)
        pick_idx = [int(np.argmin(np.abs(tgrid - x))) for x in tenor_picks]
        plt.figure(figsize=(10,4))
        for j in pick_idx:
            lo = res["curves"]["delta"]["p10"][:, j]
            md = dmed[:, j]
            hi = res["curves"]["delta"]["p90"][:, j]
            h  = np.arange(1, md.shape[0]+1)
            plt.fill_between(h, lo, hi, alpha=0.15)
            plt.plot(h, md, lw=2, label=f"{tgrid[j]:g}y")
        plt.axhline(0, color="k", lw=1)
        plt.title(f"{title_prefix}: Δ yields at selected tenors (decimals)")
        plt.xlabel("h (quarters)"); plt.legend(); plt.tight_layout(); plt.show()
        
        # --- Curve snapshots (Δ) at a few horizons
        snaps = [1,4,8,12]
        snaps = [s for s in snaps if s <= dmed.shape[0]]
        if snaps:
            cols = len(snaps)
            plt.figure(figsize=(3.2*cols, 3.6))
            for c, s in enumerate(snaps, start=1):
                ax = plt.subplot(1, cols, c)
                ax.plot(tgrid, dmed[s-1, :], lw=2)
                ax.axhline(0, color="k", lw=1)
                ax.set_title(f"Δ curve @ h={s}")
                ax.set_xlabel("tenor (years)")
            plt.tight_layout(); plt.show()


plot_macro_forecast(res_monet, var_order, title_prefix="Monetary tightening",
                    tenors_years=tenors_years, tenor_picks=(1.0, 5.0))
