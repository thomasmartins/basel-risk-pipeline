import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns  # optional, just for style

# ---- Minimal helpers (same logic as before) ----
def _unpack_B(B_draw, K, p):
    Kp = B_draw.shape[1]
    assert Kp == K * p, f"Expected Kp==K*p but got Kp={Kp}, K={K}, p={p}"
    return [B_draw[:, j*K:(j+1)*K] for j in range(p)]

def _companion_from_As(As):
    K = As[0].shape[0]; p = len(As)
    A = np.zeros((K*p, K*p))
    for j, Aj in enumerate(As):
        A[:K, j*K:(j+1)*K] = Aj
    if p > 1:
        A[K:, :-K] = np.eye(K*(p-1))
    return A

def _extract_B_Sigma_c(idata, var_B="B", var_Sigma="Sigma", var_chol="chol", var_c="c"):
    post = idata.posterior
    B = post[var_B].stack(s=("chain","draw")).transpose("s", ...).values  # (S,K,Kp)
    S_total, K, Kp = B.shape
    if var_Sigma in post:
        Sigma = (post[var_Sigma].stack(s=("chain","draw"))
                             .transpose("s", ...).values)                # (S,K,K)
    else:
        chol = (post[var_chol].stack(s=("chain","draw"))
                             .transpose("s", ...).values)                # (S,K,K)
        Sigma = np.einsum("sij,skj->sik", chol, chol)
    if var_c in post:
        c = post[var_c].stack(s=("chain","draw")).transpose("s", ...).values  # (S,K)
        if c.ndim == 3 and c.shape[-1] == 1:
            c = c[..., 0]
    else:
        c = np.zeros((S_total, K))
    return B, Sigma, c

def _state_from_history(Yhist, p):
    K = Yhist.shape[1]
    hist = Yhist[-p:, :]
    return hist[::-1].reshape(-1)

def _build_mu_H(A_list, c, Sigma, H, s0):
    K = A_list[0].shape[0]; p = len(A_list)
    Acomp = _companion_from_As(A_list)
    Sel   = np.zeros((K, K*p)); Sel[:, :K] = np.eye(K)

    mu_blocks, s = [], s0.copy()
    for _ in range(1, H+1):
        y = c.copy()
        for j, Aj in enumerate(A_list, start=1):
            y += Aj @ s[(j-1)*K : j*K]
        mu_blocks.append(y)
        s = np.r_[y, s[:(p-1)*K]]
    mu = np.concatenate(mu_blocks, axis=0)  # (H*K,)

    G = np.zeros((K*p, K)); G[:K, :K] = np.eye(K)
    H_rows = []
    for h in range(1, H+1):
        row_blocks = []
        for j in range(1, H+1):
            if j <= h:
                A_pow = np.linalg.matrix_power(Acomp, j-1)
                block = Sel @ (A_pow @ G)
            else:
                block = np.zeros((K, K))
            row_blocks.append(block)
        H_rows.append(np.hstack(row_blocks))
    Hmat = np.vstack(H_rows)  # (H*K, H*K)
    return mu, Hmat

def _conditional_gaussian(mu, Hmat, Sigma_u, Cmat, dvec):
    CH = Cmat @ Hmat
    b  = dvec - Cmat @ mu
    CHS = CH @ Sigma_u
    middle = np.linalg.pinv(CHS @ CH.T)        # robust
    m_u = Sigma_u @ CH.T @ (middle @ b)
    V_u = Sigma_u - (Sigma_u @ CH.T) @ (middle @ CHS)
    y_mean = mu + Hmat @ m_u
    y_cov  = Hmat @ V_u @ Hmat.T
    return y_mean, y_cov

# ---- Units helper ----
def _bps_to_series_units(bps_value, unit_kind):
    """
    Convert a magnitude expressed in bps to the series' unit.
    unit_kind: 'pct' (percent), 'bps' (already bps), or a scale factor.
    """
    if isinstance(unit_kind, (int, float)):
        return bps_value * float(unit_kind)
    if unit_kind == "pct":   # series in percent points
        return bps_value / 100.0
    if unit_kind == "bps":   # series already in bps
        return float(bps_value)
    raise ValueError(f"Unknown unit kind: {unit_kind}")

# ---- Main generalized API ----
def basel_conditional_curve_path(
    idata: az.InferenceData,
    Yhist_df: pd.DataFrame,
    var_order: list,
    constraints: dict,
    *,
    p: int,
    H: int,
    draws_posterior: int = 400,
    draws_conditional: int = 200,
    seed: int = 123,
    scenario_name: str = "Basel_custom"
):
    """
    Conditional forecasts given multiple variable path constraints.
    You specify magnitudes in *basis points*; we convert per variable.

    Parameters
    ----------
    idata : ArviZ InferenceData for the reduced-form VAR
    Yhist_df : DataFrame (index = dates; columns include var_order)
    var_order : list of variable names (length K)
    constraints : dict mapping variable -> dict with:
        {
          "unit": "pct" | "bps" | <scale float>,     # how the series is stored
          "horizons": [1,2,...,H0],                  # horizons to constrain
          # choose exactly one of the next two:
          "delta_bps": <float>,                      # y_{t+h} = y_t + delta
          "path_bps":  <array-like length==len(horizons)>  # explicit targets (relative to y_t)
        }
        Notes:
          - All magnitudes are interpreted in *bps* and internally converted.
          - Constraints are *equalities* on future levels of those series.
    p : int  VAR lags
    H : int  forecast horizon
    draws_posterior : posterior draws to use
    draws_conditional : conditional draws per posterior draw
    seed : RNG seed
    scenario_name : label to attach to the output

    Returns
    -------
    long DataFrame with columns:
      ["date","var","h","yhat","draw","post_draw","scenario"]
    """
    rng = np.random.default_rng(seed)
    B_all, Sg_all, c_all = _extract_B_Sigma_c(idata)
    S_total, K, Kp = B_all.shape
    p_infer = Kp // K
    if Kp != K * p:
        import warnings
        warnings.warn(f"Overriding p={p} with inferred p={p_infer}.")
        p = p_infer

    # history/state
    Yhist = Yhist_df[var_order].dropna().to_numpy()
    s0 = _state_from_history(Yhist, p)
    y0 = Yhist[-1, :]
    last_date = Yhist_df.index[-1]
    q = pd.Period(last_date, freq="Q")
    dates = pd.to_datetime([(q + i).end_time.normalize() for i in range(1, H+1)])

    # Build constraint matrix C and vector d over stacked future y (size H*K)
    C_rows, d_vals = [], []
    for vname, spec in constraints.items():
        if vname not in var_order:
            raise KeyError(f"Variable '{vname}' not in var_order.")
        idx = var_order.index(vname)
        horizons = spec["horizons"]
        unit = spec.get("unit", "pct")
        if ("delta_bps" in spec) == ("path_bps" in spec):
            raise ValueError(f"Provide exactly one of delta_bps or path_bps for '{vname}'.")
        if "delta_bps" in spec:
            bump_ser_units = _bps_to_series_units(spec["delta_bps"], unit)
            targets = [y0[idx] + bump_ser_units for _ in horizons]
        else:
            # explicit per-horizon relative bumps (bps) from y0[idx]
            series_bumps = [_bps_to_series_units(bps, unit) for bps in spec["path_bps"]]
            if len(series_bumps) != len(horizons):
                raise ValueError(f"path_bps length mismatch for '{vname}'.")
            targets = [y0[idx] + b for b in series_bumps]

        for h, target in zip(horizons, targets):
            row = np.zeros(H*K)
            row[(h-1)*K + idx] = 1.0
            C_rows.append(row)
            d_vals.append(target)

    C = np.vstack(C_rows) if C_rows else np.zeros((0, H*K))
    d = np.array(d_vals)    if d_vals else np.zeros((0,))

    # posterior subsample
    sel = rng.choice(S_total, size=min(draws_posterior, S_total), replace=False)

    rows = []
    for r, sidx in enumerate(sel):
        B = B_all[sidx]; Sigma = Sg_all[sidx]; c = c_all[sidx]
        As = _unpack_B(B, K, p)
        mu, Hmat = _build_mu_H(As, c, Sigma, H, s0)
        Sigma_u = np.kron(np.eye(H), Sigma)

        # conditional mean/cov (handles C empty as well)
        if C.shape[0] == 0:
            # Unconstrained (rare): sample from baseline N(mu, H Sigma_u H')
            y_mean = mu
            y_cov  = Hmat @ Sigma_u @ Hmat.T
        else:
            y_mean, y_cov = _conditional_gaussian(mu, Hmat, Sigma_u, C, d)

        # sample conditional draws
        evals, evecs = np.linalg.eigh(y_cov)
        evals = np.clip(evals, 0.0, None)
        Lcov = evecs @ np.diag(np.sqrt(evals))
        z = rng.normal(size=(H*K, draws_conditional))
        Y_draws = (y_mean[:, None] + Lcov @ z).T.reshape(draws_conditional, H, K)

        for d_i in range(draws_conditional):
            for k, name in enumerate(var_order):
                rows.append(pd.DataFrame({
                    "date": dates,
                    "var": name,
                    "h": np.arange(1, H+1),
                    "yhat": Y_draws[d_i, :, k],
                    "draw": d_i,
                    "post_draw": r,
                    "scenario": scenario_name
                }))
    return pd.concat(rows, ignore_index=True)

def make_ecb_basel6(
    *,
    idata,
    Yhist_df: pd.DataFrame,
    var_order: list,
    p: int,
    H: int,
    draws_posterior: int = 400,
    draws_conditional: int = 200,
    seed_base: int = 123,

    # magnitudes in basis points
    parallel_bps: float = 200.0,
    slope_bps: float = 100.0,
    short_bps: float = 200.0,

    # horizons to pin (e.g. first 4 quarters)
    horizons=[1,2,3,4],

    # how each series is stored in your data
    level_unit: str = "pct",
    slope_unit: str = "pct",
    short_unit: str = "pct",

    # variable names in your VAR state
    level_var: str = "level",
    slope_var: str = "slope_10y_1y",
    short_var: str = "policy_rate",
):
    """
    Generate conditional forecast draws for all 6 ECB-style IRRBB/Basel scenarios:
      1. Parallel Up
      2. Parallel Down
      3. Steepener
      4. Flattener
      5. Short-end Up
      6. Short-end Down

    Arguments
    ---------
    idata : InferenceData from your BVAR
    Yhist_df : historical dataframe (index: dates, columns: var_order)
    var_order : list of variables in VAR order
    p : VAR lag order
    H : forecast horizon (#quarters)
    draws_posterior : posterior draws to use per scenario
    draws_conditional : conditional draws per posterior draw
    seed_base : base RNG seed; each scenario will offset it

    parallel_bps : absolute parallel shift magnitude in bps (e.g. +200 / -200)
    slope_bps    : slope steepener magnitude in bps (10y - 1y widens)
    short_bps    : short-end move magnitude in bps (policy/short rate)

    horizons     : list of horizons (1..H0) you want to lock to the shocked value
    *_unit       : "pct", "bps", or scalar for conversion of bps -> series units
    *_var        : column names in var_order for level, slope, and short-end

    Returns
    -------
    dict with six keys:
      {
        "parallel_up":   df_parallel_up,
        "parallel_down": df_parallel_down,
        "steepener":     df_steepener,
        "flattener":     df_flattener,
        "short_up":      df_short_up,
        "short_down":    df_short_down,
        "all":           df_all_concat
      }

    Each df_* is long/tidy with columns:
      ["date","var","h","yhat","draw","post_draw","scenario"]
    """

    common_kwargs = dict(
        idata=idata,
        Yhist_df=Yhist_df,
        var_order=var_order,
        p=p,
        H=H,
        draws_posterior=draws_posterior,
        draws_conditional=draws_conditional,
    )

    # 1) Parallel Up: +parallel_bps to level
    df_parallel_up = basel_conditional_curve_path(
        **common_kwargs,
        constraints={
            level_var: {
                "unit": level_unit,
                "horizons": horizons,
                "delta_bps": +parallel_bps,
            }
        },
        seed=seed_base + 1,
        scenario_name="Basel_parallel_up"
    )

    # 2) Parallel Down: -parallel_bps to level
    df_parallel_down = basel_conditional_curve_path(
        **common_kwargs,
        constraints={
            level_var: {
                "unit": level_unit,
                "horizons": horizons,
                "delta_bps": -parallel_bps,
            }
        },
        seed=seed_base + 2,
        scenario_name="Basel_parallel_down"
    )

    # 3) Steepener:
    #   Slope rises by slope_bps (yield curve steepens).
    #   Often interpreted as long rates up / short unchanged.
    #   We pin slope +slope_bps, and optionally hold level flat (+0 bps)
    df_steepener = basel_conditional_curve_path(
        **common_kwargs,
        constraints={
            slope_var: {
                "unit": slope_unit,
                "horizons": horizons,
                "delta_bps": +slope_bps,
            },
            level_var: {
                "unit": level_unit,
                "horizons": horizons,
                "delta_bps": 0.0,
            },
        },
        seed=seed_base + 3,
        scenario_name="Basel_steepener"
    )

    # 4) Flattener:
    #   Slope falls by slope_bps (curve flattens).
    #   Often "long down / short same".
    df_flattener = basel_conditional_curve_path(
        **common_kwargs,
        constraints={
            slope_var: {
                "unit": slope_unit,
                "horizons": horizons,
                "delta_bps": -slope_bps,
            },
            level_var: {
                "unit": level_unit,
                "horizons": horizons,
                "delta_bps": 0.0,
            },
        },
        seed=seed_base + 4,
        scenario_name="Basel_flattener"
    )

    # 5) Short-end Up:
    #   policy/short rate +short_bps. You usually don't pin level here,
    #   because Basel's short-end shock is very front-loaded.
    df_short_up = basel_conditional_curve_path(
        **common_kwargs,
        constraints={
            short_var: {
                "unit": short_unit,
                "horizons": horizons,
                "delta_bps": +short_bps,
            }
        },
        seed=seed_base + 5,
        scenario_name="Basel_short_end_up"
    )

    # 6) Short-end Down:
    df_short_down = basel_conditional_curve_path(
        **common_kwargs,
        constraints={
            short_var: {
                "unit": short_unit,
                "horizons": horizons,
                "delta_bps": -short_bps,
            }
        },
        seed=seed_base + 6,
        scenario_name="Basel_short_end_down"
    )

    # Concatenate into a single frame if you want to fan-chart them all
    df_all = pd.concat(
        [
            df_parallel_up,
            df_parallel_down,
            df_steepener,
            df_flattener,
            df_short_up,
            df_short_down,
        ],
        ignore_index=True,
    )

    return {
        "parallel_up": df_parallel_up,
        "parallel_down": df_parallel_down,
        "steepener": df_steepener,
        "flattener": df_flattener,
        "short_up": df_short_up,
        "short_down": df_short_down,
        "all": df_all,
    }

sns.set(style="whitegrid", font_scale=1.1)

def plot_basel_conditional_forecasts(out, save=False, folder="figs/"):
    """
    Plots median conditional forecasts for each variable in 'out'.
    Parameters
    ----------
    out : pd.DataFrame
        DataFrame with columns ['var', 'h', 'median'].
    save : bool, optional
        If True, saves PNG files in folder.
    folder : str, optional
        Destination folder for PNGs if save=True.
    """
    variables = out["var"].unique()
    nvars = len(variables)

    for v in variables:
        dfv = out[out["var"] == v]
        plt.figure(figsize=(6, 4))
        plt.plot(dfv["h"], dfv["median"], marker="o", linewidth=2, color="tab:blue")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.title(f"Basel conditional forecast: {v}", fontsize=13)
        plt.xlabel("Horizon (quarters)")
        plt.ylabel("Median forecast")
        plt.tight_layout()

        if save:
            plt.savefig(f"{folder}{v.replace(' ', '_')}_basel_conditional.png", dpi=150)
        plt.show()
        
def summarize_fan(df, probs=(0.16, 0.5, 0.84)):
    """
    df: subset of df_all for ONE scenario and ONE variable.
    returns DataFrame with columns h, p16, p50, p84
    """
    q = (df.groupby(["h"])["yhat"]
           .quantile(probs)
           .unstack(level=-1)
           .rename(columns={probs[0]:"p16", probs[1]:"p50", probs[2]:"p84"}))
    q = q.reset_index()  # keep h
    return q

def plot_curve_termsheet(df_all,
                         curve_vars=("level","slope_10y_1y","policy_rate"),
                         outdir="figs/termsheets",
                         dpi=150):
    """
    For each scenario in df_all, create a 3-panel figure:
    level, slope, policy_rate with 68% band + median.
    Saves PNGs and also shows them.
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    scenarios = df_all["scenario"].unique()

    for scen in scenarios:
        df_s = df_all[df_all["scenario"] == scen]

        # prep figure with 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(14,4), sharex=True)
        fig.suptitle(f"{scen} – Yield curve factors", fontsize=14)

        for ax, v in zip(axes, curve_vars):
            if v not in df_s["var"].unique():
                # scenario might not have pinned this var (e.g. slope in short-end-only shock)
                ax.set_title(f"{v} (not in VAR?)", fontsize=11)
                ax.axis("off")
                continue

            df_v = df_s[df_s["var"] == v]
            fan = summarize_fan(df_v)  # h, p16, p50, p84

            ax.fill_between(fan["h"], fan["p16"], fan["p84"], alpha=0.3)
            ax.plot(fan["h"], fan["p50"], linewidth=2)
            ax.axhline(0.0, linestyle="--", linewidth=1)

            ax.set_title(v, fontsize=11)
            ax.set_xlabel("Horizon (quarters)")
            ax.set_ylabel(v)

        plt.tight_layout(rect=[0,0,1,0.92])
        fname = f"{outdir}/{scen.replace(' ','_')}_curve_termsheet.png"
        plt.savefig(fname, dpi=dpi)
        plt.show()

