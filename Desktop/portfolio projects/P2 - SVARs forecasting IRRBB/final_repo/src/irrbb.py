"""NII/EVE calculators using yield and balance-sheet projections."""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import arviz as az

def toy_sat_params():
    return dict(
        # Tenor grid you used to reconstruct the curve
        # buckets_years = [0.25, 1, 2, 3, 5, 7, 10],
        buckets_years = [1, 5, 10],

        # Balance sheet snapshot (monetary units)
        balances = dict(
            loans_float=4000, loans_fixed=8000, securities_afs=3000,
            deposits_nmd=12000, deposits_term=3000, wholesale_funding=1500
        ),

        # Which tenor each bucket references (years)
        ref_tenor = dict(
            loans_float=0.25, loans_fixed=5.0, securities_afs=5.0,
            deposits_nmd=1.0, deposits_term=2.0, wholesale_funding=0.25
        ),

        # Quarterly repricing share (0–1)
        reprice = dict(
            loans_float=1.00, loans_fixed=0.10, securities_afs=0.15,
            deposits_nmd=0.60, deposits_term=0.50, wholesale_funding=1.00
        ),

        # Pass-through to reference tenor
        beta = dict(
            loans_float=1.00, loans_fixed=1.00, securities_afs=1.00,
            deposits_nmd=0.30, deposits_term=0.70, wholesale_funding=1.00
        ),

        # Lags (quarters) in re-pricing
        lags = dict(
            loans_float=0, loans_fixed=0, securities_afs=0,
            deposits_nmd=1, deposits_term=1, wholesale_funding=0
        ),

        # Additive spreads (decimals)
        spread = dict(
            loans_float=0.015, loans_fixed=0.020, securities_afs=0.005,
            deposits_nmd=0.000, deposits_term=0.000, wholesale_funding=0.002
        ),

        # Floors for deposit rates (decimals)
        floor = dict(deposits_nmd=0.00, deposits_term=0.00, wholesale_funding=0.00),

        # Durations (years), + for assets, − for liabilities
        duration = dict(
            loans_fixed=3.5, loans_float=0.2, securities_afs=4.0,
            deposits_nmd=-1.5, deposits_term=-2.0, wholesale_funding=-0.25
        )
    )


def _nearest_idx(grid, x):
    g = np.asarray(grid, float)
    return int(np.argmin(np.abs(g - float(x))))

def _lag(arr, q):
    if q <= 0: return arr
    out = np.empty_like(arr); out[:q] = arr[0]; out[q:] = arr[:-q]
    return out


def bucket_rate_paths(curve_levels_median, tenors_years, sat):
    """
    curve_levels_median: (H, n_tenors) decimals for the scenario (levels, not deltas)
    returns dict: {bucket: (H,) decimal rate path}
    """
    H, _ = curve_levels_median.shape
    rates = {}
    for k in sat["balances"].keys():
        ref = sat["ref_tenor"][k]
        beta= sat["beta"][k]
        lag = sat["lags"][k]
        spr = sat["spread"][k]
        j = _nearest_idx(tenors_years, ref)
        rref = curve_levels_median[:, j]
        r = beta * _lag(rref, lag) + spr
        if k in sat["floor"]:
            r = np.maximum(r, sat["floor"][k])
        rates[k] = r
    return rates


def toy_nii(res_curve, tenors_years, sat):
    """
    res_curve: dict from forecast_macro(...) or forecast_basel(...).
               Uses res["curves"]["levels"]["median"] and baseline counterpart.
    Returns dict with nii_shock, nii_base, nii_delta (quarterly).
    """
    lev_s = res_curve["curves"]["levels"]["median"]          # (H, n_tenors) decimals
    lev_0 = res_curve["baseline"]["curves"]["levels"]["median"]
    H = lev_s.shape[0]

    rates_s = bucket_rate_paths(lev_s, tenors_years, sat)
    rates_0 = bucket_rate_paths(lev_0, tenors_years, sat)

    def roll(rates):
        # start coupons at t=0 using first point
        coup = {k: rates[k][0] for k in rates}
        out  = np.zeros(H)
        for t in range(H):
            q_income = 0.0
            for k, bal in sat["balances"].items():
                s = sat["reprice"][k]
                coup[k] = s * rates[k][t] + (1 - s) * coup[k]  # blended coupon
                # assets positive, liabilities negative
                sign = +1.0 if k in ("loans_float","loans_fixed","securities_afs") else -1.0
                q_income += sign * bal * coup[k] / 4.0
            out[t] = q_income
        return out

    nii_s = roll(rates_s)
    nii_0 = roll(rates_0)
    return dict(nii_shock=nii_s, nii_base=nii_0, nii_delta=nii_s - nii_0)


def toy_eve(res_curve, tenors_years, sat, snap_h=1):
    """
    ΔEVE at horizon snap_h (use Δ curve median at that quarter).
    Duration * balance * Δy at the bucket’s ref tenor; durations already carry signs.
    """
    dlev = res_curve["curves"]["delta"]["median"]            # (H, n_tenors) decimals
    H = dlev.shape[0]; h = max(1, min(H, snap_h)) - 1
    dcurve = dlev[h, :]                                      # (n_tenors,)

    dE = 0.0
    for k, bal in sat["balances"].items():
        dur = sat["duration"][k]
        ref = sat["ref_tenor"][k]
        j = _nearest_idx(tenors_years, ref)
        dy = dcurve[j]                                       # decimal change
        dE += -dur * bal * dy                                # duration PV change
    return dE


# Example: macro
# sat = toy_sat_params()
# res_monet = forecast_macro(
    # "monetary_tightening",
    # accepted_macro, panel, var_order,
    # M, tenors_years, current_curve_levels,
    # h=12, shock_horizon=1, n_paths=300, seed=123
# )

# nii = toy_nii(res_monet, tenors_years, sat)
# dE  = toy_eve(res_monet, tenors_years, sat, snap_h=1)
# print("ΔNII (first 4q):", np.round(nii["nii_delta"][:4], 2))
# print("ΔEVE @ h=1:", round(dE, 2))

def forecast_basel(
    scenario,                              # "parallel_up" | "parallel_down" | "steepener" | "flattener" | "short_up" | "short_down"
    accepted_basel_tight, panel, var_order,
    M, tenors_years, basel_mag, current_curve_levels,
    p=None, h=12, shock_horizon=1, n_paths=200, seed=123,
    alphas=None                            # optional; if None we’ll compute LS alphas on the accepted set
):
    """
    Returns a dict with the same structure used by the toy NII/EVE functions:
      res["curves"]["levels"] -> {"p10","median","p90"}  (H, n_tenors) decimals
      res["curves"]["delta"]  -> {"p10","median","p90"}  (H, n_tenors) decimals
      res["baseline"]["curves"]["levels"]["median"]      (H, n_tenors) decimals
      plus: fan/paths if you want to use them.
    """
    acc = accepted_basel_tight[scenario]
    Nacc = acc["IRFs"].shape[0]

    # Resolve alphas (LS vs Basel target) on the *accepted* set if not provided
    if alphas is None:
        alphas, _y_star_bps, _r2, _rmse = compute_alphas_pp(
            acc, var_order, tenors_years, M, basel_mag[scenario], scenario
        )  # shape (Nacc,)

    # Shocked forecast (scaled)
    fan_s, paths_s = srsvar_conditional_fanchart_scaled(
        accepted_all={scenario: acc}, panel=panel, vars_order=var_order,
        scenario=scenario, alphas=alphas, p=p, h=h,
        shock_horizon=shock_horizon, n_paths=n_paths, seed=seed
    )

    # Baseline (no shock)
    fan_0, paths_0 = srsvar_conditional_fanchart_scaled(
        accepted_all={scenario: acc}, panel=panel, vars_order=var_order,
        scenario=scenario, alphas=np.zeros_like(alphas), p=p, h=h,
        shock_horizon=0, n_paths=n_paths, seed=seed
    )

    # Factor paths → tenor curves (levels, decimals)
    curves_s = curves_from_paths_levels(paths_s, var_order, M, current_curve_levels)  # (N,H,n_ten)
    curves_0 = curves_from_paths_levels(paths_0, var_order, M, current_curve_levels)  # (N,H,n_ten)
    dcurves  = curves_s - curves_0                                                    # (N,H,n_ten)

    # Summaries
    pct = lambda x,p: np.percentile(x, p, axis=0)
    res = dict(
        fan=fan_s, paths=paths_s,
        curves=dict(
            levels=dict(p10=pct(curves_s,10), median=pct(curves_s,50), p90=pct(curves_s,90)),
            delta =dict(p10=pct(dcurves ,10), median=pct(dcurves ,50), p90=pct(dcurves ,90))
        ),
        baseline=dict(
            fan=fan_0, paths=paths_0,
            curves=dict(levels=dict(median=pct(curves_0,50)))
        ),
        alphas=alphas
    )
    return res


# # Example: Basel
# res_parallel = forecast_basel(
    # "parallel_up",
    # accepted_basel_tight, panel, var_order,
    # M, tenors_years, basel_mag, current_curve_levels,
    # h=12, shock_horizon=1, n_paths=300, seed=123
# )
# nii_b = toy_nii(res_parallel, tenors_years, sat)
# dE_b  = toy_eve(res_parallel, tenors_years, sat, snap_h=1)
# print("ΔNII (first 4q):", np.round(nii_b["nii_delta"][:4], 2))
# print("ΔEVE @ h=1:", round(dE_b, 2))

def bucket_rate_paths(curve_levels, tenors_years, sat):
    """
    curve_levels: (H, n_tenors) decimals
    returns {bucket: (H,) rate path in decimals}
    """
    H, _ = curve_levels.shape
    rates = {}
    for k in sat["balances"].keys():
        ref = sat["ref_tenor"][k]
        beta= sat["beta"][k]
        lag = sat["lags"][k]
        spr = sat["spread"][k]
        j = _nearest_idx(tenors_years, ref)
        r = beta * _lag(curve_levels[:, j], lag) + spr
        if k in sat.get("floor", {}):
            r = np.maximum(r, sat["floor"][k])
        rates[k] = r
    return rates


def toy_nii_summary(res_curve, tenors_years, sat, cet1=None,
                    bands=False, q=(10,50,90),
                    curves_paths_s=None, curves_paths_0=None):
    """
    Returns a dict with:
      - nii_shock, nii_base, nii_delta (H,)
      - delta_pct_of_starting_nii (H,)  where starting NII is baseline t=1 quarter
      - optional bands dicts (p10/median/p90)
      - optional ΔEVE% of CET1 is NOT here (that's in toy_eve_summary)

    If bands=True and curves_paths_* are provided, computes path-exact bands.
    Else uses percentile-curve shortcut bands (approx).
    """
    lev_med_s = res_curve["curves"]["levels"]["median"]     # (H, n_ten)
    lev_med_0 = res_curve["baseline"]["curves"]["levels"]["median"]

    # Build bucket rates and roll NII (quarterly accrual)
    def _roll(lev):
        rates = bucket_rate_paths(lev, tenors_years, sat)
        coup = {k: rates[k][0] for k in rates}
        H = lev.shape[0]
        out = np.zeros(H)
        for t in range(H):
            q_income = 0.0
            for k, bal in sat["balances"].items():
                s = sat["reprice"][k]
                coup[k] = s * rates[k][t] + (1 - s) * coup[k]
                sign = +1.0 if k in ("loans_float","loans_fixed","securities_afs") else -1.0
                q_income += sign * bal * coup[k] / 4.0
            out[t] = q_income
        return out

    nii_s = _roll(lev_med_s)
    nii_0 = _roll(lev_med_0)
    nii_d = nii_s - nii_0

    # % of starting NII (use baseline quarter 1)
    start_nii = nii_0[0] if nii_0.size > 0 else 1.0
    start_nii = start_nii if abs(start_nii) > 1e-9 else np.sign(start_nii + 1e-9) * 1e-9
    nii_d_pct = 100.0 * (nii_d / start_nii)

    out = dict(
        nii_shock=nii_s, nii_base=nii_0, nii_delta=nii_d,
        delta_pct_of_starting_nii=nii_d_pct,
        starting_nii= start_nii
    )

    # -------- Optional bands --------
    if bands:
        p10, p50, p90 = q
        def _roll_from_curvearr(curve_arr):  # (H,n_ten) → NII(H,)
            return _roll(curve_arr)

        if curves_paths_s is not None and curves_paths_0 is not None:
            # Path-exact bands
            N, H, nT = curves_paths_s.shape
            niiS = np.vstack([_roll_from_curvearr(curves_paths_s[i]) for i in range(N)])  # (N,H)
            nii0 = np.vstack([_roll_from_curvearr(curves_paths_0[i]) for i in range(N)])  # (N,H)
            d    = niiS - nii0
            bands_obj = dict(
                shock = dict(p10=np.percentile(niiS, p10, axis=0),
                             median=np.percentile(niiS, p50, axis=0),
                             p90=np.percentile(niiS, p90, axis=0)),
                base  = dict(p10=np.percentile(nii0, p10, axis=0),
                             median=np.percentile(nii0, p50, axis=0),
                             p90=np.percentile(nii0, p90, axis=0)),
                delta = dict(p10=np.percentile(d, p10, axis=0),
                             median=np.percentile(d, p50, axis=0),
                             p90=np.percentile(d, p90, axis=0))
            )
        else:
            # Quick percentile-curve bands (approx.)
            lev = res_curve["curves"]["levels"]
            nii_shock_p10   = _roll(lev["p10"])
            nii_shock_med   = nii_s
            nii_shock_p90   = _roll(lev["p90"])
            # baseline only has median in our structure; reuse it for bands if no per-paths
            bands_obj = dict(
                shock = dict(p10=nii_shock_p10, median=nii_shock_med, p90=nii_shock_p90),
                base  = dict(p10=nii_0,        median=nii_0,        p90=nii_0),
                delta = dict(p10=nii_shock_p10 - nii_0,
                             median=nii_shock_med - nii_0,
                             p90=nii_shock_p90 - nii_0)
            )

        out["bands"] = bands_obj

    return out


def toy_eve_summary(res_curve, tenors_years, sat, cet1=None, snap_h=1,
                    bands=False, curves_paths_s=None, curves_paths_0=None, q=(10,50,90)):
    """
    Returns dict with:
      dEVE (absolute), dEVE_pct_of_CET1 (if cet1 provided), and optional bands.
    """
    dlev = res_curve["curves"]["delta"]["median"]     # (H, n_tenors)
    H = dlev.shape[0]; h = max(1, min(H, snap_h)) - 1

    def _dE_from_dcurve(dcurve_row):
        dE = 0.0
        for k, bal in sat["balances"].items():
            dur = sat["duration"][k]
            ref = sat["ref_tenor"][k]
            j = _nearest_idx(tenors_years, ref)
            dy = dcurve_row[j]                         # decimal change at ref tenor
            dE += -dur * bal * dy
        return dE

    dE = _dE_from_dcurve(dlev[h, :])
    out = dict(dEVE=dE)
    if cet1 is not None and abs(cet1) > 1e-9:
        out["dEVE_pct_of_CET1"] = 100.0 * dE / cet1

    # -------- Optional bands --------
    if bands:
        p10, p50, p90 = q
        if curves_paths_s is not None and curves_paths_0 is not None:
            # Path-exact: build dcurve per path at snap, then PV
            N = curves_paths_s.shape[0]
            dE_arr = np.empty(N)
            for i in range(N):
                dcurve_i = curves_paths_s[i, h, :] - curves_paths_0[i, h, :]
                dE_arr[i] = _dE_from_dcurve(dcurve_i)
            out["bands"] = dict(
                p10=np.percentile(dE_arr, p10),
                median=np.percentile(dE_arr, p50),
                p90=np.percentile(dE_arr, p90)
            )
        else:
            # Quick bands using curve delta bands at snap
            dlev10 = res_curve["curves"]["delta"]["p10"][h, :]
            dlev50 = res_curve["curves"]["delta"]["median"][h, :]
            dlev90 = res_curve["curves"]["delta"]["p90"][h, :]
            out["bands"] = dict(
                p10=_dE_from_dcurve(dlev10),
                median=_dE_from_dcurve(dlev50),
                p90=_dE_from_dcurve(dlev90)
            )
            if cet1 is not None and abs(cet1) > 1e-9:
                out["bands_pct_of_CET1"] = {k: 100.0*v/cet1 for k,v in out["bands"].items()}

    return out


# sat = toy_sat_params()

# # Macro example (you already have forecast_macro)
# res_monet = forecast_macro(
    # "monetary_tightening",
    # accepted_macro, panel, var_order,
    # M, tenors_years, current_curve_levels,
    # h=12, shock_horizon=1, n_paths=300, seed=123
# )
# nii_m = toy_nii_summary(res_monet, tenors_years, sat, cet1=2500.0, bands=True,
                        # curves_paths_s=curves_from_paths_levels(res_monet["paths"], var_order, M, current_curve_levels),
                        # curves_paths_0=curves_from_paths_levels(res_monet["baseline"]["paths"], var_order, M, current_curve_levels))
# eve_m = toy_eve_summary(res_monet, tenors_years, sat, cet1=2500.0, snap_h=1, bands=True,
                        # curves_paths_s=curves_from_paths_levels(res_monet["paths"], var_order, M, current_curve_levels),
                        # curves_paths_0=curves_from_paths_levels(res_monet["baseline"]["paths"], var_order, M, current_curve_levels))

# # Basel example (new helper)
# res_parallel_up = forecast_basel(
    # "parallel_up",
    # accepted_basel_tight, panel, var_order,
    # M, tenors_years, basel_mag, current_curve_levels,
    # h=12, shock_horizon=1, n_paths=300, seed=123
# )
# nii_b = toy_nii_summary(res_parallel_up, tenors_years, sat, cet1=2500.0, bands=True,
                        # curves_paths_s=curves_from_paths_levels(res_parallel_up["paths"], var_order, M, current_curve_levels),
                        # curves_paths_0=curves_from_paths_levels(res_parallel_up["baseline"]["paths"], var_order, M, current_curve_levels))
# eve_b = toy_eve_summary(res_parallel_up, tenors_years, sat, cet1=2500.0, snap_h=1, bands=True,
                        # curves_paths_s=curves_from_paths_levels(res_parallel_up["paths"], var_order, M, current_curve_levels),
                        # curves_paths_0=curves_from_paths_levels(res_parallel_up["baseline"]["paths"], var_order, M, current_curve_levels))

def _errbars(p10, med, p90):
    lo = max(med - p10, 0.0)
    hi = max(p90 - med, 0.0)
    return np.array([[lo], [hi]])  # shape (2, 1)



# sat = toy_sat_params()

# # Example with a macro scenario you've already run:
# res_monet = forecast_macro(
    # "monetary_tightening",
    # accepted_macro, panel, var_order,
    # M, tenors_years, current_curve_levels,
    # h=12, shock_horizon=1, n_paths=300, seed=123
# )

# # (Optional) pass per-path curves for path-exact bands
# curves_paths_s = curves_from_paths_levels(res_monet["paths"], var_order, M, current_curve_levels)
# curves_paths_0 = curves_from_paths_levels(res_monet["baseline"]["paths"], var_order, M, current_curve_levels)

# plot_nii_eve(
    # res_monet, tenors_years, sat, cet1=2500.0,
    # snap_h=1, currency="€",
    # title="Monetary tightening: ΔNII and ΔEVE",
    # bands=True, curves_paths_s=curves_paths_s, curves_paths_0=curves_paths_0
# )

# # Basel example using forecast_basel_toy (from the previous message):
# res_parallel_up = forecast_basel(
    # "parallel_up",
    # accepted_basel_tight, panel, var_order,
    # M, tenors_years, basel_mag, current_curve_levels,
    # h=12, shock_horizon=1, n_paths=300, seed=123
# )
# plot_nii_eve(
    # res_parallel_up, tenors_years, sat, cet1=2500.0,
    # snap_h=1, currency="€",
    # title="Basel IRRBB: Parallel +200 bps (ΔNII & ΔEVE)",
    # bands=True
# )

def plot_eve_clean(eve_obj, currency="€", title="ΔEVE summary", show_pct=True):
    """
    eve_obj: dict from toy_eve_summary(...)
      expects keys:
        - "dEVE" (float)
        - optional "dEVE_pct_of_CET1" (float)
        - optional "bands": {"p10","median","p90"}
        - optional "bands_pct_of_CET1": {"p10","median","p90"}

    Visualization:
      - vertical line = p10–p90 band (if present)
      - dot = median (or dEVE if bands missing)
      - left y-axis: absolute currency
      - right y-axis (optional): % of CET1 marker at the same x
    """
    has_bands = isinstance(eve_obj.get("bands"), dict) and all(k in eve_obj["bands"] for k in ("p10","median","p90"))

    if has_bands:
        p10 = float(eve_obj["bands"]["p10"])
        med = float(eve_obj["bands"]["median"])
        p90 = float(eve_obj["bands"]["p90"])
        dot_val = med
        lo, hi = p10, p90
    else:
        dot_val = float(eve_obj["dEVE"])
        lo, hi = dot_val, dot_val  # no band

    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))

    x0 = 0.0
    # band (p10–p90) as a vertical line
    ax1.vlines(x0, lo, hi, color="tab:blue", lw=8, alpha=0.25)
    # lollipop median (or single point)
    ax1.plot([x0], [dot_val], "o", color="tab:blue", markersize=8)
    # zero line
    ax1.axhline(0, color="k", lw=0.8)

    # x cosmetics
    ax1.set_xlim(-0.8, 0.8)
    ax1.set_xticks([x0])
    ax1.set_xticklabels(["ΔEVE"], rotation=0)
    ax1.set_ylabel(f"{currency} (absolute)")

    # title
    ax1.set_title(title, fontsize=12)

    # Right axis: % of CET1
    if show_pct and ("dEVE_pct_of_CET1" in eve_obj or "bands_pct_of_CET1" in eve_obj):
        ax2 = ax1.twinx()
        if "bands_pct_of_CET1" in eve_obj and all(k in eve_obj["bands_pct_of_CET1"] for k in ("p10","median","p90")):
            bp = eve_obj["bands_pct_of_CET1"]
            ax2.vlines(x0+0.18, bp["p10"], bp["p90"], color="tab:orange", lw=3, alpha=0.6)
            ax2.plot([x0+0.18], [bp["median"]], "o", color="tab:orange", markersize=6, label="% of CET1")
        elif "dEVE_pct_of_CET1" in eve_obj:
            ax2.plot([x0+0.18], [eve_obj["dEVE_pct_of_CET1"]], "o", color="tab:orange", markersize=6, label="% of CET1")
        ax2.set_ylabel("% of CET1")
        ax2.grid(False)

    # Pad y-lims a bit for readability
    yvals = [lo, hi, dot_val, 0.0]
    ymin = min(yvals); ymax = max(yvals)
    pad = 0.08 * (ymax - ymin + 1e-9)
    ax1.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()
    plt.show()


def eve_table(eve_obj):
    """
    Return a tidy DataFrame with p10/median/p90 for absolute ΔEVE and % CET1 (if available).
    """
    rows = []
    def _safe(dct, key, default=np.nan):
        try:
            return float(dct.get(key, default))
        except Exception:
            return default

    if "bands" in eve_obj and isinstance(eve_obj["bands"], dict):
        rows.append(dict(metric="ΔEVE (abs)", p10=_safe(eve_obj["bands"], "p10"),
                         median=_safe(eve_obj["bands"], "median"), p90=_safe(eve_obj["bands"], "p90")))
    else:
        rows.append(dict(metric="ΔEVE (abs)", p10=np.nan, median=float(eve_obj["dEVE"]), p90=np.nan))

    if "bands_pct_of_CET1" in eve_obj and isinstance(eve_obj["bands_pct_of_CET1"], dict):
        rows.append(dict(metric="ΔEVE (% CET1)", p10=_safe(eve_obj["bands_pct_of_CET1"], "p10"),
                         median=_safe(eve_obj["bands_pct_of_CET1"], "median"),
                         p90=_safe(eve_obj["bands_pct_of_CET1"], "p90")))
    elif "dEVE_pct_of_CET1" in eve_obj:
        rows.append(dict(metric="ΔEVE (% CET1)", p10=np.nan,
                         median=float(eve_obj["dEVE_pct_of_CET1"]), p90=np.nan))

    return pd.DataFrame(rows, columns=["metric","p10","median","p90"])


# Assume you already have:
# eve_obj = toy_eve_summary(res_parallel_up, tenors_years, sat, cet1=2500.0, snap_h=1, bands=True,
#                           curves_paths_s=..., curves_paths_0=...)

# eve_obj = toy_eve_summary(
        # res_monet, tenors_years, sat, cet1=2500.0, snap_h=1,
        # bands=True, curves_paths_s=curves_paths_s, curves_paths_0=curves_paths_0
    # )

# plot_eve_clean(eve_obj, currency="€", title="Basel: Parallel +200 bps — ΔEVE", show_pct=True)

# print(eve_table(eve_obj).round(2))

def plot_irf_posterior_density(accepted, var_order, shock_name, vars_to_show, horizon=0):
    """
    accepted: dict with IRFs (Ndraws, H+1, K, K)
    horizon: which horizon to plot (0 = impact)
    """
    IRFs = np.asarray(accepted["IRFs"])
    t = np.arange(IRFs.shape[1])
    fig, axes = plt.subplots(1, len(vars_to_show), figsize=(4*len(vars_to_show),3))
    for i, var in enumerate(vars_to_show):
        k = var_order.index(var)
        data = IRFs[:, horizon, k, 0]  # response of var to identified shock
        ax = axes[i] if len(vars_to_show)>1 else axes
        ax.hist(data, bins=40, color="tab:blue", alpha=0.6)
        ax.axvline(np.median(data), color="k", lw=1.2, label="median")
        ax.set_title(f"{var} @ h={horizon}")
        ax.legend()
    fig.suptitle(f"Posterior densities of IRFs to {shock_name}")
    plt.tight_layout()
    plt.show()


# plot_irf_posterior_density(accepted_macro["monetary_tightening"], var_order,
                           # "Monetary tightening", ["infl_q_ann","gdp_q_ann","policy_rate"], horizon=0)


# currency = "€"

# summary = pd.DataFrame([
    # dict(
        # scenario="Fiscal expansion",
        # shock_scale="unit structural",
        # delta_NII_1y=f"{-1.2:.1f} {currency}",
        # delta_EVE=f"{+22:.1f} {currency}",
        # delta_EVE_pct="~ +1%",
        # qualitative="Short-run funding cost ↑; slight PV gain from curve flattening"
    # ),
    # dict(
        # scenario="Parallel up (Basel +200 bp)",
        # shock_scale="+200 bp",
        # delta_NII_1y=f"+40 {currency}",
        # delta_EVE=f"−370 {currency}",
        # delta_EVE_pct="≈ −15 % of CET1",
        # qualitative="Classic asset-sensitive: NII ↑, EVE ↓"
    # )
# ])
# print(summary.to_markdown(index=False))

### NEW FUNCTIONS

def irrbb_defaults():
    """
    Realistic-ish default balance sheet numbers.
    """
    return dict(
        baseline_nii=10_000_000_000,   # 10bn
        cet1=40_000_000_000,           # 40bn
        # target sensitivities for a +200 bps parallel up
        target_nii_pct=0.06,           # +6% NII
        target_eve_pct=-0.12           # -12% of CET1
    )


def make_basel_forecast_df(df_all, scenario, probs=(0.10, 0.50, 0.90)):
    """
    Same helper we just made: summarize Basel draws to median + bands.
    """
    q_lo, q_med, q_hi = probs
    df_s = df_all[df_all["scenario"] == scenario].copy()
    if df_s.empty:
        raise ValueError(f"Scenario {scenario!r} not found.")
    g = df_s.groupby(["var","h","date"])["yhat"]
    df_q = g.quantile([q_lo, q_med, q_hi]).unstack(-1).reset_index()
    df_q = df_q.rename(columns={q_lo:"yhat_lo", q_med:"yhat", q_hi:"yhat_hi"})
    df_q["scenario"] = scenario
    return df_q


def run_irrbb_simple(
    df_all: pd.DataFrame,
    scenario: str,
    *,
    curve_vars=("level","slope_10y_1y","policy_rate"),
    horizon_nii=4,
    par_shift_bps=200,
    defaults=None,
    baseline_level=None
):
    """
    Take Basel conditional forecasts, turn them into a simple NII/EVE impact
    with realistic magnitudes.

    Assumptions:
    - NII reacts mainly to *short* and *level* parts
    - EVE reacts mainly to *level* part
    - We scale the impacts to match target_% in irrbb_defaults()
    """
    if defaults is None:
        defaults = irrbb_defaults()

    baseline_nii = defaults["baseline_nii"]
    cet1         = defaults["cet1"]
    target_nii_pct = defaults["target_nii_pct"]
    target_eve_pct = defaults["target_eve_pct"]

    fan = make_basel_forecast_df(df_all, scenario)
    # get median paths
    level_path = fan[fan["var"] == "level"].sort_values("h")["yhat"].values
    slope_path = fan[fan["var"] == "slope_10y_1y"].sort_values("h")["yhat"].values \
                 if "slope_10y_1y" in fan["var"].unique() else np.zeros_like(level_path)
    short_path = fan[fan["var"] == "policy_rate"].sort_values("h")["yhat"].values \
                 if "policy_rate" in fan["var"].unique() else np.zeros_like(level_path)

    # we only need first horizon_nii quarters
    L = min(horizon_nii, len(level_path))
    level_path = level_path[:L]
    slope_path = slope_path[:L]
    short_path = short_path[:L]

    # we need to know how big the *intended* Basel shock was (e.g. +200 bps)
    # so we can scale from "model units" to "reg unit".
    # assume the first-period level move is the key driver:
    
    if baseline_level is None:
        raise ValueError("Pass baseline_level=... (last observed level in percent).")
    
    # last_level = panel["level"].iloc[-1]
    shocked_level_h1 = level_path[0]       # e.g. 4.19
    model_level_move = shocked_level_h1 - baseline_level
    
    # model_level_move = level_path[0] - level_path[0] * 0 + 0  # just level_path[0]
    if model_level_move == 0:
        model_level_move = 1e-6  # avoid div0

    # scaling factor: how many bps is one model unit?
    # we want model_level_move  --> par_shift_bps
    model_to_bps = par_shift_bps / (model_level_move * 10000) if abs(model_level_move) > 1e-9 else 0.0
    # (if your level is in percent, model_level_move=2.0 means 200 bps -> 2.0 * 100 = 200? adjust if needed)

    # For simplicity, let's just take the *relative* shape and then scale NII/EVE
    # NII: assume it loads 0.7 on short, 0.3 on level
    nii_factor = 0.7 * short_path + 0.3 * level_path
    # Now scale so that the *first* quarter NII change equals target_nii_pct * baseline_nii
    # for a 200 bps shock.
    # First quarter model move in "nii_factor":
    model_nii_move = nii_factor[0]
    if model_nii_move == 0:
        nii_scale = 0.0
    else:
        target_nii_amount = baseline_nii * target_nii_pct
        nii_scale = target_nii_amount / model_nii_move

    delta_nii = nii_factor * nii_scale   # (L,)

    # EVE: load only on level, scale to target_eve_pct of CET1
    model_eve_move = level_path[0]
    if model_eve_move == 0:
        eve_scale = 0.0
    else:
        target_eve_amount = cet1 * target_eve_pct
        eve_scale = target_eve_amount / model_eve_move

    delta_eve = level_path[0] * eve_scale  # just h=1 for EVE

    # build tidy outputs
    nii_df = pd.DataFrame({
        "scenario": scenario,
        "h": np.arange(1, L+1),
        "delta_nii": delta_nii,
        "delta_nii_pct_of_baseline": delta_nii / baseline_nii
    })

    eve_df = pd.DataFrame({
        "scenario": [scenario],
        "h": [1],
        "delta_eve": [delta_eve],
        "delta_eve_pct_of_cet1": [delta_eve / cet1]
    })

    return nii_df, eve_df, fan

def plot_nii_simple(nii_df):
    plt.figure(figsize=(6,4))
    plt.plot(nii_df["h"], nii_df["delta_nii_pct_of_baseline"]*100, marker="o", label="ΔNII (% of NII)")
    plt.xlabel("Horizon (quarters)")
    plt.ylabel("ΔNII (% of baseline)")
    plt.title(f"{nii_df['scenario'].iloc[0]} – NII path")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def print_eve_simple(eve_df):
    row = eve_df.iloc[0]
    print(f"{row['scenario']} – ΔEVE @ h=1: {row['delta_eve']:.0f} "
          f"({row['delta_eve_pct_of_cet1']*100:.1f}% of CET1)")

def extract_conditional_yieldpaths(df_all, curve_vars=("level","slope_10y_1y","policy_rate")):
    """
    Returns tidy DataFrame with median and 68% bands for each scenario/curve_var/horizon.
    """
    q = (df_all[df_all["var"].isin(curve_vars)]
         .groupby(["scenario","var","h"])["yhat"]
         .quantile([0.16,0.5,0.84])
         .unstack(level=-1)
         .rename(columns={0.16:"p16",0.5:"median",0.84:"p84"})
         .reset_index())
    return q
    
def compute_nii_eve_from_curvepaths(curve_paths, yield_sensitivities, horizon_nii=4):
    """
    Generic pipeline: multiply conditional yield moves by portfolio sensitivities.
    yield_sensitivities: dict like {'level': beta_level, 'slope_10y_1y': beta_slope, 'policy_rate': beta_short}
    Returns DataFrame with ΔNII (1..horizon_nii) and ΔEVE for each scenario.
    """
    results = []

    for scen in curve_paths["scenario"].unique():
        df_s = curve_paths[curve_paths["scenario"] == scen]
        df_l = df_s[df_s["var"] == "level"]
        df_sl = df_s[df_s["var"] == "slope_10y_1y"]
        df_sh = df_s[df_s["var"] == "policy_rate"]

        # compute synthetic total yield shift (weighted)
        delta_yield = (yield_sensitivities["level"] * df_l["median"].values[:horizon_nii] +
                       yield_sensitivities["slope_10y_1y"] * df_sl["median"].values[:horizon_nii] +
                       yield_sensitivities["policy_rate"] * df_sh["median"].values[:horizon_nii])

        # Simple linearized NII impact (bps * sensitivity)
        delta_nii = delta_yield * yield_sensitivities.get("nii_multiplier", 1.0)

        # EVE at h=1: approximate duration effect of level shift
        delta_eve = yield_sensitivities.get("eve_duration", -20.0) * df_l["median"].values[0]

        results.append(pd.DataFrame({
            "scenario": scen,
            "h": np.arange(1, horizon_nii+1),
            "ΔNII": delta_nii,
            "ΔEVE": delta_eve
        }))
    return pd.concat(results, ignore_index=True)
    
def build_plots_nii_eve(nii_eve):
    for scen in nii_eve["scenario"].unique():
        df_s = nii_eve[nii_eve["scenario"]==scen]
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(df_s["h"], df_s["ΔNII"], "o-", label="ΔNII (first 4q)")
        ax1.set_xlabel("Horizon (quarters)")
        ax1.set_ylabel("ΔNII")
        ax1.legend(loc="upper right")
        plt.title(f"{scen} – NII path")
        plt.tight_layout()
        plt.show()

        print(f"{scen}: ΔEVE @ h=1 = {df_s['ΔEVE'].iloc[0]:.2f}")
        
def plot_nii(
    res_curve,
    tenors_years,
    sat,
    currency="€",
    title="Scenario impact on NII",
    bands=True,
    curves_paths_s=None,
    curves_paths_0=None,
    nii_axis="currency",    # "currency" | "bps_assets" | "%assets"
    assets_total=None,      # required for "bps_assets" or "%assets"
    annualize=True          # annualize before converting to bps/%assets
):
    """
    Plot ΔNII paths under a given scenario.

    nii_axis:
      - "currency": ΔNII in currency/quarter (y-axis in `currency`)
      - "bps_assets": ΔNII as *annualized* bps of total Assets (needs assets_total)
      - "%assets": ΔNII as *annualized* % of total Assets (needs assets_total)

    res_curve: output of forecast_macro(...) or forecast_basel_toy(...)
    """

    # ----- compute summaries -----
    nii_obj = toy_nii_summary(
        res_curve,
        tenors_years,
        sat,
        cet1=None,  # not needed here
        bands=bands,
        curves_paths_s=curves_paths_s,
        curves_paths_0=curves_paths_0,
    )

    d_abs_q = nii_obj["nii_delta"]   # currency per quarter, shape (H,)
    H = d_abs_q.shape[0]
    h = np.arange(1, H + 1)

    # ---- helper: build y-series for panel (A) depending on axis mode ----
    def _to_axis(series_q):
        # series_q: per quarter (currency)
        if nii_axis == "currency":
            return series_q, f"{currency} per quarter"

        if assets_total is None:
            raise ValueError("assets_total is required for 'bps_assets' or '%assets'")

        scale = 4.0 if annualize else 1.0
        series_ann = series_q * scale

        if nii_axis == "bps_assets":
            y = (series_ann / assets_total) * 1e4  # bps per year
            label = "bps of Assets (annualized)" if annualize else "bps of Assets (quarterly)"
            return y, label

        if nii_axis == "%assets":
            y = (series_ann / assets_total) * 100.0  # % per year
            label = "% of Assets (annualized)" if annualize else "% of Assets (quarterly)"
            return y, label

        raise ValueError(f"Unknown nii_axis='{nii_axis}'")

    yA, yA_label = _to_axis(d_abs_q)

    # If we have ΔNII bands, convert those too (consistently)
    def _get_band(obj, key):
        if "bands" not in obj:
            return None
        b = obj["bands"][key]
        return b.get("p10", None), b.get("median", None), b.get("p90", None)

    nii_delta_bands = _get_band(nii_obj, "delta")
    if nii_delta_bands and all(x is not None for x in nii_delta_bands):
        lo_q, md_q, hi_q = nii_delta_bands
        loA, _ = _to_axis(lo_q)
        mdA, _ = _to_axis(md_q)
        hiA, _ = _to_axis(hi_q)
    else:
        loA = mdA = hiA = None

    # ----- layout: 2 panels (ΔNII in units, ΔNII as % of starting NII) -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    ax1, ax2 = axes

    # (A) ΔNII in chosen units
    if loA is not None:
        ax1.fill_between(h, loA, hiA, alpha=0.15)
        ax1.plot(h, mdA, lw=2)
    else:
        ax1.plot(h, yA, lw=2)

    ax1.axhline(0, lw=1)
    ax1.set_title(f"ΔNII vs baseline ({yA_label})")
    ax1.set_xlabel("h (quarters)")
    ax1.set_ylabel(yA_label)

    # (B) ΔNII as % of starting NII
    d_pct = nii_obj["delta_pct_of_starting_nii"]
    if nii_delta_bands and all(x is not None for x in nii_delta_bands):
        start_nii = nii_obj["starting_nii"]
        denom = start_nii if abs(start_nii) > 1e-12 else np.sign(start_nii + 1e-12) * 1e-12
        ax2.fill_between(h, 100.0 * lo_q / denom, 100.0 * hi_q / denom, alpha=0.15)
        ax2.plot(h, 100.0 * md_q / denom, lw=2)
    else:
        ax2.plot(h, d_pct, lw=2)

    ax2.axhline(0, lw=1)
    ax2.set_title("ΔNII as % of starting NII")
    ax2.set_xlabel("h (quarters)")
    ax2.set_ylabel("%")

    fig.suptitle(title, fontsize=13)
    fig.text(
        0.5,
        -0.02,
        "Notes: Curves in decimals; NII is quarterly. ΔNII bands are 10–90%.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.show()


# IRRBB CALC PIPELINE

# --- 0. Utilities to summarize draws ----------------------------------------

def summarize_fan(df, probs=(0.16, 0.5, 0.84)):
    """
    df: subset of df_all for ONE scenario and ONE variable.
    returns DataFrame with columns h, p16, p50, p84
    """
    q = (df.groupby(["h"])["yhat"]
           .quantile(probs)
           .unstack(level=-1)
           .rename(columns={probs[0]:"p16", probs[1]:"p50", probs[2]:"p84"}))
    return q.reset_index()  # keep h


def extract_conditional_yieldpaths(df_all, curve_vars=("level","slope_10y_1y","policy_rate")):
    """
    Collapse df_all down to (scenario, var, h, p16, median, p84).
    Used mostly for sanity checks / dashboards.
    """
    q = (df_all[df_all["var"].isin(curve_vars)]
         .groupby(["scenario","var","h"])["yhat"]
         .quantile([0.16,0.5,0.84])
         .unstack(level=-1)
         .rename(columns={0.16:"p16",0.5:"median",0.84:"p84"})
         .reset_index())
    return q


# --- 1. Scenario -> curve cube adapter --------------------------------------

def dfscenario_to_rescurve(
    df_all: pd.DataFrame,
    scenario_name: str,
    var_order: list,
    M: np.ndarray,
    tenors_years: np.ndarray,
    current_curve_levels: np.ndarray,
    horizon_H: int,
    baseline_name: str = None,
):
    """
    Translate the conditional BVAR scenario draws (df_all) into the 'res_curve'
    structure expected by irrbb.toy_nii_summary / toy_eve_summary.

    Inputs
    ------
    df_all : long dataframe with columns:
        ["date","var","h","yhat","draw","post_draw","scenario"]
        (output of make_ecb_basel6()["all"])
    scenario_name : string, e.g. "Basel_parallel_up"
    baseline_name : optional string naming an "unshocked" or baseline scenario
        in df_all to compare against. If None, we fabricate a flat baseline.
    var_order : list of variables in VAR order. Must include:
        "level", "slope_10y_1y", "curvature_ns_like"
    M : (n_tenors x 3) mapping from [level, slope, curvature] -> delta yield by tenor
        This is your term structure factor loading matrix.
    tenors_years : array of tenor buckets (e.g. sat["buckets_years"])
    current_curve_levels : (n_tenors,) last observed absolute yield curve in decimals
    horizon_H : forecast horizon (quarters)
    baseline_name : name of baseline/unconstrained scenario in df_all, if available.
                    We'll use it to compute deltas vs baseline.

    Returns
    -------
    res_curve : dict
        {
          "paths": shocked_curve_paths (N,H,n_ten),
          "baseline": {
              "paths": baseline_curve_paths (N,H,n_ten),
              "curves": {"levels": {"median": (H,n_ten)}}
          },
          "curves": {
              "levels": {"p10","median","p90"}, (H,n_ten)
              "delta":  {"p10","median","p90"}, (H,n_ten)
          },
          ...
        }
    curves_paths_s_levels : (N,H,n_ten)
    curves_paths_b_levels : (N,H,n_ten)
    """

    # Grab scenario draws
    df_s = df_all[df_all["scenario"] == scenario_name].copy()

    # Grab baseline draws, if provided
    if baseline_name is not None and baseline_name in df_all["scenario"].unique():
        df_b = df_all[df_all["scenario"] == baseline_name].copy()
    else:
        df_b = None

    # Combine post_draw and draw into a single ID so each posterior+conditional sample is one draw
    df_s["global_draw"] = list(zip(df_s["post_draw"], df_s["draw"]))
    groups_s = df_s.groupby("global_draw")

    # Map variable names to columns
    var_to_idx = {v:i for i,v in enumerate(var_order)}
    K = len(var_order)
    H = horizon_H

    # Build factor array (N_s, H, K)
    paths_fac_s = []
    for (_, _), g in groups_s:
        # fill HxK with NaN, then populate
        mat = np.full((H, K), np.nan)
        for v, sub in g.groupby("var"):
            if v not in var_to_idx:
                continue
            k = var_to_idx[v]
            for _, row in sub.iterrows():
                h_idx = int(row["h"]) - 1
                if 0 <= h_idx < H:
                    mat[h_idx, k] = row["yhat"]
        paths_fac_s.append(mat)
    paths_fac_s = np.array(paths_fac_s)  # (N_s,H,K)

    # Baseline factors, same construction if available
    if df_b is not None and not df_b.empty:
        df_b["global_draw"] = list(zip(df_b["post_draw"], df_b["draw"]))
        groups_b = df_b.groupby("global_draw")
        paths_fac_b = []
        for (_, _), g in groups_b:
            mat = np.full((H, K), np.nan)
            for v, sub in g.groupby("var"):
                if v not in var_to_idx:
                    continue
                k = var_to_idx[v]
                for _, row in sub.iterrows():
                    h_idx = int(row["h"]) - 1
                    if 0 <= h_idx < H:
                        mat[h_idx, k] = row["yhat"]
            paths_fac_b.append(mat)
        paths_fac_b = np.array(paths_fac_b)  # (N_b,H,K)
    else:
        paths_fac_b = None  # we'll fabricate flat baseline later

    # Identify which factor indices map to curve
    if "level" not in var_to_idx:
        raise ValueError("The var_order must contain 'level'.")
    if "slope_10y_1y" not in var_to_idx:
        raise ValueError("The var_order must contain 'slope_10y_1y'.")
    if "curvature_ns_like" not in var_to_idx:
        raise ValueError("The var_order must contain 'curvature_ns_like'.")

    i_level = var_to_idx["level"]
    i_slope = var_to_idx["slope_10y_1y"]
    i_curv  = var_to_idx["curvature_ns_like"]

    # Map factors -> full curve
    # We assume linear: yield(τ) ≈ current_curve_levels[τ] + M[τ,:] @ [level, slope, curvature]
    # If you already have a more refined function (e.g. Nelson-Siegel parametrization),
    # replace this inner function by that.
    def factors_to_curve(level, slope, curvature):
        vec = np.array([level, slope, curvature])  # shape (3,)
        return current_curve_levels + M @ vec      # (n_tenors,)

    def build_curve_cube(paths_fac):
        """
        paths_fac: (N,H,K)
        returns out: (N,H,n_tenors) absolute curve levels in decimals
        """
        N = paths_fac.shape[0]
        n_ten = len(tenors_years)
        out = np.zeros((N, H, n_ten))
        for i in range(N):
            for h in range(H):
                lvl = paths_fac[i,h,i_level]
                slp = paths_fac[i,h,i_slope]
                crv = paths_fac[i,h,i_curv]
                out[i,h,:] = factors_to_curve(lvl, slp, crv)
        return out

    curves_paths_s_levels = build_curve_cube(paths_fac_s)  # (N_s,H,n_ten)

    # Baseline curve cube
    if paths_fac_b is not None:
        curves_paths_b_levels = build_curve_cube(paths_fac_b)  # (N_b,H,n_ten)
    else:
        # no explicit baseline scenario => assume baseline is flat current curve
        N_s, H_, n_ten = curves_paths_s_levels.shape
        assert H_ == H
        flat = np.tile(current_curve_levels, (H,1))        # (H,n_ten)
        curves_paths_b_levels = np.tile(flat, (N_s,1,1))   # (N_s,H,n_ten)

    # Compute deltas vs baseline
    # We'll broadcast: if N_s != N_b, we'll match sizes by simple truncation/min.
    N_use = min(curves_paths_s_levels.shape[0], curves_paths_b_levels.shape[0])
    s_use = curves_paths_s_levels[:N_use]   # (N_use,H,n_ten)
    b_use = curves_paths_b_levels[:N_use]   # (N_use,H,n_ten)
    diff  = s_use - b_use                   # (N_use,H,n_ten)

    # Summaries p10/50/90 across draws
    def pct(arr,p):
        return np.percentile(arr, p, axis=0)  # (H,n_ten)

    curves_levels_summary = dict(
        p10    = pct(s_use,10),
        median = pct(s_use,50),
        p90    = pct(s_use,90)
    )
    curves_delta_summary = dict(
        p10    = pct(diff,10),
        median = pct(diff,50),
        p90    = pct(diff,90)
    )
    baseline_levels_summary = dict(
        median = pct(b_use,50)
    )

    # This matches the shape your irrbb functions expect.
    res_curve = dict(
        paths = s_use,  # shocked scenario paths (N_use,H,n_ten)
        baseline = dict(
            paths = b_use,  # baseline curve paths (N_use,H,n_ten)
            curves = dict(
                levels = baseline_levels_summary  # 'median' baseline curve (H,n_ten)
            )
        ),
        curves = dict(
            levels = curves_levels_summary,  # absolute curve
            delta  = curves_delta_summary    # shocked - baseline
        )
    )

    return res_curve, s_use, b_use


# --- 2. High-level wrapper to go from df_all -> NII/EVE objects -----------

def scenario_to_nii_eve_objects(
    df_all: pd.DataFrame,
    scenario_name: str,
    var_order: list,
    M: np.ndarray,
    tenors_years: np.ndarray,
    current_curve_levels: np.ndarray,
    horizon_H: int,
    sat: dict,
    cet1_amount: float,
    irrbb_module,
    baseline_name: str = None,
    snap_h_eve: int = 1,
    nii_axis: str = "currency",
    assets_total=None
):
    """
    One-stop helper:
      1. Build res_curve for the scenario
      2. Call toy_nii_summary / toy_eve_summary / plot_nii_eve / plot_eve_clean
      3. Return nii_obj, eve_obj, and a summary dict we can table later

    irrbb module functions :
        toy_nii_summary
        toy_eve_summary
        plot_nii_eve
        plot_eve_clean
        eve_table
    """

    # (a) scenario curves -> IRRBB input
    res_curve, curves_paths_s, curves_paths_b = dfscenario_to_rescurve(
        df_all=df_all,
        scenario_name=scenario_name,
        var_order=var_order,
        M=M,
        tenors_years=tenors_years,
        current_curve_levels=current_curve_levels,
        horizon_H=horizon_H,
        baseline_name=baseline_name
    )

    # (b) NII summary
    nii_obj = toy_nii_summary(
        res_curve,
        tenors_years=tenors_years,
        sat=sat,
        bands=True,
        curves_paths_s=curves_paths_s,
        curves_paths_0=curves_paths_b
    )

    # (c) EVE summary
    eve_obj = toy_eve_summary(
        res_curve,
        tenors_years=tenors_years,
        sat=sat,
        cet1=cet1_amount,
        snap_h=snap_h_eve,
        bands=True,
        curves_paths_s=curves_paths_s,
        curves_paths_0=curves_paths_b
    )

    # (d) Quick plots (these functions already know how to visualize ΔNII & ΔEVE)
    plot_nii(
        res_curve,
        tenors_years=tenors_years,
        sat=sat,
        currency="€",
        title=f"{scenario_name}: ΔNII & ΔEVE",
        bands=True,
        curves_paths_s=curves_paths_s,
        curves_paths_0=curves_paths_b,
        nii_axis=nii_axis,
        assets_total=assets_total
    )

    # unused:
    # plot_eve_clean(
        # eve_obj,
        # currency="€",
        # title=f"{scenario_name} – ΔEVE",
        # show_pct=True
    # )

    # (e) Tabular summary (ΔEVE etc.)
    eve_tab = eve_table(eve_obj)

    # (f) Extract core metrics we’ll want for the “summary table for all 6 scenarios”
    # We'll define:
    #   - cum_NII_4q_median: cumulative ΔNII over first 4 quarters (median)
    #   - EVE_h1_median: ΔEVE at snap_h_eve, median
    #   - EVE_h1_pctCET1: ΔEVE / CET1, median

    # nii_obj: depending on your implementation,
    # you likely stored paths for delta NII by horizon in nii_obj["nii"]["delta"]["median"]
    # If not, adjust these keys to match your actual structure.
    # nii_delta_median = nii_obj["nii"]["delta"]["median"]  # shape (H,)
    nii_delta_median = nii_obj["bands"]['delta']['median']
    cum_NII_4q_median = np.sum(nii_delta_median[:4])

    # eve_delta_median = eve_obj["eve"]["delta"]["median"][snap_h_eve-1]
    eve_delta_median = eve_obj["bands"]["median"]
    eve_pctCET1_median = eve_obj["dEVE_pct_of_CET1"]

    summary = dict(
        scenario=scenario_name,
        cum_NII_4q_median=cum_NII_4q_median,
        EVE_h1_median=eve_delta_median,
        EVE_h1_pctCET1=eve_pctCET1_median
    )

    return {
        "res_curve": res_curve,
        "nii_obj": nii_obj,
        "eve_obj": eve_obj,
        "eve_table": eve_tab,
        "summary": summary
    }

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
    
def accepted_to_df_all(
    accepted_dict: dict,
    var_order: list,
    last_date: pd.Timestamp,
    freq: str = "Q"
) -> pd.DataFrame:
    """
    Convert a dict like accepted_macro into a long DataFrame like df_all
    so it can be fed to scenario_to_nii_eve_objects or to the fan-plotters.

    Parameters
    ----------
    accepted_dict : dict
        e.g. accepted_macro["monetary_tightening"]["forecasts"] = (N, H, K)
    var_order : list[str]
        VAR variable ordering (length K)
    last_date : pd.Timestamp
        last historical date; forecasts start after this
    freq : str
        pandas frequency, usually "Q" for your setup

    Returns
    -------
    df_all_like : pd.DataFrame
        columns = ["date","var","h","yhat","draw","post_draw","scenario"]
    """
    all_frames = []

    for scen_name, scen_obj in accepted_dict.items():
        if "forecasts" not in scen_obj:
            raise ValueError(f"Scenario {scen_name} has no 'forecasts' key.")

        paths = np.asarray(scen_obj["forecasts"])   # (N, H, K)
        N, H, K = paths.shape

        # build dates for this horizon
        dates = pd.date_range(last_date, periods=H+1, freq=freq)[1:]

        # we’ll treat each N as: post_draw = n, draw = 0 (or draw = n, post_draw = 0)
        # to match your Basel schema better, we can split like this:
        #   post_draw = n
        #   draw      = 0
        rows = []
        for n in range(N):
            # (H, K) for this draw
            mat = paths[n, :, :]  # (H,K)
            df_n = pd.DataFrame({
                "date": np.repeat(dates, K),
                "var": np.tile(var_order, H),
                "h": np.repeat(np.arange(1, H+1), K),
                "yhat": mat.reshape(-1),
                "draw": 0,            # single conditional draw per posterior
                "post_draw": n,
                "scenario": scen_name
            })
            rows.append(df_n)

        all_frames.append(pd.concat(rows, ignore_index=True))

    df_all_like = pd.concat(all_frames, ignore_index=True)
    return df_all_like

def make_forecasts_from_irfs(accepted_macro, shock_size=1.0):
    """
    Create pseudo forecast paths (N,H,K) for each scenario
    from the stored IRFs.
    """
    for scen, obj in accepted_macro.items():
        IRFs = np.asarray(obj["IRFs"])   # (N, H+1, K, K)
        if IRFs.ndim == 3:  # (H+1,K,K) single draw
            IRFs = IRFs[np.newaxis, ...]

        N, H1, K, _ = IRFs.shape
        H = H1 - 1

        # apply a +1σ shock on the main identified shock
        # (often column 0 corresponds to the shock of interest)
        shock_vec = np.zeros(K)
        shock_vec[0] = shock_size

        # compute yhat_t = IRF_t @ shock
        paths = np.einsum("nhkj,j->nhk", IRFs[:,1:,:,:], shock_vec)
        accepted_macro[scen]["forecasts"] = paths  # store as (N,H,K)

    return accepted_macro

def build_irrbb_df_from_objects(nii_eve_df, out_monet=None, out_fiscal=None):
    """
    nii_eve_df: DataFrame from compute_nii_eve_from_curvepaths with cols
                ['scenario','h','ΔNII','ΔEVE']
    out_monet, out_fiscal: outputs from scenario_to_nii_eve_objects
                           (each has a 'summary' dict)
    returns: tidy df with one row per scenario, h=1
    """
    # 1. start from Basel results, keep h=1
    base = nii_eve_df[nii_eve_df["h"] == 1].copy()
    base = base[["scenario", "ΔNII", "ΔEVE"]].reset_index(drop=True)

    rows = [base]

    # 2. add monetary scenario if provided
    if out_monet is not None and "summary" in out_monet:
        s = out_monet["summary"]
        rows.append(pd.DataFrame([{
            "scenario": s.get("scenario", "monetary_tightening"),
            # your scenario_to_nii_eve_objects stored NII as "cum_NII_4q_median"
            # and EVE as "EVE_h1_median"
            "ΔNII": s.get("cum_NII_4q_median", np.nan),
            "ΔEVE": s.get("EVE_h1_median", np.nan),
        }]))

    # 3. add fiscal scenario if provided
    if out_fiscal is not None and "summary" in out_fiscal:
        s = out_fiscal["summary"]
        rows.append(pd.DataFrame([{
            "scenario": s.get("scenario", "fiscal_expansion"),
            "ΔNII": s.get("cum_NII_4q_median", np.nan),
            "ΔEVE": s.get("EVE_h1_median", np.nan),
        }]))

    full = pd.concat(rows, ignore_index=True)
    return full

def plot_irrbb_all_scenarios(df, baseline_nii=10_000_000_000, cet1=40_000_000_000):
    """
    Plot ΔEVE and ΔNII on the same axis (% values, same baseline).
    """
    df = df.copy()
    df["ΔNII_%1y"] = df["ΔNII"] / baseline_nii * 100.0
    df["ΔEVE_%CET1"] = df["ΔEVE"] / cet1 * 100.0

    # keep the sign convention consistent: EVE ↓ when rates ↑
    df = df.sort_values("scenario")
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(x - width/2, df["ΔEVE_%CET1"], width,
           label="ΔEVE (% CET1)", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, df["ΔNII_%1y"], width,
           label="ΔNII (1y, %)", color="orange", alpha=0.8)

    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["scenario"], rotation=20, ha="right")
    ax.set_ylabel("Change (%)")
    ax.set_title("IRRBB – Economic Value and Earnings Sensitivity")
    ax.legend(frameon=False, loc='best')

    fig.tight_layout()
    plt.show()

def plot_eve_tornado(df_summary: pd.DataFrame, cet1_eur: float):
    """
    df_summary must have columns: ['scenario','ΔEVE'] in € (h=1 snapshot).
    Plots ΔEVE as % of CET1 across scenarios (sorted), horizontal bars.
    """
    df = df_summary.copy()
    df["ΔEVE_%CET1"] = 100.0 * df["ΔEVE"] / cet1_eur
    df = df.sort_values("ΔEVE_%CET1")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(df["scenario"], df["ΔEVE_%CET1"])

    ax.axvline(0.0, linewidth=0.8, alpha=0.6)
    ax.set_xlabel("ΔEVE (% of CET1)")
    ax.set_ylabel("")
    ax.set_title("IRRBB — ΔEVE across scenarios (h=1)")

    # optional: add value labels
    for y, v in enumerate(df["ΔEVE_%CET1"].values):
        ax.annotate(f"{v:.1f}%", xy=(v, y), xytext=(4, -2),
                    textcoords="offset points", ha="left", va="center")

    fig.tight_layout()
    plt.show()

# old NII and EVE plot function
# def plot_nii_eve(
    # res_curve, tenors_years, sat, cet1=None,
    # snap_h=1,
    # currency="€",
    # title="Scenario impact on NII and EVE",
    # bands=True,
    # curves_paths_s=None, curves_paths_0=None,
    # # --- NEW toggles for ΔNII axis A ---
    # nii_axis="currency",            # "currency" | "bps_assets" | "%assets"
    # assets_total=None,              # required for "bps_assets" or "%assets"
    # annualize=True                  # if True, annualize NII (×4) before converting to bps/%assets
# ):
    # """
    # nii_axis:
      # - "currency": ΔNII in currency/quarter (y-axis in `currency`)
      # - "bps_assets": ΔNII as *annualized* bps of total Assets (needs assets_total)
      # - "%assets": ΔNII as *annualized* % of total Assets (needs assets_total)

    # res_curve: output of forecast_macro(...) or forecast_basel_toy(...)
    # """
    # # ----- compute summaries (reuse your toy helpers) -----
    # nii_obj = toy_nii_summary(
        # res_curve, tenors_years, sat, cet1=cet1,
        # bands=bands, curves_paths_s=curves_paths_s, curves_paths_0=curves_paths_0
    # )
    # eve_obj = toy_eve_summary(
        # res_curve, tenors_years, sat, cet1=cet1, snap_h=snap_h,
        # bands=bands, curves_paths_s=curves_paths_s, curves_paths_0=curves_paths_0
    # )

    # d_abs_q = nii_obj["nii_delta"]          # currency per quarter
    # H = d_abs_q.shape[0]
    # h = np.arange(1, H+1)

    # # ---- helper: build y-series for panel (A) depending on axis mode ----
    # def _to_axis(series_q):
        # # series_q: per quarter (currency)
        # if nii_axis == "currency":
            # return series_q, f"{currency} per quarter"
        # if assets_total is None:
            # raise ValueError("assets_total is required for 'bps_assets' or '%assets'")
        # scale = 4.0 if annualize else 1.0
        # series_ann = series_q * scale
        # if nii_axis == "bps_assets":
            # y = (series_ann / assets_total) * 1e4          # bps per year
            # return y, "bps of Assets (annualized)" if annualize else "bps of Assets (quarterly)"
        # if nii_axis == "%assets":
            # y = (series_ann / assets_total) * 100.0        # % per year
            # return y, "% of Assets (annualized)" if annualize else "% of Assets (quarterly)"
        # raise ValueError(f"Unknown nii_axis='{nii_axis}'")

    # yA, yA_label = _to_axis(d_abs_q)

    # # If we have ΔNII bands, convert those too (consistently)
    # def _get_band(obj, key):
        # if "bands" not in obj: return None
        # b = obj["bands"][key]
        # return b.get("p10", None), b.get("median", None), b.get("p90", None)

    # nii_delta_bands = _get_band(nii_obj, "delta")
    # if nii_delta_bands and all(x is not None for x in nii_delta_bands):
        # lo_q, md_q, hi_q = nii_delta_bands
        # loA, _ = _to_axis(lo_q); mdA, _ = _to_axis(md_q); hiA, _ = _to_axis(hi_q)
    # else:
        # loA = mdA = hiA = None

    # # ----- layout -----
    # fig = plt.figure(figsize=(12, 7))
    # gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], width_ratios=[1,1], hspace=0.35, wspace=0.25)

    # # (A) ΔNII in chosen units
    # ax1 = fig.add_subplot(gs[0, 0])
    # if loA is not None:
        # ax1.fill_between(h, loA, hiA, alpha=0.15)
        # ax1.plot(h, mdA, lw=2)
    # else:
        # ax1.plot(h, yA, lw=2)
    # ax1.axhline(0, lw=1)
    # ax1.set_title(f"ΔNII vs baseline ({yA_label})")
    # ax1.set_xlabel("h (quarters)")
    # ax1.set_ylabel(yA_label)

    # # (B) ΔNII as % of starting NII (same as before)
    # d_pct = nii_obj["delta_pct_of_starting_nii"]
    # ax2 = fig.add_subplot(gs[0, 1])
    # if nii_delta_bands and all(x is not None for x in nii_delta_bands):
        # start_nii = nii_obj["starting_nii"]
        # # Convert absolute bands to % bands safely (avoid div-zero)
        # denom = start_nii if abs(start_nii) > 1e-12 else np.sign(start_nii + 1e-12) * 1e-12
        # ax2.fill_between(h, 100.0*lo_q/denom, 100.0*hi_q/denom, alpha=0.15)
        # ax2.plot(h, 100.0*md_q/denom, lw=2)
    # else:
        # ax2.plot(h, d_pct, lw=2)
    # ax2.axhline(0, lw=1)
    # ax2.set_title("ΔNII as % of starting NII")
    # ax2.set_xlabel("h (quarters)")
    # ax2.set_ylabel("%")

    # # (C) ΔEVE bar (absolute + optional % of CET1 with error bars)
    # def _errbars(p10, med, p90):
        # lo = max(med - p10, 0.0)
        # hi = max(p90 - med, 0.0)
        # return np.array([[lo], [hi]])  # (2,1) non-negative

    # ax3 = fig.add_subplot(gs[1, :])
    # labels = ["ΔEVE"]
    # vals_abs = [eve_obj["dEVE"]]
    # errs_abs = None

    # if eve_obj.get("bands") is not None:
        # b10 = eve_obj["bands"].get("p10", None)
        # b50 = eve_obj["bands"].get("median", None)
        # b90 = eve_obj["bands"].get("p90", None)
        # if None not in (b10, b50, b90):
            # errs_abs = _errbars(b10, b50, b90)
            # vals_abs = [b50]  # center bar at median

    # bars = ax3.bar(labels, vals_abs, alpha=0.9, yerr=errs_abs, capsize=6)
    # ax3.axhline(0, lw=1)
    # ax3.set_ylabel(f"{currency} (absolute)")
    # title_eve = f"ΔEVE at h={snap_h}"

    # # Right axis for % of CET1
    # if cet1 is not None and abs(cet1) > 1e-12:
        # ax4 = ax3.twinx()
        # pct_cet1 = 100.0 * vals_abs[0] / cet1
        # ax4.plot([0], [pct_cet1], marker="o", color="tab:orange")
        # ax4.set_ylabel("% of CET1")
        # if eve_obj.get("bands") is not None:
            # bp = eve_obj.get("bands_pct_of_CET1")
            # if bp is not None and all(k in bp for k in ("p10","median","p90")):
                # eb = _errbars(bp["p10"], bp["median"], bp["p90"])
                # ax4.errorbar([0], [bp["median"]], yerr=eb, fmt="none", capsize=6, color="tab:orange")
        # title_eve += "  (right axis: % of CET1)"
    # ax3.set_title(title_eve)

    # fig.suptitle(title, fontsize=13)
    # fig.text(0.5, -0.02,
        # "Notes: Curves in decimals; NII is quarterly. ΔNII bands are 10–90%. "
        # "EVE uses a duration approximation at the chosen horizon; error bars sanitized to be non-negative.",
        # ha="center", fontsize=9, style="italic"
    # )
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.12)
    # plt.show()