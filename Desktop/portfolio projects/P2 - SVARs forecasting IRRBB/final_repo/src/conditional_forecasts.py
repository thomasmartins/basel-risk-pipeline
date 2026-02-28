"""Conditional forecasts (yields|macro and macro|Basel)."""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

def make_conditional_forecast(
    accepted_all: dict,
    panel: pd.DataFrame,
    var_order: list,
    scenario: str,
    *,
    p: int = None,
    h: int = 12,
    shock_size: float = 1.0,
    shock_horizon: int = 1,
    n_paths: int = 200,
    seed: int = 123,
    baseline: bool = False,
):
    """
    Run a very simple conditional SVAR-style forecast for ONE scenario
    and return a tidy dataframe with fan bands.

    Parameters
    ----------
    accepted_all : dict
        e.g. accepted_macro or accepted_basel_tight where
        accepted_all[scenario] has keys "B", "c", and either "IRFs" or "C".
    panel : DataFrame
        Your historical data (last p rows used as VAR state).
    var_order : list[str]
        VAR ordering, must match the B matrices.
    scenario : str
        Key in accepted_all, e.g. "monetary_tightening" or "parallel_up".
    p : int, optional
        VAR lag order. If None, inferred from B’s shape.
    h : int
        Forecast horizon (steps).
    shock_size : float
        Size of the *structural* shock you want to apply at t=0.
        If baseline=True, this is ignored (set to 0).
    shock_horizon : int
        For how many steps to apply the shock (usually 1).
    n_paths : int
        How many accepted draws to use (max).
    seed : int
        RNG seed.
    baseline : bool
        If True, run the same forecast but with NO shock (shock_size=0).

    Returns
    -------
    fan_df : DataFrame
        columns: ["date","var","h","yhat","yhat_lo","yhat_hi","scenario"]
    paths  : np.ndarray
        shape (n_paths, h, K) with the individual simulated paths
    """
    rng = np.random.default_rng(seed)

    # 1. pick the scenario params
    if scenario not in accepted_all:
        raise KeyError(f"Scenario {scenario!r} not in accepted_all.")
    params = accepted_all[scenario]

    B_all = np.asarray(params["B"])          # (Nacc, K, Kp)
    c_all = np.asarray(params.get("c", 0.0)) # (Nacc, K) or (K,)
    Nacc, K, Kp = B_all.shape

    # infer p
    impl_p = Kp // K
    if p is None:
        p = impl_p
    elif p != impl_p:
        raise ValueError(f"Provided p={p} but B implies p={impl_p}")

    # 2. history buffer (K x p), with most recent in column 0
    hist_mat = panel[var_order].dropna().to_numpy()
    if hist_mat.shape[0] < p:
        raise ValueError("Not enough history to form p lags.")
    hist_buf = hist_mat[-p:, :][::-1].T  # (K,p)

    # 3. choose which draws to simulate
    draw_idx = rng.choice(Nacc, size=min(n_paths, Nacc), replace=False)

    # 4. helper to get impact matrix for a given draw
    def get_impact(d):
        if "IRFs" in params:
            irf = np.asarray(params["IRFs"][d])
            # most common case in your script: (H+1, K, K)
            if irf.ndim == 3 and irf.shape[1] == K and irf.shape[2] == K:
                return irf[0, :, :]  # impact at h=0
        if "C" in params:
            return np.asarray(params["C"][d])
        raise KeyError("No impact matrix ('IRFs' or 'C') found in accepted scenario.")

    # 5. simulate all selected draws
    paths = np.zeros((draw_idx.size, h, K))
    # dates: quarterly forward from last obs
    last_date = panel.index[-1]
    dates = pd.date_range(last_date, periods=h+1, freq="QE")[1:]

    for nd, di in enumerate(draw_idx):
        B = B_all[di]
        c = c_all[di] if c_all.ndim == 2 else (c_all if c_all.ndim == 1 else np.zeros(K))
        As = [B[:, j*K:(j+1)*K] for j in range(p)]
        Cimp = get_impact(di)

        buf = hist_buf.copy()
        for t in range(h):
            yhat = c.copy()
            for j, Aj in enumerate(As):
                yhat += Aj @ buf[:, j]
            # impose structural shock at t < shock_horizon
            if (not baseline) and (t < shock_horizon):
                eps = np.zeros(K)
                eps[0] = shock_size   # assume identified shock is column 0
                yhat = yhat + Cimp @ eps
            paths[nd, t, :] = yhat
            # roll history
            buf = np.column_stack([yhat, buf[:, :-1]])

    # 6. aggregate to fan (10/50/90)
    med = np.median(paths, axis=0)
    lo  = np.percentile(paths, 10, axis=0)
    hi  = np.percentile(paths, 90, axis=0)

    rows = []
    for k, name in enumerate(var_order):
        rows.append(pd.DataFrame({
            "date": dates,
            "var": name,
            "h": np.arange(1, h+1),
            "yhat": med[:, k],
            "yhat_lo": lo[:, k],
            "yhat_hi": hi[:, k],
            "scenario": scenario if not baseline else scenario + "_baseline"
        }))
    fan_df = pd.concat(rows, ignore_index=True)
    return fan_df, paths
    
def plot_conditional_forecast(
    fan_df: pd.DataFrame,
    vars_to_plot=None,
    title_prefix="Scenario"
):
    """
    Very simple visualizer for the tidy fan df produced by make_conditional_forecast.
    Plots one figure per variable.

    Parameters
    ----------
    fan_df : DataFrame
        must have columns ["date","var","yhat","yhat_lo","yhat_hi","scenario"]
    vars_to_plot : list[str] or None
        if None, plot all variables in this fan_df
    title_prefix : str
        text to prepend to plot titles
    """
    scen_names = fan_df["scenario"].unique()
    if len(scen_names) == 1:
        scen = scen_names[0]
    else:
        # if user passed multiple scenarios at once, we just show the first in the title
        scen = ", ".join(scen_names)

    if vars_to_plot is None:
        vars_to_plot = fan_df["var"].unique()

    for v in vars_to_plot:
        sub = fan_df[fan_df["var"] == v].sort_values("date")
        plt.figure(figsize=(7,4))
        plt.fill_between(sub["date"], sub["yhat_lo"], sub["yhat_hi"], alpha=0.25, label="10–90%")
        plt.plot(sub["date"], sub["yhat"], lw=2, label="median")
        plt.axhline(0, color="k", lw=0.7)
        plt.title(f"{title_prefix}: {scen} – {v}")
        plt.xlabel("date")
        plt.ylabel(v)
        plt.legend()
        plt.tight_layout()
        plt.show()
