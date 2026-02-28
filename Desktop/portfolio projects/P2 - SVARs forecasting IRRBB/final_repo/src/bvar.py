"""Bayesian reduced-form VAR (PyMC/ArviZ)."""
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az


# 1) Load data (you already created this file)
df = pd.read_csv("data/quarterly_panel_modelvars.csv", parse_dates=["date"], index_col="date")

vars_used = [
    "infl_q_ann", "gdp_q_ann", "policy_rate", "gg_deficit_pct_gdp", "gg_debt_pct_gdp",
    "level", "slope_10y_1y", "curvature_ns_like"
]
Ydf = df[vars_used].dropna()
Y = Ydf.to_numpy()
T, K = Y.shape

# 2) Build lag matrices
def build_var_design(Y, p):
    T, K = Y.shape
    Xlags = [Y[p - i : T - i] for i in range(1, p + 1)]
    X = np.hstack(Xlags)              # (T-p, K*p)
    Yt = Y[p:]                        # (T-p, K)
    return X, Yt

p = 2
X, Yt = build_var_design(Y, p)        # X=(T_eff, K*p), Yt=(T_eff, K)
T_eff = Yt.shape[0]

# 3) Minnesota-style prior scales (simple, effective defaults)
def minnesota_scales(K, p, sigma=None, lam1=0.2, lam2=0.5, lam3=1.0):
    """
    lam1: overall tightness; lam2: cross-variable penalty; lam3: lag decay.
    sigma: per-equation scale (use sd of Y as fallback).
    """
    if sigma is None:
        sigma = Ydf.std().values
    S = np.zeros((K, K * p))
    for i in range(K):               # equation i
        for L in range(1, p + 1):    # lag L
            for j in range(K):       # predictor j
                col = (L - 1) * K + j
                cross = lam2 if i != j else 1.0
                S[i, col] = (sigma[i] / (sigma[j] + 1e-12)) * (lam1 * cross / (L ** lam3))
    return S

S_beta = minnesota_scales(K, p, sigma=Ydf.std().values, lam1=0.20, lam2=0.5, lam3=1.0)

# 4) PyMC model (LKJ Cholesky prior on Σu; gallery pattern)

with pm.Model() as bvar_model:
    # Residual covariance Σu with LKJ prior
    sd_dist = pm.HalfNormal.dist(1.0, shape=K)
    chol, corr, sds = pm.LKJCholeskyCov(
        "chol", n=K, eta=2.0, sd_dist=sd_dist, compute_corr=True, store_in_trace=True
    )
    Sigma = pm.Deterministic("Sigma", chol @ chol.T)

    # Coefficients: B is K x (K*p); intercept c is K
    B = pm.Normal("B", mu=0.0, sigma=S_beta, shape=(K, K * p))
    c = pm.Normal("c", mu=0.0, sigma=5.0, shape=K)
    
    # COVID dummy
    Z = np.zeros((T-p, 1), dtype=float)
    Z[[63-p, 64-p], 0] = 1.0
    q = Z.shape[1]                 # number of dummy columns (1 or 2)
    Z_shared = pm.Data("Z", Z)     # register as data for PyMC
    # Coefficients for exogenous block: Γ has shape (K x q), one coefficient per VAR equation per dummy
    tau_D = pm.HalfNormal("tau_D", sigma=0.5)   # global shrinkage
    Gamma = pm.Normal("Gamma", mu=0.0, sigma=tau_D, shape=(K, q))


    # Mean and likelihood (vectorized over T_eff rows)
    mu = c + pt.dot(X, B.T) + pt.dot(Z_shared, Gamma.T) # (T_eff, K)
    pm.MvNormal("y", mu=mu, chol=chol, observed=Yt)     # use chol for stability
pm.model_to_graphviz(bvar_model)

# Sampler (BlackJAX NUTS if installed; fallback to pm.sample)
# from pymc.sampling_jax import sample_blackjax_nuts
#with bvar_model:
#    try:
#        idata = sample_blackjax_nuts(draws=1500, tune=1500, chains=4, target_accept=0.9, random_seed=123)
#    except Exception:
#        idata = pm.sample(2000, nuts_sampler="numpyro", init="jitter+adapt_diag", tune=1000, chains=4, target_accept=0.9, adapt_step_size=True, random_seed=123, idata_kwargs={"log_likelihood": True})

idata = az.from_netcdf("results/bvar_results.nc")

pm.summary(idata)

# Keep helpful attrs
idata.attrs["vars_used"] = vars_used
idata.attrs["lags"] = p

# forecasting functions, I won't be using those in the main notebook

def forecast_dates_from_index(last_date, h):
    # assumes a DatetimeIndex at quarter-end
    q = pd.Period(last_date, freq="Q")
    periods = [ (q + i).end_time.normalize() for i in range(1, h+1) ]
    return pd.to_datetime(periods)

def forecast_bvar(idata, Yhist_df, p=2, h=8, draws=500, model_name="BVAR", return_draws=False, seed=123):
    vars_used = idata.attrs["vars_used"]
    Yh = Yhist_df[vars_used].dropna().to_numpy()
    post = idata.posterior
    B_all = post["B"].stack(s=("chain","draw")).values    # (K, Kp, S)
    c_all = post["c"].stack(s=("chain","draw")).values    # (K, S)
    S = B_all.shape[-1]
    rng = np.random.default_rng(seed)
    sel = rng.choice(S, size=min(draws, S), replace=False)

    K = len(vars_used); Kp = B_all.shape[1]; assert Kp % K == 0 and (Kp//K)==p
    hist = Yh[-p:, :].copy()  # (p, K)

    paths = np.zeros((len(sel), h, K))
    for d, sidx in enumerate(sel):
        B = B_all[:, :, sidx]; c = c_all[:, sidx]
        A = [B[:, j*K:(j+1)*K] for j in range(p)]
        hist_buf = hist[::-1].T.copy()
        for t in range(h):
            yhat = c.copy()
            for j, Aj in enumerate(A):
                yhat += Aj @ hist_buf[:, j]
            paths[d, t, :] = yhat
            hist_buf = np.column_stack([yhat, hist_buf[:, :-1]])

    mean = paths.mean(axis=0)  # (h, K)
    dates = forecast_dates_from_index(Yhist_df.index[-1], h)
    out_rows = []
    if return_draws:
        # long form with draws (for fan charts)
        for d in range(paths.shape[0]):
            for k, name in enumerate(vars_used):
                out_rows.append(pd.DataFrame({
                    "date": dates, "var": name, "h": np.arange(1, h+1),
                    "yhat": paths[d, :, k], "model": model_name, "draw": d
                }))
        return pd.concat(out_rows, ignore_index=True)
    else:
        for k, name in enumerate(vars_used):
            out_rows.append(pd.DataFrame({
                "date": dates, "var": name, "h": np.arange(1, h+1),
                "yhat": mean[:, k], "model": model_name
            }))
        return pd.concat(out_rows, ignore_index=True)

def backtest_bvar(fit_fn, df, vars_used, p=2, horizons=(1,4,8), split=0.6,
                  draws=400, tune=400, seed=42, model_name="BVAR"):
    """
    fit_fn: callable that fits a BVAR on a DataFrame (returns idata with attrs['vars_used'])
            signature like: fit_fn(Ytrain_df, p, draws, tune, seed) -> idata
    """
    out = []
    T = len(df); start = int(split*T)
    idx = df.index
    for t in range(start, T - max(horizons) + 1):
        train = df.iloc[:t][vars_used].dropna()
        idata = fit_fn(train, p=p, draws=draws, tune=tune, seed=seed)
        fc = forecast_bvar(idata, train, p=p, h=max(horizons), draws=draws, model_name=model_name)
        fc = fc[fc["h"].isin(horizons)]
        # attach actuals
        act = (df.iloc[:t+max(horizons)][vars_used]
                 .reset_index().melt(id_vars=["date"], var_name="var", value_name="y"))
        merged = fc.merge(act, on=["date","var"], how="left")
        out.append(merged)
    return pd.concat(out, ignore_index=True)

bvar_fc = forecast_bvar(idata, Ydf, p=p, h=8, draws=800)
bvar_fc
