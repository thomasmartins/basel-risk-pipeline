"""SVAR identification and sign restrictions for macro shocks."""
import os
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import json

# -------- Utilities --------

def extract_B_Sigma(idata, var_B="B", var_chol="chol", var_Sigma="Sigma"):
    """
    Return arrays:
      B_draws   : (S, K, Kp)  stacked lag blocks per draw (A1|A2|...|Ap)
      Sigma_draws : (S, K, K)
    Works whether you stored 'Sigma' or 'chol' in the trace.
    """
    post = idata.posterior

    # B
    B = post[var_B].stack(s=("chain", "draw")).values  # (K, Kp, S) or (S, K, Kp) depending on dims
    if B.shape[0] != B.shape[-1]:  # typical case (K,Kp,S)
        B = np.moveaxis(B, -1, 0)  # -> (S,K,Kp)

    # Sigma (prefer direct if present; else from chol)
    if var_Sigma in post:
        Sg = post[var_Sigma].stack(s=("chain", "draw")).values
        if Sg.shape[0] != Sg.shape[-1]:
            Sg = np.moveaxis(Sg, -1, 0)  # -> (S,K,K)
        Sigma = Sg
    else:
        chol = post[var_chol].stack(s=("chain", "draw")).values  # (K,K,S) or (S,K,K)
        if chol.shape[0] != chol.shape[-1]:
            chol = np.moveaxis(chol, -1, 0)  # -> (S,K,K)
        Sigma = np.einsum("sij,skj->sik", chol, chol)  # chol @ chol.T per draw

    return B, Sigma

def unpack_B(B_draw, K, p):
    """Return list [A1,...,Ap] with each Ai (K×K) from stacked B_draw (K×Kp)."""
    return [B_draw[:, j*K:(j+1)*K] for j in range(p)]

def companion_from_As(As):
    """Build VAR companion matrix A_comp from [A1..Ap]."""
    K = As[0].shape[0]; p = len(As)
    A = np.zeros((K*p, K*p))
    for j, Aj in enumerate(As):
        A[:K, j*K:(j+1)*K] = Aj
    if p > 1:
        A[K:, :-K] = np.eye(K*(p-1))
    return A

def random_orthonormal(k, rng=None):
    """Uniform Haar draw: Q from QR of Gaussian matrix with det(Q)=+1."""
    rng = np.random.default_rng() if rng is None else rng
    Z = rng.normal(size=(k, k))
    Q, R = np.linalg.qr(Z)
    d = np.sign(np.diag(R))
    Q = Q @ np.diag(d)
    return Q

def irfs_from_BC(B_draw, C, H):
    """
    IRFs Φ_h for h=0..H:
      B_draw : (K, Kp) stacked [A1|..|Ap]
      C      : (K, K) impact matrix (u_t = C ε_t, ε ~ N(0,I))
    Returns Φ of shape (H+1, K, K): response of variables (axis1) to shocks (axis2).
    """
    K, Kp = B_draw.shape
    p = Kp // K
    As = unpack_B(B_draw, K, p)
    Acomp = companion_from_As(As)
    S = np.zeros((K, K*p)); S[:, :K] = np.eye(K)

    Phi = np.zeros((H+1, K, K))
    Phi[0] = C.copy()
    Ak = np.eye(K*p)
    for h in range(1, H+1):
        Ak = Acomp @ Ak
        Phi[h] = (S @ Ak @ S.T) @ C
    return Phi

def is_stable(B_draw, tol=1 - 1e-10):
    """All eigenvalues of the companion inside the unit circle."""
    K, Kp = B_draw.shape
    p = Kp // K
    As = unpack_B(B_draw, K, p)
    Acomp = companion_from_As(As)
    eigvals = np.linalg.eigvals(Acomp)
    return np.all(np.abs(eigvals) < tol)

# -------- Step 1: candidate rotations & IRFs --------

def make_candidates_from_idata(
    idata,
    p: int,
    H: int = 20,
    max_draws: int = 1000,
    rotations_per_draw: int = 5,
    require_stable: bool = True,
    seed: int = 123,
):
    """
    From reduced-form posterior (B, Sigma, c), generate candidate impact matrices:
      - take up to max_draws reduced-form draws,
      - for each, try 'rotations_per_draw' random Q to build C = P Q,
      - compute IRFs up to horizon H,
      - (optionally) keep only stable VAR draws.

    Returns dict with arrays:
      {'B': (N,K,Kp), 'Sigma': (N,K,K), 'C': (N,K,K), 'IRFs': (N,H+1,K,K), 'c': (N,K)}
    """
    rng = np.random.default_rng(seed)

    # You already have this:
    B_draws, Sigma_draws = extract_B_Sigma(idata)       # (S_total,K,Kp), (S_total,K,K)
    S_total, K, Kp = B_draws.shape
    p_infer = Kp // K
    assert p_infer == p, f"Posterior has p={p_infer}, but you passed p={p}"

    # NEW: pull intercept draws ('c') from idata (adjust this to your naming)
    # Expect shape (S_total, K) after flattening chains/draws.
    c_draws = extract_param(idata, key_candidates=("c", "const", "intercept", "mu"))
    if c_draws is None:
        # If your model truly has no constant (demeaned VAR), keep zeros:
        c_draws = np.zeros((S_total, K))

    sel = np.arange(S_total)
    if max_draws < S_total:
        sel = rng.choice(S_total, size=max_draws, replace=False)

    out_B, out_S, out_C, out_IRF, out_c = [], [], [], [], []
    for s in sel:
        B = B_draws[s]
        if require_stable and not is_stable(B):
            continue
        Sigma = Sigma_draws[s]

        # Lower-triangular Cholesky
        try:
            P = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            continue

        for _ in range(rotations_per_draw):
            Q = random_orthonormal(K, rng)
            C = P @ Q
            Phi = irfs_from_BC(B, C, H)      # (H+1,K,K)

            out_B.append(B)
            out_S.append(Sigma)
            out_C.append(C)
            out_IRF.append(Phi)
            out_c.append(c_draws[s])         # <-- replicate intercept for each rotation

    if not out_B:
        raise RuntimeError("No candidates produced. Relax stability or increase draws/rotations.")

    return {
        "B":    np.array(out_B),     # (N, K, Kp)
        "Sigma":np.array(out_S),     # (N, K, K)
        "C":    np.array(out_C),     # (N, K, K) impact
        "IRFs": np.array(out_IRF),   # (N, H+1, K, K)
        "c":    np.array(out_c),     # (N, K)    intercept
    }


# Helper: try a few names and flatten chains/draws from idata.posterior
def extract_param(idata, key_candidates=("c",), required_dim_last=None):
    """
    Returns np.ndarray of shape (S_total, K) if found, else None.
    Adjust to your InferenceData structure if needed.
    """
    try:
        import xarray as xr  # ArviZ/xarray usually available with PyMC
    except Exception:
        xr = None

    post = getattr(idata, "posterior", None)
    if post is None:
        return None

    for key in key_candidates:
        if hasattr(post, key):
            arr = getattr(post, key)
        elif key in getattr(post, "data_vars", {}):
            arr = post[key]
        else:
            continue

        # Typical shapes: (chain, draw, K) or (chain, draw, K, 1)
        # Collapse (chain, draw) into one 'sample' axis:
        try:
            stacked = arr.stack(sample=("chain", "draw")).transpose("sample", ...).values
        except Exception:
            # Fallback: just convert to numpy and reshape best-effort
            a = np.asarray(arr)
            if a.ndim >= 2:
                a = a.reshape(a.shape[0]*a.shape[1], *a.shape[2:])
            else:
                return None
            stacked = a

        # Reduce to (S_total, K)
        if stacked.ndim == 2:            # (S_total, K)
            return stacked
        if stacked.ndim == 3 and stacked.shape[-1] == 1:
            return stacked[..., 0]
        if stacked.ndim > 2:
            # Try to squeeze singleton dims and keep last dimension as K
            squeezed = np.squeeze(stacked)
            if squeezed.ndim == 2:
                return squeezed
        # If not compatible, try the next key
    return None


# idata is the PyMC result from your BVAR fit; p is the VAR lag length used
idata = az.from_netcdf("results/bvar_results.nc")
cands = make_candidates_from_idata(
    idata=idata,
    p=idata.attrs.get("lags", 2),
    H=20,
    max_draws=2000,           # cap for speed; increase later
    rotations_per_draw=15,    # 5–20 is typical; increase if acceptance is low later
    require_stable=True,
    seed=123
)

# print("Candidates generated:", len(cands["C"]))
# print("Shapes B, Sigma, C, IRFs:", cands["B"].shape, cands["Sigma"].shape, cands["C"].shape, cands["IRFs"].shape)



# ---------- core helpers ----------

def check_irf_signs(irfs: np.ndarray,
                    spec: Dict[int, List[Tuple[int,int,int]]],
                    tol_zero: float = 1e-8) -> bool:
    """
    irfs: (H+1, K, K) responses of variables (axis=1) to shocks (axis=2)
    spec: {(var_idx): [(h0,h1, sign), ...], ...}, sign ∈ {-1,0,+1}
    """
    H = irfs.shape[0] - 1
    for i_var, windows in spec.items():
        for (h0, h1, sgn) in windows:
            a, b = max(h0, 0), min(h1, H)
            path = irfs[a:b+1, i_var, :]
            if sgn == +1:
                if np.any(path < 0): return False
            elif sgn == -1:
                if np.any(path > 0): return False
            elif sgn == 0:
                if np.any(np.abs(path) > tol_zero): return False
            else:
                raise ValueError("sign must be -1, 0, or +1")
    return True

def find_matching_column(irfs: np.ndarray,
                         spec_one_shock: Dict[int, List[Tuple[int,int,int]]],
                         tol_zero: float = 1e-8):
    """
    Try all K columns; return j if any satisfies spec_one_shock, else None.
    spec_one_shock: {var_idx: [(h0,h1,sign), ...]}
    """
    H, K, _ = irfs.shape[0]-1, irfs.shape[1], irfs.shape[2]
    for j in range(K):
        ok = True
        for i_var, windows in spec_one_shock.items():
            for (h0,h1,sgn) in windows:
                a, b = max(h0,0), min(h1,H)
                path = irfs[a:b+1, i_var, j]
                if sgn == +1 and np.any(path < 0): ok=False; break
                if sgn == -1 and np.any(path > 0): ok=False; break
                if sgn == 0  and np.any(np.abs(path) > tol_zero): ok=False; break
            if not ok: break
        if ok: return j
    return None

def reorder_columns(C: np.ndarray, irfs: np.ndarray, j_first: int):
    """Make column j_first the first column (shock 0)."""
    K = C.shape[1]
    perm = [j_first] + [c for c in range(K) if c != j_first]
    return C[:, perm], irfs[:, :, perm], perm


### OLD:

# def build_sign_specs(var2idx: Dict[str,int]):
#     # indices
#     i_pi   = var2idx["infl_q_ann"]
#     i_gdp  = var2idx["gdp_q_ann"]
#     i_def  = var2idx["gg_deficit_pct_gdp"]
#     i_debt = var2idx["gg_debt_pct_gdp"]
#     i_pol  = var2idx["policy_rate"]
#     i_lev  = var2idx["level"]
#     i_slp  = var2idx["slope_10y_1y"]
#     i_curv = var2idx["curvature_ns_like"]

#     # Fiscal expansion (FTPL-friendly)
#     spec_fiscal = {
#         i_def:  [(0,2,-1)],     # deficit higher for a couple of quarters (adjust sign if your series is negative)
#         i_lev:  [(0,0,+1)],     # long-end/level up on impact
#         i_pi:   [(1,3,+1)],     # inflation rises within a year (no impact requirement)
#         i_gdp:  [(1,3,+1)],     # output supports demand with a lag
#         i_debt: [(3,8,+1)],     # debt rises medium-run
#     # slope/curvature/policy left free
#     }

#     # Monetary tightening (short-rate ↑; curve flattens; disinflation and slower growth)
#     spec_monet = {
#         i_pol: [(0,0,+1)],     # policy rate jumps on impact
#         i_slp: [(0,0,-1)],     # curve flattens immediately (10y−1y falls)
#         i_pi : [(1,4,-1)],     # disinflation starts within a year (allow impact noise)
#         i_gdp: [(1,4,-1)],     # activity weakens with a lag
#     # i_lev, i_def, i_debt, i_curv left free (ambiguous short-run signs)
#     }
#     return spec_fiscal, spec_monet


# 8-variable order
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

def build_sign_specs_8var():
    """
    Returns (spec_fiscal, spec_monet) for use with your sign-restriction code.
    Each spec is a dict: { var_index : [(h0,h1, sign), ...] }, with sign in {-1,0,+1}.
    Horizons are in quarters, inclusive (e.g., (0,4) = impact through 1 year).
    """

    # -------- Fiscal expansion shock --------
    # Narrative: primary deficit rises; long rates/level up on impact; curve at least not flatter on impact;
    # inflation rises over the first year; GDP turns positive with a short lag;
    # debt ratio rises with a lag (stock-flow); policy may react mildly (optional + after 1q).
    spec_fiscal = {
        i_def:  [(0,2,-1)],     # deficit + on impact and near term. deficits are negative and surpluses are positive, so deficit increase has negative sign
        i_lev:  [(0,0,+1)],     # level + at impact
        #i_slp:  [(0,2,+1)],     # steepening (≥ 0) at impact
        i_pi:   [(1,4,+1)],     # inflation ≥ 0 in year 1
        i_gdp:  [(1,4,+1)],     # GDP ≥ 0 with a lag
        i_debt: [(2,6,+1)],     # debt ratio rises over 0.5–2 years
        # i_pol:  [(1,4,+1)],   # OPTIONAL: monetary reaction (tightening) after 1q
        # curvature free
    }

    # -------- Monetary tightening shock --------
    # Narrative: policy rate jumps up; curve flattens on impact; inflation disinflates over a year;
    # output falls with a lag; level typically up on impact (short end drags average up).
    # Deficit/debt left free: short-run sign is ambiguous (valuation vs cycle).
    spec_monet = {
        i_pol:  [(0,0,+1)],     # policy ↑ on impact
        i_slp:  [(0,2,-1)],     # flattening at impact
        i_pi:   [(1,4,-1)],     # disinflation within year 1
        i_gdp:  [(1,4,-1)],     # output ↓ with a lag
        # i_lev:  [(0,1,+1)],     # average rates up on impact (mild)
        # deficit, debt, curvature free
    }

    return spec_fiscal, spec_monet

def apply_sign_restrictions(
    cands: dict,
    spec_one_shock: Dict[int, List[Tuple[int,int,int]]],
    shock_name: str,
    max_accept: int = 400,
    tol_zero: float = 1e-8
):
    """
    Returns accepted dict with arrays and metadata:
      {
        'B','Sigma','C','IRFs','perm','shock_name',
        'accepted','tried','accept_rate','sel_idx',
        'c' (optional)
      }
    """
    B_all, S_all, C_all, IRF_all = cands["B"], cands["Sigma"], cands["C"], cands["IRFs"]
    N, H1, K, _ = IRF_all.shape

    acc_B, acc_S, acc_C, acc_IRF, perms = [], [], [], [], []
    sel_idx = []  # record which candidate indices were accepted
    tried = 0

    for n in range(N):
        tried += 1
        irf = IRF_all[n]  # (H+1, K, K)
        j = find_matching_column(irf, spec_one_shock, tol_zero=tol_zero)
        if j is None:
            continue
        C_re, irf_re, perm = reorder_columns(C_all[n], irf, j_first=j)
        acc_B.append(B_all[n]); acc_S.append(S_all[n]); acc_C.append(C_re); acc_IRF.append(irf_re); perms.append(perm)
        sel_idx.append(n)
        if len(acc_B) >= max_accept:
            break

    if not acc_B:
        out = {
            "B": np.empty((0,)), "Sigma": np.empty((0,)), "C": np.empty((0,)),
            "IRFs": np.empty((0,)), "perm": [], "shock_name": shock_name,
            "accepted": 0, "tried": tried, "accept_rate": 0.0, "sel_idx": np.array([], dtype=int)
        }
        # Attach empty 'c' only if provided
        if "c" in cands:
            out["c"] = np.empty((0, K)) if np.asarray(cands["c"]).ndim >= 1 else np.empty((0,))
        return out

    out = {
        "B": np.array(acc_B),
        "Sigma": np.array(acc_S),
        "C": np.array(acc_C),
        "IRFs": np.array(acc_IRF),
        "perm": perms,
        "shock_name": shock_name,
        "accepted": len(acc_B),
        "tried": tried,
        "accept_rate": len(acc_B) / max(1, tried),
        "sel_idx": np.array(sel_idx, dtype=int),
    }

    # ---- attach intercept *after* selection so it cannot influence acceptance ----
    if "c" in cands:
        c_all = np.asarray(cands["c"])
        if c_all.ndim == 1 and c_all.shape[0] == K:
            # same intercept for all candidates → replicate for each accepted draw
            out["c"] = np.repeat(c_all[np.newaxis, :], len(sel_idx), axis=0)
        elif c_all.ndim == 2 and c_all.shape[0] == B_all.shape[0] and c_all.shape[1] == K:
            # per-candidate intercepts → slice by sel_idx
            out["c"] = c_all[out["sel_idx"]]
        else:
            raise ValueError(f"Unexpected shape for cands['c']: {c_all.shape}; expected (K,) or (N,K) with N={B_all.shape[0]}, K={K}")

    return out


# Map your variable names to indices, in the order used in the BVAR
var_order = [
    "infl_q_ann", "gdp_q_ann", "policy_rate", "gg_deficit_pct_gdp", "gg_debt_pct_gdp",
    "level", "slope_10y_1y", "curvature_ns_like"
]
var2idx = {v:i for i,v in enumerate(var_order)}

# spec_fiscal, spec_monet = build_sign_specs(var2idx)
spec_fiscal, spec_monet = build_sign_specs_8var()

accepted_fiscal = apply_sign_restrictions(
    cands, spec_one_shock=spec_fiscal, shock_name="fiscal_expansion",
    max_accept=800, tol_zero=1e-8
)
#print("Fiscal: accepted", accepted_fiscal["accepted"], "of", accepted_fiscal["tried"],
#      f"(rate={accepted_fiscal['accept_rate']:.2%})")

accepted_monet = apply_sign_restrictions(
    cands, spec_one_shock=spec_monet, shock_name="monetary_tightening",
    max_accept=800, tol_zero=1e-8
)
#print("Monetary: accepted", accepted_monet["accepted"], "of", accepted_monet["tried"],
#      f"(rate={accepted_monet['accept_rate']:.2%})")


accepted_macro = {'fiscal_expansion' : accepted_fiscal, "monetary_tightening" : accepted_monet}

def diagnose_bottlenecks(cands, spec, tol_zero=1e-4, mode="mean", sample=400, var_order=None):
    IRF = cands["IRFs"]
    N, H1, K, _ = IRF.shape
    N = min(N, sample)
    fails = {(i, span): 0 for i, spans in spec.items() for span in spans}

    def ok(seg, sgn):
        if sgn == +1: return seg.mean() >= -tol_zero if mode=="mean" else (seg >= -tol_zero).all()
        if sgn == -1: return seg.mean() <=  tol_zero if mode=="mean" else (seg <=  tol_zero).all()
        if sgn ==  0: return (abs(seg).mean() <= tol_zero) if mode=="mean" else (abs(seg) <= tol_zero).all()
        return False

    for n in range(N):
        cube = IRF[n]
        for i, spans in spec.items():
            for (h0,h1,sgn) in spans:
                good_any = False
                for j in range(K):  # best structural column qualifies the draw
                    seg = cube[h0:h1+1, i, j]
                    if ok(seg, sgn):
                        good_any = True
                        break
                if not good_any:
                    fails[(i,(h0,h1,sgn))] += 1

    rows = []
    for (i, span), cnt in sorted(fails.items(), key=lambda x: -x[1]):
        name = var_order[i] if var_order else i
        rows.append((name, span, cnt))
    return rows

# Example:
#print("Fiscal bottlenecks:")
#for name, span, cnt in diagnose_bottlenecks(cands, spec_fiscal, var_order=var_order)[:6]:
#    print(f"{name:>20} {span} -> {cnt} fails")

#print("\nMonetary bottlenecks:")
#for name, span, cnt in diagnose_bottlenecks(cands, spec_monet, var_order=var_order)[:6]:
#    print(f"{name:>20} {span} -> {cnt} fails")


# def summarize_irfs(acc: dict, H: int = 20):
#     """
#     acc['IRFs']: (Nacc, H+1, K, K), column 0 is the identified shock by construction.
#     Returns median, p10, p90 arrays of shape (H+1, K).
#     """
#     if acc["accepted"] == 0:
#         return None
#     irfs = acc["IRFs"][:, :H+1, :, 0]  # take the identified shock (col 0)
#     med = np.median(irfs, axis=0)
#     lo  = np.percentile(irfs, 10, axis=0)
#     hi  = np.percentile(irfs, 90, axis=0)
#     return med, lo, hi

# f_med, f_lo, f_hi = summarize_irfs(accepted_fiscal, H=20)
# m_med, m_lo, m_hi = summarize_irfs(accepted_monet, H=20)

def summarize_irfs(acc, var_order, H=None, q_lo=10, q_hi=90, alpha_scaling=None):
    """
    acc: dict from apply_sign_restrictions(...)
         expects acc["IRFs"] with shape (Nacc, H+1, K, K)
         where column 0 is the identified shock (we reordered earlier).
         or you can pass the IRFs object as an array
    Returns:
      med, lo, hi: arrays (H+1, K)
      tidy_df: long DataFrame with columns [shock, var, h, med, p10, p90]
    """
    if acc["accepted"] == 0:
        raise ValueError(f"No accepted draws for shock={acc.get('shock_name', '')}")

    if alpha_scaling is None:
        IRFs = acc["IRFs"]                    # (N, H+1, K, K)
    else:
        IRFs = acc["IRFs"] * alpha_scaling[:, None, None, None]           
    if H is None:
        H = IRFs.shape[1] - 1
                  

    # IRFs to the identified shock (col 0 after reordering)
    irf_draws = IRFs[:, :H+1, :, 0]                     # (N, H+1, K)

    med = np.median(irf_draws, axis=0)                  # (H+1, K)
    lo  = np.percentile(irf_draws, q_lo, axis=0)
    hi  = np.percentile(irf_draws, q_hi, axis=0)

    # tidy
    rows = []
    for k, name in enumerate(var_order):
        rows.append(pd.DataFrame({
            "shock": acc.get("shock_name","shock"),
            "var": name,
            "h": np.arange(H+1),
            "med": med[:, k],
            f"p{q_lo}": lo[:, k],
            f"p{q_hi}": hi[:, k],
        }))
    tidy_df = pd.concat(rows, ignore_index=True)
    return med, lo, hi, tidy_df

# Plotting macroeconomic shock IRFs: scaling to 1 SD

# Robust σ (optional) so pandemic outliers don’t dominate
def robust_sigma(x, winsor=(5,95)):
    x = pd.Series(x).dropna().to_numpy()
    if winsor is not None:
        lo, hi = np.percentile(x, winsor)
        x = np.clip(x, lo, hi)
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826 * mad  # ≈ SD for Gaussian data

def unit_impacts(accepted, var_order, var_name):
    """Impact (h=0) of a unit structural shock on `var_name` (pp), shape (Nacc,)."""
    IRFs = np.asarray(accepted["IRFs"])           # (Nacc, H+1, K, K)
    i = var_order.index(var_name)
    return IRFs[:, 0, i, 0]

def alphas_one_sigma(accepted, panel, var_order, anchor_var,
                     sigma=None, robust=True, winsor=(5,95), eps=1e-12):
    """
    Scale so the impact on `anchor_var` equals its σ (pp).
    If sigma=None, compute (robust) σ from `panel[anchor_var]`.
    """
    imp_abs = np.abs(unit_impacts(accepted, var_order, anchor_var))  # (Nacc,)
    if sigma is None:
        if robust:
            sigma = robust_sigma(panel[anchor_var], winsor=winsor)
        else:
            sigma = float(pd.Series(panel[anchor_var]).std())
    return sigma / (imp_abs + eps), float(sigma)


panel = pd.read_csv("data/quarterly_panel_modelvars.csv", parse_dates=["date"], index_col="date")
# Monetary tightening, 1σ on policy_rate
accF = accepted_macro["fiscal_expansion"]
alphas_1sd_fisc, sig_def_f = alphas_one_sigma(accF, panel, var_order, "gg_deficit_pct_gdp")

# Fiscal expansion, 1σ on deficit
accM = accepted_macro["monetary_tightening"]
alphas_1sd_monet, sig_pol_m = alphas_one_sigma(accM, panel, var_order, "policy_rate")

var_order = [
    "infl_q_ann","gdp_q_ann","policy_rate","gg_deficit_pct_gdp","gg_debt_pct_gdp",
    "level","slope_10y_1y","curvature_ns_like"
]

f_med, f_lo, f_hi, df_irf_fiscal  = summarize_irfs(accepted_fiscal, var_order, H=20, alpha_scaling=alphas_1sd_fisc)
m_med, m_lo, m_hi, df_irf_monet   = summarize_irfs(accepted_monet, var_order, H=20, alpha_scaling=alphas_1sd_monet)

os.makedirs("results", exist_ok=True)

# Save for report/notebook plotting if you like
df_irf = pd.concat([df_irf_fiscal, df_irf_monet], ignore_index=True)
df_irf.to_csv("results/irf_summary_sign.csv", index=False)
np.save("results/accepted_irfs_fiscal.npy",  accepted_fiscal["IRFs"])
np.save("results/accepted_irfs_monetary.npy",accepted_monet["IRFs"])

def plot_irf_panel(med, lo, hi, var_order, shock_label="shock", H=None, fname=None):
    """
    med/lo/hi: arrays (H+1, K) from summarize_irfs
    """
    if H is None:
        H = med.shape[0] - 1
    K = med.shape[1]
    ncols = 3
    nrows = int(np.ceil(K / ncols))

    plt.figure(figsize=(12, 3.4*nrows))
    t = np.arange(H+1)
    for i, name in enumerate(var_order):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.plot(t, med[:, i], label="median")
        ax.fill_between(t, lo[:, i], hi[:, i], alpha=0.2)
        ax.axhline(0, linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("h (quarters)")
    plt.suptitle(f"IRFs to {shock_label} (median & {int(100*(hi.size/med.size))}% band not literal)")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
        plt.close()

# Plot both shocks
# plot_irf_panel(f_med, f_lo, f_hi, var_order, shock_label="Fiscal expansion",
#                fname="results/irf_panel_fiscal.png")
# plot_irf_panel(m_med, m_lo, m_hi, var_order, shock_label="Monetary tightening",
#                fname="results/irf_panel_monetary.png")

#plot_irf_panel(f_med, f_lo, f_hi, var_order, shock_label="Fiscal expansion")
#plt.show()

#plot_irf_panel(m_med, m_lo, m_hi, var_order, shock_label="Monetary tightening")
#plt.show()

def acceptance_report(acc):
    return {
        "shock": acc.get("shock_name",""),
        "accepted": acc.get("accepted",0),
        "tried": acc.get("tried",0),
        "accept_rate": acc.get("accept_rate",0.0)
    }

with open("results/acceptance_stats.json","w") as f:
    json.dump({
        "fiscal": acceptance_report(accepted_fiscal),
        "monetary": acceptance_report(accepted_monet)
    }, f, indent=2)


def plot_irfs_cloud(irf_draws, var_idx, var_name, shock_name, H=20, nplot=200):
    """
    irf_draws: array (N, H+1, K) of IRFs for a given shock
    var_idx: index of the variable to plot
    """
    N = min(nplot, irf_draws.shape[0])
    t = np.arange(H+1)
    plt.figure(figsize=(6,4))
    for i in range(N):
        plt.plot(t, irf_draws[i, :, var_idx], color="lightgrey", alpha=0.5)
    plt.axhline(0, linewidth=0.8)
    plt.title(f"Unfiltered IRFs: {var_name} to {shock_name}")
    plt.xlabel("h (quarters)")
    plt.ylabel("Response")
    plt.show()

#plot_irfs_cloud(cands["IRFs"][:, :, :, 3], 0, "inflation", "deficit", H=20, nplot=200)

def plot_irf_filtered(df_irf, shock, var, band=(10,90)):
    sub = df_irf[(df_irf["shock"]==shock) & (df_irf["var"]==var)]
    plt.figure(figsize=(6,4))
    plt.plot(sub.h, sub.med, label="median")
    plt.fill_between(sub.h, sub[f"p{band[0]}"], sub[f"p{band[1]}"], alpha=0.3)
    plt.axhline(0, linewidth=0.8)
    plt.title(f"Filtered IRF: {var} to {shock}")
    plt.ylim(-10,10)
    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    ax = plt.gca()  # get current axes
    ax.xaxis.set_major_locator(loc)
    plt.xlim(0,20)
    plt.xlabel("h (quarters)")
    plt.ylabel("Response")
    plt.legend()
    plt.show()

#plot_irf_filtered(df_irf[df_irf['shock'] == "fiscal_expansion"], "fiscal_expansion", "infl_q_ann", band=(10,90))

#plot_irf_filtered(df_irf[df_irf['shock'] == "fiscal_expansion"], "fiscal_expansion", "gdp_q_ann", band=(10,90))

def plot_irf_comparison(irf_unfiltered, df_filtered, shock, var, var_idx, H=20,
                        band=(10,90), nplot=200):
    """
    irf_unfiltered: array (N, H+1, K) with IRFs to the chosen shock (before filtering).
    df_filtered: tidy DataFrame from summarize_irfs()
    shock: str, shock label (e.g. "Fiscal expansion")
    var: str, variable name to plot
    var_idx: int, index of var in var_order
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    t = np.arange(H+1)

    # left: cloud (unfiltered)
    N = min(nplot, irf_unfiltered.shape[0])
    for i in range(N):
        axes[0].plot(t, irf_unfiltered[i, :H+1, var_idx], color="lightgrey", alpha=0.5)
    axes[0].axhline(0, linewidth=0.8, c='black')
    axes[0].set_title(f"Unfiltered IRFs: {var} to {shock}")
    axes[0].set_xlabel("h (quarters)")

    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
    axes[0].xaxis.set_major_locator(loc)
    axes[0].set_xlim(0,20)
    
    # right: filtered (median + bands)
    sub = df_filtered[(df_filtered["shock"]==shock) & (df_filtered["var"]==var)]
    axes[1].plot(sub.h, sub.med, label="median", color="C0")
    axes[1].fill_between(sub.h, sub[f"p{band[0]}"], sub[f"p{band[1]}"],
                         alpha=0.3, color="C0")
    axes[1].axhline(0, linewidth=0.8, c='black')
    axes[1].set_title(f"Filtered IRFs: {var} to {shock}")
    axes[1].set_xlabel("h (quarters)")
    axes[1].set_ylim(-3,3)
    axes[1].xaxis.set_major_locator(loc)
    axes[1].set_xlim(0,20)
    axes[1].tick_params(axis="y", which="both", labelleft=True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


#plot_irf_comparison(cands["IRFs"][:, :, :, 3], df_irf_fiscal, "fiscal_expansion",
#                    var="infl_q_ann", var_idx=0, H=20)
#plt.show()

def plot_hist_impact(acc, var_idx, var_name, shock_name):
    Phi0 = acc["IRFs"][:, 0, var_idx, 0]  # horizon 0, variable var_idx, shock col 0
    plt.hist(Phi0, bins=30, alpha=0.6)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Distribution of impact responses: {var_name} to {shock_name}")
    plt.show()

#plot_hist_impact(accepted_fiscal, 0, "fiscal_expansion", "infl_q_ann")

def quick_diag_summary(acc, var_order, shock_label):
    """
    acc: dict from apply_sign_restrictions (must have "IRFs", "accepted", "tried")
    var_order: list of variable names in VAR order
    shock_label: str, label for the identified shock
    """
    report = {}
    report["shock"] = shock_label
    report["accepted"] = acc.get("accepted", 0)
    report["tried"] = acc.get("tried", 0)
    report["accept_rate"] = round(report["accepted"] / report["tried"], 4) if report["tried"] else 0

    if report["accepted"] > 0:
        # Horizon 0 median impacts
        Phi0 = acc["IRFs"][:, 0, :, 0]  # (N, K) responses at h=0 to identified shock
        med_imp = np.median(Phi0, axis=0)
        lo_imp  = np.percentile(Phi0, 10, axis=0)
        hi_imp  = np.percentile(Phi0, 90, axis=0)
        impacts = {
            var_order[k]: {
                "median": round(med_imp[k], 3),
                "p10": round(lo_imp[k], 3),
                "p90": round(hi_imp[k], 3)
            }
            for k in range(len(var_order))
        }
        report["impact_responses"] = impacts
    else:
        report["impact_responses"] = "No accepted draws"

    # Print nicely
    print(f"\n=== Diagnostics summary: {shock_label} ===")
    print(f"Accepted: {report['accepted']} / {report['tried']} "
          f"({report['accept_rate']*100:.2f}% rate)")
    if isinstance(report["impact_responses"], dict):
        print("Median impact responses (h=0):")
        for v, stats in report["impact_responses"].items():
            print(f"  {v:20s}: {stats}")
    else:
        print("  No accepted draws.")

    return report

# Example usage
# var_order = [
    # "infl_q_ann","gdp_q_ann","policy_rate","gg_deficit_pct_gdp","gg_debt_pct_gdp",
    # "level","slope_10y_1y","curvature_ns_like"
# ]

#rep_fiscal  = quick_diag_summary(accepted_fiscal,  var_order, "Fiscal expansion")
#rep_monet   = quick_diag_summary(accepted_monet,   var_order, "Monetary tightening")

# Optionally save to JSON for your report
#with open("results/diag_summary.json","w") as f:
#    json.dump({"fiscal":rep_fiscal, "monetary":rep_monet}, f, indent=2)

def diag_table(acc, var_order, shock_label):
    """
    Build a tidy DataFrame of diagnostics for a given identified shock.
    """
    accepted = acc.get("accepted", 0)
    tried = acc.get("tried", 0)
    acc_rate = round(accepted / tried, 4) if tried else 0
    
    if accepted == 0:
        return pd.DataFrame([{
            "shock": shock_label,
            "var": "N/A",
            "median": "N/A",
            "p10": "N/A",
            "p90": "N/A",
            "accepted": accepted,
            "tried": tried,
            "accept_rate": acc_rate
        }])

    # h=0 responses across accepted draws
    Phi0 = acc["IRFs"][:, 0, :, 0]   # (N, K)
    med = np.median(Phi0, axis=0)
    lo  = np.percentile(Phi0, 10, axis=0)
    hi  = np.percentile(Phi0, 90, axis=0)

    rows = []
    for k, name in enumerate(var_order):
        rows.append({
            "shock": shock_label,
            "var": name,
            "median": round(med[k], 3),
            "p10": round(lo[k], 3),
            "p90": round(hi[k], 3),
            "accepted": accepted,
            "tried": tried,
            "accept_rate": f"{acc_rate*100:.2f}%"
        })
    return pd.DataFrame(rows)

# # Example: build tables
# table_fiscal  = diag_table(accepted_fiscal,  var_order, "Fiscal expansion")
# table_monet   = diag_table(accepted_monet,   var_order, "Monetary tightening")

# # Combine into one table for display
# diag_tables = pd.concat([table_fiscal, table_monet], ignore_index=True)
# # diag_tables

def build_impact_matrix(tables: dict, var_order: list):
    """
    tables: dict like {"Fiscal expansion": table_fiscal, "Monetary tightening": table_monet}
            where each value is the DataFrame returned by diag_table(...)
    var_order: list of variables in VAR order (rows of the heatmap)
    Returns: (M, shocks, vars) where M is a 2D numpy array [n_vars x n_shocks]
    """
    shocks = list(tables.keys())
    M = np.zeros((len(var_order), len(shocks)))
    for j, sk in enumerate(shocks):
        df = tables[sk]
        med_map = dict(zip(df["var"], df["median"]))
        for i, v in enumerate(var_order):
            M[i, j] = med_map.get(v, np.nan)
    return M, shocks, var_order

def plot_impact_heatmap(impact_matrix, shocks, vars_, title="Median impact responses (h=0)"):
    """
    impact_matrix: 2D array [n_vars x n_shocks] with median h=0 impacts
    shocks: list of shock names (columns)
    vars_: list of variable names (rows)
    """
    fig, ax = plt.subplots(figsize=(1.2*len(shocks)+4, 0.45*len(vars_)+2))
    im = ax.imshow(impact_matrix, aspect="auto")
    # Axis ticks/labels
    ax.set_xticks(np.arange(len(shocks)))
    ax.set_yticks(np.arange(len(vars_)))
    ax.set_xticklabels(shocks, rotation=30, ha="right")
    ax.set_yticklabels(vars_)
    ax.set_title(title)
    # Grid lines
    ax.set_xticks(np.arange(-.5, len(shocks), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(vars_), 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=1)
    # Annotate values
    for i in range(len(vars_)):
        for j in range(len(shocks)):
            val = impact_matrix[i, j]
            if np.isnan(val): 
                disp = "NA"
            else:
                disp = f"{val:.2f}"
            ax.text(j, i, disp, ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# ----- Example usage -----
# var_order used in your VAR:
# var_order = [
    # "infl_q_ann","gdp_q_ann","policy_rate","gg_deficit_pct_gdp","gg_debt_pct_gdp",
    # "level","slope_10y_1y","curvature_ns_like"
# ]

# # Build the tables first (from earlier step):
# table_fiscal  = diag_table(accepted_fiscal, var_order, "Fiscal expansion")
# table_monet   = diag_table(accepted_monet,  var_order, "Monetary tightening")

# # Compose heatmap input and plot
# impact_M, shocks, vars_ = build_impact_matrix(
    # {"Fiscal expansion": table_fiscal, "Monetary tightening": table_monet},
    # var_order
# )
# plot_impact_heatmap(impact_M, shocks, vars_, title="Median (h=0) responses by shock")
