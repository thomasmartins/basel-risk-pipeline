"""NII attribution under matched-funded FTP.

For each cashflow row i with sign s_i (asset +1 / liability -1), notional N_i,
customer rate c_i, and behavioural maturity m_b_i, the annualised NII (margin)
contribution is decomposed as

    nii_i = s_i · N_i · (c_i - r_f)

where r_f is the wholesale overnight funding rate. Under FTP this is split
into a *commercial* and a *funding* leg:

    customer_margin_i   = s_i · N_i · (c_i - ftp_b_i)
    funding_margin_i    = s_i · N_i · (ftp_b_i - r_f)
    nii_i               = customer_margin_i + funding_margin_i

with ftp_b_i := FTP rate at behavioural maturity m_b_i.

A third diagnostic — the *behavioural value* — captures the slice of
customer_margin attributable to pricing NMDs at behavioural maturity rather
than contractual maturity:

    ftp_c_i             = FTP rate at contractual maturity m_c_i
    behavioral_value_i  = s_i · N_i · (ftp_c_i - ftp_b_i)   (0 for non-NMDs)

For an upward-sloping curve and a deposit (s = -1, m_b > m_c, ftp_b > ftp_c)
the behavioural value is positive — the deposit business unit is credited
with a long-term funding rate as commercial margin, and treasury bears the
associated rolling-funding risk (which the IRRBB engine captures separately).

All margins are annualised; multiply by a horizon (in years) if you want a
period NII rather than an annualised rate × notional product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from basel_risk_engine.ftp.curve import FTPCurve

# Same asset/liability sign convention used by the EVE engine.
_ASSET_PRODUCTS = ("loan", "bond")


@dataclass(frozen=True)
class AttributionResult:
    per_row: pd.DataFrame      # one row per cashflow, all components
    book_total: pd.Series      # column-wise sums (customer_margin, funding_margin, ...)


def _signs(products: pd.Series) -> np.ndarray:
    return np.where(products.isin(_ASSET_PRODUCTS), 1.0, -1.0)


def compute_attribution(
    cashflows: pd.DataFrame,
    ftp_curve: FTPCurve,
    *,
    customer_rate_col: str = "customer_rate",
    behavioral_col: str = "behavioral_maturity_years",
    maturity_days_col: str = "maturity_days",
    amount_col: str = "amount",
    product_col: str = "product",
    is_nmd_col: str = "is_nmd",
) -> AttributionResult:
    """Compute per-row + book-level NII attribution under FTP.

    Required columns: `customer_rate`, `behavioral_maturity_years`,
    `maturity_days`, `amount`, `product`, `is_nmd`.
    The NMD overlay must have been applied beforehand so that the behavioural
    and is_nmd columns are present.
    """
    required = {customer_rate_col, behavioral_col, maturity_days_col, amount_col, product_col, is_nmd_col}
    missing = required - set(cashflows.columns)
    if missing:
        raise KeyError(f"FTP attribution missing columns: {sorted(missing)}")

    df = cashflows.copy()
    signs = _signs(df[product_col])
    amounts = df[amount_col].astype(float).to_numpy()
    customer = df[customer_rate_col].astype(float).to_numpy()
    behavioural_yrs = df[behavioral_col].astype(float).to_numpy()
    contractual_yrs = df[maturity_days_col].astype(float).to_numpy() / 365.0
    is_nmd = df[is_nmd_col].astype(bool).to_numpy()

    ftp_b = ftp_curve.ftp_yield(behavioural_yrs)
    ftp_c = ftp_curve.ftp_yield(contractual_yrs)
    r_f = ftp_curve.overnight_funding_rate()

    customer_margin = signs * amounts * (customer - ftp_b)
    funding_margin = signs * amounts * (ftp_b - r_f)
    behavioral_value = np.where(is_nmd, signs * amounts * (ftp_c - ftp_b), 0.0)
    nii_total = customer_margin + funding_margin

    out = df[[product_col, amount_col, is_nmd_col]].copy()
    out["customer_rate"] = customer
    out["ftp_behavioral"] = ftp_b
    out["ftp_contractual"] = ftp_c
    out["overnight_rate"] = r_f
    out["sign"] = signs
    out["customer_margin"] = customer_margin
    out["funding_margin"] = funding_margin
    out["behavioral_value"] = behavioral_value
    out["nii_total"] = nii_total

    totals = out[["customer_margin", "funding_margin", "behavioral_value", "nii_total"]].sum()
    return AttributionResult(per_row=out, book_total=totals)
