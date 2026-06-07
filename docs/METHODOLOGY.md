# Methodology

What each engine in this repo actually does, why it does it that way, and
what it does *not* do. 

Code paths are given as `package.module:object` so this document can be
re-derived from the source if it ever drifts.

---

## 1. Short-rate model: Hull-White 1F

> `basel_risk_engine.rate_models.hull_white:HullWhiteModel`

### SDE and curve fit

Under the risk-neutral measure,

```
dr_t = (θ(t) − a·r_t) dt + σ dW_t
```

with `θ(t)` bootstrapped from the initial forward curve `f^M(0, t)` so that

```
α(t) = f^M(0, t) + σ² / (2a²) · (1 − e^{−a t})²
r_t = x_t + α(t),   x_t ~ OU(0, a, σ)
```

The model reprices the input zero curve exactly:
`P_model(0, τ; r_0) = P_market(0, τ)` whenever `r_0 = f^M(0, 0)`. This is
enforced in `basel_risk_engine.run:_calibrate` by pinning `r_0` to the
instantaneous forward at the valuation date; the resulting curve-fit
residual is at machine epsilon (~1e-16) and is surfaced on the IRRBB
page lineage panel.

### Calibration

`(a, σ)` come from a closed-form OLS regression on the AR(1)
representation of `x_t = r_t − f^M(0, t)`. The estimator is unbiased in
log-price space; the small-sample bias in the mean-reversion speed (Yu 2009) is the same one Vasicek κ shows, and the property tests assert
recovery within tolerance rather than at-point.

`θ(t)` is read directly off the curve — no fitting.

### MC simulation

Exact (not Euler) one-step transition:

```
r_{k+1} = e^{−a Δt} r_k
        + α(t_{k+1}) − e^{−a Δt} α(t_k)
        + σ · √((1 − e^{−2 a Δt}) / (2a)) · ε_k
```

Antithetic variates halve variance. Default: 2 000 paths × 60 monthly
steps, ~5 y horizon.

### Limitations

- Single factor → cannot reproduce steepener/flattener moves
  independently of parallel shifts. BCBS 368 prescribes the steepener
  and flattener as explicit deterministic shocks for exactly this
  reason; the MC distribution captures only the parallel/level risk
  in the short rate.
- Calibration is purely historical: no swaption / caplet vol inputs.
  Real production HW1F would calibrate σ to caplet implied vols at the
  hedging tenor.

---

## 2. NMD behavioural overlay

> `basel_risk_engine.behavioral.nmd:apply_nmd_overlay`

A row is classified NMD if `product == "deposit"` AND
`maturity_days ≤ short_maturity_threshold_days` (default 365). Its
effective repricing tenor becomes a volume-weighted blend:

```
m_b = stable_core_pct · core_behavioral_maturity_yrs
    + (1 − stable_core_pct) · m_contractual
```

Non-NMD rows: `m_b = m_contractual`. The output column
`behavioral_maturity_years` is what every downstream engine consumes for
discounting.

Default parameters (`NMDParams`): stable core 70 %, behavioural maturity
5 y, deposit β 0.40. β is consumed by the NII engine only.

---

## 3. Mortgage CPR + amortisation

> `basel_risk_engine.behavioral.mortgage_cpr`

### Level-payment schedule

Constant monthly payment

```
P = N · (c/12) / (1 − (1 + c/12)^{−n})     (c > 0)
P = N / n                                  (c → 0 limit)
```

decomposes into `I_t = (c/12)·B_{t−1}` and `A_t = P − I_t`. The
bookkeeping identity `Σ A_t = N` holds by construction.

### CPR model

Refinancing-incentive:

```
CPR(r) = clip(cpr_base + β · max(0, c − r), 0, cpr_cap)
SMM    = 1 − (1 − CPR)^(1/12)
```

with `r` the market refi rate at the *remaining-term* tenor (the curve
yield at `(n − t)/12`, floored at 12 months). Defaults
(`CPRParams`): base 6 % p.a., β 8, cap 60 %.

CPR-adjusted recursion:

```
B_t   = (B_{t−1} − A_t) · (1 − SMM_t)
prepay_t        = (B_{t−1} − A_t) · SMM_t
total_principal_t = A_t + prepay_t
```

`Σ total_principal_t = N` holds exactly (no defaults are modelled).

### Limitations

- No burnout (refinancers self-select; later prepayments are slower).
- No turnover (housing transactions).
- Single rate-sensitive driver. PSA-style ramps not supported.
- CPR is locked at the base curve when computing the MC ΔEVE
  distribution. Per-path CPR would require pricing each mortgage
  schedule under each path-implied curve (~2 000 × 100 × 360 ops);
  the deterministic BCBS 368 path captures the full curve sensitivity.

---

## 4. Black-76 callable bond pricing

> `basel_risk_engine.valuation.black76`

For a European call with expiry `T` and strike `K` (in unit-notional
terms) on a zero-coupon bond maturing at `S > T`, the Brigo–Mercurio
HW1F closed form (§3.3) is

```
ZBC(0, T, S, K) = P(0,S) · Φ(h) − K · P(0,T) · Φ(h − σ_P)

B(T, S)  = (1 − e^{−a (S − T)}) / a
σ_P      = σ · B(T, S) · √((1 − e^{−2 a T}) / (2a))
h        = ln(P(0,S) / (K · P(0,T))) / σ_P + σ_P / 2
```

`σ_P` is the **integrated** lognormal volatility of the forward bond
price over `[0, T]` (*not* annualised). The formula collapses to the
no-time-value intrinsic `max(P(0,S) − K·P(0,T), 0)` as `σ → 0` or
`T → 0`.

For a notional-`N` bond redeemed at strike `pct%` of par:
`call_value = N · ZBC(0, T_call, T_mat, pct/100)` and
`PV_callable = N · P(0, T_mat) − call_value`.

### Synthetic-data adaptation

The cashflows table models bonds as zero-coupon bullets (no coupon
schedule). A par-strike (`K = 1.0`) European call on such a bond is
deep-out-of-the-money in a positive-yield world — the holder always
gets `≥ K` at maturity anyway. Real callable bonds carry coupons that
make them trade above par, so a par-strike call is in the money by
roughly the cumulative coupon spread.

The ingestion proxy for this: callable bond strikes are randomised in
**[75, 88] % of par** in `basel_ingestion.generate`. Documented in the
generator and in the 2.1c CHANGELOG entry.

### Limitations

- European exercise only. Real callable bonds have Bermudan call
  schedules with step-down strikes. Pricing those properly under HW1F
  needs a Jamshidian decomposition; out of scope.
- HW1F vol calibrated to history; no swaption-vol calibration. Call
  values are correct given the calibrated `(a, σ)`, i.e., they just sit
  lower than what swaption vols would imply.

---

## 5. EVE engine

> `basel_risk_engine.valuation.eve:EVEEngine`

`EVE = Σ(asset PVs) − Σ(liability PVs)`. Sign convention: loans /
mortgages / bonds → +1, deposits → −1.

### Deterministic BCBS 368

`value(book, curve)` branches per cashflow type:

| Type                                    | Pricing                                                            |
|-----------------------------------------|--------------------------------------------------------------------|
| Bullet (loans, deposits, non-callable bonds) | `signed_amount · DF(behavioural_tau)` — vectorised             |
| Mortgages (`amortization_type == "level"`)   | CPR-adjusted schedule, each cashflow discounted at its own tenor |
| Callable bonds (HW1F only)                   | Straight bullet PV − Black-76 call value                       |

Six BCBS 368 §132 scenarios are applied as **bucket-resolved curve
shifts** (parallel ±200 bps, short ±250/180/100/40/0 bps with decay,
steepener −65/−30/+30/+70/+90, flattener +90/+50/0/−40/−65) and the
book is repriced under each shocked curve.

EBA supervisory outlier test:

```
ratio = max_s |ΔEVE_s| / Tier1
breach = ratio > 0.15
```

both the deterministic worst and the MC distributional `|ΔEVE|₉₉` are
reported on the dashboard.

### MC ΔEVE distribution

A flat per-payment table is built once at the base curve:

- bullets and callable bonds → one row at `behavioural_maturity_years`
- mortgages → one row per scheduled month under base-curve CPR
  (schedule locked across MC)

For each path's terminal short rate `r_h` at the 1y horizon, the table
is discounted via the HW1F instantaneous-shock approximation
`P(0, τ; r_h) = A(0, τ)·e^{−B(0, τ)·r_h}`. ΔEVE per path = path PV −
base-curve PV of the same table.

**Coherence note.** MC ΔEVE captures the duration / linear risk of the
book under stochastic curve moves; the deterministic BCBS 368 path
captures the convexity from optionality (CPR responding to rates,
Black-76 call kicking in). The two are reported alongside on purpose.

---

## 6. FTP engine

> `basel_risk_engine.ftp.attribution:compute_attribution`

Internal FTP curve = wholesale base curve + tenor-dependent liquidity
premium (linear interp on bps, flat extrapolation at the long end).

Per cashflow `i` with sign `s_i`, notional `N_i`, customer rate `c_i`,
contractual maturity `m_c`, behavioural maturity `m_b`:

```
ftp_b = FTP(m_b)                         (transfer-priced funding cost)
ftp_c = FTP(m_c)                         (contractual counterfactual)
r_f   = FTP(τ → 0)                       (wholesale overnight)

customer_margin   = s · N · (c − ftp_b)         (commercial)
funding_margin    = s · N · (ftp_b − r_f)       (treasury earns the term spread)
behavioural_value = s · N · (ftp_c − ftp_b)     (only on NMDs; else 0)
nii_total         = customer_margin + funding_margin
```

`nii_total` is **invariant to the FTP choice** — only the customer /
funding split moves. Behavioural value is positive on an upward curve
for NMD deposits (`m_b > m_c`, `ftp_b > ftp_c`, `s = −1`), which is the
deposit business's reward for sticky funding.

### Limitations

- Static — point-in-time decomposition, not a distributional NII.
- LP curve is a parametric input, not back-tested.

---

## 7. Liquidity survival horizon (ALMM)

> `basel_risk_engine.liquidity.survival:compute_survival_horizon`

### Three preset stresses (`LIQUIDITY_STRESS_SCENARIOS`)

| Parameter                  | Idiosyncratic | Market-wide | Combined |
|----------------------------|---------------|-------------|----------|
| Retail stable runoff       | 5 %           | 5 %         | 15 %     |
| Retail unstable runoff     | 10 %          | 10 %        | 30 %     |
| Wholesale runoff           | 40 %          | 100 %       | 100 %    |
| Asset inflow haircut       | 5 %           | 10 %        | 15 %     |
| HQLA L2A haircut           | 15 %          | 25 %        | 30 %     |
| HQLA L2B haircut           | 50 %          | 60 %        | 65 %     |

These are aligned with EBA LCR baselines and widened above the floors
for market-wide and combined stresses (fire-sale conditions).

### Algorithm

1. **t = 0 HQLA stock.** For each row with `direction == "inflow"` and
   `hqla_type ∈ {Level1, Level2A, Level2B}`,
   `cbc_contribution = amount · (1 − haircut(level))`.
   `initial_cbc = Σ cbc_contribution`. HQLA inflows are then removed
   from the maturity ladder (no double-count).

2. **Stress overlays on the ladder.**

   - `direction == "outflow"` & `product == "deposit"`:
     factor = retail-stable / retail-unstable / wholesale runoff
     depending on `(counterparty, is_nmd)`.
   - `direction == "outflow"` & `product != "deposit"`: pass-through
     (bond redemptions are contractual).
   - `direction == "inflow"`: factor = `1 − asset_inflow_haircut`.

3. **Aggregate by day**, fill in a 0..`max_horizon_days` grid (default
   365). Cashflows past the horizon are excluded — not clipped to the
   horizon edge.

4. **LCR-style 75 % inflow cap** per day:
   `capped_inflow = min(stressed_inflow, stressed_outflow · 0.75)`.

5. **Walk forward.**
   `running_cbc[d] = initial_cbc + cumulative_net_cashflow(0..d)`.
   Survival horizon = first day `running_cbc < 0`; if it never breaches,
   reported as the max horizon with `is_breached = False`.

### Headline numbers (current synthetic data)

Per scenario × stress: survival horizons under combined stress fall
around 220–270 days (within the realistic ALMM stress range), with
strict monotonicity `combined ≤ market_wide ≤ idiosyncratic` across all
four input scenarios.

### Limitations

- No intraday liquidity (Basel BCBS 248).
- Single currency (no FX liquidity stratification).
- No reverse stress testing (we set the stresses, we don't solve for
  "what runoff kills us"). The stress parameters themselves are
  configurable in code, not estimated from data.

---

## 8. Synthetic data: deliberate calibrations worth knowing

> `basel_ingestion.generate`

- **Short-rate history.** 5 y of monthly observations under a Vasicek
  SDE with `κ = 0.5, θ = 0.025, σ = 0.01, r_0 = 0.03`. Used to
  calibrate HW1F `(a, σ)` and to seed the initial curve.
- **HQLA mix.** `hqlatype` is weighted **15 / 5 / 5 / 75 %** across
  Level1 / Level2A / Level2B / None (*not* uniform). Uniform gave 75 %
  HQLA classification and a synthetically over-liquid bank that never
  breached any ALMM stress; the weighted mix yields a realistic ~30 M
  EUR HQLA stock with combined-stress breaches around day 220.
- **Mortgage share.** 30 % of loans get `amortization_type = "level"`
  with term 5–30 y.
- **Callable bond share.** 40 % of bonds longer than 5 y get a half-life
  European call with strike in [75, 88] % of par (premium-coupon proxy
  under the zero-coupon synthetic model. See §4 above).
- **Customer rate.** Wholesale base + per-product spread (loan +200 bps,
  bond +50 bps, deposit −150 bps) + ±10 bps idiosyncratic jitter.

---

## 9. Reading the dashboard

Every page now carries the same **Model lineage** expander at the top
(`src.lineage:render_model_lineage`). It shows: short-rate model family
+ version, calibrated `(a, σ)`, half-life, curve-fit residual,
calibration sample size, MC settings, NMD overlay parameters, CPR
parameters, and the three ALMM stress configurations. The dashboard
never re-runs the engine, i.e., it only reads marts produced by the offline
risk-engine run.

---

## 10. What's deliberately out of scope

- **FRTB / market-risk capital.** 
- **Multi-factor rate models** (HW2F, G2++). HW1F is sufficient for an
  IRRBB demonstration and keeps the calibration / closed-forms tractable.
- **Coupon-bond cashflow modelling.** Cashflows are bullet payments at
  maturity; the callable-strike adjustment (§4) is the workaround for
  the missing coupon stream.
- **Real bid/ask spreads, repo haircuts, FX, intraday liquidity.** Not
  modelled.
- **Distributional MC NII attribution.** The FTP attribution is static
  (point-in-time). Distributional NII would require running the FTP
  decomposition under each MC short-rate path; explicitly dropped at the
  Phase 2.1 scope freeze (2026-05-27).
- **Sobol / quasi-MC.** Pseudo-random Mersenne Twister + antithetics
  only. Same scope-freeze decision.
