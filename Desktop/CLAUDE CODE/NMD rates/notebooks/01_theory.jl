### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ title
md"""
# Affine Term Structure Models & Riccati ODEs
### Application to NMD Rate Modelling

**Reference:** Utkarsh et al. (2023) — *Automated Translation and Accelerated Solving of Differential Equations on Multiple GPU Platforms* ([arXiv:2304.06835](https://arxiv.org/abs/2304.06835))

---
"""

# ╔═╡ section1
md"""
## 1  The Short Rate and Bond Pricing

Let $r(t)$ denote the continuously-compounded instantaneous short rate.
The **time-$t$ price of a zero-coupon bond** maturing at $T$ is the
risk-neutral expectation of the discounted payoff:

$$P(t, T) = \mathbb{E}^Q\!\left[\exp\!\left(-\int_t^T r(s)\,ds\right)\Bigg|\mathcal{F}_t\right]$$

The continuously-compounded **zero yield** at maturity $\tau = T - t$ is:

$$y(t,\tau) = -\frac{\ln P(t,T)}{\tau}$$

This yield curve $\tau \mapsto y(t,\tau)$ is the central object we model.
"""

# ╔═╡ section2
md"""
## 2  Affine Term Structure Models

A model is called **affine** if the log bond price is affine (linear + constant)
in the state variable $x(t)$:

$$\ln P(t,T) = A(\tau) - B(\tau)\, x(t), \qquad \tau = T - t$$

This is attractive because:
- Yields are linear in $x(t)$: $\;y(\tau) = -A(\tau)/\tau + B(\tau)\,x(t)/\tau$
- $A$ and $B$ are **deterministic** functions of $\tau$ only → solved once, reused forever
- The entire yield curve collapses to a 1-D (or $n$-D) state

**Sufficient conditions** (Duffie & Kan 1996): the risk-neutral dynamics of $x(t)$ must be of the form

$$dx(t) = \kappa\,(\theta - x(t))\,dt + \sqrt{\alpha_0 + \alpha_1\,x(t)}\;dW^Q(t)$$

with $r(t) = \delta_0 + \delta_1\,x(t)$.

| Model | $\alpha_0$ | $\alpha_1$ | Closed form? |
|-------|-----------|-----------|-------------|
| **Vasicek (1977)** | $\sigma^2$ | $0$ | ✓ |
| **CIR (1985)** | $0$ | $\sigma^2$ | ✓ |
| General affine | any | any | ODE only |
"""

# ╔═╡ section3
md"""
## 3  The Riccati ODE System

Substituting the affine ansatz $\ln P = A(\tau) - B(\tau)\,x$ into the
Feynman–Kač PDE and matching coefficients yields the **Riccati system**:

$$\boxed{\frac{dB}{d\tau} = \delta_1 - \kappa\,B - \frac{\alpha_1}{2}\,B^2}$$

$$\boxed{\frac{dA}{d\tau} = -\delta_0 + \kappa\,\theta\,B - \frac{\alpha_0}{2}\,B^2}$$

with boundary conditions $A(0) = B(0) = 0$ (at maturity the bond pays 1, so $\ln P = 0$).

**Key observations:**
1. The $B$ equation is a scalar Riccati ODE — nonlinear in $B$.
2. The $A$ equation is **linear** in $A$ once $B$ is known — it's an integral.
3. For Vasicek ($\alpha_1 = 0$), both equations become linear → closed form.
4. For CIR ($\alpha_0 = 0$), the $B$ equation is a Bernoulli ODE → closed form.
5. In general we **solve numerically** with an ODE solver.
"""

# ╔═╡ section4
md"""
## 4  Vasicek Closed Form

For Vasicek: $\delta_0 = 0$, $\delta_1 = 1$, $\alpha_0 = \sigma^2$, $\alpha_1 = 0$.
The Riccati ODE for $B$ becomes:

$$\frac{dB}{d\tau} = 1 - \kappa\,B \implies B(\tau) = \frac{1 - e^{-\kappa\tau}}{\kappa}$$

Integrating the $A$ equation:

$$A(\tau) = \left(B(\tau) - \tau\right)\!\left(\theta - \frac{\sigma^2}{2\kappa^2}\right) - \frac{\sigma^2 B(\tau)^2}{4\kappa}$$

The **zero yield** at maturity $\tau$ is:

$$y(\tau) = -\frac{A(\tau)}{\tau} + \frac{B(\tau)}{\tau}\,r(t)$$

This converges to the long-run mean as $\tau \to \infty$:

$$\lim_{\tau\to\infty} y(\tau) = \theta - \frac{\sigma^2}{2\kappa^2}$$

(the risk-neutral long rate is lower than $\theta$ due to the convexity term $-\sigma^2/2\kappa^2$).
"""

# ╔═╡ section5
md"""
## 5  Why This Matters for NMD Rates

Non-Maturity Deposits (NMDs) — savings accounts, current accounts — pay
a bank-set rate $d(t)$ that adjusts sluggishly to market conditions.
The standard regulatory framework (BCBS 368) models NMD repricing as:

$$d(\tau) = \alpha + \beta \cdot y(\tau)$$

where:
- $y(\tau)$ is the **model yield** at reference maturity $\tau$ → from our affine model
- $\beta \in [0,1]$ is the **pass-through** coefficient (estimated empirically)
- $\alpha$ is the bank margin / constant spread

**Uncertainty propagates as follows:**

$$\underbrace{p(\kappa, \theta, \sigma, r_0 \mid \text{data})}_{\text{posterior}} \;\longrightarrow\; \underbrace{y(\tau; \kappa, \theta, \sigma, r_0)}_{\text{ODE solve per draw}} \;\longrightarrow\; \underbrace{d(\tau) = \alpha + \beta y(\tau)}_{\text{NMD rate}}$$

With $N = 10{,}000$ posterior draws, we need $10{,}000$ Riccati ODE solves.
This is where **DiffEqGPU.jl** provides the 20–100× speedup over
CPU or JAX/PyTorch approaches (Utkarsh et al. 2023).
"""

# ╔═╡ section6
md"""
## 6  Roadmap

| Notebook | Content |
|----------|---------|
| **01 (this)** | Theory: affine models, Riccati ODEs, Vasicek |
| **02** | CPU implementation: ODE vs closed-form validation |
| **03** | GPU scaling: when does GPU pay off? |
| **04** | Bayesian calibration: credible bands for NMD rates |

Proceed to **`02_vasicek_cpu.jl`** →
"""
