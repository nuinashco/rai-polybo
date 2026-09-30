import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="medium",
    app_title="Gaussian Processes and Bayesian Optimization",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import cho_solve, solve_triangular
    from scipy.stats import norm, qmc

    def cholesky(A, lower=True):
        """Lower Cholesky factor with a clean zero upper triangle (safe to multiply by directly)."""
        return np.linalg.cholesky(A)

    return cho_solve, cholesky, mo, norm, np, plt, qmc, solve_triangular


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gaussian Processes and Bayesian Optimization

    *An interactive summary of `docs/gp-and-bo.md`.*

    Bayesian optimization (BO) finds the maximum of an **expensive black-box function** by fitting a probabilistic
    surrogate, usually a Gaussian process (GP), and choosing each next query to balance **exploring** uncertain regions
    against **exploiting** promising ones. It shines when one evaluation costs hours, dollars or a lab experiment, and
    you can afford tens to a few hundred of them.

    We want $x^* = \arg\max_x f(x)$ over a box $\mathcal{X} \subset \mathbb{R}^d$. Observations are
    $y = f(x) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$, and $D_n = \{(x_i, y_i)\}_{i=1}^n$.
    BO needs two ingredients:

    - a **surrogate model** that turns $D_n$ into a posterior belief over $f$ (the GP), and
    - an **acquisition function** $\alpha(x \mid D_n)$ that scores every candidate; we query its $\arg\max$.

    Everything below is plain NumPy/SciPy, so every formula in the doc maps to a few lines of code you can read.
    Move the sliders: every figure recomputes live.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        flowchart LR
            A[Initial design<br/>Sobol / LHS] --> B[Fit GP surrogate<br/>posterior + hyperparameters]
            B --> C[Maximize acquisition<br/>x = argmax α]
            C --> D[Evaluate f at x<br/>expensive]
            D --> E{Budget left?}
            E -- yes --> B
            E -- no --> F[Recommend best x]
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, plt):
    # Palette (categorical slots in fixed order) and a quiet matplotlib style.
    C = {
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "aqua": "#1baf7a",
        "yellow": "#eda100",
        "magenta": "#e87ba4",
        "green": "#008300",
        "violet": "#4a3aa7",
        "red": "#e34948",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "grid": "#e4e3df",
    }
    SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"], C["red"]]

    plt.rcParams.update(
        {
            "font.weight": "normal",
            "mathtext.fontset": "cm",  # LaTeX-looking math in figure text ($...$)  # override a global matplotlibrc that may request thin weights
            "figure.dpi": 110,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": C["ink2"],
            "axes.labelcolor": C["ink2"],
            "axes.titlesize": 10,
            "axes.titlecolor": C["ink"],
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": C["grid"],
            "grid.linewidth": 0.6,
            "xtick.color": C["ink2"],
            "ytick.color": C["ink2"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "font.size": 9,
        }
    )

    def show(fig):
        """Render a matplotlib figure to HTML and free it."""
        fig.tight_layout()
        html = mo.as_html(fig)
        plt.close(fig)
        return html

    def plot_posterior(ax, xs, mu, sd, X=None, y=None, f_true=None, color=None, label="posterior mean"):
        color = color or C["blue"]
        if f_true is not None:
            ax.plot(xs, f_true, color=C["ink2"], ls="--", lw=1.2, label=r"true $f$")
        ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=color, alpha=0.18, lw=0, label=r"$\pm 2\sigma$")
        ax.plot(xs, mu, color=color, label=label)
        if X is not None and len(X):
            ax.scatter(X, y, s=28, color=C["ink"], zorder=5, edgecolor="#fcfcfb", linewidth=1.2, label="data")

    return C, SERIES, plot_posterior, show


@app.cell(hide_code=True)
def _(cho_solve, cholesky, norm, np, solve_triangular):
    # ---------- kernels ----------
    def _as2d(a):
        a = np.asarray(a, dtype=float)
        return a[:, None] if a.ndim == 1 else a

    def dist(A, B, ls=1.0):
        A, B = _as2d(A) / ls, _as2d(B) / ls
        d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
        return np.sqrt(np.maximum(d2, 0.0))

    def k_se(A, B, ls=0.2, sf2=1.0):
        return sf2 * np.exp(-0.5 * dist(A, B, ls) ** 2)

    def k_m12(A, B, ls=0.2, sf2=1.0):
        return sf2 * np.exp(-dist(A, B, ls))

    def k_m32(A, B, ls=0.2, sf2=1.0):
        r = np.sqrt(3) * dist(A, B, ls)
        return sf2 * (1 + r) * np.exp(-r)

    def k_m52(A, B, ls=0.2, sf2=1.0):
        r = np.sqrt(5) * dist(A, B, ls)
        return sf2 * (1 + r + r**2 / 3) * np.exp(-r)

    def k_periodic(A, B, ls=1.0, period=0.3, sf2=1.0):
        r = dist(A, B, 1.0)
        return sf2 * np.exp(-2 * np.sin(np.pi * r / period) ** 2 / ls**2)

    def k_linear(A, B, sf2=1.0, c=0.5):
        return sf2 * (_as2d(A) - c) @ (_as2d(B) - c).T

    KERNELS = {"Matérn 1/2": k_m12, "Matérn 3/2": k_m32, "Matérn 5/2": k_m52, "Squared exponential": k_se}

    # ---------- GP regression ----------
    def gp_posterior(X, y, Xs, kern, noise=1e-6, full_cov=False):
        """Posterior mean and variance (or covariance) via one Cholesky factorization."""
        Xs = np.asarray(Xs, float)
        if X is None or len(X) == 0:
            cov = kern(Xs, Xs)
            return np.zeros(len(Xs)), (cov if full_cov else np.diag(cov).copy())
        K = kern(X, X) + (noise + 1e-9) * np.eye(len(X))
        L = cholesky(K, lower=True)
        Ks = kern(X, Xs)
        mu = Ks.T @ cho_solve((L, True), y)
        V = solve_triangular(L, Ks, lower=True)
        if full_cov:
            return mu, kern(Xs, Xs) - V.T @ V
        kss = np.diag(kern(Xs, Xs))
        return mu, np.maximum(kss - (V**2).sum(0), 1e-12)

    def log_ml_terms(X, y, kern, noise):
        """(data fit, complexity penalty, constant) of the log marginal likelihood."""
        K = kern(X, X) + (noise + 1e-9) * np.eye(len(X))
        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), y)
        return -0.5 * y @ alpha, -np.log(np.diag(L)).sum(), -0.5 * len(X) * np.log(2 * np.pi)

    def fit_lengthscale(X, y, base_kern, noise, grid=np.logspace(-2, 0, 30)):
        """Type-II ML over a lengthscale grid (cheap, robust stand-in for L-BFGS with restarts)."""
        scores = [sum(log_ml_terms(X, y, lambda a, b, l=l: base_kern(a, b, ls=l), noise)) for l in grid]
        return grid[int(np.argmax(scores))]

    def sample_paths(mu, cov, n, rng):
        return rng.multivariate_normal(mu, cov + 1e-9 * np.eye(len(mu)), size=n, method="eigh")

    # ---------- acquisition functions (maximization) ----------
    def acq_pi(mu, sd, best, xi=0.0):
        return norm.cdf((mu - best - xi) / sd)

    def acq_ei(mu, sd, best):
        z = (mu - best) / sd
        return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)

    def acq_ucb(mu, sd, beta=4.0):
        return mu + np.sqrt(beta) * sd

    # ---------- the running 1D test problem: negated, rescaled Forrester function ----------
    def objective(x):
        x = np.asarray(x, float)
        return -((6 * x - 2) ** 2 * np.sin(12 * x - 4)) / 6.02

    XS = np.linspace(0, 1, 400)
    F_TRUE = objective(XS)
    F_STAR = F_TRUE.max()
    X_STAR = XS[F_TRUE.argmax()]
    return (
        F_STAR,
        F_TRUE,
        KERNELS,
        XS,
        acq_ei,
        acq_pi,
        acq_ucb,
        fit_lengthscale,
        gp_posterior,
        k_linear,
        k_m12,
        k_m32,
        k_m52,
        k_periodic,
        k_se,
        log_ml_terms,
        objective,
        sample_paths,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Gaussian processes

    ### Everything starts with Gaussian conditioning

    If $(\mathbf f, \mathbf f_*)$ are jointly Gaussian, then

    $$
    \mathbf{f}_* \mid \mathbf{f} \sim \mathcal{N}\!\left( \mathbf{m}_* + K_*^\top K^{-1} (\mathbf{f} - \mathbf{m}),\; K_{**} - K_*^\top K^{-1} K_* \right)
    $$

    Two facts to notice in the figure: the conditional **mean moves linearly** with the observed value, and the
    conditional **variance does not depend on the observed value at all**, only on the correlation. That second fact is
    what makes "fantasy" batch BO work later.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rho = mo.ui.slider(-0.95, 0.95, step=0.05, value=0.8, label="correlation ρ", show_value=True)
    f1_obs = mo.ui.slider(-2.5, 2.5, step=0.1, value=1.2, label="observed f₁", show_value=True)
    return f1_obs, rho


@app.cell(hide_code=True)
def _(C, f1_obs, mo, norm, np, plt, rho, show):
    _r, _a = rho.value, f1_obs.value
    _g = np.linspace(-3.2, 3.2, 200)
    _A, _B = np.meshgrid(_g, _g)
    _dens = np.exp(-(_A**2 - 2 * _r * _A * _B + _B**2) / (2 * (1 - _r**2)))

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    _ax1.contour(_A, _B, _dens, levels=6, colors=C["blue"], linewidths=1.0)
    _ax1.axvline(_a, color=C["orange"], lw=1.6)
    _ax1.set(xlabel=r"$f_1$ (observed)", ylabel=r"$f_2$ (predict)", title=r"Joint prior $p(f_1, f_2)$", aspect="equal")

    _cm, _cs = _r * _a, np.sqrt(1 - _r**2)
    _ax2.plot(_g, norm.pdf(_g), color=C["ink2"], ls="--", lw=1.2, label=r"prior $p(f_2)$")
    _ax2.fill_between(_g, norm.pdf(_g, _cm, _cs), color=C["orange"], alpha=0.2, lw=0)
    _ax2.plot(_g, norm.pdf(_g, _cm, _cs), color=C["orange"], label=r"conditional $p(f_2 \mid f_1)$")
    _ax2.set(xlabel=r"$f_2$", title=rf"$\mathbb{{E}}[f_2 \mid f_1] = \rho \cdot f_1 = {_cm:.2f}, \quad \mathrm{{Var}}[f_2 \mid f_1] = 1 - \rho^2 = {_cs**2:.2f}$")
    _ax2.legend(loc="upper left")
    mo.vstack([mo.hstack([rho, f1_obs], justify="start"), show(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### From vectors to functions: the posterior predictive

    A GP $f \sim \mathcal{GP}(m, k)$ is a collection of random variables any finite subset of which is jointly
    Gaussian. Marginalization means we only ever touch the Gaussian over the training and test points. With a zero
    prior mean and noisy observations,

    $$
    \mu_n(x_*) = \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y},
    \qquad
    \sigma_n^2(x_*) = k(x_*, x_*) - \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{k}_* .
    $$

    The mean is kernel ridge regression; the variance collapses near data and returns to the prior far from it,
    which is exactly what an optimizer needs: **it knows where it has not looked**. In code this is one Cholesky
    factorization, $O(n^3)$ once, then $O(n)$ per mean and $O(n^2)$ per variance.
    """)
    return


@app.cell(hide_code=True)
def _(KERNELS, mo):
    gp_kernel = mo.ui.dropdown(list(KERNELS), value="Matérn 5/2", label="kernel")
    gp_ls = mo.ui.slider(0.02, 0.5, step=0.01, value=0.15, label="lengthscale ℓ", show_value=True)
    gp_noise = mo.ui.slider(0.0, 0.3, step=0.01, value=0.02, label="noise σₙ", show_value=True)
    gp_n = mo.ui.slider(0, 15, step=1, value=5, label="observations n", show_value=True)
    return gp_kernel, gp_ls, gp_n, gp_noise


@app.cell(hide_code=True)
def _(
    C,
    F_TRUE,
    KERNELS,
    SERIES,
    XS,
    gp_kernel,
    gp_ls,
    gp_n,
    gp_noise,
    gp_posterior,
    mo,
    np,
    objective,
    plot_posterior,
    plt,
    sample_paths,
    show,
):
    _rng = np.random.default_rng(3)
    _Xall = _rng.uniform(0, 1, 15)
    _eps = _rng.standard_normal(15)
    _X = _Xall[: gp_n.value]
    _y = objective(_X) + gp_noise.value * _eps[: gp_n.value]
    _kern = lambda a, b: KERNELS[gp_kernel.value](a, b, ls=gp_ls.value)
    _xs = XS[::2]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    _mu0, _cov0 = gp_posterior(None, None, _xs, _kern, full_cov=True)
    for _i, _s in enumerate(sample_paths(_mu0, _cov0, 3, np.random.default_rng(0))):
        _ax1.plot(_xs, _s, color=SERIES[_i], lw=1.2)
    _ax1.fill_between(_xs, -2, 2, color=C["blue"], alpha=0.1, lw=0)
    _ax1.set(title="Prior: 3 sample functions", xlabel=r"$x$", ylim=(-3, 3))

    _mu, _cov = gp_posterior(_X, _y, _xs, _kern, noise=gp_noise.value**2, full_cov=True)
    _sd = np.sqrt(np.maximum(np.diag(_cov), 1e-12))
    plot_posterior(_ax2, _xs, _mu, _sd, _X, _y, F_TRUE[::2])
    for _i, _s in enumerate(sample_paths(_mu, _cov, 3, np.random.default_rng(0))):
        _ax2.plot(_xs, _s, color=SERIES[_i], lw=0.8, alpha=0.7)
    _ax2.set(title=f"Posterior after {gp_n.value} observations", xlabel=r"$x$")
    _ax2.legend(loc="lower left", ncols=2)
    mo.vstack([mo.hstack([gp_kernel, gp_ls], justify="start"), mo.hstack([gp_noise, gp_n], justify="start"), show(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Learning hyperparameters from the marginal likelihood

    $$
    \log p(\mathbf{y} \mid X, \theta) =
    \underbrace{-\tfrac{1}{2} \mathbf{y}^\top K_y^{-1} \mathbf{y}}_{\text{data fit}}
    \;\underbrace{-\tfrac{1}{2} \log |K_y|}_{\text{complexity penalty}}
    \;-\; \tfrac{n}{2} \log 2\pi
    $$

    Short lengthscales fit the data but pay a large complexity penalty; long ones are simple but fit badly. The
    balance is an **automatic Occam's razor**. The right-hand panel shows the surface is non-convex: an
    "everything is noise" explanation (long $\ell$, large $\sigma_n$) competes with interpolation (short $\ell$,
    small $\sigma_n$), which is why practical fitting uses restarts and priors.
    """)
    return


@app.cell(hide_code=True)
def _(
    C,
    F_TRUE,
    XS,
    gp_posterior,
    k_se,
    log_ml_terms,
    np,
    objective,
    plot_posterior,
    plt,
    show,
):
    _rng = np.random.default_rng(7)
    _X = np.sort(_rng.uniform(0, 1, 12))
    _y = objective(_X) + 0.1 * _rng.standard_normal(12)
    _ym, _ys = _y.mean(), _y.std()
    _yz = (_y - _ym) / _ys
    _noise = (0.1 / _ys) ** 2

    _ls = np.logspace(-2.3, 0.5, 160)
    _terms = np.array([log_ml_terms(_X, _yz, lambda a, b, l=l: k_se(a, b, ls=l), _noise) for l in _ls])
    _total = _terms.sum(1)
    _best = _ls[_total.argmax()]

    _fig = plt.figure(figsize=(11, 6.4))
    _gs = _fig.add_gridspec(2, 3)
    _ax = _fig.add_subplot(_gs[0, :2])
    _ax.plot(_ls, _terms[:, 0], color=C["orange"], label=r"data fit $-\frac{1}{2}\mathbf{y}^\top K_y^{-1}\mathbf{y}$")
    _ax.plot(_ls, _terms[:, 1], color=C["aqua"], label=r"complexity $-\frac{1}{2}\log|K_y|$")
    _ax.plot(_ls, _total, color=C["blue"], lw=2.2, label=r"$\log p(\mathbf{y} \mid X, \theta)$")
    _ax.axvline(_best, color=C["ink2"], ls=":", lw=1)
    _ax.set(xscale="log", xlabel=r"lengthscale $\ell$", ylim=(_total.max() - 40, max(_terms[:, 1].max(), 5) + 3),
            title=rf"Decomposition (SE kernel, fixed noise). ML optimum $\ell \approx {_best:.3f}$")
    _ax.legend(loc="lower right")

    _axc = _fig.add_subplot(_gs[0, 2])
    _lg, _ng = np.logspace(-2.3, 0.5, 50), np.logspace(-2.5, 0.2, 50)
    _Z = np.array([[sum(log_ml_terms(_X, _yz, lambda a, b, l=l: k_se(a, b, ls=l), s**2)) for l in _lg] for s in _ng])
    _axc.contourf(_lg, _ng, np.maximum(_Z, _Z.max() - 25), levels=20, cmap="Blues")
    _axc.set(xscale="log", yscale="log", xlabel=r"lengthscale $\ell$", ylabel=r"noise $\sigma_n$ (standardized)",
             title=r"$\log p(\mathbf{y} \mid X, \theta)$ over $(\ell, \sigma_n)$")
    _axc.grid(False)

    for _j, (_l, _name) in enumerate([(0.012, "too short: interpolates noise"), (_best, "ML optimum"),
                                      (2.5, "too long: calls it noise")]):
        _a = _fig.add_subplot(_gs[1, _j])
        _mu, _var = gp_posterior(_X, _yz, XS, lambda a, b, l=_l: k_se(a, b, ls=l), _noise)
        plot_posterior(_a, XS, _mu * _ys + _ym, np.sqrt(_var) * _ys, _X, _y, F_TRUE)
        _a.set(title=rf"$\ell = {_l:.3f}$ ({_name})", ylim=(-2.8, 1.8), xlabel=r"$x$")
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The weight-space view.** A GP is Bayesian linear regression $f(x)=\phi(x)^\top w$ in a (possibly infinite)
    feature space with $k(x,x') = \phi(x)^\top \Sigma_p \phi(x')$. Infinitely wide Bayesian neural nets converge to GPs
    (Neal 1996; Lee et al. 2018), and random-feature approximations (Section 6) come straight from this view.

    ## 2. Kernels: what you believe about $f$

    The kernel **is** the prior. It decides smoothness, how far information travels, which inputs matter, and whether
    $f$ is periodic or additive. In BO the kernel choice often matters more than the acquisition function.

    A Matérn-$\nu$ process is $k$ times mean-square differentiable exactly when $\nu > k$; $\nu \to \infty$ gives the
    squared exponential. **Matérn-5/2 is the standard BO default** (Snoek et al. 2012): SE is unrealistically smooth,
    which makes the GP overconfident between observations and starves exploration.
    """)
    return


@app.cell(hide_code=True)
def _(SERIES, cholesky, k_m12, k_m32, k_m52, k_se, np, plt, show):
    _xs = np.linspace(0, 1, 300)
    _z = np.random.default_rng(11).standard_normal((300, 2))
    _ks = [("Matérn 1/2 (rough)", k_m12), ("Matérn 3/2", k_m32), ("Matérn 5/2 (BO default)", k_m52),
           (r"Squared exponential ($\nu \to \infty$)", k_se)]
    _fig = plt.figure(figsize=(11, 5.6))
    _gs = _fig.add_gridspec(2, 4)
    for _j, (_name, _k) in enumerate(_ks):
        _a = _fig.add_subplot(_gs[0, _j])
        _L = cholesky(_k(_xs, _xs, ls=0.2) + 1e-6 * np.eye(300), lower=True)
        for _i in range(2):
            _a.plot(_xs, _L @ _z[:, _i], color=SERIES[_i], lw=1.1)
        _a.set(title=_name, ylim=(-3, 3), xlabel=r"$x$")

    _r = np.linspace(0, 1, 200)
    _ax = _fig.add_subplot(_gs[1, :2])
    _sp = _fig.add_subplot(_gs[1, 2:])
    _s = np.linspace(0, 6, 300)
    _ell = 0.2
    for _i, ((_name, _k), _nu) in enumerate(zip(_ks, [0.5, 1.5, 2.5, np.inf])):
        _ax.plot(_r, _k(np.zeros(1), _r, ls=_ell)[0], color=SERIES[_i], label=_name.split(" (")[0])
        if np.isinf(_nu):
            _S = np.exp(-2 * np.pi**2 * _ell**2 * _s**2)
        else:
            _S = (1 + 4 * np.pi**2 * _s**2 * _ell**2 / (2 * _nu)) ** (-(_nu + 0.5))
        _sp.plot(_s, _S, color=SERIES[_i], label=_name.split(" (")[0])
    _ax.set(title=r"Kernel $k(r)$, $\ell = 0.2$", xlabel=r"distance $r$")
    _ax.legend()
    _sp.set(yscale="log", ylim=(1e-8, 2), xlabel=r"frequency $s$",
            title="Spectral density (Bochner): heavier tails ⇒ rougher samples")
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Building kernels from parts, and ARD

    Valid kernels are closed under **sums** (independent additive components) and **products** (similar only if
    similar under both). With one lengthscale per dimension, $r^2 = \sum_d (x_d - x'_d)^2/\ell_d^2$
    (**automatic relevance determination**): a large $\ell_d$ switches dimension $d$ off.
    """)
    return


@app.cell(hide_code=True)
def _(SERIES, cholesky, k_linear, k_periodic, k_se, np, plt, show):
    _xs = np.linspace(0, 1, 250)
    _z = np.random.default_rng(5).standard_normal((250, 3))
    _combos = [
        (r"Periodic ($p = 0.3$)", lambda a, b: k_periodic(a, b, ls=1.0, period=0.3)),
        (r"Periodic $\times$ SE: locally periodic", lambda a, b: k_periodic(a, b, ls=1.0, period=0.15) * k_se(a, b, ls=0.4)),
        ("Linear", lambda a, b: k_linear(a, b, sf2=4.0)),
        ("Linear + SE: trend + wiggles", lambda a, b: k_linear(a, b, sf2=4.0) + 0.2 * k_se(a, b, ls=0.08)),
    ]
    _fig = plt.figure(figsize=(11, 5.8))
    _gs = _fig.add_gridspec(2, 4)
    for _j, (_name, _k) in enumerate(_combos):
        _a = _fig.add_subplot(_gs[0, _j])
        _L = cholesky(_k(_xs, _xs) + 1e-6 * np.eye(250), lower=True)
        for _i in range(3):
            _a.plot(_xs, _L @ _z[:, _i], color=SERIES[_i], lw=1.1)
        _a.set(title=_name, xlabel=r"$x$")

    _g = np.linspace(0, 1, 36)
    _G = np.stack(np.meshgrid(_g, _g), -1).reshape(-1, 2)
    _zz = np.random.default_rng(1).standard_normal(len(_G))
    for _j, (_ls, _title) in enumerate([(np.array([0.2, 0.2]), r"Isotropic: $\ell = (0.2, 0.2)$"),
                                        (np.array([0.1, 1.5]), r"ARD: $\ell = (0.1, 1.5) \Rightarrow x_2$ irrelevant")]):
        _a = _fig.add_subplot(_gs[1, 2 * _j: 2 * _j + 2])
        _L = cholesky(k_se(_G, _G, ls=_ls) + 1e-6 * np.eye(len(_G)), lower=True)
        _a.imshow((_L @ _zz).reshape(36, 36), origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", aspect="auto")
        _a.set(title=_title, xlabel=r"$x_1$", ylabel=r"$x_2$")
        _a.grid(False)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The RKHS trap.** Every kernel defines an RKHS, and the GP mean is the minimum-norm interpolant in it. But for
    the kernels used in practice, GP **sample paths almost surely do not lie in their own RKHS** (Kanagawa et al.
    2018). Bayesian regret analyses ($f \sim \mathcal{GP}$) and frequentist ones ($\|f\|_{\mathcal H} \le B$) are
    therefore about different function classes.

    ## 3. The Bayesian optimization loop and acquisition functions

    From Kushner (1964, PI) and Močkus (1975, EI) to EGO (Jones et al. 1998) and Spearmint (Snoek et al. 2012), the
    loop has stayed the same: initial design → fit GP → maximize $\alpha$ → evaluate → repeat. Throughout, $f^+$ is the
    incumbent (best value so far) and $z = (\mu - f^+)/\sigma$.

    | Family | Acquisition | Formula |
    | :--- | :--- | :--- |
    | Improvement | PI | $\Phi\big((\mu - f^+ - \xi)/\sigma\big)$ |
    | Improvement | EI | $(\mu - f^+)\Phi(z) + \sigma\phi(z)$ |
    | Optimism | GP-UCB | $\mu + \sqrt{\beta_t}\,\sigma$ |
    | Posterior sampling | Thompson | $\arg\max$ of one posterior sample $\tilde f$ |
    | Information | KG / ES / PES / MES | value of information about $\max\mu$, $x^*$ or $f^*$ |

    Change the margin $\xi$ and the exploration weight $\beta$ below and watch where each rule wants to query next.
    PI with $\xi = 0$ hugs the incumbent; UCB with large $\beta$ chases uncertainty.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    acq_n = mo.ui.slider(2, 10, step=1, value=4, label="observations n", show_value=True)
    acq_xi = mo.ui.slider(0.0, 0.5, step=0.01, value=0.0, label="PI margin ξ", show_value=True)
    acq_beta = mo.ui.slider(0.1, 16.0, step=0.1, value=4.0, label="UCB β", show_value=True)
    acq_seed = mo.ui.slider(0, 20, step=1, value=0, label="Thompson sample seed", show_value=True)
    return acq_beta, acq_n, acq_seed, acq_xi


@app.cell(hide_code=True)
def _(
    C,
    F_TRUE,
    XS,
    acq_beta,
    acq_ei,
    acq_n,
    acq_pi,
    acq_seed,
    acq_ucb,
    acq_xi,
    fit_lengthscale,
    gp_posterior,
    k_m52,
    mo,
    np,
    objective,
    plot_posterior,
    plt,
    show,
):
    _X = np.array([0.05, 0.35, 0.95, 0.55, 0.2, 0.85, 0.65, 0.45, 0.1, 0.75])[: acq_n.value]
    _y = objective(_X)
    _ym, _ys = _y.mean(), _y.std() + 1e-9
    _yz = (_y - _ym) / _ys
    _ls = fit_lengthscale(_X, _yz, k_m52, 1e-6)
    _kern = lambda a, b: k_m52(a, b, ls=_ls)
    _xs = XS[::2]
    _mu, _cov = gp_posterior(_X, _yz, _xs, _kern, full_cov=True)
    _sd = np.sqrt(np.maximum(np.diag(_cov), 1e-12))
    _best = _yz.max()
    _ts = np.random.default_rng(acq_seed.value).multivariate_normal(_mu, _cov + 1e-9 * np.eye(len(_xs)), method="eigh")

    _acqs = [
        (rf"PI ($\xi = {acq_xi.value:.2f}$)", acq_pi(_mu, _sd, _best, acq_xi.value)),
        ("EI", acq_ei(_mu, _sd, _best)),
        (rf"UCB ($\beta = {acq_beta.value:.1f}$)", acq_ucb(_mu, _sd, acq_beta.value)),
        ("Thompson sample", _ts),
    ]
    _fig, _axes = plt.subplots(5, 1, figsize=(9, 8.4), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]})
    plot_posterior(_axes[0], _xs, _mu * _ys + _ym, _sd * _ys, _X, _y, F_TRUE[::2])
    _axes[0].axhline(_y.max(), color=C["ink2"], lw=0.8, ls=":")
    _axes[0].set(title=rf"GP posterior (Matérn 5/2, ML lengthscale $\ell = {_ls:.2f}$); dotted line = incumbent $f^+$")
    _axes[0].legend(loc="lower left", ncols=4)
    for _a, (_name, _v) in zip(_axes[1:], _acqs):
        _i = int(np.argmax(_v))
        _a.fill_between(_xs, _v.min(), _v, color=C["orange"], alpha=0.15, lw=0)
        _a.plot(_xs, _v, color=C["orange"])
        _a.axvline(_xs[_i], color=C["ink"], lw=1, ls="--")
        _a.set_ylabel(_name, rotation=0, ha="right", va="center")
        _a.set_yticks([])
        _axes[0].plot(_xs[_i], (_mu * _ys + _ym)[_i], marker="v", ms=8, color=C["orange"], zorder=6)
    _axes[-1].set_xlabel(r"$x$    (dashed line = next query, ▼ on the posterior)")
    mo.vstack([mo.hstack([acq_n, acq_xi], justify="start"), mo.hstack([acq_beta, acq_seed], justify="start"), show(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EI's hidden numerical problem, and LogEI

    Far from the incumbent ($z \ll 0$), EI and its gradient **underflow to exactly zero** in float64 (below
    $z \approx -38$), and the naive formula $z\Phi(z) + \phi(z)$ steadily loses precision to cancellation on the way.
    In practice $\mu$ and $\sigma$ are themselves noisy floats, so the acquisition is effectively flat long before that. Gradient-based inner optimization then sees a flat
    landscape and stalls. **LogEI** (Ament et al. 2023) computes $\log \mathrm{EI}$ stably with an asymptotic
    expansion: same maximizer, well-behaved landscape. Many past "better acquisition" results were partly
    inner-optimization failures of plain EI.
    """)
    return


@app.cell(hide_code=True)
def _(C, norm, np, plt, show):
    # h(z) = z Φ(z) + φ(z) is EI for σ = 1. A cancellation-free log h(z), in the spirit of LogEI:
    #   z > -1:  direct formula (no cancellation there)
    #   z ≤ -1:  with t = -z, h = φ(z)(1 - t·R(t)) where R is the Mills ratio. Its continued fraction
    #            R = 1/g₀, g_k = t + (k+1)/g_{k+1}, gives 1 - t·R = 1/(g₀ g₁) with no subtraction at all.
    def _log_h(z, K=3000):
        z = np.asarray(z, float)
        out = np.empty_like(z)
        m = z > -1
        out[m] = np.log(z[m] * norm.cdf(z[m]) + norm.pdf(z[m]))
        t = -z[~m]
        g = t.copy()
        for k in range(K, 0, -1):
            g = t + (k + 1) / g
        out[~m] = norm.logpdf(z[~m]) - np.log(g * (t + 1 / g))
        return out

    _z = np.linspace(-45, 3, 3000)
    _naive = _z * norm.cdf(_z) + norm.pdf(_z)
    _grad = norm.cdf(_z)
    _lh = _log_h(_z)
    _dead = _z[(_naive <= 0) | (_grad <= 0)].max()
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    _ax1.plot(_z, _lh / np.log(10), color=C["blue"], lw=2.2, label=r"$\log_{10}\,\mathrm{EI}$ via LogEI (stable)")
    with np.errstate(divide="ignore"):
        _ax1.plot(_z, np.where(_grad > 0, np.log10(_grad), np.nan), color=C["orange"], lw=1.4,
                  label=r"$\log_{10}\,\partial\mathrm{EI}/\partial\mu = \log_{10}\Phi(z)$, naive")
    _ax1.axvspan(_z.min(), _dead, color=C["orange"], alpha=0.08, lw=0)
    _ax1.annotate("naive EI and its\ngradient are exactly 0", (_dead - 0.5, -150), ha="right", color=C["ink2"], fontsize=8)
    _ax1.set(xlabel=r"$z = (\mu - f^+)/\sigma$", ylabel=r"$\log_{10}$ value", title=rf"Float64 underflow below $z \approx {_dead:.0f}$")
    _ax1.legend(loc="lower right")

    with np.errstate(divide="ignore", invalid="ignore"):
        _rel = np.abs(np.exp(np.log(np.where(_naive > 0, _naive, np.nan)) - _lh) - 1)
    _ax2.plot(_z, np.log10(np.maximum(_rel, 1e-17)), color=C["orange"], lw=0.6, alpha=0.8)
    _ax2.axhline(0, color=C["ink2"], ls=":", lw=1)
    _ax2.set(xlabel=r"$z$", ylabel=r"$\log_{10}$ relative error", xlim=(-40, 3), ylim=(-17, 1),
             title="Naive EI: cancellation erodes ~6 digits, then total failure")
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Watch the loop run

    Three random initial points, then 12 iterations. Each step refits the lengthscale by marginal likelihood,
    maximizes the acquisition on a dense grid (the "inner problem"), and evaluates $f$. Step through iterations and
    compare policies.
    """)
    return


@app.cell(hide_code=True)
def _(
    XS,
    acq_ei,
    acq_pi,
    acq_ucb,
    fit_lengthscale,
    gp_posterior,
    k_m52,
    np,
    objective,
):
    def run_bo(policy, seed=0, n_init=3, n_iter=12, beta=4.0, xi=0.01, keep_history=True):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, n_init)
        y = objective(X)
        xs = XS[::2]
        history = []
        for _t in range(n_iter):
            if policy == "Random":
                x_next = rng.uniform(0, 1)
                if keep_history:
                    history.append(None)
            else:
                ym, ys = y.mean(), y.std() + 1e-9
                yz = (y - ym) / ys
                ls = fit_lengthscale(X, yz, k_m52, 1e-6)
                kern = lambda a, b, l=ls: k_m52(a, b, ls=l)
                if policy == "Thompson":
                    mu, cov = gp_posterior(X, yz, xs, kern, full_cov=True)
                    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
                    a = rng.multivariate_normal(mu, cov + 1e-9 * np.eye(len(xs)), method="eigh")
                else:
                    mu, var = gp_posterior(X, yz, xs, kern)
                    sd = np.sqrt(var)
                    a = {"EI": lambda: acq_ei(mu, sd, yz.max()),
                         "PI": lambda: acq_pi(mu, sd, yz.max(), xi),
                         "UCB": lambda: acq_ucb(mu, sd, beta)}[policy]()
                x_next = xs[int(np.argmax(a))]
                if keep_history:
                    history.append(dict(X=X.copy(), y=y.copy(), mu=mu * ys + ym, sd=sd * ys, acq=a, x_next=x_next))
            X = np.append(X, x_next)
            y = np.append(y, objective(x_next))
        return X, y, history

    POLICIES = ["EI", "PI", "UCB", "Thompson"]
    BO_RUNS = {p: run_bo(p, seed=2) for p in POLICIES}
    return BO_RUNS, POLICIES, run_bo


@app.cell(hide_code=True)
def _(POLICIES, mo):
    bo_policy = mo.ui.radio(POLICIES, value="EI", label="policy", inline=True)
    bo_step = mo.ui.slider(0, 11, step=1, value=0, label="iteration", show_value=True)
    return bo_policy, bo_step


@app.cell(hide_code=True)
def _(
    BO_RUNS,
    C,
    F_STAR,
    F_TRUE,
    XS,
    bo_policy,
    bo_step,
    mo,
    objective,
    plot_posterior,
    plt,
    show,
):
    _Xf, _yf, _hist = BO_RUNS[bo_policy.value]
    _h = _hist[bo_step.value]
    _xs = XS[::2]
    _fig, (_a1, _a2) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    plot_posterior(_a1, _xs, _h["mu"], _h["sd"], _h["X"], _h["y"], F_TRUE[::2])
    _a1.scatter([_h["x_next"]], [objective(_h["x_next"])], marker="*", s=160, color=C["orange"], zorder=7,
                edgecolor=C["ink"], linewidth=0.6, label="next query")
    _gap = F_STAR - _h["y"].max()
    _a1.set(title=rf"{bo_policy.value}, iteration {bo_step.value + 1}: $n = {len(_h['X'])}$, "
                  rf"simple regret $f^* - \max\, y = {_gap:.3f}$", ylim=(-2.6, 1.8))
    _a1.legend(loc="lower left", ncols=5)
    _a2.fill_between(_xs, _h["acq"].min(), _h["acq"], color=C["orange"], alpha=0.15, lw=0)
    _a2.plot(_xs, _h["acq"], color=C["orange"])
    _a2.axvline(_h["x_next"], color=C["ink"], ls="--", lw=1)
    _a2.set(ylabel=r"$\alpha(x)$", xlabel=r"$x$", yticks=[])
    mo.vstack([mo.hstack([bo_policy, bo_step], justify="start"), show(_fig)])
    return


@app.cell(hide_code=True)
def _(C, F_STAR, SERIES, np, plt, run_bo, show):
    _pols = ["EI", "PI", "UCB", "Thompson", "Random"]
    _seeds = range(16)
    _fig, _ax = plt.subplots(figsize=(9, 3.4))
    for _i, _p in enumerate(_pols):
        _R = []
        for _s in _seeds:
            _X, _y, _ = run_bo(_p, seed=100 + _s, n_iter=15, keep_history=False)
            _R.append(np.maximum(F_STAR - np.maximum.accumulate(_y), 1e-4))
        _R = np.log10(np.array(_R))
        _t = np.arange(1, _R.shape[1] + 1)
        _col = C["ink2"] if _p == "Random" else SERIES[_i]
        _ax.fill_between(_t, np.percentile(_R, 25, 0), np.percentile(_R, 75, 0), color=_col, alpha=0.1, lw=0)
        _ax.plot(_t, np.median(_R, 0), color=_col, ls="--" if _p == "Random" else "-", label=_p)
    _ax.axvline(3, color=C["ink2"], ls=":", lw=1)
    _ax.set(xlabel="evaluations (first 3 = random initial design)", ylabel=r"$\log_{10}$ simple regret $r_T$",
            title=r"Median (IQR band) simple regret over 16 seeds, 1-D test function (floored at $10^{-4}$)", xlim=(1, 18))
    _ax.legend(loc="lower left", ncols=5)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Which acquisition should you use?** (from the doc)

    | Situation | Good default | Why |
    | :--- | :--- | :--- |
    | Low noise, sequential, $d \lesssim 20$ | LogEI | Robust, no tuning, easy inner optimization |
    | Noisy observations | Noisy EI (LogNEI) or KG | Handle an uncertain incumbent properly |
    | Large batches or high $d$ | Thompson sampling, often in a trust region | Cheap, parallel, diverse |
    | Multi-fidelity or cost-aware | KG or MES variants | Value information without needing a candidate |
    | You want theory to match practice | GP-UCB | Cleanest regret guarantees |

    Information-theoretic methods target the optimum itself: **ES** the location $p(x^*\mid D)$, **PES** swaps
    roles via mutual-information symmetry, and **MES** targets the scalar $f^*$, whose truncated-Gaussian entropy is
    closed form, making it the practical one. **KG** values how much an observation raises $\max_x \mu_{n+1}$, so it
    can value points that will never be the best.

    ## 4. Theory: what the guarantees actually say

    Bounds are on **regret**, simple $r_T = f(x^*) - f(\hat x_T)$ or cumulative $R_T = \sum_t f(x^*) - f(x_t)$.
    The key quantity is the **maximum information gain**
    $\gamma_T = \max_{|A|=T} \tfrac12 \log|I + \sigma_n^{-2}K_A|$, which measures kernel complexity. GP-UCB achieves
    $R_T = O^*(\sqrt{T \beta_T \gamma_T})$ (Srinivas et al. 2010), sublinear whenever $\gamma_T$ is.

    Below, $\gamma_T$ is computed greedily (a $(1-1/e)$-approximation: always add the point of highest posterior
    variance). Rougher kernels have larger $\gamma_T$, and the Matérn exponent $d/(2\nu+d)$ tends to 1 with dimension,
    which is the curse of dimensionality in one line.
    """)
    return


@app.cell(hide_code=True)
def _(SERIES, k_m12, k_m32, k_m52, k_se, np, plt, show):
    def _greedy_gain(kern, T=60, noise=0.01):
        xs = np.linspace(0, 1, 400)
        K = kern(xs, xs)
        var = np.diag(K).copy()
        V = np.zeros((0, len(xs)))
        gains = []
        for _ in range(T):
            i = int(np.argmax(var))
            gains.append(0.5 * np.log1p(var[i] / noise))
            v = (K[i] - V[:, i] @ V) / np.sqrt(var[i] + noise)
            V = np.vstack([V, v])
            var = np.maximum(var - v**2, 0)
        return np.cumsum(gains)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for _i, (_name, _k) in enumerate([("Matérn 1/2", k_m12), ("Matérn 3/2", k_m32), ("Matérn 5/2", k_m52),
                                      ("Squared exp.", k_se)]):
        _ax1.plot(np.arange(1, 61), _greedy_gain(lambda a, b, k=_k: k(a, b, ls=0.1)), color=SERIES[_i], label=_name)
    _ax1.set(xlabel=r"$T$", ylabel=r"$\gamma_T$ (nats)", title=r"Greedy information gain, $d = 1$, $\ell = 0.1$")
    _ax1.legend()

    _d = np.arange(1, 51)
    for _i, _nu in enumerate([0.5, 1.5, 2.5]):
        _ax2.plot(_d, _d / (2 * _nu + _d), color=SERIES[_i], label=rf"Matérn $\nu = {_nu}$")
    _ax2.axhline(1, color="#52514e", ls=":", lw=1)
    _ax2.set(xlabel=r"dimension $d$", ylabel=r"exponent $d/(2\nu+d)$", ylim=(0, 1.05),
             title=r"$\gamma_T = \tilde{O}(T^{d/(2\nu+d)})$: exponent $\to 1$ (near-linear) as $d$ grows")
    _ax2.legend(loc="lower right")
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Kernel | Growth of $\gamma_T$ |
    | :--- | :--- |
    | Linear | $O(d \log T)$ |
    | Squared exponential | $O((\log T)^{d+1})$ |
    | Matérn-$\nu$ | $\tilde O(T^{d/(2\nu + d)})$ |

    The theory explains why smoother kernels learn faster, why dimension hurts without structure, and why exploration
    must never be switched off. It does **not** tell you what $\beta$ to use (theoretical schedules over-explore), and
    it almost always assumes known hyperparameters, while real BO learns them from the same few points.

    ## 5. Making BO work in practice

    ### Hyperparameters are the silent failure mode

    With very few points, the ML lengthscale can be badly wrong. Snoek et al. (2012) **integrate the acquisition over
    the hyperparameter posterior**, $\hat\alpha(x) = \int \alpha(x;\theta)\,p(\theta\mid D)\,d\theta$. The cheaper fix
    is good priors (a log-normal lengthscale prior scaled with $\log d$, now BoTorch's default). Always standardize
    outputs and normalize inputs to the unit cube.
    """)
    return


@app.cell(hide_code=True)
def _(
    C,
    XS,
    acq_ei,
    gp_posterior,
    k_m52,
    log_ml_terms,
    np,
    objective,
    plt,
    show,
):
    _X = np.array([0.1, 0.45, 0.62])
    _y = objective(_X)
    _ym, _ys = _y.mean(), _y.std()
    _yz = (_y - _ym) / _ys
    _lg = np.logspace(-2, 0.5, 120)
    _loglik = np.array([sum(log_ml_terms(_X, _yz, lambda a, b, l=l: k_m52(a, b, ls=l), 1e-6)) for l in _lg])
    _logprior = -0.5 * ((np.log(_lg) - np.log(0.2)) / 0.75) ** 2  # log-normal prior on ℓ
    _lp = _loglik + _logprior
    _post = np.exp(_lp - _lp.max())
    _post /= _post.sum()
    _ml = _lg[_loglik.argmax()]

    _xs = XS[::2]
    def _ei_at(l):
        mu, var = gp_posterior(_X, _yz, _xs, lambda a, b: k_m52(a, b, ls=l), 1e-6)
        return acq_ei(mu, np.sqrt(var), _yz.max())
    _ei_ml = _ei_at(_ml)
    _ei_bayes = sum(w * _ei_at(l) for w, l in zip(_post, _lg) if w > 1e-4)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    _ax1.plot(_lg, _post / _post.max(), color=C["blue"], label=r"posterior $p(\ell \mid D)$ (scaled)")
    _ax1.plot(_lg, np.exp(_loglik - _loglik.max()), color=C["orange"], label="likelihood (scaled)")
    _ax1.axvline(_ml, color=C["orange"], ls=":", lw=1)
    _ax1.set(xscale="log", xlabel=r"lengthscale $\ell$", title=rf"3 noiseless points: ML runs to the grid edge, $\ell = {_ml:.2f}$")
    _ax1.legend(loc="center right")
    _ax2.plot(_xs, _ei_ml / _ei_ml.max(), color=C["orange"], label=r"EI at ML $\ell$")
    _ax2.plot(_xs, _ei_bayes / _ei_bayes.max(), color=C["blue"], label=r"EI integrated over $p(\ell \mid D)$")
    for _x in _X:
        _ax2.axvline(_x, color=C["ink"], lw=0.8, ls=":")
    _ax2.set(xlabel=r"$x$", ylabel="EI (normalized)", title=r"Marginalizing $\theta$ changes where you look next", ylim=(0, 1.3))
    _ax2.legend(loc="upper left", ncols=2)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Monte Carlo acquisitions and batch BO

    Batch acquisitions like $q\mathrm{EI}(X) = \mathbb E[\max(\max_j f(x_j) - f^+, 0)]$ have no closed form, so we
    estimate them with the reparameterization $f(X) = \mu(X) + L(X)\varepsilon$. BoTorch's trick is to **fix the
    base samples** $\varepsilon_i$ (often quasi-MC) during inner optimization: the estimate becomes a smooth
    deterministic function of $X$ (sample average approximation), and L-BFGS-B just works.

    For batches, **fantasies** exploit the Section 1 fact: posterior variance does not depend on observed values,
    so conditioning on a fake $y$ at a chosen point correctly shrinks uncertainty there (kriging believer).
    """)
    return


@app.cell(hide_code=True)
def _(
    C,
    SERIES,
    XS,
    acq_ei,
    cholesky,
    gp_posterior,
    k_m52,
    np,
    objective,
    plt,
    qmc,
    show,
):
    _X = np.array([0.05, 0.3, 0.5, 0.95])
    _y = objective(_X)
    _ym, _ys = _y.mean(), _y.std()
    _yz = (_y - _ym) / _ys
    _kern = lambda a, b: k_m52(a, b, ls=0.15)
    _xs = XS[::2]
    _best = _yz.max()

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.5))
    # (a) fantasies: same σ, different μ
    _xb = 0.75
    _mu0, _v0 = gp_posterior(_X, _yz, _xs, _kern)
    _axes[0].plot(_xs, np.sqrt(_v0), color=C["ink2"], ls="--", lw=1.2, label=r"$\sigma$ before")
    _sds = []
    for _i, _fy in enumerate([-1.5, 0.0, 1.5]):
        _mu, _v = gp_posterior(np.append(_X, _xb), np.append(_yz, _fy), _xs, _kern)
        _axes[0].plot(_xs, _mu, color=SERIES[_i], lw=1.1, label=rf"$\mu \mid$ fantasy $y = {_fy:+.1f}$")
        _sds.append(np.sqrt(_v))
    assert np.allclose(_sds[0], _sds[2])
    _axes[0].plot(_xs, _sds[0], color=C["violet"], lw=2.4, label=r"$\sigma$ after (identical for every $y$)")
    _axes[0].axvline(_xb, color=C["ink"], lw=0.8, ls=":")
    _axes[0].set(title=r"Fantasy at $x = 0.75$: three means, one $\sigma$", xlabel=r"$x$")
    _axes[0].legend(fontsize=7, loc="lower left")

    # (b) kriging believer batch of q = 4 points by EI
    _Xb, _yb = _X.copy(), _yz.copy()
    _mu, _v = gp_posterior(_Xb, _yb, _xs, _kern)
    _axes[1].plot(_xs, acq_ei(_mu, np.sqrt(_v), _best) / acq_ei(_mu, np.sqrt(_v), _best).max(),
                  color=C["ink2"], lw=1.2, ls="--", label="EI before batch")
    for _j in range(4):
        _mu, _v = gp_posterior(_Xb, _yb, _xs, _kern)
        _e = acq_ei(_mu, np.sqrt(_v), _best)
        _i = int(np.argmax(_e))
        _axes[1].plot(_xs[_i], 1.05, marker="v", color=C["orange"], ms=9)
        _axes[1].annotate(str(_j + 1), (_xs[_i], 1.1), ha="center", color=C["ink"], fontsize=8)
        _Xb, _yb = np.append(_Xb, _xs[_i]), np.append(_yb, _mu[_i])  # believe the mean
    for _x in _X:
        _axes[1].axvline(_x, color=C["ink"], lw=0.8, ls=":")
    _axes[1].set(title="Kriging believer: batch of 4 (▼ in pick order)", xlabel=r"$x$", ylim=(0, 1.2), yticks=[])
    _axes[1].legend(loc="center right")

    # (c) qEI with q = 2: fresh iid samples vs fixed QMC base samples
    _x1 = _xs[int(np.argmax(acq_ei(_mu0, np.sqrt(_v0), _best)))]
    def _qei(x2, eps):
        mu, cov = gp_posterior(_X, _yz, np.array([_x1, x2]), _kern, full_cov=True)
        L = cholesky(cov + 1e-9 * np.eye(2), lower=True)
        f = mu + eps @ L.T
        return np.maximum(f.max(1) - _best, 0).mean()
    _grid = np.linspace(0, 1, 160)
    _rng = np.random.default_rng(0)
    _base = qmc.MultivariateNormalQMC(np.zeros(2), seed=1).random(64)
    _ref = qmc.MultivariateNormalQMC(np.zeros(2), seed=2).random(8192)
    _axes[2].plot(_grid, [_qei(g, _rng.standard_normal((64, 2))) for g in _grid], color=C["orange"], lw=1,
                  label=r"$N=64$, fresh iid each $x_2$")
    _axes[2].plot(_grid, [_qei(g, _base) for g in _grid], color=C["blue"], lw=2, label=r"$N=64$, fixed QMC base")
    _axes[2].plot(_grid, [_qei(g, _ref) for g in _grid], color=C["ink2"], ls="--", lw=1.2, label=r"reference ($N=8192$)")
    _axes[2].axvline(_x1, color=C["ink"], lw=0.8, ls=":")
    _axes[2].set(title=rf"$q\mathrm{{EI}}(x_1 = {_x1:.2f}, x_2)$ estimate", xlabel=r"$x_2$")
    _axes[2].set_ylim(top=_axes[2].get_ylim()[1] + 0.08)
    _axes[2].legend(fontsize=7, loc="upper center", ncols=2)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Other batch strategies: **local penalization** (González et al. 2016), **joint MC** optimization of
    qEI/qNEI/qKG, and **Thompson sampling** with $q$ independent samples (scales to hundreds). In practice, use
    [BoTorch](https://botorch.org/)/[Ax](https://ax.dev/) for research and production, GPyTorch for scalable GPs,
    SMAC3 for mixed/conditional spaces, and Optuna for everyday tuning.

    ## 6. Scaling Gaussian processes

    Exact GPs cost $O(n^3)$ time and $O(n^2)$ memory. Three families of fixes:

    - **Inducing points / sparse variational GPs**: summarize $f$ through $u = f(Z)$ at $m \ll n$ points.
      Titsias (2009) maximizes
      $\log\mathcal N(\mathbf y\mid 0, Q_{nn} + \sigma_n^2 I) - \tfrac{1}{2\sigma_n^2}\mathrm{tr}(K_{nn} - Q_{nn})$,
      $O(nm^2)$; SVGP (Hensman et al. 2013) makes it minibatchable. Weakness for BO: it smooths over detail.
    - **Structured kernels + iterative solvers**: KISS-GP, GPyTorch's conjugate gradients on GPUs, exact GPs on a
      million points.
    - **Random Fourier features** (Rahimi & Recht 2007) and **pathwise conditioning** (Wilson et al. 2020) for cheap,
      accurate function samples.

    Drag the number of inducing points: with too few, the sparse GP misses the sharp peak BO cares about most.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sparse_m = mo.ui.slider(2, 40, step=1, value=6, label="inducing points m", show_value=True)
    return (sparse_m,)


@app.cell(hide_code=True)
def _(
    C,
    XS,
    cho_solve,
    cholesky,
    gp_posterior,
    k_se,
    mo,
    np,
    objective,
    plot_posterior,
    plt,
    show,
    sparse_m,
):
    _rng = np.random.default_rng(4)
    _n, _sn2 = 300, 0.1**2
    _X = np.sort(_rng.uniform(0, 1, _n))
    _y = objective(_X) + 0.1 * _rng.standard_normal(_n)
    _kern = lambda a, b: k_se(a, b, ls=0.07, sf2=1.0)
    _xs = XS[::2]
    _Z = np.linspace(0, 1, sparse_m.value)

    # Titsias SGPR predictive: Σ = (Kmm + σ⁻² Kmn Knm)⁻¹,  μ* = σ⁻² K*m Σ Kmn y,  var* = k** − Q** + K*m Σ Km*
    _Kmm = _kern(_Z, _Z) + 1e-8 * np.eye(len(_Z))
    _Kmn = _kern(_Z, _X)
    _Ksm = _kern(_xs, _Z)
    _A = _Kmm + _Kmn @ _Kmn.T / _sn2
    _La = cholesky(_A, lower=True)
    _Lm = cholesky(_Kmm, lower=True)
    _mu_s = _Ksm @ cho_solve((_La, True), _Kmn @ _y) / _sn2
    _var_s = (1.0 - (_Ksm * cho_solve((_Lm, True), _Ksm.T).T).sum(1) + (_Ksm * cho_solve((_La, True), _Ksm.T).T).sum(1))
    _mu_e, _v_e = gp_posterior(_X, _y, _xs, _kern, _sn2)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    _ax1.scatter(_X, _y, s=4, color=C["ink2"], alpha=0.35, lw=0)
    plot_posterior(_ax1, _xs, _mu_e, np.sqrt(_v_e), f_true=None)
    _ax1.set(title=rf"Exact GP, $n = {_n}$  ($O(n^3)$)", xlabel=r"$x$", ylim=(-2.8, 1.8))
    _ax2.scatter(_X, _y, s=4, color=C["ink2"], alpha=0.35, lw=0)
    plot_posterior(_ax2, _xs, _mu_s, np.sqrt(np.maximum(_var_s, 1e-12)), color=C["violet"], label="sparse mean")
    _ax2.plot(_Z, np.full_like(_Z, -2.6), "|", ms=10, color=C["orange"], mew=2, label=r"inducing inputs $Z$")
    _ax2.set(title=rf"Sparse variational GP, $m = {sparse_m.value}$  ($O(nm^2)$)", xlabel=r"$x$")
    _ax2.set_ylim(-2.8, 2.6)
    _ax2.legend(loc="upper center", ncols=3, fontsize=7)
    mo.vstack([sparse_m, show(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Random features, variance starvation and pathwise conditioning

    Sampling frequencies from the spectral density gives $k(x,x') \approx \phi(x)^\top\phi(x')$ with
    $\phi(x) = \sqrt{2/D}\cos(\Omega x + b)$: the GP becomes Bayesian linear regression, and a posterior sample is an
    explicit function you can maximize (what Thompson sampling needs). The catch is **variance starvation**: with many
    data points and few features the posterior becomes overconfident away from the data. **Matheron's rule** fixes it:

    $$
    (f \mid \mathbf{y})(\cdot) \overset{d}{=} f(\cdot) + k(\cdot, X)(K + \sigma_n^2 I)^{-1}\big(\mathbf{y} - f(X) - \boldsymbol\varepsilon\big)
    $$

    with the prior sample $f$ from random features and the correction from the exact kernel.
    """)
    return


@app.cell(hide_code=True)
def _(
    C,
    SERIES,
    XS,
    cho_solve,
    cholesky,
    gp_posterior,
    k_se,
    np,
    objective,
    plt,
    show,
):
    _ell = 0.08
    _rng = np.random.default_rng(0)

    def _features(x, W, b):
        return np.sqrt(2 / len(W)) * np.cos(np.outer(x, W) + b)

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.4))
    # (a) kernel approximation quality
    _r = np.linspace(0, 0.4, 200)
    _axes[0].plot(_r, np.exp(-0.5 * (_r / _ell) ** 2), color=C["ink"], lw=2.2, label="exact SE")
    for _i, _D in enumerate([10, 100, 1000]):
        _W, _b = _rng.normal(0, 1 / _ell, _D), _rng.uniform(0, 2 * np.pi, _D)
        _axes[0].plot(_r, (_features(np.zeros(1), _W, _b) @ _features(_r, _W, _b).T).ravel(),
                      color=SERIES[_i], lw=1.1, label=rf"RFF, $D = {_D}$")
    _axes[0].set(xlabel=r"$r$", title=r"Random Fourier features: $\phi(x)^\top\phi(x')$ vs $k(r)$", ylim=(-0.4, 1.1))
    _axes[0].legend()

    # (b, c) many points with a gap in (0.4, 0.75); RFF-only vs pathwise posterior samples
    _n, _sn = 1000, 0.1
    _X = np.sort(np.r_[_rng.uniform(0, 0.4, _n // 2), _rng.uniform(0.75, 1.0, _n // 2)])
    _y = objective(_X) + _sn * _rng.standard_normal(_n)
    _xs = XS[::2]
    _kern = lambda a, b: k_se(a, b, ls=_ell)
    _mu, _var = gp_posterior(_X, _y, _xs, _kern, _sn**2)
    _D = 20
    _K = _kern(_X, _X) + _sn**2 * np.eye(_n)
    _L = cholesky(_K, lower=True)
    _Kxs = _kern(_xs, _X)
    for _ax, _mode in zip(_axes[1:], [r"RFF-only posterior ($D = 20$)", "Pathwise: RFF prior + exact update"]):
        _ax.fill_between(_xs, _mu - 2 * np.sqrt(_var), _mu + 2 * np.sqrt(_var), color=C["blue"], alpha=0.18, lw=0,
                         label=r"exact $\pm 2\sigma$")
        _ax.scatter(_X, _y, s=3, color=C["ink2"], alpha=0.3, lw=0)
        _W, _b = _rng.normal(0, 1 / _ell, _D), _rng.uniform(0, 2 * np.pi, _D)
        _P, _Ps = _features(_X, _W, _b), _features(_xs, _W, _b)
        for _s in range(5):
            if _mode.startswith("RFF"):
                _A = _P.T @ _P / _sn**2 + np.eye(_D)
                _La = cholesky(_A, lower=True)
                _wm = cho_solve((_La, True), _P.T @ _y / _sn**2)
                _w = _wm + np.linalg.solve(_La.T, _rng.standard_normal(_D))
                _f = _Ps @ _w
            else:
                _w = _rng.standard_normal(_D)
                _f = _Ps @ _w + _Kxs @ cho_solve((_L, True), _y - _P @ _w - _sn * _rng.standard_normal(_n))
            _ax.plot(_xs, _f, color=C["orange"], lw=0.9, alpha=0.85, label="posterior samples" if _s == 0 else None)
        _ax.axvspan(0.4, 0.75, color=C["ink2"], alpha=0.05, lw=0)
        _ax.set(title=_mode, xlabel=r"$x$", ylim=(-3.5, 3.5))
        _ax.legend(loc="lower left", fontsize=7)
    _axes[1].annotate("gap in the data:\nsamples too confident", (0.575, 2.6), ha="center", color=C["ink2"], fontsize=8)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. High-dimensional BO

    Folk wisdom said GP-BO stops working beyond ~20 dimensions. Four problems compound: **distances concentrate**, so
    a stationary kernel sees every pair of points as equally far; covering the space needs exponentially many points;
    ARD must learn $d$ lengthscales from few data; and the acquisition optimization itself becomes hard.

    The field responded with structure:

    - **Embeddings** (REMBO, HeSBO, ALEBO, BAxUS): assume $f(x) = g(P^\top x)$ with low effective dimension.
    - **Additivity** (Kandasamy et al. 2015): additive kernels make regret scale linearly in $d$.
    - **Trust regions** (TuRBO, SCBO): a local GP around the incumbent, expanded on success, shrunk on failure.
    - **Sparsity priors** (SAASBO): half-Cauchy priors on inverse squared lengthscales switch most dimensions off.

    **The 2024 surprise** (Hvarfner, Hellsten & Nardi): the real culprit was the lengthscale prior. With $\ell \sim 1$
    in a $d$-dimensional unit cube, every pair of points looks independent (middle panel). Scaling the prior location
    like $\sqrt d$ restores correlation, and plain GP-BO then matches the specialized methods.
    """)
    return


@app.cell(hide_code=True)
def _(C, SERIES, np, plt, show):
    _rng = np.random.default_rng(0)
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.4))
    _ds = [1, 10, 100, 1000]
    _med_fixed, _med_scaled, _dgrid = [], [], np.unique(np.logspace(0, 3, 25).astype(int))
    for _i, _d in enumerate(_ds):
        _P = _rng.uniform(0, 1, (300, _d))
        _D = np.sqrt(((_P[:150] - _P[150:]) ** 2).sum(1))
        _axes[0].hist(_D / _D.mean(), bins=np.linspace(0, 2.5, 50), histtype="step", lw=1.6, color=SERIES[_i],
                      density=True, label=rf"$d = {_d}$")
    _axes[0].set(xlabel="pairwise distance / mean", title=r"Distances concentrate as $d$ grows", yticks=[])
    _axes[0].legend()

    for _d in _dgrid:
        _P = _rng.uniform(0, 1, (400, _d))
        _r2 = ((_P[:200] - _P[200:]) ** 2).sum(1)
        _med_fixed.append(np.median(np.exp(-0.5 * _r2 / 0.5**2)))
        _med_scaled.append(np.median(np.exp(-0.5 * _r2 / (0.5 * np.sqrt(_d)) ** 2)))
    _axes[1].plot(_dgrid, _med_fixed, color=C["orange"], label=r"$\ell = 0.5$ (fixed)")
    _axes[1].plot(_dgrid, _med_scaled, color=C["blue"], label=r"$\ell = 0.5\sqrt{d}$ (dimension-scaled)")
    _axes[1].set(xscale="log", xlabel=r"dimension $d$", ylabel=r"median $k(x, x')$",
                 title="Prior correlation of random point pairs")
    _axes[1].legend(loc="center right")

    _D = 30
    for _i in range(3):
        _tau = np.abs(0.1 * np.tan(np.pi * (_rng.uniform() - 0.5)))
        _rho = np.abs(_tau * np.tan(np.pi * (_rng.uniform(size=_D) - 0.5)))
        _axes[2].vlines(np.arange(_D) + 0.25 * (_i - 1), 0, _rho / _rho.max(), color=SERIES[_i], lw=2,
                        label=f"prior draw {_i + 1}")
    _axes[2].set(xlabel=r"input dimension $d$", ylabel=r"$\rho_d / \max\,\rho$,   $\rho_d = 1/\ell_d^2$", ylim=(0, 1.3),
                 title="SAAS prior: most dimensions ≈ off, a few escape")
    _axes[2].legend(loc="upper center", ncols=3, fontsize=7)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Beyond vanilla BO

    The recipe is always the same: **model each unknown quantity with its own GP, then redefine the acquisition** to
    value what you actually care about.

    - **Constraints**: $\alpha_{\text{cEI}}(x) = \alpha_{\text{EI}}(x)\prod_k P(c_k(x) \le 0)$ (Gardner et al. 2014;
      Gelbart et al. 2014).
    - **Multi-fidelity / cost-aware**: add a fidelity input $s$ and model $f(x, s)$ (MF-GP-UCB, taKG, MF-MES), or use
      bandit-style early stopping (Hyperband, BOHB).
    - **Multiple objectives**: maximize the **hypervolume** dominated by the Pareto front (qEHVI, qNEHVI, ParEGO).
    - **Other spaces**: categorical/mixed kernels, graph kernels, or BO in a generative model's latent space.
    - **Non-GP surrogates**: random forests (SMAC), TPE (Optuna), DNGO, BNNs, deep kernels, and PFNs that do
      Bayesian inference in-context with a transformer.
    """)
    return


@app.cell(hide_code=True)
def _(
    C,
    F_TRUE,
    XS,
    acq_ei,
    gp_posterior,
    k_m52,
    norm,
    np,
    objective,
    plot_posterior,
    plt,
    show,
):
    def _constraint(x):
        return np.sin(9 * x) + 0.2  # feasible where c(x) ≤ 0

    _X = np.array([0.08, 0.3, 0.52, 0.9])
    _y, _c = objective(_X), _constraint(_X)
    _xs = XS[::2]
    _kern = lambda a, b: k_m52(a, b, ls=0.18)
    _ym, _ys = _y.mean(), _y.std()
    _mu, _v = gp_posterior(_X, (_y - _ym) / _ys, _xs, _kern)
    _muc, _vc = gp_posterior(_X, _c, _xs, _kern)
    _feas = _c <= 0
    _best = ((_y[_feas] - _ym) / _ys).max()
    _ei = acq_ei(_mu, np.sqrt(_v), _best)
    _pf = norm.cdf(-_muc / np.sqrt(_vc))

    _fig = plt.figure(figsize=(12, 4.6))
    _gs = _fig.add_gridspec(3, 2, width_ratios=[1.4, 1])
    _a0 = _fig.add_subplot(_gs[0, 0])
    plot_posterior(_a0, _xs, _mu * _ys + _ym, np.sqrt(_v) * _ys, _X, _y, F_TRUE[::2])
    _infeas = _constraint(_xs) > 0
    _a0.fill_between(_xs, -2.6, 1.6, where=_infeas, color=C["red"], alpha=0.07, lw=0, label="infeasible (truth)")
    _a0.set(title=r"Objective GP (shaded: true $c(x) > 0$)", ylim=(-2.6, 1.6))
    _a0.legend(loc="lower left", ncols=3, fontsize=7)
    _a0.tick_params(labelbottom=False)
    _a1 = _fig.add_subplot(_gs[1, 0], sharex=_a0)
    _a1.plot(_xs, _ei / _ei.max(), color=C["orange"], label="EI (normalized)")
    _a1.plot(_xs, _pf, color=C["aqua"], label=r"$P(c(x) \leq 0)$")
    _a1.legend(loc="upper left", fontsize=7)
    _a1.tick_params(labelbottom=False)
    _a2 = _fig.add_subplot(_gs[2, 0], sharex=_a0)
    _cei = _ei * _pf
    _a2.fill_between(_xs, 0, _cei / _cei.max(), color=C["violet"], alpha=0.15, lw=0)
    _a2.plot(_xs, _cei / _cei.max(), color=C["violet"], label=r"$\alpha_{\mathrm{cEI}} = \alpha_{\mathrm{EI}} \cdot P(c \leq 0)$")
    _a2.axvline(_xs[_cei.argmax()], color=C["ink"], ls="--", lw=1)
    _a2.set(xlabel=r"$x$")
    _a2.legend(loc="upper left", fontsize=7)

    # Multi-objective: Pareto front and dominated hypervolume (maximize both)
    _rng = np.random.default_rng(3)
    _t = _rng.uniform(0, 1, 40)
    _o1 = _t + 0.08 * _rng.standard_normal(40)
    _o2 = 1 - _t**0.6 + 0.12 * _rng.uniform(-1, 0, 40) - 0.25 * _rng.uniform(0, 1, 40) ** 3
    _P = np.c_[_o1, _o2]
    _dom = np.array([np.any(np.all(_P >= p, 1) & np.any(_P > p, 1)) for p in _P])
    _front = _P[~_dom]
    _front = _front[np.argsort(-_front[:, 0])]
    _ref = np.array([_P[:, 0].min() - 0.05, _P[:, 1].min() - 0.05])
    _am = _fig.add_subplot(_gs[:, 1])
    # _front is sorted by objective 1 descending (so objective 2 ascending)
    _hv = sum((_front[i, 0] - _ref[0]) * (_front[i, 1] - (_front[i - 1, 1] if i else _ref[1])) for i in range(len(_front)))
    _asc = _front[::-1]
    _sx, _sy = [_ref[0]], [_asc[0, 1]]
    for _i, _p in enumerate(_asc):
        _sx += [_p[0], _p[0]]
        _sy += [_p[1], _asc[_i + 1, 1] if _i + 1 < len(_asc) else _ref[1]]
    _am.fill(np.r_[_ref[0], _sx], np.r_[_ref[1], _sy], color=C["blue"], alpha=0.12, lw=0, label="dominated hypervolume")
    _am.plot(_sx, _sy, color=C["blue"], lw=1.2)
    _am.scatter(*_P[_dom].T, s=16, color=C["ink2"], alpha=0.6, lw=0, label="dominated")
    _am.scatter(*_front.T, s=36, color=C["orange"], zorder=5, edgecolor=C["ink"], lw=0.6, label="Pareto front")
    _am.scatter(*_ref, marker="x", color=C["ink"], s=40, label="reference point")
    _am.set(xlabel=r"objective $f_1$ (maximize)", ylabel=r"objective $f_2$ (maximize)", title=rf"Hypervolume $= {_hv:.3f}$")
    _am.legend(loc="upper right", fontsize=7)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Takeaways and open problems

    **The practical recipe (2026)**

    1. Normalize inputs to $[0,1]^d$, standardize outputs.
    2. GP with **Matérn-5/2 + ARD** and a **dimension-scaled lengthscale prior**; consider fully Bayesian
       hyperparameters when data is scarce.
    3. **LogEI / LogNEI** for sequential or noisy problems; **Thompson sampling** (pathwise) or joint MC acquisitions
       for batches; trust regions for large budgets in high $d$.
    4. Take the inner optimization seriously: multi-start L-BFGS-B with fixed QMC base samples.
    5. Use BoTorch/Ax rather than writing it yourself.

    **Open problems**: robustness to misspecified kernels and hyperparameters (with guarantees); honest
    benchmarking (LogEI and vanilla-BO showed baselines are often mis-set); non-myopic BO; when to stop; injecting
    expert and LLM priors (πBO, PFNs, LLAMBO); and one framework that works for both a few dozen and millions of
    evaluations.

    **Read next**: [Garnett, *Bayesian Optimization*](https://bayesoptbook.com/) (2023, free online),
    [Frazier's tutorial](https://arxiv.org/abs/1807.02811), [Rasmussen & Williams](http://gaussianprocess.org/gpml/),
    and the Distill articles on [GPs](https://distill.pub/2019/visual-exploration-gaussian-processes/) and
    [BO](https://distill.pub/2020/bayesian-optimization/). Full references are in `docs/gp-and-bo.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
