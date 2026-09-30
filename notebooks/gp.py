import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="medium",
    app_title="Gaussian Processes",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import cho_solve, solve_triangular
    from scipy.optimize import minimize
    from scipy.stats import norm

    def cholesky(A, lower=True):
        """Lower Cholesky factor with a clean zero upper triangle (safe to multiply by directly)."""
        return np.linalg.cholesky(A)

    return cho_solve, cholesky, minimize, mo, norm, np, plt, solve_triangular


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gaussian Processes

    *An interactive companion to the GP sections of `docs/gp-and-bo.md`, plus a section on how GPs fail.*

    A Gaussian process (GP) is a **probability distribution over functions**. You state what you believe about an
    unknown function $f$ before seeing data (smoothness, typical size, how far information travels), observe a few
    noisy values, and get back a full posterior: a best guess **and** an honest error bar everywhere.

    Observations are $y = f(x) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$, and the data are
    $D = \{(x_i, y_i)\}_{i=1}^n$, stacked into $X$ and $\mathbf{y}$.

    The workflow this notebook follows:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        flowchart LR
            A[Prior<br/>mean + kernel] --> B[Fit hyperparameters<br/>marginal likelihood]
            B --> C[Condition on data<br/>posterior]
            C --> D[Predict and sample]
            D --> E{Diagnostics OK?<br/>LOO, failure modes}
            E -- no --> A
            E -- yes --> F[Use the model]
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
            "font.weight": "normal",  # override a global matplotlibrc that may request thin weights
            "mathtext.fontset": "cm",  # LaTeX-looking math in figure text ($...$)
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
def _(cho_solve, cholesky, np, solve_triangular):
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
        """Posterior mean and variance (or covariance) of f via one Cholesky factorization.

        `noise` is a scalar variance or a per-point vector of variances.
        """
        Xs = np.asarray(Xs, float)
        if X is None or len(X) == 0:
            cov = kern(Xs, Xs)
            return np.zeros(len(Xs)), (cov if full_cov else np.diag(cov).copy())
        K = kern(X, X) + np.diag(np.broadcast_to(noise + 1e-9, (len(X),)))
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

    def log_ml(X, y, kern, noise):
        return sum(log_ml_terms(X, y, kern, noise))

    def fit_lengthscale(X, y, base_kern, noise, grid=np.logspace(-2, 0, 30)):
        """Type-II ML over a lengthscale grid (cheap, robust stand-in for L-BFGS with restarts)."""
        scores = [log_ml(X, y, lambda a, b, l=l: base_kern(a, b, ls=l), noise) for l in grid]
        return grid[int(np.argmax(scores))]

    def loo(X, y, kern, noise):
        """Closed-form leave-one-out predictive mean and variance of each y_i."""
        K = kern(X, X) + (noise + 1e-9) * np.eye(len(X))
        Kinv = cho_solve((cholesky(K, lower=True), True), np.eye(len(X)))
        d = np.diag(Kinv)
        return y - (Kinv @ y) / d, 1.0 / d

    def sample_paths(mu, cov, n, rng):
        return rng.multivariate_normal(mu, cov + 1e-9 * np.eye(len(mu)), size=n, method="eigh")

    # ---------- the running 1D test function (rescaled Forrester) ----------
    def f_test(x):
        x = np.asarray(x, float)
        return -((6 * x - 2) ** 2 * np.sin(12 * x - 4)) / 6.02

    XS = np.linspace(0, 1, 400)
    F_TRUE = f_test(XS)
    return (
        F_TRUE,
        KERNELS,
        XS,
        f_test,
        fit_lengthscale,
        gp_posterior,
        k_linear,
        k_m12,
        k_m32,
        k_m52,
        k_periodic,
        k_se,
        log_ml,
        log_ml_terms,
        loo,
        sample_paths,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Everything starts with Gaussian conditioning

    The whole GP machinery rests on one fact. If two blocks of variables are jointly Gaussian, then the conditional
    of one given the other is Gaussian, with a closed form:

    $$
    \begin{bmatrix} \mathbf{f} \\ \mathbf{f}_* \end{bmatrix} \sim \mathcal{N}\!\left( \mathbf{0}, \begin{bmatrix} K & K_* \\ K_*^\top & K_{**} \end{bmatrix} \right)
    \quad\Longrightarrow\quad
    \mathbf{f}_* \mid \mathbf{f} \sim \mathcal{N}\!\left( K_*^\top K^{-1} \mathbf{f},\; K_{**} - K_*^\top K^{-1} K_* \right)
    $$

    With two unit-variance variables and correlation $\rho$, this is just $\mathbb{E}[f_2 \mid f_1] = \rho f_1$ and
    $\mathrm{Var}[f_2 \mid f_1] = 1 - \rho^2$. Two facts to notice in the figure:

    - the conditional **mean moves linearly** with the observed value;
    - the conditional **variance does not depend on the observed value at all**, only on the correlation.
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
    ## 2. From vectors to functions: the posterior

    A **Gaussian process** $f \sim \mathcal{GP}(m, k)$ is a collection of random variables, any finite subset of
    which is jointly Gaussian. The mean function $m(x) = \mathbb{E}[f(x)]$ is usually set to zero (after
    standardizing $y$), and the **kernel** $k(x, x') = \mathrm{Cov}(f(x), f(x'))$ does the modeling. The
    infinitely many other points integrate out for free, so we only ever touch the Gaussian over the training and
    test points.

    **Where the posterior comes from.** Let $\mathbf{f} = f(X)$ and $f_* = f(x_*)$. Bayes' rule gives the posterior
    over $\mathbf{f}$, and averaging over it gives the prediction:

    $$
    p(\mathbf{f} \mid \mathbf{y}) = \frac{p(\mathbf{y} \mid \mathbf{f})\, p(\mathbf{f})}{p(\mathbf{y})},
    \qquad
    p(f_* \mid \mathbf{y}) = \int p(f_* \mid \mathbf{f})\, p(\mathbf{f} \mid \mathbf{y})\, d\mathbf{f}
    $$

    Every density here is Gaussian, so there is a shortcut. Since $\mathbf{y} = \mathbf{f} + \boldsymbol\varepsilon$
    with independent noise, $\mathbf{y}$ and $f_*$ are jointly Gaussian. The noise adds $\sigma_n^2 I$ to the
    observed block only:

    $$
    \begin{bmatrix} \mathbf{y} \\ f_* \end{bmatrix} \sim \mathcal{N}\left( \mathbf{0}, \begin{bmatrix} K + \sigma_n^2 I & \mathbf{k}_* \\ \mathbf{k}_*^\top & k(x_*, x_*) \end{bmatrix} \right)
    $$

    Conditioning (Section 1) then gives the posterior mean and variance at any test point:

    $$
    \mu(x_*) = \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y},
    \qquad
    \sigma^2(x_*) = k(x_*, x_*) - \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{k}_*
    $$

    Here $K_{ij} = k(x_i, x_j)$ and $[\mathbf{k}_*]_i = k(x_i, x_*)$. The variance starts at the prior variance and
    shrinks near data; far from data it returns to the prior. In code this is **one Cholesky factorization**
    $LL^\top = K + \sigma_n^2 I$: $O(n^3)$ once, then $O(n)$ per mean and $O(n^2)$ per variance. Nobody inverts $K$
    (Section 6.5 shows why).
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
    f_test,
    gp_kernel,
    gp_ls,
    gp_n,
    gp_noise,
    gp_posterior,
    mo,
    np,
    plot_posterior,
    plt,
    sample_paths,
    show,
):
    _rng = np.random.default_rng(3)
    _Xall = _rng.uniform(0, 1, 15)
    _eps = _rng.standard_normal(15)
    _X = _Xall[: gp_n.value]
    _y = f_test(_X) + gp_noise.value * _eps[: gp_n.value]
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
    **The variance ignores the values.** The formula for $\sigma^2(x_*)$ contains $X$ but not $\mathbf{y}$: how
    uncertain the GP is depends only on **where** you observed, not on **what** you saw. Below, a fifth point is
    added at $x = 0.75$ with three different values. The means go three different ways; the standard deviation is
    identical to machine precision. (A consequence: if the kernel is wrong, surprising data will never make the
    error bars wider. Only refitting the hyperparameters can.)
    """)
    return


@app.cell(hide_code=True)
def _(C, SERIES, XS, f_test, gp_posterior, k_m52, np, plt, show):
    _X = np.array([0.05, 0.3, 0.5, 0.95])
    _y = f_test(_X)
    _yz = (_y - _y.mean()) / _y.std()
    _kern = lambda a, b: k_m52(a, b, ls=0.15)
    _xs = XS[::2]
    _xb = 0.75

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    _mu0, _v0 = gp_posterior(_X, _yz, _xs, _kern)
    _sds = []
    for _i, _fy in enumerate([-1.5, 0.0, 1.5]):
        _mu, _v = gp_posterior(np.append(_X, _xb), np.append(_yz, _fy), _xs, _kern)
        _ax1.fill_between(_xs, _mu - 2 * np.sqrt(_v), _mu + 2 * np.sqrt(_v), color=SERIES[_i], alpha=0.08, lw=0)
        _ax1.plot(_xs, _mu, color=SERIES[_i], lw=1.3, label=rf"observe $y = {_fy:+.1f}$ at $x = 0.75$")
        _sds.append(np.sqrt(_v))
    assert np.allclose(_sds[0], _sds[1]) and np.allclose(_sds[0], _sds[2])
    _ax1.scatter(_X, _yz, s=28, color=C["ink"], zorder=5, edgecolor="#fcfcfb", linewidth=1.2)
    _ax1.axvline(_xb, color=C["ink"], lw=0.8, ls=":")
    _ax1.set(title="Three different values: three means", xlabel=r"$x$")
    _ax1.legend(loc="lower left", fontsize=7)

    _ax2.plot(_xs, np.sqrt(_v0), color=C["ink2"], ls="--", lw=1.2, label=r"$\sigma(x)$ with 4 points")
    _ax2.plot(_xs, _sds[0], color=C["violet"], lw=2.2, label=r"$\sigma(x)$ with 5 points (same for all three $y$)")
    _ax2.axvline(_xb, color=C["ink"], lw=0.8, ls=":")
    _ax2.set(title=r"...but one $\sigma(x)$", xlabel=r"$x$", ylim=(0, 1.25))
    _ax2.legend(loc="upper left", fontsize=7)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Learning hyperparameters from the marginal likelihood

    The kernel has hyperparameters $\theta$: lengthscale $\ell$, output scale $\sigma_f^2$, noise $\sigma_n^2$. They
    change the answer a lot (compare the three fits below), so they have to be learned from the data.

    **The idea.** For a candidate $\theta$, ask: *if functions really behaved the way $\theta$ says, how likely would
    exactly these $\mathbf{y}$ be?* We don't know $f$, so we average over every function the prior allows, weighted
    by its prior probability. That is "$f$ integrated out", and the result is the **marginal likelihood**
    (or evidence):

    $$
    p(\mathbf{y} \mid X, \theta) = \int p(\mathbf{y} \mid \mathbf{f})\, p(\mathbf{f} \mid X, \theta)\, d\mathbf{f}
    = \int \mathcal{N}(\mathbf{y} \mid \mathbf{f}, \sigma_n^2 I)\; \mathcal{N}(\mathbf{f} \mid \mathbf{0}, K_\theta)\, d\mathbf{f}
    $$

    No sampling is needed. The integral is the density of $\mathbf{y} = \mathbf{f} + \boldsymbol\varepsilon$, a sum
    of two independent Gaussians, so $\mathbf{y} \sim \mathcal{N}(\mathbf{0}, K_y)$ with
    $K_y = K_\theta + \sigma_n^2 I$. $K_y$ is built from the **inputs** $X$ and $\theta$ only. Its log-density is

    $$
    \log p(\mathbf{y} \mid X, \theta) =
    \underbrace{-\tfrac{1}{2} \mathbf{y}^\top K_y^{-1} \mathbf{y}}_{\text{data fit}}
    \;\underbrace{-\tfrac{1}{2} \log |K_y|}_{\text{complexity penalty}}
    \;-\; \tfrac{n}{2} \log 2\pi
    $$

    **Why it doesn't overfit.** A probability distribution must sum to one. A flexible model (short $\ell$) can
    produce almost any dataset, so it spreads its probability thinly and gives little to the one you observed; that
    is the complexity term. A rigid model (long $\ell$) concentrates on smooth datasets and fits wiggly data badly;
    that is the data-fit term. The maximum is the simplest model that still explains the data: an **automatic
    Occam's razor**. Choosing $\theta$ this way is called **type-II maximum likelihood** (or empirical Bayes).
    """)
    return


@app.cell(hide_code=True)
def _(C, F_TRUE, XS, f_test, gp_posterior, k_se, log_ml_terms, np, plot_posterior, plt, show):
    _rng = np.random.default_rng(7)
    _X = np.sort(_rng.uniform(0, 1, 12))
    _y = f_test(_X) + 0.1 * _rng.standard_normal(12)
    _ym, _ys = _y.mean(), _y.std()
    _yz = (_y - _ym) / _ys
    _noise = (0.1 / _ys) ** 2

    _ls = np.logspace(-2.3, 0.5, 160)
    _terms = np.array([log_ml_terms(_X, _yz, lambda a, b, l=l: k_se(a, b, ls=l), _noise) for l in _ls])
    _total = _terms.sum(1)
    _best = _ls[_total.argmax()]

    _fig = plt.figure(figsize=(11, 6.4))
    _gs = _fig.add_gridspec(2, 3)
    _ax = _fig.add_subplot(_gs[0, :])
    _ax.plot(_ls, _terms[:, 0], color=C["orange"], label=r"data fit $-\frac{1}{2}\mathbf{y}^\top K_y^{-1}\mathbf{y}$")
    _ax.plot(_ls, _terms[:, 1], color=C["aqua"], label=r"complexity $-\frac{1}{2}\log|K_y|$")
    _ax.plot(_ls, _total, color=C["blue"], lw=2.2, label=r"$\log p(\mathbf{y} \mid X, \theta)$")
    _ax.axvline(_best, color=C["ink2"], ls=":", lw=1)
    _ax.set(xscale="log", xlabel=r"lengthscale $\ell$", ylim=(_total.max() - 40, max(_terms[:, 1].max(), 5) + 3),
            title=rf"Decomposition (SE kernel, fixed noise, 12 points). ML optimum $\ell \approx {_best:.3f}$")
    _ax.legend(loc="lower right")

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
    ### Optimizing it: gradients, L-BFGS and restarts

    There is no formula for the best $\theta$, so we climb the surface step by step. The gradient is closed-form,
    from two matrix identities, $\partial K^{-1} = -K^{-1} (\partial K) K^{-1}$ and
    $\partial \log|K| = \mathrm{tr}(K^{-1} \partial K)$:

    $$
    \frac{\partial}{\partial \theta_j} \log p(\mathbf{y} \mid X, \theta)
    = \tfrac{1}{2} \operatorname{tr}\left( \left( \boldsymbol{\alpha}\boldsymbol{\alpha}^\top - K_y^{-1} \right) \frac{\partial K_y}{\partial \theta_j} \right),
    \qquad \boldsymbol{\alpha} = K_y^{-1} \mathbf{y}
    $$

    The usual optimizer is **L-BFGS**, a quasi-Newton method: it uses the last few gradients to estimate curvature,
    which gives much better steps than plain gradient ascent (the "-B" variant supports box bounds). Optimization is
    done over $\log \ell$ and $\log \sigma_n$, so the parameters stay positive.

    **The catch: the surface has several hills**, and L-BFGS climbs the one nearest its start. Below, 8 noisy points
    from $\sin(4\pi x)$, and L-BFGS started from 12 places. The runs end at three different optima: a sensible fit,
    an **interpolating** one (short $\ell$, tiny noise) and an **"everything is noise"** one (huge $\ell$, large
    noise). The cures are **multiple restarts** (keep the best) and **priors** on $\theta$ that rule out the silly
    extremes.
    """)
    return


@app.cell(hide_code=True)
def _(C, SERIES, XS, gp_posterior, k_se, log_ml, minimize, np, plot_posterior, plt, show):
    _rng = np.random.default_rng(0)
    _X = np.sort(_rng.uniform(0, 1, 8))
    _y = np.sin(4 * np.pi * _X) + 0.3 * _rng.standard_normal(8)
    _ym, _ys = _y.mean(), _y.std()
    _yz = (_y - _ym) / _ys
    _ftrue = (np.sin(4 * np.pi * XS) - _ym) / _ys

    def _lml(t):
        _l, _s = np.exp(t)
        try:
            return log_ml(_X, _yz, lambda a, b: k_se(a, b, ls=_l), _s**2 + 1e-8)
        except np.linalg.LinAlgError:
            return np.nan

    _bounds = [(np.log(0.005), np.log(10.0)), (np.log(1e-3), np.log(3.0))]
    _starts = [(np.log(l), np.log(s)) for l in [0.02, 0.1, 0.5, 2.0] for s in [0.02, 0.2, 1.0]]
    _runs = []
    for _s0 in _starts:
        _path = [np.array(_s0)]
        _r = minimize(lambda t: -_lml(t), _s0, method="L-BFGS-B", bounds=_bounds,
                      callback=lambda xk: _path.append(np.array(xk)))
        _runs.append((np.array(_path), _r.x, -_r.fun))

    # distinct optima: keep the best run per basin (separated in log-lengthscale)
    _optima = []
    for _p, _x, _f in sorted(_runs, key=lambda r: -r[2]):
        if all(abs(_x[0] - o[0][0]) > 0.5 for o in _optima):
            _optima.append((_x, _f))
    _optima = _optima[:3]

    def _basin(x):
        return int(np.argmin([abs(x[0] - o[0][0]) for o in _optima]))

    _fig = plt.figure(figsize=(12.5, 6.2))
    _gs = _fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    _axc = _fig.add_subplot(_gs[0, :])
    _lg, _sg = np.logspace(np.log10(0.005), 1, 70), np.logspace(-3, np.log10(3), 70)
    _Z = np.array([[_lml((np.log(l), np.log(s))) for l in _lg] for s in _sg])
    _cf = _axc.contourf(_lg, _sg, np.clip(_Z, np.nanmax(_Z) - 12, None), levels=24, cmap="Greys", alpha=0.55)
    _axc.grid(False)
    for _p, _x, _f in _runs:
        _col = SERIES[_basin(_x)]
        _axc.plot(np.exp(_p[:, 0]), np.exp(_p[:, 1]), color=_col, lw=1.1, alpha=0.9)
        _axc.plot(*np.exp(_p[0]), "o", color=_col, ms=4)
    for _i, (_x, _f) in enumerate(_optima):
        _axc.plot(*np.exp(_x), "*", color=SERIES[_i], ms=16, mec=C["ink"], mew=0.8, zorder=6, clip_on=False)
        _axc.annotate(str(_i + 1), np.exp(_x), xytext=(-16, 10), textcoords="offset points", fontsize=9,
                      color="#fcfcfb", weight="bold", ha="center", va="center", zorder=7, annotation_clip=False,
                      bbox=dict(boxstyle="circle,pad=0.25", fc=SERIES[_i], ec=C["ink"], lw=0.8))
    _axc.set(xscale="log", yscale="log", xlabel=r"lengthscale $\ell$", ylabel=r"noise $\sigma_n$ (standardized)",
             title=r"$\log p(\mathbf{y} \mid X, \theta)$ (darker = higher) and 12 L-BFGS runs ($\circ$ start, $\star$ end)")

    for _i, (_x, _f) in enumerate(_optima):
        _l, _s = np.exp(_x)
        _a = _fig.add_subplot(_gs[1, _i])
        _mu, _v = gp_posterior(_X, _yz, XS, lambda a, b: k_se(a, b, ls=_l), _s**2 + 1e-8)
        plot_posterior(_a, XS, _mu, np.sqrt(_v), _X, _yz, _ftrue, color=SERIES[_i])
        _a.set(title=rf"optimum {_i + 1}: $\ell = {_l:.3f}$, $\sigma_n = {_s:.3f}$, $\log p = {_f:.2f}$",
               xlabel=r"$x$", ylim=(-3.2, 3.2))
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### When data is scarce: integrate $\theta$ out too

    With very few points the likelihood over $\theta$ is flat or points somewhere absurd. Below, with 3 noiseless
    points, maximum likelihood runs to the smallest lengthscale on the grid: it concludes the points are unrelated,
    so the prediction collapses to the prior between them. The fully Bayesian alternative puts a prior on $\theta$
    and averages the predictions over the posterior $p(\theta \mid D)$:

    $$
    p(f_* \mid D) = \int p(f_* \mid D, \theta)\, p(\theta \mid D)\, d\theta \;\approx\; \sum_s w_s\, p(f_* \mid D, \theta^{(s)})
    $$

    The result is a mixture of GPs. Its mean is the weighted average of the means, and its variance adds the spread
    between the means to the average variance. It is more expensive but much more robust when $n$ is tiny.
    """)
    return


@app.cell(hide_code=True)
def _(C, XS, f_test, gp_posterior, k_m52, log_ml, np, plot_posterior, plt, show):
    _X = np.array([0.1, 0.45, 0.62])
    _y = f_test(_X)
    _ym, _ys = _y.mean(), _y.std()
    _yz = (_y - _ym) / _ys
    _lg = np.logspace(-2, 0.5, 120)
    _loglik = np.array([log_ml(_X, _yz, lambda a, b, l=l: k_m52(a, b, ls=l), 1e-6) for l in _lg])
    _logprior = -0.5 * ((np.log(_lg) - np.log(0.2)) / 0.75) ** 2  # log-normal prior on ℓ
    _lp = _loglik + _logprior
    _post = np.exp(_lp - _lp.max())
    _post /= _post.sum()
    _ml = _lg[_loglik.argmax()]

    _xs = XS[::2]
    _mu_ml, _v_ml = gp_posterior(_X, _yz, _xs, lambda a, b: k_m52(a, b, ls=_ml), 1e-6)
    _m1 = np.zeros_like(_xs)
    _m2 = np.zeros_like(_xs)
    for _w, _l in zip(_post, _lg):
        if _w > 1e-4:
            _mu, _v = gp_posterior(_X, _yz, _xs, lambda a, b, l=_l: k_m52(a, b, ls=l), 1e-6)
            _m1 += _w * _mu
            _m2 += _w * (_v + _mu**2)
    _wsum = _post[_post > 1e-4].sum()
    _mix_mu, _mix_var = _m1 / _wsum, _m2 / _wsum - (_m1 / _wsum) ** 2

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    _ax1.plot(_lg, _post / _post.max(), color=C["blue"], label=r"posterior $p(\ell \mid D)$ (scaled)")
    _ax1.plot(_lg, np.exp(_loglik - _loglik.max()), color=C["orange"], label="likelihood (scaled)")
    _ax1.axvline(_ml, color=C["orange"], ls=":", lw=1)
    _ax1.set(xscale="log", xlabel=r"lengthscale $\ell$", title=rf"3 noiseless points: ML runs to the grid edge, $\ell = {_ml:.2f}$")
    _ax1.legend(loc="center right")

    _ftrue = (f_test(_xs) - _ym) / _ys
    plot_posterior(_ax2, _xs, _mix_mu, np.sqrt(np.maximum(_mix_var, 1e-12)), _X, _yz, _ftrue,
                   label=r"averaged over $p(\ell \mid D)$")
    _ax2.plot(_xs, _mu_ml, color=C["orange"], lw=1.2, label=rf"ML $\ell = {_ml:.2f}$: mean")
    _ax2.plot(_xs, _mu_ml + 2 * np.sqrt(_v_ml), color=C["orange"], lw=0.8, ls="--", label=r"ML: $\pm 2\sigma$")
    _ax2.plot(_xs, _mu_ml - 2 * np.sqrt(_v_ml), color=C["orange"], lw=0.8, ls="--")
    _ax2.set(xlabel=r"$x$", title="Predictions: ML point estimate vs integrated over $\\ell$", ylim=(-3.2, 3.8))
    _ax2.legend(loc="upper center", ncols=3, fontsize=7)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The weight-space view and the kernel trick

    There is a second way to see a GP. Take a linear model over features, $f(x) = \phi(x)^\top w$, with random
    Gaussian weights $w \sim \mathcal{N}(0, \Sigma_p)$. Every draw of $w$ gives one function, so a prior on weights is
    a prior on functions. $f$ at any set of points is a linear combination of Gaussians, hence jointly Gaussian, so
    $f$ is a GP with kernel

    $$
    k(x, x') = \mathrm{Cov}\big(f(x), f(x')\big) = \phi(x)^\top \Sigma_p\, \phi(x')
    $$

    **The kernel trick** runs this backwards. If a method only uses features through dot products
    $\phi(x)^\top\phi(x')$, replace each one with a function $k(x, x')$ that returns the same number, and never build
    the features. For example, take 2D inputs and quadratic features:

    $$
    \phi(x) = \big[\, x_1^2,\;\; \sqrt{2}\, x_1 x_2,\;\; x_2^2 \,\big]
    \qquad\Longrightarrow\qquad
    \phi(x)^\top \phi(z) = (x^\top z)^2
    $$

    Three features, computed with one two-dimensional dot product. For degree 5 in 100 dimensions, the same trick
    replaces about 92 million features.

    The squared exponential corresponds to **infinitely many** features. One way to see it: put a Gaussian bump at
    every location $c$ and give each an independent random weight $w_c$:

    $$
    f(x) = \sum_c w_c\, \phi_c(x), \qquad \phi_c(x) = \exp\!\left( -\frac{(x - c)^2}{\ell^2} \right)
    $$

    Summing over $c$ (an integral in the limit) gives exactly the SE kernel with lengthscale $\ell$. The figure
    builds this: with few bumps the covariance is lumpy and the samples look like a row of bumps; with many, both
    match the SE kernel.

    The GP posterior formulas contain the features only through $K_{ij} = k(x_i, x_j)$, $\mathbf{k}_*$ and
    $k(x_*, x_*)$. So **a GP is Bayesian linear regression in a feature space you never construct**, and choosing a
    kernel means choosing the building blocks $f$ is made of. The view pays off twice: random-feature approximations
    (Section 7) and the link to neural networks (an infinitely wide random network is a GP; Neal 1996,
    Lee et al. 2018).
    """)
    return


@app.cell(hide_code=True)
def _(C, SERIES, XS, k_se, np, plt, show):
    _ell = 0.1
    _xs = XS[::2]

    def _features(x, m):
        _c = np.linspace(-0.3, 1.3, m)
        _delta = _c[1] - _c[0]
        _scale = np.sqrt(_delta / (_ell * np.sqrt(np.pi / 2)))  # makes Var f(x) ≈ 1 for dense bumps
        return _scale * np.exp(-(((x[:, None] - _c[None, :]) / _ell) ** 2))

    _rng = np.random.default_rng(2)
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 3.2))

    _P = _features(_xs, 14)
    _w = _rng.standard_normal(14)
    for _j in range(14):
        _axes[0].plot(_xs, _P[:, _j] * _w[_j], color=C["ink2"], lw=0.8, alpha=0.6,
                      label=r"$w_c\, \phi_c(x)$" if _j == 0 else None)
    _axes[0].plot(_xs, _P @ _w, color=C["blue"], lw=2.2, label=r"$f(x) = \sum_c w_c \phi_c(x)$")
    _axes[0].set(title="A function built from 14 weighted bumps", xlabel=r"$x$", ylim=(-3, 3.6))
    _axes[0].legend(loc="upper center", ncols=2, fontsize=7)

    _x0 = np.array([0.5])
    for _i, _m in enumerate([6, 25, 400]):
        _axes[1].plot(_xs, (_features(_x0, _m) @ _features(_xs, _m).T).ravel(), color=SERIES[_i], lw=1.2,
                      label=rf"$m = {_m}$ bumps")
    _axes[1].plot(_xs, k_se(_x0, _xs, ls=_ell)[0], color=C["ink"], ls="--", lw=1.4,
                  label=r"exact SE, $\ell = 0.1$")
    _axes[1].set(title=r"Covariance $\phi(0.5)^\top \phi(x)$", xlabel=r"$x$")
    _axes[1].legend(loc="upper right", fontsize=7)

    for _ax, _m in zip(_axes[2:], [6, 400]):
        _P = _features(_xs, _m)
        for _i in range(3):
            _ax.plot(_xs, _P @ _rng.standard_normal(_m), color=SERIES[_i], lw=1.1)
        _ax.set(title=rf"Prior samples with $m = {_m}$ bumps", xlabel=r"$x$", ylim=(-3, 3))
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Kernels: what you believe about $f$

    The kernel **is** the prior. It decides smoothness, how far information travels, which inputs matter, and
    whether $f$ is periodic or additive. For stationary kernels, the correlation depends only on the distance
    $r = |x - x'|$, so the kernel is a single curve $k(r)$.

    **Matérn family.** A Matérn-$\nu$ process is $k$ times differentiable exactly when $\nu > k$. $\nu = 1/2$ gives
    rough, Brownian-like paths; $\nu \to \infty$ gives the squared exponential (SE), which is infinitely
    differentiable. **Matérn-5/2 is a common default**: SE is often unrealistically smooth, which makes the GP
    overconfident between observations. What matters is the shape of $k(r)$ near $r = 0$: a sharp corner means
    close neighbours decorrelate fast (rough), a flat top means they stay correlated (smooth).

    **Spectral view (Bochner's theorem).** Every stationary kernel is the Fourier transform of a non-negative
    **spectral density** $S(s)$: how much of each frequency $s$ the random functions contain. Same information as
    $k(r)$, expressed through waves. Heavy high-frequency tails mean fine wiggles (rough samples); SE's tail
    collapses, so its samples have essentially no fine structure. A Matérn tail decays like $s^{-(2\nu + 1)}$.
    """)
    return


@app.cell(hide_code=True)
def _(SERIES, cholesky, k_m12, k_m32, k_m52, k_se, np, plt, show):
    _xs = np.linspace(0, 1, 300)
    _z = np.random.default_rng(11).standard_normal((300, 2))
    _ks = [("Matérn 1/2 (rough)", k_m12), ("Matérn 3/2", k_m32), ("Matérn 5/2 (common default)", k_m52),
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

    Valid kernels are closed under **sums** ($f$ is a sum of independent functions, one per kernel; in weight space,
    the feature sets are concatenated) and **products** (points are similar only if similar under both kernels). That
    gives a small grammar: periodic × SE is "almost periodic", linear + SE is "trend plus wiggles".

    **Automatic relevance determination (ARD)** gives each input its own lengthscale,
    $r^2 = \sum_d (x_d - x'_d)^2/\ell_d^2$. A small $\ell_d$ means $f$ changes quickly along input $d$; a large one
    switches the input off. It is "automatic" because the marginal likelihood learns the $\ell_d$: raising the
    lengthscale of an irrelevant input makes the model simpler at no cost in fit, so Occam's razor pushes it up.
    Normalize inputs first, or the lengthscales aren't comparable.
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
    ### The RKHS connection, and a subtle trap

    Every kernel defines a **reproducing kernel Hilbert space** (RKHS): the functions you can build from kernel bumps,
    $f = \sum_i \alpha_i k(x_i, \cdot)$ (and limits), with a norm $\|f\|_{\mathcal H}^2 = \boldsymbol\alpha^\top K
    \boldsymbol\alpha$ that measures complexity by the kernel's standards. In the spectral view,
    $\|f\|^2_{\mathcal H} = \int |\hat f(s)|^2 / S(s)\, ds$: frequencies the kernel considers rare are expensive.

    - The GP mean is the **minimum-norm interpolant** of the data (noise-free), or the kernel ridge regression
      solution (noisy).
    - In the noise-free case, $\sigma(x)$ is the **worst-case error** over all functions with
      $\|f\|_{\mathcal H} \le 1$ that match the data.

    **The trap:** GP samples are almost surely **not** in the RKHS of their own kernel. In weight space, a sample
    has i.i.d. $\mathcal N(0, 1)$ weights, and there are infinitely many of them, so

    $$
    \|f\|^2_{\mathcal H} = \sum_i w_i^2 \approx 1 + 1 + 1 + \dots = \infty
    $$

    The figure checks this. Interpolate a function at $n$ points and compute the norm of the interpolant: for a
    smooth function it converges, for a GP sample from the same kernel it grows like $n$ without bound. So "$f$ is a
    GP sample" and "$f$ has bounded RKHS norm" describe **different function classes**. Keep this in mind whenever
    a theorem assumes one of them.
    """)
    return


@app.cell(hide_code=True)
def _(C, cholesky, f_test, k_m32, np, plt, show):
    _g = np.linspace(0, 1, 1025)
    _Kg = k_m32(_g, _g, ls=0.2)
    _sample = cholesky(_Kg + 1e-10 * np.eye(len(_g)), lower=True) @ np.random.default_rng(0).standard_normal(len(_g))
    _smooth = f_test(_g) / f_test(_g).std()

    _ns, _n_sample, _n_smooth = [], [], []
    for _step in [128, 64, 32, 16, 8, 4, 2, 1]:
        _idx = np.arange(0, len(_g), _step)
        _Kn = k_m32(_g[_idx], _g[_idx], ls=0.2) + 1e-10 * np.eye(len(_idx))
        _ns.append(len(_idx))
        _n_sample.append(_sample[_idx] @ np.linalg.solve(_Kn, _sample[_idx]))
        _n_smooth.append(_smooth[_idx] @ np.linalg.solve(_Kn, _smooth[_idx]))

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10.5, 3.3))
    _ax1.plot(_g, _sample, color=C["orange"], lw=1.1, label=r"sample from $\mathcal{GP}(0, k_{\mathrm{Matern\ 3/2}})$")
    _ax1.plot(_g, _smooth, color=C["blue"], lw=1.6, label="a smooth function")
    _ax1.set(title="Two functions on [0, 1]", xlabel=r"$x$")
    _ax1.legend(loc="lower left", fontsize=7)
    _ax2.plot(_ns, _n_sample, "o-", color=C["orange"], label="GP sample: grows without bound")
    _ax2.plot(_ns, _n_smooth, "o-", color=C["blue"], label="smooth function: converges")
    _ax2.set(xscale="log", yscale="log", xlabel=r"interpolation points $n$",
             ylabel=r"$\|\mathrm{interpolant}\|^2_{\mathcal{H}} = \mathbf{f}^\top K^{-1} \mathbf{f}$",
             title="RKHS norm of the interpolant")
    _ax2.legend(loc="upper left", fontsize=7)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. How GPs fail

    A GP always returns a smooth mean and tidy error bars, even when its assumptions are badly wrong. That makes its
    failures quiet. Most of them come from one of the assumptions baked into the standard model:

    | Assumption | What breaks when it's wrong | Section |
    | :--- | :--- | :--- |
    | Zero (or constant) prior mean | Extrapolation reverts to the mean | 6.1 |
    | Stationary kernel: one $\ell$ everywhere | Wiggly and flat regions can't both be modeled | 6.2 |
    | Same noise level everywhere | Error bars too wide in some places, too narrow in others | 6.3 |
    | Gaussian noise | One outlier drags the whole fit | 6.4 |
    | Exact linear algebra | Ill-conditioned $K$ gives garbage or crashes | 6.5 |
    | Distances are informative | In high dimensions every pair looks equally far | 6.6 |

    Hyperparameter pathologies (Section 3) belong on this list too: local optima of the marginal likelihood, and
    point estimates from too little data.

    ### 6.1 Extrapolation reverts to the prior mean

    Far from data, the posterior returns to the prior, and the prior mean is zero. A stationary kernel therefore
    **cannot extrapolate a trend**: the prediction bends back to zero past the last observation, while the error
    bars widen to the prior. Fixes: a **linear** (or polynomial) term in the kernel or the mean function, when you
    believe the trend continues; otherwise, simply don't trust a GP outside the data.
    """)
    return


@app.cell(hide_code=True)
def _(C, gp_posterior, k_linear, k_se, np, plot_posterior, plt, show):
    _rng = np.random.default_rng(1)
    _ftr = lambda x: 2 * x + 0.25 * np.sin(15 * x)
    _X = np.sort(_rng.uniform(0, 0.6, 15))
    _y = _ftr(_X) + 0.05 * _rng.standard_normal(15)
    _xs = np.linspace(0, 1.4, 300)
    _kerns = [
        ("SE kernel: mean falls back to 0", lambda a, b: k_se(a, b, ls=0.1)),
        ("Linear + SE kernel: trend continues", lambda a, b: k_linear(a, b, sf2=4.0, c=0.0) + 0.1 * k_se(a, b, ls=0.1)),
    ]
    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
    for _ax, (_title, _k) in zip(_axes, _kerns):
        _mu, _v = gp_posterior(_X, _y, _xs, _k, 0.05**2)
        _ax.axvspan(0.6, 1.4, color=C["ink2"], alpha=0.05, lw=0)
        plot_posterior(_ax, _xs, _mu, np.sqrt(_v), _X, _y, _ftr(_xs))
        _ax.axhline(0, color=C["ink2"], lw=0.8)
        _ax.set(title=_title, xlabel=r"$x$", ylim=(-2.5, 4.5))
        _ax.text(1.0, 4.0, "no data: extrapolation", ha="center", color=C["ink2"], fontsize=8)
    _axes[0].legend(loc="lower left", ncols=2, fontsize=7)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Nonstationarity: one lengthscale for the whole domain

    A stationary kernel assumes $f$ is equally wiggly everywhere. Below, $f(x) = \sin(20x^3)$ is flat on the left
    and oscillates fast on the right. Maximum likelihood has to pick **one** $\ell$ and compromises: short enough
    for the right side, so the flat left side gets a wobbly mean and error bars that bulge between points. In the
    warped input the left side is modeled cleanly; the wider band on the far right is honest, because in $u$ the
    data there are sparse.

    Fixes: **input warping** (fit the GP on a transformed input $u = w(x)$ where $f$ is stationary; here
    $u = x^3$, and in practice $w$ is learned, e.g. a Kumaraswamy CDF per input), nonstationary kernels, local
    models, or deep kernels. Log-transforming inputs such as learning rates is the everyday version of the same idea.
    """)
    return


@app.cell(hide_code=True)
def _(C, fit_lengthscale, gp_posterior, k_m52, np, plot_posterior, plt, show):
    _rng = np.random.default_rng(3)
    _fns = lambda x: np.sin(20 * x**3)
    _X = np.sort(_rng.uniform(0, 1, 40))
    _y = _fns(_X) + 0.05 * _rng.standard_normal(40)
    _xs = np.linspace(0, 1, 400)
    _noise = 0.05**2

    _l_x = fit_lengthscale(_X, _y, k_m52, _noise, grid=np.logspace(-2.3, 0, 60))
    _mu1, _v1 = gp_posterior(_X, _y, _xs, lambda a, b: k_m52(a, b, ls=_l_x), _noise)
    _l_u = fit_lengthscale(_X**3, _y, k_m52, _noise, grid=np.logspace(-2.3, 0, 60))
    _mu2, _v2 = gp_posterior(_X**3, _y, _xs**3, lambda a, b: k_m52(a, b, ls=_l_u), _noise)

    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
    plot_posterior(_axes[0], _xs, _mu1, np.sqrt(_v1), _X, _y, _fns(_xs))
    _axes[0].set(title=rf"Stationary GP in $x$ (ML $\ell = {_l_x:.3f}$)", xlabel=r"$x$", ylim=(-2.2, 2.2))
    _axes[0].legend(loc="lower left", ncols=2, fontsize=7)
    plot_posterior(_axes[1], _xs, _mu2, np.sqrt(_v2), _X, _y, _fns(_xs), color=C["aqua"])
    _axes[1].set(title=rf"GP in warped input $u = x^3$ (ML $\ell_u = {_l_u:.3f}$)", xlabel=r"$x$")
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 Heteroscedastic noise, and how to spot it with leave-one-out

    The standard likelihood has one noise level $\sigma_n$ everywhere. Below, the true noise grows from almost
    nothing on the left to large on the right. The fitted GP uses an average $\sigma_n$: its predictive band for $y$
    is **too wide** on the left and **too narrow** on the right.

    **Leave-one-out (LOO) residuals** catch this, and for a GP they are free: no refitting needed. With
    $K_y^{-1}$ in hand, the prediction for $y_i$ from all the other points is

    $$
    \mu_{-i} = y_i - \frac{[K_y^{-1}\mathbf{y}]_i}{[K_y^{-1}]_{ii}}, \qquad \sigma^2_{-i} = \frac{1}{[K_y^{-1}]_{ii}}
    $$

    If the model is right, the standardized residuals $z_i = (y_i - \mu_{-i}) / \sigma_{-i}$ look like
    $\mathcal N(0, 1)$ noise: an even cloud around 0, about 95% inside $\pm 2$, with no pattern in $x$.

    **Top row (the problem).** The band is too wide on the left and too narrow on the right. The residuals give it
    away: squashed near 0 on the left (the model expected more noise than it saw), spread past $\pm 2$ on the right
    (more noise than it expected). The fraction outside $\pm 2$ looks almost fine, which is why you have to look at
    the **pattern**, not just the count.

    **Bottom row (the fix).** Let the noise variance depend on $x$: here
    $\log \sigma_n^2(x) = a + b\,x + c\,x^2$, learned together with $\ell$ by maximizing the marginal likelihood (the
    only change is the diagonal of $K_y$, from $\sigma_n^2$ to $\sigma_n^2(x_i)$). The band now follows the true
    noise, and the residuals become an even cloud across $x$. Other fixes: a second GP on $\log \sigma_n^2(x)$ for
    complex patterns, known per-point noise from replicates, or a variance-stabilizing transform of $y$ such as
    $\log y$.
    """)
    return


@app.cell(hide_code=True)
def _(C, gp_posterior, k_m52, log_ml, loo, minimize, np, plt, show):
    _rng = np.random.default_rng(5)
    _fh = lambda x: 0.8 * np.sin(2 * np.pi * x)
    _sd_true = lambda x: 0.03 + 0.5 * x**2
    _X = np.sort(_rng.uniform(0, 1, 100))
    _y = _fh(_X) + _sd_true(_X) * _rng.standard_normal(100)
    _xs = np.linspace(0, 1, 300)

    # (a) standard GP: one noise level, fitted with ℓ by marginal likelihood
    _lg, _sg = np.logspace(-1.5, 0, 25), np.logspace(-2, -0.3, 25)
    _Z = np.array([[log_ml(_X, _y, lambda a, b, l=l: k_m52(a, b, ls=l), s**2) for l in _lg] for s in _sg])
    _si, _li = np.unravel_index(np.argmax(_Z), _Z.shape)
    _l0, _s0 = _lg[_li], _sg[_si]

    # (b) fix: noise variance depends on x, log σ²(x) = a + b·x + c·x², learned jointly with ℓ
    def _noise(t, x):
        return np.exp(t[1] + t[2] * x + t[3] * x**2)

    def _nlml(t):
        return -log_ml(_X, _y, lambda a, b: k_m52(a, b, ls=np.exp(t[0])), _noise(t, _X))

    _t = minimize(_nlml, [np.log(_l0), np.log(_s0**2), 0.0, 0.0], method="L-BFGS-B").x

    _models = [
        ("Standard GP: one noise level", _l0, np.full(100, _s0**2), np.full(300, _s0**2), C["orange"]),
        (r"Fixed: noise level learned as a function of $x$", np.exp(_t[0]), _noise(_t, _X), _noise(_t, _xs), C["blue"]),
    ]
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 6.6))
    for _row, (_title, _l, _nX, _nxs, _col) in enumerate(_models):
        _kern = lambda a, b, l=_l: k_m52(a, b, ls=l)
        _mu, _v = gp_posterior(_X, _y, _xs, _kern, _nX)
        _band = 2 * np.sqrt(_v + _nxs)
        _ax = _axes[_row, 0]
        _ax.fill_between(_xs, _mu - _band, _mu + _band, color=_col, alpha=0.18, lw=0, label=r"model: $\pm 2$ sd of $y$")
        _ax.plot(_xs, _mu, color=_col, label="posterior mean")
        _ax.plot(_xs, _fh(_xs) + 2 * _sd_true(_xs), color=C["ink2"], ls="--", lw=1.1, label=r"truth: $\pm 2$ sd of $y$")
        _ax.plot(_xs, _fh(_xs) - 2 * _sd_true(_xs), color=C["ink2"], ls="--", lw=1.1)
        _ax.scatter(_X, _y, s=10, color=C["ink"], lw=0, alpha=0.7)
        _ax.set(title=_title, xlabel=r"$x$", ylim=(-2.6, 2.6))
        _ax.legend(loc="lower left", fontsize=7, ncols=2)

        _lm, _lv = loo(_X, _y, _kern, _nX)
        _z = (_y - _lm) / np.sqrt(_lv)
        _out = np.abs(_z) > 2
        _az = _axes[_row, 1]
        _az.axhspan(-2, 2, color=_col, alpha=0.07, lw=0)
        _az.axhline(0, color=C["ink2"], lw=0.8)
        _az.scatter(_X[~_out], _z[~_out], s=14, color=C["ink2"], lw=0, label=r"$|z_i| \leq 2$")
        _az.scatter(_X[_out], _z[_out], s=20, color=C["red"], lw=0, label=r"$|z_i| > 2$")
        _az.set(title=rf"LOO residuals $z_i$: {_out.mean():.0%} outside $\pm 2$ (expect ≈5%)", xlabel=r"$x$",
                ylabel=r"$z_i$", ylim=(-4, 4))
        _az.legend(loc="upper left", fontsize=7)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.4 Outliers: the Gaussian likelihood has thin tails

    Under Gaussian noise, a point 40 standard deviations off is essentially impossible, so the GP bends the whole
    function toward it instead of calling it an error. Below, one corrupted observation drags the mean far from the
    truth over a whole lengthscale. (If $\sigma_n$ is being fitted, the outlier also inflates it, which widens the
    error bars everywhere.)

    The fix is a heavy-tailed likelihood such as **Student-t**. It has no closed-form posterior, but a simple EM
    loop works: a Student-t is a Gaussian whose variance is random, so each point gets its own noise
    $\sigma_n^2 / w_i$ with weight $w_i = (\nu + 1)/(\nu + r_i^2/\sigma_n^2)$ computed from its residual $r_i$.
    Points that don't fit get small weights and stop pulling on the curve. Laplace, EP and variational inference are
    the usual production methods for the same model.
    """)
    return


@app.cell(hide_code=True)
def _(C, F_TRUE, XS, f_test, gp_posterior, k_m52, np, plot_posterior, plt, show):
    _rng = np.random.default_rng(8)
    _X = np.sort(_rng.uniform(0, 1, 25))
    _y = f_test(_X) + 0.05 * _rng.standard_normal(25)
    _io = int(np.argmin(np.abs(_X - 0.35)))
    _y_bad = _y.copy()
    _y_bad[_io] += 2.0
    _kern = lambda a, b: k_m52(a, b, ls=0.12)
    _s2, _nu = 0.05**2, 4.0

    _mu_clean, _ = gp_posterior(np.delete(_X, _io), np.delete(_y, _io), XS, _kern, _s2)
    _mu_g, _v_g = gp_posterior(_X, _y_bad, XS, _kern, _s2)

    _w = np.ones(25)
    for _ in range(50):
        _m, _v = gp_posterior(_X, _y_bad, _X, _kern, _s2 / _w)
        _w = (_nu + 1) / (_nu + ((_y_bad - _m) ** 2 + _v) / _s2)
    _mu_t, _v_t = gp_posterior(_X, _y_bad, XS, _kern, _s2 / _w)

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(13, 3.3), gridspec_kw={"width_ratios": [1, 1, 0.75]})
    for _ax, _mu, _v, _col, _title in [(_ax1, _mu_g, _v_g, C["orange"], "Gaussian likelihood"),
                                       (_ax2, _mu_t, _v_t, C["blue"], r"Student-t likelihood ($\nu = 4$, EM)")]:
        plot_posterior(_ax, XS, _mu, np.sqrt(_v), _X, _y_bad, F_TRUE, color=_col)
        _ax.scatter(_X[_io], _y_bad[_io], s=70, facecolor="none", edgecolor=C["red"], lw=1.5, zorder=6,
                    label="outlier")
        _ax.set(title=_title, xlabel=r"$x$", ylim=(-2.4, 2.4))
    _ax1.legend(loc="lower left", ncols=3, fontsize=7)
    _ax3.vlines(_X, 0, _w, color=C["blue"], lw=2)
    _ax3.plot(_X[_io], _w[_io], "o", color=C["red"], ms=6)
    _ax3.set(title=r"Student-t weights $w_i$", xlabel=r"$x$", ylim=(0, 1.35))
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.5 Numerical failure: ill-conditioned kernel matrices

    $K$ is positive definite in exact arithmetic, but in floating point it can be numerically singular. This happens
    when points are close together relative to $\ell$: rows of $K$ become nearly identical. Smooth kernels are the
    worst: with 50 random points, the SE matrix is numerically singular (condition number past $10^{16}$, the float64
    limit) for any $\ell \gtrsim 0.035$, and the Cholesky factorization fails. Matérn kernels degrade much more
    gracefully.

    Two defenses, both standard in GP libraries:

    - **Jitter**: add a small $\epsilon I$ (around $10^{-6}$ on standardized data) to $K$. It caps the condition
      number at about $1/\epsilon$, at the cost of treating the data as very slightly noisy.
    - **Never invert $K$**: use a Cholesky solve. The right panel runs the same posterior-variance formula two ways
      on an ill-conditioned noiseless problem. The explicit inverse returns **negative variances**; Cholesky with
      jitter is fine.
    """)
    return


@app.cell(hide_code=True)
def _(C, cholesky, k_m52, k_se, np, plt, show, solve_triangular):
    _rng = np.random.default_rng(1)
    _X = np.sort(_rng.uniform(0, 1, 50))
    _ls = np.logspace(-2.3, 0.3, 50)

    def _cond(K):
        _ev = np.linalg.eigvalsh(K)
        return _ev.max() / max(_ev.min(), _ev.max() * 1e-18)

    def _chol_ok(K):
        try:
            np.linalg.cholesky(K)
            return True
        except np.linalg.LinAlgError:
            return False

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    _cases = [("SE, no jitter", k_se, 0.0, C["orange"]), (r"SE + $10^{-6} I$ jitter", k_se, 1e-6, C["blue"]),
              ("Matérn 5/2, no jitter", k_m52, 0.0, C["aqua"])]
    for _name, _k, _jit, _col in _cases:
        _Ks = [_k(_X, _X, ls=l) + _jit * np.eye(50) for l in _ls]
        _c = np.array([_cond(K) for K in _Ks])
        _ok = np.array([_chol_ok(K) for K in _Ks])
        _ax1.plot(_ls, _c, color=_col, label=_name)
        if (~_ok).any():
            _ax1.plot(_ls[~_ok], _c[~_ok], "x", color=_col, ms=5, label="Cholesky fails")
    _ax1.axhline(1 / np.finfo(float).eps, color=C["ink2"], ls="--", lw=1)
    _ax1.text(_ls[0], 1.5 / np.finfo(float).eps, r"$1/\epsilon_{\mathrm{machine}}$", color=C["ink2"], fontsize=8)
    _ax1.set(xscale="log", yscale="log", xlabel=r"lengthscale $\ell$", ylabel=r"condition number of $K$",
             title="50 random points in [0, 1]", ylim=(1, 1e20))
    _ax1.legend(loc="lower right", fontsize=7)

    _X2 = np.sort(_rng.uniform(0, 1, 20))
    _xs = np.linspace(0, 1, 400)
    _K = k_se(_X2, _X2, ls=0.3)
    _Ks = k_se(_X2, _xs, ls=0.3)
    _v_inv = 1 - (_Ks * (np.linalg.inv(_K) @ _Ks)).sum(0)
    _V = solve_triangular(cholesky(_K + 1e-6 * np.eye(20), lower=True), _Ks, lower=True)
    _v_chol = 1 - (_V**2).sum(0)
    _ax2.plot(_xs, _v_inv, color=C["orange"], lw=1.1, label=r"np.linalg.inv($K$)")
    _ax2.plot(_xs, _v_chol, color=C["blue"], lw=2, label=r"Cholesky of $K + 10^{-6} I$")
    _ax2.axhline(0, color=C["ink2"], lw=0.8)
    _ax2.plot(_X2, np.zeros(20), "|", color=C["ink"], ms=8)
    _ax2.set(title=rf"Posterior variance $\sigma^2(x)$, SE $\ell = 0.3$, cond$(K) \approx 10^{{{np.log10(_cond(_K)):.0f}}}$",
             xlabel=r"$x$", ylabel=r"$\sigma^2(x)$ (must be $\geq 0$)")
    _ax2.legend(loc="upper right", fontsize=7)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.6 High dimensions: distances stop being informative

    A stationary kernel only sees distances. In $d$ dimensions, the distances between random points **concentrate**:
    they all become nearly equal (left). With a lengthscale suited to 1D, every pair of points then looks
    uncorrelated, so the GP learns nothing from its neighbours and falls back to the prior (middle). ARD makes it
    worse, because $d$ lengthscales must be learned from few points.

    What helps:

    - **Dimension-scaled lengthscale priors** (Hvarfner, Hellsten & Nardi 2024): a prior whose location grows like
      $\sqrt d$ keeps typical correlations reasonable. Now the default in BoTorch.
    - **Sparsity priors** such as SAAS: a half-Cauchy prior on inverse squared lengthscales that switches most
      inputs off unless the data insists (right).
    - **Structure**: additive kernels over groups of inputs, or low-dimensional embeddings $f(x) = g(P^\top x)$.
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
    ## 7. Scaling Gaussian processes

    Exact inference costs $O(n^3)$ time and $O(n^2)$ memory: instant at $n = 1{,}000$, hours and 80 GB at
    $n = 100{,}000$. The main fixes, each avoiding the full $n \times n$ computation differently:

    - **Inducing points** (Titsias 2009; SVGP, Hensman et al. 2013): summarize $f$ through its values at
      $m \ll n$ locations, $O(nm^2)$. Titsias chooses them by maximizing a lower bound on the marginal likelihood,
      $\log\mathcal N(\mathbf y\mid 0, Q_{nn} + \sigma_n^2 I) - \tfrac{1}{2\sigma_n^2}\mathrm{tr}(K_{nn} - Q_{nn})$
      with $Q_{nn} = K_{nm}K_{mm}^{-1}K_{mn}$; SVGP makes it minibatchable for millions of points. The weakness: a
      summary blurs fine detail.
    - **Iterative solvers** (GPyTorch): conjugate gradients needs only products $K\mathbf v$, which GPUs do fast.
      Exact GPs on a million points.
    - **Random Fourier features** and **pathwise conditioning**: below.

    Drag the number of inducing points: with too few, the sparse GP smooths over the sharp features.
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
    f_test,
    gp_posterior,
    k_se,
    mo,
    np,
    plot_posterior,
    plt,
    show,
    sparse_m,
):
    _rng = np.random.default_rng(4)
    _n, _sn2 = 300, 0.1**2
    _X = np.sort(_rng.uniform(0, 1, _n))
    _y = f_test(_X) + 0.1 * _rng.standard_normal(_n)
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

    The spectral density is a distribution over frequencies. Sample $D$ of them and the kernel becomes a dot product
    of explicit cosine features, $k(x,x') \approx \phi(x)^\top\phi(x')$ with $\phi(x) = \sqrt{2/D}\cos(\Omega x + b)$
    (Rahimi & Recht 2007). The GP becomes Bayesian linear regression on $D$ features, and a posterior sample is an
    explicit function you can evaluate anywhere.

    The catch is **variance starvation**: with many data points and few features, the data pin down all $D$ weights,
    and the posterior becomes overconfident where there is no data (middle panel, in the gap). **Matheron's rule**
    fixes it (Wilson et al. 2020): a posterior sample is a prior sample plus a data-driven correction,

    $$
    (f \mid \mathbf{y})(\cdot) \overset{d}{=} f(\cdot) + k(\cdot, X)(K + \sigma_n^2 I)^{-1}\big(\mathbf{y} - f(X) - \boldsymbol\varepsilon\big)
    $$

    Take the prior sample $f$ from random features (accurate for the prior at any $n$) and the correction from the
    exact kernel, and the error bars are right again (right panel).
    """)
    return


@app.cell(hide_code=True)
def _(C, SERIES, XS, cho_solve, cholesky, f_test, gp_posterior, k_se, np, plt, show):
    _ell = 0.08
    _rng = np.random.default_rng(0)

    def _features(x, W, b):
        return np.sqrt(2 / len(W)) * np.cos(np.outer(x, W) + b)

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.4))
    _r = np.linspace(0, 0.4, 200)
    _axes[0].plot(_r, np.exp(-0.5 * (_r / _ell) ** 2), color=C["ink"], lw=2.2, label="exact SE")
    for _i, _D in enumerate([10, 100, 1000]):
        _W, _b = _rng.normal(0, 1 / _ell, _D), _rng.uniform(0, 2 * np.pi, _D)
        _axes[0].plot(_r, (_features(np.zeros(1), _W, _b) @ _features(_r, _W, _b).T).ravel(),
                      color=SERIES[_i], lw=1.1, label=rf"RFF, $D = {_D}$")
    _axes[0].set(xlabel=r"$r$", title=r"Random Fourier features: $\phi(x)^\top\phi(x')$ vs $k(r)$", ylim=(-0.4, 1.1))
    _axes[0].legend()

    _n, _sn = 1000, 0.1
    _X = np.sort(np.r_[_rng.uniform(0, 0.4, _n // 2), _rng.uniform(0.75, 1.0, _n // 2)])
    _y = f_test(_X) + _sn * _rng.standard_normal(_n)
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
    ## 8. Takeaways

    **A practical recipe**

    1. Normalize inputs to $[0, 1]^d$ (log-transform inputs that act multiplicatively) and standardize $y$.
    2. Start with **Matérn-5/2 + ARD**. Add structure you actually believe in: a linear term for trends, periodic
       terms for cycles, additive kernels for groups of inputs.
    3. Fit hyperparameters by marginal likelihood with **several restarts** and **sensible priors**. With very little
       data, integrate them out.
    4. Use a Cholesky solve with a little jitter; never invert $K$.
    5. **Check the model** before trusting it: LOO residuals, and the failure modes below.

    **Symptom → likely cause → fix**

    | Symptom | Likely cause | Fix |
    | :--- | :--- | :--- |
    | Predictions fall back to the mean outside the data | Zero mean, stationary kernel | Linear term or mean function; don't extrapolate |
    | Error bars too wide in flat regions, fit too stiff in busy ones | Nonstationary $f$ | Input warping, nonstationary or local models |
    | LOO residuals fan out with $x$ | Input-dependent noise | Heteroscedastic likelihood, transform $y$ |
    | A few huge LOO residuals, fit bends toward them | Outliers | Student-t likelihood |
    | Spiky mean that interpolates noise, or flat mean with huge noise | Bad local optimum of the marginal likelihood | Restarts, priors on $\theta$ |
    | Cholesky fails, negative variances | Ill-conditioned $K$ | Jitter, Matérn instead of SE, remove duplicates |
    | Everything reverts to the prior in many dimensions | Distance concentration | Dimension-scaled priors, sparsity priors, structure |
    | Too slow or out of memory | $O(n^3)$ exact inference | GPyTorch, inducing points, random features |

    **Read next**: [Rasmussen & Williams, *Gaussian Processes for Machine Learning*](http://gaussianprocess.org/gpml/)
    (free online; chapters 2, 4 and 5), the Distill article
    [*A Visual Exploration of Gaussian Processes*](https://distill.pub/2019/visual-exploration-gaussian-processes/),
    Duvenaud's [kernel cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/),
    [Kanagawa et al. (2018)](https://arxiv.org/abs/1807.02582) on GPs and kernel methods, and
    [Liu et al. (2020)](https://arxiv.org/abs/1807.01065) on scaling. Full references are in `docs/gp-and-bo.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
