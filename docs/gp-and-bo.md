# Gaussian Processes and Bayesian Optimization

Sep 26, 2026 · @Ivan Havlytskyi

Bayesian optimization (BO) finds the maximum of an expensive black-box function by fitting a probabilistic surrogate, usually a Gaussian process (GP), and choosing each next query to balance exploring uncertain regions against exploiting promising ones. It shines when one evaluation costs hours, dollars or a lab experiment, and you can afford tens to a few hundred of them.

The setting is simple to state. We want $x^* = \arg\max_{x} f(x)$ over a compact domain $\mathcal{X} \subset \mathbb{R}^d$. We can query $f$, possibly with noise, but we get no gradients and no closed form. Each query is expensive, so the question is not how to optimize, but where to look next.

BO answers that question with two ingredients. A **surrogate model** turns the evaluations so far into a posterior belief over $f$. An **acquisition function** turns that belief into a score for every candidate $x$, and we query the $\arg\max$. The GP is the default surrogate because it gives calibrated uncertainty, closed-form updates and a clean way to encode assumptions like smoothness.

The list of places BO is used is long. It tunes deep-learning hyperparameters ([Snoek et al. 2012](https://arxiv.org/abs/1206.2944)), tuned AlphaGo before its match with Lee Sedol ([Chen et al. 2018](https://arxiv.org/abs/1812.06855)), runs online A/B experiments at Meta ([Letham et al. 2019](https://arxiv.org/abs/1706.07094)), and drives self-driving labs in chemistry and materials science.

This post goes from the math of GPs to the current state of BO:

- **Gaussian processes:** conditioning, the posterior, and learning hyperparameters from the marginal likelihood.
- **Kernels:** what a kernel says about the functions you believe in, and the link to RKHS theory.
- **The BO loop and acquisition functions:** from probability of improvement to entropy search, with derivations.
- **Theory:** what regret bounds actually guarantee.
- **Practice:** Monte Carlo acquisition functions, batch BO and BoTorch.
- **Scaling:** sparse GPs, structured kernels and pathwise sampling.
- **High dimensions:** embeddings, trust regions, sparsity priors, and the 2024 result that vanilla BO works better than we thought.
- **Beyond vanilla BO:** constraints, multiple fidelities, multiple objectives and non-GP surrogates.

Notation: we maximize $f$. Observations are $y = f(x) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$. The dataset after $n$ evaluations is $D_n = \{(x_i, y_i)\}_{i=1}^n$.

## Background: Gaussian processes

### Everything starts with Gaussian conditioning

The whole GP machinery rests on one fact: if two vectors are jointly Gaussian, the conditional of one given the other is Gaussian with a closed form. Split a jointly Gaussian vector into a block $\mathbf{f}$ we observe and a block $\mathbf{f}_*$ we want to predict:

$$
\begin{bmatrix} \mathbf{f} \\ \mathbf{f}_* \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mathbf{m} \\ \mathbf{m}_* \end{bmatrix}, \begin{bmatrix} K & K_* \\ K_*^\top & K_{**} \end{bmatrix} \right)
\quad\Longrightarrow\quad
\mathbf{f}_* \mid \mathbf{f} \sim \mathcal{N}\left( \mathbf{m}_* + K_*^\top K^{-1} (\mathbf{f} - \mathbf{m}),\; K_{**} - K_*^\top K^{-1} K_* \right)
$$

Two things are worth noticing. The conditional mean moves linearly with the observed values. The conditional covariance does not depend on the observed values at all, only on where they were observed. That second fact will matter a lot for batch BO.

### From vectors to functions

A **Gaussian process** is a collection of random variables, any finite subset of which is jointly Gaussian. We write $f \sim \mathcal{GP}(m, k)$, where $m(x) = \mathbb{E}[f(x)]$ is the mean function and $k(x, x') = \mathrm{Cov}(f(x), f(x'))$ is the covariance function, or kernel.

The definition sounds infinite-dimensional and scary, but the marginalization property of Gaussians saves us. To reason about $f$ at the $n$ training inputs and a test input, we only ever touch the $(n+1)$-dimensional Gaussian over those points. The infinitely many other points integrate out for free. That is why a GP is computable at all.

The kernel must produce a positive semi-definite matrix $K_{ij} = k(x_i, x_j)$ for every finite set of inputs. The mean function is often set to a constant, because the kernel does most of the modeling work.

### The posterior predictive

With noisy observations $y = f(X) + \varepsilon$, the observed vector has covariance $K + \sigma_n^2 I$. Applying the conditioning rule with a zero prior mean gives the posterior at a test point $x_*$:

$$
\mu_n(x_*) = \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y},
\qquad
\sigma_n^2(x_*) = k(x_*, x_*) - \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{k}_*
$$

Here $\mathbf{k}_*$ is the vector of covariances between $x_*$ and the $n$ training inputs. The posterior mean is a weighted sum of kernel functions centered on the data, $\mu_n(x) = \sum_i \alpha_i k(x_i, x)$ with $\alpha = (K + \sigma_n^2 I)^{-1} \mathbf{y}$. That is exactly kernel ridge regression with regularizer $\sigma_n^2$, which is the first of several places where GPs and kernel methods turn out to be the same object seen from two angles.

The posterior variance starts at the prior variance $k(x_*, x_*)$ and shrinks by an amount that depends on how well the training inputs "explain" the test point. Near data it collapses toward the noise level; far from data it returns to the prior. This behavior is exactly what an optimizer needs: it knows where it has not looked.

In practice nobody inverts $K$. You compute a Cholesky factorization $LL^\top = K + \sigma_n^2 I$ once, at $O(n^3)$ time and $O(n^2)$ memory. Each prediction then costs $O(n)$ for the mean and $O(n^2)$ for the variance. For typical BO budgets of a few hundred points, this is cheap.

### Learning hyperparameters from the marginal likelihood

The kernel has hyperparameters $\theta$, such as lengthscales, an output scale and the noise variance. The standard way to set them is type-II maximum likelihood: maximize the probability of the observed data with $f$ integrated out. For a GP this integral is closed-form:

$$
\log p(\mathbf{y} \mid X, \theta) =
\underbrace{-\tfrac{1}{2} \mathbf{y}^\top K_y^{-1} \mathbf{y}}_{\text{data fit}}
\;\underbrace{-\tfrac{1}{2} \log |K_y|}_{\text{complexity penalty}}
\;-\; \tfrac{n}{2} \log 2\pi,
\qquad K_y = K_\theta + \sigma_n^2 I
$$

The first term rewards fitting the data. The second term penalizes flexible models: a short lengthscale spreads prior mass over many wiggly functions, which inflates the determinant. The balance between them is an automatic Occam's razor, and it is the main reason GPs rarely overfit badly despite being nonparametric.

The gradient is also closed-form, which makes L-BFGS the usual optimizer:

$$
\frac{\partial}{\partial \theta_j} \log p(\mathbf{y} \mid X, \theta)
= \tfrac{1}{2} \operatorname{tr}\left( \left( \boldsymbol{\alpha}\boldsymbol{\alpha}^\top - K_y^{-1} \right) \frac{\partial K_y}{\partial \theta_j} \right),
\qquad \boldsymbol{\alpha} = K_y^{-1} \mathbf{y}
$$

The marginal likelihood is non-convex in $\theta$ and often multimodal. A common failure is a solution that explains everything as noise with a very long lengthscale, and another is one that interpolates noise with a very short lengthscale. Multiple restarts and sensible priors on $\theta$, discussed in the practice section, are the usual cures.

### The weight-space view

There is a second, equally useful way to see a GP. Take a linear model $f(x) = \phi(x)^\top w$ over some features $\phi$, with a Gaussian prior $w \sim \mathcal{N}(0, \Sigma_p)$. Then $f$ is a GP with kernel $k(x, x') = \phi(x)^\top \Sigma_p \phi(x')$.

The kernel trick runs this backwards. A kernel like the squared exponential corresponds to infinitely many features, so a GP is Bayesian linear regression in a feature space you never have to construct. [Neal (1996)](https://doi.org/10.1007/978-1-4612-0745-0) showed that a one-hidden-layer Bayesian neural network converges to a GP as its width goes to infinity. [Lee et al. (2018)](https://arxiv.org/abs/1711.00165) extended this to deep networks (the NNGP). The weight-space view also underlies the random-feature approximations in the scaling section.

For the full treatment, Chapters 2 and 5 of [Rasmussen & Williams (2006)](http://gaussianprocess.org/gpml/) remain the reference. For intuition first, the interactive [Distill article on GPs](https://distill.pub/2019/visual-exploration-gaussian-processes/) is excellent.

## Kernels: what you believe about f

The kernel is the whole prior. It decides how smooth sample functions are, how far information travels from an observation, which inputs matter, and whether $f$ is periodic or additive. In BO the kernel choice often matters more than the acquisition function.

### Stationary workhorses

Most BO uses stationary kernels, which depend only on $r = \|x - x'\|$ (after scaling each dimension by its lengthscale). The two you will meet constantly are the squared exponential (also called RBF) and the Matérn family:

$$
k_{\text{SE}}(r) = \sigma_f^2 \exp\left( -\frac{r^2}{2\ell^2} \right),
\qquad
k_{\nu}(r) = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{\sqrt{2\nu}\, r}{\ell} \right)^{\nu} K_{\nu}\left( \frac{\sqrt{2\nu}\, r}{\ell} \right)
$$

Here $\ell$ is the lengthscale, $\sigma_f^2$ the output scale, and $K_\nu$ a modified Bessel function. The smoothness parameter $\nu$ controls differentiability: a Matérn-$\nu$ process is $k$ times mean-square differentiable exactly when $\nu > k$. As $\nu \to \infty$ the Matérn kernel becomes the squared exponential, whose samples are infinitely differentiable.

The half-integer cases have simple closed forms. $\nu = 1/2$ gives the exponential kernel, whose samples are continuous but rough, like Brownian motion. $\nu = 3/2$ and $\nu = 5/2$ give once- and twice-differentiable samples. Matérn-5/2 is the standard BO default:

$$
k_{5/2}(r) = \sigma_f^2 \left( 1 + \frac{\sqrt{5}\, r}{\ell} + \frac{5 r^2}{3 \ell^2} \right) \exp\left( -\frac{\sqrt{5}\, r}{\ell} \right)
$$

The reason for that default comes from [Snoek et al. (2012)](https://arxiv.org/abs/1206.2944). They argued that the squared exponential is unrealistically smooth for real objectives like validation loss, and Matérn-5/2 performed better in their experiments. A too-smooth prior makes the GP overconfident between observations, which starves exploration.

### Automatic relevance determination

With one lengthscale per input dimension, $r$ becomes a weighted distance: $r^2 = \sum_d (x_d - x'_d)^2 / \ell_d^2$. This is called automatic relevance determination (ARD). A large $\ell_d$ means $f$ barely changes along dimension $d$, so the marginal likelihood can effectively switch irrelevant inputs off.

ARD is almost always on in modern BO. It is also where high-dimensional problems start to hurt, because $d$ lengthscales must be learned from very few points. The high-dimensional section returns to this.

### Building kernels from parts

Valid kernels are closed under addition and multiplication. That gives a small grammar for encoding structure:

- **Sum, $k_1 + k_2$:** $f$ is a sum of independent functions, one from each kernel. Summing kernels over disjoint groups of inputs gives an additive model, which is the basis of additive BO.
- **Product, $k_1 \times k_2$:** two points are similar only if they are similar under both kernels. Multiplying a periodic kernel by a squared exponential gives a locally periodic function whose pattern can drift.
- **Periodic:** $k(r) = \sigma_f^2 \exp(-2\sin^2(\pi r / p) / \ell^2)$ for a known or learned period $p$.
- **Linear:** $k(x, x') = \sigma_f^2 x^\top x'$, which recovers Bayesian linear regression.

[Duvenaud et al. (2013)](https://arxiv.org/abs/1302.4922) searched over this grammar with the marginal likelihood as the score, which became the "Automatic Statistician". [Duvenaud's Kernel Cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/) is the best short visual guide to what each combination does.

### The spectral view

Bochner's theorem says a continuous stationary kernel is the Fourier transform of a non-negative measure, its spectral density. The squared exponential has a Gaussian spectral density, and the Matérn kernel has a Student-t-like density whose heavier tails put more power at high frequencies, which is where the roughness comes from.

This view has two practical payoffs. [Wilson & Adams (2013)](https://arxiv.org/abs/1302.4245) modeled the spectral density as a Gaussian mixture, which gives a highly expressive "spectral mixture" kernel. And sampling frequencies from the spectral density yields the random Fourier features covered in the scaling section.

### The RKHS connection, and a subtle trap

Every kernel defines a reproducing kernel Hilbert space (RKHS), a space of functions with norm $\|f\|_{\mathcal{H}}$. The GP posterior mean is the minimum-norm interpolant in that space when there is no noise, and the kernel ridge regression solution when there is. In the noise-free case, the posterior standard deviation equals the worst-case interpolation error over the unit ball of the RKHS.

The trap: for the kernels used in practice, GP sample paths almost surely do **not** lie in the RKHS of their own kernel. They are rougher than every function in it. [Kanagawa et al. (2018)](https://arxiv.org/abs/1807.02582) is the definitive review of these connections and the ways the two worlds differ.

This matters for BO theory. Some regret analyses assume $f$ is drawn from the GP prior (the Bayesian setting). Others assume $f$ is a fixed function with bounded RKHS norm (the frequentist setting). The two assumptions describe different function classes and give different bounds.

## The Bayesian optimization loop

### A short history

The core ideas are older than machine learning's interest in them. [Kushner (1964)](https://doi.org/10.1115/1.3653121) modeled a one-dimensional objective as a Brownian motion and picked points by their probability of improving on the best value so far. [Močkus (1975)](https://doi.org/10.1007/3-540-07165-2_55) proposed expected improvement and framed the whole problem in decision-theoretic terms.

The method became practical with [Jones, Schonlau & Welch (1998)](https://doi.org/10.1023/A:1008306431147). Their Efficient Global Optimization (EGO) algorithm paired a kriging model, which is a GP by another name from geostatistics, with expected improvement. EGO is still the template for most BO today.

Machine learning adopted it around 2010–2012. [Srinivas et al. (2010)](https://arxiv.org/abs/0912.3995) gave the first regret bounds, and [Snoek et al. (2012)](https://arxiv.org/abs/1206.2944) showed BO could tune deep networks better than human experts. Their open-source Spearmint made BO a standard tool for hyperparameter tuning.

### The algorithm

Given a prior GP, a budget of $N$ evaluations and an acquisition function $\alpha$:

1. **Initial design.** Evaluate $f$ at a handful of space-filling points, typically a scrambled Sobol sequence or Latin hypercube. A common rule of thumb is $2d$ to $10d$ points, or fewer when evaluations are very costly.
2. **Fit the surrogate.** Condition the GP on $D_n$ and fit its hyperparameters, by maximum likelihood or by sampling.
3. **Optimize the acquisition.** Find $x_{n+1} = \arg\max_x \alpha(x \mid D_n)$ over $\mathcal{X}$.
4. **Evaluate.** Observe $y_{n+1} = f(x_{n+1}) + \varepsilon$ and add it to the data.
5. **Repeat** until the budget runs out, then **recommend** a point: either the best observed $y$, or the maximizer of the posterior mean when observations are noisy.

### Two optimization problems, not one

BO trades one hard problem for a sequence of easier ones. The outer problem is expensive and black-box. The inner problem, maximizing $\alpha$, is cheap to evaluate and differentiable, but it is still non-convex and often highly multimodal.

The inner problem deserves more respect than it usually gets. Acquisition surfaces tend to be flat almost everywhere with sharp peaks near promising regions. The standard recipe is multi-start gradient ascent: score a large batch of random or quasi-random candidates, start L-BFGS-B from the best few, and keep the winner. Weak inner optimization quietly turns a good acquisition function into a bad one, and several "improvements" in the literature turned out to be improvements in inner optimization ([Ament et al. 2023](https://arxiv.org/abs/2310.20708)).

### Exploration versus exploitation

Every acquisition function resolves the same tension. Querying where $\mu_n$ is high exploits what we know. Querying where $\sigma_n$ is high explores and reduces uncertainty. A pure exploiter gets stuck on the first decent local optimum; a pure explorer does uniform sampling with extra steps.

BO is closely related to the multi-armed bandit problem with infinitely many correlated arms. The kernel supplies the correlation: pulling one arm tells you about its neighbors. That is why bandit ideas like UCB and Thompson sampling carry over almost directly, and why the regret framework from bandits is the natural language for BO theory.

## Acquisition functions

Acquisition functions fall into three families: those that reward **improvement** over the incumbent, those that use **optimism** in the face of uncertainty, and those that maximize **information** about the optimum. Throughout, $\mu(x)$ and $\sigma(x)$ are the current posterior mean and standard deviation, $f^+$ is the best value observed so far, and $\Phi$ and $\phi$ are the standard normal CDF and PDF.

### Probability of improvement (PI)

The oldest idea: pick the point most likely to beat the incumbent.

$$
\alpha_{\text{PI}}(x) = P\left( f(x) > f^+ + \xi \right) = \Phi\left( \frac{\mu(x) - f^+ - \xi}{\sigma(x)} \right)
$$

PI ignores how much a point might improve, only whether it will. With $\xi = 0$ it happily picks points a hair above the incumbent with near certainty, so it over-exploits. The margin $\xi$ helps, but tuning it well is its own problem.

### Expected improvement (EI)

EI fixes PI by weighting improvements by their size. Define the improvement $I(x) = \max(f(x) - f^+, 0)$ and take its expectation under the posterior. Writing $z = (\mu(x) - f^+)/\sigma(x)$ and substituting $f(x) = \mu(x) + \sigma(x) u$ with $u$ standard normal, the expectation splits into two integrals over $u > -z$:

$$
\alpha_{\text{EI}}(x)
= \int_{-z}^{\infty} \sigma(x)(u + z)\, \phi(u)\, du
= \big(\mu(x) - f^+\big)\, \Phi(z) + \sigma(x)\, \phi(z)
$$

The first term uses $\int \phi(u)\,du = \Phi(z)$ over that range, and the second uses $\int u\,\phi(u)\,du = \phi(z)$, since $\phi'(u) = -u\,\phi(u)$. The two terms read naturally. The first rewards a high mean, which is exploitation. The second rewards high uncertainty, which is exploration.

EI has no tuning parameter, a closed form and a closed-form gradient, which is why it has been the default for 25 years. It is also myopic: it is the one-step Bayes-optimal policy only if you plan to stop right after the next evaluation and the final answer is the best observed value.

**EI has a numerical problem that went mostly unnoticed until 2023.** In regions far from the incumbent, EI and its gradient underflow to exactly zero in floating point. Gradient-based inner optimization then sees a flat landscape and stalls wherever it started. [Ament et al. (2023)](https://arxiv.org/abs/2310.20708) introduced LogEI, which computes log EI stably through an asymptotic expansion. It has the same maximizer, but its landscape is well-behaved. LogEI and its batch variants beat standard EI across many benchmarks and match or beat several more complex acquisition functions, which suggests many past comparisons were partly measuring inner-optimization failures.

**Noise breaks the incumbent.** With noisy observations, $f^+$ is itself uncertain; plugging in the best noisy $y$ rewards lucky draws. Noisy EI ([Letham et al. 2019](https://arxiv.org/abs/1706.07094)) integrates EI over the posterior of the function values at the already-observed points, using quasi-Monte Carlo. It is the standard choice in online experimentation.

### Upper confidence bound (GP-UCB)

The optimism principle from bandits: act as if the function is as good as it plausibly could be.

$$
\alpha_{\text{UCB}}(x) = \mu(x) + \sqrt{\beta_t}\, \sigma(x)
$$

The parameter $\beta_t$ sets the exploration level directly. [Srinivas et al. (2010)](https://arxiv.org/abs/0912.3995) showed that with $\beta_t$ growing roughly logarithmically in $t$, GP-UCB has sublinear regret (see the theory section). The theoretical schedules are very conservative, so practitioners usually use a small constant $\beta$ instead. UCB is simple and cheap, but that knob is its weakness.

### Thompson sampling (TS)

Draw one function $\tilde f$ from the posterior and query its maximizer. Points are selected with exactly their posterior probability of being the optimum, so exploration is automatic and parameter-free ([Russo & Van Roy 2014](https://arxiv.org/abs/1301.2609)).

The catch is that you need a whole function sample, not a sample at a fixed set of points. Exact joint sampling costs $O(m^3)$ for $m$ candidate points. Random Fourier features and pathwise conditioning (in the scaling section) make TS practical. TS is also naturally parallel: draw $q$ samples, get $q$ diverse points. That is why it underlies TuRBO and many large-batch methods.

### Knowledge gradient (KG)

EI assumes you will report the best observed point. If you will instead report the maximizer of the final posterior mean, the value of a new observation is how much it raises that maximum. The knowledge gradient ([Frazier, Powell & Dayanik 2009](https://doi.org/10.1287/ijoc.1080.0314)) measures exactly this:

$$
\alpha_{\text{KG}}(x) = \mathbb{E}_{y}\left[ \max_{x'} \mu_{n+1}(x') \;\middle|\; x_{n+1} = x \right] - \max_{x'} \mu_n(x')
$$

KG can value a point even if it will never itself be the best, because observing it can change your beliefs elsewhere. That makes it strong with noise and in multi-fidelity problems, where cheap evaluations are informative without being candidates. The cost is a nested optimization inside an expectation. The one-shot formulation in BoTorch ([Balandat et al. 2020](https://arxiv.org/abs/1910.06403)) turns it into a single joint optimization over $x$ and a set of fantasy maximizers.

### Information-theoretic acquisitions

These choose the query that most reduces uncertainty about the optimum itself, measured by entropy.

**Entropy search (ES)** ([Hennig & Schuler 2012](https://arxiv.org/abs/1112.1217)) targets the distribution of the maximizer location, $p(x^* \mid D)$. It asks how much observing $y$ at $x$ would shrink the entropy of that distribution. It is principled but expensive, because $p(x^* \mid D)$ has no closed form and must be approximated on a discretization.

**Predictive entropy search (PES)** ([Hernández-Lobato et al. 2014](https://arxiv.org/abs/1406.2541)) uses the symmetry of mutual information to swap the roles of $y$ and $x^*$. Instead of the entropy of $x^*$, you compute entropies of the one-dimensional predictive distribution of $y$, with and without conditioning on sampled optima:

$$
\alpha_{\text{PES}}(x) = H\big[ p(y \mid D, x) \big] - \mathbb{E}_{x^* \sim p(x^* \mid D)} \Big[ H\big[ p(y \mid D, x, x^*) \big] \Big]
$$

The conditioning on $x^*$ still needs expectation propagation, which makes PES accurate but intricate to implement.

**Max-value entropy search (MES)** ([Wang & Jegelka 2017](https://arxiv.org/abs/1703.01968)) makes the key simplification: target the optimal value $f^*$, a scalar, rather than the location. Conditioned on $f^*$, the predictive distribution of $f(x)$ is a Gaussian truncated at $f^*$, whose entropy is closed-form. With $\gamma(x) = (f^* - \mu(x))/\sigma(x)$ and a set $F$ of sampled max-values:

$$
\alpha_{\text{MES}}(x) \approx \frac{1}{|F|} \sum_{f^* \in F} \left[ \frac{\gamma(x)\, \phi(\gamma(x))}{2\, \Phi(\gamma(x))} - \log \Phi(\gamma(x)) \right]
$$

MES matches or beats PES at a fraction of the cost, and it made information-based BO practical. Joint entropy search ([Hvarfner et al. 2022](https://arxiv.org/abs/2206.04771)) later targeted the location and value together.

### Which one should you use?

There is no universal winner, but the evidence points to a sensible default order:

| Situation | Good default | Why |
| --- | --- | --- |
| Low noise, sequential, $d \lesssim 20$ | LogEI | Robust, no tuning, easy inner optimization |
| Noisy observations | Noisy EI (LogNEI) or KG | Handle an uncertain incumbent properly |
| Large batches or high $d$ | Thompson sampling, often inside a trust region | Cheap, parallel, diverse |
| Multi-fidelity or cost-aware | KG or MES variants | Value information without needing a candidate |
| You want theory to match practice | GP-UCB | The cleanest regret guarantees |

Mixing acquisitions is also an option. GP-Hedge ([Hoffman, Brochu & de Freitas 2011](https://arxiv.org/abs/1009.5419)) treats acquisition functions as arms of a bandit and learns which ones pay off on the current problem.

## Theory: what the guarantees actually say

BO theory mostly bounds **regret**, and the key quantity in almost every bound is the **maximum information gain** $\gamma_T$, which measures how complex the kernel's function class is. The bounds are informative about how kernels and dimensions scale, but far too loose to set hyperparameters with.

### Regret

Let $x^*$ be the true maximizer. After $T$ queries, two quantities matter:

$$
r_T = f(x^*) - f(\hat{x}_T) \quad \text{(simple regret)},
\qquad
R_T = \sum_{t=1}^{T} \big( f(x^*) - f(x_t) \big) \quad \text{(cumulative regret)}
$$

Simple regret is what optimization cares about: how good is the final recommendation? Cumulative regret is the bandit quantity: how much did we lose along the way? A sublinear $R_T$ ($R_T / T \to 0$) implies the best point found so far has simple regret going to zero, so bounds on $R_T$ are the usual route to both.

### Maximum information gain

The information gain from observing $y_A$ at a set of points $A$ is the mutual information between those observations and $f$. For a GP with Gaussian noise it is closed-form, and $\gamma_T$ is its worst case over $T$ points:

$$
\gamma_T = \max_{A \subset X,\, |A| = T} \; \tfrac{1}{2} \log \left| I + \sigma_n^{-2} K_A \right|
$$

$\gamma_T$ grows slowly for simple kernels and fast for rough or high-dimensional ones. It is determined by how fast the kernel's eigenvalues decay:

| Kernel | Growth of $\gamma_T$ | Source |
| --- | --- | --- |
| Linear | $O(d \log T)$ | [Srinivas et al. 2010](https://arxiv.org/abs/0912.3995) |
| Squared exponential | $O((\log T)^{d+1})$ | [Srinivas et al. 2010](https://arxiv.org/abs/0912.3995) |
| Matérn-$\nu$ | $\tilde{O}(T^{d/(2\nu+d)})$ | [Vakili et al. 2021](https://arxiv.org/abs/2009.06966) |

The Matérn row shows the curse of dimensionality directly. As $d$ grows relative to $\nu$, the exponent approaches 1, and $\gamma_T$ becomes nearly linear in $T$.

### GP-UCB bounds in two settings

In the **Bayesian setting**, $f$ is a sample from the GP prior. [Srinivas et al. (2010)](https://arxiv.org/abs/0912.3995) proved that GP-UCB with a suitable $\beta_t$, logarithmic in $t$, achieves with high probability:

$$
R_T = O^*\left( \sqrt{T\, \beta_T\, \gamma_T} \right)
$$

This is sublinear whenever $\gamma_T$ is, which covers all the kernels in the table.

In the **frequentist setting**, $f$ is a fixed function with RKHS norm at most $B$. Confidence bounds must now hold uniformly over that ball, so $\beta_t$ has to grow with $\gamma_t$ itself. [Chowdhury & Gopalan (2017)](https://arxiv.org/abs/1704.00445) gave the sharpest such schedule, which leads to $R_T = O(\gamma_T \sqrt{T})$. For Matérn kernels in high dimensions this bound can fail to be sublinear.

Is that GP-UCB's fault or the analysis's? Lower bounds for the RKHS setting ([Scarlett, Bogunovic & Cevher 2017](https://arxiv.org/abs/1706.00090)) scale like $T^{(\nu+d)/(2\nu+d)}$, and more elaborate algorithms come close to them. Whether plain GP-UCB is suboptimal was posed as a COLT 2021 open problem ([Vakili, Scarlett & Javidi 2021](https://proceedings.mlr.press/v134/vakili21a.html)).

### Expected improvement

EI is harder to analyze because it has no explicit exploration parameter. [Bull (2011)](https://arxiv.org/abs/1101.3501) showed that for $f$ in a Matérn RKHS and fixed kernel hyperparameters, EI converges at near-optimal rates. He also showed a failure mode: with hyperparameters re-estimated by maximum likelihood, EI can fail to converge on some functions. He fixed it with occasional random exploration.

### What the theory is good for

The theory explains three things practitioners observe. Smoother kernels learn faster. Dimension hurts exponentially unless the function has structure. And exploration must not be switched off, which is why purely greedy policies fail.

It does not tell you what $\beta$ to use: the theoretical schedules explore far more than works in practice. Nearly all analyses also assume known hyperparameters, while real BO learns them from the same few points it is optimizing over. That gap between theory and practice is still open.

## Making BO work in practice

Modern BO practice rests on three ideas: treat GP hyperparameters with care, write acquisition functions as Monte Carlo expectations you can differentiate, and select batches of points jointly. BoTorch packages all three.

### Hyperparameters are the silent failure mode

With 10 data points and 10 ARD lengthscales, maximum likelihood is fitting almost as many parameters as it has observations. The point estimate can be badly wrong, and a wrong lengthscale makes the GP overconfident exactly where exploration is needed.

[Snoek et al. (2012)](https://arxiv.org/abs/1206.2944) proposed integrating the acquisition function over the hyperparameter posterior instead:

$$
\hat{\alpha}(x) = \int \alpha(x; \theta)\, p(\theta \mid D)\, d\theta \;\approx\; \frac{1}{S} \sum_{s=1}^{S} \alpha(x; \theta^{(s)}),
\qquad \theta^{(s)} \sim p(\theta \mid D)
$$

They drew the samples with slice sampling. Fully Bayesian treatment is expensive, but it is the most reliable fix when data is scarce, and SAASBO (in the high-dimensional section) builds on it.

The cheaper fix is good priors. [Hvarfner, Hellsten & Nardi (2024)](https://arxiv.org/abs/2402.02229) showed that a log-normal lengthscale prior whose location grows with $\log d$ avoids most of the pathologies of high-dimensional fitting. BoTorch has since adopted a dimension-scaled prior of this kind as its default.

Two more routine steps matter more than their simplicity suggests. **Standardize outputs** to zero mean and unit variance, so default priors on the output scale are sensible. **Normalize inputs** to the unit cube, so lengthscale priors mean the same thing in every problem. For objectives with skewed sensitivity, such as a learning rate that matters on a log scale, input warping ([Snoek et al. 2014](https://arxiv.org/abs/1402.0929)) learns a monotone transformation of each input.

### Monte Carlo acquisition functions

Many acquisition functions are expectations of a utility $u$ over the joint posterior at a set of $q$ points $X = (x_1, \dots, x_q)$. The batch version of EI is an example:

$$
\alpha_{q\text{EI}}(X) = \mathbb{E}\left[ \max\Big( \max_{j=1..q} f(x_j) - f^+,\; 0 \Big) \right]
$$

That expectation has no closed form for $q > 1$, but it is easy to estimate by sampling. The reparameterization trick makes the estimate differentiable. Write each posterior sample as a deterministic function of standard normal noise, using the Cholesky factor $L(X)$ of the posterior covariance:

$$
f(X) = \mu(X) + L(X)\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
\qquad\Longrightarrow\qquad
\alpha(X) \approx \frac{1}{N} \sum_{i=1}^{N} u\big( \mu(X) + L(X)\, \varepsilon_i \big)
$$

Gradients with respect to $X$ then flow through $\mu$ and $L$ by automatic differentiation. [Wilson, Hutter & Deisenroth (2018)](https://arxiv.org/abs/1805.10196) worked this out for a family of acquisition functions. They also showed that many of them are submodular, so building a batch greedily, one point at a time, comes with approximation guarantees.

BoTorch ([Balandat et al. 2020](https://arxiv.org/abs/1910.06403)) added one more trick: fix the base samples $\varepsilon_i$, often using quasi-Monte Carlo, for the entire inner optimization. The estimate becomes a deterministic, smooth function of $X$, so deterministic L-BFGS-B works. This is sample average approximation, and the paper proves the maximizers converge as $N$ grows. The same machinery supports noisy EI, KG, MES and multi-objective acquisitions with a single code path.

### Batch BO

When you can run $q$ evaluations in parallel, you need $q$ points that are good individually and diverse jointly. The main approaches are:

- **Fantasies.** Choose one point, pretend you observed a value there, condition on it, and choose the next. The kriging believer uses the posterior mean as the fake value; the constant liar uses a fixed value ([Ginsbourger et al. 2010](https://doi.org/10.1007/978-3-642-10701-6_6)). This works because, as noted in the GP section, the posterior variance does not depend on the observed value, so the fantasy correctly shrinks uncertainty around chosen points.
- **Local penalization.** Multiply the acquisition by penalties around already chosen points ([González et al. 2016](https://arxiv.org/abs/1505.08052)).
- **Joint MC acquisitions.** Optimize qEI, qNEI or qKG over all $q$ points at once, or greedily one at a time, using the machinery above.
- **Thompson sampling.** Draw $q$ posterior samples and take each sample's maximizer. It is embarrassingly parallel and scales to batches of hundreds ([Hernández-Lobato et al. 2017](https://arxiv.org/abs/1706.01825)).

### Software

The current ecosystem is mature enough that implementing BO from scratch is rarely the right choice outside of learning:

| Library | Built on | Best for |
| --- | --- | --- |
| [BoTorch](https://botorch.org/) and [Ax](https://ax.dev/) | PyTorch, GPyTorch | Research and production; the widest set of modern acquisitions |
| [GPyTorch](https://gpytorch.ai/) | PyTorch | Scalable GP modeling on GPUs |
| [Trieste](https://github.com/secondmind-labs/trieste) | TensorFlow, GPflow | BO in the GPflow ecosystem |
| [SMAC3](https://github.com/automl/SMAC3) | Random forests and GPs | Hyperparameter optimization with mixed and conditional spaces |
| [Optuna](https://optuna.org/) | TPE, CMA-ES and a GP sampler | Everyday hyperparameter tuning |

## Scaling Gaussian processes

Exact GP inference costs $O(n^3)$ time and $O(n^2)$ memory, which is fine for classic BO budgets but not for high-throughput experiments, trust-region methods with tens of thousands of points, or GP regression on large datasets. Three families of fixes dominate: inducing points, structured kernels with iterative solvers, and random features.

### Inducing points and sparse variational GPs

The idea is to summarize the function through its values $u = f(Z)$ at $m \ll n$ **inducing points** $Z$. Everything else is predicted from $u$ through the conditional $f \mid u$, which gives the low-rank approximation $Q_{nn} = K_{nm} K_{mm}^{-1} K_{mn}$ to the full kernel matrix.

Early methods like FITC ([Snelson & Ghahramani 2006](https://papers.nips.cc/paper/2857-sparse-gaussian-processes-using-pseudo-inputs)) modified the model itself, which could distort it. [Titsias (2009)](https://proceedings.mlr.press/v5/titsias09a.html) instead kept the exact model and chose $Z$ by maximizing a variational lower bound on the marginal likelihood:

$$
\log p(\mathbf{y}) \;\geq\; \log \mathcal{N}\big( \mathbf{y} \mid \mathbf{0},\; Q_{nn} + \sigma_n^2 I \big) \;-\; \frac{1}{2\sigma_n^2} \operatorname{tr}\big( K_{nn} - Q_{nn} \big)
$$

The trace term is the key: it penalizes inducing points that fail to explain the full covariance, so adding more of them can only tighten the bound. The cost drops to $O(nm^2)$.

[Hensman, Fusi & Lawrence (2013)](https://arxiv.org/abs/1309.6835) kept the variational distribution over $u$ explicit, which makes the bound a sum over data points. That allows minibatch stochastic optimization at $O(m^3)$ per step, independent of $n$. This stochastic variational GP (SVGP) also handles non-Gaussian likelihoods, such as classification. It is the standard way to fit GPs to millions of points.

The weakness for BO is that sparse GPs smooth over detail. BO needs accurate uncertainty exactly near the few best points, which is where a summary through $m$ inducing points tends to be too coarse.

### Structured kernels and iterative solvers

A different route keeps all $n$ points but never forms or factorizes $K$. Conjugate gradients solves $(K + \sigma_n^2 I)^{-1} \mathbf{y}$ using only matrix–vector multiplications (MVMs). Stochastic Lanczos quadrature estimates the log determinant the same way. If an MVM is cheap, so is the whole GP.

**KISS-GP** ([Wilson & Nickisch 2015](https://arxiv.org/abs/1503.01057)) makes MVMs cheap by interpolating the kernel from a regular grid of inducing points $U$: $K \approx W K_{UU} W^\top$, with sparse interpolation weights $W$. On a grid, $K_{UU}$ has Toeplitz or Kronecker structure, so an MVM costs roughly $O(n + m \log m)$. It works best in low dimensions, where grids are affordable.

**GPyTorch** ([Gardner et al. 2018](https://arxiv.org/abs/1809.11165)) turned the MVM approach into a general engine. Its blackbox matrix–matrix inference runs batched conjugate gradients on GPUs with a pivoted-Cholesky preconditioner. [Wang et al. (2019)](https://arxiv.org/abs/1903.08114) used it to fit exact GPs to over a million points, and found they often beat sparse approximations.

### Random Fourier features

Bochner's theorem suggests another approximation. Sample $D$ frequencies $\omega$ from the kernel's spectral density, and the kernel becomes an inner product of explicit features ([Rahimi & Recht 2007](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines)):

$$
k(x, x') \approx \phi(x)^\top \phi(x'),
\qquad
\phi(x) = \sqrt{2/D}\, \cos(\Omega x + \mathbf{b}),
\quad \omega_i \sim p(\omega),\; b_i \sim U[0, 2\pi]
$$

The GP becomes Bayesian linear regression on $D$ features, which costs $O(nD^2)$. More importantly for BO, a posterior sample is now an explicit function you can evaluate anywhere and maximize with gradients. That is what Thompson sampling needs.

The catch is **variance starvation**. With many data points and a fixed number of features, the feature-based posterior becomes overconfident far from the data, so TS stops exploring. Mutný & Krause (2018) used deterministic quadrature Fourier features to reduce the approximation error for additive kernels.

### Pathwise conditioning

[Wilson et al. (2020)](https://arxiv.org/abs/2002.09309) fixed variance starvation with an old identity from geostatistics, Matheron's rule. A posterior sample equals a prior sample plus a data-dependent correction:

$$
(f \mid \mathbf{y})(\cdot) \;\overset{d}{=}\; f(\cdot) + k(\cdot, X)\,(K + \sigma_n^2 I)^{-1} \big( \mathbf{y} - f(X) - \boldsymbol{\varepsilon} \big),
\qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma_n^2 I)
$$

The trick is to decouple the two parts. The prior sample $f$ uses random Fourier features, which are accurate for the prior regardless of $n$. The correction uses the exact kernel. The result is a cheap function sample with accurate uncertainty both near and far from data. Pathwise sampling is now the standard way to do Thompson sampling in BoTorch.

For a broad survey of these methods, see [Liu et al. (2020)](https://arxiv.org/abs/1807.01065), "When Gaussian Process Meets Big Data".

## High-dimensional BO

Folk wisdom long held that GP-based BO stops working beyond about 20 dimensions. The field responded with three kinds of structural assumptions: low effective dimensionality, additivity, and locality. Then, in 2024, it turned out that much of the problem had been bad lengthscale priors.

### Why high dimensions are hard

Several problems compound. Distances between random points concentrate, so a stationary kernel sees every pair of points as roughly equally far apart. Covering the space needs exponentially many points, and $\gamma_T$ grows accordingly. ARD adds $d$ hyperparameters to fit from very few observations. And maximizing the acquisition function becomes a hard high-dimensional problem of its own, with the useful regions occupying a vanishing fraction of the volume.

### Low effective dimensionality: embeddings

Many real objectives vary along only a few directions. If $f(x) = g(P^\top x)$ for some unknown $d_e$-dimensional subspace, you can optimize in a random low-dimensional embedding instead.

**REMBO** ([Wang et al. 2013](https://arxiv.org/abs/1301.1942)) draws a random matrix $A$ of size $D \times d$ and optimizes $g(y) = f(Ay)$ over a small box in $y$. If $d \ge d_e$, the embedding contains an optimum with high probability. In practice, points $Ay$ often fall outside the original box and must be clipped back in, which distorts the geometry.

Follow-ups fixed the embedding's problems. HeSBO ([Nayebi, Munteanu & Poloczek 2019](https://proceedings.mlr.press/v97/nayebi19a.html)) used a count-sketch embedding that never leaves the box. ALEBO ([Letham et al. 2020](https://arxiv.org/abs/2001.11659)) added linear constraints and a kernel suited to the embedding. BAxUS (Papenmeier, Nardi & Poloczek 2022) starts with a small nested subspace and grows it during the run, removing the need to guess $d$.

### Additivity

If $f$ is a sum of functions over small groups of variables, $f(x) = \sum_j f_j(x^{(j)})$, then an additive kernel matches the structure. [Kandasamy, Schneider & Póczos (2015)](https://arxiv.org/abs/1503.01673) showed that $\gamma_T$, and hence regret, then scales linearly in $d$ rather than exponentially. The acquisition can also be optimized group by group.

The decomposition is rarely known. Later work learned it from data, for example by sampling decompositions with MCMC ([Gardner et al. 2017](https://proceedings.mlr.press/v54/gardner17a.html)) or allowing overlapping groups ([Rolland et al. 2018](https://proceedings.mlr.press/v84/rolland18a.html)).

### Locality: trust regions

**TuRBO** ([Eriksson et al. 2019](https://arxiv.org/abs/1910.01739)) gives up on a global model. It fits a GP inside a hyperrectangle around the current best point, with side lengths scaled by the ARD lengthscales, and picks batches with Thompson sampling inside it. After a run of successes the region expands, after a run of failures it shrinks, and when it becomes too small the method restarts elsewhere.

The local model avoids fitting one GP to a highly heterogeneous function, and it keeps the acquisition optimization local. TuRBO was one of the first GP methods to compete with evolutionary strategies like CMA-ES on problems with tens to hundreds of dimensions and thousands of evaluations. SCBO ([Eriksson & Poloczek 2021](https://arxiv.org/abs/2002.08526)) extended it to many black-box constraints.

### Sparsity priors: SAASBO

**SAASBO** ([Eriksson & Jankowiak 2021](https://arxiv.org/abs/2103.00349)) puts a hierarchical sparsity prior on the inverse squared lengthscales $\rho_d$:

$$
\tau \sim \mathcal{HC}(\alpha), \qquad \rho_d \sim \mathcal{HC}(\tau) \quad \text{for } d = 1, \dots, D
$$

The half-Cauchy (HC) priors concentrate near zero, meaning most dimensions are "switched off" by default, but their heavy tails let a few relevant dimensions escape. Inference uses the No-U-Turn sampler rather than maximum likelihood. SAASBO performs very well on problems with hundreds of dimensions and a budget of a few hundred evaluations. The MCMC makes each iteration slow, so it suits truly expensive objectives.

### The 2024 surprise: vanilla BO works

[Hvarfner, Hellsten & Nardi (2024)](https://arxiv.org/abs/2402.02229) asked why a standard GP with EI fails in high dimensions. Their answer: default lengthscale priors assume functions that are far too complex relative to $d$. With lengthscales of order 1 in a unit cube with hundreds of dimensions, almost every pair of points looks independent, and the GP cannot learn anything.

The fix is a lengthscale prior whose location scales with the dimension, roughly as $\sqrt{d}$. With that one change, plain GP-based BO matched or beat the specialized methods above on standard benchmarks with hundreds of dimensions. Around the same time, Xu & Zhe (2024), in "Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization", reported a similar finding. Follow-up work has examined why lengthscale fitting fails, pointing to vanishing gradients at poor initializations.

The lesson is a humbling one for the field. The assumption that "GPs fail in high dimensions" partly reflected a modeling default rather than a fundamental limit. Structural methods remain valuable when the structure is really there, but a well-specified vanilla GP is now the baseline to beat.

## Beyond vanilla BO

Real problems rarely look like "maximize one noiseless function over a box". Most extensions follow the same recipe: model each unknown quantity with its own GP, then redefine the acquisition function to value what you actually care about.

### Constraints

Often a design must satisfy black-box constraints $c_k(x) \leq 0$ that are as expensive to check as $f$. [Gardner et al. (2014)](https://proceedings.mlr.press/v32/gardner14.html) and [Gelbart, Snoek & Adams (2014)](https://arxiv.org/abs/1403.5607) model each constraint with its own GP and weight the acquisition by the probability of feasibility:

$$
\alpha_{\text{cEI}}(x) = \alpha_{\text{EI}}(x) \prod_{k} P\big( c_k(x) \leq 0 \big)
$$

When no feasible point has been found yet, EI is undefined, and the usual fallback is to maximize the probability of feasibility alone. Noisy constraints are handled by the same integration trick as noisy EI ([Letham et al. 2019](https://arxiv.org/abs/1706.07094)).

### Multi-fidelity and cost-aware BO

Many objectives come with cheaper approximations: train for fewer epochs, on a data subset, or run a coarser simulation. Multi-fidelity BO adds a fidelity parameter $s$ to the GP input, models $f(x, s)$ jointly, and chooses both where and how accurately to evaluate. The acquisition must weigh information against cost.

Key methods include MF-GP-UCB ([Kandasamy et al. 2016](https://arxiv.org/abs/1603.06288)), which evaluates at low fidelity until the cheap model is no longer informative. The trace-aware knowledge gradient ([Wu et al. 2019](https://arxiv.org/abs/1903.04703)) exploits the fact that training a network produces a whole learning curve for the cost of its final point. Multi-fidelity MES ([Takeno et al. 2020](https://arxiv.org/abs/1901.08275)) extends max-value entropy search.

A parallel line from AutoML uses bandit-style early stopping instead of a joint model. Hyperband ([Li et al. 2017](https://arxiv.org/abs/1603.06560)) allocates budgets by successive halving, and BOHB ([Falkner, Klein & Hutter 2018](https://arxiv.org/abs/1807.01774)) combines it with a model-based sampler. These are often the practical choice for neural network hyperparameter tuning.

### Multiple objectives

With several objectives there is no single optimum, only a Pareto front of trade-offs. The standard quality measure is the **hypervolume**: the volume of objective space dominated by the current front, relative to a reference point.

Expected hypervolume improvement (EHVI) is the multi-objective analogue of EI. It was historically expensive to compute. [Daulton, Balandat & Bakshy (2020)](https://arxiv.org/abs/2006.05078) made it parallel and differentiable (qEHVI) with the MC machinery from BoTorch, and their follow-up qNEHVI ([Daulton et al. 2021](https://arxiv.org/abs/2105.08195)) handled noise. The cheaper alternative is random scalarization: ParEGO ([Knowles 2006](https://doi.org/10.1109/TEVC.2005.851274)) optimizes a randomly weighted combination of the objectives at each step.

### Other input spaces

The GP only needs a kernel, so BO extends to anything you can define similarity over. Categorical and mixed spaces use kernels over one-hot or learned encodings. Combinatorial problems use graph kernels or discrete surrogates. For molecules and other structured objects, a common approach is to run BO in the latent space of a generative model ([Gómez-Bombarelli et al. 2018](https://arxiv.org/abs/1610.02415)).

### Non-GP surrogates

GPs are not the only choice, and they are sometimes not the best one:

| Surrogate | Example | Strengths |
| --- | --- | --- |
| Random forests | SMAC ([Hutter, Hoos & Leyton-Brown 2011](https://doi.org/10.1007/978-3-642-25566-3_40)) | Categorical and conditional spaces, scales to many points |
| Density ratio estimators | TPE ([Bergstra et al. 2011](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization)) | Very cheap; the default in Optuna and Hyperopt |
| Bayesian linear regression on neural features | DNGO ([Snoek et al. 2015](https://arxiv.org/abs/1502.05700)) | Linear cost in $n$ |
| Bayesian neural networks | BOHAMIANN ([Springenberg et al. 2016](https://papers.nips.cc/paper/6117-bayesian-optimization-with-robust-bayesian-neural-networks)) | Flexible, scalable uncertainty |
| Deep kernels | [Wilson et al. 2016](https://arxiv.org/abs/1511.02222) | A GP with a learned neural feature map |
| Prior-data fitted networks | PFNs4BO ([Müller et al. 2023](https://arxiv.org/abs/2305.17535)) | A transformer trained on prior samples predicts the posterior in-context, in one forward pass |

The PFN line is worth watching. The network is trained once on functions sampled from a prior, including GP priors with sampled hyperparameters, and then approximates Bayesian inference for new data without fitting. That removes the hyperparameter-fitting step entirely. LLM-based surrogates and proposal mechanisms, such as LLAMBO ([Liu et al. 2024](https://arxiv.org/abs/2402.03921)), push further in the same direction, bringing in prior knowledge from text descriptions of the problem.

## Open problems

BO has matured into reliable tooling, but several basic questions remain open, and the 2023–2024 results on LogEI and lengthscale priors are a reminder that some "known" limitations were really implementation details.

- **Model misspecification and hyperparameters.** Almost every theoretical guarantee assumes the kernel and its hyperparameters are known. In practice they are learned from the same handful of points being optimized. How to make BO robust to a wrong prior, with guarantees, is largely unsolved.
- **Benchmarking.** Results are sensitive to initial designs, inner-optimization budgets and hyperparameter priors. As the LogEI and vanilla-BO papers showed, a baseline that is set up badly can make a new method look better than it is. Careful, shared baselines matter more than new acquisition functions.
- **Non-myopic BO.** Nearly all acquisition functions look one step ahead. Multi-step lookahead is formally the right objective for a fixed budget, but it is expensive, and the gains shown so far are modest.
- **When to stop.** Deciding that further evaluations are not worth their cost is essential in real experiments and still mostly done by hand.
- **Priors from experts and from language models.** Practitioners know things about their problems that a stationary kernel cannot express. Methods like πBO (Hvarfner et al. 2022) inject user beliefs about where the optimum lies. Pretrained surrogates such as PFNs, and LLM-based priors, may make this routine, but when they help and when they mislead is still unclear.
- **Scale in both directions.** Self-driving labs and simulation pipelines produce more data than exact GPs like, while many scientific problems still allow only a few dozen evaluations. One framework that is reliable at both ends does not yet exist.

If you want one next step after this post, read [Garnett's *Bayesian Optimization*](https://bayesoptbook.com/) (2023, free online). It derives everything here from decision theory, with a rigor and consistency that no survey can match. For a shorter formal treatment, [Frazier's tutorial](https://arxiv.org/abs/1807.02811) remains the best compact reference.

## References

**Books, surveys and tutorials**

1. Rasmussen & Williams. [Gaussian Processes for Machine Learning](http://gaussianprocess.org/gpml/). MIT Press, 2006.
2. Garnett. [Bayesian Optimization](https://bayesoptbook.com/). Cambridge University Press, 2023.
3. Frazier. [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811). arXiv, 2018.
4. Shahriari et al. [Taking the Human Out of the Loop: A Review of Bayesian Optimization](https://doi.org/10.1109/JPROC.2015.2494218). Proceedings of the IEEE, 2016.
5. Kanagawa et al. [Gaussian Processes and Kernel Methods: A Review on Connections and Equivalences](https://arxiv.org/abs/1807.02582). arXiv, 2018.
6. Liu et al. [When Gaussian Process Meets Big Data: A Review of Scalable GPs](https://arxiv.org/abs/1807.01065). IEEE TNNLS, 2020.
7. Görtler, Kehlbeck & Deussen. [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/). Distill, 2019.
8. Agnihotri & Batra. [Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/). Distill, 2020.
9. Duvenaud. [The Kernel Cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/).

**Foundations and acquisition functions**

10. Kushner. [A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise](https://doi.org/10.1115/1.3653121). Journal of Basic Engineering, 1964.
11. Močkus. [On Bayesian Methods for Seeking the Extremum](https://doi.org/10.1007/3-540-07165-2_55). Optimization Techniques, 1975.
12. Jones, Schonlau & Welch. [Efficient Global Optimization of Expensive Black-Box Functions](https://doi.org/10.1023/A:1008306431147). Journal of Global Optimization, 1998.
13. Srinivas et al. [Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design](https://arxiv.org/abs/0912.3995). ICML 2010.
14. Snoek, Larochelle & Adams. [Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/abs/1206.2944). NeurIPS 2012.
15. Frazier, Powell & Dayanik. [The Knowledge-Gradient Policy for Correlated Normal Beliefs](https://doi.org/10.1287/ijoc.1080.0314). INFORMS Journal on Computing, 2009.
16. Hennig & Schuler. [Entropy Search for Information-Efficient Global Optimization](https://arxiv.org/abs/1112.1217). JMLR, 2012.
17. Hernández-Lobato, Hoffman & Ghahramani. [Predictive Entropy Search for Efficient Global Optimization of Black-box Functions](https://arxiv.org/abs/1406.2541). NeurIPS 2014.
18. Wang & Jegelka. [Max-value Entropy Search for Efficient Bayesian Optimization](https://arxiv.org/abs/1703.01968). ICML 2017.
19. Hvarfner, Hutter & Nardi. [Joint Entropy Search for Maximally-Informed Bayesian Optimization](https://arxiv.org/abs/2206.04771). NeurIPS 2022.
20. Russo & Van Roy. [Learning to Optimize via Posterior Sampling](https://arxiv.org/abs/1301.2609). Mathematics of Operations Research, 2014.
21. Letham et al. [Constrained Bayesian Optimization with Noisy Experiments](https://arxiv.org/abs/1706.07094). Bayesian Analysis, 2019.
22. Ament et al. [Unexpected Improvements to Expected Improvement for Bayesian Optimization](https://arxiv.org/abs/2310.20708). NeurIPS 2023.
23. Hoffman, Brochu & de Freitas. [Portfolio Allocation for Bayesian Optimization](https://arxiv.org/abs/1009.5419). UAI 2011.

**Theory**

24. Chowdhury & Gopalan. [On Kernelized Multi-armed Bandits](https://arxiv.org/abs/1704.00445). ICML 2017.
25. Scarlett, Bogunovic & Cevher. [Lower Bounds on Regret for Noisy Gaussian Process Bandit Optimization](https://arxiv.org/abs/1706.00090). COLT 2017.
26. Vakili, Khezeli & Picheny. [On Information Gain and Regret Bounds in Gaussian Process Bandits](https://arxiv.org/abs/2009.06966). AISTATS 2021.
27. Vakili, Scarlett & Javidi. [Open Problem: Tight Online Confidence Intervals for RKHS Elements](https://proceedings.mlr.press/v134/vakili21a.html). COLT 2021.
28. Bull. [Convergence Rates of Efficient Global Optimization Algorithms](https://arxiv.org/abs/1101.3501). JMLR, 2011.

**Practice, batch BO and software**

29. Wilson, Hutter & Deisenroth. [Maximizing Acquisition Functions for Bayesian Optimization](https://arxiv.org/abs/1805.10196). NeurIPS 2018.
30. Balandat et al. [BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization](https://arxiv.org/abs/1910.06403). NeurIPS 2020.
31. Snoek et al. [Input Warping for Bayesian Optimization of Non-Stationary Functions](https://arxiv.org/abs/1402.0929). ICML 2014.
32. Ginsbourger, Le Riche & Carraro. [Kriging Is Well-Suited to Parallelize Optimization](https://doi.org/10.1007/978-3-642-10701-6_6). 2010.
33. González et al. [Batch Bayesian Optimization via Local Penalization](https://arxiv.org/abs/1505.08052). AISTATS 2016.
34. Hernández-Lobato et al. [Parallel and Distributed Thompson Sampling for Large-scale Accelerated Exploration of Chemical Space](https://arxiv.org/abs/1706.01825). ICML 2017.

**Kernels and scalable GPs**

35. Duvenaud et al. [Structure Discovery in Nonparametric Regression through Compositional Kernel Search](https://arxiv.org/abs/1302.4922). ICML 2013.
36. Wilson & Adams. [Gaussian Process Kernels for Pattern Discovery and Extrapolation](https://arxiv.org/abs/1302.4245). ICML 2013.
37. Lee et al. [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165). ICLR 2018.
38. Titsias. [Variational Learning of Inducing Variables in Sparse Gaussian Processes](https://proceedings.mlr.press/v5/titsias09a.html). AISTATS 2009.
39. Hensman, Fusi & Lawrence. [Gaussian Processes for Big Data](https://arxiv.org/abs/1309.6835). UAI 2013.
40. Wilson & Nickisch. [Kernel Interpolation for Scalable Structured Gaussian Processes (KISS-GP)](https://arxiv.org/abs/1503.01057). ICML 2015.
41. Gardner et al. [GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration](https://arxiv.org/abs/1809.11165). NeurIPS 2018.
42. Wang et al. [Exact Gaussian Processes on a Million Data Points](https://arxiv.org/abs/1903.08114). NeurIPS 2019.
43. Wilson et al. [Efficiently Sampling Functions from Gaussian Process Posteriors](https://arxiv.org/abs/2002.09309). ICML 2020.

**High-dimensional BO**

44. Wang et al. [Bayesian Optimization in a Billion Dimensions via Random Embeddings (REMBO)](https://arxiv.org/abs/1301.1942). JAIR, 2016.
45. Kandasamy, Schneider & Póczos. [High Dimensional Bayesian Optimisation and Bandits via Additive Models](https://arxiv.org/abs/1503.01673). ICML 2015.
46. Letham et al. [Re-Examining Linear Embeddings for High-Dimensional Bayesian Optimization (ALEBO)](https://arxiv.org/abs/2001.11659). NeurIPS 2020.
47. Eriksson et al. [Scalable Global Optimization via Local Bayesian Optimization (TuRBO)](https://arxiv.org/abs/1910.01739). NeurIPS 2019.
48. Eriksson & Jankowiak. [High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces (SAASBO)](https://arxiv.org/abs/2103.00349). UAI 2021.
49. Hvarfner, Hellsten & Nardi. [Vanilla Bayesian Optimization Performs Great in High Dimensions](https://arxiv.org/abs/2402.02229). ICML 2024.

**Extensions and other surrogates**

50. Gelbart, Snoek & Adams. [Bayesian Optimization with Unknown Constraints](https://arxiv.org/abs/1403.5607). UAI 2014.
51. Kandasamy et al. [Gaussian Process Bandit Optimisation with Multi-fidelity Evaluations](https://arxiv.org/abs/1603.06288). NeurIPS 2016.
52. Li et al. [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560). JMLR, 2018.
53. Falkner, Klein & Hutter. [BOHB: Robust and Efficient Hyperparameter Optimization at Scale](https://arxiv.org/abs/1807.01774). ICML 2018.
54. Daulton, Balandat & Bakshy. [Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization](https://arxiv.org/abs/2006.05078). NeurIPS 2020.
55. Hutter, Hoos & Leyton-Brown. [Sequential Model-Based Optimization for General Algorithm Configuration (SMAC)](https://doi.org/10.1007/978-3-642-25566-3_40). LION 2011.
56. Snoek et al. [Scalable Bayesian Optimization Using Deep Neural Networks (DNGO)](https://arxiv.org/abs/1502.05700). ICML 2015.
57. Müller et al. [PFNs4BO: In-Context Learning for Bayesian Optimization](https://arxiv.org/abs/2305.17535). ICML 2023.
58. Chen et al. [Bayesian Optimization in AlphaGo](https://arxiv.org/abs/1812.06855). arXiv, 2018.
