# PolyBO: Structure-Aware Machine Learning for Materials Discovery

## Installation and Usage

### 1. Install uv and necessary tools

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For alternatives, see [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Create and activate the virtual environment

```bash
uv sync
source .venv/bin/activate
```

### 3. Run the notebooks

```bash
uv run marimo edit notebooks/gp.py
```

See the [marimo docs](https://docs.marimo.io/) for more.

## Notebooks

| Notebook | What it covers |
| --- | --- |
| [`notebooks/gp.py`](notebooks/gp.py) | **Gaussian processes.** Conditioning, the posterior, learning hyperparameters (marginal likelihood, L-BFGS restarts, integrating out θ), the weight-space view and kernel trick, kernels (Matérn, spectral density, ARD, RKHS), how GPs fail (extrapolation, nonstationarity, heteroscedastic noise, outliers, ill-conditioning, high dimensions) and scaling (sparse GPs, random features, pathwise sampling). |
| [`notebooks/gp_and_bo.py`](notebooks/gp_and_bo.py) | **Gaussian processes and Bayesian optimization.** The GP background plus acquisition functions (PI, EI, UCB, Thompson sampling), LogEI, the BO loop, regret, batch and Monte Carlo acquisitions, high-dimensional and constrained / multi-objective BO. |

Both notebooks are written in plain NumPy/SciPy so every formula maps to a few readable lines. They accompany the write-up in [`docs/gp-and-bo.md`](docs/gp-and-bo.md).
