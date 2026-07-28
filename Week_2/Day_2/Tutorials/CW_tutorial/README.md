# Continuous Wave (CW) Tutorial

Two-part tutorial on continuous gravitational waves in pulsar timing arrays.

*Adapted from Polina Petrov and Caitlin Witt's work (IPTA GW school 2024 CW tutorial). Simulation code adapted from Matt Miles.*

## Notebooks

| notebook | contents |
|---|---|
| `CW_tutorial_1_simulation_and_signal_models.ipynb` | Simulate your own PTA dataset (CW + optional red noise / GWB), antenna patterns, Earth term vs pulsar term, the `enterprise` and `discovery` CW signal models, likelihood scans |
| `CW_tutorial_2_searches.ipynb` | Search the simulated data: Fp/Fe frequentist statistics, full Bayesian search with `QuickCW` (with an explicit pulsar-term option), and an optional **GPU-only** `Prometheus` (NUTS) section |

`*_solutions.ipynb` are the completed versions with all exercise cells filled in and executed.

Run Tutorial 1 first — it writes the simulated dataset to `sim_data/`, which Tutorial 2 searches. A pre-generated `sim_data/` ships with the repo, so Tutorial 2 also works standalone.

## Data

- `data_products/` — 12 synthetic pulsars (feather files, mimicking the IPTA DR2 dataset): realistic sky positions, 13–22 yr baselines, ~0.1–0.5 µs TOA errors, measured distances.
- `sim_data/` — the simulated residuals (`cw_sim_residuals.npz`) and injection parameters (`cw_injection_params.json`) produced by Tutorial 1.

## Helper files

- `cw_utils.py` — modified Fe-statistic class (from the 2024 tutorial).
- `QuickCW_v2.py` — thin wrapper around `QuickCW` with a simplified white-noise model appropriate for the simulated data.

## Installation

**The school environment already covers these notebooks — no extra packages are needed.** From the repository root:

```bash
uv sync
```

Then run the notebooks with that environment, e.g.

```bash
uv run jupyter lab
```

or select `.venv/bin/python` as the kernel in your editor. Both notebooks are tested against this environment (verified with numpy 2.4, scipy 1.18, jax 0.11, numba 0.66, Python 3.13).

Everything the notebooks import — `enterprise`, `enterprise_extensions`, `discovery`, `QuickCW`, plus `healpy`, `corner`, `h5py`, `astropy`, `pyarrow`, `scikit-sparse` — is already resolved by the root `pyproject.toml`, most of it transitively through `enterprise-extensions`.

Notes:
- Tutorial 1 installs a small **SciPy ≥ 1.18 compatibility shim** before importing `discovery` (SciPy now returns the `cho_factor` `lower` flag as a 0-d array, which JAX rejects as a non-hashable static argument). It is a no-op on older SciPy and can be deleted once `discovery` handles this upstream.
- `scikit-sparse` is only used to speed up simulating an HD-correlated GWB; the notebooks fall back to a dense Cholesky if it is unavailable.
- These notebooks do **not** use `PTMCMCSampler`, which is convenient because importing it in the synced environment currently fails with `RuntimeError: cannot load MPI library` (its `mpi4py` dependency finds no system MPI). Other tutorials that do import it will need `libopenmpi` installed, or `mpi4py` removed.
- See https://github.com/nanograv/QuickCW/blob/main/docs/how_to_run_QuickCW.md for QuickCW details.

The optional Prometheus section requires an NVIDIA GPU with CUDA-enabled JAX:

```bash
pip install git+https://github.com/XGI-MSU/prometheus.git
```

## Maintenance

The notebooks are generated from `builders/build_nb1.py` and `builders/build_nb2.py` (via `nbformat`) — each writes both the student and `_solutions` variant. To change a notebook, edit the builder, rerun it, then execute the solutions notebook to refresh outputs and `sim_data/`.
