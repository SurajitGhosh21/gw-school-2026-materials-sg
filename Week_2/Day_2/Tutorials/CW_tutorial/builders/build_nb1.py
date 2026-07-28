"""Generate CW tutorial notebook 1 (student + solution versions)."""
import nbformat as nbf
import sys

OUTDIR = "/home/mattm/projects/gw-school-2026-materials/Week_2/Day_2/Tutorials/CW_tutorial"

# Each cell: (cell_type, source) for solution. Student version swaps cells whose
# id appears in STUDENT_OVERRIDES with the alternate source (or drops if None).
cells = []
overrides = {}

def md(src, sid=None):
    cells.append(("markdown", src, sid))

def code(src, sid=None):
    cells.append(("code", src, sid))

def student(sid, src):
    overrides[sid] = src

# ----------------------------------------------------------------------------
md(r"""# Continuous Gravitational Waves in Pulsar Timing Arrays
## Tutorial 1: Simulating the data & understanding the signal model

*Adapted from Polina Petrov and Caitlin Witt's work (IPTA GW school 2024 CW tutorial). Simulation code adapted from Matt Miles.*

In addition to the stochastic gravitational wave background (GWB), we expect a number of closer, louder sources to stand out above the rest. These are individual high-mass, inspiraling supermassive black hole binaries (SMBHBs) with orbital frequencies roughly between 1 and 100 nHz, which evolve very slowly over the course of PTA observations. Because of their slow evolution, PTAs don't witness these binaries changing much in orbital frequency, i.e., we don't see the distinct "chirp" that LIGO sees. We therefore call these signals "continuous gravitational waves", or CWs.

CW searches are a bit different from GWB searches. A CW is a *deterministic* signal: for a single binary emitting GWs we can model the entire waveform! In theory this makes the search simpler, but in practice there are a lot of parameters to model.

In this tutorial you will:

1. **Simulate** your own PTA dataset containing a CW, and optionally intrinsic pulsar red noise and a GWB, and see how the data change depending on what you put in them.
2. Understand **how the CW signal models in `enterprise` and `discovery` work**: the antenna pattern, the Earth term and the pulsar term, and how the model parameters shape the waveform.
3. Use the `discovery` (JAX) likelihood to see how the likelihood responds to a CW in the data.

In Tutorial 2 we will take the dataset you make here and *search* for the signal with frequentist statistics, `QuickCW`, and (optionally, on a GPU) `Prometheus`.""")

md(r"""## 0. Packages

This notebook needs `enterprise`, `enterprise_extensions`, `discovery` (and JAX), `scikit-sparse`, and `healpy`. Everything runs on a laptop CPU.""")

code(r"""# imports
import glob, json
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import scipy.linalg as sl
import scipy.sparse as ss

# scikit-sparse speeds up simulating an HD-correlated GWB, but is optional
# (we fall back to a dense Cholesky if it isn't installed)
try:
    from sksparse.cholmod import cholesky
    HAVE_SKSPARSE = True
except ImportError:
    HAVE_SKSPARSE = False

import astropy.units as u

from enterprise.signals import signal_base, selections, white_signals, gp_signals, utils
import enterprise.signals.parameter as parameter
from enterprise.signals.gp_signals import get_timing_model_basis, BasisGP
from enterprise.signals.parameter import function
from enterprise.signals.utils import create_gw_antenna_pattern
from enterprise import constants as const

from enterprise_extensions import load_feathers
from enterprise_extensions.blocks import common_red_noise_block
from enterprise_extensions.deterministic import cw_delay, CWSignal

import healpy as hp""")

md(r"""## 1. The detector: our pulsar timing array

Our detector is the pulsar timing array itself. We've stored the timing models and observations for 12 synthetic pulsars (mimicking the IPTA DR2 dataset) as "feather" files. Each pulsar has a realistic sky position, timing baseline (13–22 years), TOA uncertainties (~0.1–0.5 µs), and a measured distance with uncertainty (`psr.pdist`) — that last one will turn out to matter a lot for CWs.

Any `dmx` or `_pdist` warnings from the loader can be ignored.""")

code(r"""feather_dir = './data_products/'

# Reference epoch for every CW model in these tutorials. A CW's phase (and,
# since the binary evolves, its frequency) are quoted AT some reference time,
# so injection and search must agree on it. QuickCW hard-codes MJD 53000, so
# we adopt that everywhere and all our numbers stay comparable.
TREF = 53000.0 * 86400.0    # seconds

# load the pulsars as enterprise Pulsar objects
psrs = load_feathers.load_feathers_from_folder(feather_dir)

# back up the pulsar distances, and wipe the residuals to zero --
# we are going to replace them with our own simulated data
for psr in psrs:
    psr._pdist = psr.pdist
    psr.residuals = np.array(psr.toas) * 0.0

Npulsars = len(psrs)
for p in psrs:
    print(f'{p.name}:  {len(p.toas):5d} TOAs over {(p.toas.max()-p.toas.min())/86400/365.25:5.1f} yr,  '
          f'distance {p.pdist[0]:.2f} +/- {p.pdist[1]:.2f} kpc')""")

md(r"""## 2. Simulating a CW in the array

The neat trick here: an `enterprise` PTA object is not just a likelihood — it's also a *generative model*. It knows the deterministic delays (like a CW) and the covariance of every Gaussian process (white noise, red noise, GWB) in the model. So to simulate data we build a PTA model containing everything we want in the data, then draw one random realisation of every stochastic process and add the deterministic signals on top.

The function below does exactly that (this is the same simulation code used for research-grade PTA simulations, just simplified to a single CW source).""")

code(r'''def simulate(pta, params, sparse_cholesky=True):
    """Simulate residuals with enterprise (instead of libstempo/PINT).

    Draws one realisation of every Gaussian process in the PTA model
    (red noise, GWB, ...), adds white noise at the TOA uncertainties,
    and adds all deterministic delays (the CW!) evaluated at `params`.
    """
    delays, ndiags, fmats, phis = (pta.get_delay(params=params),
                                   pta.get_ndiag(params=params),
                                   pta.get_basis(params=params),
                                   pta.get_phi(params=params))

    gpresiduals = []
    if pta._commonsignals:
        # correlated signals (e.g. an HD-correlated GWB) need the full
        # inter-pulsar covariance matrix
        if sparse_cholesky and HAVE_SKSPARSE:
            cf = cholesky(ss.csc_matrix(phis))
            gp = np.zeros(phis.shape[0])
            gp[cf.P()] = np.dot(cf.L().toarray(), np.random.randn(phis.shape[0]))
        else:
            gp = np.dot(sl.cholesky(phis, lower=True), np.random.randn(phis.shape[0]))

        i = 0
        for fmat in fmats:
            j = i + fmat.shape[1]
            gpresiduals.append(np.dot(fmat, gp[i:j]))
            i = j
        assert len(gp) == i
    else:
        for fmat, phi in zip(fmats, phis):
            if phi is None:
                gpresiduals.append(0)
            elif phi.ndim == 1:
                gpresiduals.append(np.dot(fmat, np.sqrt(phi) * np.random.randn(phi.shape[0])))
            else:
                raise NotImplementedError

    whiteresiduals = []
    for delay, ndiag in zip(delays, ndiags):
        if ndiag is None:
            whiteresiduals.append(0)
        elif isinstance(ndiag, signal_base.ShermanMorrison):
            n = np.diag(ndiag._nvec)
            for j, s in zip(ndiag._jvec, ndiag._slices):
                n[s, s] += j
            whiteresiduals.append(delay + np.dot(sl.cholesky(n, lower=True), np.random.randn(n.shape[0])))
        elif ndiag.ndim == 1:
            whiteresiduals.append(delay + np.sqrt(ndiag) * np.random.randn(ndiag.shape[0]))
        else:
            raise NotImplementedError

    return [np.array(g + w) for g, w in zip(gpresiduals, whiteresiduals)]


def set_residuals(psr, y):
    """Push simulated residuals into an enterprise Pulsar object."""
    psr._residuals[psr._isort] = y


@function
def tm_prior(weights, toas, variance=1e40):
    return weights * variance * len(toas)


def TimingModel(coefficients=False, name='linear_timing_model',
                use_svd=False, normed=True, prior_variance=1e40):
    """Class factory for marginalized linear timing model signals."""
    basis = get_timing_model_basis(use_svd, normed)
    prior = tm_prior(variance=prior_variance)
    BaseClass = BasisGP(prior, basis, coefficients=coefficients, name=name)

    class TimingModel(BaseClass):
        signal_type = 'basis'
        signal_name = 'linear timing model'
        signal_id = name + '_svd' if use_svd else name

    return TimingModel''')

md(r"""### Build the injection model

Now we build the PTA model that defines what goes *into* the data:

- the (marginalized) linear **timing model**,
- instrumental **white noise** at the stated TOA uncertainties,
- optionally, pulsar-intrinsic **red noise** (a power law, independent per pulsar),
- optionally, a Hellings–Downs-correlated **GWB**,
- one **CW source**. This is a circular binary that evolves according to general relativity: **both the Earth term and the pulsar term are always injected into the data** (more on what those are in Section 3!). One subtlety worth noticing in the code: we do *not* give the model an explicit pulsar-term phase parameter (`p_phase`). When it's absent, `cw_delay` derives the pulsar-term phase from the pulsar distance and the binary's GR evolution — a *phase–distance connected* injection, which is the physically self-consistent choice.""")

code(r'''def build_injection_pta(psrs, include_rednoise=False, include_gwb=False, components=30):
    """PTA object defining everything we want in the simulated data."""
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    selection = selections.Selection(selections.by_backend)
    efac = parameter.Constant(1)
    equad = parameter.Constant(-8)

    # Reference time for the CW model: phase0 (and, for an evolving binary,
    # the frequency) are defined AT this epoch. We use MJD 53000, which is
    # what QuickCW hard-codes, so the values we inject here are directly
    # comparable to what the search recovers in Tutorial 2.
    tref = TREF

    tm = TimingModel(coefficients=False, name='linear_timing_model',
                     use_svd=False, normed=True, prior_variance=1e-14)

    # the CW source: circular binary, GR evolution, Earth term + pulsar term.
    # note there is NO p_phase parameter: with p_phase absent, the pulsar-term
    # phase is computed from the pulsar distance (phase-connected injection).
    # p_dist is per pulsar: the distance OFFSET from the measured value,
    # in units of its uncertainty (so p_dist = 0 means the measured distance).
    cw_wf = cw_delay(cos_gwtheta=parameter.Uniform(-1, 1)('cw_cos_gwtheta'),
                     gwphi=parameter.Uniform(0, 2*np.pi)('cw_gwphi'),
                     log10_h=parameter.Uniform(-18, -11)('cw_log10_h'),
                     log10_mc=parameter.Uniform(6, 10)('cw_log10_mc'),
                     log10_fgw=parameter.Uniform(-9, -7)('cw_log10_fgw'),
                     cos_inc=parameter.Uniform(-1, 1)('cw_cos_inc'),
                     psi=parameter.Uniform(0, np.pi)('cw_psi'),
                     phase0=parameter.Uniform(0, 2*np.pi)('cw_phase0'),
                     p_dist=parameter.Normal(0, 1),
                     psrTerm=True, evolve=True, tref=tref)
    cw = CWSignal(cw_wf, psrTerm=True, name='cw')

    models = []
    for p in psrs:
        s = tm
        s += white_signals.MeasurementNoise(efac=efac, selection=selection)
        s += white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)

        if include_rednoise:
            pl = utils.powerlaw(log10_A=parameter.Uniform(-18, -11),
                                gamma=parameter.Uniform(0, 7))
            s += gp_signals.FourierBasisGP(spectrum=pl, components=components,
                                           Tspan=Tspan, name='rednoise')

        if include_gwb:
            s += common_red_noise_block(psd='powerlaw', prior='log-uniform',
                                        components=components, orf='hd', name='gwb')

        s += cw

        models.append(s(p))

    pta = signal_base.PTA(models)
    pta.set_default_params({})
    return pta''')

md(r"""### Decide what to inject

A CW from a single circular binary is described by 8 "global" parameters (shared by every pulsar):

| parameter | meaning |
|---|---|
| `cw_cos_gwtheta`, `cw_gwphi` | sky location of the source (polar/azimuthal, celestial coords) |
| `cw_log10_fgw` | GW frequency (= **twice** the orbital frequency) [Hz] |
| `cw_log10_h` | GW strain amplitude |
| `cw_log10_mc` | chirp mass [M$_\odot$] |
| `cw_cos_inc` | orbital inclination |
| `cw_psi` | GW polarization angle |
| `cw_phase0` | initial GW phase |

plus one parameter *per pulsar*: the pulsar distance `<psr>_cw_p_dist`, expressed as an **offset from the measured distance in units of its uncertainty** (so 0 = the measured distance). It sets where the pulsar term lives (Section 3). For the injection we place every pulsar at its measured distance.

We'll put the source at the approximate sky location of the Virgo cluster and make it **loud** ($h = 10^{-13.3}$, about 0.4 µs of induced residual — a few times our TOA errors) so we can see it by eye.

NB: most CWs that PTAs could realistically detect are expected to have $\mathcal{M}\sim10^{9-10} M_\odot$ and $f_{\rm gw}\sim$ 5–15 nHz. Blind searches use much wider ranges than that.""")

code(r"""# sky location of the Virgo cluster
ra_in = (12 + 27/60) * 15      # degrees (hours x 15 = degrees)
dec_in = 12 + 43/60            # degrees
gwtheta_in = np.pi/2 - dec_in*np.pi/180   # theta = pi/2 - dec
gwphi_in = ra_in*np.pi/180

cw_injection = {
    'cw_cos_gwtheta': np.cos(gwtheta_in),   # cosine of polar angle
    'cw_gwphi':       gwphi_in,             # azimuthal angle [rad]
    'cw_log10_fgw':   np.log10(2e-8),       # GW frequency: 20 nHz
    'cw_log10_h':     -13.3,                # loud strain amplitude
    'cw_log10_mc':    9.5,                  # chirp mass: 3e9 Msun
    'cw_cos_inc':     0.5,                  # inclination
    'cw_psi':         1.0,                  # polarization angle [rad]
    'cw_phase0':      np.pi/3,              # initial GW phase [rad]
}""")

md(r"""To keep everything reproducible and easy to play with, the function below assembles the full injection parameter dictionary and simulates one realisation of the data. The toggles are the interesting bit:

- `include_rednoise` — adds pulsar-intrinsic red noise, with amplitudes/slopes drawn from realistic PTA ranges
- `include_gwb` — adds an HD-correlated GWB at a realistic amplitude
- `cw_params` — the CW injection dictionary (set `None` for no CW at all)""")

code(r'''def simulate_dataset(psrs, cw_params=cw_injection, include_rednoise=False,
                     include_gwb=False, gwb_log10_A=-14.5, seed=1234):
    """Build an injection PTA, assemble parameters, and simulate residuals.

    Returns (residual dict {psr name: array}, injection parameter dict).
    """
    np.random.seed(seed)
    pta = build_injection_pta(psrs, include_rednoise=include_rednoise,
                              include_gwb=include_gwb)

    params = {}
    if include_rednoise:
        # red noise drawn from realistic parameter ranges based on PTA analyses
        params.update({p: np.random.uniform(-18, -13) for p in pta.param_names
                       if 'rednoise_log10_A' in p})
        params.update({p: np.random.uniform(2, 6) for p in pta.param_names
                       if 'rednoise_gamma' in p})
    if include_gwb:
        params.update({'gwb_gamma': 4.333, 'gwb_log10_A': gwb_log10_A})
    if cw_params is not None:
        params.update(cw_params)
        # each pulsar term is placed at the measured pulsar distance
        # (p_dist is an offset from the measurement in units of its sigma)
        for psr in psrs:
            params[f'{psr.name}_cw_p_dist'] = 0.0

    resids = simulate(pta, params, sparse_cholesky=True)
    return {p.name: r for p, r in zip(psrs, resids)}, params


def plot_residuals(resid_dict, psrs=psrs, title=None, compare=None, labels=None):
    """Plot residuals for every pulsar. Optionally overlay a second dataset."""
    ncols = 3
    nrows = int(np.ceil(len(psrs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.5*nrows), sharex=True)
    for ax, psr in zip(axes.ravel(), psrs):
        mjd = (psr.toas*u.s).to(u.day).value
        ax.errorbar(mjd, resid_dict[psr.name]/1e-6, psr.toaerrs/1e-6,
                    fmt='.', ms=3, alpha=0.6,
                    label=(labels[0] if labels else None))
        if compare is not None:
            ax.errorbar(mjd, compare[psr.name]/1e-6, psr.toaerrs/1e-6,
                        fmt='.', ms=3, alpha=0.4, color='C3',
                        label=(labels[1] if labels else None))
        ax.set_title(psr.name, fontsize=10)
        ax.set_ylabel('res [µs]')
    for ax in axes[-1]:
        ax.set_xlabel('MJD')
    if labels:
        axes[0, 0].legend(fontsize=8)
    if title:
        fig.suptitle(title, y=1.005, fontsize=14)
    fig.tight_layout()
    plt.show()''')

code(r"""# simulate the simplest dataset: white noise + our loud CW, nothing else
resid_cw, injection_params = simulate_dataset(psrs, include_rednoise=False,
                                              include_gwb=False, seed=1234)
plot_residuals(resid_cw, title='White noise + CW')""")

md(r"""You should clearly see a sinusoid-ish oscillation in several pulsars — that's the CW! Notice:

- The **amplitude and phase differ from pulsar to pulsar** even though it's one source. That's the detector response (antenna pattern) and the pulsar term at work — Section 3.
- Some pulsars barely show the signal at all. Look at where they are on the sky relative to the source.

### Exercise 1a: play with the data ✏️

Use `simulate_dataset` and `plot_residuals` to build intuition. Things to try, with questions to discuss:

1. **Add a GWB** (`include_gwb=True`). How is the GWB visually different from the CW? Could you tell them apart by eye? (Try `compare=`/`labels=` in `plot_residuals` to overlay datasets.)
2. **Add intrinsic red noise** (`include_rednoise=True`). Red noise differs pulsar-to-pulsar. Rerun with a few different `seed`s — how does the by-eye detectability of the CW change?
3. **Make the CW quieter** (`cw_log10_h` = −13.5, −14, ...). At what amplitude do you lose it by eye? Compare with your neighbours — this is basically a by-eye sensitivity curve!
4. **Move the source** (change `cw_cos_gwtheta`, `cw_gwphi`). Which pulsars respond most strongly now?
5. **Lower the frequency** (`cw_log10_fgw` → 3 nHz). What starts to happen when the CW period approaches the data span? Which noise process could you now confuse it with?""", sid="ex1a_intro")

code(r"""# --- solution / example exploration ---

# 1. white noise + CW  vs  white noise + CW + GWB
resid_gwb, _ = simulate_dataset(psrs, include_gwb=True, seed=1234)
plot_residuals(resid_cw, compare=resid_gwb, labels=['CW only', 'CW + GWB'],
               title='Adding a GWB (A=10^-14.5)')

# 2. also with intrinsic red noise
resid_rn, _ = simulate_dataset(psrs, include_rednoise=True, include_gwb=True, seed=1234)
plot_residuals(resid_rn, title='CW + GWB + intrinsic red noise')

# 3. a much quieter CW
quiet_cw = dict(cw_injection, cw_log10_h=-14.0)
resid_quiet, _ = simulate_dataset(psrs, cw_params=quiet_cw, seed=1234)
plot_residuals(resid_quiet, title='Quiet CW (log10_h = -14): invisible by eye')""", sid="ex1a_solution")
student("ex1a_solution", r"""# space to explore! e.g.:
# resid_gwb, _ = simulate_dataset(psrs, include_gwb=True, seed=1234)
# plot_residuals(resid_cw, compare=resid_gwb, labels=['CW only', 'CW + GWB'])

""")

md(r"""Some things you probably noticed:

- The **GWB** looks like smooth long-timescale wandering that is *different in every pulsar* (it's a random process — the famous Hellings–Downs correlation between pulsars is far too weak to see by eye).
- **Intrinsic red noise** can look a lot like a low-frequency CW in a *single* pulsar. What distinguishes a CW is that it appears **coherently across pulsars** at a common frequency with a specific sky-dependent amplitude/phase pattern. This is why CW searches are fundamentally multi-pulsar exercises.
- A CW with a period approaching the data span gets absorbed by red noise models and the timing model fit — PTA sensitivity drops steeply at both ends of the band.""", sid="ex1a_discussion")

md(r"""## 3. Anatomy of the CW signal

### What does our detector look like?

Before dissecting the waveform, let's get familiar with the PTA's response to a CW. The cells below plot the **antenna pattern**: how strongly a pulsar at any sky position responds to a GW source at a given location. The response has separate patterns for the two GW polarizations ($+$ and $\times$), and we also plot $\cos\mu$, the angle between the source and pulsar directions.""")

code(r'''def plot_apf(psrs, gwtheta, gwphi):
    """Plot the PTA's antenna response pattern for a GW source at (gwtheta, gwphi)."""
    nside = 8
    npix = hp.nside2npix(nside)
    data_p = np.zeros(npix)
    data_x = np.zeros(npix)
    data_m = np.zeros(npix)
    for pix in range(npix):
        theta, phi = hp.pix2ang(nside, pix)
        pos = np.array([np.sin(theta)*np.cos(phi),
                        np.sin(theta)*np.sin(phi),
                        np.cos(theta)])
        data_p[pix], data_x[pix], data_m[pix] = create_gw_antenna_pattern(pos, gwtheta, gwphi)

    names = ['Plus', 'Cross', r'$\cos\mu$']
    fig, axes = plt.subplots(1, 3, figsize=(18, 12))
    for d, dat in enumerate([data_p, data_x, data_m]):
        plt.axes(axes[d])
        hp.mollview(dat, rot=180, title=names[d], hold=True, cmap='binary_r')
        for i, psr in enumerate(psrs):
            hp.visufunc.projscatter(psr.theta, psr.phi, marker='*', s=200,
                                    edgecolor='w', color=f'C{i%10}')
        hp.visufunc.projscatter(gwtheta, gwphi, marker='D', s=100,
                                edgecolor='w', color='r')
        hp.graticule(15, 30)
    plt.show()

plot_apf(psrs, gwtheta_in, gwphi_in)''')

md(r"""The stars are our 12 pulsars and the red diamond is the injected GW source. Positive responses are white and negative black — the response is informative either way, while a response near 0 means insensitive.

**Based on the antenna patterns, which pulsars do you expect to be most/least sensitive to our source? Scroll back up to your residual plots and check!**

### Earth term + pulsar term

A CW is not just a sinusoid. The signal from a single binary is

$$s(t) = F^+(\theta,\phi,\psi)\,[s_+(t_p) - s_+(t)] \;+\; F^\times(\theta,\phi,\psi)\,[s_\times(t_p) - s_\times(t)]$$

where $F^{+,\times}$ are the antenna patterns above. $s_{+,\times}(t)$ is the signal imprinted when the GW passes the **Earth** (the "Earth term"), and $s_{+,\times}(t_p)$ when it passed the **pulsar** (the "pulsar term"), at

$$t_p = t - L\,(1 - \cos\mu)$$

with $L$ the pulsar distance. The pulsar term is a snapshot of the binary **hundreds to thousands of years in the past** — the binary was slower then, so the pulsar term sits at a *lower frequency* than the Earth term. Every pulsar carries its own pulsar term (different $L$, different $\mu$), while the Earth term is common to all.

Let's decompose our injected signal into the two pieces with `enterprise_extensions`'s `cw_delay` — this is *the* CW waveform function that `enterprise`-based searches (including `QuickCW` in Tutorial 2) use as their signal model.""")

code(r"""# the same reference time we injected with (and the one QuickCW uses)
tref = TREF

nshow = 6  # first 6 pulsars; change to look at the others
plt.figure(figsize=(16, 3.2*int(np.ceil(nshow/2))))
for i, psr in enumerate(psrs[:nshow]):
    ax = plt.subplot(int(np.ceil(nshow/2)), 2, i+1)

    common = dict(cos_gwtheta=cw_injection['cw_cos_gwtheta'],
                  gwphi=cw_injection['cw_gwphi'],
                  log10_h=cw_injection['cw_log10_h'],
                  log10_fgw=cw_injection['cw_log10_fgw'],
                  log10_mc=cw_injection['cw_log10_mc'],
                  cos_inc=cw_injection['cw_cos_inc'],
                  psi=cw_injection['cw_psi'],
                  phase0=cw_injection['cw_phase0'],
                  p_dist=0, evolve=True, phase_approx=False, tref=tref)

    # Earth term ONLY (psrTerm=False)
    cw_e = cw_delay(psr.toas.copy(), psr.pos, pdist=psr.pdist, psrTerm=False, **common)
    # FULL signal (psrTerm=True)
    cw_total = cw_delay(psr.toas.copy(), psr.pos, pdist=psr.pdist, psrTerm=True, **common)
    # there's no pulsar-term-only option, so: (full) - (Earth) = (pulsar term)
    cw_p = cw_total - cw_e

    mjd = (psr.toas*u.s).to(u.day).value
    ax.plot(mjd, cw_e, 'k--', alpha=0.7, lw=2, label='Earth')
    ax.plot(mjd, cw_p, 'k--', alpha=0.3, lw=2, label='Pulsar')
    ax.plot(mjd, cw_total, color=f'C{i%10}', lw=2.5, label='Total')
    ax.set_title(psr.name)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('MJD'); ax.set_ylabel('Residual (s)')
plt.tight_layout()
plt.show()""")

md(r"""**Can you see that the pulsar term oscillates more slowly than the Earth term?** The total signal is the beat of the two — that's why the residuals didn't look like clean sinusoids.

### Pulsar distances

The pulsar distance $L$ sets the pulsar-term lag $L(1-\cos\mu)$. Here's the catch: the GW wavelength is $\sim$ 1 pc, while typical pulsar distance *uncertainties* are tens to hundreds of pc. So a small change in the assumed distance completely scrambles the pulsar-term phase. Below we draw a few distances from each pulsar's measured distance uncertainty and watch the waveform change.""")

code(r"""Ndraws = 10
nshow = 3
np.random.seed(42)

plt.figure(figsize=(16, 3.2*nshow))
for i, psr in enumerate(psrs[:nshow]):
    ax = plt.subplot(nshow, 1, i+1)
    mjd = (psr.toas*u.s).to(u.day).value

    common = dict(cos_gwtheta=cw_injection['cw_cos_gwtheta'],
                  gwphi=cw_injection['cw_gwphi'],
                  log10_h=cw_injection['cw_log10_h'],
                  log10_fgw=cw_injection['cw_log10_fgw'],
                  log10_mc=cw_injection['cw_log10_mc'],
                  cos_inc=cw_injection['cw_cos_inc'],
                  psi=cw_injection['cw_psi'],
                  phase0=cw_injection['cw_phase0'],
                  psrTerm=True, evolve=True, phase_approx=False, tref=tref)

    # the injected waveform, at the measured distance
    cw_in = cw_delay(psr.toas.copy(), psr.pos, pdist=psr.pdist, p_dist=0, **common)

    # same signal, but drawing the distance from its measurement uncertainty
    # (p_dist is the offset from the measured distance, in units of its sigma)
    for _ in range(Ndraws):
        cwd = cw_delay(psr.toas.copy(), psr.pos, pdist=psr.pdist,
                       p_dist=np.random.randn(), **common)
        ax.plot(mjd, cwd, color=f'C{i%10}', alpha=0.2)

    ax.plot(mjd, cw_in, color=f'C{i%10}', lw=2.5)
    ax.set_title(f'{psr.name}   (L = {psr.pdist[0]:.2f} ± {psr.pdist[1]:.2f} kpc)')
    ax.set_xlabel('MJD'); ax.set_ylabel('Residual (s)')
plt.tight_layout()
plt.show()""")

md(r"""So why bother with the pulsar term at all? **Because it lets us measure more of the source's properties.** From the Earth term alone, all we can constrain is the strain amplitude

$$h_0 = \frac{2\,\mathcal{M}^{5/3}(\pi f_{\rm gw})^{2/3}}{d_L},$$

a degenerate combination of chirp mass, frequency, and distance. But the pulsar terms are snapshots of the binary at earlier times: the frequency difference between the pulsar terms and the Earth term measures how fast the binary is evolving,

$$\frac{\mathrm{d}\omega}{\mathrm{d}t} = \frac{96}{5}\mathcal{M}^{5/3}\omega^{11/3},$$

and hence the chirp mass — which breaks the degeneracy. (You still can't get *everything*: searches sample either $h_0$ or $d_L$, not both.)

### Exercise 3a: CW parameter exploration ✏️

Use the space below and `cw_delay` to explore the waveform. **Which parameters affect the strength of the signal? Which produce a bigger difference between Earth-term and pulsar-term frequencies?** Compare with friends, and consult the hints at the end of the section.""", sid="ex3a_intro")

code(r"""# --- solution / example exploration ---
psr = psrs[0]
mjd = (psr.toas*u.s).to(u.day).value

base = dict(cos_gwtheta=cw_injection['cw_cos_gwtheta'], gwphi=cw_injection['cw_gwphi'],
            log10_h=-13.0, log10_fgw=np.log10(2e-8), log10_mc=9.5,
            cos_inc=0.5, psi=1.0, phase0=np.pi/3, p_dist=0,
            psrTerm=True, evolve=True, phase_approx=False, tref=tref)

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# louder: higher strain
for lh, c in zip([-13.5, -13.0, -12.7], ['C0', 'C1', 'C3']):
    kw = dict(base, log10_h=lh)
    axes[0].plot(mjd, cw_delay(psr.toas.copy(), psr.pos, pdist=psr.pdist, **kw),
                 color=c, label=f'log10_h = {lh}')
axes[0].legend(); axes[0].set_ylabel('Residual (s)')
axes[0].set_title(f'{psr.name}: strain sets the overall amplitude')

# higher chirp mass: faster evolution, bigger Earth/pulsar-term frequency split
for lmc, c in zip([8.5, 9.5, 10.0], ['C0', 'C1', 'C3']):
    kw = dict(base, log10_mc=lmc)
    axes[1].plot(mjd, cw_delay(psr.toas.copy(), psr.pos, pdist=psr.pdist, **kw),
                 color=c, label=f'log10_mc = {lmc}')
axes[1].legend(); axes[1].set_ylabel('Residual (s)'); axes[1].set_xlabel('MJD')
axes[1].set_title('chirp mass sets the binary evolution → Earth vs pulsar term frequency split')
plt.tight_layout()
plt.show()""", sid="ex3a_solution")
student("ex3a_solution", r"""# space to explore the CW signal

""")

md(r"""#### Hints

From the strain equation, more massive and higher-frequency systems are louder, with the chirp mass $\mathcal{M}$ having the larger effect; closer systems (smaller $d_L$) are louder too.

From the frequency evolution equation, a higher chirp mass or higher orbital frequency makes the binary evolve faster — so the Earth term and pulsar terms separate further in frequency.""")

md(r"""## 4. The same signal model in `discovery`

`discovery` is a JAX rewrite of the PTA likelihood. Two things make it interesting for CWs:

1. The whole likelihood is **differentiable** — you get gradients (and Hessians) of $\ln \mathcal{L}$ for free, which enables gradient-based samplers (HMC/NUTS — this is exactly what `Prometheus` exploits on GPUs in Tutorial 2).
2. It's **fast**, especially vectorised over many parameter values.

To use it we need the CW waveform written in JAX. Rather than importing one, we'll write it ourselves in the next cell — it's ~50 lines, it's the exact same physics as the `enterprise` `cw_delay` model we injected (GR-evolving circular binary, pulsar-term phase derived from the pulsar distance), and seeing the waveform laid out end-to-end is half the point of this section. We'll then *verify* it against `enterprise` before poking the likelihood.""")

code(r"""import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # remove this line to run JAX on a GPU

# Compatibility shim for SciPy >= 1.18, which returns the `lower` flag from
# cho_factor as a 0-d array instead of a bool. JAX requires that flag to be
# hashable (it is a static argument), so discovery's jitted likelihood raises
# "Non-hashable static arguments are not supported" without this. Harmless on
# older SciPy. Remove once discovery handles it upstream.
import scipy.linalg as _sl
_cho_factor = _sl.cho_factor
_sl.cho_factor = lambda *a, **kw: (lambda r: (r[0], bool(r[1])))(_cho_factor(*a, **kw))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import discovery as ds""")

code(r'''def make_phase_connected_binary(pulsarterm=True, tref=0.0):
    """The evolving circular-binary CW delay, written in JAX.

    Same waveform as enterprise_extensions' cw_delay(evolve=True) with no
    explicit pulsar-term phase: the pulsar-term phase follows from the
    pulsar distance. NOTE: here p_dist is the ABSOLUTE distance in kpc.
    (No @jax.jit needed here -- the whole likelihood gets jitted later.)
    """
    def binary_delay(toas, pos, cos_gwtheta, gwphi, cos_inc, log10_mc,
                     log10_fgw, log10_h, phase0, psi, p_dist=1.0):
        t = jnp.asarray(toas) - tref
        x, y, z = pos

        # antenna patterns and cos(mu), same math as
        # enterprise.signals.utils.create_gw_antenna_pattern
        sin_phi, cos_phi = jnp.sin(gwphi), jnp.cos(gwphi)
        cos_theta = cos_gwtheta
        sin_theta = jnp.sqrt(1.0 - cos_theta**2)
        m_dot = sin_phi*x - cos_phi*y
        n_dot = -cos_theta*cos_phi*x - cos_theta*sin_phi*y + sin_theta*z
        om_dot = -sin_theta*cos_phi*x - sin_theta*sin_phi*y - cos_theta*z
        fplus = 0.5*(m_dot**2 - n_dot**2) / (1.0 + om_dot)
        fcross = (m_dot*n_dot) / (1.0 + om_dot)
        cos_mu = -om_dot

        mc = (10.0**log10_mc) * const.Tsun          # chirp mass in seconds
        w0 = jnp.pi * (10.0**log10_fgw)             # orbital frequency at tref
        inc = jnp.arccos(cos_inc)
        phase0_orb = 0.5*phase0                     # GW phase -> orbital phase
        dist = 2.0 * mc**(5/3) * w0**(2/3) / (10.0**log10_h)

        # pulsar time: the GW passed the pulsar L(1 - cos mu) ago
        tp = t - (const.kpc/const.c) * p_dist * (1.0 - cos_mu)

        def evolve(time):
            # GR frequency evolution of a circular binary, and the
            # phase obtained by integrating it -- this is what makes the
            # model "phase-connected": pulsar-term phase follows from L
            omega = w0 * (1.0 - (256/5) * mc**(5/3) * w0**(8/3) * time)**(-3/8)
            phase = phase0_orb + (1.0/(32.0*mc**(5/3))) * (w0**(-5/3) - omega**(-5/3))
            return omega, phase

        omega, phase = evolve(t)        # Earth term
        omega_p, phase_p = evolve(tp)   # pulsar term

        At = -0.5*jnp.sin(2*phase)*(3.0 + jnp.cos(2*inc))
        Bt = 2.0*jnp.cos(2*phase)*jnp.cos(inc)
        At_p = -0.5*jnp.sin(2*phase_p)*(3.0 + jnp.cos(2*inc))
        Bt_p = 2.0*jnp.cos(2*phase_p)*jnp.cos(inc)

        alpha = mc**(5/3) / (dist * omega**(1/3))
        alpha_p = mc**(5/3) / (dist * omega_p**(1/3))

        rplus = alpha * (-At*jnp.cos(2*psi) + Bt*jnp.sin(2*psi))
        rcross = alpha * (At*jnp.sin(2*psi) + Bt*jnp.cos(2*psi))
        rplus_p = alpha_p * (-At_p*jnp.cos(2*psi) + Bt_p*jnp.sin(2*psi))
        rcross_p = alpha_p * (At_p*jnp.sin(2*psi) + Bt_p*jnp.cos(2*psi))

        if pulsarterm:
            return fplus*(rplus_p - rplus) + fcross*(rcross_p - rcross)
        return -fplus*rplus - fcross*rcross

    return binary_delay''')

code(r"""# load the same pulsars as discovery Pulsar objects
disco_psrs = [ds.Pulsar.read_feather(f)
              for f in sorted(glob.glob(feather_dir + '*.feather'))]
print([p.name for p in disco_psrs])""")

code(r"""# --- check: discovery's CW waveform == enterprise's CW waveform ---

# enterprise: ask the injection PTA for its deterministic delay (the CW)
pta_check = build_injection_pta(psrs)   # white noise + CW only
ent_delays = pta_check.get_delay(params=injection_params)

# our JAX waveform, evaluated at the same parameters
cw_disco = make_phase_connected_binary(pulsarterm=True, tref=tref)

i = 0  # pick a pulsar
dpsr = [p for p in disco_psrs if p.name == psrs[i].name][0]
disco_delay = cw_disco(dpsr.toas, dpsr.pos,
                       cos_gwtheta=cw_injection['cw_cos_gwtheta'],
                       gwphi=cw_injection['cw_gwphi'],
                       cos_inc=cw_injection['cw_cos_inc'],
                       log10_mc=cw_injection['cw_log10_mc'],
                       log10_fgw=cw_injection['cw_log10_fgw'],
                       log10_h=cw_injection['cw_log10_h'],
                       phase0=cw_injection['cw_phase0'],
                       psi=cw_injection['cw_psi'],
                       p_dist=psrs[i].pdist[0])

mjd = (np.array(dpsr.toas)*u.s).to(u.day).value
plt.figure(figsize=(12, 3.5))
plt.plot(mjd, np.array(ent_delays[i]), lw=3, alpha=0.5, label='enterprise (injected)')
plt.plot(mjd, np.array(disco_delay), 'k--', lw=1.2, label='discovery')
plt.xlabel('MJD'); plt.ylabel('CW delay (s)')
plt.title(f'{psrs[i].name}: the two signal models agree')
plt.legend()
plt.show()""")

md(r"""### A differentiable CW likelihood

Now let's build the full `discovery` likelihood for our simulated dataset: white noise + timing model + the CW delay (with pulsar term). We hand it the residuals we simulated in Section 2, JIT-compile it, and take gradients.""")

code(r'''# whether to include the pulsar term in the *model*
# (it is always in the *data*!)
pulsar_term = True

# fixed white-noise parameters, matching the injection
noisedict = {}
for psr in disco_psrs:
    noisedict[psr.name + '_KAT_MKBF_efac'] = 1.0
    noisedict[psr.name + '_KAT_MKBF_log10_t2equad'] = -8.0
    noisedict[psr.name + '_KAT_MKBF_log10_ecorr'] = -8.0

noise_terms = {psr.name: ds.makenoise_measurement(psr, noisedict=noisedict)
               for psr in disco_psrs}
timing_terms = {psr.name: ds.makegp_timing(psr, variance=1e-14)
                for psr in disco_psrs}

# real searches always model intrinsic red noise alongside the CW, so we
# include a power-law red-noise GP for every pulsar in the search model too
# (we didn't inject any red noise, so we'll evaluate it at a tiny amplitude)
Tspan_disco = ds.getspan(disco_psrs)
common_gp = ds.makecommongp_fourier(disco_psrs, ds.powerlaw, 30, Tspan_disco,
                                    name='rednoise')


class CWDelay:
    """Single CW source (Earth term + optional pulsar term) for one pulsar."""
    def __init__(self, psr, include_pterm=True):
        self.psr = psr
        self.include_pterm = include_pterm
        self.waveform = make_phase_connected_binary(pulsarterm=include_pterm,
                                                    tref=tref)
        self.params = ['cw_cos_gwtheta', 'cw_gwphi', 'cw_cos_inc', 'cw_log10_mc',
                       'cw_log10_fgw', 'cw_log10_h', 'cw_phase0', 'cw_psi']
        if include_pterm:
            self.params.append(f'{psr.name}_cw_p_dist')

    def __call__(self, params):
        kwargs = dict(cos_gwtheta=params['cw_cos_gwtheta'], gwphi=params['cw_gwphi'],
                      cos_inc=params['cw_cos_inc'], log10_mc=params['cw_log10_mc'],
                      log10_fgw=params['cw_log10_fgw'], log10_h=params['cw_log10_h'],
                      phase0=params['cw_phase0'], psi=params['cw_psi'])
        if self.include_pterm:
            kwargs['p_dist'] = params[f'{self.psr.name}_cw_p_dist']
        return self.waveform(self.psr.toas, self.psr.pos, **kwargs)


def build_likelihood(residual_map, include_pterm=True):
    """Construct a discovery ArrayLikelihood over all pulsars."""
    pulsar_likes = [
        ds.PulsarLikelihood([
            np.array(residual_map[psr.name], copy=True),
            noise_terms[psr.name],
            timing_terms[psr.name],
            CWDelay(psr, include_pterm=include_pterm),
        ])
        for psr in disco_psrs
    ]
    fml = ds.ArrayLikelihood(pulsar_likes, commongp=common_gp)
    logl = fml.logL

    # base parameter values: the injection, plus red noise evaluated
    # "off" (log10_A = -20) since we didn't inject any.
    # our JAX model uses ABSOLUTE pulsar distances [kpc] (the enterprise
    # injection used offsets from the measurement -- offset 0 = measured)
    measured_dist = {p.name: p.pdist[0] for p in psrs}
    base = {}
    for k in logl.params:
        if k.endswith('_cw_p_dist'):
            base[k] = float(measured_dist[k.replace('_cw_p_dist', '')])
        elif k in injection_params:
            base[k] = injection_params[k]
        elif 'rednoise_log10_A' in k:
            base[k] = -20.0
        elif 'rednoise_gamma' in k:
            base[k] = 4.0
        else:
            raise KeyError(f'no base value for parameter {k}')
    param_keys = list(base.keys())
    base_values = jnp.array([base[k] for k in param_keys], dtype=jnp.float64)
    return logl, param_keys, base_values


logl, param_keys, base_values = build_likelihood(resid_cw, include_pterm=pulsar_term)
print('model parameters:', param_keys)''')

code(r"""def logl_wrapped(x_array):
    params = {k: v for k, v in zip(param_keys, x_array)}
    return logl(params)

logl_fn = jax.jit(logl_wrapped)
grad_fn = jax.jit(jax.grad(logl_fn))

print('lnL at the injected parameters:', logl_fn(base_values))
print('gradient wrt cw_log10_h:',
      grad_fn(base_values)[param_keys.index('cw_log10_h')])""")

md(r"""### What does the likelihood see?

The payoff: scan $\ln\mathcal{L}$ along one parameter at a time, holding everything else at the injected values, and watch how the likelihood responds to the CW. Because the `discovery` likelihood is JAX-vectorised, evaluating it at hundreds of parameter values takes seconds.""")

code(r'''def scan_lnL(scan_param, scan_min, scan_max, gridsteps=300, chunk_size=50):
    """Evaluate lnL on a grid over one parameter, others fixed at injection."""
    scan_idx = param_keys.index(scan_param)
    scan_values = jnp.linspace(scan_min, scan_max, gridsteps)

    batched = jax.jit(jax.vmap(logl_wrapped))
    template = jnp.repeat(base_values[None, :], chunk_size, axis=0)

    out = []
    for start in range(0, gridsteps, chunk_size):
        current = scan_values[start:start+chunk_size]
        pad = chunk_size - current.shape[0]
        padded = current if pad == 0 else jnp.pad(current, (0, pad),
                                                  constant_values=current[-1])
        block = template.at[:, scan_idx].set(padded)
        out.append(np.array(batched(block))[:current.shape[0]])
    return np.array(scan_values), np.concatenate(out)


# scan the CW amplitude
scan_values, logls = scan_lnL('cw_log10_h', -15.0, -12.5)

plt.figure(figsize=(9, 4))
plt.plot(scan_values, logls)
plt.axvline(injection_params['cw_log10_h'], color='k', ls='--', alpha=0.5,
            label='injected value')
plt.xlabel('cw_log10_h'); plt.ylabel('ln L')
plt.title('The likelihood peaks at the injected amplitude')
plt.legend()
plt.show()''')

md(r"""### The pulsar-term likelihood surface

Now for the pulsar-term parameters — the pulsar distances. Here a single number sets the scale of everything: the **GW wavelength**, $\lambda_{\rm gw} = c/f_{\rm gw} \approx 0.5$ pc for our source. Shifting a pulsar's distance by half a wavelength flips the sign of its pulsar term; shifting it by a full wavelength brings it back. Meanwhile the *uncertainty* on a pulsar distance is tens to hundreds of parsecs — hundreds of wavelengths!

To see what that does to the likelihood, we'll map $\ln\mathcal{L}$ over the distances of **two** pulsars at once, in a tiny window (±3 pc) around the true values, with everything else held at the injection. The red X marks the truth.""")

code(r"""# --- the pulsar-term likelihood surface over TWO pulsar distances ---

# not every pulsar has a strong pulsar term (check the antenna patterns!),
# so first find the two whose pulsar terms matter most for the likelihood:
# pulsar-term rms over the TOA uncertainty, times sqrt(N_toa)
def pterm_snr(psr):
    kw = dict(cos_gwtheta=cw_injection['cw_cos_gwtheta'], gwphi=cw_injection['cw_gwphi'],
              log10_h=cw_injection['cw_log10_h'], log10_fgw=cw_injection['cw_log10_fgw'],
              log10_mc=cw_injection['cw_log10_mc'], cos_inc=cw_injection['cw_cos_inc'],
              psi=cw_injection['cw_psi'], phase0=cw_injection['cw_phase0'],
              p_dist=0, evolve=True, phase_approx=False, tref=tref)
    cw_e = cw_delay(psr.toas.copy(), psr.pos, pdist=psr.pdist, psrTerm=False, **kw)
    cw_t = cw_delay(psr.toas.copy(), psr.pos, pdist=psr.pdist, psrTerm=True, **kw)
    return np.std(cw_t - cw_e) / np.median(psr.toaerrs) * np.sqrt(len(psr.toas))

ranked = sorted(psrs, key=pterm_snr, reverse=True)
psr_x, psr_y = ranked[0], ranked[1]
print(f'loudest pulsar terms: {psr_x.name}, {psr_y.name}')

ix = param_keys.index(f'{psr_x.name}_cw_p_dist')
iy = param_keys.index(f'{psr_y.name}_cw_p_dist')

# the GW wavelength is ~0.5 pc, so scan just +/- 1.5 pc around the truth
half_width = 0.0015   # kpc
gridsteps = 120
xvals = np.linspace(-half_width, half_width, gridsteps) + psr_x.pdist[0]
yvals = np.linspace(-half_width, half_width, gridsteps) + psr_y.pdist[0]
XX, YY = np.meshgrid(xvals, yvals)
pts = np.column_stack([XX.ravel(), YY.ravel()])

# evaluate lnL on the grid (vectorised with JAX, in chunks)
batched = jax.jit(jax.vmap(logl_wrapped))
chunk = 200
template = jnp.repeat(base_values[None, :], chunk, axis=0)
out = []
for s in range(0, pts.shape[0], chunk):
    block_pts = pts[s:s+chunk]
    n = block_pts.shape[0]
    if n < chunk:
        block_pts = np.vstack([block_pts, np.repeat(block_pts[-1:], chunk - n, axis=0)])
    block = (template.at[:, ix].set(jnp.asarray(block_pts[:, 0]))
                     .at[:, iy].set(jnp.asarray(block_pts[:, 1])))
    out.append(np.array(batched(block))[:n])
logl_grid = np.concatenate(out).reshape(gridsteps, gridsteps)""")

code(r"""# plot the surface: heatmap + 3D view, truth marked with an X
xpc = (xvals - psr_x.pdist[0]) * 1e3   # offsets from truth in pc
ypc = (yvals - psr_y.pdist[0]) * 1e3

fig = plt.figure(figsize=(15, 6))

ax1 = fig.add_subplot(1, 2, 1)
pcm = ax1.pcolormesh(xpc, ypc, logl_grid, shading='auto')
fig.colorbar(pcm, ax=ax1, label='ln L')
ax1.axhline(0, color='w', lw=0.6, alpha=0.5)
ax1.axvline(0, color='w', lw=0.6, alpha=0.5)
ax1.plot(0, 0, 'rx', ms=14, mew=3, label='true distances')
ax1.set_xlabel(f'{psr_x.name} distance offset [pc]')
ax1.set_ylabel(f'{psr_y.name} distance offset [pc]')
ax1.set_title('The pulsar-term likelihood surface')
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(*np.meshgrid(xpc, ypc), logl_grid, cmap='viridis',
                 rstride=1, cstride=1, linewidth=0, antialiased=False)
ax2.set_xlabel(f'{psr_x.name} offset [pc]')
ax2.set_ylabel(f'{psr_y.name} offset [pc]')
ax2.set_zlabel('ln L')
ax2.set_title('...as a 3D surface')

plt.tight_layout()
plt.show()""")

md(r"""This plot is one of the most important (and painful) facts about CW searches, so let's unpack why the surface looks like this:

- Moving along the **x-axis** changes only the first pulsar's term; along the **y-axis**, only the second pulsar's. Each direction oscillates with its own period of roughly $\lambda_{\rm gw}/(1-\cos\mu) \sim$ 1 pc — the phase of that pulsar's term wrapping through full cycles. The two directions are (nearly) independent, which is why you get an egg-carton grid rather than diagonal stripes.
- The truth (red X) sits on *a* peak — but look how many other peaks are almost as tall, even within this ±3 pc box. Each one corresponds to sliding some pulsar's term by a whole number of wavelengths. Now remember the real distance uncertainties are tens to hundreds of pc: the full prior range contains **hundreds of near-degenerate maxima per pulsar**, and the joint space is that comb raised to the $N_{\rm psr}$-th power.
- This is exactly why the pulsar term is hard: the likelihood knows the *phase* of each pulsar term precisely, but that only pins the distance down modulo a wavelength. Samplers need special jump proposals (in distance and pulsar phase) to hop between these maxima — you'll meet them in `QuickCW` in Tutorial 2.

### Exercise 4a ✏️

1. Scan `cw_log10_fgw` around the injection (try ±0.1 dex with lots of grid points). Why is the frequency likelihood also sharply peaked/multimodal?
2. Rebuild the likelihood with `include_pterm=False` (Earth term only — remember, the *data* still contain pulsar terms). Redo the `cw_log10_h` scan. Does the model still favour the injected amplitude? What does this tell you about ignoring the pulsar term?
3. Remake the distance surface with two *different* pulsars, or widen `half_width`. The oscillation period along each axis is $\lambda_{\rm gw}/(1-\cos\mu)$ — can you confirm that pulsars closer to the source on the sky (larger $\cos\mu$) show *slower* oscillations?""", sid="ex4a_intro")

code(r"""# --- solution ---

# 1. frequency scan: the CW must stay phase-coherent over ~20 years, so a tiny
#    frequency change decoheres the template from the data.
scan_values, logls = scan_lnL('cw_log10_fgw',
                              cw_injection['cw_log10_fgw'] - 0.1,
                              cw_injection['cw_log10_fgw'] + 0.1, gridsteps=500)
plt.figure(figsize=(9, 4))
plt.plot(scan_values, logls)
plt.axvline(cw_injection['cw_log10_fgw'], color='k', ls='--', alpha=0.5, label='injected')
plt.xlabel('cw_log10_fgw'); plt.ylabel('ln L'); plt.legend()
plt.title('Frequency: sharply peaked, with coherence sidelobes')
plt.show()

# 2. Earth-term-only model on pulsar-term data
logl, param_keys, base_values = build_likelihood(resid_cw, include_pterm=False)
logl_fn = jax.jit(logl_wrapped)
scan_values, logls = scan_lnL('cw_log10_h', -15.0, -12.5)
plt.figure(figsize=(9, 4))
plt.plot(scan_values, logls)
plt.axvline(injection_params['cw_log10_h'], color='k', ls='--', alpha=0.5, label='injected')
plt.xlabel('cw_log10_h'); plt.ylabel('ln L'); plt.legend()
plt.title('Earth-term-only model: still detects the CW, but biased & weaker')
plt.show()

# restore the full model for anything below
logl, param_keys, base_values = build_likelihood(resid_cw, include_pterm=True)
logl_fn = jax.jit(logl_wrapped)""", sid="ex4a_solution")
student("ex4a_solution", r"""# space for exercise 4a

""")

md(r"""## 5. Save the dataset for Tutorial 2

Finally, save the simulated residuals and the injection parameters. Tutorial 2 will load these and try to *recover* the source with real search pipelines. We save the clean version (white noise + CW only) so the searches run fast — once you've been through Tutorial 2, come back here, regenerate the data with a GWB or red noise (or a quieter CW!), and see how the searches cope.""")

code(r"""np.savez('sim_data/cw_sim_residuals.npz', **resid_cw)
with open('sim_data/cw_injection_params.json', 'w') as f:
    json.dump({k: float(v) for k, v in injection_params.items()}, f, indent=2)
print('saved sim_data/cw_sim_residuals.npz and sim_data/cw_injection_params.json')""")

md(r"""---
**Continue to Tutorial 2**, where we search for this signal with the $\mathcal{F}_e$/$\mathcal{F}_p$ statistics, `QuickCW`, and (with a GPU) `Prometheus`.""")

# ----------------------------------------------------------------------------

def build(fname, student_version=False):
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                              "name": "python3"}
    for ctype, src, sid in cells:
        if student_version and sid in overrides:
            if overrides[sid] is None:
                continue
            src = overrides[sid]
        if student_version and sid and sid.endswith("_discussion"):
            continue  # keep interpretation for the solution version
        cell = nbf.v4.new_markdown_cell(src) if ctype == "markdown" else nbf.v4.new_code_cell(src)
        nb.cells.append(cell)
    nbf.write(nb, fname)
    print("wrote", fname)

# pass --student-only to regenerate just the student notebook, leaving the
# already-executed solutions notebook (and its outputs) untouched
if '--student-only' not in sys.argv:
    build(f"{OUTDIR}/CW_tutorial_1_simulation_and_signal_models_solutions.ipynb", student_version=False)
build(f"{OUTDIR}/CW_tutorial_1_simulation_and_signal_models.ipynb", student_version=True)
