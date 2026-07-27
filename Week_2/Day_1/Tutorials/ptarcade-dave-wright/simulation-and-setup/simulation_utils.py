"""Utilities for data simulation."""

from pathlib import Path
import discovery as ds
import discovery.samplers.numpyro as ds_numpyro
import numpyro
import jax


def load_pulsars(
    data_dir: Path,
    n_pulsars: int | None = None,
) -> list:
    """Load from feather files."""
    feather_files = sorted(data_dir.glob("v1p1*[JB]*.feather"))
    if n_pulsars is not None:
        feather_files = feather_files[:n_pulsars]
    return [ds.Pulsar.read_feather(f) for f in feather_files]


def build_likelihood_for_sampling(
    psrs: list,
    psd_func,
    psd_func_common_argnames: list[str],
    irn_n_freqs: int | None = None,
    gw_n_freqs: int | None = None,
) -> ds.GlobalLikelihood:
    t_span = ds.getspan(psrs)
    psl_models = [
        ds.PulsarLikelihood(
            [
                psr.residuals,
                ds.makenoise_measurement(psr, psr.noisedict),
                ds.makegp_ecorr(psr, psr.noisedict),
                ds.makegp_timing(psr, svd=True, variance=1e-12),
                ds.makegp_fourier(
                    psr,
                    ds.powerlaw,
                    T=t_span,
                    components=irn_n_freqs,
                    name="red_noise",
                ),
                ds.makegp_fourier(
                    psr,
                    psd_func,
                    gw_n_freqs,
                    T=t_span,
                    name="gw",
                    common=["gw_" + arg for arg in psd_func_common_argnames],
                ),
            ],
        )
        for psr in psrs
    ]
    return ds.GlobalLikelihood(psl_models)

def build_curn_arraylikelihood(psrs: list, common_psd_fn, n_freqs: int, components, crn_prefix="crn"):
    Tspan = ds.getspan(psrs)
    psl_models = [
        ds.PulsarLikelihood(
            [
                psr.residuals,
                ds.makenoise_measurement(psr, psr.noisedict),
                ds.makegp_ecorr(psr, psr.noisedict),
                ds.makegp_timing(psr, svd=True),
            ]
        )
        for psr in psrs
    ]
    common_psd, crn_params = ds.make_combined_crn(n_freqs, ds.powerlaw, common_psd_fn, crn_prefix=crn_prefix)
    commongp = ds.makecommongp_fourier(
        psrs,
        common_psd,
        components,
        T=Tspan,
        common=crn_params,
        name="red_noise",
    )
    return ds.ArrayLikelihood(psl_models, commongp=commongp), crn_params

def run_freespec_mcmc(model, rng_seed: int = 42, mcmc_kwargs=dict()):
    logl = model.logL
    npmodel = ds_numpyro.makemodel_transformed(logl, priordict ={"log10_rho": [-9, -4]})

    sampler = numpyro.infer.MCMC(
        numpyro.infer.NUTS(npmodel),
        **mcmc_kwargs,
        progress_bar=True,
    )
    sampler.run(jax.random.key(rng_seed))
    return sampler, npmodel.to_df(sampler.get_samples())
