import ptarcade.models_utils as aux
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import gammaln

# Tabulated relativistic dof, pulled out of ptarcade so we can interpolate with jnp.interp
# aux.g_s/aux.g_rho use np.interp, which chokes on jax tracers.
_T_tab = aux.gs[:, 0]
_g_s_tab = aux.gs[:, 2]
_g_rho_tab = aux.gs[:, 3]

_g_s_eq = aux.g_s(aux.T_eq) # entropic relativistic dof at equality (constant)


def S(x, a, b, c):
    """
    | Spectral shape as a function of x=f/f_peak
    """
    return (a + b)**c / (b * x**(-a/c) + a * x**(b/c))**c



def spectrum(f, df, log10_alpha, log10_T_star, log10_H_R, a, b, c):
    """
    | Returns the timing-residual cross-power spectral density integrated
    | over each frequency bin, S(f) * df, in units of s**2 -- the convention
    | discovery's makegp_fourier expects (cf. discovery.powerlaw).
    |
    | Note that f and df are supplied by discovery's fourierbasis already
    | repeated twice (sin/cos), so no np.repeat is needed here.
    |
    | Parameters:
    |   - f, df (Hz)
    |   - log10(alpha)
    |   - log10(T_star/Gev)
    |   - log10(H*R)
    |   - spectral shape parameters a,b,c
    """

    alpha = 10**log10_alpha
    T_star = 10**log10_T_star
    H_R = 10**log10_H_R

    H_beta = H_R * (8 * np.pi)**(-1/3)

    delta = 0.48 / (1 + 5.3 + 5) # velocity factor from 1605.01403
    f_peak = 0.35 / (1+ 0.69 +0.069) # peak frequency at emission (beta norm.) from 1605.01403
    p = 2 # alpha coefficient
    q = 2 # rate coefficient
    kappa = 1 # efficiency factor

    g_s_star = jnp.interp(T_star, _T_tab, _g_s_tab) # entropic relativistic dof at time of emission
    g_star = jnp.interp(T_star, _T_tab, _g_rho_tab) # relativistic dof at time of emission

    # normalization factor
    n = (a+b)/c
    norm = (
            (b/a)**(a/n)
            * (n * c / b)**c
            * jnp.exp(gammaln(a/n) + gammaln(b/n) - gammaln(c))
            / n
    )

    # dilution factor
    dil = (
            np.pi**2 / 90
            * g_star * (_g_s_eq / g_s_star)**(4/3)
            * aux.T_0**4 / (aux.M_pl * aux.H_0)**2
            )


    # peak frequency today in Hz
    f_0 = (
            90**(-1/2)
            * np.pi * g_star**(1/2) * (_g_s_eq/g_s_star)**(1/3)
            * aux.T_0 / aux.M_pl
            * T_star * f_peak * H_beta**-1
            * aux.gev_to_hz
            )

    # GW energy density as a fraction of the closure density
    omega = (
            1/norm
            * dil
            * delta
            * (H_beta)**q
            * (kappa * alpha / (1 + alpha)) ** p
            * S(f / f_0, a, b, c)
            )

    # Omega -> S(f) [s**3] -> S(f) * df [s**2]
    return aux.H_0_Hz**2 * omega / (8 * np.pi**4 * f**5) * df
