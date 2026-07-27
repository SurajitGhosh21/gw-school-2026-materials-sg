"""Gravitational waves from bubble collisions in a first-order phase transition."""
import numpy as np
from scipy.special import gamma

import ptarcade.models_utils as aux
from ptarcade.models_utils import prior

name = "pt_bubble"   # names the output directory
smbhb = False        # don't add the black hole binary background

parameters = {
    "log10_alpha": prior("Uniform", -2, 1),    # transition strength
    "log10_T_star": prior("Uniform", -4, 4),   # temperature / GeV
    "log10_H_R": prior("Uniform", -3, 0),      # bubble size / Hubble radius
    "a": prior("Uniform", 1, 3),               # low-frequency slope
    "b": prior("Uniform", 1, 3),               # high-frequency slope
    "c": prior("Uniform", 1, 3),               # sharpness of the peak
}


def S(x, a, b, c):
    """Spectral shape vs x = f / f_peak: rises as x^a, falls as x^-b."""
    return (a + b) ** c / (b * x ** (-a / c) + a * x ** (b / c)) ** c


def spectrum(f, log10_alpha, log10_T_star, log10_H_R, a, b, c):
    """Return h^2 Omega_gw at each frequency in f [Hz]."""
    alpha = 10**log10_alpha
    T_star = 10**log10_T_star
    H_R = 10**log10_H_R

    H_beta = H_R * (8 * np.pi) ** (-1 / 3)   # Hubble rate / transition rate

    # numbers from the bubble-collision simulations of 1605.01403
    delta = 0.48 / (1 + 5.3 + 5)             # efficiency of converting energy to GWs
    f_peak = 0.35 / (1 + 0.69 + 0.069)       # peak frequency at emission
    p, q, kappa = 2, 2, 1

    # relativistic degrees of freedom: at matter-radiation equality, and at emission
    g_s_eq = aux.g_s(aux.T_eq)
    g_s_star = aux.g_s(T_star)
    g_star = aux.g_rho(T_star)

    # normalise S so its peak value is 1
    n = (a + b) / c
    norm = ((b / a) ** (a / n) * (n * c / b) ** c
            * gamma(a / n) * gamma(b / n) / (n * gamma(c)))

    # redshifting of the GW energy density from emission until today
    dil = (np.pi**2 / 90 * g_star * (g_s_eq / g_s_star) ** (4 / 3)
           * aux.T_0**4 / (aux.M_pl * aux.H_0) ** 2)

    # peak frequency today, in Hz
    f_0 = (90 ** (-1 / 2) * np.pi * g_star ** (1 / 2) * (g_s_eq / g_s_star) ** (1 / 3)
           * aux.T_0 / aux.M_pl * T_star * f_peak * H_beta**-1 * aux.gev_to_hz)

    return (1 / norm * aux.h**2 * dil * delta * H_beta**q
            * (kappa * alpha / (1 + alpha)) ** p * S(f / f_0, a, b, c))
