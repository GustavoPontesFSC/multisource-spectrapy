"""Symmetric pseudo-Voigt profile."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .profiles import gaussian, lorentzian


def pseudo_voigt(
    x: ArrayLike,
    A: float,
    eta: float,
    w: ArrayLike,
    xc: float,
) -> NDArray[np.float64]:
    """Evaluate a Gaussian-Lorentzian pseudo-Voigt mixture.

    Parameters
    ----------
    x : array-like
        Coordinates at which the profile is evaluated.
    A : float
        Scale of the profile. For constant ``w``, it is the analytic area.
    eta : float
        Lorentzian fraction. ``eta=0`` is Gaussian and ``eta=1`` is
        Lorentzian.
    w : array-like
        Shared FWHM of the Gaussian and Lorentzian components.
    xc : float
        Center of the profile.

    Returns
    -------
    numpy.ndarray
        Pseudo-Voigt values evaluated at ``x``.
    """
    sigma = np.asarray(w) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gaussian_component = gaussian(x, 1.0, sigma, xc)
    lorentzian_component = lorentzian(x, 1.0, w, xc)
    return A * (
        (1.0 - eta) * gaussian_component + eta * lorentzian_component
    )
