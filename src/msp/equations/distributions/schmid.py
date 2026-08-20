"""Schmid asymmetric pseudo-Voigt profile."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..special_function import sigmoid
from .pseudo_voigt import pseudo_voigt


def schmid_pseudo_voigt(
    x: ArrayLike,
    A: float,
    eta: float,
    w0: float,
    a: float,
    b: float,
    xc: float,
) -> NDArray[np.float64]:
    """Evaluate the asymmetric pseudo-Voigt profile proposed by Schmid et al.

    The model replaces the constant FWHM of a pseudo-Voigt profile with a
    sigmoidal width. ``A`` is a scale parameter and is not generally equal to
    the integrated area when ``a`` is nonzero.

    Parameters
    ----------
    x : array-like
        Coordinates at which the profile is evaluated.
    A : float
        Scale of the profile. It equals the analytic area only in the symmetric
        limit ``a=0``.
    eta : float
        Lorentzian fraction.
    w0 : float
        FWHM in the symmetric limit.
    a : float
        Asymmetry parameter. ``a=0`` recovers a symmetric pseudo-Voigt.
    b : float
        Displacement of the sigmoid center relative to ``xc``.
    xc : float
        Peak center.

    Returns
    -------
    numpy.ndarray
        Asymmetric pseudo-Voigt values evaluated at ``x``.

    References
    ----------
    Schmid, M., Steinrueck, H.-P., and Gottfried, J. M. (2014).
    "A new asymmetric Pseudo-Voigt function for more efficient fitting of XPS
    lines." Surface and Interface Analysis, 46(8), 505-511.
    https://doi.org/10.1002/sia.5521
    """
    x = np.asarray(x, dtype=float)
    delta = x - xc
    w = sigmoid(delta, w0, a, b)
    return pseudo_voigt(x, A, eta, w, xc)
