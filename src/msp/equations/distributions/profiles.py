"""Area-normalized Gaussian, Lorentzian, and Voigt profiles."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import voigt_profile


def gaussian(
    x: ArrayLike, A: float, sigma: ArrayLike, xc: float
) -> NDArray[np.float64]:
    """Evaluate an area-normalized Gaussian distribution.

    Parameters
    ----------
    x : array-like
        Coordinates at which the distribution is evaluated.
    A : real
        Analytic area. Negative values represent an inverted peak.
    sigma : array-like
        Standard deviation. Values must be greater than zero.
    xc : real
        Center of the distribution.

    Returns
    -------
    numpy.ndarray
        Gaussian values evaluated at ``x``. For constant ``sigma``, the
        integral over the infinite domain is ``A``.
    """
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = (x - xc) / sigma
    return (A / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z**2)


def lorentzian(
    x: ArrayLike, A: float, w: ArrayLike, xc: float
) -> NDArray[np.float64]:
    """Evaluate an area-normalized Lorentzian distribution.

    Parameters
    ----------
    x : array-like
        Coordinates at which the distribution is evaluated.
    A : real
        Analytic area. Negative values represent an inverted peak.
    w : array-like
        Full width at half maximum (FWHM). Values must be greater than zero.
    xc : real
        Center of the distribution.

    Returns
    -------
    numpy.ndarray
        Lorentzian values evaluated at ``x``. For constant ``w``, the integral
        over the infinite domain is ``A``.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    return (2.0 * A / np.pi) * w / (4.0 * (x - xc) ** 2 + w**2)


def voigt(
    x: ArrayLike,
    A: float,
    sigma: ArrayLike,
    w: ArrayLike,
    xc: float,
) -> NDArray[np.float64]:
    """Evaluate an area-normalized Voigt distribution.

    Parameters
    ----------
    x : array-like
        Coordinates at which the distribution is evaluated.
    A : real
        Analytic area. Negative values represent an inverted peak.
    sigma : array-like
        Standard deviation of the Gaussian component.
    w : array-like
        FWHM of the Lorentzian component.
    xc : real
        Center of the distribution.

    Returns
    -------
    numpy.ndarray
        Voigt values evaluated at ``x``. For constant widths, the integral over
        the infinite domain is ``A``.
    """
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    gamma = np.asarray(w, dtype=float) / 2.0
    return A * voigt_profile(x - xc, sigma, gamma)
