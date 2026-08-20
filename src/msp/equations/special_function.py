"""Special mathematical functions used by spectral equations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit


def sigmoid(
    x: ArrayLike, w0: float, a: float, b: float
) -> NDArray[np.float64]:
    """Evaluate the sigmoidal width used by the Schmid pseudo-Voigt model.

    The function approaches zero on one side and ``2 * w0`` on the other.
    When ``a=0``, it is constant and equal to ``w0``.

    Parameters
    ----------
    x : array-like
        Coordinates relative to the peak center.
    w0 : float
        Width in the symmetric limit.
    a : float
        Asymmetry parameter controlling direction and steepness.
    b : float
        Displacement of the sigmoid center relative to the peak center.
    """
    x = np.asarray(x, dtype=float)
    return 2.0 * w0 * expit(a * (x - b))
    
