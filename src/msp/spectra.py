import warnings

import numpy as np


class Spectra:
    """Represent a collection of spectra sharing the same x-axis.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional shared spectral axis, such as wavelength, frequency,
        or wavenumber.
    Y : numpy.ndarray
        Two-dimensional numeric dataset with one x-axis point per row and one
        spectrum per column. A single spectrum is selected as ``Y[:, i]``.
    allow_single : bool, default=False
        Allow ``Y`` to contain only one spectrum. By default, single-spectrum
        data are rejected with a recommendation to use ``Spectrum`` instead.

    Notes
    -----
    The data are stored in ascending ``x`` order. The rows of ``Y`` are
    reordered with ``x`` to preserve the correspondence between each spectral
    point and its measurements.

    When ``allow_single=True`` is used with a one-column ``Y``, be aware that
    methods requiring multiple spectra may fail.

    Raises
    ------
    TypeError
        If ``x`` or ``Y`` cannot be converted to a numeric NumPy array.
    ValueError
        If ``x`` is not one-dimensional, ``Y`` is not two-dimensional, or a
        column of ``Y`` does not contain exactly one value for each point in
        ``x``. It is also raised when ``Y`` contains no columns, or when its
        single valid column is not explicitly permitted by
        ``allow_single=True``.
    """

    def __init__(
        self, x: np.ndarray, Y: np.ndarray, allow_single: bool = False
    ):
        try:
            x = np.asarray(x, dtype=float)
            Y = np.asarray(Y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "x and Y must be convertible to numeric NumPy arrays."
            ) from exc

        if x.ndim != 1:
            raise ValueError("x must be one-dimensional.")

        if Y.ndim != 2:
            raise ValueError("Y must be two-dimensional.")

        if x.shape[0] != Y.shape[0]:
            raise ValueError(
                "The number of rows in Y must match the number of points in x."
            )

        if Y.shape[1] == 0:
            raise ValueError("Y must contain at least one spectrum.")

        if Y.shape[1] == 1 and not allow_single:
            raise ValueError(
                "Y contains only one spectrum. Use Spectrum for single-spectrum "
                "data, or set allow_single=True to preserve the two-dimensional "
                "Spectra representation."
            )

        if Y.shape[1] == 1:
            warnings.warn(
                "Spectra was created with only one spectrum. Be aware that "
                "methods requiring multiple spectra may fail.",
                UserWarning,
                stacklevel=2,
            )

        order = np.argsort(x, kind="stable")
        
        self.x, self.Y = x[order], Y[order, :]
        self.setup()

    def setup(self):
        pass
