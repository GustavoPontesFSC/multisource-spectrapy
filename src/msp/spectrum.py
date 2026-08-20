import numpy as np
from .core import (
    _x_range_mask,
    _x_out_range_mask,
    baseline_linear_regression
)
class Spectrum:
    """Represent a single spectrum as a pair of one-dimensional numeric axes.

    Parameters
    ----------
    x : numpy.ndarray
        Independent spectral axis, such as wavelength, frequency, or wavenumber.
    y : numpy.ndarray
        Measured values associated with ``x``, such as intensity,
        absorbance, or transmittance.
    order : {"ascending", "descending"}, default="ascending"
        Direction used to arrange ``x``. The same ordering is applied to
        ``y`` so that every coordinate-measurement pair is preserved.

    Notes
    -----
    By default, the data are stored in ascending ``x`` order. Set ``order`` to
    ``"descending"`` to store them in descending order.

    Raises
    ------
    TypeError
        If either axis cannot be converted to a numeric NumPy array.
    ValueError
        If ``x`` is not one-dimensional, the axis shapes differ, or ``order``
        is not ``"ascending"`` or ``"descending"``.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        order: str = "ascending",
    ):
        try:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "x and y must be convertible to numeric NumPy arrays."
            ) from exc

        if x.ndim != 1:
            raise ValueError("x must be one-dimensional.")

        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")
        if order not in {"ascending", "descending"}:
            raise ValueError('order must be "ascending" or "descending".')

        indices = np.argsort(
            x,
            stable=True,
            descending=order == "descending",
        )
        self.x, self.y = x[indices], y[indices]
        self.order = order
        self.setup()

    def setup(self):
        pass
    def x_range_mask(self, x_range):
        """Return a mask selecting x-values inside an inclusive interval.

        The two interval boundaries may be provided in any order, regardless
        of whether ``self.x`` is stored in ascending or descending order.
        """
        return _x_range_mask(x = self.x, x_range = x_range)
    def x_out_range_mask(self, x_range):
        """Return a mask selecting x-values strictly outside an interval.

        The two interval boundaries may be provided in any order. Values equal
        to either boundary belong to the interval and are not selected.
        """
        return _x_out_range_mask(x = self.x, x_range = x_range)
    def remove_baseline(self,
                         x_range : tuple[float,float],
                         method: str = "linear regression",
                         remove: bool = True,
                         **kwargs) -> np.ndarray:
        """Estimate and optionally remove a baseline from an x-axis interval.

        Parameters
        ----------
        x_range : tuple of float
            Boundaries of the interval used for baseline estimation and
            correction. They may be provided in either order.
        method : str, default="linear regression"
            Baseline estimation method. Currently, ``"linear regression"``
            and its alias ``"lr"`` are supported.
        remove : bool, default=True
            If ``True``, subtract the estimated baseline from ``self.y``
            inside the selected interval. If ``False``, leave the spectrum
            unchanged and only return the fitted coefficients.
        **kwargs
            Method-specific options. ``n_edge_points`` sets how many points
            from each end of the interval are used by linear regression and
            defaults to 3.

        Returns
        -------
        numpy.ndarray
            Linear baseline coefficients in descending power order: slope and
            intercept.

        Raises
        ------
        TypeError
            If ``n_edge_points`` is not an integer greater than or equal to 1.
        """
        mask = self.x_range_mask(x_range)
        if method.lower() == 'lr' or method.lower() == 'linear regression':
            if not 'n_edge_points' in kwargs:
                n_edge_points = 3
            elif type(kwargs.get('n_edge_points')) is not int or kwargs.get('n_edge_points') < 1:
                raise TypeError('The number of points must be an integer and greater than or equal to 1!')
            else:
                n_edge_points = kwargs.get("n_edge_points")
            coef = baseline_linear_regression(self.x[mask], self.y[mask], n_edge_points)
            if remove:
                Y = np.poly1d(coef)
                self.y[mask] = self.y[mask] - Y(self.x[mask])
            return coef
