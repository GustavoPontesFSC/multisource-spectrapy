import numpy as np
import pybaselines

def linear_regression(x,y,n_edge_points):
    """Fit a linear baseline using points from both ends of a spectrum.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional spectral coordinates from the selected interval.
    y : numpy.ndarray
        One-dimensional spectral values corresponding to ``x``.
    n_edge_points : int
        Number of points selected from each end of the interval.

    Returns
    -------
    numpy.ndarray
        Linear coefficients in descending power order: slope and intercept.
    """
    regress_x, regress_y = np.zeros(2*n_edge_points), np.zeros(2*n_edge_points)
    for i in range(n_edge_points):
        regress_x[i] = x[i]
        regress_y[i] = y[i]
        regress_x[-i-1] = x[-i-1]
        regress_y[-i-1] = y[-i-1]
    coef = np.polyfit(regress_x, regress_y, deg=1)                
    return coef
