import numpy as np

def _x_range_mask(x, x_range: tuple[float,float]):
    """Return a boolean mask for a selected interval of the x-axis.

    Parameters
    ----------
    x_range : tuple of float
        Two interval boundaries. They may be provided in either order,
        independently of the ordering of ``x``. Both boundaries are
        included in the selection.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the same shape as ``x``. A value is
        ``True`` when the corresponding x-coordinate lies inside the
        requested interval.

    Raises
    ------
    ValueError
        If ``x_range`` does not contain exactly two boundaries.
    """
    if not len(x_range) == 2:
        raise ValueError('x_range must be a tuple with the wanted interval!')
    
    return (x >= np.min(x_range)) & (x <= np.max(x_range))
            
def _x_out_range_mask(x, x_range: tuple[float,float]):
    """Return a boolean mask for values outside an interval of the x-axis.

    Parameters
    ----------
    x_range : tuple of float
        Two interval boundaries. They may be provided in either order,
        independently of the ordering of ``x``. Both boundaries
        belong to the interval and are therefore excluded from the
        outside-range selection.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the same shape as ``x``. A value is
        ``True`` when the corresponding x-coordinate lies strictly below
        the lower boundary or strictly above the upper boundary.

    Raises
    ------
    ValueError
        If ``x_range`` does not contain exactly two boundaries.

    Notes
    -----
    This mask is the logical complement of ``x_range_mask`` for the same
    interval.
    """
    return ~_x_range_mask(x, x_range)
                
