"""Go from a particle distribution to a concentration field

Presently by counting (possibly weighted) particles in grid cells
"""

# ------------------------------
# Bjørn Ådlandsvik <bjorn@hi.no>
# Institute of Marine Research
# ------------------------------

from typing import Optional, Union

import numpy as np
import xarray as xr

Array = Union[list[float], np.ndarray, xr.DataArray]
Limits = tuple[int, int, int, int]


def cellcount(
    X: Array,
    Y: Array,
    W: Optional[Array] = None,
    grid_limits: Optional[Limits] = None,
) -> xr.DataArray:
    """Count the (possibly weighted) number of particles in grid cells

    Parameters
    ----------
    X, Y : 1D arrays, length = n
        Particle position in grid coordinates
    W : 1D array, length = n
        Weight of particles, default=None for unweighted
    grid_limits : 4-tuple (i0, i1, j0, j1) or None
        Limitation of grid to consider,
        Default=None gives the bounding box of the particle positions

    Returns
    -------
    C : 2D xarray.DataArray, shape = (j1-j0, i1-i0)
        Particle counts

    Note: particles outside the grid limits are silently ignored
    Integer indices are at the center of the grid cells

    """

    # Subgrid specification

    if grid_limits is None:
        i0 = int(round(np.min(X)))
        i1 = int(round(np.max(X))) + 1
        j0 = int(round(np.min(Y)))
        j1 = int(round(np.max(Y))) + 1
    elif len(grid_limits) == 4:
        i0, i1, j0, j1 = grid_limits
    else:
        raise TypeError("Illegal grid_limits")

    # Count
    x_edges = np.arange(i0 - 0.5, i1)
    y_edges = np.arange(j0 - 0.5, j1)
    if W is None:
        C = np.histogram2d(np.asarray(Y), np.asarray(X), bins=[y_edges, x_edges])
    else:
        C = np.histogram2d(
            np.asarray(Y),
            np.asarray(X),
            weights=np.asarray(W),
            bins=[y_edges, x_edges],
        )

    coords = dict(Y=np.arange(j0, j1), X=np.arange(i0, i1))
    return xr.DataArray(C[0], coords=coords, dims=coords.keys())
