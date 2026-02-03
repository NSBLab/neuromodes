"""
Module for generating models of cortical structural connectomes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike

def model_connectome(
    emodes: ArrayLike,
    evals: ArrayLike,
    r: float = 9.53,
    k: int = 108,
    checks: bool = True
) -> NDArray:
    """
    Generate a vertex-wise structural connectivity matrix using the Green's function approach
    described in Normand et al., 2025.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_verts is the number of vertices 
        and n_modes is the number of eigenmodes.
    evals : array-like
        The eigenvalues array of shape (n_modes,).
    r : float, optional
        Spatial scale parameter for the Green's function, in millimeters. Default is `9.53`.
    k : int, optional
        Number of eigenmodes to use. Default is `108`.
    checks : bool, optional
        Whether to verify shapes of `emodes` and `evals` before computation. Default is `True`.

    Returns
    -------
    np.ndarray
        The generated vertex-wise structural connectivity matrix.

    Raises
    ------
    ValueError
        If `emodes` does not have shape (n_verts, n_modes) where n_verts > n_modes.
    ValueError
        If `evals` does not have shape (n_modes,).
    ValueError
        If `r` is not a positive number.
    ValueError
        If `k` is not a positive integer in the range [1, n_modes].

    Notes
    -----
    If comparing this model to empirical connectomes, consider thresholding the returned matrix to
    match the density of the empirical data.
    """
    # Format / validate arguments
    if checks:
        emodes = np.asarray_chkfinite(emodes)
        evals = np.asarray_chkfinite(evals)

        if emodes.ndim != 2 or emodes.shape[0] <= emodes.shape[1]:
            raise ValueError("`emodes` must have shape (n_verts, n_modes), with n_verts > n_modes.")
    n_modes = emodes.shape[1]
    if checks and evals.shape != (n_modes,):
        raise ValueError(f"`evals` must have shape (n_modes,) = {(n_modes,)}, matching the number "
                        "of columns in `emodes`.")
    if not isinstance(r, (int, float)) or r <= 0:
        raise ValueError("Parameter `r` must be a positive number.")
    if not isinstance(k, int) or k <= 0 or k > n_modes:
        raise ValueError(f"Parameter `k` must be an integer in the range [1, {n_modes}].")

    # Compute the Geometric Eigenmode Model
    denom = 1/(1 + evals[:k] * r**2)
    gem = emodes[:, :k] @ np.diag(denom) @ np.linalg.pinv(emodes[:, :k])

    # Replace diagonal and negative values with zero
    np.fill_diagonal(gem, 0)
    gem = np.maximum(gem, 0)

    # Symmetrise
    gem = (gem + gem.T) / 2

    # Normalise
    return gem / np.max(gem)