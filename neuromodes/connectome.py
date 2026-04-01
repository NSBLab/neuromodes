"""
Module for generating models of cortical structural connectomes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from neuromodes.eigen import EigenData

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray
    from neuromodes.eigen import _CheckKind

def model_connectome(
    emodes: NDArray[floating],
    evals: NDArray[floating],
    r: float = 9.53,
    k: int = 108,
    checks: _CheckKind = 'shape'
) -> NDArray[floating]:
    """
    Generate a vertex-wise structural connectivity matrix using the Geometric Eigenmode Model [1]_.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape ``(n_verts, n_modes)``, where ``n_verts`` is the number of
        vertices and ``n_modes`` is the number of eigenmodes.
    evals : array-like
        The eigenvalues array of shape ``(n_modes,)``.
    r : float, optional
        Spatial scale parameter for the Green's function, in millimeters. Default is ``9.53``.
    k : int, optional
        Number of eigenmodes to use. Default is ``108``.
    checks : bool, optional
        Whether to verify types and shapes of ``emodes`` and ``evals`` before computation. Default
        is ``True``.

    Returns
    -------
    np.ndarray
        The generated vertex-wise structural connectivity matrix.

    Raises
    ------
    ValueError
        If ``r`` is not a positive number.
    ValueError
        If ``k`` is not a positive integer in the range [1, ``n_modes``].

    Notes
    -----
    If comparing this model to empirical connectomes, consider thresholding the returned matrix to
    match the density of the empirical data.

    Prior work has treated ``r`` and ``k`` as free parameters to fit empirical data [1]_, with the
    default value reflecting an optimal fit to human diffusion MRI data. Consider adjusting these
    parameters, as their optima can vary across analyses (e.g., different surfaces, heterogeneous
    modes, parcellated connectomes, empirical data, fitting metrics, etc.).
    
    While the model can be computed using non-cortical modes, users should consider whether this is
    theoretically sensible and physiologically plausible.

    References
    ----------
    ..  [1] Normand, F., et al. (2025). Geometric constraints on the architecture of mammalian
        cortical connectomes. BioRxiv. https://doi.org/10.1101/2025.09.17.676944
    """
    # Format / validate arguments
    if checks is not False:
        ved = EigenData(emodes=emodes, evals=evals, checks=checks)
        emodes, evals = ved.emodes, ved.evals

    r = float(r)
    n_modes = emodes.shape[1]
    if r <= 0:
        raise ValueError("Parameter r must be a positive number.")
    if k != int(k) or k <= 0 or k > n_modes:
        raise ValueError(f"Parameter k must be an integer in the range [1, n_modes = {n_modes}].")

    # Compute the Geometric Eigenmode Model
    denom = 1/(1 + evals[:k] * r**2)
    gem = emodes[:, :k] @ (denom[:, np.newaxis] * np.linalg.pinv(emodes[:, :k]))

    # Replace diagonal and negative values with zero
    np.fill_diagonal(gem, 0)
    gem = np.maximum(gem, 0)

    # Symmetrise
    gem = (gem + gem.T) / 2

    # Normalise
    return gem / np.max(gem)