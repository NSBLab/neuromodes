"""
Module for generating models of cortical structural connectomes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from neuromodes.eigen import EigenData

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from neuromodes.eigen import _CheckKind
    from scipy.sparse import spmatrix

def compute_gem(
    emodes: NDArray[np.floating],
    evals: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None = None,
    r: float = 9.53,
    k: int = 108,
    weighted_pinv: bool = True,
    checks: _CheckKind = 'shape'
) -> NDArray[np.floating]:
    """
    Generate a model structural connectome using the Geometric Eigenmode Model [1]_.

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
    mass : spmatrix | NDArray[np.floating] | None, optional
        The mass matrix of shape ``(n_verts, n_verts)``. If ``weighted_pinv`` is ``False``, this
        parameter is ignored. If ``None``, the identity matrix is used. Default is ``None``.
    weighted_pinv : bool, optional
        Whether to incorporate mass-weighting when computing the pseudoinverse of eigenmodes (see
        Notes). This accounts for mesh irregularities but differs from the original implementation
        [1]_. Default is ``True``.
    checks : bool, optional
        Whether to validate types and shapes of ``emodes``, ``evals``, and ``mass`` before
        computation. Default is ``True``.

    Returns
    -------
    np.ndarray
        The generated structural connectivity matrix of shape ``(n_verts, n_verts)``.

    Raises
    ------
    ValueError
        If ``r`` is not a positive number.
    ValueError
        If ``k`` is not a positive integer in the range [1, ``n_modes``].

    Notes
    -----
    If comparing this model to empirical connectomes, it is recommended to threshold the returned
    matrix to match the density of the empirical data.

    Prior work has treated ``r`` and ``k`` as free parameters to fit empirical data [1]_, with the
    default value reflecting an optimal fit to human diffusion MRI data. Consider adjusting these
    parameters, as their optima can vary across analyses (e.g., different surfaces, heterogeneous
    modes, parcellated connectomes, empirical data, fitting metrics, etc.).
    
    While the model can be computed using non-cortical modes, users should consider whether this is
    theoretically sensible and physiologically plausible.

    In Euclidean space, the pseudoinverse of a matrix A (with full column rank) is given by A⁺ =
    (AᵀA)⁻¹Aᵀ. For a linear system Ax = b, x = A⁺b minimizes the Euclidean L2 norm of the residuals,
    (b - Ax)ᵀ(b - Ax). However, since this vector is defined on a discrete manifold (e.g., a
    cortical map), the L2 norm becomes the mass-weighted inner product (b - Ax)ᵀM(b - Ax), where M
    is the mass matrix. The pseudoinverse must therefore adjust accordingly to minimize this
    mass-weighted L2 norm. Its formula can be derived from the property of the residuals vector
    being mass-orthogonal to all columns of A:

    AᵀM(b - Ax) = 0
         AᵀMb = AᵀMAx
            x = (AᵀMA)⁻¹AᵀMb
              = A⁺b
           A⁺ = (AᵀMA)⁻¹AᵀM
    
    For an orthonormal basis Ψ (e.g., eigenmodes), this simplifies to Ψ⁺ = ΨᵀM, which is used when
    ``weighted_pinv`` is ``True``.

    References
    ----------
    ..  [1] Normand, F., et al. (2025). Geometric constraints on the architecture of mammalian
        cortical connectomes. BioRxiv. https://doi.org/10.1101/2025.09.17.676944
    """
    # Format / validate arguments
    if checks is not False:
        ved = EigenData(emodes=emodes, evals=evals, mass=mass, checks=checks)
        emodes, evals, mass = ved.emodes, ved.evals, ved.mass

    r = float(r)
    n_modes = emodes.shape[1]
    if r <= 0:
        raise ValueError("Parameter r must be a positive number.")
    if k != int(k) or k <= 0 or k > n_modes:
        raise ValueError(f"Parameter k must be an integer in the range [1, n_modes = {n_modes}].")

    # Compute pseudoinverse of eigenmodes
    if weighted_pinv:
        pinv_emodes = emodes[:, :k].T if mass is None else emodes[:, :k].T @ mass
    else:
        pinv_emodes = np.linalg.pinv(emodes[:, :k])

    # Construct model
    denom = 1/(1 + evals[:k] * r**2)
    gem = emodes[:, :k] @ (denom[:, None] * pinv_emodes)

    # Replace diagonal and negative values with zero
    np.fill_diagonal(gem, 0)
    gem = np.maximum(gem, 0)

    # Symmetrise (only removes floating point error if weighted_pinv is True)
    gem = (gem + gem.T) / 2

    # Normalise
    return gem / np.max(gem)