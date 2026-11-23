"""
Module for generating models of cortical structural connectomes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

from neuromodes.stats import _dot_nd, _multiply_nd
from neuromodes.validation import _process_mass_matrix, _check_n_verts, _check_n_modes, _process_mode_ids

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.sparse import csc_matrix

    from neuromodes.eigen import _CheckKind
    from neuromodes.validation import ModeCountInput, ModeIdInput

def compute_gem(
    emodes: NDArray[np.floating],
    evals: NDArray[np.floating],
    r: float = 9.53,
    mass: csc_matrix | None = None,
    mode_counts: ModeCountInput | None = None,
    mode_ids: ModeIdInput | None = None,
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
    k : int, optional # TODO P1 fix this to be mode_counts
        Number of eigenmodes to use. Default is ``None``. To use the optimal results as derived in 
        [1]_, set ``mode_counts=108``.
    checks : bool, optional
        Whether to validate types and shapes of ``emodes`` and ``evals`` before computation. Default
        is ``True``.

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

    References
    ----------
    ..  [1] Normand, F., et al. (2025). Geometric constraints on the architecture of mammalian
        cortical connectomes. BioRxiv. https://doi.org/10.1101/2025.09.17.676944
    """
    # Format / validate arguments
    # if checks is not False:
    #     ved = EigenData(emodes=emodes, evals=evals, checks=checks)
    #     emodes, evals = ved.emodes, ved.evals

    r = float(r)
    n_verts, n_modes = emodes.shape
    if r < 0:
        raise ValueError("Parameter r must be a non-negative number.")

    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, n_modes)

    _check_n_modes(emodes=emodes, evals=evals)
    output_dtype = np.result_type(emodes, evals)

    if weighted_pinv: 
        mass = _process_mass_matrix(mass, n_verts, dtype=output_dtype)
    elif mass is not None:
        warn("Mass matrix provided but weighted_pinv is False; mass will be ignored.", UserWarning)
    
    out = np.empty((n_verts, n_verts, len(mode_ids)), dtype=output_dtype)

    filtered_emodes = emodes / (1 + evals * r**2)
    for j, mids in enumerate(mode_ids): 
        if weighted_pinv:
            assert mass is not None
            c = _dot_nd(emodes[:, mids], mass)
        else: 
            c = np.linalg.pinv(emodes[:, mids])

        curr = _multiply_nd(filtered_emodes[:, mids], c)
        np.fill_diagonal(curr, 0)
        curr = np.maximum(curr, 0)
        curr = (curr + curr.T) / 2

        out[:, :, j] = curr / np.max(curr)

    return out.squeeze(axis=-1) if squeeze_output else out


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