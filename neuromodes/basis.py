"""
Module for expressing brain maps as linear combinations of orthogonal basis vectors, such as
geometric eigenmodes.
"""

from __future__ import annotations
from typing import Tuple, Literal, TYPE_CHECKING
from warnings import warn
import numpy as np
from scipy.sparse import spmatrix
from scipy.spatial.distance import cdist
from neuromodes.eigen import _validate_eigenvars

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray, ArrayLike
    from scipy.spatial.distance import _MetricCallback, _MetricKind 

nan_warning = ("data contains NaNs and/or Infs; these will be disregarded during decomposition by "
               "masking corresponding vertices from data and emodes.")

def decompose(
    data: ArrayLike,
    emodes: ArrayLike,
    method: Literal['project', 'regress'] = 'project',
    mass: spmatrix | ArrayLike | None = None,
    checks: bool = True,
) -> NDArray[floating]:
    """
    Calculate the decomposition of the given data onto a basis set.

    Parameters
    ----------
    data : array-like
        The input data array of shape ``(n_verts,)`` or ``(n_verts, n_maps)``, where ``n_verts`` is
        the number of vertices and ``n_maps`` is the number of maps.
    emodes : array-like
        The vectors array of shape ``(n_verts, n_modes)``, where ``n_modes`` is the number of basis
        vectors.
    method : str, optional
        The method used for the decomposition, either ``'project'`` to project data into a
        mass-orthonormal space or ``'regress'`` for least-squares fitting. Note that the beta values
        from ``'regress'`` tend towards those from ``'project'`` when more basis vectors are
        provided. For a non-orthonormal basis set, ``'regress'`` must be used. Default is
        ``'project'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)`` used for the decomposition when method is
        ``'project'``. If vectors are orthonormal in Euclidean space, leave as ``None``. See
        :func:`eigen.is_orthonormal_basis` for more details. Default is ``None``.
    checks : bool, optional
        Whether to verify types, shapes, and orthonormality of ``emodes`` and ``mass`` before
        decomposition. Default is ``True``.

    Returns
    -------
    numpy.ndarray
        The beta coefficients array of shape ``(n_modes, n_maps)``, obtained from the decomposition.
    
    Raises
    ------
    ValueError
        If ``data`` does not have shape ``(n_verts,)`` or ``(n_verts, n_maps)``.
    ValueError
        If ``method`` is not ``'project'`` or ``'regress'``.

    Notes
    -----
    If ``data`` contains NaNs or Infs, these will be disregarded during decomposition by masking
    corresponding vertices from ``data``, ``emodes``, and ``mass``. Note that this can lead to
    unexpected behaviour, such as extreme values in affected areas of the reconstructed data, or
    extreme beta values. This appears particularly prevalent when using the ``'regress'`` method. 
    """
    # Format / validate inputs
    if method == 'regress' and mass is not None:
        if checks:  # Skip warning for EigenSolver, where mass is always passed in
            warn("mass is ignored when method='regress'.")
    
    if checks:
        check_ortho = (method == 'project')
        emodes, _, mass = _validate_eigenvars(emodes=emodes, mass=mass, check_ortho=check_ortho)[:3]

    if method not in ['project', 'regress']:
        raise ValueError(f"Invalid method '{method}'; must be 'project' or 'regress'.")

    n_verts, n_modes = emodes.shape
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.ndim != 2 or data.shape[0] != n_verts:
        raise ValueError("data must have shape (n_verts,) or (n_verts, n_maps), where n_verts is "
                         f"the number of rows in emodes ({n_verts}).")

    data_finite = np.isfinite(data)
    if data_finite.all():
        # Standard decomposition
        return _calc_beta(data, emodes, method, mass)
    
    # Handle NaNs and Infs by masking out afflicted vertices
    warn(nan_warning)
    
    # Decompose separarely for each NaN/Inf pattern
    masks, mask_indices = np.unique(data_finite, axis=1, return_inverse=True)

    beta = np.empty((n_modes, data.shape[1]))
    for i, mask in enumerate(masks.T):
        # Get indices of maps with this NaN/Inf pattern
        map_indices = np.where(mask_indices == i)[0]

        # Remove verts with NaNs/Inf in this group from data and emodes
        data_masked = data[mask, :][:, map_indices]
        emodes_masked = emodes[mask, :]
        mass_masked = mass[mask, :][:, mask] if mass is not None else None # TODO: recalculate diagonal?

        # Calculate beta coefficients for subset of data
        beta[:, map_indices] = _calc_beta(data_masked, emodes_masked, method, mass_masked)
    
    return beta

def reconstruct(
    data: ArrayLike,
    emodes: ArrayLike,
    method: Literal['project', 'regress'] = 'project',
    mass: spmatrix | ArrayLike | None = None,
    mode_counts: ArrayLike | None = None,
    metric: _MetricCallback | _MetricKind | None = 'correlation',
    checks: bool = True,
    **cdist_kwargs
) -> Tuple[NDArray[floating], NDArray[floating], list[NDArray[floating]]]:
    """
    Calculate and score the reconstruction of the given independent data using the provided
    orthogonal vectors (e.g., geometric eigenmodes).

    Parameters
    ----------
    data : array-like
        The input data array of shape ``(n_verts,)`` or ``(n_verts, n_maps)``, where ``n_verts`` is
        the number of vertices and ``n_maps`` is the number of brain maps.
    emodes : array-like
        The vectors array of shape ``(n_verts, n_modes)``, where ``n_modes`` is the number of
        orthogonal vectors.
    method : str, optional
        The method used for the decomposition, either ``'project'`` to project data into a
        mass-orthonormal space or ``'regress'`` for least-squares fitting. Note that the beta values
        from ``'regress'`` tend towards those from ``'project'`` when more basis vectors are
        provided. For a non-orthonormal basis set, ``'regress'`` must be used. Default is
        ``'project'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)`` used for the decomposition when method is
        ``'project'``. If vectors are orthonormal in Euclidean space, leave as ``None``. See
        :func:`eigen.is_orthonormal_basis` for more details. Default is ``None``.
    mode_counts : array-like, optional
        The sequence of vectors to be used for reconstruction, of shape ``(n_recons,)``. For
        example, ``mode_counts=np.array([10,20,30])`` will run three analyses: with the first 10
        vectors, with the first 20 vectors, and with the first 30 vectors. Default is ``None``,
        which uses all vectors provided.
    metric : str, optional
        The metric used for calculating reconstruction error. Should be one of the options from
        ``scipy.spatial.distance.cdist``, or ``None`` if no scoring is required. Default is
        ``'correlation'``.
    checks : bool, optional
        Whether to verify types, shapes, and orthonormality of ``emodes`` and ``mass`` before
        reconstruction. Default is ``True``.
    **cdist_kwargs
        Additional keyword arguments to pass to ``scipy.spatial.distance.cdist``.

    Returns
    -------
    recon : numpy.ndarray
        The reconstructed data array of shape ``(n_verts, n_recons, n_maps)``, where ``n_recons`` is
        the number of different reconstructions ordered in ``mode_counts``. Each slice is the
        independent reconstruction of each map. Note that if ``mode_counts`` includes any constant
        vector (e.g., the first geometric eigenmode), the reconstructions will be constant for that
        value of ``mode_counts`` (this may also result in warnings/nans for ``recon_error``). 
    recon_error : numpy.ndarray
        The reconstruction error array of shape ``(n_recons, n_maps)``. Each value represents the
        reconstruction error of one map. If ``metric`` is None, this will be empty. 
    beta : list of numpy.ndarray
        A list of beta coefficients calculated for each vector.
    
    Raises
    ------
    ValueError
        If ``mode_counts`` is not a 1D array-like of integers within the range [1, ``n_modes``].

    Notes
    -----
    If ``data`` contains NaNs or Infs, these will be disregarded during decomposition by masking
    corresponding vertices from ``data``, ``emodes``, and ``mass``. Note that this can lead to
    unexpected behaviour, such as extreme values in affected areas of the reconstructed data, or
    extreme beta values. This appears particularly prevalent when using the ``'regress'`` method.
    """
    # Format / validate arguments
    if checks:
        check_ortho = (method == 'project')
        emodes, _, mass = _validate_eigenvars(emodes=emodes, mass=mass, check_ortho=check_ortho)[:3]

    n_verts, n_modes = emodes.shape
    data = np.asarray(data) # chkfinite in decompose
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_maps = data.shape[1]
    
    if mode_counts is None:
        mode_counts = np.arange(n_modes) + 1
    else:
        mode_counts = np.asarray(mode_counts)
        if (mode_counts.ndim != 1 or (mode_counts != mode_counts.astype(int)).any()
            or mode_counts.min() < 1 or mode_counts.max() > n_modes):
            raise ValueError("mode_counts must be a 1D array-like of integers within the range "
                             f"[1, n_modes = {n_modes}].")
    n_recons = len(mode_counts)

    # Decompose maps to get beta coefficients
    if method == 'project':
        # only need to decompose once (with n=max modes) if using orthogonal method
        tmp = decompose(data, emodes[:, :np.max(mode_counts)], mass=mass,
                        method=method, checks=checks)
        beta = [tmp[:mq, :] for mq in mode_counts]
    else:  # method == 'regress' (TODO: just add mode_counts to decompose() to clean this up?)
        data_finite = np.isfinite(data)
        if data_finite.all():
            beta = [
                decompose(data, emodes[:, :mq], mass=None, method=method, checks=False)
                for mq in mode_counts
            ]
        else:
            # Handle NaNs/Infs by masking out afflicted vertices
            warn(nan_warning)
            
            # Decompose separarely for each NaN/Inf pattern
            masks, mask_indices = np.unique(data_finite, axis=1, return_inverse=True)

            beta = [np.empty((mq, data.shape[1])) for mq in mode_counts]
            for i, mask in enumerate(masks.T):
                # Get indices of maps with this NaN/Inf pattern
                map_indices = np.where(mask_indices == i)[0]

                # Remove verts with NaNs/Inf in this group from data and emodes
                data_masked = data[mask, :][:, map_indices]
                emodes_masked = emodes[mask, :]

                # Calculate beta coefficients for subset of data
                for i, mq in enumerate(mode_counts):
                    beta[i][:, map_indices] = decompose(data_masked, emodes_masked[:, :mq], method,
                                                         mass=None, checks=False)

    # Reconstruct maps from beta coefficients
    recon = np.empty((n_verts, n_recons, n_maps))
    for i in range(n_recons):
        recon[:, i, :] = emodes[:, :mode_counts[i]] @ beta[i]

    # Score reconstructions
    recon_error = np.empty((n_recons, n_maps))
    if metric is not None:
        for i in range(n_maps):
            recons = recon[:, :, i]
            empirical = data[:, [i]]

            # Handle NaNs/Infs
            if method == 'regress' and not (mask := data_finite[:, i]).all():
                recons = recons[mask, :]
                empirical = empirical[mask, :]

            recon_error[:, i] = cdist(recons.T, empirical.T, metric=metric, **cdist_kwargs)[:, 0]

    return recon, recon_error, beta

def reconstruct_timeseries(
    timeseries: ArrayLike,
    emodes: ArrayLike,
    method: Literal['project', 'regress'] = 'project',
    mass: spmatrix | ArrayLike | None = None,
    mode_counts: ArrayLike | None = None,
    metric: _MetricCallback | _MetricKind | None = 'correlation',
    checks: bool = True,
    **cdist_kwargs
) -> Tuple[NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating],
           list[NDArray[floating]]]:
    """
    Calculate and score the reconstruction of the given timeseries data using the provided
    orthogonal vectors (e.g., geometric eigenmodes).

    Parameters
    ----------
    timeseries : array-like
        The input timeseries array of shape ``(n_verts, n_timepoints)``, where ``n_verts`` is the
        number of vertices and ``n_timepoints`` is the number of timepoints.
    emodes : array-like
        The vectors array of shape ``(n_verts, n_modes)``, where ``n_modes`` is the number of
        orthogonal vectors.
    method : str, optional
        The method used for the decomposition, either ``'project'`` to project data into a
        mass-orthonormal space or ``'regress'`` for least-squares fitting. Note that the beta values
        from ``'regress'`` tend towards those from ``'project'`` when more basis vectors are
        provided. For a non-orthonormal basis set, ``'regress'`` must be used. Default is
        ``'project'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)`` used for the decomposition when method is
        ``'project'``. If vectors are orthonormal in Euclidean space, leave as ``None``. See
        :func:`eigen.is_orthonormal_basis` for more details. Default is ``None``.
    mode_counts : array-like, optional
        The sequence of vectors to be used for reconstruction. For example, ``mode_counts =
        np.array([10,20,30])`` will run three analyses: with the first 10 vectors, with the first 20
        vectors, and with the first 30 vectors. Default is ``None``, which uses all vectors
        provided.
    metric : str, optional
        The metric used for calculating reconstruction error. Should be one of the options from
        ``scipy.spatial.distance.cdist``, or ``None`` if no scoring is required. Default is
        ``'correlation'``.
    checks : bool, optional
        Whether to verify types, shapes, and orthonormality of ``emodes`` and ``mass`` before
        reconstruction. Default is ``True``.
    **cdist_kwargs
        Additional keyword arguments to pass to ``scipy.spatial.distance.cdist``.

    Returns
    -------
    fc_recon : numpy.ndarray
        The reconstructed functional connectivity array of shape ``(n_edges, n_recons)``, where
        ``n_edges = n_verts*(n_verts-1)/2`` and ``n_recons`` is the number of different
        reconstructions ordered in ``mode_counts``. The FC matrix returned is r-to-z (arctanh)
        transformed and vectorized. Note that if ``mode_counts`` includes any constant vector (e.g.,
        the first geometric eigenmode), the reconstructions will be constant for that value of
        ``mode_counts`` (this may also result in warnings/nans for ``recon_error``). 
    fc_recon_error : numpy.ndarray
        The functional reconstruction accuracy of shape ``(n_recons,)``. If ``metric`` is ``None``,
        this will be empty.
    recon : numpy.ndarray
        The reconstructed timeseries array of shape ``(n_verts, n_recons, n_timepoints)``, where
        ``n_recons`` is the number of different reconstructions ordered in ``mode_counts``. Each
        slice is the independent reconstruction of each timepoint. Note that if ``mode_counts``
        includes any constant vector (e.g., the first geometric eigenmode), the reconstructions will
        be constant for that value of ``mode_counts`` (this may also result in warnings/nans for
        ``recon_error``).
    recon_error : numpy.ndarray
        The reconstruction error array of shape ``(n_recons, n_timepoints)``. Each value represents
        the reconstruction error at one timepoint. If ``metric`` is ``None``, this will be empty. 
    beta : list of numpy.ndarray
        A list of beta coefficients calculated for each vector.
    
    Raises
    ------
    ValueError
        If ``timeseries`` does not have shape ``(n_verts, n_timepoints)``.
    """
    # Format / validate arguments
    if checks:
        timeseries = np.asarray_chkfinite(timeseries)
    if np.ndim(timeseries) != 2:
        raise ValueError("timeseries must have shape (n_verts, n_timepoints).")
    
    # Use reconstruct to get independent reconstructions
    recon, recon_error, beta = reconstruct(
        timeseries,
        emodes, 
        method=method,
        mass=mass,
        mode_counts=mode_counts,
        metric=metric,
        checks=checks
    )

    # Calculate FC of original timeseries
    fc = calc_vec_fc(timeseries)
    n_edges = len(fc)

    # Calculate FC of reconstructed timeseries
    n_recons = recon.shape[1]
    fc_recon = np.empty((n_edges, n_recons))
    for i in range(n_recons):
        fc_recon[:, i] = calc_vec_fc(recon[:, i, :])

    # Score FC of reconstructions    
    fc_recon_error = cdist(
        fc_recon.T,
        fc[np.newaxis, :],
        metric=metric,
        **cdist_kwargs
        )[:, 0] if metric is not None else np.empty(n_recons)

    return fc_recon, fc_recon_error, recon, recon_error, beta

def calc_norm_power(
    beta: ArrayLike
) -> NDArray[floating]:
    """
    Transform beta coefficients from a decomposition into normalised power.

    Parameters
    ----------
    beta : array-like
        The beta coefficients array of shape ``(n_modes,)`` or ``(n_modes, n_maps)``, where
        ``n_modes`` is the number of orthogonal vectors and ``n_maps`` is the number of brain maps.

    Returns
    -------
    numpy.ndarray
        The normalized power array of shape ``(n_modes,)`` or ``(n_modes, n_maps)``, where each
        element represents the proportion of power contributed by the corresponding orthogonal
        vector to each brain map.
    """
    # TODO: add option to normalize by total power if all modes were used (map.T @ mass @ map)
    beta_sq = np.asarray_chkfinite(beta)**2
    total_power = np.sum(beta_sq, axis=0)

    return beta_sq / total_power

def calc_vec_fc(
    timeseries: ArrayLike
) -> NDArray[floating]:
    """
    Compute Fisher-z-transformed vectorized functional connectivity from timeseries data.
    
    Parameters
    ----------
    timeseries : array-like
        The input timeseries data of shape ``(n_verts, n_timepoints)``.

    Returns
    -------
    numpy.ndarray
        The Fisher-z-transformed vectorized functional connectivity array of shape ``(n_edges,)``,
        where ``n_edges = n_verts*(n_verts-1)/2``.
    """
    fc = np.corrcoef(timeseries)
    vec_fc = fc[np.triu_indices_from(fc, k=1)]
    return np.arctanh(vec_fc)

def _calc_beta(
    data: NDArray[floating],
    emodes: NDArray[floating],
    method: str,
    mass: spmatrix | NDArray[floating] | None,
) -> NDArray[floating]:
    """Helper function to perform decomposition after validating arguments and masking NaNs/Infs."""
    if method == 'project':
        if mass is None:
            return emodes.T @ data
        return emodes.T @ mass @ data

    # method == 'regress'
    return np.linalg.lstsq(emodes, data, rcond=None)[0]