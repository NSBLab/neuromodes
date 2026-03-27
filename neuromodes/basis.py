"""
Module for expressing brain maps as linear combinations of orthogonal basis vectors, such as
geometric eigenmodes.
"""

from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from warnings import warn
import numpy as np
from scipy.spatial.distance import cdist
from neuromodes.eigen import EigenData
from neuromodes.mesh import mask_laplacian

if TYPE_CHECKING:   
    from numpy import floating
    from numpy.typing import NDArray, ArrayLike
    from scipy.spatial.distance import _MetricCallback, _MetricKind 
    from scipy.sparse import csc_matrix

nan_warning = ("data contains NaNs and/or Infs; these will be disregarded during decomposition by "
               "masking corresponding vertices from data and emodes.")

def decompose(
    data: NDArray,
    emodes: NDArray[floating],
    method: str = 'project',
    mass: csc_matrix | None = None,
    mode_counts: int | List | Tuple | NDArray | None = None,
    mode_ids: List | Tuple | None = None,
    checks: bool | str = True,
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
        The beta coefficients array of shape ``(n_modes, n_maps, nq)``, obtained from the decomposition.
    
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
    if method not in ['project', 'regress']:
        raise ValueError(f"Invalid method '{method}'; must be 'project' or 'regress'.")
    
    # Skip warning for EigenSolver, where mass is always passed in
    if (method == 'regress') and (mass is not None) and (checks is True or checks == 'ortho'):  
            warn("mass is ignored when method='regress'.")
    
    if checks is not False: 
        ved = EigenData(emodes=emodes, mass=mass, data=data, checks=checks)
        emodes, mass, data = ved.emodes, ved.mass, ved.data

    # mode_counts is just shorthand for mode_ids
    # If mode_counts is provided, reformat into mode_ids
    squeeze_output = False
    if mode_ids is None and mode_counts is None:
        mode_ids = (np.arange(emodes.shape[1]),)
        squeeze_output = True
    elif mode_counts is not None: 
        if isinstance(mode_counts, int):
            mode_counts = [mode_counts]
            squeeze_output = True
        mode_ids = [np.arange(mc) for mc in mode_counts]
    if not isinstance(mode_ids, (list, tuple)):
        raise ValueError("mode_ids must be a list or tuple of arrays of mode indices.")
    n_counts = len(mode_ids)

    # Manipulate input/output shapes
    n_verts, n_modes = emodes.shape
    input_shape = data.shape                                        # (n_verts, ...)
    output_shape = (n_modes,) + input_shape[1:] + (n_counts,)       # (n_modes, ..., n_counts)
    data_reshaped = data.reshape(n_verts, -1)                       # (n_verts, n_maps_all)
    if data_reshaped.ndim == 1: data_reshaped = data_reshaped[:,np.newaxis]
    output_reshaped = np.empty((n_modes, data_reshaped.shape[1], n_counts))   # (n_modes, n_maps_all, n_counts)
    
    # TODO : only need to decompose once (with n=max modes) if using orthogonal method
    # Handle NaNs and Infs by masking out afflicted vertices (separately for each NaN/Inf pattern)
    data_finite = np.isfinite(data_reshaped)
    masks, mask_indices = np.unique(data_finite, axis=1, return_inverse=True)
    for j in range(n_counts):
        for i, mask in enumerate(masks.T):
            # Get indices of maps with this NaN/Inf pattern
            # Remove verts with NaNs/Inf in this group from data and emodes
            # Calculate beta coefficients for subset of data
            map_indices = np.where(mask_indices == i)[0]
            output_reshaped[:, map_indices, j] = _calc_beta(
                data = data_reshaped[:, map_indices], 
                emodes = emodes[:, mode_ids[j]], 
                method = method,
                mass = mass, 
                mask = mask
            )
    
    output = np.reshape(output_reshaped, output_shape)
    if squeeze_output:
        output = np.squeeze(output, axis=-1)
    return output

def reconstruct(
    data: ArrayLike,
    emodes: NDArray,
    method: str = 'project',
    mass: csc_matrix | None = None,
    mode_counts: ArrayLike | None = None,
    metric: _MetricCallback | _MetricKind | None = 'correlation',
    checks: bool | str = True,
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
    if checks is not False:
        ved = EigenData(emodes=emodes, mass=mass, checks=checks)
        emodes, mass = ved.emodes, ved.mass

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
    timeseries: NDArray,
    emodes: NDArray,
    method: str = 'project',
    mass: csc_matrix | None = None,
    mode_counts: ArrayLike | None = None,
    metric: _MetricCallback | _MetricKind | None = 'correlation',
    checks: bool | str = True,
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
    beta_sq = np.asarray_chkfinite(beta)**2
    total_power = np.sum(beta_sq, axis=0)

    return beta_sq / total_power

def calc_vec_fc(
    timeseries: NDArray
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
    mass: csc_matrix | None,
    mask: NDArray
) -> NDArray[floating]:
    """Helper function to perform decomposition after validating arguments and masking NaNs/Infs."""
    d = data[mask, :]
    e = emodes[mask, :]
    if method == 'project' and mass is not None:
        m = mask_laplacian(stiffness=None, mass=mass, mask=mask)[1]
        return e.T @ m @ d
    elif method == 'project' and mass is None:
        return e.T @ d
    elif method == 'regress':
        return np.linalg.lstsq(e, d, rcond=None)[0]
    else:
        raise ValueError(f"Invalid method '{method}'; must be 'project' or 'regress'.")