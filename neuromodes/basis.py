"""
Module for expressing brain maps as linear combinations of orthogonal basis vectors, such as
geometric eigenmodes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from warnings import warn
import numpy as np
from scipy.spatial.distance import cdist
from neuromodes.eigen import EigenData
from neuromodes.mesh import mask_mass

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.spatial.distance import _MetricCallback, _MetricKind 
    from scipy.sparse import csc_matrix
    from neuromodes.eigen import _CheckKind
    from neuromodes.basis import _DecompositionKind, _IntSequenceKind, _SeqSequenceKind

nan_warning = ("data contains NaNs and/or Infs; these will be disregarded during decomposition by "
               "masking corresponding vertices from data and emodes.")

def decompose(
    data: NDArray[np.floating],
    emodes: NDArray[np.floating],
    method: _DecompositionKind = 'project',
    mass: csc_matrix | None = None,
    mode_counts: _IntSequenceKind | int | None = None,
    mode_ids: _SeqSequenceKind | None = None,
    checks: _CheckKind = None,  
) -> NDArray[np.floating] | list[NDArray[np.floating]]:
    """
    Calculate the decomposition of the given data onto a basis set.

    Parameters
    ----------
    data : array-like
        The input data array of shape ``(n_verts, ...)``, where ``n_verts`` is the number of
        vertices and ``n_maps`` is the number of maps.
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
    checks : str or bool, optional
        Whether to verify types, shapes, and orthonormality of ``emodes`` and ``mass`` before
        decomposition. Default is ``True``.

    Returns
    -------
    numpy.ndarray or list
        The beta coefficients array of shape ``(n_modes, ...)`` or ``list of (n_modes, ...)``
        arrays, obtained from the decomposition.
    
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
    
    if checks is None:
        if method == 'regress':
            checks = 'maps'
        else:
            checks = True
    if checks is not False: 
        ved = EigenData(emodes=emodes, mass=mass, data=data, checks=checks)
        emodes, mass, data = ved.emodes, ved.mass, ved.data

    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, emodes.shape[1])
    n_modes = [len(x) for x in mode_ids]

    # Manipulate input/output shapes
    output_shapes = [(i,) + data.shape[1:] for i in n_modes]
    beta = [np.empty(shape, dtype=data.dtype) for shape in output_shapes]
    data_reshaped = data.reshape(data.shape[0], -1) # guaranteed 2d
    
    # Handle NaNs and Infs by masking out afflicted vertices (separately for each NaN/Inf pattern)
    data_finite = np.isfinite(data_reshaped)
    # masks, mask_indices = np.unique(data_finite, axis=1, return_inverse=True)
    if np.all(data_finite):
        masks = np.ones((data_reshaped.shape[0], 1), dtype=bool)
        mask_indices = np.zeros(data_reshaped.shape[1], dtype=int)
    else:
        masks, mask_indices = np.unique(data_finite, axis=1, return_inverse=True)

    if method == 'project': 
        # Find the unique mode IDs requested, and the inverse mapping back to mode_ids
        unique_mids, inv = np.unique(np.concatenate(mode_ids), return_inverse=True)
        inv = np.split(inv, np.cumsum([len(m) for m in mode_ids[:-1]])) # back in the same list pattern as mode_ids
        
        # For each nan/inf pattern, get the beta values for all the unique modes
        beta_all = np.empty((len(unique_mids), data_reshaped.shape[1]))
        for i, mask in enumerate(masks.T):
            map_indices = np.where(mask_indices == i)[0]
            beta_all[:, map_indices] = _calc_beta(
                data = data_reshaped[:, map_indices], 
                emodes = emodes[:, unique_mids],
                method = method,
                mass = mass, 
                mask = mask
            )
        
        # Map the unique results back to the specific mode_ids requested
        for j, idxs in enumerate(inv):
            beta[j] = beta_all[idxs, :].reshape(output_shapes[j])

    elif method == 'regress':
        # Have to loop over each set of mode indices
        for j in range(len(mode_ids)):
            beta_current = np.empty((n_modes[j], data_reshaped.shape[1]), dtype=data.dtype)
            # as well as each NaN patter
            for i, mask in enumerate(masks.T):
                # Get indices of maps with this NaN/Inf pattern
                # Remove verts with NaNs/Inf in this group from data and emodes
                # Calculate beta coefficients for subset of data
                map_indices = np.where(mask_indices == i)[0]
                beta_current[:, map_indices] = _calc_beta(
                    data = data_reshaped[:, map_indices], 
                    emodes = emodes[:, mode_ids[j]], 
                    method = method,
                    mass = mass, 
                    mask = mask
                )
            beta[j] = beta_current.reshape(output_shapes[j])

    return beta[0] if squeeze_output else beta # convert back to array if mode_counts was None/scalar

def reconstruct(
    data: NDArray,
    emodes: NDArray,
    method: _DecompositionKind = 'project',
    mass: csc_matrix | None = None,
    mode_counts: _IntSequenceKind | int | None = None,
    mode_ids: _SeqSequenceKind | None = None,
    checks: _CheckKind = None,
    metric: _MetricCallback | _MetricKind | None = 'correlation',
    **cdist_kwargs
) -> tuple[NDArray[np.floating], NDArray[np.floating], list[NDArray[np.floating]] | NDArray[np.floating]]:
    """
    Calculate and score the reconstruction of the given independent data using the provided
    orthogonal vectors (e.g., geometric eigenmodes).

    Parameters
    ----------
    data : array-like
        The input data array of shape ``(n_verts, ...)``, where ``n_verts`` is
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
        The reconstructed data array of shape ``(n_verts, ..., n_recons)``, where ``n_recons`` is
        the number of different reconstructions ordered in ``mode_counts``. Each slice is the
        independent reconstruction of each map. Note that if ``mode_counts`` includes any constant
        vector (e.g., the first geometric eigenmode), the reconstructions will be constant for that
        value of ``mode_counts`` (this may also result in warnings/nans for ``recon_error``). 
    recon_error : numpy.ndarray
        The reconstruction error array of shape ``(..., n_recons)``. Each value represents the
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
    # Format / validate inputs
    if checks is None:
        if method == 'regress':
            checks = 'maps'
        else:
            checks = True
    if checks is not False:
        ved = EigenData(emodes=emodes, mass=mass, data=data, checks=checks)
        emodes, mass, data = ved.emodes, ved.mass, ved.data
    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, emodes.shape[1])

    # Prepare inputs and outputs in right shapes
    n_recons = len(mode_ids)
    recon_output_shape = data.shape + (n_recons,)
    error_output_shape = data.shape[1:] + (n_recons,)

    data_2d = data.reshape(data.shape[0], -1)
    recon_flat_shape = data_2d.shape + (n_recons,) # the flat data will just be np.reshaped into the size above
    error_flat_shape = (data_2d.shape[1],) + (n_recons,)

    # Main computation
    # a. Decomposition
    beta = decompose(data, emodes, method=method, mass=mass, mode_ids=mode_ids, checks=False)

    # b. Reconstructions: Need to loop over recons as betas are different sizes
    recon_flat = np.empty(recon_flat_shape, dtype=data.dtype)
    for j, mids in enumerate(mode_ids):
        recon_flat[:, :, j] = emodes[:, mids] @ beta[j].reshape(len(mids), -1) # convert to col vec if 1D

    # c. Errors: Need to loop over maps for cdist
    recon_error_flat = np.full(error_flat_shape, None, dtype=data.dtype)
    if metric is not None:
        for i in range(data_2d.shape[1]):
            recon_error_flat[i, :] = cdist(data_2d[:, [i]].T, recon_flat[:, i, :].T, 
                                           metric=metric, **cdist_kwargs)

    # Reshape outputs
    recon = recon_flat.reshape(recon_output_shape)
    recon_error = recon_error_flat.reshape(error_output_shape)

    if squeeze_output: 
        recon = np.squeeze(recon, axis=-1)
        recon_error = np.squeeze(recon_error, axis=-1)
        beta = beta[0]

    return recon, recon_error, beta

def reconstruct_timeseries(
    timeseries: NDArray,
    emodes: NDArray,
    method: _DecompositionKind = 'project',
    mass: csc_matrix | None = None,
    mode_counts: _IntSequenceKind | int | None = None,
    mode_ids: _SeqSequenceKind | None = None,
    metric: _MetricCallback | _MetricKind | None = 'correlation',
    checks: _CheckKind = None,
    **cdist_kwargs
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating],
           list[NDArray[np.floating]] | NDArray[np.floating]]:
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
    if checks is None:
        if method == 'regress':
            checks = 'maps'
        else:
            checks = True
    if checks is not False:
        ved = EigenData(emodes=emodes, mass=mass, data=timeseries, checks=checks)
        emodes, mass, timeseries = ved.emodes, ved.mass, ved.data
    if timeseries.ndim != 2:
        raise ValueError("timeseries must have shape (n_verts, n_timepoints).")

    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, emodes.shape[1])
    
    # Use reconstruct to get independent reconstructions
    recon, recon_error, beta = reconstruct(
        timeseries,
        emodes, 
        method=method,
        mass=mass,
        mode_ids=mode_ids,
        metric=metric,
        checks=False, 
        **cdist_kwargs
    )

    # Calculate FC of original timeseries
    fc = calc_vec_fc(timeseries)
    n_edges = len(fc)
    n_recons = len(beta)

    # Calculate FC of reconstructed timeseries
    fc_recon = np.full((n_edges,) + recon.shape[2:], None, dtype=timeseries.dtype)
    for i in range(n_recons):
        fc_recon[:, i] = calc_vec_fc(recon[..., i])

    # Score FC of reconstructions
    fc_recon_error = np.full(n_recons, None, dtype=timeseries.dtype)
    if metric is not None:
        fc_recon_error = cdist(
            fc_recon.T,
            fc[np.newaxis, :],
            metric=metric,
            **cdist_kwargs
            )[:, 0]
    
    if squeeze_output:
        fc_recon = np.squeeze(fc_recon, axis=-1)
        fc_recon_error = np.squeeze(fc_recon_error, axis=-1)
        recon = np.squeeze(recon, axis=-1)
        recon_error = np.squeeze(recon_error, axis=-1)
        beta = beta[0]

    return fc_recon, fc_recon_error, recon, recon_error, beta

def calc_norm_power(
    beta: NDArray
) -> NDArray[np.floating]:
    """
    Transform beta coefficients from a decomposition into normalised power.

    Parameters
    ----------
    beta : array-like
        The beta coefficients array of shape ``(n_modes,...)``, where ``n_modes`` is the number of
        orthogonal vectors and the remaining dimensions are the number of brain maps.

    Returns
    -------
    numpy.ndarray
        The normalized power array of shape ``(n_modes,...)``, where each element represents the
        proportion of power contributed by the corresponding orthogonal vector to each brain map.
    """
    # TODO: add option to normalize by total power if all modes were used (map.T @ mass @ map)
    beta_sq = np.asarray_chkfinite(beta)**2
    total_power = np.sum(beta_sq, axis=0)

    return beta_sq / total_power

def calc_vec_fc(
    timeseries: NDArray
) -> NDArray[np.floating]:
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
    data: NDArray[np.floating],
    emodes: NDArray[np.floating],
    method: str,
    mass: csc_matrix | None,
    mask: NDArray
) -> NDArray[np.floating]:
    """Helper function to perform decomposition after validating arguments and masking NaNs/Infs."""
    d = data[mask, :]
    e = emodes[mask, :]
    if method == 'project' and mass is not None:
        m = mask_mass(mass=mass, mask=mask)
        return e.T @ m @ d
    elif method == 'project' and mass is None:
        return e.T @ d
    elif method == 'regress':
        return np.linalg.lstsq(e, d, rcond=None)[0]
    else:
        raise ValueError(f"Invalid method '{method}'; must be 'project' or 'regress'.")

def _process_mode_ids(
    mode_counts: _IntSequenceKind | int | None,
    mode_ids: _SeqSequenceKind | None,
    n_modes: int
) -> tuple[_SeqSequenceKind, bool]: 
    # mode_counts is just shorthand for mode_ids
    # If mode_counts is provided, reformat into mode_ids
    if mode_counts is not None and mode_ids is not None:
        raise UserWarning("Both mode_counts and mode_ids provided; mode_counts will be ignored.")
    
    if isinstance(mode_ids, (list, tuple, np.ndarray)):
        output = mode_ids
    elif mode_ids is not None: 
        raise ValueError("mode_ids must be a list or tuple of arrays of mode indices.")
    elif mode_counts is None:
        output = (np.arange(n_modes),)
    elif isinstance(mode_counts, int):
        output = (np.arange(mode_counts),)
    else: 
        output = [np.arange(mc) for mc in mode_counts]
    
    squeeze_output = (mode_ids is None) and (mode_counts is None or isinstance(mode_counts, int))

    return output, squeeze_output
