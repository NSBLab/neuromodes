"""
Module for expressing brain maps as linear combinations of orthogonal basis vectors, such as
geometric eigenmodes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
from scipy.sparse import eye, issparse, csc_matrix
from scipy.spatial.distance import cdist, pdist
from neuromodes.eigen import EigenData

if TYPE_CHECKING:
    from typing import Any, TypeAlias, Literal
    from collections.abc import Sequence
    from numpy.typing import NDArray
    from scipy.spatial.distance import _MetricCallback, _MetricKind
    from neuromodes.eigen import _CheckKind
    # from neuromodes.basis import _DecompositionKind, _IntSequenceKind, _SeqSequenceKind

    _IntSequenceKind: TypeAlias = Sequence[int] | NDArray[np.integer]
    _SeqSequenceKind: TypeAlias = Sequence[_IntSequenceKind] | NDArray[Any]
    _DecompositionKind: TypeAlias = Literal['project', 'regress']

@overload
def decompose(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: int | None = ...,
	mode_ids: None = ...,
	checks: _CheckKind = ...
) -> NDArray[np.floating]: ...

# 2. mode_counts is Sequence -> List of Arrays
@overload
def decompose(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: _IntSequenceKind,
	mode_ids: None = ...,
	checks: _CheckKind = ...
) -> list[NDArray[np.floating]]: ...

# 3. mode_ids is Sequence -> List of Arrays
@overload
def decompose(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: None = ...,
	mode_ids: _SeqSequenceKind,
	checks: _CheckKind = ...
) -> list[NDArray[np.floating]]: ...

def decompose(
    data: NDArray[np.floating],
    emodes: NDArray[np.floating],
    method: _DecompositionKind = 'project',
    *,
    mass: csc_matrix | None = None,
    mode_counts: _IntSequenceKind | int | None = None,
    mode_ids: _SeqSequenceKind | None = None,
    checks: _CheckKind | None = None,  
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
    
    if checks is None:
        if method == 'regress':
            checks = 'maps'
        else:
            checks = True
    if checks is not False: 
        ved = EigenData(emodes=emodes, mass=mass, data=data, checks=checks)
        emodes, mass, data = ved.emodes, ved.mass, ved.data

    if mass is None: 
        warn("No mass matrix provided; assuming that area at each vertex is 1")
        mass = csc_matrix(eye(emodes.shape[0], format='csc', dtype=cast(type[float], data.dtype)))

    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, emodes.shape[1])
    n_modes = [len(x) for x in mode_ids]

    # Manipulate input/output shapes
    output_shapes = [(i,) + data.shape[1:] for i in n_modes]
    beta = [np.empty(shape, dtype=data.dtype) for shape in output_shapes]
    data_reshaped = data.reshape(data.shape[0], -1) # guaranteed 2d
    
    # Handle NaNs and Infs by masking out afflicted vertices (separately for each NaN/Inf pattern)
    data_finite = np.isfinite(data_reshaped)
    if np.all(data_finite):
        masks = np.ones((data_reshaped.shape[0], 1), dtype=bool)
        mask_indices = np.zeros(data_reshaped.shape[1], dtype=int)
    elif method == 'regress':
        if checks is True or checks == 'maps':
            warn("NaN/Inf values detected in data; these will be disregarded during decomposition by"
                " masking corresponding vertices from data and emodes. This may lead to extreme "
                "values in affected areas of the reconstructed data. Consider instead interpolating"
                " missing data prior to decomposition via EigenSolver.inpaint().")
        masks, mask_indices = np.unique(data_finite, axis=1, return_inverse=True)
    else:  # method == 'project'
        raise ValueError("data contains NaNs/Infs; consider interpolating missing data prior to "
                         "decomposition via EigenSolver.inpaint() or set method='regress' to mask "
                         "out afflicted vertices during decomposition.")

    if method == 'project': 
        # Find the unique mode IDs requested, and the inverse mapping back to mode_ids
        unique_mids, inv = np.unique(np.concatenate(mode_ids), return_inverse=True)
        inv = np.split(inv, np.cumsum([len(m) for m in mode_ids[:-1]])) # back in the same list pattern as mode_ids
        
        # For each nan/inf pattern, get the beta values for all the unique modes
        beta_all = emodes[:, unique_mids].T @ (mass @ data_reshaped) 
        # beta_all = _calc_beta(
        #     data = data_reshaped, 
        #     emodes = emodes[:, unique_mids],
        #     method = method,
        #     mass = mass,
        #     mask = masks[:, 0]
        #     )
        
        # Map the unique results back to the specific mode_ids requested
        for j, idxs in enumerate(inv):
            beta[j] = beta_all[idxs, :].reshape(output_shapes[j])

    elif method == 'regress':
        # Have to loop over each set of mode indices
        for j in range(len(mode_ids)):
            beta_current = np.empty((n_modes[j], data_reshaped.shape[1]), dtype=data.dtype)
            # as well as each NaN pattern
            for i, mask in enumerate(masks.T):
                # Get indices of maps with this NaN/Inf pattern
                # Remove verts with NaNs/Inf in this group from data and emodes
                # Calculate beta coefficients for subset of data
                map_indices = np.where(mask_indices == i)[0]
                beta_current[:, map_indices] = spatial_lstsq(
                    emodes[:, mode_ids[j]],
                    data_reshaped[:, map_indices],
                    w=mass,
                    mask=mask
                )[0]
                # beta_current[:, map_indices] = _calc_beta(
                #     data = data_reshaped[:, map_indices], 
                #     emodes = emodes[:, mode_ids[j]], 
                #     method = method,
                #     mass = mass, 
                #     mask = mask
                # )
            beta[j] = beta_current.reshape(output_shapes[j])

    return beta[0] if squeeze_output else beta # convert back to array if mode_counts was None/scalar

def reconstruct(
    emodes: NDArray,
    data: NDArray | None = None,
    coefficients: list[NDArray] | NDArray | None = None,
    method: _DecompositionKind = 'project',
    mass: csc_matrix | None = None,
    mode_counts: _IntSequenceKind | int | None = None,
    mode_ids: _SeqSequenceKind | None = None,
    checks: _CheckKind | None = None
) -> NDArray[np.floating]:
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

    # This should stay here as it needs to be run in both coefficients and data modes
    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, emodes.shape[1])

    # Validate that exactly one of coefficients/data is provided & decompose if only data is provided
    if coefficients is not None and data is not None:
        raise ValueError("Exactly one of 'coefficients' or 'data' must be provided.")
    elif coefficients is not None:
        if isinstance(coefficients, np.ndarray): # equivalent to `if squeeze_output`, but keeps pyright happy
            coefficients = [coefficients]
    elif data is not None: # coefficients will never be squeezed in this case (as mode_ids is passed)
        coefficients = decompose(data, emodes, method=method, mass=mass, mode_ids=mode_ids, checks=False)
    else: # neither provided (this order keeps pyright happy)
        raise ValueError("Exactly one of 'coefficients' or 'data' must be provided.")
    n_recons = len(coefficients)

    # Main computation 
    recon_flat_shape = (emodes.shape[0], int(np.prod(coefficients[0].shape[1:])), n_recons)
    recon_flat = np.empty(recon_flat_shape, dtype=coefficients[0].dtype)
    for j, mids in enumerate(mode_ids):
        recon_flat[:, :, j] = emodes[:, mids] @ coefficients[j].reshape(len(mids), -1) # convert to col vec if 1D

    # Reshape outputs
    if squeeze_output: 
        recon_output_shape = (emodes.shape[0],) + coefficients[0].shape[1:] 
    else: 
        recon_output_shape = (emodes.shape[0],) + coefficients[0].shape[1:] + (n_recons,)
    recon = recon_flat.reshape(recon_output_shape)

    return recon

def reconstruction_error(
    data: NDArray,
    recon: NDArray,
    mass: csc_matrix | None = None,
    metric: _MetricCallback | _MetricKind = 'correlation',
    checks: _CheckKind = 'maps',
    **cdist_kwargs
) -> NDArray[np.floating]:
    # Format / validate checks
    if checks is not False: 
        ved = EigenData(emodes=None, mass=mass, data=data, checks=checks)
        mass, data = ved.mass, ved.data
        ved = EigenData(emodes=None, mass=None, data=recon, checks=checks)
        recon = ved.data

    # Get and check data/recon shapes
    data_shape = data.shape
    recon_shape = recon.shape
    squeeze_output = len(data_shape) == len(recon_shape)
    if squeeze_output:
        if data_shape != recon_shape:
            raise ValueError(f"data and recon must have the same shape; got {data_shape} and {recon_shape}.")
        n_recons = 1
    else: 
        if data_shape != recon_shape[:-1]:
            raise ValueError(f"data and recon must have the same shape except for the last dimension; got {data_shape} and {recon_shape}.")
        n_recons = recon_shape[-1]

    w = _process_vertex_areas(mass, data.shape[0]) 
    # w = mass.diagonal() if mass is not None else np.ones(data.shape[0])

    # Main computation
    data_2d = data.reshape(data.shape[0], -1)
    recon_3d = recon.reshape(recon.shape[0], -1, n_recons)

    error_flat_shape = (data_2d.shape[1],) + (n_recons,)
    recon_error_flat = np.empty(error_flat_shape, dtype=data.dtype)
    for i in range(data_2d.shape[1]):
        recon_error_flat[i, :] = spatial_cdist(data_2d[:, [i]], recon_3d[:, i, :],
                                        w=w, metric=metric, **cdist_kwargs)
    
    recon_error = recon_error_flat.reshape(recon.shape[1:])
    return recon_error

def spatial_lstsq(a, b, w, mask=None, rcond=None):
    if mask is None: 
        mask = np.ones(a.shape[0], dtype=bool) 
    w = _process_vertex_areas(w, a.shape[0])
    w = np.sqrt(w[mask, np.newaxis])
    return np.linalg.lstsq(w * a[mask], w * b[mask], rcond=rcond)

def spatial_pdist(X, w, mask=None, **kwargs):
    if mask is None: 
        mask = np.ones(X.shape[0], dtype=bool) 
    return pdist(X[mask].T, w=_process_vertex_areas(w, X.shape[0], mask=mask), **kwargs)

def spatial_cdist(XA, XB, w, mask=None, **kwargs): 
    if mask is None: 
        mask = np.ones(XA.shape[0], dtype=bool) 
    return cdist(XA[mask].T, XB[mask].T, w=_process_vertex_areas(w, XA.shape[0], mask=mask), **kwargs)

def _process_vertex_areas(w, n_verts, mask=None):
    if w is None: 
        warn("Mass matrix not provided; assuming that area at each vertex is 1")
        output = np.ones(n_verts)
    elif issparse(w):
        output = w.diagonal()
    elif isinstance(w, np.ndarray):
        output = np.squeeze(w)
        if w.ndim == 2:
            output = np.diag(w)
    else: 
        raise ValueError("Mass matrix must be a sparse matrix or a numpy array.")
    
    # TODO : consider adding support for using mask to do unmasking! 
    # if nnz(mask) = n_verts: (output_unmasked[mask] = output)
    output = np.asanyarray(output).reshape((n_verts,))
    if mask is not None:
        assert (len(mask) == n_verts) and (mask.ndim == 1) and (mask.dtype == bool)
        output = output[mask]
    
    return output

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
