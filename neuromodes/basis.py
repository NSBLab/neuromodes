"""
Module for expressing brain maps as linear combinations of orthogonal basis vectors, such as
geometric eigenmodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload
from warnings import warn
from collections.abc import Sequence, Callable

import numpy as np
from scipy.sparse import diags, issparse

from neuromodes.stats import cdistw, lstsqw, _multiply_nd, _dot_nd
from neuromodes.validation import _process_mass_matrix, _process_mode_ids, _check_n_verts

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal, TypeAlias, TypeVar

    from numpy.typing import NDArray
    from scipy.sparse import csc_matrix
    from scipy.spatial.distance import _MetricCallback, _MetricKind

    from neuromodes.eigen import _CheckKind
    # from neuromodes.validation import _IntSequenceKind, _SeqSequenceKind
    from neuromodes.validation import ModeCountInput, ModeIdInput

    _DecompositionKind: TypeAlias = Literal['project', 'regress']
    CoeffsT = TypeVar("CoeffsT", NDArray[np.floating], list[NDArray[np.floating]])

@overload
def decompose(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: int | None = ...,
	mode_ids: None = ...,
	checks: _CheckKind | None = ...
) -> NDArray[np.floating]: ...

# 2. mode_counts is Sequence -> List of Arrays
@overload
def decompose(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: Sequence[int],
	mode_ids: None = ...,
	checks: _CheckKind | None = ...
) -> list[NDArray[np.floating]]: ...

# 3. mode_ids is Sequence -> Arrays
@overload
def decompose(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: None = ...,
	mode_ids: Sequence[int],
	checks: _CheckKind | None = ...
) -> NDArray[np.floating]: ...

# 3. mode_ids is Sequence[Sequence] -> List of Arrays
@overload
def decompose(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: None = ...,
	mode_ids: Sequence[Sequence[int]],
	checks: _CheckKind | None = ...
) -> list[NDArray[np.floating]]: ...

# TODO P3: add `integrate` option like LaPy for when data are already mass-corrected
# TODO P4? add support for data=None which is equivalent to data=speye(n_verts)
# TODO P4: add type hints for data to be csc
def decompose(
    data: NDArray[np.floating],
    emodes: NDArray[np.floating],
    method: _DecompositionKind = 'project',
    *,
    mass: csc_matrix | None = None,
    mode_counts: ModeCountInput | None = None,
    mode_ids: ModeIdInput | None = None,
    checks: _CheckKind | None = None,  
) -> NDArray[np.floating] | list[NDArray[np.floating]]:
    """
    Calculate the decomposition of the given data onto a basis set.

    Parameters
    ----------
    data : array-like
        The empirical data array of shape ``(n_verts, ...)``, where ``n_verts`` is the number of
        vertices and additional axes represent maps to be decomposed independently. E.g.,
        ``data`` could be a single map of shape ``(n_verts,)`` or multiple maps of shape
        ``(n_verts, n_maps)`` or multiple maps for different subjects of shape 
        ``(n_verts, n_maps, n_subjects)``.
    emodes : array-like
        The basis vectors array of shape ``(n_verts, n_modes)``, where ``n_modes`` is the number of
        vectors.
    method : str, optional
        The method used for the decomposition, either ``'project'`` or ``'regress'``. Note that
        ``'project'`` is faster and more accurate, while ``'regress'`` should only be used when
        ``data`` contains missing values (NaNs; see Notes). The methods are otherwise equivalent.
        Default is ``'project'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)``. If vectors are orthonormal in Euclidean
        space, leave as ``None``. See :func:`eigen.is_orthonormal_basis` for more details. Default
        is ``None``.
    mode_counts : array-like, optional
        The sequence of vectors to be used for decomposition, of shape ``(n_recons,)``. For
        example, ``mode_counts=np.array([10,20,30])`` will run three analyses: with the first 10
        vectors, with the first 20 vectors, and with the first 30 vectors. Default is ``None``,
        which uses ``n_modes``.
    mode_ids : array-like, optional
        The indices of the modes to be used for reconstruction, overriding ``mode_counts``. If
        ``None``, all modes are used. Default is ``None``.
    checks : str or bool, optional
        Whether to validate arguments prior to analysis. Default is ``None``.

    Returns
    -------
    numpy.ndarray or list
        The coefficients array of shape ``(n_modes, ...)`` or ``list of (n_modes, ...)`` 
        arrays, obtained from the decomposition.
    
    Raises
    ------
    ValueError
        If ``method`` is not ``'project'`` or ``'regress'``.
    ValueError
        If ``method`` is ``'project'`` and ``data`` contains NaNs.

    Notes
    -----
    If ``data`` contains NaNs, these will be disregarded prior to decomposition (via
    ``method='regress'``) by removing corresponding entries from ``data``, ``emodes``, and ``mass``.
    Note that extreme values may accumulate in the reconstructed data at these vertices, where data
    provides no constraint. Consider instead interpolating missing data prior to decomposition.
    """
    # Format / validate inputs
    if method not in ['project', 'regress']:
        raise ValueError(f"Invalid method '{method}'; must be 'project' or 'regress'.")
    
    # if checks is None:
    #     checks = True if method == 'project' else 'maps'
    # if checks is not False: 
    #     ved = EigenData(emodes=emodes, mass=mass, data=data, checks=checks)
    #     emodes, mass, data = ved.emodes, ved.mass, ved.data

    mass = _process_mass_matrix(mass, data.shape[0])
    _check_n_verts(data=data, emodes=emodes, mass=mass)
    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, emodes.shape[1])

    # Manipulate input/output shapes
    n_modes = [len(x) for x in mode_ids]
    output_shapes = [(i,) + data.shape[1:] for i in n_modes]
    coeffs = [np.empty(shape, dtype=data.dtype) for shape in output_shapes]

    if method == 'project': 
        # Find the unique mode IDs requested, and the inverse mapping back to mode_ids
        unique_mids, inv = np.unique(np.concatenate(mode_ids), return_inverse=True)
        inv = np.split(inv, np.cumsum([len(m) for m in mode_ids[:-1]])) # back in the same list pattern as mode_ids
        
        # Get the coeffs for all the unique modes
        coeffs_all = _multiply_nd(_dot_nd(emodes[:, unique_mids], mass), data)
        
        # Map the unique results back to the specific mode_ids requested
        for j, idxs in enumerate(inv):
            coeffs[j] = coeffs_all[idxs]

    elif method == 'regress':
        # Find all the nan patterns
        data_2d = data.reshape(data.shape[0], -1) # guaranteed 2d
        data_isnan = np.isnan(data_2d)
        if np.any(data_isnan):
            masks, mask_indices = np.unique(~data_isnan, axis=1, return_inverse=True)
        else:
            masks = np.ones((data_2d.shape[0], 1), dtype=bool)
            mask_indices = np.zeros(data_2d.shape[1], dtype=int)

        # Have to loop over each set of mode indices
        for j in range(len(mode_ids)):
            coeffs_current = np.empty((n_modes[j], data_2d.shape[1]), dtype=data.dtype)
            # as well as each NaN pattern
            for i, mask in enumerate(masks.T):
                # Get indices of maps with this NaN/Inf pattern
                # Remove verts with NaNs/Inf in this group from data and emodes
                # Calculate coefficients for subset of data
                map_indices = np.where(mask_indices == i)[0]
                coeffs_current[:, map_indices] = lstsqw(
                    emodes[mask][:, mode_ids[j]],
                    data_2d[mask][:, map_indices],
                    mass=mass[mask][:, mask]
                )[0]
            coeffs[j] = coeffs_current.reshape(output_shapes[j])

    return coeffs[0] if squeeze_output else coeffs # convert back to array if mode_counts was None/scalar

# TODO P1? add checks to ensure that coeffs and mode_ids/mode_counts are compatible 
# (namely that len(mode_ids) == len(coeffs) and coeffs[j].shape[0] == len(mode_ids[j]) for all j)

@overload
def reconstruct(
    coeffs: NDArray[np.floating],
    emodes: NDArray[np.floating],
    mode_counts: ModeCountInput | None = ...,
    mode_ids: ModeIdInput | None = ...,
) -> NDArray[np.floating]: ...

@overload
def reconstruct(
    coeffs: list[NDArray[np.floating]],
    emodes: NDArray[np.floating],
    mode_counts: ModeCountInput | None = ...,
    mode_ids: ModeIdInput | None = ...,
) -> list[NDArray[np.floating]]: ...

def reconstruct(
    coeffs: NDArray[np.floating] | list[NDArray[np.floating]],
    emodes: NDArray[np.floating],
    mode_counts: ModeCountInput | None = None,
    mode_ids: ModeIdInput | None = None,
) -> NDArray[np.floating] | list[NDArray[np.floating]]:
    """
    Calculate the reconstruction of the given independent data using the provided orthonormal
    vectors (e.g., geometric eigenmodes).

    Parameters
    ----------
    coeffs : array-like, optional
        The modal coefficients array of shape ``(n_modes, ...)`` or ``list of (n_modes, ...)``
        arrays, obtained from the decomposition. If ``None``, ``data`` must be provided. Default is
        ``None``.
    emodes : array-like
        The basis vectors array of shape ``(n_verts, n_modes)``, where ``n_modes`` is the number of
        vectors.
    mode_counts : array-like, optional
        The sequence of vectors to be used for reconstruction, of shape ``(n_recons,)``. For
        example, ``mode_counts=np.array([10,20,30])`` will run three analyses: with the first 10
        vectors, with the first 20 vectors, and with the first 30 vectors. Default is ``None``,
        which uses ``n_modes``.
    mode_ids : array-like, optional
        The indices of the modes to be used for reconstruction, overriding ``mode_counts``. If
        ``None``, all modes are used. Default is ``None``.
        Default is ``None``.
    checks : str or bool, optional
        Whether to validate arguments prior to analysis. Default is ``True``.

    Returns
    -------
    recon : numpy.ndarray
        The reconstructed data array of shape ``(n_verts, ..., n_recons)``, where ``n_recons`` is
        the number of different reconstructions ordered in ``mode_counts``. Each slice is the
        independent reconstruction of each map. Note that if ``mode_counts`` includes any constant
        vector (e.g., the first geometric eigenmode), the reconstructions will be constant for that
        value of ``mode_counts`` (this may also result in warnings/nans for :func:`calc_recon_error`). 
    
    Raises
    ------
    ValueError
        If neither or both of ``data`` and ``coeffs`` are provided.

    Notes
    -----
    If ``data`` contains NaNs, these will be disregarded prior to decomposition (via
    ``method='regress'``) by removing corresponding entries from ``data``, ``emodes``, and ``mass``.
    Note that extreme values may accumulate in the reconstructed data at these vertices, where data
    provides no constraint. Consider instead interpolating missing data prior to decomposition.
    """
    # Format / validate inputs
    # if checks is None:
    #     if method == 'regress':
    #         checks = 'maps'
    #     else:
    #         checks = True
    # if checks is not False:
    #     ved = EigenData(emodes=emodes, mass=mass, checks=checks)
    #     emodes, mass = ved.emodes, ved.mass

    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, emodes.shape[1])
    # coeffs_same_shape = all(c.shape[1:] == coeffs[0].shape[1:] for c in coeffs)
    # output_dtype = np.result_type(*[c.dtype for c in coeffs], emodes.dtype)

    if squeeze_output and isinstance(coeffs, np.ndarray): # mode_counts is None or scalar, so coeffs is a single array
        recon_nd = _multiply_nd(emodes[:, mode_ids[0]], coeffs)
    elif squeeze_output: 
        raise ValueError("If mode_counts is None or scalar, coeffs must be a single array, not a list.")
    # elif isinstance(coeffs, np.ndarray):
    #     output_dtype = np.result_type(coeffs.dtype, emodes.dtype)
    #     recon_nd = np.empty((emodes.shape[0], *coeffs.shape[1:]), dtype=output_dtype)
    #     for j, mids in enumerate(mode_ids):
    #         recon_nd[..., j] = _multiply_nd(emodes[:, mids], coeffs[...,j])
    # elif coeffs_same_shape:
    #     recon_nd = np.empty((emodes.shape[0], *coeffs[0].shape[1:], len(coeffs)), dtype=output_dtype)
    #     for j, mids in enumerate(mode_ids):
    #         recon_nd[..., j] = _multiply_nd(emodes[:, mids], coeffs[j])
    else:
        recon_nd = [_multiply_nd(emodes[:, mids], coeffs[j]) for j, mids in enumerate(mode_ids)]
    return recon_nd

# TODO P1: add to EigenSolver
# TODO P2: add support for data=None which is equivalent to data=speye(n_verts)
# TODO P3: add `integrate` option like LaPy for when data are already mass-corrected
def filter_spectrum(
        data: NDArray[np.floating], 
        emodes: NDArray[np.floating], 
        evals: NDArray[np.floating], 
        transfer_function: Callable[[NDArray[np.floating]], NDArray[np.floating] | csc_matrix] = lambda evals: np.ones_like(evals), 
        mass: csc_matrix | None = None, 
        method: _DecompositionKind = 'project',
        mode_ids: ModeIdInput | None = None, 
        mode_counts: ModeCountInput | None = None,
) -> NDArray[np.floating]:
    """
    filter should be a function that takes in the vector of eigenvalues and returns either
    (i) a vector of the same length (which will be converted to a diagonal matrix), or 
    (ii) a numpy or scipy.sparse square matrix of shape of the same length.
    Note that only the corresponding modes in mode_ids/mode_counts will be used for each iteration.

    if mode_counts is used, or if mode_ids is used to request multiple sets of modes,
    output will be of shape (n_verts, ..., n_recons), where n_recons is the number of reconstructions
    otherwise, output will be of shape (n_verts, ...), where ... is the shape of data after the first axis.
    """

    mode_ids, squeeze_output = _process_mode_ids(mode_counts, mode_ids, emodes.shape[1])

    # check n_verts and check_mass are inside decompose, so we don't need to do them here
    coeffs = decompose(data, emodes, method=method, mass=mass, mode_ids=mode_ids)

    # TODO add check_n_modes 

    filtered_evals = transfer_function(evals)
    if filtered_evals.ndim == 1:
        filtered_evals = diags(filtered_evals, dtype=evals.dtype).tocsc()
    if (filtered_evals.shape is None 
        or filtered_evals.shape[0] != filtered_evals.shape[1] 
        or filtered_evals.shape[0] != emodes.shape[1]): 
        raise ValueError(f"Transfer function output must be a square matrix of shape (n_modes, n_modes); got {filtered_evals.shape}.")
    filtered_emodes = _multiply_nd(emodes, filtered_evals)

    output = np.empty((*data.shape, len(mode_ids)), 
                      dtype=np.result_type(data, emodes, filtered_evals))

    for j, mids in enumerate(mode_ids):
        output[..., j] = _multiply_nd(filtered_emodes[:, mids], coeffs[j])

    if squeeze_output:
        output = np.squeeze(output, axis=-1)

    return output

# def _extract_submatrix(A, rows, cols) -> NDArray[np.floating] | csc_matrix:
#     if issparse(A):
#         return A.tocsr()[rows, :].tocsc()[:, cols]
#     return A[np.ix_(rows, cols)]

# TODO P5: support list input for recon (the output of reconstruct)
# TODO P5: add keepdims
# TODO P5: see if this can be rewritten with some kind of 'cdistw_diagonal' function
# which finds the diagonal of the distance matrix without calculating the full matrix
def calc_recon_error(
    data: NDArray[np.floating],
    recon: NDArray[np.floating],
    mass: csc_matrix | None = None,
    metric: _MetricCallback | _MetricKind = 'correlation',
    checks: _CheckKind = 'maps',
    **cdist_kwargs
) -> NDArray[np.floating]:
    """
    Calculate the reconstruction error between the given data and each reconstruction, using the
    specified metric.

    Parameters
    ----------
    data : array-like
        The empirical data array of shape ``(n_verts, ...)``, where ``n_verts`` is the number 
        of vertices and additional axes represent maps.
    recon : array-like
        The reconstructed data array of shape ``(n_verts, ..., n_recons)``, where ``n_recons`` 
        is the number of different reconstructions ordered in ``mode_counts`` (see 
        :func:`basis.decompose`). Each slice contains the independent reconstruction(s) of each the
        corresponding map in ``data``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)``. If vectors are orthonormal in 
        Euclidean space, leave as ``None``. See :func:`eigen.is_orthonormal_basis` for more 
        details. Default is ``None``.
    metric : str or callable, optional
        The distance metric to use for calculating reconstruction error. Can be any metric accepted
        by ``scipy.spatial.distance.cdist``, or a custom metric function. Default is
        ``'correlation'``.
    checks : str or bool, optional
        Whether to validate arguments prior to analysis. Default is ``'maps'``.
    **cdist_kwargs
        Additional keyword arguments to be passed to ``scipy.spatial.distance.cdist`` when
        calculating reconstruction error.

    Returns
    -------
    recon_error : numpy.ndarray
        The reconstruction error array of shape ``(..., n_recons)``, where ``n_recons`` is the
        number of different reconstructions ordered in ``mode_counts``. Each slice contains the
        error(s) of the corresponding map in ``data``.

    Raises
    ------
    ValueError
        If ``data`` and ``recon`` have incompatible shapes.
    """

    mass = _process_mass_matrix(mass, data.shape[0])
    _check_n_verts(data=data, recon=recon, mass=mass)

    # Get and check data/recon shapes
    data_shape = data.shape
    recon_shape = recon.shape
    squeeze_output = len(data_shape) == len(recon_shape)
    if squeeze_output:
        if data_shape != recon_shape:
            raise ValueError(f"data and recon must have the same shape; got {data_shape} and "
                             f"{recon_shape}.")
        n_recons = 1
    else: 
        if data_shape != recon_shape[:-1]:
            raise ValueError("data and recon must have the same shape except for the last "
                             f"dimension; got {data_shape} and {recon_shape}.")
        n_recons = recon_shape[-1]

    # Main computation
    data_2d = data.reshape(data.shape[0], -1)
    recon_3d = recon.reshape(recon.shape[0], -1, n_recons)

    error_2d_shape = (data_2d.shape[1],) + (n_recons,)
    recon_error_2d = np.empty(error_2d_shape, dtype=data.dtype)
    for i in range(data_2d.shape[1]):
        recon_error_2d[i, :] = cdistw(
            data_2d[:, [i]], recon_3d[:, i, :], mass=mass, metric=metric, **cdist_kwargs
        )

    recon_error = recon_error_2d.reshape(recon.shape[1:])
    return recon_error
