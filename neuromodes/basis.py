"""
Module for expressing brain maps as linear combinations of orthogonal basis vectors.
"""

from __future__ import annotations
from typing import Union, Tuple, TYPE_CHECKING
from warnings import warn
import numpy as np
from scipy.sparse import spmatrix
from scipy.spatial.distance import cdist
from neuromodes.eigen import is_orthonormal_basis

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from scipy.spatial.distance import _MetricCallback, _MetricKind 

def decompose(
    data: ArrayLike,
    emodes: ArrayLike,
    method: str = 'project',
    mass: Union[spmatrix, ArrayLike, None] = None,
    checks: bool = True,
) -> NDArray:
    """
    Calculate the decomposition of the given data onto a basis set.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts,) or (n_verts, n_maps), where n_verts is the number
        of vertices and n_maps is the number of maps.
    emodes : array-like
        The vectors array of shape (n_verts, n_modes), where n_modes is the number of basis vectors.
    method : str, optional
        The method used for the decomposition, either `'project'` to project data into a
        mass-orthonormal space or `'regress'` for least-squares fitting. Note that the beta values
        from `'regress'` tend towards those from `'project'` when more basis vectors are provided.
        For a non-orthonormal basis set, `'regress'` must be used. Default is `'project'`.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is
        `'project'`. If vectors are orthonormal in Euclidean space, leave as `None`. See
        `eigen.is_orthonormal_basis` for more details. Default is `None`.
    checks : bool, optional
        Whether to verify shapes and orthonormality of `emodes` and `mass` before decomposition.
        Default is `True`.

    Returns
    -------
    numpy.ndarray
        The beta coefficients array of shape (n_modes, n_maps), obtained from the decomposition.
    
    Raises
    ------
    ValueError
        If `data` does not have shape (n_verts,) or (n_verts, n_maps).
    ValueError
        If `method='project'` and `emodes` columns do not form an orthonormal basis set.
    ValueError
        If `method` is not 'project' or 'regress'.
    """
    # Format / validate inputs
    data = np.asarray_chkfinite(data)

    if method not in ['project', 'regress']:
        raise ValueError(f"Invalid `method` '{method}'; must be 'project' or 'regress'.")
    if checks:
        emodes = np.asarray(emodes)  # chkfinite in is_orthonormal_basis
        if method == 'project':
            if not isinstance(mass, (spmatrix, type(None))):
                mass = np.asarray(mass)  # chkfinite in is_orthonormal_basis
            if not is_orthonormal_basis(emodes, mass):
                err_str = "in Euclidean space" if mass is None else "with the provided mass matrix"
                raise ValueError(
                    f"The columns of `emodes` do not form an orthonormal basis set {err_str}. "
                    "Either provide a suitable `mass` matrix such that `emodes.T @ mass @ emodes = "
                    "I`, use `method='regress'`, or set `checks=False`."
                )
        elif mass is not None:  # method == 'regress'
            warn("`mass` is ignored when `method='regress'`.")
    n_verts = emodes.shape[0]
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.ndim != 2 or data.shape[0] != n_verts:
        raise ValueError("`data` must have shape (n_verts,) or (n_verts, n_maps), where n_verts is "
                         f"the number of rows in `emodes` ({n_verts}).")

    # Decomposition
    if method == 'project':
        return emodes.T @ data if mass is None else emodes.T @ mass @ data
    else:  # method == 'regress'
        return np.linalg.lstsq(emodes, data)[0]

def reconstruct(
    data: ArrayLike,
    emodes: ArrayLike,
    method: str = 'project',
    mass: Union[spmatrix, ArrayLike, None] = None,
    mode_counts: Union[ArrayLike, None] = None,
    metric: Union[_MetricCallback, _MetricKind, None] = 'correlation',
    checks: bool = True,
    **cdist_kwargs
) -> Tuple[NDArray, NDArray, list[NDArray]]:
    """
    Calculate and score the reconstruction of the given independent data using the provided
    orthogonal vectors (e.g., geometric eigenmodes).

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts,) or (n_verts, n_maps), where n_verts is the number
        of vertices and n_maps is the number of brain maps.
    emodes : array-like
        The vectors array of shape (n_verts, n_modes), where n_modes is the number of orthogonal
        vectors.
    method : str, optional
        The method used for the decomposition, either `'project'` to project data into a
        mass-orthonormal space or `'regress'` for least-squares fitting. Note that the beta values
        from `'regress'` tend towards those from `'project'` when more basis vectors are provided.
        For a non-orthonormal basis set, `'regress'` must be used. Default is `'project'`.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is
        `'project'`. If vectors are orthonormal in Euclidean space, leave as `None`. See
        `eigen.is_orthonormal_basis` for more details. Default is `None`.
    mode_counts : array-like, optional
        The sequence of vectors to be used for reconstruction. For example,
        `mode_counts=np.array([10,20,30])` will run three analyses: with the first 10 vectors, with
        the first 20 vectors, and with the first 30 vectors. Default is `None`, which uses all
        vectors provided.
    metric : str, optional
        The metric used for calculating reconstruction error. Should be one of the options from
        `scipy.spatial.distance.cdist`, or `None` if no scoring is required. Default is
        `'correlation'`.
    checks : bool, optional
        Whether to verify shapes and orthonormality of `emodes` and `mass` before reconstruction.
        Default is `True`.
    **cdist_kwargs
        Additional keyword arguments to pass to `scipy.spatial.distance.cdist`.

    Returns
    -------
    recon : numpy.ndarray
        The reconstructed data array of shape (n_verts, nq, n_maps), where nq is the number of
        different reconstructions ordered in `mode_counts`. Each slice is the independent
        reconstruction of each map. Note that if `mode_counts` includes any constant vector (e.g.,
        the first geometric eigenmode), the reconstructions will be constant for that value of
        `mode_counts` (this may also result in warnings/nans for `recon_error`). 
    recon_error : numpy.ndarray
        The reconstruction error array of shape (nq, n_maps). Each value represents the
        reconstruction error of one map. If `metric` is None, this will be empty. 
    beta : list of numpy.ndarray
        A list of beta coefficients calculated for each vector.
    
    Raises
    ------
    ValueError
        If `mode_counts` is not a 1D array of integers within the range [1, n_modes].
    """
    # Format / validate arguments
    data = np.asarray(data) # chkfinite in decompose
    if data.ndim == 1:
        data = data[:, np.newaxis]
    emodes = np.asarray(emodes) # chkfinite in decompose
    if mode_counts is None:
        mode_counts = np.arange(emodes.shape[1]) + 1
    else:
        mode_counts = np.asarray(mode_counts)
        if (mode_counts.ndim != 1 or not np.issubdtype(mode_counts.dtype, np.integer)
            or mode_counts.min() < 1 or mode_counts.max() > emodes.shape[1]):
            raise ValueError("`mode_counts` must be a 1D array-like of integers within the range "
                             f"[1, {emodes.shape[1]}].")

    # Decompose the data to get beta coefficients
    if method == 'project':
        # only need to decompose once (with n=max modes) if using orthogonal method
        tmp = decompose(data, emodes[:, :np.max(mode_counts)], mass=mass,
                        method=method, checks=checks)
        beta = [tmp[:mq, :] for mq in mode_counts]
    else:
        beta = [
            decompose(data, emodes[:, :mq], mass=None, method=method, checks=checks)
            for mq in mode_counts
        ]

    # Reconstruct and calculate error
    recon = np.stack([emodes[:, :mode_counts[i]] @ beta[i] for i in range(len(beta))], axis=1)
    recon_error = np.stack([
        cdist(recon[:, :, i].T, data[:, [i]].T, metric=metric, **cdist_kwargs)
        for i in range(data.shape[1])
    ], axis=1) if metric is not None else np.empty(0)

    return recon, recon_error, beta

def reconstruct_timeseries(
    data: ArrayLike,
    emodes: ArrayLike,
    method: str = 'project',
    mass: Union[spmatrix, ArrayLike, None] = None,
    mode_counts: Union[ArrayLike, None] = None,
    metric: Union[_MetricCallback, _MetricKind, None] = 'correlation',
    checks: bool = True,
    **cdist_kwargs
) -> Tuple[NDArray, NDArray, NDArray, NDArray, list[NDArray]]:
    """
    Calculate and score the reconstruction of the given time-series data using the provided
    orthogonal vectors (e.g., geometric eigenmodes).

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_timepoints), where n_verts is the number of
        vertices and n_timepoints is the number of timepoints.
    emodes : array-like
        The vectors array of shape (n_verts, n_modes), where n_modes is the number of orthogonal
        vectors.
    method : str, optional
        The method used for the decomposition, either `'project'` to project data into a
        mass-orthonormal space or `'regress'` for least-squares fitting. Note that the beta values
        from `'regress'` tend towards those from `'project'` when more basis vectors are provided.
        For a non-orthonormal basis set, `'regress'` must be used. Default is `'project'`.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is
        `'project'`. If vectors are orthonormal in Euclidean space, leave as `None`. See
        `eigen.is_orthonormal_basis` for more details. Default is `None`.
    mode_counts : array-like, optional
        The sequence of vectors to be used for reconstruction. For example, `mode_counts =
        np.array([10,20,30])` will run three analyses: with the first 10 vectors, with the first 20
        vectors, and with the first 30 vectors. Default is `None`, which uses all vectors provided.
    metric : str, optional
        The metric used for calculating reconstruction error. Should be one of the options from
        `scipy.spatial.distance.cdist`, or `None` if no scoring is required. Default is
        `'correlation'`.
    checks : bool, optional
        Whether to verify shapes and orthonormality of `emodes` and `mass` before reconstruction.
        Default is `True`.
    **cdist_kwargs
        Additional keyword arguments to pass to `scipy.spatial.distance.cdist`.

    Returns
    -------
    fc_recon : numpy.ndarray
        The functional connectivity reconstructed data array of shape (n_edges, nq), where n_edges =
        n_verts*(n_verts-1)/2 and nq is the number of different reconstructions ordered in
        `mode_counts`. The FC matrix returned is r-to-z (arctanh) transformed and vectorized. Note 
        that if `mode_counts` includes any constant vector (e.g., the first geometric eigenmode),
        the reconstructions will be constant for that value of `mode_counts` (this may also result
        in warnings/nans for `recon_error`). 
    fc_recon_error : numpy.ndarray
        The functional reconstruction accuracy of shape (nq,). If `metric` is `None`, this will be
        empty.
    recon : numpy.ndarray
        The reconstructed data array of shape (n_verts, nq, n_timepoints), where nq is the number of
        different reconstructions ordered in `mode_counts`. Each slice is the independent
        reconstruction of each timepoint. Note that if `mode_counts` includes any constant vector
        (e.g., the first geometric eigenmode), the reconstructions will be constant for that value
        of `mode_counts` (this may also result in warnings/nans for `recon_error`).
    recon_error : numpy.ndarray
        The reconstruction error array of shape (nq, n_timepoints). Each value represents the
        reconstruction error at one timepoint. If `metric` is `None`, this will be empty. 
    beta : list of numpy.ndarray
        A list of beta coefficients calculated for each vector.
    
    Raises
    ------
    ValueError
        If `data` does not have shape (n_verts, n_timepoints).
    """
    # Format / validate arguments
    if np.ndim(data) != 2:
        raise ValueError("`data` must have shape (n_verts, n_timepoints).")
    
    # Use reconstruct to get independent reconstructions
    recon, recon_error, beta = reconstruct(
        data,
        emodes, 
        method=method,
        mass=mass,
        mode_counts=mode_counts,
        metric=metric,
        checks=checks
    )

    fc = calc_vec_fc(data)[np.newaxis, :]
    fc_recon = np.stack([calc_vec_fc(recon[:, i, :]) for i in range(recon.shape[1])], axis=1)
    fc_recon_error = (cdist(fc_recon.T, fc, metric=metric, **cdist_kwargs)[:, 0]
                      if metric is not None else np.empty(0))

    return fc_recon, fc_recon_error, recon, recon_error, beta

def calc_norm_power(
    beta: ArrayLike
) -> NDArray:
    """
    Transform beta coefficients from a decomposition into normalised power.

    Parameters
    ----------
    beta : array-like
        The beta coefficients array of shape (n_modes,) or (n_modes, n_maps), where n_modes is the
        number of orthogonal vectors and n_maps is the number of brain maps.

    Returns
    -------
    numpy.ndarray
        The normalized power array of shape (n_modes,) or (n_modes, n_maps), where each element
        represents the proportion of power contributed by the corresponding orthogonal vector to
        each brain map.
    """
    beta_sq = np.asarray_chkfinite(beta)**2
    total_power = np.sum(beta_sq, axis=0)

    return beta_sq / total_power

def calc_vec_fc(
    timeseries: ArrayLike
) -> NDArray:
    """
    Compute Fisher-z-transformed vectorized functional connectivity from timeseries data.
    
    Parameters
    ----------
    timeseries : array-like
        The input timeseries data of shape (n_verts, n_timepoints).

    Returns
    -------
    numpy.ndarray
        The Fisher-z-transformed vectorized functional connectivity array of shape (n_edges,), where
        n_edges = n_verts*(n_verts-1)/2.
    """
    fc = np.corrcoef(timeseries)
    vec_fc = fc[np.triu_indices_from(fc, k=1)]
    return np.arctanh(vec_fc)