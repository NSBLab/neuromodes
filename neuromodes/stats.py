"""
Mass (or simply area)-weighted adaptations of common statistical functions for spatial maps.
Conventional functions are equivalent to setting mass to identity, representing a mesh where each
vertex has Voronoi area/volume of 1.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, TYPE_CHECKING
from warnings import warn
from scipy.spatial.distance import squareform, cdist
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, diags
from scipy.stats import norm
from neuromodes.eigen import EigenData

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.spatial.distance import _MetricCallback, _MetricKind

def gramw(
    data: NDArray[np.floating],
    data_b: NDArray[np.floating] | None = None,
    *,
    mass: spmatrix | NDArray[np.floating] | None
) -> NDArray[np.floating]:
    """
    Dot product between all brain maps (pairwise), equivalent to ``data.T @ (mass @ data_b)``.

    Parameters
    ----------
    data : array-like
        The first set of spatial maps, of shape ``(n_verts, n_maps)``.
    data_b : array-like, optional
        The second set of spatial maps, of shape ``(n_verts, n_maps_b)``. If not provided, the
        function computes the Gram matrix of ``data`` with itself. Default is ``None``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.

    Returns
    -------
    np.ndarray
        The Gram matrix of shape ``(n_maps, n_maps_b)`` if ``data_b`` is provided, or
        ``(n_maps, n_maps)`` if not.
    """
    ved = EigenData(data=(data, data_b), mass=mass)
    a, b = ved.data
    mass = _process_vertex_areas(ved.mass, a.shape[0])

    if b is None:
        b = a
    return a.T @ (mass @ b)

# TODO: ensure that all functions support nD input, not just 1D/2D
def dotw(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    keepdims: bool = False
) -> NDArray[np.floating]:
    """
    Dot product between corresponding brain maps (not pairwise), equivalent to ``sum(data_a * (mass
    @ data_b), axis=0)``.

    Parameters
    ----------
    data_a : array-like
        The first set of spatial maps, of shape ``(n_verts, n_maps)``.
    data_b : array-like
        The second set of spatial maps, of shape ``(n_verts, n_maps)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    keepdims : bool, optional
        If True, the output will have the same number of dimensions as the input arrays. Default is
        False.

    Returns
    -------
    np.ndarray
        The dot product of shape ``(n_maps,)`` if ``keepdims=False``, or ``(1, n_maps)`` if
        ``keepdims=True``.

    Raises
    ------
    ValueError
        If ``data_a`` and ``data_b`` do not have the same number of maps (columns).
    """
    ved = EigenData(data=(data_a, data_b), mass=mass)
    a, b = ved.data

    if a.shape != b.shape:
        raise ValueError(f"data_a and data_b must have the same shape; got {a.shape} and "
                         f"{b.shape}.")

    n_verts = a.shape[0]
    mass = _process_vertex_areas(ved.mass, n_verts)
    a_2d = a.reshape(n_verts, -1)
    b_2d = b.reshape(n_verts, -1)
    out = np.sum(a_2d * (mass @ b_2d), axis=0).reshape(a.shape[1:])
    return np.expand_dims(out, axis=0) if keepdims else out

def ssqw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    keepdims: bool = False
) -> NDArray[np.floating]:
    """
    Sums of squares of each brain map, equivalent to ``sum(data * (mass @ data), axis=0)``.

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    keepdims : bool, optional
        If True, the output will have the same number of dimensions as the input array. Default is
        False.

    Returns
    -------
    np.ndarray
        The sums of squares of shape ``(n_maps,)`` if ``keepdims=False``, or ``(1, n_maps)`` if
        ``keepdims=True``.
    """
    return dotw(data, data, mass=mass, keepdims=keepdims)

def meanw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    keepdims: bool = False
) -> float:
    """
    Area-weighted mean of each brain map, equivalent to ``sum(mass @ data) / mass.sum()``.

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    keepdims : bool, optional
        If True, the output will have the same number of dimensions as the input array. Default is
        False.

    Returns
    -------
    np.ndarray
        The area-weighted mean of shape ``(n_maps,)`` if ``keepdims=False``, or ``(1, n_maps)`` if
        ``keepdims=True``.
    """
    ved = EigenData(data=data, mass=mass)  # FIXME: areas vector
    data, mass = ved.data, ved.mass

    areas = _mass_to_areas(mass, data.shape[0])
    out = np.average(data, axis=0, weights=areas)
    return out[None, :] if keepdims else out  # TODO: test for data.ndim /= 2 cases

def demeanw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None
) -> NDArray[np.floating]:
    """
    Removes the area-weighted mean from each brain map.

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.

    Returns
    -------
    np.ndarray
        The demeaned spatial maps, of shape ``(n_verts, n_maps)``.
    """
    data = np.asarray(data)
    return data - meanw(data, mass)

def varw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    keepdims: bool = False
) -> float:
    """
    Mass-weighted variance of each brain map, equivalent to ``sum((data - mean) * (mass @ (data -
    mean))) / mass.sum()``. Note that this function does not offer Bessel's correction, as mesh
    vertices are not IID samples and maps typically display spatial autocorrelation.

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    keepdims : bool, optional
        If True, the output will have the same number of dimensions as the input array. Default is
        False.

    Returns
    -------
    np.ndarray
        The area-weighted variance of shape ``(n_maps,)`` if ``keepdims=False``, or ``(1, n_maps)``
        if ``keepdims=True``.
    """
    data = np.asarray(data)
    mass = _process_vertex_areas(mass, data.shape[0])

    B = demeanw(data, mass)
    out = ssqw(B, mass) / mass.sum()
    return out[None, :] if keepdims else out  # TODO: test for data.ndim /= 2 cases

def momentw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    order: int,
    keepdims: bool = False
) -> float:
    """
    Mass-weighted central moment of a given order, of each brain map. Central moments are computed
    about the weighted mean. For order 1, the moment is always 0. For order 2, the moment is the
    variance. For higher orders, the moment is approximated using vertex areas (i.e., lumped mass). 

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    order : int
        The order of the moment to compute.
    keepdims : bool, optional
        If True, the output will have the same number of dimensions as the input array. Default is
        False.

    Returns
    -------
    np.ndarray
        The area-weighted central moment of shape ``(n_maps,)`` if ``keepdims=False``, or ``(1,
        n_maps)`` if ``keepdims=True``.
    """
    data = np.asarray(data)

    if order == 1:
        out = np.zeros(data.shape[1]) if data.ndim == 2 else 0
        return out[None, :] if keepdims else out
    elif order == 2:
        return varw(data, mass, keepdims=keepdims)
    else:
        # Approximate by lumping
        areas = _mass_to_areas(mass, data.shape[0])
        data_dm = demeanw(data, areas)
        # Sum rows of the sparse matrix to get a lumped vector, safely flattened
        return meanw(data_dm ** order, areas, keepdims=keepdims)

def stdw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    keepdims: bool = False
) -> float:
    """
    Mass-weighted standard deviation of each brain map.

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    keepdims : bool, optional
        If True, the output will have the same number of dimensions as the input array. Default is
        False.

    Returns
    -------
    np.ndarray
        The mass-weighted standard deviation of shape ``(n_maps,)`` if ``keepdims=False``, or ``(1,
        n_maps)`` if ``keepdims=True``.
    """
    return np.sqrt(varw(data, mass, keepdims=keepdims))

def zscorew(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None
) -> NDArray[np.floating]:
    """
    Mass-weighted z-scoring of each brain map.
    
    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.

    Returns
    -------
    np.ndarray
        The mass-weighted z-scored spatial maps, of shape ``(n_verts, n_maps)``.
    """
    # TODO: consider inefficiency of demeaning twice, could be expanded
    return demeanw(data, mass) / stdw(data, mass)

def covw(
    data: NDArray[np.floating],
    data_b: NDArray[np.floating] | None = None,
    *,
    mass: spmatrix | NDArray[np.floating] | None
) -> NDArray[np.floating]:
    """
    Mass-weighted covariance amongst or between brain maps.
    
    Usage:
    - ``covw(A, mass=mass)`` computes covariance of maps ``A`` amongst themselves.
    - ``covw(A, B, mass=mass)`` computes cross-covariance between maps ``A`` and maps ``B``.
    
    Note that this function does not offer Bessel's correction, as mesh vertices are not IID samples
    and maps typically display spatial autocorrelation.

    Parameters
    ----------
    data : array-like
        The first set of spatial maps, of shape ``(n_verts, n_maps)``.
    data_b : array-like, optional
        The second set of spatial maps, of shape ``(n_verts, n_maps_b)``. If not provided, the
        function computes the covariance of ``data`` with itself. Default is ``None``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.

    Returns
    -------
    np.ndarray
        The mass-weighted covariance matrix, of shape ``(n_maps, n_maps_b)`` if ``data_b`` is
        provided, or ``(n_maps, n_maps)`` if not.
    """
    a = np.asarray(data)

    mass = _process_vertex_areas(mass, a.shape[0])
    total_area = mass.sum()
    
    a_dm = demeanw(a, mass)
    b_dm = demeanw(data_b, mass) if data_b is not None else a_dm
    gram = gramw(a_dm, b_dm, mass=mass)
    
    return gram / total_area

def vecnormw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    p: int = 2,
    keepdims: bool = False
) -> NDArray[np.floating]:
    """
    Calculates the mass-weighted L^p norm of each brain map.
    
    Cases
    - ``p=2``: exact, square root of sum of squares
    - ``p=np.inf``: exact, maximum absolute value (no mass needed)
    - ``p!=2 and p!=np.inf``: approximate using vertex areas (i.e., lumped mass matrix)

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    p : int, optional
        The order of the norm. Default is 2.
    keepdims : bool, optional
        If True, the output will have the same number of dimensions as the input array. Default is
        False.

    Returns
    -------
    np.ndarray
        The mass-weighted L^p norm of each brain map.
    """
    data = np.asarray(data)
    
    if p == 2: # Exact (well-defined)
        return np.sqrt(ssqw(data, mass, keepdims=keepdims))
    
    elif p == np.inf: # Exact (ignores the mass matrix)
        out = np.max(np.abs(data), axis=0)
        return out[None, :] if (keepdims and data.ndim == 2) else out
    
    else: # Approximate by lumping
        areas = _mass_to_areas(mass, data.shape[0])
        out = np.sum(areas * (np.abs(data) ** p), axis=0) ** (1 / p)
        return out[None, :] if (keepdims and data.ndim == 2) else out

def cdistw(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    metric: _MetricCallback | _MetricKind = 'euclidean'
) -> NDArray[np.floating]:
    """
    Mass-weighted pairwise distances between two sets of brain maps. Some metrics are
    reimplementation as exact calculations, while others are approximated using vertex areas (i.e.,
    lumped mass matrix; :func:`~scipy.spatial.distance.cdist` with weights).

    Parameters
    ----------
    data_a : array-like
        The first set of spatial maps, of shape ``(n_verts, n_maps_a)``.
    data_b : array-like
        The second set of spatial maps, of shape ``(n_verts, n_maps_b)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    metric : str or callable, optional
        The distance metric to use. Can be a string recognized by ``scipy.spatial.distance.cdist``
        or a custom function. Default is ``'euclidean'``.

    Returns
    -------
    np.ndarray
        The mass-weighted distance matrix of shape ``(n_maps_a, n_maps_b)``.
    """
    # a lot of redundancy, especially for correlation. TODO: consider adding `checks`, and `checks='mass'` to EigenData
    ved = EigenData(data=(data_a, data_b), mass=mass)
    a, b = ved.data
    mass = ved.mass

    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]

    # Reimplement these to support unlumped mass matrix
    if metric == 'sqeuclidean':
        D = ssqw(a, mass)[:, None] + ssqw(b, mass)[None, :] - 2 * gramw(a, b, mass=mass)
        
    elif metric == 'euclidean':
        D = np.sqrt(cdistw(a, b, mass, 'sqeuclidean'))
        
    elif metric == 'cosine':
        Num = gramw(a, b, mass=mass)
        DenX = vecnormw(a, mass, 2)[:, None]
        DenY = vecnormw(b, mass, 2)[None, :]
        D = 1 - Num / (DenX * DenY)
        
    elif metric == 'correlation':
        D = cdistw(demeanw(a, mass), demeanw(b, mass), mass, 'cosine')
        
    else:
        areas = _mass_to_areas(mass, a.shape[0])
        D = cdist(a, b, metric=metric, mass=areas)

    return np.maximum(D, 0)

def pdistw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    metric: _MetricCallback | _MetricKind = 'euclidean'
) -> NDArray[np.floating]:
    """
    Mass-weighted pairwise distances amongst a set of brain maps, outputting a condensed vector.

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    metric : str or callable, optional
        The distance metric to use. Can be a string recognized by ``scipy.spatial.distance.cdist``
        or a custom function. Default is ``'euclidean'``.

    Returns
    -------
    np.ndarray
        The mass-weighted distance vector of shape ``(n_maps * (n_maps - 1) // 2,)``.
    """
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"data must be a 2D array with at least 2 columns; got shape {data.shape}.")

    D2 = cdistw(data, data, mass, metric)
    np.fill_diagonal(D2, 0) # Ensures exact 0 on diagonal
    return squareform(D2, checks=False)

def solvew(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None
) -> NDArray[np.floating]:
    """
    Solves the weighted least squares problem using the normal equations ``(aᵀMa)x = aᵀMb``, where
    ``M`` is the mass matrix. See https://en.wikipedia.org/wiki/Weighted_least_squares#Motivation
    for details. Consider instead using :func:`lstsqw` for a numerically stable approximation.

    Parameters
    ----------
    data_a : array-like
        The first set of spatial maps, of shape ``(n_verts, n_maps_a)``.
    data_b : array-like
        The second set of spatial maps, of shape ``(n_verts, n_maps_b)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.

    Returns
    -------
    np.ndarray
        The solution to the weighted least squares problem, of shape ``(n_maps_a, n_maps_b)``.
    """
    ved = EigenData(data=(data_a, data_b), mass=mass)
    a, b = ved.data
    mass = _process_vertex_areas(ved.mass, a.shape[0])
    # Solves (a'Wa)x = (a'Wb)
    return np.linalg.solve(a.T @ mass @ a, a.T @ mass @ b)

def lstsqw(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    rcond: float | None = None
) -> tuple[NDArray[np.floating], int, float, NDArray[np.floating]]:
    """
    Solve the weighted least squares problem using the vertex areas (i.e., lumped mass matrix),
    equivalent to the approximation ``(√(areas)a)x ≈ √(areas)b``.

    Parameters
    ----------
    data_a : array-like
        The first set of spatial maps, of shape ``(n_verts, n_maps_a)``.
    data_b : array-like
        The second set of spatial maps, of shape ``(n_verts, n_maps_b)``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    rcond : float or None, optional
        Cut-off ratio for small singular values of ``data_a``. For the purposes of rank
        determination, singular values are treated as zero if they are smaller than ``rcond`` times
        the largest singular value of ``data_a``. The default uses the machine precision times
        ``max(n_verts, n_maps_a)``. Passing -1 will use machine precision. 

    Returns
    -------
    np.ndarray
        Least-squares solution. Shape is ``(n_maps_a, n_maps_b)`` if ``data_b`` is 2D, or
        ``(n_maps_a,)`` if ``data_b`` is 1D.
    """
    ved = EigenData(data=(data_a, data_b), mass=mass)
    a, b = ved.data

    va = np.sqrt(_mass_to_areas(ved.mass, a.shape[0])) # (n_verts,)
    aw = a * va[:, np.newaxis]
    bw = b * va[:, np.newaxis] if b.ndim != 1 else b * va
    return np.linalg.lstsq(aw, bw, rcond=rcond)

def parcellate(
    data: NDArray[np.floating],
    parcellation: NDArray[np.integer],
    mass: spmatrix | NDArray[np.floating] | None,
    method: Literal['mean', 'sum', 'var', 'std'] = 'mean',
    checks: bool = True
) -> NDArray[np.floating]:
    """
    Area-weighted parcellation of each brain map.

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    parcellation : array-like
        The parcellation map, of shape ``(n_verts,)``, where each value is an integer representing
        the parcel ID for the corresponding vertex.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    method : {'mean', 'sum', 'var', 'std'}, optional
        The method for aggregating vertex values within each parcel. If 'mean', the function
        computes the area-weighted mean of each parcel. If 'sum', the function computes the
        area-weighted sum of each parcel. If 'var', the function computes the area-weighted
        variance of each parcel. If 'std', the function computes the area-weighted standard
        deviation of each parcel. Default is 'mean'.
    checks : bool, optional
        If True, the function will perform input validation and formatting. Default is True.

    Returns
    -------
    np.ndarray
        The parcellated spatial maps, of shape ``(n_parcels, n_maps)``.
    TODO consider also returning the parcellation matrix

    Raises
    ------
    ValueError
        If ``method`` is not 'mean', 'sum', 'var', or 'std'.
    ValueError
        If ``data`` is not 1D or 2D.
    ValueError
        If ``parcellation`` is not 1D.
    """
    # Format / validate arguments    
    ved = EigenData(data=(data, parcellation), mass=mass, checks=checks)
    data, parcellation = ved.data

    if data.ndim > 2: # TODO: consider supporting nD input, not just 1D/2D
        raise ValueError("data must be 1D or 2D.")
    is_data_vec = (data.ndim == 1)
    data_2d = data[:, np.newaxis] if is_data_vec else data
    
    parcellation = np.asarray(parcellation, dtype=int, copy=True)
    if parcellation.ndim != 1:
        raise ValueError("Parcellation map must be 1D.")

    # Construct parcellation matrix as sparse matrix with shape (n_parcels, n_verts)
    n_verts = data.shape[0]
    parc_mat = csr_matrix(
        (_mass_to_areas(ved.mass, n_verts),
         (parcellation, np.arange(n_verts))),
        shape=(np.max(parcellation)+1, n_verts)
        )
    if method in ['mean', 'var', 'std']: # Divide by parcel areas
        parc_mat /= parc_mat @ np.ones((n_verts,1)) # TODO consider replacing with sum?

    # Apply parcellation matrix to data (data_parc is the output)
    if method in ['mean', 'sum']: # TODO benchmark against weighted average / sum (np.bincount/nnp.average)
        data_parc = parc_mat @ data_2d
    elif method in ['var', 'std']:
        data_parc = parc_mat @ (data_2d ** 2) - (parc_mat @ data_2d) ** 2 # var = E[X^2] - (E[X])^2
        data_parc = np.maximum(data_parc, 0) # Remove numerical error
        if method == 'std': 
            data_parc = np.sqrt(data_parc)
    else: # check at the end to appease pyright
        raise ValueError(f"method must be 'mean', 'sum', 'var', or 'std'; got {method}.")

    return data_parc.squeeze(axis=1) if is_data_vec else data_parc

def calc_homogeneity(
    data: NDArray[np.floating],
    parcellation: NDArray[np.integer],
    mass: spmatrix | NDArray[np.floating] | None
) -> NDArray[np.floating]:
    """
    Calculates the parcel homogeneity of each brain map, defined as the average variance of each
    parcel weighted by the area of each parcel.

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    parcellation : array-like
        The parcellation map, of shape ``(n_verts,)``, where each value is an integer representing
        the parcel ID for the corresponding vertex.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.

    Returns
    -------
    np.ndarray
        The parcel homogeneity of shape ``(n_maps,)``.
    """
    # Format / validate arguments
    ved = EigenData(data=(data, parcellation), mass=mass)
    data, parcellation = ved.data

    # Get variance of each parcel
    var_parc = parcellate(data, parcellation, mass=ved.mass, method='var', checks=False)

    # Get area of each parcel (sum of vertex areas)
    parcel_areas = np.bincount(parcellation, weights=_mass_to_areas(ved.mass, parcellation.shape[0]))

    # Calculate weighted average of parcel variances
    homogeneity = parcellate(var_parc, np.arange(len(parcel_areas)), mass=diags(parcel_areas), method='mean', checks=False)
    # homogeneity = np.average(var_parc, axis=0, weights=parcel_areas)

    return homogeneity.squeeze(axis=0) if data.ndim == 1 else homogeneity

def resample(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    method: Literal['exact', 'mean', 'range', 'affine', 'qnorm'] = 'exact',
    mass: spmatrix | NDArray[np.floating] | None = None
) -> NDArray[np.floating]:
    """
    Resamples the values of ``data_a`` to match the distribution or summary statistics of
    ``data_b``.
    
    Parameters
    ----------
    data_a : array-like
        The set of spatial maps to be resampled, of shape ``(n_verts,)`` or ``(n_verts, ...,
        n_maps_b)``. For the latter case, each map in ``data_a[:, ..., i]`` will be resampled
        according to ``data_b[:, i]``.
    data_b : array-like
        The set of spatial maps to match against, of shape ``(n_verts,)`` or ``(n_verts,
        n_maps_b)``.
    method : {'exact', 'mean', 'range', 'affine', 'qnorm'}, optional
        The resampling method to use.
        - ``'exact'``: Rank-matches ``data_a`` to preserve the exact distribution of ``data_b``.
        - ``'mean'``: Adjusts the mean of ``data_a`` to match that of ``data_b``.
        - ``'range'``: Adjusts the range of ``data_a`` to match that of ``data_b``.
        - ``'affine'``: Adjusts both the mean and standard deviation of ``data_a`` to match those of
        ``data_b``.
        - ``'qnorm'``: Transforms ``data_a`` to a Gaussian distribution with the same mean and
        standard deviation as ``data_b`` (quantile normalization).
    mass : array-like, optional
        The mass matrix, of shape ``(n_verts, n_verts)``.

    Returns
    -------
    ndarray
        The resampled data, of shape ``(n_verts,)`` or ``(n_verts, ..., n_maps_b)``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the specified options.
    ValueError
        If ``data_b`` is 2D and its last axis length does not match that of ``data_a``.
    ValueError
        If ``data_b`` is not 1D or 2D.
    """
    # TODO: consider making data_b optional and add parameters for mean / std / min / max
    # Format / validate arguments
    ved = EigenData(data=(data_a, data_b), mass=mass)
    data_a, data_b = ved.data[0].copy(), ved.data[1].copy()  # a bit ugly

    # reshape to 3D for consistency
    data_a_shape = data_a.shape
    if data_a.ndim == 1:
        data_a = data_a[:, None, None]
    elif data_a.ndim == 2:
        data_a = data_a[:, None, :]
    elif data_a.ndim != 3:
        data_a = data_a.reshape(data_a.shape[0], -1, data_a.shape[-1])

    if data_b.ndim == 1:
        data_b = data_b[:, None, None]
    elif data_b.ndim == 2:
        if data_b.shape[-1] != data_a.shape[-1]:
            raise ValueError(
                "data_b consists of multiple maps, but the length of its last axis is not equal to "
                "that of data_a. This is required as each map in data_a[:, ..., i] will be "
                "resampled according to data_b[:, i]."
                )
        data_b = data_b[:, None, :]
    else:
        raise ValueError("data_b must be 1D or 2D.")

    # Resample according to method
    # TODO: consider whether to support NaNs, see previous implementation at
    # https://github.com/NSBLab/neuromodes/blob/3f1b773772ff916eff66daf5664e88324863f197/neuromodes/nulls.py#L391
    match method:
        case 'mean':
            data_a = demeanw(data_a, ved.mass) + meanw(data_b, ved.mass, keepdims=True)
        case 'affine':
            data_a = (zscorew(data_a, ved.mass) * stdw(data_b, ved.mass, keepdims=True)
                      + meanw(data_b, ved.mass, keepdims=True))
        case 'range':
            data_a_min = data_a.min(axis=0, keepdims=True)
            data_b_min = data_b.min(axis=0, keepdims=True)
            data_a_range = data_a.max(axis=0, keepdims=True) - data_a_min
            data_b_range = (data_b.max(axis=0, keepdims=True) - data_b_min)

            data_a = (data_a - data_a_min) / data_a_range * data_b_range + data_b_min
        case 'exact':
            data_b_sorted = np.sort(data_b, axis=0)
            data_a_ranks = np.argsort(np.argsort(data_a, axis=0), axis=0)
            data_a = np.take_along_axis(data_b_sorted, data_a_ranks, axis=0)
        case 'qnorm':
            data_a_ranks = np.argsort(np.argsort(data_a, axis=0), axis=0)
            data_a_ranks = (data_a_ranks + 1) / (data_a.shape[0] + 1)  # avoids 0 and 1
            data_a_stdnorm = norm.ppf(data_a_ranks)
            data_a = (zscorew(data_a_stdnorm, ved.mass) * stdw(data_b, ved.mass, keepdims=True)
                      + meanw(data_b, ved.mass, keepdims=True))
        case _:
            raise ValueError(f"Unknown method: {method}")

    return data_a.reshape(data_a_shape) if data_a_shape != data_a.shape else data_a

def sigmoid_rescale(
    data: NDArray[np.floating],
    steepness: float = 1.0,
    upper: float = 1.0,
    lower: float = 0.0,
    center: float = 0.0,
    checks: bool = True
) -> NDArray[np.floating]:
    """
    Rescales the input data to be within the range ``(lower, upper)`` by applying to each ``data``
    value ``x`` the sigmoid function ``f(x) = lower + (upper - lower) / (1 + exp(-steepness * (x -
    center)))``.

    If scaling heterogeneity maps for use in ``EigenSolver``, it is recommended to first z-score
    each map, then use ``lower=0``, ``upper=2``, and ``center=0``. In this case, ``steepness`` will
    match the alpha parameter used in previous work [1]_.

    Parameters
    ----------
    data : array-like
        The data to be rescaled, of shape ``(n_verts, ...)``. Maps are along the remaining axes and
        are rescaled independently.
    steepness : float, optional
        The steepness of the sigmoid function. Negative values will flip the function. Default is
        ``1.0``.
    upper : float, optional
        The upper bound of the rescaled data. Default is ``1.0``.
    lower : float, optional
        The lower bound of the rescaled data. Default is ``0.0``.
    center : float, optional
        The center of the sigmoid function, such that ``f(center) = (upper + lower) / 2``. Default
        is ``0.0``.


    Returns
    -------
    ndarray
        The scaled data, of shape ``(n_verts, ...)``.

    References
    ----------
    ..  [1] Barnes, V., et al. (2026). Regional heterogeneity shapes macroscopic wave dynamics of
        the human and non-human primate cortex. bioRxiv. https://doi.org/10.64898/2026.01.22.701178
    """
    # Format / validate arguments
    if checks is not False:
        data = EigenData(data=data).data
    
    return lower + (upper - lower) / (1 + np.exp(-steepness * (data - center)))

def _mass_to_areas(
    mass: spmatrix | NDArray[np.floating] | None = None,
    n_verts: int | None = None
) -> NDArray[np.floating]:
    mass = _process_vertex_areas(mass=mass, n_verts=n_verts)
    return np.asarray(mass.sum(axis=0)).ravel()

# TODO: consider adding dtype parameter to _process_vertex_areas for w=None case
# TODO: consider moving to EigenData
def _process_vertex_areas(
    mass: spmatrix | NDArray[np.floating] | None = None,
    n_verts: int | None = None
) -> csc_matrix:
    
    if mass is None and n_verts is None:
        raise ValueError("Either mass or n_verts must be provided.")
    
    elif mass is None and n_verts is not None: # appease: pyright
        warn("Mass matrix not provided; setting to identity (i.e., every vertex has area 1).")
        output = diags(np.ones(n_verts), format='csc')

    elif isinstance(mass, (np.ndarray, list)):        
        mass_arr = np.asarray(mass)
        if mass_arr.ndim == 2 and mass_arr.shape[0] == mass_arr.shape[1]:
            output = mass_arr
        elif mass_arr.ndim == 1 or mass_arr.shape[0] == 1 or mass_arr.shape[1] == 1:
            output = diags(np.ravel(mass_arr), format='csc')
        else: 
            raise ValueError(f"Mass matrix has invalid shape: {mass_arr.shape} (should be square "
                             "or vector).")

    elif isinstance(mass, spmatrix) and mass.shape is not None: # appease: pyright
        if mass.shape[0] == mass.shape[1]: 
            output = mass
        elif mass.shape[0] == 1 or mass.shape[1] == 1:
            output = diags(mass.toarray().flatten(), format='csc')
        else: 
            raise ValueError(f"Sparse mass matrix has invalid shape: {mass.shape} (should be "
                             "square or vector).")
            
    else:
        raise TypeError("mass must be a 1D array of vertex areas, a 2D mass matrix, or a sparse "
                        "matrix.")

    output = csc_matrix(output)
    
    if output.shape is None: 
        raise ValueError("Mass matrix has undefined shape.")
    elif output.shape[0] != output.shape[1]:
        raise ValueError(f"Mass matrix must be square; got shape {output.shape}.")
    if n_verts is not None and output.shape != (n_verts, n_verts):
        raise ValueError(f"Mass matrix has invalid shape: {output.shape} (should be ({n_verts}, "
                         f"{n_verts})).")

    return output