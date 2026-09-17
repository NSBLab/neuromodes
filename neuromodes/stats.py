"""
Mass (or simply area)-weighted adaptations of common statistical functions for spatial maps.
Conventional functions are equivalent to setting mass to identity, representing a mesh where each
vertex has Voronoi area/volume of 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from warnings import warn

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags, spmatrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import splu
from scipy.spatial.distance import cdist, squareform

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
    Area-weighted mean of each brain map, equivalent to ``sum(mass @ data) / sum(mass)``.

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
    ved = EigenData(data=data, mass=mass)
    data, mass = ved.data, ved.mass

    areas = _mass_to_areas(mass, data.shape[0])
    out = np.average(data, axis=0, weights=areas)
    return out[None, :] if keepdims else out

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
    Mass-weighted variance of each brain map, equivalent to ``sum((data - meanw(data, mass)) * (mass
    @ (data - meanw(data, mass)))) / mass.sum()``. Note that this function does not offer Bessel's
    correction, as mesh vertices are not IID samples and maps typically display spatial
    autocorrelation.

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
    return out[None, :] if keepdims else out

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

def lstsqw(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    rcond: float | None = None
) -> tuple[NDArray[np.floating], int, float, NDArray[np.floating]]:
    """
    Solve the linear system Ax = b, where A and b are sets of spatial maps, by minimising the
    mass-weighted squared L₂ norm (b-Ax)ᵀ M (b-Ax) (see Notes).

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

    Notes
    -----
    For discretized spatial maps, we want to minimize the mass-weighted L₂ norm. By leveraging the
    Cholesky decomposition of the symmetric positive definite mass matrix (M = LLᵀ), we can
    transform the mass-weighted norm into a typical Euclidean norm:
    
    ||b-Ax||₂²_mass = (b-Ax)ᵀM(b-Ax)
                    = (b-Ax)ᵀLLᵀ(b-Ax)
                    = (Lᵀ(b-Ax))ᵀ(Lᵀ(b-Ax))
                    = ((Lᵀb)-(LᵀA)x)ᵀ((Lᵀb)-(LᵀA)x)
                    = ||(Lᵀb)-(LᵀA)x||₂²_Euclidean
    
    The above tells us that we can still use a standard least squares solver if we simply scale the
    data by Lᵀ. This is equivalent to using ``np.linalg.solve(a.T @ mass @ a, a.T @ mass @ b)`` but
    is more efficient and numerically stable. Note that for lumped (diagonal) mass, L = Lᵀ = √M
    (element-wise).
    """
    ved = EigenData(data=(data_a, data_b), mass=mass)
    a, b = ved.data

    # concatenate both data arrays for efficient Cholesky multiplication
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    ab = np.concatenate([a, b], axis=1)

    # multiply and unpack
    LT_ab = _mult_by_cholesky(ab, mass, transpose=True)
    LT_a = LT_ab[:, :a.shape[1]]
    LT_b = LT_ab[:, a.shape[1]:]

    # Get results from standard least squares solver
    return np.linalg.lstsq(LT_a, LT_b, rcond=rcond)

def parcellate(
    data: NDArray[np.floating],
    parcellation: NDArray[np.integer],
    mass: spmatrix | NDArray[np.floating] | None,
    method: Literal['mean', 'sum'] = 'mean'  # TODO: add 'var' (e.g., for parcel homogeneity)
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
    method : {'mean', 'sum'}, optional
        The method for aggregating vertex values within each parcel. If 'mean', the function
        computes the area-weighted mean of each parcel. If 'sum', the function computes the
        area-weighted sum of each parcel. Default is 'mean'.

    Returns
    -------
    np.ndarray
        The parcellated spatial maps, of shape ``(n_parcels, n_maps)``.

    Raises
    ------
    ValueError
        If ``method`` is not 'mean' or 'sum'.
    ValueError
        If ``data`` is not 1D or 2D.
    ValueError
        If ``parcellation`` is not 1D.
    """
    # Format / validate arguments
    if method not in ('mean', 'sum'):
        raise ValueError(f"method must be 'mean' or 'sum'; got {method}.")
    
    ved = EigenData(data=(data, parcellation), mass=mass)
    data, parcellation = ved.data
    n_verts = data.shape[0]
    areas = _mass_to_areas(ved.mass, n_verts)
    parcellation = np.asarray(parcellation, dtype=int, copy=True)
    
    if data.ndim > 2:
        raise ValueError("data must be 1D or 2D.")
    if parcellation.ndim != 1:
        raise ValueError("Parcellation map must be 1D.")
    
    parc_ids = np.unique(parcellation)
    n_parcels = len(parc_ids)
    is_data_vec = (data.ndim == 1)
    data_2d = data[:, np.newaxis] if is_data_vec else data

    # Construct sparse parcellation matrix as (n_parcels, n_verts)
    parc_mat = csr_matrix(
        (np.ones(n_verts),
         (parcellation, np.arange(n_verts))),
        shape=(n_parcels, n_verts)
        )

    # Adjust parcellation matrix for vertex areas
    parc_areas = parc_mat @ areas
    parc_mat = parc_mat.multiply(areas)
    if method == 'mean':
        parc_mat /= parc_areas[:, np.newaxis]

    # Apply parcellation matrix to data
    data_parc = parc_mat @ data_2d
    return data_parc.squeeze(axis=1) if is_data_vec else data_parc

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
        warn("Mass matrix not provided; assuming that area at each vertex is 1")
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

def _mult_by_cholesky(
    data: NDArray[np.floating],
    matrix: spmatrix | NDArray[np.floating],
    transpose: bool = False
) -> NDArray[np.floating]:
    """
    Multiplies the input data by the Cholesky factor L (or its transpose Lᵀ) of the given matrix
    (i.e., matrix = LLᵀ). If the matrix is diagonal, L = Lᵀ = √(matrix) (element-wise). Note that
    ``matrix`` must be symmetric positive definite for the Cholesky factorization to exist.

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    matrix : array-like
        The matrix to multiply the data by, of shape ``(n_verts, n_verts)``.
    transpose : bool, optional
        If True, multiplies by ``L.T`` instead of ``L``. Default is ``False``.

    Returns
    -------
    np.ndarray
        The input data multiplied by the Cholesky factor of the matrix, of shape ``(n_verts,
        n_maps)``.

    Raises
    ------
    ValueError
        If the matrix is not symmetric or not positive definite.
    """
    ved = EigenData(data=data, mass=matrix)  # a bit dodgy
    data, matrix = ved.data, ved.mass

    # check symmetry of matrix
    if (matrix != matrix.T).nnz > 0:
        raise ValueError("matrix is not symmetric.")

    if matrix.nnz == matrix.shape[0]:
        # Diagonal matrix; L = Lᵀ = √(matrix)
        return data * np.sqrt(matrix.diagonal())[:, None]

    # Non-diagonal matrix; compute Cholesky factorization via LU decomposition.

    # Manually permute rows/cols of matrix to reduce its bandwidth and thus increase sparsity of L
    # and U factors for efficiency. splu usually does this internally, but does not offer the option
    # to ensure symmetric permutation, as is needed for L_lu U = L_lu D L_luᵀ (see below).
    perm = reverse_cuthill_mckee(matrix, symmetric_mode=True)
    matrix_perm = matrix[perm, :][:, perm]

    # Factorize, while forcing splu to avoid permuting and pivoting
    # matrix = L_lu U = L_lu D L_luᵀ = (L_lu √D) (L_lu √D)ᵀ = L Lᵀ
    lu = splu(matrix_perm, permc_spec='NATURAL', diag_pivot_thresh=0)
    L_lu = lu.L.tocsr()
    D = lu.U.diagonal()[:, None]

    # Check that matrix is SPD (D must be positive)
    # this is a necessary but not sufficient condition, but computationally easy
    # TODO: consider more robust checks
    if np.any(D <= 0):
        raise ValueError("matrix is not positive definite.")

    # Scale permuted data
    if transpose:
        # Lᵀ = (L_lu √D)ᵀ = √D L_luᵀ
        data_rescaled = np.sqrt(D) * (L_lu.T @ data[perm, :])
    else:
        # L = L_lu √D
        data_rescaled = L_lu @ (np.sqrt(D) * data[perm, :])

    # Reverse the permutation
    inv_perm = np.argsort(perm)
    return data_rescaled[inv_perm, :]