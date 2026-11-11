"""
Mass (or simply vertex area/volume)-weighted adaptations of common statistical functions for spatial
maps. Conventional functions are equivalent to setting mass to identity, representing a mesh where
each vertex has Voronoi area/volume of 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from warnings import warn

import numpy as np
from scipy.sparse import csc_matrix, diags, issparse, spmatrix
from scipy.sparse.linalg import splu
from scipy.spatial.distance import cdist, squareform

from neuromodes.eigen import EigenData

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.spatial.distance import _MetricCallback, _MetricKind

def gramw(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating] | None = None,
    *,
    mass: spmatrix | NDArray[np.floating] | None
) -> NDArray[np.floating]:
    """
    Mass-weighted inner product between all brain maps (pairwise), equivalent to ``data_a.T @ (mass
    @ data_b)``.

    Parameters
    ----------
    data_a : array-like
        The first set of spatial maps, of shape ``(n_verts, n_maps)``.
    data_b : array-like, optional
        The second set of spatial maps, of shape ``(n_verts, n_maps_b)``. If not provided, the
        function computes the Gram matrix of ``data_a`` with itself. Default is ``None``.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.

    Returns
    -------
    np.ndarray
        The Gram matrix of shape ``(n_maps, n_maps_b)`` if ``data_b`` is provided, or
        ``(n_maps, n_maps)`` if not.
    """
    ved = EigenData(data=(data_a, data_b), mass=mass)
    a, b = ved.data
    mass = _create_mass_matrix(ved.mass, a.shape[0])

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
    Mass-weighted inner products between corresponding brain maps (not pairwise), equivalent to
    ``sum(data_a * (mass @ data_b), axis=0)``.

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
        The inner products of shape ``(n_maps,)`` if ``keepdims=False``, or ``(1, n_maps)`` if
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
    mass = _create_mass_matrix(ved.mass, n_verts)
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
    Mass-weighted sums of squares of each brain map, equivalent to ``sum(data * (mass @ data),
    axis=0)``.

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
    mass = _create_mass_matrix(mass, data.shape[0])

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
        If ``True``, the output will have the same number of dimensions as the input array. Default is
        ``False``.

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
        If ``True``, the output will have the same number of dimensions as the input array. Default is
        ``False``.

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
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating] | None = None,
    *,
    mass: spmatrix | NDArray[np.floating] | None
) -> NDArray[np.floating]:
    """
    Mass-weighted covariance amongst or between brain maps.

    Parameters
    ----------
    data_a : array-like
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

    Notes
    -----
    Usage:
    - ``covw(A, mass=mass)`` computes covariance of maps ``A`` amongst themselves.
    - ``covw(A, B, mass=mass)`` computes cross-covariance between maps ``A`` and maps ``B``.
    
    Note that this function does not offer Bessel's correction, as mesh vertices are not IID samples
    and maps typically display spatial autocorrelation.
    """
    a = np.asarray(data_a)

    mass = _create_mass_matrix(mass, a.shape[0])
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

    Notes
    -----
    Cases:
    - ``p=2``: exact, square root of sum of squares
    - ``p=np.inf``: exact, maximum absolute value (no mass needed)
    - ``p!=2 and p!=np.inf``: approximate using vertex areas (i.e., lumped mass matrix)
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
    data_a: NDArray[np.floating] | spmatrix,
    data_b: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating] | None,
    rcond: float | None = None
) -> tuple[NDArray[np.floating], int, float, NDArray[np.floating]]:
    """
    Solve the linear system Ax = b, where A and b are sets of spatial maps (e.g., basis maps
    and target maps, respectively), by minimising the mass-weighted squared L₂ norm 
    (b-Ax)ᵀM(b-Ax) (see Notes).

    Parameters
    ----------
    data_a : array-like
        The basis maps of shape ``(n_verts, n_maps_a)``.
    data_b : array-like
        The target maps of shape ``(n_verts, n_maps_b)``.
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
    For discretized spatial maps, we want to minimize the mass-weighted L₂ norm. By leveraging a
    Cholesky-like decomposition of the symmetric positive definite mass matrix (M = FFᵀ), we can
    transform the mass-weighted norm into a typical Euclidean norm:
    
    ||b-Ax||₂²_mass = (b-Ax)ᵀM(b-Ax)
                    = (b-Ax)ᵀFFᵀ(b-Ax)
                    = (Fᵀ(b-Ax))ᵀ(Fᵀ(b-Ax))
                    = ((Fᵀb)-(FᵀA)x)ᵀ((Fᵀb)-(FᵀA)x)
                    = ||(Fᵀb)-(FᵀA)x||₂²_Euclidean
    
    The above tells us that we can still use a standard least squares solver if we simply scale the
    data by Fᵀ. This is equivalent to using ``np.linalg.solve(a.T @ mass @ a, a.T @ mass @ b)`` but
    is more efficient and numerically stable. Note that for lumped (diagonal) mass, F = Fᵀ = √M
    (element-wise).
    """
    ved = EigenData(data=(data_a, data_b), mass=mass)
    a, b = ved.data

    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]

    # scale data by transposed Cholesky-like factor of mass matrix
    chol = _CholeskyFactor(ved.mass)
    if issparse(a):
        # convert output to dense for np.linalg.lstsq(). no good alternative as scipy's splu()
        # cannot handle rectangular systems, scipy has no sparse QR solver, iterative solvers like
        # lsqr() and lsmr() are inefficient for multi RHS, and spsolve() is not stable for
        # ill-conditioned systems.
        FT_a = chol.multiply(a, transpose=True).toarray()

        # b is dense so need to multiply separately
        FT_b = chol.multiply(b, transpose=True)
    else:
        # concatenate and multiply for efficiency
        FT_a, FT_b = np.split(
            chol.multiply(np.concatenate([a, b], axis=1), transpose=True),
            [a.shape[1]],
            axis=1
            )

    # Get results from standard least squares solver
    return np.linalg.lstsq(FT_a, FT_b, rcond=rcond)

def parcellate(
    data: NDArray[np.floating],
    parcellation: NDArray[np.integer],
    mass: spmatrix | NDArray[np.floating] | None,
    method: Literal['mean', 'sum'] = 'mean'  # TODO: add 'var' (e.g., for parcel homogeneity)
) -> NDArray[np.floating]:
    """
    Mass-weighted parcellation of each brain map.

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    parcellation : array-like
        The parcellation map, of shape ``(n_verts,)``, where each value is an integer representing
        the parcel ID for the corresponding vertex. Negative values are treated as outside the mask
        (e.g., medial wall) and are ignored in the parcellation.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    method : {'mean', 'sum'}, optional
        The method for aggregating vertex values within each parcel. If ``'mean'``, the function
        computes the mass-weighted mean of each parcel. If ``'sum'``, the function computes the
        mass-weighted sum of each parcel. Default is ``'mean'``.

    Returns
    -------
    np.ndarray
        The parcellated spatial maps, of shape ``(n_parcels, n_maps)``.

    Raises
    ------
    ValueError
        If ``method`` is not ``'mean'`` or ``'sum'``.
    ValueError
        If ``data`` is not 1D or 2D.
    """
    # Format / validate arguments
    ved = EigenData(data=(data, parcellation), mass=mass)
    data, parcellation = ved.data
    if data.ndim > 2:
        raise ValueError("data must be 1D or 2D.")
    
    if method not in ('mean', 'sum'):
        raise ValueError(f"method must be 'mean' or 'sum'; got {method}.")

    mass = _create_mass_matrix(ved.mass, data.shape[0])
    parc_mat = _create_parcellation_matrix(parcellation)

    # Parcellate data u via parcellation matrix P and mass matrix M:
    # 'sum' -> x = PᵀMu
    # 'mean' -> x = (PᵀMP)⁻¹PᵀMu (least-squares minimiser for unparcellation Px ≈ u)
    parc_data = (gramw(parc_mat, data, mass=mass) if method == 'sum'
                 else lstsqw(parc_mat, data, mass=mass)[0])
    return parc_data[:, 0] if data.ndim == 1 else parc_data

# TODO add support for passing through parcellation matrix without modification
# TODO add support for dtypes
def _create_parcellation_matrix(
    parcellation: NDArray[np.integer]
) -> csc_matrix:
    """
    Converts a parcellation vector to a sparse parcellation matrix.

    Parameters
    ----------
    parcellation : array-like
        The parcellation map, of shape ``(n_verts,)``, where each value is an integer representing
        the parcel ID for the corresponding vertex.

    Returns
    -------
    scipy.sparse.csc_matrix
        The sparse parcellation matrix, of shape ``(n_verts, n_parcels)``.

    Raises
    ------
    ValueError
        If ``parcellation`` is not 1D.
    """
    parcellation = np.asarray(parcellation, copy=True) # user has to ensure parc is integer

    if parcellation.ndim != 1:
        raise ValueError("Parcellation map must be 1D.")
    
    n_verts = parcellation.shape[0]
    mask = parcellation >= 0 # medial wall should be negative
    
    # Construct sparse parcellation matrix as (n_parcels, n_verts)
    parc_mat = csc_matrix(
        (np.ones(np.sum(mask)),
         (np.arange(n_verts)[mask], parcellation[mask])),
        shape=(n_verts, np.max(parcellation)+1)
        )
    
    return parc_mat

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
    mass = _create_mass_matrix(mass=mass, n_verts=n_verts)
    return np.asarray(mass.sum(axis=0)).ravel()

# TODO: consider adding dtype parameter to _create_mass_matrix for w=None case
# TODO: consider moving to EigenData
def _create_mass_matrix(
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

class _CholeskyFactor:
    """
    Computes a Cholesky-derived square-root factor of a sparse symmetric positive definite
    ``matrix`` (A). For a non-diagonal matrix, a fill-reducing permutation P is applied first. The
    permuted matrix is factorized as

        PAPᵀ = LLᵀ,

    where L is lower triangular. The factor applied in the original ordering is given by

        F = PᵀLP,

    such that A = FFᵀ. Due to permutation, F is generally not triangular. If A is diagonal, L = Lᵀ =
    F = Fᵀ = √A (element-wise; P = I).

    Parameters
    ----------
    matrix : spmatrix
        The symmetric positive definite matrix, of shape ``(n_verts, n_verts)``.
    """
    def __init__(self, matrix: spmatrix):
        self.is_diag = matrix.nnz == np.count_nonzero(matrix.diagonal())

        if not self.is_diag:
            # Let SuperLU choose a fill-reducing symmetric ordering (min degree on A+Aᵀ) and force
            # diagonal pivots such that PAPᵀ = L_lu U_lu = L_lu D L_luᵀ = (Uᵀ/√D)(Uᵀ/√D)ᵀ = LLᵀ
            lu = splu(
                matrix,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0,
                options={"SymmetricMode": True},
            )
            self.U = lu.U
            self.perm_c = lu.perm_c

        self.sqrt_diag = np.sqrt((matrix if self.is_diag else lu.U).diagonal())[:, None]

    def multiply(
        self,
        data: NDArray[np.floating] | spmatrix,
        transpose: bool = False,
    ) -> NDArray[np.floating] | spmatrix:
        """
        Multiplies the input data by the Cholesky-derived factor F or its transpose Fᵀ.

        Parameters
        ----------
        data : array-like
            The spatial maps, of shape ``(n_verts, n_maps)``.
        transpose : bool, optional
            If ``True``, multiplies by Fᵀ instead of F. Default is ``False``.

        Returns
        -------
        np.ndarray or scipy.sparse.spmatrix
            The result of the multiplication, of shape ``(n_verts, n_maps)``.
        """
        if issparse(data):
            if self.is_diag:
                # F = Fᵀ = √A
                return data.multiply(self.sqrt_diag)

            # Permute rows of data according to match matrix ordering (CSR for efficiency)
            data_perm = data.tocsr()[np.argsort(self.perm_c)]

            # Scale data by factor
            data_perm_scaled = ((self.U @ data_perm).multiply(1 / self.sqrt_diag) if transpose
                                else self.U.T @ (data_perm.multiply(1 / self.sqrt_diag)))

            # Reverse permutation and return to original spmatrix type
            return data.__class__(data_perm_scaled.tocsr()[self.perm_c], copy=False)

        if self.is_diag:
            # F = Fᵀ = √A
            return data * self.sqrt_diag

        # Permute rows of data according to match matrix ordering
        data_perm = np.empty_like(data)
        data_perm[self.perm_c] = data

        # Scale data by factor and reverse permutation
        return ((self.U @ data_perm) / self.sqrt_diag if transpose
                else self.U.T @ (data_perm / self.sqrt_diag))[self.perm_c]