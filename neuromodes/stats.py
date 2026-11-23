"""
Mass (or simply vertex area/volume)-weighted adaptations of common statistical functions for spatial
maps. Conventional functions are equivalent to setting mass to identity, representing a mesh where
each vertex has Voronoi area/volume of 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload
from warnings import warn

import numpy as np
from scipy.sparse import csc_matrix, spmatrix, diags, issparse
from scipy.spatial.distance import cdist as scipy_cdist, squareform as scipy_squareform

# from neuromodes.eigen import EigenData
from neuromodes.validation import _create_mass_matrix, _process_parcellation_matrix, _mass_to_areas, _check_n_verts

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    from scipy.spatial.distance import _MetricCallback, _MetricKind

@overload
def _multiply_nd( # pyright: ignore[reportOverlappingOverload]
    A: csc_matrix, 
    B: csc_matrix
) -> csc_matrix: ...

@overload
def _multiply_nd(
    A: csc_matrix | NDArray[np.floating], 
    B: csc_matrix | NDArray[np.floating]
) -> NDArray[np.floating]: ...
    
def _multiply_nd(
    A: csc_matrix | NDArray[np.floating], 
    B: csc_matrix | NDArray[np.floating]
) -> csc_matrix | NDArray[np.floating]:
    """Compute A @ B where contraction occurs over axis 1 of A and axis 0 of B.
    
    Preserves sparse matrix types when possible and handles N-D dense arrays.
    """
    if A.shape is None or B.shape is None: # make pyright happy
        raise ValueError("Input arrays must have defined shapes.")
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        # dense @ dense -> N-D tensor contraction
        return np.tensordot(A, B, axes=(1, 0))
    elif issparse(A) and issparse(B):
        # sparse @ sparse -> sparse matrix
        return A @ B
    elif issparse(A):
        # sparse @ dense -> dense array (reshaped to maintain B's trailing axes)
        B_2d = B.reshape(B.shape[0], -1)
        output_shape = (A.shape[0],) + B.shape[1:]
        return (A @ B_2d).reshape(output_shape)
    elif issparse(B):
        # dense @ sparse -> dense array (reshaped to maintain A's trailing axes)
        A_2d = A.reshape(A.shape[0], -1)
        output_shape = (A.shape[0],) + B.shape[1:]
        return (A_2d @ B).reshape(output_shape)
    else:
        raise TypeError(f"Unsupported types for _multiply_nd: {type(A)} and {type(B)}")

@overload
def _dot_nd( # pyright: ignore[reportOverlappingOverload]
    A: csc_matrix, 
    B: csc_matrix
) -> csc_matrix: ...

@overload
def _dot_nd(
    A: csc_matrix | NDArray[np.floating], 
    B: csc_matrix | NDArray[np.floating]
) -> NDArray[np.floating]: ...

def _dot_nd(
    A: csc_matrix | NDArray[np.floating], 
    B: csc_matrix | NDArray[np.floating]
) -> csc_matrix | NDArray[np.floating]:
    """Compute <A, B> by contracting over axis 0 (the spatial dimension, n_verts).
    
    Equivalent to A.T @ B for 2D inputs, but supports any rank N-D arrays.
    """
    if A.shape is None or B.shape is None: # make pyright happy
        raise ValueError("Input arrays must have defined shapes.")
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        # dense.T @ dense -> N-D tensor contraction
        return np.tensordot(A, B, axes=(0, 0))
    elif issparse(A) and issparse(B):
        # sparse.T @ sparse -> works natively in SciPy
        return A.T @ B
    elif issparse(A):
        # sparse.T @ dense -> flatten B's trailing axes, perform sparse matmul, reshape back
        B_2d = B.reshape(B.shape[0], -1)
        output_shape = A.shape[1:] + B.shape[1:]
        return (A.T @ B_2d).reshape(output_shape)
    elif issparse(B):
        # dense.T @ sparse -> flatten A's trailing axes, perform sparse matmul, reshape back
        A_2d = A.reshape(A.shape[0], -1)
        output_shape = A.shape[1:] + B.shape[1:]
        return (A_2d.T @ B).reshape(output_shape)
    else:
        raise TypeError(f"Unsupported types for _dot_nd: {type(A)} and {type(B)}")

def _pad_dims_right(arr: NDArray[np.floating], target_ndim: int) -> NDArray[np.floating]:
    """Pad the dimensions of an array on the right with singleton dimensions to reach target_ndim."""
    return arr[(...,) + (None,) * max(0, target_ndim - arr.ndim)]

def _dot_diagonal_nd(
    a: csc_matrix | NDArray[np.floating],
    b: csc_matrix | NDArray[np.floating], 
    keepdims: bool = False
) -> NDArray[np.floating]:
    """Compute the dot product of two arrays along their diagonal elements.

    This is a specialized function for computing the dot product of two arrays
    where only the diagonal elements are considered.

    Parameters
    ----------
    A : array-like
        The first array.
    B : array-like
        The second array.

    Returns
    -------
    np.ndarray
        The result of the dot product.
    """
    
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        out = np.einsum('v...,v...->...', 
                        _pad_dims_right(a, max(a.ndim, b.ndim)), 
                        _pad_dims_right(b, max(a.ndim, b.ndim))
                        )
        if keepdims:
            out = np.expand_dims(out, axis=0)
        return out
    
    # Case 2: Both Sparse, with extra checks for making pyright happy
    if issparse(a) and issparse(b) and not isinstance(a, np.ndarray) and not isinstance(b, np.ndarray):
        if a.shape is None or b.shape is None:
            raise ValueError("Input arrays must have defined shapes.")
        if a.shape[1] == 1:
            out = (a.T @ b).toarray()
        elif b.shape[1] == 1:
            out = (b.T @ a).toarray()
        else: 
            out = np.asarray(a.multiply(b).sum(axis=0))
        if not keepdims:
            out = out.squeeze(axis=0)
        return out

    # Case 3: One Sparse, One Dense, with explicit checks for making pyright happy
    if issparse(a) and not isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        sparse_mat, dense_arr = a, b
    elif issparse(b) and not isinstance(b, np.ndarray) and isinstance(a, np.ndarray):
        sparse_mat, dense_arr = b, a
    else:
        raise TypeError(f"Unsupported types for _dot_diagonal_nd: {type(a)} and {type(b)}")
    if sparse_mat.shape is None or dense_arr.shape is None:
        raise ValueError("Input arrays must have defined shapes.")
    
    b_s = sparse_mat.shape[1] # csc_matrix is always exactly 2D
    b_d = dense_arr.shape[1] if dense_arr.ndim > 1 else 0

    if dense_arr.ndim == 1:
        out = np.asarray(sparse_mat.T @ dense_arr)

    elif dense_arr.ndim == 2:
        if b_s == 1:
            out = sparse_mat.T @ dense_arr
        elif b_d == 1:
            out = dense_arr.T @ sparse_mat
        elif b_s == b_d:
            out = sparse_mat.multiply(dense_arr).sum(axis=0)
        else:
            raise ValueError(
                f"Cannot broadcast column dimension {b_s} of sparse matrix with {b_d} of dense array."
            )
        out = np.asarray(out).squeeze(axis=0)

    else:
        coo = sparse_mat.tocoo()
        s_vals = coo.data[(...,) + (None,) * (dense_arr.ndim - 2)]
        out = np.zeros((max(b_s, b_d),) + dense_arr.shape[2:], 
                        dtype=np.result_type(sparse_mat.dtype, dense_arr.dtype))

        if b_s == 1: # sparse (a, 1) vs dense (a, b, ...)
            d_slices = dense_arr[coo.row, :]
            out = np.sum(s_vals[:, None] * d_slices, axis=0)
        elif b_d == 1: # sparse (a, b) vs dense (a, 1, ...)
            d_slices = dense_arr[coo.row, 0]
            np.add.at(out, coo.col, s_vals * d_slices)
        elif b_s == b_d: # sparse (a, b) vs dense (a, b, ...)
            d_slices = dense_arr[coo.row, coo.col]
            np.add.at(out, coo.col, s_vals * d_slices)
        else: 
            raise ValueError(
                f"Cannot broadcast column dimension {b_s} of sparse matrix with {b_d} of dense array."
            )

    if keepdims:
        out = np.expand_dims(out, axis=0)

    return out

def gramw(
    data_a: csc_matrix | NDArray[np.floating],
    data_b: csc_matrix | NDArray[np.floating] | None = None,
    *,
    mass: csc_matrix | NDArray[np.floating] | None
) -> csc_matrix | NDArray[np.floating]:
    """
    Dot product between all brain maps (pairwise), equivalent to ``data.T @ (mass @ data_b)``.

    Parameters
    ----------
    data_a : array-like
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
    # ved = EigenData(data=(data_a, data_b), mass=mass)
    # a, b = ved.data
    # mass = _create_mass_matrix(ved.mass, a.shape[0])

    a, b = data_a, data_b
    b = a if b is None else b
    mb = b if mass is None else _multiply_nd(mass, b)
    amb = _dot_nd(a, mb)
    return amb

def dotw(
    data_a: csc_matrix | NDArray[np.floating],
    data_b: csc_matrix | NDArray[np.floating],
    mass: csc_matrix | NDArray[np.floating] | None = None,
    keepdims: bool = False
) -> NDArray[np.floating]:
    """Calculate the elementwise weighted dot product between spatial maps across axis 0.

    Equivalent to ``np.sum(data_a * (mass @ data_b), axis=0)``. Trailing dimensions
    beyond axis 0 are broadcasted automatically.

    Parameters
    ----------
    data_a : NDArray
        First set of spatial maps of shape ``(n_verts, ...)``.
    data_b : NDArray
        Second set of spatial maps of shape ``(n_verts, ...)``.
    mass : csc_matrix or NDArray, optional
        Mass matrix of shape ``(n_verts, n_verts)``.
    keepdims : bool, optional
        Whether to retain the contracted spatial dimension (axis 0) with length 1.
        Default is False.

    Returns
    -------
    NDArray
        The weighted inner product array broadcasted over trailing dimensions.
    """
    # ved = EigenData(data=(data_a, data_b), mass=mass)
    # a, b = ved.data
    # mass = _create_mass_matrix(ved.mass, a.shape[0])

    # 1. Apply mass operator to b: M @ b  (shape: (n_verts, ...))
    a, b = data_a, data_b
    mb = b if mass is None else _multiply_nd(mass, b)
    amb = _dot_diagonal_nd(a, mb, keepdims=keepdims)
    return amb


def ssqw(
    data: NDArray[np.floating],
    mass: csc_matrix | NDArray[np.floating] | None,
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
    mass: csc_matrix | NDArray[np.floating] | None,
    keepdims: bool = False
) -> NDArray[np.floating]:
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
    # ved = EigenData(data=data, mass=mass)
    # data, mass = ved.data, ved.mass
    data = np.asarray(data)
    areas = _mass_to_areas(mass, data.shape[0], keepdims=keepdims)
    num = _dot_nd(areas, data)
    return num / areas.sum() # np.float

def demeanw(
    data: NDArray[np.floating],
    mass: csc_matrix | NDArray[np.floating] | None
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
    return data - meanw(data, mass, keepdims=True)

def varw(
    data: NDArray[np.floating],
    mass: csc_matrix | NDArray[np.floating] | None,
    keepdims: bool = False
) -> NDArray[np.floating]:
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
    out = ssqw(demeanw(data, mass), mass, keepdims=keepdims)
    return out / mass.sum() if mass is not None else out / data.shape[0]

def momentw(
    data: NDArray[np.floating],
    mass: csc_matrix | NDArray[np.floating] | None,
    order: int,
    keepdims: bool = False
) -> NDArray[np.floating]:
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
        out = np.zeros(data.shape[1:], dtype=data.dtype)
        return np.expand_dims(out, axis=0) if keepdims else out
    elif order == 2:
        return varw(data, mass, keepdims=keepdims)
    else:
        # Approximate by lumping
        areas = _mass_to_areas(mass, data.shape[0])
        data_dm = demeanw(data, areas)
        return meanw(data_dm ** order, areas, keepdims=keepdims)

def stdw(
    data: NDArray[np.floating],
    mass: csc_matrix | NDArray[np.floating] | None,
    keepdims: bool = False
) -> NDArray[np.floating]:
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
    mass: csc_matrix | NDArray[np.floating] | None
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
    # TODO future: consider inefficiency of demeaning twice, could be expanded
    return demeanw(data, mass) / stdw(data, mass, keepdims=True)

def covw(
    data: NDArray[np.floating],
    data_b: NDArray[np.floating] | None = None,
    *,
    mass: csc_matrix | NDArray[np.floating] | None
) -> NDArray[np.floating]:
    """
    Mass-weighted covariance amongst or between brain maps.

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

    Notes
    -----
    Usage:
    - ``covw(A, mass=mass)`` computes covariance of maps ``A`` amongst themselves.
    - ``covw(A, B, mass=mass)`` computes cross-covariance between maps ``A`` and maps ``B``.
    
    Note that this function does not offer Bessel's correction, as mesh vertices are not IID samples
    and maps typically display spatial autocorrelation.
    """

    a_dm = demeanw(data, mass)
    b_dm = demeanw(data_b, mass) if data_b is not None else a_dm
    gram = gramw(a_dm, b_dm, mass=mass)
    
    return gram / mass.sum() if mass is not None else gram / data.shape[0]

def vecnormw(
    data: NDArray[np.floating],
    mass: csc_matrix | NDArray[np.floating] | None,
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
        return np.max(np.abs(data), axis=0, keepdims=keepdims)
    
    else: # Approximate by lumping
        return dotw(np.ones((data.shape[0],)), (np.abs(data) ** p), mass=mass, keepdims=keepdims) ** (1 / p)

def cdistw(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: csc_matrix | NDArray[np.floating] | None,
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
    # ved = EigenData(data=(data_a, data_b), mass=mass)
    # a, b = ved.data
    # mass = ved.mass

    a, b = np.asarray(data_a), np.asarray(data_b)
    target_ndim = (a.ndim - 1) + (b.ndim - 1)

    if metric == 'sqeuclidean':
        ssqwa = _pad_dims_right(ssqw(a, mass), target_ndim)
        ssqwb = ssqw(b, mass)
        gramd = gramw(a, b, mass=mass)
        D = ssqwa + ssqwb - 2 * gramd
        
    elif metric == 'euclidean':
        D = np.sqrt(cdistw(a, b, mass, 'sqeuclidean'))
        
    elif metric == 'cosine':
        Num = gramw(a, b, mass=mass)
        DenX = _pad_dims_right(vecnormw(a, mass, 2), target_ndim)
        DenY = vecnormw(b, mass, 2)
        D = 1 - Num / (DenX * DenY)
        
    elif metric == 'correlation':
        D = cdistw(demeanw(a, mass), demeanw(b, mass), mass, 'cosine')
        
    else:
        areas = _mass_to_areas(mass, a.shape[0])
        D = scipy_cdist(a, b, metric=metric, w=areas)

    return np.maximum(D, 0)

def pdistw(
    data: NDArray[np.floating],
    mass: csc_matrix | NDArray[np.floating] | None,
    metric: _MetricCallback | _MetricKind = 'euclidean', 
    squareform: bool | None = None
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
    D2 = cdistw(data, data, mass, metric) # will be either 0d (if input is 1D) or >=2d (if input is >=2D)
    if squareform is None: 
        squareform = D2.ndim == 2
    if squareform:
        return scipy_squareform(D2, force='tovector', checks=False)
    return D2

def solvew(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: csc_matrix | NDArray[np.floating] | None
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
    # ved = EigenData(data=(data_a, data_b), mass=mass)
    # a, b = ved.data
    a, b = data_a, data_b
    mass = _create_mass_matrix(mass, a.shape[0])
    _check_n_verts(a=a, b=b, mass=mass)
    # Solves (a'Wa)x = (a'Wb)
    return np.linalg.solve(a.T @ mass @ a, a.T @ mass @ b)

def lstsqw(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: csc_matrix | NDArray[np.floating] | None,
    rcond: float | None = None
# ) -> tuple[NDArray[np.floating], int, float, NDArray[np.floating]]:
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], np.int32, NDArray[np.floating[Any]]]: 
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
    # ved = EigenData(data=(data_a, data_b), mass=mass)
    # a, b = ved.data
    mass = _create_mass_matrix(mass, data_a.shape[0])
    a, b = data_a, data_b
    _check_n_verts(a=a, b=b, mass=mass)

    va = np.sqrt(_mass_to_areas(mass, a.shape[0])) # (n_verts,)
    aw = a * va[:, np.newaxis]
    bw = b * va[:, np.newaxis] if b.ndim != 1 else b * va
    return np.linalg.lstsq(aw, bw, rcond=rcond)

def parcellate(
    data: csc_matrix | NDArray[np.floating],
    parcellation: NDArray[np.integer],
    mass: csc_matrix | NDArray[np.floating] | None,
    method: Literal['mean', 'sum'] = 'mean'
) -> csc_matrix | NDArray[np.floating]:
    """
    Area-weighted parcellation of each brain map.

    Parameters
    ----------
    data : array-like
        The spatial maps, of shape ``(n_verts, n_maps)``.
    parcellation : array-like
        The parcellation map, of shape ``(n_verts,)``, where each value is an integer representing
        the parcel ID for the corresponding vertex. Values of -1 are treated as medial wall and 
        ignored in the parcellation.
    mass : array-like
        The mass matrix, of shape ``(n_verts, n_verts)``.
    method : {'mean', 'sum'}, optional
        The method for aggregating vertex values within each parcel. If ``'mean'``, the function
        computes the area-weighted mean of each parcel. If ``'sum'``, the function computes the
        area-weighted sum of each parcel. Default is ``'mean'``.

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
    data = np.asarray(data)
    if method not in ('mean', 'sum'):
        raise ValueError(f"method must be 'mean' or 'sum'; got {method}.")

    # Construct p = sparse parcellation matrix as (n_verts, n_parcels)
    # output = p.T @ mass @ data; or 
    # output = p.T @ mass @ data / (p @ mass @ 1) if method == 'mean'
    parc_mat = _process_parcellation_matrix(parcellation)
    parc_mass = _dot_nd(parc_mat, _create_mass_matrix(mass, data.shape[0]))
    if method == 'mean':
        parc_mass /= parc_mass.sum(axis=1) 

    data_parc = _multiply_nd(parc_mass, data)
    return data_parc


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
    # if checks is not False:
    #     data = EigenData(data=data).data
    
    return lower + (upper - lower) / (1 + np.exp(-steepness * (data - center)))
