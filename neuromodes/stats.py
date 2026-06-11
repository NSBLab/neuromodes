"""
Mass (or simply area)-weighted adaptations of common statistical functions for spatial maps.
Conventional functions are equivalent to setting mass to identity, representing a mesh where each
vertex has Voronoi area/volume of 1.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from warnings import warn
from scipy.spatial.distance import squareform, cdist
from scipy.sparse import csc_matrix, spmatrix, diags
from neuromodes.eigen import EigenData

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.spatial.distance import _MetricCallback, _MetricKind

def gramw(
    data: NDArray[np.floating],
    data_b: NDArray[np.floating] | None = None,
    *,
    mass: spmatrix | NDArray[np.floating]
) -> NDArray[np.floating]:
    """Dot product between all columns (pairwise)."""
    ved = EigenData(data=(data, data_b), mass=mass)
    a, b = ved.data
    mass = _process_vertex_areas(ved.mass, a.shape[0])

    if b is None:
        b = a
    return a.T @ (mass @ b)

# TODO: ensure that all functions support nD input, not just 2D
def dotw(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating],
    keepdims: bool = False
) -> NDArray[np.floating]:
    """Dot product between corresponding brain maps (not pairwise).
    """
    ved = EigenData(data=(data_a, data_b), mass=mass)
    a, b = ved.data

    if a.shape != b.shape:
        raise ValueError(f"data_a and data_b must have matching shapes; got {a.shape} and {b.shape}.")

    n_verts = a.shape[0]
    mass = _process_vertex_areas(ved.mass, n_verts)
    a_2d = a.reshape(n_verts, -1)
    b_2d = b.reshape(n_verts, -1)
    out = np.sum(a_2d * (mass @ b_2d), axis=0).reshape(a.shape[1:])
    return np.expand_dims(out, axis=0) if keepdims else out

def ssqw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating],
    keepdims: bool = False
) -> NDArray[np.floating]:
    """Energy (sum of squares) of each column."""
    return dotw(data, data, mass=mass, keepdims=keepdims)

def meanw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating],
    keepdims: bool = False
) -> float:
    """Area-weighted mean. Note that this is equivalent to the more general
    sum(mass @ data) / mass.sum()."""
    ved = EigenData(data=data, mass=mass)  # FIXME: areas vector
    data, mass = ved.data, ved.mass

    areas = _mass_to_areas(mass, data.shape[0])
    out = np.average(data, axis=0, weights=areas)
    return out[None, :] if keepdims else out  # TODO: test for data.ndim /= 2 cases

def demeanw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating]
) -> NDArray[np.floating]:
    """Remove the area-weighted mean."""
    data = np.asarray(data)
    return data - meanw(data, mass)

def varw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating],
    keepdims: bool = False
) -> float:
    """Area-weighted variance."""
    data = np.asarray(data)
    mass = _process_vertex_areas(mass, data.shape[0])

    B = demeanw(data, mass)
    out = ssqw(B, mass) / mass.sum()
    return out[None, :] if keepdims else out  # TODO: test for data.ndim /= 2 cases

def momentw(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating],
    order: int,
    keepdims: bool = False
) -> float:
    """Area-weighted statistical moment of a given order."""
    data = np.asarray(data)

    if order == 1:
        n_maps = data.shape[1] if data.ndim == 2 else 1
        return np.zeros(n_maps)
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
    mass: spmatrix | NDArray[np.floating],
    keepdims: bool = False
) -> float:
    """Area-weighted standard deviation."""
    return np.sqrt(varw(data, mass, keepdims=keepdims))

def zscorew(
    data: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating]
) -> NDArray[np.floating]:
    """Z-score using area-weighted mean and standard deviation."""
    return demeanw(data, mass) / stdw(data, mass)

def covw(
    data: NDArray[np.floating],
    data_b: NDArray[np.floating] | None = None,
    *,
    mass: spmatrix | NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Weighted covariance.
    Usage: covw(A, mass=w) computes covariance of A with itself.
           covw(A, B, mass=w) computes cross-covariance between A and B.
    No bias correction is applied, equivalent to np.cov(..., bias=True). Since mesh vertices are not
    IID samples and maps typically display spatial autocorrelation, Bessel's N-1 correction is not
    appropriate.
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
    mass: spmatrix | NDArray[np.floating],
    p: int = 2,
    keepdims: bool = False
) -> NDArray[np.floating]:
    """Calculates the area-weighted L^p norm of spatial maps."""
    if p == 2: # Exact (well-defined)
        return np.sqrt(ssqw(data, mass, keepdims=keepdims))
    
    elif p == np.inf: # Exact (ignores the mass matrix)
        out = np.max(np.abs(data), axis=0)
        return out[None, :] if keepdims else out
    
    else: # Approximate by lumping
        data = np.asarray(data)
        areas = _mass_to_areas(mass, data.shape[0])
        out = np.sum(areas * (np.abs(data) ** p), axis=0) ** (1 / p)
        return out[None, :] if keepdims else out

def cdistw(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating],
    metric: _MetricCallback | _MetricKind = 'euclidean'
) -> NDArray[np.floating]:
    """Pairwise distance between columns of X and Y, accounting for mass matrix. Some functions support
    exact calculation (these have been reimplemented); some functions are approximated by lumping
    (scipy.spatial.distance.cdist with weights)."""
    ved = EigenData(data=(data_a, data_b), mass=mass)  # a lot of redundancy, especially for correlation. TODO: consider adding `checks`, and `checks='mass'` to EigenData
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
    mass: spmatrix | NDArray[np.floating],
    metric: _MetricCallback | _MetricKind = 'euclidean'
) -> NDArray[np.floating]:
    """Pairwise distances between observations in X, outputting a condensed vector."""
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"data must be a 2D array with at least 2 columns; got shape {data.shape}.")

    D2 = cdistw(data, data, mass, metric)
    np.fill_diagonal(D2, 0) # Ensures exact 0 on diagonal
    return squareform(D2, checks=False)

def solvew(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Use method of normal equations to give area-weighted least squares error.
    See https://en.wikipedia.org/wiki/Weighted_least_squares#Motivation
    """
    ved = EigenData(data=(data_a, data_b), mass=mass)
    a, b = ved.data
    mass = _process_vertex_areas(ved.mass, a.shape[0])
    # Solves (a'Wa)x = (a'Wb)
    return np.linalg.solve(a.T @ mass @ a, a.T @ mass @ b)

def lstsqw(
    data_a: NDArray[np.floating],
    data_b: NDArray[np.floating],
    mass: spmatrix | NDArray[np.floating],
    rcond: float | None = None
) -> tuple[NDArray[np.floating], int, float, NDArray[np.floating]]:
    """
    Solve the weighted least squares by lumping the mass matrix and weighting each vertex.
    """
    ved = EigenData(data=(data_a, data_b), mass=mass)
    a, b = ved.data

    va = np.sqrt(_mass_to_areas(ved.mass, a.shape[0])) # (n_verts,)
    aw = a * va[:, np.newaxis]
    bw = b * va[:, np.newaxis] if b.ndim != 1 else b * va
    return np.linalg.lstsq(aw, bw, rcond=rcond)

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