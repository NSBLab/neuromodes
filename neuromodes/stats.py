from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from warnings import warn
from scipy.spatial.distance import squareform, cdist
from scipy.stats import rankdata
from scipy.sparse import csc_matrix, diags

if TYPE_CHECKING:
    from scipy.spatial.distance import _MetricCallback, _MetricKind

def dotw(A, B, w):
    """Dot product between corresponding columns (not pairwise)."""
    W = _process_vertex_areas(w, A.shape[0])
    return np.sum(A * (W @ B), axis=0)

def gramw(A, B, w):
    """Dot product between all columns (pairwise)."""
    W = _process_vertex_areas(w, A.shape[0])
    return A.T @ (W @ B)

def ssqw(A, w):
    """Energy (sum of squares) of each column."""
    return dotw(A, A, w)

# TODO : consider adding keepdims parameter
def meanw(A, w):
    """Area-weighted mean."""
    W = _process_vertex_areas(w, A.shape[0])
    sa = W.sum()
    return (W @ A).sum(axis=0) / sa

def demeanw(A, w):
    """Remove the area-weighted mean."""
    return A - meanw(A, w)

def varw(A, w):
    """Area-weighted variance."""
    W = _process_vertex_areas(w, A.shape[0])
    sa = W.sum()
    B = demeanw(A, w)
    return ssqw(B, w) / sa

def momentw(A, w, order):
    """Area-weighted statistical moment of a given order."""
    if order == 1:
        return np.zeros(A.shape[1])
    elif order == 2:
        return varw(A, w)
    else:
        # Approximate by lumping
        W = _process_vertex_areas(w, A.shape[0])
        sa = W.sum()
        B = demeanw(A, w)
        # Sum rows of the sparse matrix to get a lumped vector, safely flattened
        w_lumped = np.asarray(W.sum(axis=1)) 
        return np.sum(w_lumped * (B ** order)) / sa

def stdw(A, w):
    """Area-weighted standard deviation."""
    return np.sqrt(varw(A, w))

def normalizew(A, w):
    """Z-score normalize using area-weighted mean and standard deviation."""
    return demeanw(A, w) / stdw(A, w)

def covw(A, arg2, arg3=None):
    """
    Weighted covariance.
    Usage: covw(A, w) computes covariance of A with itself.
           covw(A, B, w) computes cross-covariance between A and B.
    """
    if arg3 is None:
        B = A
        w = arg2
    else:
        B = arg2
        w = arg3

    W = _process_vertex_areas(w, A.shape[0])
    sa = W.sum()
    
    A_d = demeanw(A, w)
    B_d = demeanw(B, w)
    
    return gramw(A_d, B_d, w) / sa

def vecnormw(A, w, p=2):
    """Calculates the area-weighted L^p norm of spatial maps."""
    if p == 2: # Exact (well-defined)
        return np.sqrt(ssqw(A, w))
    
    elif p == np.inf: # Exact (ignores the mass matrix)
        return np.max(np.abs(A), axis=0)
    
    else: # Approximate by lumping
        W = _process_vertex_areas(w, A.shape[0])
        w_lumped = np.asarray(W.sum(axis=1))
        return np.sum(w_lumped * (np.abs(A) ** p), axis=0) ** (1 / p)

def cdistw(
        X, 
        Y, 
        w, 
        metric: _MetricCallback | _MetricKind = 'euclidean'
    ) -> np.ndarray:
    """Pairwise distance between rows of X and Y, accounting for mass matrix. Some functions support
    exact calculation (these have been reimplemented); some functions are approximated by lumping
    (scipy.spatial.distance.cdist with weights)."""
    
    W = _process_vertex_areas(w, X.shape[0])
    
    # Reimplement these to support unlumped mass matrix
    if metric == 'sqeuclidean':
        D = ssqw(X, w)[:, None] + ssqw(Y, w)[None, :] - 2 * gramw(X, Y, w)
        
    elif metric == 'euclidean':
        D = np.sqrt(cdistw(X, Y, w, 'sqeuclidean'))
        
    elif metric == 'cosine':
        Num = gramw(X, Y, w)
        DenX = vecnormw(X, w, 2)[:, None]
        DenY = vecnormw(Y, w, 2)[None, :]
        D = 1 - Num / (DenX * DenY)
        
    elif metric == 'correlation':
        D = cdistw(demeanw(X, w), demeanw(Y, w), w, 'cosine')
        
    else:
        weights = _mass_to_areas(w, X.shape[0])
        D = cdist(X, Y, metric=metric, w=weights)

    return np.maximum(D, 0)

def pdistw(
        X, 
        w, 
        metric: _MetricCallback | _MetricKind = 'euclidean'
    ) -> np.ndarray:
    """Pairwise distances between observations in X, outputting a condensed vector."""
    D2 = cdistw(X, X, w, metric)
    np.fill_diagonal(D2, 0) # Ensures exact 0 on diagonal
    return squareform(D2, checks=False)

def correlationw(XA, XB, w, metric='pearsonr'):    
    if metric=='spearmanr':
        dataA = rankdata(XA, axis=0)
        dataB = rankdata(XB, axis=0)
    elif metric=='pearsonr':
        dataA = XA
        dataB = XB
    else: 
        raise ValueError(f"Invalid metric '{metric}'; must be 'pearsonr' or 'spearmanr'.")
    return 1 - cdistw(dataA, dataB, w=w, metric='correlation')

## TODO 
def compare_images(): 
    raise NotImplementedError("compare_images is not yet implemented.")

def solvew(A, B, w):
    """
    Use method of normal equations to give area-weighted least squares error.
    See https://en.wikipedia.org/wiki/Weighted_least_squares#Motivation
    """
    W = _process_vertex_areas(w, A.shape[0])
    # Solves (A'WA)x = (A'WB)
    return np.linalg.solve(A.T @ W @ A, A.T @ W @ B)

def lstsqw(a, b, w, rcond=None):
    """
    Solve the weighted least squares by lumping the mass matrix and weighting each vertex.
    """
    va = np.sqrt(_mass_to_areas(w, a.shape[0])) # (n_verts,)
    aw = a * va[:, np.newaxis]
    bw = b * va[:, np.newaxis] if b.ndim != 1 else b * va
    return np.linalg.lstsq(aw, bw, rcond=rcond)

def pinv(A, w, rcond=1e-15):
    raise NotImplementedError("pinv is not yet implemented.")
    # W = _process_vertex_areas(w, A.shape[0])
    # return np.linalg.pinv(A.T @ W @ A, rcond=rcond) @ A.T @ W

# TODO
def pcaw(): 
    raise NotImplementedError("pcaw is not yet implemented.")

def dmew(): 
    raise NotImplementedError("dmew is not yet implemented.")    

def _mass_to_areas(w=None, n_verts=None) -> np.ndarray:
    mass = _process_vertex_areas(w=w, n_verts=n_verts)
    return np.asarray(mass.sum(axis=0)).ravel()

# TODO: consider adding dtype parameter to _process_vertex_areas for w=None case
def _process_vertex_areas(
        w: csc_matrix | np.ndarray | list | None, 
        n_verts: int | None = None
) -> csc_matrix:
    
    if w is None and n_verts is None:
        raise ValueError("Either w or n_verts must be provided.")
    
    elif w is None and n_verts is not None: # appease: pyright
        warn("Mass matrix not provided; assuming that area at each vertex is 1")
        output = diags(np.ones(n_verts), format='csc')

    elif isinstance(w, (np.ndarray, list)):        
        w_arr = np.asarray(w)
        if w_arr.ndim == 2 and w_arr.shape[0] == w_arr.shape[1]:
            output = w_arr
        elif w_arr.ndim == 1 or w_arr.shape[0] == 1 or w_arr.shape[1] == 1:
            output = diags(np.ravel(w_arr), format='csc')
        else: 
            raise ValueError(f"Mass matrix has invalid shape: {w_arr.shape} (should be square or vector).")

    elif isinstance(w, csc_matrix) and w.shape is not None: # appease: pyright
        if w.shape[0] == w.shape[1]: 
            output = w
        elif w.shape[0] == 1 or w.shape[1] == 1:
            output = diags(w.toarray().flatten(), format='csc')
        else: 
            raise ValueError(f"Sparse mass matrix has invalid shape: {w.shape} (should be square or vector).")
            
    else:
        raise TypeError("w must be a 1D array of vertex areas, a 2D mass matrix, or a sparse matrix.")

    output = csc_matrix(output)
    
    if output.shape is None: 
        raise ValueError("Mass matrix has undefined shape.")
    elif output.shape[0] != output.shape[1]:
        raise ValueError(f"Mass matrix must be square; got shape {output.shape}.")
    if n_verts is not None and output.shape != (n_verts, n_verts):
        raise ValueError(f"Mass matrix has invalid shape: {output.shape} (should be ({n_verts}, {n_verts})).")

    return output

# TODO
def compare_correlation_matrices(): 
    raise NotImplementedError("compare_correlation_matrices is not yet implemented.")
