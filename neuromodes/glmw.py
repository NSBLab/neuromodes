from __future__ import annotations
from types import SimpleNamespace
from typing import TYPE_CHECKING
import numpy as np
import scipy.stats as sps

from neuromodes.eigen import EigenData
from neuromodes.stats import ssqw, solvew

if TYPE_CHECKING:
    from typing import Literal
    from scipy.sparse import csc_matrix

# TODO check performace on unequal variance data (Welch's t-test/ANOVA)

def _glmw(
    design: np.ndarray, # (n_vertices, n_predictors)
    response: np.ndarray, # (n_vertices, n_data)
    contrast: np.ndarray, # (n_contrasts, n_predictors)
    mass: csc_matrix, # (n_vertices, n_vertices)
):
    # 0. Input validation
    ved = EigenData(data=(design, response), mass=mass)
    design, response = ved.data
    mass = ved.mass

    if design.ndim != 2:
        raise ValueError(f"Design matrix must be 2D. Received shape: {design.shape}.")
    if response.ndim != 2:
        raise ValueError(f"Response matrix must be 2D. Received shape: {response.shape}.")
    if design.shape[0] != response.shape[0]:
        raise ValueError(
            f"Number of rows in design and response matrices must match. "
            f"Received design shape: {design.shape}, response shape: {response.shape}."
        )
    
    contrast = np.asarray(contrast)
    if contrast.ndim == 0:
        contrast = np.atleast_2d(contrast)
    if contrast.ndim != 2:
        raise ValueError("Contrast must be a 2D array or a scalar. Received shape: {contrast.shape}.")
    if contrast.shape[1] != design.shape[1]:
        raise ValueError(
            f"Contrast matrix must have the same number of columns as design matrix. "
            f"Received contrast shape: {contrast.shape}, design shape: {design.shape}."
        )

    # TODO check use of solve here? what if input is deficient rank? overdetermined design matrix?
    # 1. Fit GLM. TODO Should probably return beta 
    beta = solvew(design, response, mass) # Shape: (n_predictors, n_data)
    
    # 2. Shared Data Metrics
    df = design.shape[0] - np.linalg.matrix_rank(design) # TODO check this is correct for different mass matrices

    ssr = ssqw(response - design.dot(beta), mass)
    sigma_sq = ssr / df # finish ssr
    
    # 3. Contrast Analysis
    D = contrast.dot(beta)  # Shape: (n_contrasts, n_data)

    A = (mass.dot(design)).T.dot(design)
    G = contrast.dot(np.linalg.solve(A, contrast.T))  # Shape: (n_contrasts, n_contrasts)
    V = np.linalg.solve(G, D)

    Q = np.maximum(np.sum(D * V, axis=0), 0) / sigma_sq
    
    return Q, D, df  

# TODO P5 consider writing wrapper for ttest_1samp, f_oneway etc. 
def ttestw(
    design: np.ndarray, 
    response: np.ndarray, 
    contrast: np.ndarray, 
    mass: csc_matrix, 
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
) -> SimpleNamespace:
    """Weighted t-test for one or two samples."""

    if  np.atleast_2d(contrast).shape[0] != 1:
        raise ValueError(
            f"ttestw requires exactly 1 contrast (1 degree of freedom). "
            f"Received {contrast.shape[0]}. Use ftestw for multi-row contrasts."
        )

    Q, D, df = _glmw(design, response, contrast, mass)
    
    # Recover directional t-statistic: t = sign(D) * sqrt(Q)
    statistic = np.sign(D.flatten()) * np.sqrt(Q)

    if alternative == 'two-sided':
        pvalue = 2 * sps.t.sf(np.abs(statistic), df=df)
    elif alternative == 'greater':
        pvalue = sps.t.sf(statistic, df=df)
    elif alternative == 'less':
        pvalue = sps.t.cdf(statistic, df=df)
    else:
        raise ValueError(f"Invalid alternative: {alternative}. Expected 'two-sided', 'greater', or 'less'.")
       
    # return a SimpleNamespace for compatibility with sps.ttest_*
    # (sps returns a TTestResult object with statistic, pvalue, and df attributes)
    # we don't return a confidence_interval method (?yet)
    return SimpleNamespace(
        statistic=statistic, 
        pvalue=pvalue,
        df=df
    )

# TODO add tests for this
def ztestw(
    design: np.ndarray, 
    response: np.ndarray, 
    contrast: np.ndarray, 
    mass: csc_matrix, 
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
) -> SimpleNamespace:
    """Weighted z-test derived from ttestw."""
    res = ttestw(design, response, contrast, mass, alternative=alternative)
    
    # Convert t-statistic to Z-score using probability integral transform
    p = sps.t.logsf(np.abs(res.statistic), df=res.df) # log p half
    # statistic = np.sign(res.statistic) * sps.norm.isf(np.exp(p))

    # TODO confirm that we can restrict to scipy 1.12+
    from scipy.special import ndtri_exp
    statistic = np.sign(res.statistic) * -ndtri_exp(p)
    
    return SimpleNamespace(
        statistic=statistic,
        pvalue=res.pvalue,
        df=res.df
    )

def ftestw(
    design: np.ndarray, 
    response: np.ndarray, 
    contrast: np.ndarray, 
    mass: csc_matrix
) -> tuple:
    """Weighted F-test for one-way ANOVA and ANCOVA."""
    Q, _, df = _glmw(design, response, contrast, mass)
    
    # Compute omnibus F-statistic: F = Q / rank(C)
    rank_C = np.linalg.matrix_rank(contrast)
    statistic = Q / rank_C
    pvalue = sps.f.sf(statistic, dfn=rank_C, dfd=df)
    
    # same return as f_oneway
    return statistic, pvalue
