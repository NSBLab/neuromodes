from __future__ import annotations
from collections.abc import Sequence, Iterable, Iterator, Sized
from dataclasses import dataclass
from itertools import islice
from functools import partial
from typing import Literal, TYPE_CHECKING, overload
from warnings import warn

import numpy as np
from scipy.stats import genpareto
from scipy.sparse import csr_matrix, csc_matrix, eye as speye
from scipy.interpolate import make_interp_spline

import neuromodes.glmw as glmw

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray

SeedLike = np.random.SeedSequence | int | None
StatLiteral = Literal['t', 'f', 'z']
PermutationLiteral = Literal["flip", "shuffle", "flip_shuffle"]
AlternativeLiteral = Literal['two-sided', 'greater', 'less']

@dataclass(frozen=True, slots=True)
class TailFitResult:
    """GPD tail-fit diagnostics aligned with `stat_map`.

    Attributes
    ----------
    n : NDArray[np.intp]
        Number of extreme null samples used for the accepted GPD fit,
        or ``-1`` where no GPD fit was performed or accepted.
    fit : NDArray[np.float64]
        Array of shape ``(*stat_map.shape, 3)`` containing ``(shape, threshold, scale)``,
        or ``NaN`` where unfitted. ``threshold`` is in the original scale of
        ``stat_map``. To evaluate tail probabilities
        directly, orient exceedances via ``sign = np.where(tail_dir, -1.0, 1.0)``
        and call ``scipy.stats.genpareto.sf(sign * (stat_map - fit[..., 1]), fit[..., 0], 0.0, fit[..., 2])``.
    pvalue : NDArray[np.float64]
        Anderson-Darling goodness-of-fit p-value for the accepted fit,
        or ``NaN`` where unfitted.
    """
    n: NDArray[np.intp]
    fit: NDArray[np.float64]
    pvalue: NDArray[np.float64]

# TODO rename all variables to be more descriptive
# TODO remove mbm_ from function names

# TODO P5: stat testing support 3d data so that this can be done without a loop
# Currently only supports one ttest at a time (1 contrast). For multiple contrasts, use ftestw/ANOVA.
# doing multiple ttests concurrently is dangerous in terms of multiple comparisons, so we don't want to encourage it.
# If you want to do multiple ttests, they have to be run separately and then corrected for multiple comparisons (e.g. FDR).
# If you want to see this functionality added, please open an issue on GitHub and we can discuss the best way to implement it.
# It would probably involve different tests (contrasts) being stacked along axis=2
def compute_stat_map(
    design: NDArray,     # (n_subjects, n_predictors)
    response: NDArray,   # (n_subjects, n_data)
    contrast: NDArray,   # (n_contrasts, n_predictors)
    statistic: StatLiteral,
) -> NDArray:            
    ident = csc_matrix(speye(response.shape[0], format='csc')) # pyright

    if statistic == 't':
        stat_map = glmw.ttestw(design, response, contrast, ident).statistic
    elif statistic == 'z':
        stat_map = glmw.ztestw(design, response, contrast, ident).statistic
    elif statistic == 'f':
        stat_map = glmw.ftestw(design, response, contrast, ident)[0]
    else:
        raise ValueError(f"Unsupported stat test: {statistic}")

    return stat_map 

# TODO consider moving these two functions to a more generic location
def _iter_spawned_rng(seed: SeedLike = None) -> Iterator[np.random.Generator]:
    """Yields an infinite, prefix-invariant stream of independent Generators."""
    rng = np.random.default_rng(seed)
    while True:
        yield rng.spawn(1)[0]

def _stack_arrays(
    arrays: Iterable[NDArray],
    n: int | None = None,
) -> NDArray:
    """
    Materializes `n` arrays from `arrays` along a new trailing axis (axis=-1).

    If `n` is None, `arrays` must be Sized and all elements are consumed.
    """
    # Validate inputs
    if n is None:
        if not isinstance(arrays, Sized):
            raise TypeError("n must be specified when Iterable is not Sized.")
        n = len(arrays)
    elif isinstance(arrays, Sized) and n > len(arrays):
        raise ValueError(f"n ({n}) exceeds the length of arrays ({len(arrays)}).")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")

    # Main loop
    i, out = -1, None
    for i, arr in enumerate(islice(arrays, n)):
        if out is None:
            out = np.empty((*arr.shape, n), dtype=arr.dtype, order="F")
        out[..., i] = arr

    # Validate outputs
    if i != n - 1 or out is None:
        raise ValueError(f"Expected {n} arrays, but iterable stopped after {i + 1}.")

    return out

# These could also use `def _stream(): for.... return _stream()` if desired
def iter_flip(
    n: int,
    seed: SeedLike = None,
) -> Iterator[csr_matrix]:
    """Lazily yields (n_subjects, n_subjects) diagonal sign-flip sparse matrices."""
    indptr = np.arange(n + 1, dtype=np.intp)
    indices = np.arange(n, dtype=np.intp)

    for rng in _iter_spawned_rng(seed):
        signs = rng.integers(0, 2, size=n, dtype=np.int8) * 2 - 1
        yield csr_matrix((signs, indices, indptr), shape=(n, n))

def iter_shuffle(
    n: int,
    seed: SeedLike = None,
) -> Iterator[csr_matrix]:
    """Lazily yields (n_subjects, n_subjects) row-permutation sparse matrices."""
    indptr = np.arange(n + 1, dtype=np.intp)
    signs = np.ones(n, dtype=np.int8)

    for rng in _iter_spawned_rng(seed):
        indices = rng.permutation(n)
        yield csr_matrix((signs, indices, indptr), shape=(n, n))

def iter_flip_shuffle(
    n: int,
    seed: SeedLike = None,
) -> Iterator[csr_matrix]:
    """Lazily yields (n_subjects, n_subjects) signed permutation sparse matrices."""
    indptr = np.arange(n + 1, dtype=np.intp)

    for rng in _iter_spawned_rng(seed):
        signs = rng.integers(0, 2, size=n, dtype=np.int8) * 2 - 1
        indices = rng.permutation(n)
        yield csr_matrix((signs, indices, indptr), shape=(n, n))

def iter_permute_design(
    design: NDArray,
    columns: NDArray,
    method: PermutationLiteral = 'flip_shuffle',
    seed: SeedLike = None,
) -> Iterator[NDArray]:
    """
    Lazily yields permuted copies of `design` of shape (n_subjects, n_predictors).

    Only columns selected by `columns` are transformed; nuisance columns
    remain untouched.
    """

    # Validate inputs
    if design.ndim != 2:
        raise ValueError(f"Design matrix must be 2D. Received shape: {design.shape}")

    if method == 'flip':
        operators = iter_flip
    elif method == 'shuffle':
        operators = iter_shuffle
    elif method == 'flip_shuffle':
        operators = iter_flip_shuffle
    else:
        raise ValueError(f"Unsupported permutation method: {method}")

    target_design = np.ascontiguousarray(design[:, columns])
    if target_design.size == 0: 
        raise ValueError("No columns selected for permutation. Please provide at least one column to permute.")

    # Main
    def _stream() -> Iterator[NDArray]:
        for op in operators(design.shape[0], seed):
            perm_design = design.copy()
            perm_design[:, columns] = op @ target_design
            yield perm_design
    return _stream()

def permute_design(
    design: NDArray,
    columns: NDArray,
    n_permutations: int,
    method: PermutationLiteral = 'flip_shuffle',
    seed: SeedLike = None,
) -> NDArray:
    """
    Materializes a 3D array of permuted copies of `design` of shape
    (n_subjects, n_predictors, n_permutations).

    Only columns selected by `columns` are transformed; nuisance columns
    remain untouched.

    If `n_permutations` is None, all permutations are generated (if finite).
    """
    stream = iter_permute_design(design, columns, method, seed)
    return _stack_arrays(stream, n=n_permutations)

def iter_null_stat_maps(
    design: NDArray,
    response: NDArray,
    contrast: NDArray | np.number | int | float,
    statistic: StatLiteral,
    method: PermutationLiteral = "flip_shuffle",
    seed: SeedLike = None,
) -> Iterator[NDArray]:
    """Lazily yields null statistic maps of shape (n_data,)."""
    # Convert contrast to target_columns 
    contrast_arr = np.asarray(contrast)
    if contrast_arr.ndim == 0:
        contrast_arr = np.atleast_2d(contrast_arr)
    if contrast_arr.ndim != 2:
        raise ValueError(f"Contrast must be a 2D array or a scalar. Received shape: {contrast_arr.shape}")
    target_cols = np.any(contrast_arr != 0, axis=0)
    if not np.any(target_cols):
        raise ValueError("Contrast must have at least one non-zero entry.")

    # Main
    def _stream() -> Iterator[NDArray]:
        for perm_design in iter_permute_design(design, target_cols, method, seed):
            yield compute_stat_map(perm_design, response, contrast_arr, statistic)
    return _stream()

def generate_null_stat_maps(
    design: NDArray,
    response: NDArray,
    contrast: NDArray | np.number | int | float,
    statistic: StatLiteral,
    n_permutations: int,
    method: PermutationLiteral = 'flip_shuffle',
    seed: SeedLike = None,
) -> NDArray:
    """
    Materializes a 2D null statistic matrix of shape (n_data, n_permutations).

    If `n_permutations` is None, all permutations are generated (if finite).
    """
    stream = iter_null_stat_maps(design, response, contrast, statistic, method, seed)
    return _stack_arrays(stream, n=n_permutations)

@overload
def compute_p_values(
        stat_map: NDArray[np.floating],
        stat_map_nulls: NDArray[np.floating],
        alternative: AlternativeLiteral = ..., 
        fit_trigger: int | None = ..., 
        fit_candidate_sizes: int | Sequence[int] = ..., 
        fit_alpha: float | None = ..., 
        *,
        return_tail: Literal[False] = False,
        return_fit: Literal[False] = False,
) -> NDArray[np.floating]: ...

@overload
def compute_p_values(
        stat_map: NDArray[np.floating],
        stat_map_nulls: NDArray[np.floating],
        alternative: AlternativeLiteral = ..., 
        fit_trigger: int | None = ..., 
        fit_candidate_sizes: int | Sequence[int] = ..., 
        fit_alpha: float | None = ..., 
        *,
        return_tail: Literal[True],
        return_fit: Literal[False] = False,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]: ...

@overload
def compute_p_values(
        stat_map: NDArray[np.floating],
        stat_map_nulls: NDArray[np.floating],
        alternative: AlternativeLiteral = ..., 
        fit_trigger: int | None = ..., 
        fit_candidate_sizes: int | Sequence[int] = ..., 
        fit_alpha: float | None = ..., 
        *,
        return_tail: Literal[False] = False,
        return_fit: Literal[True],
) -> tuple[NDArray[np.floating], TailFitResult]: ...

@overload
def compute_p_values(
        stat_map: NDArray[np.floating],
        stat_map_nulls: NDArray[np.floating],
        alternative: AlternativeLiteral = ..., 
        fit_trigger: int | None = ..., 
        fit_candidate_sizes: int | Sequence[int] = ..., 
        fit_alpha: float | None = ..., 
        *,
        return_tail: Literal[True],
        return_fit: Literal[True],
) -> tuple[NDArray[np.floating], NDArray[np.bool_], TailFitResult]: ...

# TODO FUTURE look at ways of vectorising this across vertices ?Newton method for MLE ?method of moments
# TODO FUTURE support observed_stat_map being 2D for alternative == two-sided ?recursion
# TODO P2 add tests for the data returned return_tail and return_fit
def compute_p_values(
        # TODO change names!
        stat_map: NDArray[np.floating], # (n_data,) or (n_data, n_obs) if alternative != 'two-sided'
        stat_map_nulls: NDArray[np.floating], # (n_data, n_permutations)
        alternative: AlternativeLiteral = 'two-sided', 
        fit_trigger: int | None = 9, # None to disable recalculation, 0 to only do something past all the nulls, 9 to match Knijenberg, n_permutations to do all
        fit_candidate_sizes: int | Sequence[int] = tuple(range(250, 2, -10)), # match Knijenburg
        fit_alpha: float | None = 0.05, # match Knijnenburg, but perhaps 0.10 is better?
        *,
        return_tail: bool = False,
        return_fit: bool = False,
) -> NDArray[np.floating] \
    | tuple[NDArray[np.floating], NDArray[np.bool_]] \
    | tuple[NDArray[np.floating], TailFitResult] \
    | tuple[NDArray[np.floating], NDArray[np.bool_], TailFitResult]: 

    # Note that fit_candidate_sizes and fit_trigger should be specified as 
    # integers (like in Knijnenberg et al. 2009), NOT as fractions (like in PALM).

    # 0. Input validation
    if is_vector_obs := stat_map.ndim == 1:
        stat_map = stat_map[:, None] # reshape to (n_data, 1) for consistent processing
    elif stat_map.ndim == 2:
        if alternative == 'two-sided':
            # If you would like this to be implemented, please leave a GitHub issue and we can have a look
            raise NotImplementedError("observedStatMap cannot be 2D for two-sided alternative. Please provide a 1D array of observed statistics.")
    else:
        raise ValueError(f"observedStatMap must be 1D or 2D, got shape {stat_map.shape}")

    if stat_map_nulls.ndim != 2:
        raise ValueError(f"statMapNull must be 2D, got shape {stat_map_nulls.shape}")
    if stat_map_nulls.shape[0] != stat_map.shape[0]:
        raise ValueError(f"statMapNull and observedStatMap must have the same number of rows. Got {stat_map_nulls.shape[0]} and {stat_map.shape[0]}.")
    n_permutations = stat_map_nulls.shape[1]

    # If we need to do tail estimation
    if fit_trigger is not None:
        # check fit_trigger
        if fit_trigger < 0:
            warn(f"fit_trigger ({fit_trigger}) is less than 0. No p-values will be recalculated.")
        if fit_trigger > n_permutations: # Don't warn for n_permutations (allow user to force recalculation))
            warn(f"fit_trigger ({fit_trigger}) is greater than n_permutations ({n_permutations}). All p-values will be recalculated.")

        # and check fit_candidate_sizes
        if isinstance(fit_candidate_sizes, int):
            fit_candidate_sizes = (fit_candidate_sizes,)
        if not isinstance(fit_candidate_sizes, Sequence):
            raise TypeError(f"fit_candidate_sizes must be an integer or iterable, got {type(fit_candidate_sizes)}")
        for n in fit_candidate_sizes:
            if not isinstance(n, int):
                raise TypeError(f"All values in fit_candidate_sizes must be integers, got {type(n)}")
            if n > n_permutations:
                warn(f"Some values in fit_candidate_sizes are greater than n_permutations ({n_permutations}). These will be ignored.")
            if n < 2:
                warn(f"Some values in fit_candidate_sizes are less than 2. These will be ignored.")
        nte_copy = [nte for nte in fit_candidate_sizes if 2 <= nte <= n_permutations]
        if fit_candidate_sizes and not nte_copy:
            warn(f"All values in fit_candidate_sizes ({fit_candidate_sizes}) are invalid. No p-values will be recalculated.")
    else: # for pyright
        nte_copy = []
    
    # 1. Get naive p values (P'_ecdf) and setup for calculation of P_gpd
    tail_counts, tail_dir = _count_tails(stat_map, stat_map_nulls, alternative)
    mask_2d = (tail_counts <= fit_trigger
            if fit_trigger is not None
            else np.zeros_like(tail_counts, dtype=bool))
    mask_1d = np.any(mask_2d, axis=1)

    # Orient observed statistics and null distributions so all tails act as upper/right tails
    factor = 1 - 2 * tail_dir[mask_1d].astype(int) # (n_recalc, n_obs)
    obs = stat_map[mask_1d] * factor  # (n_recalc, n_obs)
    nulls_nondesc = np.sort(stat_map_nulls[mask_1d, :] * factor[:,:1], axis=1)  # (n_recalc, n_permutations)

    fit_n = np.full(stat_map.shape, -1, dtype=np.intp)
    fit_params = np.full((*stat_map.shape, 3), np.nan, dtype=np.float64)
    fit_pvalue = np.full(stat_map.shape, np.nan, dtype=np.float64)

    # 2. Main loop: iterate over each row that needs recalculation
    # Try different numbers of tail elements for GPD fitting, until AD test passes
    for ob, nulls_row, mask_row, ii in zip(obs, nulls_nondesc, mask_2d[mask_1d], np.where(mask_1d)[0]):
        # TODO FUTURE get unique values of nulls_row[-nte_copy] in stable order
        # as we only need to test each unique threshold
        for nte in nte_copy: 
            # TODO FUTURE investigate whether Knijnenburg midpoint threshold or exact threshold is better
            # Note that if using the exact threshold, the zero elements can (?should) be removed from AD test

            # Find the threshold and the exceedances for the current tail size estimation
            thr, exceedances = _get_threshold_and_exceedances(nulls_row, nte)
            if np.var(exceedances) == 0.0:
                if nte == np.max(nte_copy):
                    warn(f"All tail values are identical for vertex {ii}.")
                    break
                else:
                    continue

            # This will only be used after the fitting, but we compute it early as we may need to early exit
            ob_mask = mask_row & (ob > thr) 
            if not np.any(ob_mask):
                if nte == np.max(nte_copy):
                    warn(f"Observed statistic {ob} is below the minimum threshold {thr} for vertex {ii}.")
                    break
                else: 
                    continue

            # Fit GPD with location fixed to threshold t
            # Could use try-catch to continue to other nte/vertices if one fit errors out
            # TODO FUTURE: consider how to return these fit values and AD stats/result to user
            fit, adp = _fit_and_ad_exceedances(exceedances, fit_alpha)
            if fit[0] <= -0.5 or fit[0] > 1.0 or fit[2] <= 0.0: # shape parameter k <= -0.5 or scale parameter a <= 0.0
                continue
            if fit_alpha is not None and adp <= fit_alpha: # next iter if fit is poor
                continue

            tail_counts[ii, ob_mask] = exceedances.size * genpareto.sf(ob[ob_mask] - thr, *fit)
            fit_n[ii, ob_mask] = nte
            fit_params[ii, ob_mask, 0] = fit[0]
            fit_params[ii, ob_mask, 1] = np.where(tail_dir[ii, ob_mask], -thr, thr)
            fit_params[ii, ob_mask, 2] = fit[2]
            fit_pvalue[ii, ob_mask] = adp

            mask_row[ob_mask] = False
            if not np.any(mask_row):
                break

        if np.any(mask_row):
            warn(f"GPD fitting failed for vertex {ii}. Using non-parametric p-value.")

    # 3. Output
    p_values = _counts_to_pvals(tail_counts, n_permutations, alternative)

    if is_vector_obs:
        sqz1 = partial(np.squeeze, axis=1)
        p_values = sqz1(p_values)
        tail_dir = sqz1(tail_dir)
        fit_n = sqz1(fit_n)
        fit_params = sqz1(fit_params)
        fit_pvalue = sqz1(fit_pvalue)

    fit_result = TailFitResult(n=fit_n, fit=fit_params, pvalue=fit_pvalue)

    if not return_tail and not return_fit:
        return p_values
    elif return_tail and not return_fit:
        return p_values, tail_dir
    elif not return_tail and return_fit:
        return p_values, fit_result
    else: # return_tail and return_fit
        return p_values, tail_dir, fit_result

def _count_tails(
        obs_2d: NDArray[np.floating], 
        nulls_2d: NDArray[np.floating], 
        alternative: AlternativeLiteral
) -> tuple[NDArray[np.integer], NDArray[np.bool_]]: 
    obs_3d = obs_2d[:, np.newaxis, :]
    null_3d = nulls_2d[:, :, np.newaxis]
    geq = np.sum(null_3d >= obs_3d, axis=1).astype(float)
    leq = np.sum(null_3d <= obs_3d, axis=1).astype(float)
    if alternative == 'greater':
        tail_counts = geq.copy()
        tail_dir = np.zeros_like(geq, dtype=bool)
    elif alternative == 'less':
        tail_counts = leq.copy()
        tail_dir = np.ones_like(leq, dtype=bool)
    elif alternative == 'two-sided':
        tail_counts = np.minimum(geq, leq).copy() # will be doubled at the end
        tail_dir = leq < geq
    else:
        raise ValueError(f"Unsupported alternative hypothesis: {alternative}")
    return tail_counts, tail_dir

def _counts_to_pvals(
        tail_counts: NDArray[np.integer], 
        n_permutations: int, 
        alternative: AlternativeLiteral
) -> NDArray[np.floating]:
    if alternative == 'two-sided':
        tail_counts = 2 * tail_counts
    return np.clip(tail_counts / n_permutations, 0.0, 1.0)

# This function can be used (and perhaps formalised) to get p-values without any tail fitting
def _naive_pvals(
        obs_2d: NDArray[np.floating], 
        nulls_2d: NDArray[np.floating], 
        alternative: AlternativeLiteral
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    tail_counts, tail_dir = _count_tails(obs_2d, nulls_2d, alternative)
    return _counts_to_pvals(tail_counts, nulls_2d.shape[1], alternative), tail_dir

def _get_threshold_and_exceedances(
        nulls_nondesc: NDArray[np.floating], 
        n_tail: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    tidx = np.searchsorted(nulls_nondesc, nulls_nondesc[-n_tail], side='left')
    if tidx != 0: 
        thr = (nulls_nondesc[tidx-1] + nulls_nondesc[tidx]) / 2.0
    else: # tidx = 0: take a half step before the first value (from PALM)
        thr = (3 * nulls_nondesc[0] - nulls_nondesc[1]) / 2.0
    exceedances = nulls_nondesc[tidx:] - thr # same as nulls_nondesc[nulls_nondesc > thr] 
    return thr, exceedances

def _fit_and_ad_exceedances(
        exceedences: NDArray[np.floating], 
        fit_alpha: float | None = 0.05
) -> tuple[tuple[float, float, float], float]:
    fit = genpareto.fit(exceedences, floc=0.0) # floc forced to 0
    if fit_alpha is not None: 
        adp = _andersondarling(
            np.sort(genpareto.cdf(exceedences, *fit)), 
            k=-fit[0])[0] # Choulakian & Stephens parameterize shape parameter k = -c
    else: 
        adp = np.nan # adp = 1.0 # could instead use dummy p value
    return fit, adp

def _andersondarling(
        z_ascending: NDArray[np.floating], 
        k: float
) -> tuple[float, np.floating]:
    """
    Compute the Anderson-Darling statistic and return an
    approximated p-value based on the tables provided in:
    * Choulakian V, Stephens M A. Goodness-of-Fit Tests
      for the Generalized Pareto Distribution. Technometrics.
      2001;43(4):478-484.
    
    Note that z should be sorted in asvending order and should be 0 < z < 1.
    Note that k should be within the range of the table provided in the paper -0.9 < k < 0.5.
    """
    # Constants
    # Table 2 of the paper (Case 3: a and k unknown, bold values)
    K_TABLE = np.array([-0.90, -0.50, -0.20, -0.10,  0.00,  0.10,  0.20,  0.30,  0.40,  0.50])
    P_TABLE = np.array([0.500, 0.250, 0.100, 0.050, 0.025, 0.010, 0.005, 0.001])
    A2_TABLE = np.array([
        [0.339, 0.471, 0.641, 0.771, 0.905, 1.086, 1.226, 1.559],
        [0.356, 0.499, 0.685, 0.830, 0.978, 1.180, 1.336, 1.707],
        [0.376, 0.534, 0.741, 0.903, 1.069, 1.296, 1.471, 1.893],
        [0.386, 0.550, 0.766, 0.935, 1.110, 1.348, 1.532, 1.966],
        [0.397, 0.569, 0.796, 0.974, 1.158, 1.409, 1.603, 2.064],
        [0.410, 0.591, 0.831, 1.020, 1.215, 1.481, 1.687, 2.176],
        [0.426, 0.617, 0.873, 1.074, 1.283, 1.567, 1.788, 2.314],
        [0.445, 0.649, 0.924, 1.140, 1.365, 1.672, 1.909, 2.475],
        [0.468, 0.688, 0.985, 1.221, 1.465, 1.799, 2.058, 2.674],
        [0.496, 0.735, 1.061, 1.321, 1.590, 1.958, 2.243, 2.922]
    ])

    # Input validation
    z = np.asarray(z_ascending)
    if z.ndim != 1: 
        raise ValueError("Input array 'z' must be one-dimensional.")
    if np.any(np.diff(z) < 0): # non-descending (ties ok)
        raise ValueError("Input array 'z' must be sorted in ascending order.")

    if k < np.min(K_TABLE) or k > np.max(K_TABLE):
        warn(
            f"Shape parameter k={k} is outside the table bounds [{np.min(K_TABLE)}, {np.max(K_TABLE)}]. "
            "Critical values will be extrapolated and may be unreliable.",
            UserWarning
        )
    
    # Main computation
    n = z.size
    a2 = -n - np.mean( np.log( z*(1-z[::-1]) ) * np.arange(1,2*n,2) )

    k_to_a2 = make_interp_spline(K_TABLE, A2_TABLE, k=1, axis=0)   # Interpolate critical values for k
    a2_to_p = make_interp_spline(k_to_a2(k), P_TABLE, k=1, axis=0) # and then p-values for A2
    p = a2_to_p(a2)
    
    return np.clip(p, 0.0, 1.0), np.clip(a2, 0.0, np.inf)

# # These functions may be useful if freedman-lane permutations are incorporated in the future, 
# # or for regression testing (comparing with MATLAB using a known set of permutated designs)
# def iter_permuted_design_stat_maps(
#     permuted_designs: Iterable[NDArray],
#     response: NDArray,
#     contrast: NDArray,
#     statistic: StatLiteral,
# ) -> Iterator[NDArray]:
#     for perm_design in permuted_designs:
#         yield stat_map(perm_design, response, contrast, statistic)

# def compute_permuted_design_stat_maps(
#     permuted_designs: Iterable[NDArray],
#     response: NDArray,
#     contrast: NDArray,
#     statistic: StatLiteral,
#     n_permutations: int | None = None,
# ) -> NDArray:
#     stream = iter_permuted_design_stat_maps(permuted_designs, response, contrast, statistic)
#     return _stack_arrays(stream, n=n_permutations)
