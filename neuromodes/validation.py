from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn
from collections.abc import Sequence

import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.csgraph import connected_components

if TYPE_CHECKING:
    from typing import Any, TypeAlias, TypeGuard

    from numpy.typing import NDArray

    ModeCountInput: TypeAlias = Sequence[int] | int
    ModeIdInput: TypeAlias = Sequence[Sequence[int]] | Sequence[int]

def _check_n_verts(
    n_verts: int | np.integer | None = None,
    **kwargs
) -> int:
    """Validate that all supplied matrices share the same spatial dimension (axis 0).

    Parameters
    ----------
    n_verts : int, np.integer, or None, optional
        Expected number of vertices. If provided, all matrices in kwargs must match this.
        If None, the spatial dimension is inferred from the first non-None matrix.
    **kwargs : array-like or None
        Named arrays/matrices (e.g. ``data_a=X, data_b=Y, mass=M``). Arguments set to
        ``None`` are automatically skipped.

    Returns
    -------
    int or None
        The inferred or validated number of vertices (``shape[0]``), or ``None`` if
        no non-None matrices were provided and ``n_verts`` was not specified.

    Raises
    ------
    TypeError
        If ``n_verts`` is a boolean/non-integer, or if an argument lacks a ``shape`` attribute.
    ValueError
        If ``n_verts <= 0``, or if any matrix's ``shape[0]`` fails to match ``n_verts``.
    """
    if n_verts is not None:
        if not _is_positive_integer(n_verts):
            raise ValueError(f"n_verts must be a positive integer; got {n_verts} of type {type(n_verts).__name__}.")

    for name, arr in kwargs.items():
        if arr is None:
            continue

        if not hasattr(arr, "shape") or len(arr.shape) == 0:
            raise TypeError(
                f"Argument '{name}' must be an array or matrix with a valid .shape attribute; "
                f"got type {type(arr).__name__}."
            )

        curr_height = int(arr.shape[0])
        if n_verts is None:
            n_verts = curr_height

        if curr_height != n_verts:
            msg = (
                f"Spatial dimension mismatch in argument '{name}': got shape[0] = {curr_height}, "
                f"but expected {n_verts}"
            )
            if n_verts is None:
                msg += " (inferred from preceding inputs)."
            else:
                msg += " (specified by n_verts)."
            raise ValueError(msg)

    if n_verts is None:
        raise ValueError("No valid arrays were provided to infer n_verts, and n_verts was not specified.")
    return n_verts

def _check_n_modes(
        emodes: NDArray | None = None,
        evals: NDArray | None = None,
        n_modes: int | None = None, # the number of modes you are seeking to get from the emodes/evals
) -> int:
    if emodes is not None and evals is not None:
        if emodes.shape[1] != evals.shape[0]: # TODO P1 change wording of this error and above (and tests)
            raise ValueError(f"emodes ({emodes.shape[1]}) and evals ({evals.shape[0]}) must have shapes that are compatible")
    if n_modes is None:
        if emodes is not None:
            n_modes = emodes.shape[1]
        elif evals is not None:
            n_modes = evals.shape[0]
    if n_modes is None: 
        raise ValueError("Unable to determine n_modes; please provide at least one of n_modes, emodes, or evals.")
    if emodes is not None and emodes.shape[1] < n_modes:
        raise ValueError(f"emodes ({emodes.shape[1]}) must have at least n_modes ({n_modes}) columns")
    if evals is not None and evals.shape[0] < n_modes:
        raise ValueError(f"evals ({evals.shape[0]}) must have at least n_modes ({n_modes}) entries")
    return n_modes

def _check_csc(
    mat: csc_matrix,
    name: str = "Input",
) -> None:
    if not isinstance(mat, csc_matrix):
        raise TypeError(f"{name} must be a scipy sparse csc_matrix; got {type(mat)}.")

def _check_symmetric(
    mat: csc_matrix,
    name: str = "Input",
    rtol: float = 1e-8,
    atol: float = 1e-12,
) -> None:
    if mat.shape is None or mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} matrix must be square; got shape {mat.shape}.")

    diff = mat - mat.T
    if diff.nnz > 0:
        max_entry = float(np.max(np.abs(mat.data))) if mat.nnz > 0 else 0.0
        if float(np.max(np.abs(diff.data))) > atol + rtol * max_entry:
            raise ValueError(
                f"{name} matrix must be symmetric within rtol={rtol}, atol={atol}."
            )

def _check_one_component(
    mat: csc_matrix,
    name: str = "Input",
) -> None:
    n_components = connected_components(mat, directed=False, return_labels=False)
    if n_components != 1:
        raise ValueError(
            f"{name} matrix must have exactly 1 connected component; got {n_components}."
        )

def _is_diagonal_dominant(
    sqsymmatrix: csc_matrix,
    strict: bool = False,
    rtol: float = 1e-8,
    atol: float = 1e-12,
) -> bool:
    diag = sqsymmatrix.diagonal()
    col_sum = np.asarray(abs(sqsymmatrix).sum(axis=0)).ravel()
    off_diag_sum = col_sum - np.abs(diag)
    margin = atol + rtol * np.abs(diag)

    if strict:
        return bool(np.all(diag > margin) and np.all(diag - off_diag_sum > margin))
    return bool(np.all(diag >= -margin) and np.all(diag - off_diag_sum >= -margin))

def _check_mass(
    mass: csc_matrix,
    lump: bool | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-12,
) -> None:
    _check_csc(mass, "Mass")
    _check_symmetric(mass, "Mass", rtol=rtol, atol=atol)

    if not _is_diagonal_dominant(mass, strict=False, rtol=rtol, atol=atol):
        raise ValueError("Mass matrix is not positive semi-definite (diagonally dominant).")

    diag = mass.diagonal()
    is_lumped = np.count_nonzero(np.abs(mass.data) > atol) == np.count_nonzero(np.abs(diag) > atol)

    if lump is not None:
        if not lump and is_lumped:
            raise ValueError("Mass matrix expected not to be lumped, but it is lumped.")
        if lump and not is_lumped:
            raise ValueError("Mass matrix expected to be lumped, but it is not lumped.")

    if not is_lumped:
        _check_one_component(mass, "Mass")
        col_sum = np.asarray(mass.sum(axis=0)).ravel()
        if not np.allclose(col_sum, 2.0 * diag, rtol=rtol, atol=atol):
            raise ValueError(
                "Unlumped mass matrix column sums must equal twice the diagonal."
            )

    if np.any(np.isclose(diag, 0.0, atol=atol)):
        warn(
            "Mass matrix diagonal contains values close to zero; "
            "this may lead to numerical instability. Are any vertices isolated?",
            UserWarning,
        )

def _check_stiffness(
    stiffness: csc_matrix,
    rtol: float = 1e-8,
    atol: float = 1e-12,
) -> None:
    _check_csc(stiffness, "Stiffness")
    _check_symmetric(stiffness, "Stiffness", rtol=rtol, atol=atol)
    _check_one_component(stiffness, "Stiffness")

    if np.any(stiffness.diagonal() < -atol):
        raise ValueError("Stiffness matrix has negative diagonal entries (not SPSD).")

    col_sum = np.asarray(stiffness.sum(axis=0)).ravel()
    if not np.allclose(col_sum, 0.0, atol=atol):
        raise ValueError("Stiffness matrix column sums must equal zero.")

    if not _is_diagonal_dominant(stiffness, strict=False, rtol=rtol, atol=atol):
        warn(
            "Stiffness matrix is not diagonally dominant (e.g. non-Delaunay mesh with obtuse triangles); "
            "positive semi-definiteness is not guaranteed.",
            UserWarning,
        )

def _check_emodes_evals(
    emodes: np.ndarray,
    evals: np.ndarray,
    mass: csc_matrix,
    stiffness: csc_matrix,
    rtol: float = 1e-8,
    atol: float = 1e-12,
) -> None:

    n_modes = _check_n_modes(emodes=emodes, evals=evals)
    _check_n_verts(emodes=emodes, mass=mass, stiffness=stiffness)
    
    # 1. Check M-orthonormality: emodes.T @ mass @ emodes == I
    gram_mass = emodes.T @ (mass @ emodes)
    if not np.allclose(gram_mass, np.eye(n_modes), rtol=rtol, atol=atol):
        raise ValueError("Eigenmodes are not M-orthonormal (emodes.T @ mass @ emodes != I).")

    # 2. Check S-orthogonality / Rayleigh-Ritz diagonalization: emodes.T @ stiffness @ emodes == diag(evals)
    gram_stiff = emodes.T @ (stiffness @ emodes)
    if not np.allclose(gram_stiff, np.diag(evals), rtol=rtol, atol=atol):
        raise ValueError(
            "Eigenmodes do not diagonalize stiffness (emodes.T @ stiffness @ emodes != diag(evals))."
        )

    # 3. Check first eigenmode (and only the first) is constant
    mode0 = emodes[:, 0]
    if not np.allclose(mode0, np.mean(mode0), rtol=rtol, atol=atol):
        raise ValueError("First eigenmode must be constant across all vertices.")

    for j in range(1, n_modes):
        mode_j = emodes[:, j]
        if np.allclose(mode_j, np.mean(mode_j), rtol=rtol, atol=atol):
            raise ValueError(f"Eigenmode {j} is constant; only the first eigenmode may be constant.")

    # 4. Check first eigenvalue (and only the first) is zero
    if not np.isclose(evals[0], 0.0, atol=atol):
        raise ValueError(f"First eigenvalue must be zero; got {evals[0]}.")

    if np.any(np.isclose(evals[1:], 0.0, atol=atol)):
        raise ValueError("Only the first eigenvalue may be zero; all others must be positive.")

    # 5. Check eigenvalues are non-decreasing
    if np.any(np.diff(evals[1:]) < 0):
        raise ValueError("Eigenvalues must be sorted in non-decreasing order.")
    
def _process_parcellation_matrix(
    parcellation: csc_matrix | NDArray[np.integer], 
    mask: NDArray[np.bool] | None = None
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

    if parcellation.shape is None: 
        raise ValueError("Parcellation map has undefined shape.")
    n_verts = _check_n_verts(parcellation=parcellation, mask=mask)
    mask = mask if mask is not None else np.ones(n_verts, dtype=bool)

    if parcellation.ndim == 2:
        parc_mat = csc_matrix(parcellation[mask])
    elif parcellation.ndim == 1: # will be dense
        parc_mat = csc_matrix(
            (np.ones(mask.sum()),
            (np.arange(n_verts)[mask], parcellation[mask])),
            shape=(n_verts, (parcellation.max())+1)
            )
    else: 
        raise ValueError(f"Parcellation map must be 1D or 2D; got {parcellation.ndim}D.")

    return parc_mat

def _process_area_vector(
    mass: csc_matrix | NDArray[np.floating] | None = None,
    n_verts: int | None = None, 
    dtype: np.dtype | None = None,
    keepdims: bool = False
) -> NDArray[np.floating]:
    areas = np.asarray(_process_mass_matrix(mass=mass, n_verts=n_verts, dtype=dtype).sum(axis=1))
    if not keepdims:
        areas = areas.squeeze(axis=1)
    return areas

def _process_mass_matrix(
    mass: csc_matrix | NDArray[np.floating] | None = None,
    n_verts: int | None = None, 
    dtype: np.dtype | None = None
) -> csc_matrix:
    
    if mass is None:
        if n_verts is None:
            raise ValueError("Either mass or n_verts must be provided.")
        else:
            warn("Mass matrix not provided; assuming that area at each vertex is 1", UserWarning)
            output = csc_matrix(diags(np.ones(n_verts, dtype=dtype)))

    elif mass.ndim == 2 and mass.shape is not None: # appease: pyright 
        if mass.shape[0] != mass.shape[1]:
            raise ValueError(f"2D mass matrix must be square; got shape {mass.shape}.")
        output = csc_matrix(mass)

    elif isinstance(mass, np.ndarray):
        if mass.ndim == 1:
            output = csc_matrix(diags(mass))
        else: 
            raise ValueError(f"Dense mass matrix has invalid ndim: {mass.shape} (should be 1D or 2D).")

    else: 
        raise TypeError(f"Mass matrix must be a scipy sparse matrix or numpy dense array; got {type(mass)}.")
    
    if output.shape is None: 
        raise ValueError("Mass matrix has undefined shape.")
    elif output.shape[0] != output.shape[1]:
        raise ValueError(f"Mass matrix must be square; got shape {output.shape}.")
    if n_verts is not None and output.shape != (n_verts, n_verts):
        raise ValueError(f"Mass matrix has invalid shape: {output.shape} (should be ({n_verts}, "
                         f"{n_verts})).")

    return output

def _process_mode_ids(
    mode_counts: ModeCountInput | None,
    mode_ids: ModeIdInput | None,
    n_modes_max: int | np.integer # the max number of modes that the emodes/evals contain
) -> tuple[list[list[int]], bool]:
    """
    Process mode_counts and mode_ids into a consistent format for use in decompose/reconstruct.
    """
    # mode_counts is just shorthand for mode_ids
    # If mode_counts is provided, reformat into mode_ids
    if mode_counts is not None and mode_ids is not None:
        raise UserWarning("Both mode_counts and mode_ids provided; mode_counts will be ignored.")

    # We don't do squeeze_output = len(output) == 1 to have better type checking
    # ie determine squeeze_output based on the input type rather than the output length
    if _is_integer_sequence(mode_ids):
        output = [list(mode_ids),]
        squeeze_output = True
    elif _is_integer_sequence_of_sequences(mode_ids):
        output = [list(mids) for mids in mode_ids]
        squeeze_output = False
    elif mode_ids is not None:
        raise TypeError(f"mode_ids must be of type Sequence[int] or Sequence[Sequence[int]]; got {type(mode_ids)}.")
    elif mode_counts is None:
        output = [list(range(n_modes_max)),]
        squeeze_output = True
    elif _is_integer(mode_counts):
        output = [list(range(mode_counts)),]
        squeeze_output = True
    elif _is_integer_sequence(mode_counts):
        output = [list(range(mc)) for mc in mode_counts]
        squeeze_output = False
    else: 
        raise TypeError(f"mode_counts must be of type int or Sequence[int]; got {type(mode_counts)}.")

    for i in output:
        if len(i) == 0:
            raise ValueError("mode_counts/mode_ids must not be empty.")
        for j in i: 
            if not _is_integer(j):
                raise TypeError(f"mode_counts/mode_ids must be integers; got {j} of type {type(j)}.")
            if j < 0 or j >= n_modes_max:
                raise ValueError(f"mode_counts/mode_ids must be in the range [0, n_modes = {n_modes_max}).")

    return output, squeeze_output

def _is_nonstring_sequence(x) -> TypeGuard[Sequence[Any]]:
    """
    Check if x is an sequence of numeric types (int or float).
    """
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes))

def _is_integer(x) -> TypeGuard[int]:
    """
    Check if x is an integer or an sequence of integers.
    """
    return isinstance(x, (int, )) and not isinstance(x, bool) 

def _is_positive_integer(x) -> TypeGuard[int]:
    """
    Check if x is a positive integer.
    """
    return _is_integer(x) and x > 0

def _is_nonnegative_integer(x) -> TypeGuard[int]:
    """
    Check if x is a non-negative integer.
    """
    return _is_integer(x) and x >= 0

def _is_integer_sequence(x) -> TypeGuard[Sequence[int]]:
    """
    Check if x is an sequence of integers.
    """
    return _is_nonstring_sequence(x) and all(_is_integer(i) for i in x)

def _is_integer_sequence_of_sequences(x) -> TypeGuard[Sequence[Sequence[int]]]:
    """
    Check if x is an sequence of sequences of integers.
    """
    return _is_nonstring_sequence(x) and all(_is_integer_sequence(i) for i in x)
