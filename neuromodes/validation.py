from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn
from collections.abc import Sequence

import numpy as np
from scipy.sparse import diags, spmatrix, csc_matrix

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

    target_nverts = n_verts

    for name, arr in kwargs.items():
        if arr is None:
            continue

        if not hasattr(arr, "shape") or len(arr.shape) == 0:
            raise TypeError(
                f"Argument '{name}' must be an array or matrix with a valid .shape attribute; "
                f"got type {type(arr).__name__}."
            )

        curr_height = int(arr.shape[0])
        if target_nverts is None:
            target_nverts = curr_height

        if curr_height != target_nverts:
            msg = (
                f"Spatial dimension mismatch in argument '{name}': got shape[0] = {curr_height}, "
                f"but expected {target_nverts}"
            )
            if n_verts is None:
                msg += " (inferred from preceding inputs)."
            else:
                msg += " (specified by n_verts)."
            raise ValueError(msg)

    if target_nverts is None:
        raise ValueError("No valid arrays were provided to infer n_verts, and n_verts was not specified.")
    return target_nverts

# TODO P1 this function!
def _check_n_modes(
        n_modes_max = None, 
        mode_counts = None, 
        mode_ids = None,
        emodes = None, 
        evals = None,
):
    if emodes is not None and evals is not None:
        if emodes.shape[1] != evals.shape[0]: # TODO P1 change wording of this error and above (and tests)
            raise ValueError(f"must have shape Number of modes in emodes ({emodes.shape[1]}) does not match number of evals ({evals.shape[0]}).")
        n_modes_max = emodes.shape[1]
    # raise NotImplementedError("This function is not yet implemented. It will validate that the number of modes specified by mode_counts or mode_ids is consistent with the number of modes in emodes and evals.")


# check square
# check csc
def _check_mass(mass, 
                lump = None
                ):
    raise NotImplementedError("This function is not yet implemented. It will validate that the mass matrix is consistent with the number of vertices in emodes and evals.")
def _check_stiffness(stiffness,
                            ):
        raise NotImplementedError("This function is not yet implemented. It will validate that the stiffness matrix is consistent with the number of vertices in emodes and evals.")

def _check_emodes_evals(emodes,
                  evals,
                  mass, 
                  stiffness
                  ):
    # check emodes.T @ mass @ emodes == I
    # check emodes.T @ stiffness @ emodes == diag(evals)
    # check first emode (only) is constant
    # check first eval (only) is zero 
    raise NotImplementedError("This function is not yet implemented. It will validate that the eigenmodes are consistent with the mass and stiffness matrices and the eigenvalues.")



def _create_parcellation_matrix(
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


def _mass_to_areas(
    mass: csc_matrix | NDArray[np.floating] | None = None,
    n_verts: int | None = None,
    keepdims: bool = False
) -> NDArray[np.floating]:
    mass = np.asarray(_create_mass_matrix(mass=mass, n_verts=n_verts).sum(axis=1))
    if not keepdims:
        mass = mass.squeeze(axis=1)
    return mass

# TODO P3: consider adding dtype parameter to _create_mass_matrix for w=None case
def _create_mass_matrix(
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
    n_modes_max: int | np.integer
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


# TODO consider raising an error in these functions
# TODO consider returning the cleaned data of a specific type 
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
