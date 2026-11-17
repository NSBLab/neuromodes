"""Functions for generating hierarchical parcellations of a surface or volume mesh."""

from __future__ import annotations

from heapq import heappop, heappush

import numpy as np
from lapy import Solver, TetMesh, TriaMesh
from numpy.typing import NDArray

from neuromodes.mesh import check_surf, check_vol, mask_mesh


def _queue_child_or_finalize(
    geometry: TriaMesh | TetMesh,
    child_mask: NDArray[np.bool_],
    heap: list,
    parcel_masks: list[NDArray[np.bool_]],
    counter: int,
    seed: int | None,
) -> int:
    """Queue a splittable child or record it as one terminal parcel."""
    if not np.any(child_mask):
        # Empty mask, cannot split
        return counter
    
    min_vertices = (3 if isinstance(geometry, TriaMesh) else 4)
    if child_mask.sum() < min_vertices:
        # Cannot split the atom (no complete cells)
        parcel_masks.append(child_mask)
        return counter
    
    try:
        submesh, child_mapping = mask_mesh(
            geometry, child_mask, return_mapping=True
        )
    except ValueError as error:
        if "mask does not retain any complete cell" not in str(error):
            raise
        # Cannot split the atom (no complete cells)
        parcel_masks.append(child_mask)
        return counter

    # Compute the second eigenpair and push split onto heap
    evals, emodes = Solver(submesh).eigs(2, rng=seed)
    queued_mask = np.zeros(len(child_mask), dtype=bool)
    queued_mask[child_mapping] = True

    counter += 1
    heappush(heap, (evals[1], counter, emodes[:, 1] >= 0, queued_mask))
    return counter

def make_parcellation(
    geometry: TriaMesh | TetMesh,
    n_parcels: int | list[int] | tuple[int, ...],
    check_mesh: bool = True,
    seed: int | None = None
) -> NDArray[np.floating]:
    """
    Generate a parcellation for a surface or volume mesh using the hierarchical algorithm from Pang
    et al. (2025) [1]_, where the mesh is iteratively split using the sign of the second (first
    non-constant) Laplace-Beltrami eigenmode. Vertices omitted by broken induced cells remain ``-1``
    in the returned integer labels.

    Parameters
    ----------
    geometry : lapy.TriaMesh or lapy.TetMesh
        Surface or volume mesh containing ``n_verts`` vertices. Must be manifold with one connected
        component.
    n_parcels : int or sequence of int
        Number(s) of parcels requested.
    check_mesh : bool
        Whether to check that the input mesh is manifold with one connected component.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        The parcellation array of shape ``(n_verts,)`` if ``n_parcels`` is an integer, or
        ``(n_verts, n_requests)`` if ``n_parcels`` is a sequence of integers.

    Raises
    ------
    TypeError
        If ``geometry`` is not an instance of ``lapy.TriaMesh`` or ``lapy.TetMesh``.
    ValueError
        If ``n_parcels`` is not an integer or a sequence of integers in the range [1, ``n_verts``].
    RuntimeError
        If the highest requested number of parcels failed to be reached due to mesh topology.

    References
    ----------
    ..  [1] Pang, J.C., et al. (2025). Geometric influences on the regional organization of the
        mammalian brain. bioRxiv. https://doi.org/10.1101/2025.01.30.635820
    """
    # TODO: add equal_areas=False for MGH mode
    # TODO: add repair=True to repair masks with unreferenced vertices
    # Format / validate inputs
    if isinstance(geometry, TriaMesh):
        if check_mesh:
            check_surf(geometry)
    elif isinstance(geometry, TetMesh):
        if check_mesh:
            check_vol(geometry)
    else:
        raise TypeError("geometry must be an instance of either lapy.TriaMesh or lapy.TetMesh")
    n_verts = geometry.v.shape[0]

    err_msg = ("n_parcels must be an integer or a sequence of integers in the range [1, n_verts = "
               f"{n_verts}]")
    if scalar_request := isinstance(n_parcels, (int, np.integer)):
        n_parcels = [n_parcels]
    elif not (isinstance(n_parcels, (list, tuple)) and len(n_parcels) > 0
              and all(isinstance(val, (int, np.integer)) for val in n_parcels)):
        raise ValueError(err_msg)
    if any(value < 1 or value > n_verts for value in n_parcels):
        raise ValueError(err_msg)
    
    n_parcels_set = sorted(set(n_parcels))
    parcellations = {}

    # Handle special / trivial case of one parcel
    if 1 in n_parcels_set:
        parcellations[1] = np.ones(n_verts, dtype=int)
    if n_parcels_set == [1]:
        return parcellations[1] if scalar_request else parcellations[1][:, None]

    # Initialise the heap with the first eigenpair
    evals, emodes = Solver(geometry).eigs(2, rng=seed)
    counter = 0  # Tiebreaker for degen evals
    heap = [(evals[1], counter, emodes[:, 1] >= 0, np.ones(n_verts, dtype=bool))]

    # Split parcel with lowest eigenvalue until we reach the largest requested parcel count
    parcel_masks = []
    for parcel_count in range(2, n_parcels_set[-1] + 1):
        emode2_pos, parent_mask = heappop(heap)[2:]

        # Split the parent parcel into two child parcels according to emode2 sign
        daughter_mask = np.zeros(len(parent_mask), dtype=bool)
        son_mask = daughter_mask.copy()
        son_mask[parent_mask] = emode2_pos
        daughter_mask[parent_mask] = ~emode2_pos

        for child_mask in (son_mask, daughter_mask):
            counter = _queue_child_or_finalize(
                geometry, child_mask, heap, parcel_masks, counter, seed
            )

        current_count = len(parcel_masks) + len(heap)
        err_msg = (f"Cannot create the highest requested number of parcels; reached {current_count}"
                   " parcels")
        if any(target < current_count and target not in parcellations for target in n_parcels_set):
            raise RuntimeError(err_msg)
        if not heap and current_count < n_parcels_set[-1]:
            raise RuntimeError(err_msg)

        if parcel_count in n_parcels_set:
            # Use the current masks to create the parcellation
            parcellation_i = np.full(n_verts, -1, dtype=int)
            for j, parcel in enumerate(parcel_masks + [item[3] for item in heap], start=1):
                parcellation_i[parcel] = j
            parcellations[parcel_count] = parcellation_i

    parcellations = np.column_stack([parcellations[i] for i in n_parcels])
    return parcellations[:, 0] if scalar_request else parcellations