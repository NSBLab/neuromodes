"""Functions for generating hierarchical parcellations of a surface or volume mesh."""

from __future__ import annotations

from collections.abc import Sequence
from heapq import heappop, heappush
from itertools import count
from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np
from lapy import Solver, TetMesh, TriaMesh

from neuromodes.mesh import check_surf, check_vol, is_vol, mask_mesh

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

def make_parcellation(
    geometry: TriaMesh | TetMesh,
    n_parcels: int | Sequence[int],
    check_geometry: bool = True,
    # TODO: consider adding repair="remesh", if it's meaningfully different and not too slow
    # e.g., when an unreferenced vertex appears, add supporting vertices to the mesh where the
    # emode2 sign (determined by interpolation) is the same and resulting cell quality is acceptable
    repair: Literal["nearest"] | None = "nearest",
    seed: int | Generator | None = None
    # TODO: add equal_areas=False for MGH method
    # TODO: add use_cholmod=False iff faster
) -> NDArray[np.integer]:
    """
    Generate a parcellation for a surface or volume mesh using the hierarchical algorithm from Pang
    et al. (2025) [1]_, where the mesh is iteratively split using the sign of the second (first
    non-constant) Laplace-Beltrami eigenmode.

    Parameters
    ----------
    geometry : lapy.TriaMesh or lapy.TetMesh
        Surface or volume mesh containing ``n_verts`` vertices. Must be manifold with one connected
        component.
    n_parcels : int or sequence of int
        Number(s) of parcels requested.
    check_geometry : bool
        Whether to check that the input mesh ``geometry`` is manifold with one connected component.
    repair : 'nearest' or None, optional
        Method for assigning vertices omitted from complete induced cells. ``'nearest'`` assigns to
        the closest labelled adjacent vertex.
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
    TypeError
        If ``n_parcels`` is not an integer or a sequence of integers.
    ValueError    
        If any of the values in ``n_parcels`` are not in the range [1, ``n_verts``].
    RuntimeError
        If the highest requested number of parcels failed to be reached due to mesh topology.
    ValueError
        If ``repair`` is not ``None`` or ``'nearest'``.

    References
    ----------
    ..  [1] Pang, J.C., et al. (2025). Geometric influences on the regional organization of the
        mammalian brain. bioRxiv. https://doi.org/10.1101/2025.01.30.635820
    """
    # Format / validate inputs
    if check_geometry:
        if is_vol(geometry):
            check_vol(geometry)
        else:
            check_surf(geometry)
    n_verts = geometry.v.shape[0]

    scalar_request = isinstance(n_parcels, (int, np.integer))
    n_parcels = np.atleast_1d(n_parcels)
    if any(val < 1 or val > n_verts for val in n_parcels):
        raise ValueError("n_parcels must be an integer or a sequence of integers in the range "
                         f"[1, n_verts = {n_verts}]")
    n_parcels_set = np.unique(n_parcels)

    # Handle trivial case of one parcel only
    if np.array_equal(n_parcels_set, 1):
        return np.zeros((n_verts,) if scalar_request else (n_verts, len(n_parcels)), dtype=int)

    parcellations_set = np.full((n_verts, len(n_parcels_set)), -1, dtype=int)

    # Handle trivial case of one parcel amongst other requests
    if n_parcels_set[0] == 1:
        parcellations_set[:, 0] = 0

    # Initialise the heap with the first eigenpair
    evals, emodes = Solver(geometry).eigs(2, rng=seed)
    counter = count() # Tiebreaker for degen evals
    heap = [(evals[1], next(counter), emodes[:, 1] >= 0, np.ones(n_verts, dtype=bool))]

    # Split parcel with lowest eigenvalue until we reach the largest requested parcel count
    leaf_masks = []
    for parcel_count in range(2, n_parcels_set[-1] + 1):
        # TODO: consider a multiple connected components approach for potential speedup
        _split_parcel(geometry, leaf_masks, heap, counter, seed)

        current_count = len(leaf_masks) + len(heap)
        if not heap and current_count < n_parcels_set[-1]:
            # TODO: warn and return -1 cols for unachievable parcel counts
            raise RuntimeError("Cannot create the highest requested number of parcels; reached "
                               f"{current_count} parcels")

        if parcel_count in n_parcels_set:
            # Use the current masks to create the parcellation
            col = (n_parcels_set == parcel_count).nonzero()[0][0]
            for j, parcel in enumerate([item[3] for item in heap] + leaf_masks):
                parcellations_set[parcel, col] = j

    if repair == "nearest":
        parcellations_set = _fill_omitted_vertices(geometry, parcellations_set)
    elif repair is not None:
        assert_never(repair)

    # create parcellations array of shape (n_verts, n_requests), with columns in the order of
    # n_parcels, feom parcellations_set
    parcellations = parcellations_set[:, np.searchsorted(n_parcels_set, n_parcels)]
    return parcellations[:, 0] if scalar_request else parcellations

def _split_parcel(
    geometry: TriaMesh | TetMesh,
    leaf_masks: list[NDArray[np.bool_]],
    heap: list,
    counter: count,
    seed: int | Generator | None,
) -> None:
    """
    Split a parcel using its second Laplace-Beltrami eigenmode and push the resulting child parcels
    onto the heap. If a child parcel does not contain any cells (triangles for ``TriaMesh`` or
    tetrahedra for ``TetMesh``), its second eigenpair cannot be computed and so it is not split
    further (leaf node in the hierarchy).

    Parameters
    ----------
    geometry : lapy.TriaMesh or lapy.TetMesh
        Surface or volume mesh containing ``n_verts`` vertices. Must be manifold with one connected
        component.
    leaf_masks: list
        List of small terminated parcel masks (leaf nodes in the hierarchy).
    heap: list
        List of tuples (eigenvalue, tiebreaker, eigenmode_sign, mask) representing the current state
        of the hierarchy.
    counter: count
        A counter to break ties in the heap (degenerate eigenvalues).
    seed: int | Generator | None
        Random seed for reproducibility.
    """
    # Get parcel with lowest eigenvalue
    emode2_ispos, parent_mask = heappop(heap)[2:]

    # Create two child masks using the emode2 sign
    empty_mask = np.zeros(geometry.v.shape[0], dtype=bool)
    child1_mask = empty_mask.copy()
    child2_mask = empty_mask.copy()
    child1_mask[parent_mask] = emode2_ispos
    child2_mask[parent_mask] = ~emode2_ispos

    for child_mask in (child1_mask, child2_mask):
        try:
            submesh, mapping = mask_mesh(geometry, child_mask, return_mapping=True)
        except ValueError:  # TODO: make custom error in mask_mesh()
            # Parcel has no cells, so it cannot be split further (leaf node in the hierarchy)
            leaf_masks.append(child_mask)
            continue

        # Compute the second eigenpair of child parcel and push with updated mask onto heap
        evals, emodes = Solver(submesh).eigs(2, rng=seed)
        child_mask = empty_mask.copy()
        child_mask[mapping] = True

        heappush(heap, (evals[1], next(counter), emodes[:, 1] >= 0, child_mask))

def _fill_omitted_vertices(
    geometry: TriaMesh | TetMesh,
    parcellations_set: NDArray[np.integer],
) -> NDArray[np.integer]:
    """
    Relabel parcellations' omitted vertices (denoted by -1) using the label of the nearest
    non-omitted neighbour.

    Parameters
    ----------
    geometry : lapy.TriaMesh or lapy.TetMesh
        Surface or volume mesh containing ``n_verts`` vertices. Must be manifold with one connected
        component.
    parcellations_set : NDArray[np.integer]
        Array of parcellations of shape ``(n_verts, n_requests)``, where omitted vertices are
        denoted by -1. 

    Returns
    -------
    NDArray[np.integer]
        Array of parcellations of shape ``(n_verts, n_requests)``, with omitted vertices relabelled.

    Raises
    ------
    RuntimeError
        If a vertex cannot be repaired because it has no labelled neighbours.
    """
    verts_omitted = (parcellations_set == -1).nonzero()
    if not np.any(verts_omitted):
        return parcellations_set
    verts_omitted_union = np.unique(verts_omitted[0])

    adj_sym = geometry.adj_sym.tocsr()
    neighbours_max = {}
    neighbour_distances_max = {}
    for vertex in verts_omitted_union:
        # Get distances to all neighbours
        neighbours = adj_sym[vertex].indices
        distances = np.linalg.norm(geometry.v[neighbours] - geometry.v[vertex], axis=1)

        # Cache sorted distances and neighbour indices
        order = np.argsort(distances, kind="stable")
        neighbours_max[vertex] = neighbours[order]
        neighbour_distances_max[vertex] = distances[order]

    # Repair each parcellation independently
    for i in range(parcellations_set.shape[1]):
        parcellation = parcellations_set[:, i]

        # get indices of omitted vertices in this parcellation from verts_omitted
        for vertex in verts_omitted[0][verts_omitted[1] == i]:
            # Get indices of non-omitted neighbours
            neighbours = neighbours_max[vertex]
            candidates = (parcellation[neighbours] + 1).nonzero()[0]  # +1 includes label 0 and excludes omitted (-1)
            if len(candidates) == 0:
                # TODO: consider if this can happen and if we should expand search to 2 steps
                raise RuntimeError("cannot repair an unresolved vertex with no labelled neighbour")

            # Assign label of nearest non-omitted neighbour (for ties, use lowest label)
            distances = neighbour_distances_max[vertex][candidates]
            nearest_indices = neighbours[candidates[distances == np.min(distances)]]
            parcellation[vertex] = np.min(parcellation[nearest_indices])

        parcellations_set[:, i] = parcellation

    return parcellations_set