"""
Module for validating and manipulating meshes of brain structures.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np
from lapy import TetMesh, TriaMesh
from nibabel.gifti.gifti import GiftiImage

from neuromodes.io import fs_extensions

if TYPE_CHECKING:
    from numpy.typing import NDArray

MeshT = TypeVar("MeshT", TriaMesh, TetMesh)

class EmptyMaskError(Exception):
    """Raised when a mask does not retain any complete cell in a mesh."""

def is_vol(
    geometry: TetMesh | TriaMesh | GiftiImage | str | Path | dict
) -> bool:
    """
    Determine whether the given geometry represents a volume or surface mesh.

    Parameters
    ----------
    geometry : lapy.TetMesh, lapy.TriaMesh, nibabel.gifti.GiftiImage, str, Path, or dict
        The geometry to check. Can be an instance of `lapy.TetMesh`, `lapy.TriaMesh`,
        `nibabel.gifti.GiftiImage`, a path-like string, or a dictionary with mesh data.

    Returns
    -------
    bool
        True if the geometry is a volume mesh, False if it is a surface mesh.

    Raises
    ------
    ValueError
        If the geometry is a path-like string with an unrecognized file extension.
    ValueError
        If the geometry is a dictionary that does not have keys 'vertices' and 'cells', or if
        'cells' does not reference an array with shape (n_tetras, 4) for volumes or (n_trias, 3) for
        surfaces.
    """
    # Instances
    if isinstance(geometry, TetMesh):
        return True
    if isinstance(geometry, (TriaMesh, GiftiImage)):
        return False
    
    # Paths
    if isinstance(geometry, (str, Path)):
        if str(geometry).endswith('.tetra.vtk'):
            return True
        elif str(geometry).endswith(('.vtk', '.gii') + fs_extensions):
            return False
        raise ValueError(
            'Received path-like string for `geometry`, but file extension is not recognized. '
            'Please provide a path-like string to a mesh file for a surface (.vtk, .gii, '
            f'{", ".join(fs_extensions)}) or volume (.tetra.vtk).')
    
    # Dictionary
    if isinstance(geometry, dict):
        err_str = ('Received an invalid dictionary for `geometry`. `vertices` key should reference '
                   'an array of shape (n_verts, 3) and `cells` key should reference an array of '
                   'shape (n_tetras, 4) for volumes or (n_trias, 3) for surfaces.')
        if 'vertices' not in geometry:
            raise ValueError(err_str)
        try:
            verts_per_face = np.asarray(geometry['cells']).shape[1]
        except (ValueError, KeyError, IndexError):
            raise ValueError(err_str)
        if verts_per_face == 4:
            return True
        elif verts_per_face == 3:
            return False
        raise ValueError(err_str)

@overload
def mask_mesh(
    geometry: MeshT,
    mask: NDArray[np.bool_],
    return_mapping: Literal[False] = False,
) -> MeshT: ...

@overload
def mask_mesh(
    geometry: MeshT,
    mask: NDArray[np.bool_],
    return_mapping: Literal[True],
) -> tuple[MeshT, NDArray[np.integer]]: ...

def mask_mesh(
    geometry: MeshT,
    mask: NDArray[np.bool_],
    return_mapping: bool = False,
) -> MeshT | tuple[MeshT, NDArray[np.integer]]:
    """
    Remove vertices and cells not fully contained by a vertex mask. Vertices retained by ``mask``
    that do not belong to a retained cell are removed as well, so the returned mesh has no
    unreferenced vertices.

    Parameters
    ----------
    geometry : lapy.TriaMesh or lapy.TetMesh
        The input mesh.
    mask : array-like
        A boolean array indicating which vertices to keep (``True``) or remove (``False``).
    return_mapping : bool, optional
        If ``True``, also return the original indices of vertices in the masked mesh.

    Returns
    -------
    lapy.TriaMesh, lapy.TetMesh, or tuple
        The masked mesh, or a tuple containing the mesh, local-to-global mapping.

    Raises
    ------
    ValueError
        If ``mask`` does not have a length matching the number of vertices in ``geometry``.
    EmptyMaskError
        If the mask does not retain any complete cell.
    """
    # Format / validate arguments
    mask = np.asarray_chkfinite(mask, dtype=bool)
    if mask.shape != (geometry.v.shape[0],):
        raise ValueError(f"mask must have shape ({geometry.v.shape[0]},), matching the number of "
                         "vertices in geometry.")

    # Keep only cells whose vertices are all retained
    elem_mask = np.all(mask[geometry.t], axis=1)
    if not np.any(elem_mask):
        raise EmptyMaskError("mask does not retain any complete cell")
    cells = geometry.t[elem_mask]

    # Remove unreferenced vertices and remap cell indices
    mapping = np.unique(cells)
    inverse = np.full(len(geometry.v), -1, dtype=geometry.t.dtype)
    inverse[mapping] = np.arange(len(mapping), dtype=geometry.t.dtype)

    # Create new mesh
    masked = geometry.__class__(v=geometry.v[mapping], t=inverse[cells])
    return (masked, mapping) if return_mapping else masked

def unmask_data(
    data: NDArray[np.floating],
    mask: NDArray[np.bool_],
    fill_value: float = np.nan
) -> NDArray[np.floating]:
    """
    Unmasks data by inserting it into a full array with the same length as the medial wall mask.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be unmasked, of shape ``(n_verts)`` or ``(n_verts, ...)``, where ``n_verts`` is
        the number of vertices in the masked mesh and additional axes represent maps to be unmasked
        independently.
    mask : numpy.ndarray
        A boolean array-like of shape ``(n_verts + n_extra_verts)``, where ``True`` indicates the
        positions of the data in the full array. Must contain exactly ``n_verts`` ``True`` values.
    fill_value : float, optional
        The value to fill in the positions outside the mask. Default is NaN.

    Returns
    -------
    numpy.ndarray
        The unmasked data of shape ``(n_verts + n_extra_verts)`` or
        ``(n_verts + n_added_verts, n_maps)``.

    Raises
    ------
    ValueError
        If ``mask`` is not a 1D boolean array.
    ValueError
        If ``data`` does not have shape ``(n_verts,)`` or ``(n_verts, n_maps)``.
    """
    # Format / validate arguments (TODO: use EigenData)
    mask = np.asarray_chkfinite(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("`mask` must be a 1D boolean array.")
    
    data = np.asarray(data)
    if data.shape[0] != np.sum(mask):
        raise ValueError("`data` must have shape (n_verts,...), where n_verts "
                         f"matches the number of True values in `mask` ({np.sum(mask)}).")
    
    # out_size
    out_size = (len(mask),) + data.shape[1:]

    # out_dtype: the safest dtype that can hold BOTH the data and the fill_value
    out_dtype = np.result_type(data.dtype, np.array(fill_value).dtype)
    
    # Initialise array of fill values
    data_unmasked = np.full(out_size, fill_value, dtype=out_dtype)

    # Overwrite rows with data where mask is True
    data_unmasked[mask, ...] = data

    return data_unmasked

def normalize_vol(
    geometry: TetMesh
) -> TetMesh:
    """
    Translate the mesh centroid to the origin and rescale to unit volume.

    Parameters
    ----------
    geometry : lapy.TetMesh
        The input volume mesh, to be modified in-place.
    """
    # Make copy to avoid in-place modification
    geometry = TetMesh(geometry.v, geometry.t)

    # Get edge vectors for each tetrahedron
    t0 = geometry.t[:, 0]
    t1 = geometry.t[:, 1]
    t2 = geometry.t[:, 2]
    t3 = geometry.t[:, 3]

    v0 = geometry.v[t0, :]
    v1 = geometry.v[t1, :]
    v2 = geometry.v[t2, :]
    v3 = geometry.v[t3, :]

    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0

    # Compute volume of each tetrahedron using triple product formula: V = |(e1 . (e2 x e3))| / 6
    tetra_vols = np.abs(np.einsum('ij,ij->i', e1, np.cross(e2, e3))) / 6

    # Compute centroid of each tetrahedron as simple average of its vertices
    tetra_centroids = (v0 + v1 + v2 + v3) / 4

    # Compute mesh centroid as volume-weighted average of tetrahedron centroids
    # Note: this is equivalent to LaPy's TriaMesh.centroid()
    centroid = np.sum(tetra_vols[:, np.newaxis] * tetra_centroids, axis=0) / np.sum(tetra_vols)

    # Translate centroid to origin
    geometry.v -= centroid

    # Rescale to unit volume
    geometry.v /= geometry.boundary_tria().volume() ** (1/3)

    return geometry

def check_vol(
    vol: TetMesh
) -> None:
    """
    Check if the volume mesh is manifold, has no unreferenced vertices, and has a contiguous surface
    boundary.
    
    Parameters
    ----------
    vol : lapy.TetMesh
        The volume mesh to check.

    Raises
    ------
    ValueError
        If the volume mesh contains unreferenced vertices.
    ValueError
        If the volume mesh is not manifold (i.e., contains triangles shared by more than two
        tetrahedra).
    """
    if vol.has_free_vertices():
        raise ValueError('Volume mesh contains unreferenced vertices (i.e., not part of any '
                         'tetrahedron).')
    
    # Ensure volume is manifold (i.e., no cells shared by more than two tets)
    if not _is_vol_manifold(vol):
        raise ValueError('Volume mesh is not manifold: contains cells shared by more than two '
                         'tetrahedra.')

    # Validate surface boundary
    vol_boundary = vol.boundary_tria()
    vol_boundary.rm_free_vertices_()
    vol_boundary.orient_()
    check_surf(vol_boundary)

def _is_vol_manifold(vol) -> bool:
    """Check if the tetrahedral mesh is manifold.

    Returns
    -------
    bool
        True if every triangle face is shared by at most two tetrahedra.
    """
    # Extract all 4 triangles from each tetrahedron
    trias = np.concatenate([
        vol.t[:, [0, 1, 2]],
        vol.t[:, [0, 1, 3]],
        vol.t[:, [0, 2, 3]],
        vol.t[:, [1, 2, 3]],
    ])

    # Order vertices within each triangle for consistent representation
    trias.sort(axis=1)

    # Manifold if no triangle occurs more than twice
    tria_counts = np.unique(trias, axis=0, return_counts=True)[1]
    return np.all(tria_counts <= 2)

def check_surf(
    surf: TriaMesh
) -> None:
    """
    Check if the surface mesh is contiguous, manifold, and contains no unreferenced vertices.
    
    Parameters
    ----------
    surf : lapy.TriaMesh
        The surface mesh to check.

    Raises
    ------
    ValueError
        If the surface mesh contains unreferenced (i.e., faceless) vertices.
    ValueError
        If the surface mesh is not contiguous.
    ValueError
        If the surface mesh is not manifold (i.e., contains edges belonging to more than two cells).
    """
    # Ensure surface has no unreferenced vertices
    referenced = np.zeros(len(surf.v), dtype=bool)
    referenced[surf.t] = True
    if not np.all(referenced):
        raise ValueError(f'Surface mesh contains {np.sum(~referenced)} unreferenced '
                         '(i.e., faceless) vertices.')

    # Ensure surface is contiguous
    n_components = surf.connected_components()[0]
    if n_components != 1:
        raise ValueError(f'Surface mesh is not contiguous: {n_components} connected components '
                         'found.')

    # Ensure surface is manifold
    if not surf.is_manifold():
        raise ValueError('Surface mesh is not manifold: contains edges belonging to more than two '
                         'cells.')

# TODO: consider adding _is_face_contiguous() for parcellations