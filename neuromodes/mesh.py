"""
Module for reading, validating, manipulating, and creating meshes of brain structures.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from typing import TypeVar
    from numpy import bool_
    from numpy.typing import NDArray
    from lapy import TriaMesh, TetMesh
    MeshType = TypeVar('MeshType', TriaMesh, TetMesh)

def mask_mesh(
    geometry: MeshType,
    mask: NDArray[bool_]
) -> MeshType:
    """
    Remove specified vertices and corresponding elements from a triangular surface mesh. Returns a
    ``lapy.TriaMesh`` object.

    Parameters
    ----------
    geometry : lapy.TriaMesh or lapy.TetMesh
        The input surface or volume mesh.
    mask : array-like
        A boolean array indicating which vertices to keep (``True``) or remove (``False``).

    Returns
    -------
    lapy.TriaMesh or lapy.TetMesh
        The masked surface or volume mesh.

    Raises
    ------
    ValueError
        If ``mask`` does not have a length matching the number of vertices in ``geometry``.
    """
    # Format / validate arguments
    mask = np.asarray_chkfinite(mask, dtype=bool)
    if mask.shape != (geometry.v.shape[0],):
        raise ValueError(f"mask must have shape ({geometry.v.shape[0]},), matching the number of "
                         "vertices in geometry.")

    # Remove vertices not in mask
    v_masked = geometry.v[mask] # inherit original type

    # Update vertex indices of elements
    v_map = unmask_data(np.arange(np.sum(mask)), mask, fill_value=0).astype(geometry.t.dtype) 
    t_remapped = v_map[geometry.t]
    
    # Keep only elements where all vertices are in the mask
    elem_mask = np.all(mask[geometry.t], axis=1)
    t_masked = t_remapped[elem_mask]

    # Create a new TriaMesh or TetMesh with the masked vertices and elements
    return geometry.__class__(v=v_masked, t=t_masked)

# TODO : investigate generic type hinting/overloads here
def unmask_data(
    data: NDArray,
    mask: NDArray[bool_],
    fill_value: float | int | bool = np.nan
) -> NDArray:
    """
    Unmasks data by inserting it into a full array with the same length as the medial wall mask.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be unmasked, of shape ``(n_verts,)`` or ``(n_verts, ...)``.
    mask : numpy.ndarray
        A boolean array-like of shape ``(n_verts + n_extra_verts)`` where ``True`` indicates the
        positions of the data in the full array. Must contain exactly ``n_verts`` ``True`` values.
    fill_value : float, optional
        The value to fill in the positions outside the mask. Default is NaN.

    Returns
    -------
    numpy.ndarray
        The unmasked data of shape ``(n_verts + n_extra_verts,)`` or
        ``(n_verts + n_added_verts, ...)``.

    Raises
    ------
    ValueError
        If ``mask`` is not a 1D boolean array.
    ValueError
        If ``data`` does not have shape ``(n_verts,)`` or ``(n_verts, ...)``.
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

def check_surf(
    surf: TriaMesh
) -> None:
    """
    Check if the surface mesh is contiguous with no unreferenced vertices.
    
    Parameters
    ----------
    surf : lapy.TriaMesh
        The surface mesh to check.

    Raises
    ------
    ValueError
        If the surface mesh contains unreferenced vertices.
    ValueError
        If the surface mesh is not contiguous.
    ValueError
        If the surface mesh is not manifold (i.e., contains edges belonging to more than two faces).
    """
    # Ensure surface has no unreferenced vertices
    referenced = np.zeros(len(surf.v), dtype=bool)
    referenced[surf.t] = True
    if not np.all(referenced):
        raise ValueError(f'Surface mesh contains {np.sum(~referenced)} unreferenced '
                         'vertices (i.e., not part of any face).')

    # Ensure surface is contiguous
    n_components = surf.connected_components()[0]
    if n_components != 1:
        raise ValueError(f'Surface mesh is not contiguous: {n_components} connected components '
                         'found.')

    # Ensure surface is manifold
    if not surf.is_manifold():
        raise ValueError('Surface mesh is not manifold: contains edges belonging to more than two '
                         'faces.')
