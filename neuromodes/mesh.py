"""
Module for validating and manipulating meshes of brain structures.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from lapy import TetMesh, TriaMesh
from nibabel.affines import apply_affine
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load
from nibabel.nifti1 import Nifti1Image
from scipy.interpolate import RBFInterpolator, griddata
from scipy.ndimage import binary_closing

from neuromodes.io import fs_extensions

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
        If the geometry is a dictionary that does not have keys 'vertices' and 'faces', or if
        'faces' does not reference an array with shape (n_tetras, 4) for volumes or (n_trias, 3) for
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
                   'an array of shape (n_verts, 3) and `faces` key should reference an array of '
                   'shape (n_tetras, 4) for volumes or (n_trias, 3) for surfaces.')
        if 'vertices' not in geometry:
            raise ValueError(err_str)
        try:
            verts_per_face = np.asarray(geometry['faces']).shape[1]
        except Exception:
            raise ValueError(err_str)
        if verts_per_face == 4:
            return True
        elif verts_per_face == 3:
            return False
        raise ValueError(err_str)

def mask_mesh(
    geometry: TriaMesh,
    mask: NDArray[np.bool_]
) -> TriaMesh:
    """
    Remove specified vertices and corresponding faces from a triangular surface mesh. Note that this
    may produce a non-contiguous mesh with unreferenced vertices--use :func:`check_surf` to validate
    the resulting mesh.

    Parameters
    ----------
    geometry : lapy.TriaMesh
        The input surface mesh.
    mask : array-like
        A boolean array indicating which vertices to keep (``True``) or remove (``False``).

    Returns
    -------
    lapy.TriaMesh
        The masked surface mesh.

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

    # Create a new TriaMesh with the masked vertices and elements
    return geometry.__class__(v=v_masked, t=t_masked)

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

def check_vol(
    vol: TetMesh
) -> None:
    """
    Check if the volume mesh has no unreferenced vertices and a contiguous surface boundary.
    
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
    
    # Ensure volume is manifold (i.e., no faces shared by more than two tets)
    if not _is_vol_manifold(vol):
        raise ValueError('Volume mesh is not manifold: contains faces shared by more than two '
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
    Check if the surface mesh is contiguous with no unreferenced vertices.
    
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
        If the surface mesh is not manifold (i.e., contains edges belonging to more than two faces).
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
                         'faces.')

# TODO: expose function below but remove interpolation and validate one-to-one mapping
#
# When this is done, we can plot modes in NIFTI space in the tutorial (e.g.,
# nilearn.plotting.plot_stat_map(mode_nifti, display_mode='ortho', cmap='seismic_r',
# symmetric_cbar=True, title='Mode 2')
#
# def tetmesh_to_nifti(
#     data: ArrayLike,
#     tetmesh: TetMesh,
#     nifti_mask: str | Path | Nifti1Image,
#     method: str = 'nearest',
#     **rbf_kwargs
# ) -> Nifti1Image:
#     """
#     Project data defined on a tetrahedral mesh to a volumetric NIFTI space.
    
#     Parameters
#     ----------
#     nifti_mask : str, Path, or Nifti1Image
#         Input NIFTI file path or loaded Nifti1Image object defining the target volume space.
#     data : array-like
#         Data values defined on the vertices of the tetmesh (shape should be (n_vertices,))
#     tetmesh : lapy.TetMesh
#         The tetrahedral mesh on which the data is defined. Must have vertex coordinates similar to
#         the NIFTI space, after applying the appropriate affine transformation.
#     """
#     # Format / validate arguments
#     data = np.asarray_chkfinite(data)
#     if data.shape != (tetmesh.v.shape[0],):
#         raise ValueError(f"data must have shape ({tetmesh.v.shape[0]},), got {data.shape}.")
#     if not isinstance(tetmesh, TetMesh):
#         raise TypeError("tetmesh must be an instance of lapy.TetMesh.")
#     if isinstance(nifti_mask, (str, Path)):
#         nifti_mask = load(nifti_mask)
#     elif not isinstance(nifti_mask, Nifti1Image):
#         raise TypeError("nifti_mask must be a Nifti1Image object or a path-like string to a valid "
#                          "`.nii` or `.nii.gz` file.")
    
#     # Get coordinates of nonzero voxels in physical space
#     x, y, z = np.asarray(nifti_mask.get_fdata() != 0).nonzero()
#     vox_coords = np.column_stack([x, y, z])
#     vox_coords = apply_affine(nifti_mask.affine, vox_coords)

#     # Initialise NIFTI array
#     interp_data = np.zeros(nifti_mask.shape, dtype=np.result_type(data, np.float32))

#     # Linear interpolation within convex hull of vertices, store at ROI coordinates
#     interp_data[x, y, z] = griddata(tetmesh.v, data, vox_coords, method=method)

#     # Use Rbf to interpolate at any voxels outside the convex hull (griddata returns NaNs)
#     nan_mask = np.isnan(interp_data)
#     if np.any(nan_mask):
#         rbf = RBFInterpolator(tetmesh.v, data, **rbf_kwargs)
#         interp_data[nan_mask] = rbf(vox_coords[nan_mask])

#     # Create a new NIFTI image with the interpolated values
#     header = nifti_mask.header.copy().set_data_dtype(interp_data.dtype)
#     return Nifti1Image(interp_data, nifti_mask.affine, header=header)

# TODO: consider replacing linear interpolation and RBF with biharmonic inpainting
def nifti_data_to_tetmesh(
    nifti_data: str | Path | Nifti1Image,
    tetmesh: TetMesh,
    nifti_mask: str | Path | Nifti1Image | None = None,
    **rbf_kwargs
) -> NDArray[np.floating]:
    """
    Project data from volumetric NIFTI space to a tetrahedral mesh using linear interpolation and
    radial basis functions for extrapolation.
    
    Parameters
    ----------
    nifti_data : str, Path, or Nifti1Image
        Input NIFTI file path or loaded ``Nifti1Image`` object.
    tetmesh : lapy.TetMesh
        The tetrahedral mesh to which data is projected.
    nifti_mask : str, Path, or Nifti1Image or None
        Optional mask to define the region of interest. If ``None``, nonzero voxels are used.
        Default is ``None``.
    **rbf_kwargs
        Additional keyword arguments to pass to ``scipy.interpolate.RBFInterpolator``, such as 
        ``function`` and ``smooth``.
    
    Returns
    -------
    ndarray
        Data interpolated at mesh vertices, shape (n_vertices,).
    """
    # Load NIFTI if needed
    if isinstance(nifti_data, (str, Path)):
        nifti_data = load(nifti_data)
    elif not isinstance(nifti_data, Nifti1Image):
        raise TypeError("nifti_data must be a Nifti1Image object or path")
    
    # Get mask
    data = nifti_data.get_fdata()
    if nifti_mask is None:
        mask = data != 0
    else:
        if isinstance(nifti_mask, (str, Path)):
            nifti_mask = load(nifti_mask)
        mask = nifti_mask.get_fdata() != 0
    
    # Get voxel coordinates and values
    x, y, z = np.where(mask)
    vox_coords = np.column_stack([x, y, z])
    vox_coords = apply_affine(nifti_data.affine, vox_coords)
    
    # Linear interpolation within convex hull of ROI coordinates, store at mesh vertices
    interp_data = griddata(vox_coords, data[mask], tetmesh.v, method='linear')

    # Use Rbf to interpolate at any vertices outside the convex hull (griddata returns NaNs)
    nan_mask = np.isnan(interp_data)
    if np.any(nan_mask):
        rbf = RBFInterpolator(vox_coords, data[mask], **rbf_kwargs)
        interp_data[nan_mask] = rbf(tetmesh.v[nan_mask])

    return interp_data

def nifti_mask_to_tetmesh(
    nifti_mask: str | Path | Nifti1Image,
    closings: int = 0,
    discard_components: bool = False,
    **tetgen_kwargs
) -> TetMesh:
    """
    Tetrahedral meshing from a binary voxel mask NIFTI via marching cubes and Delaunay
    tetrahedralization. Note that mesh quality may be poor when the NIFTI has nighly anisotropic
    voxel dimensions. 

    Parameters
    ----------
    nifti_mask : str, Path, or Nifti1Image
        Input NIFTI file path or loaded Nifti1Image object containing a binary mask of the region of
        interest (ROI).
    closings : int, optional
        Number of binary closing operations to perform on the ROI mask before meshing. This can help
        fill small holes and merge disconnected pieces in the ROI. Default is ``0``.
    discard_components : bool, optional
        If True, only the largest connected component of the generated surface mesh will be kept, 
        discarding the rest. This can be useful for removing small, irrelevant pieces from the mesh.
        Default is ``False``.
    **tetgen_kwargs
        Additional keyword arguments to pass to the `tetgen.TetGen.tetrahedralize`.

    Returns
    -------
    lapy.TetMesh
        The generated tetrahedral mesh of the ROI.

    Raises
    ------
    ImportError
        If the required packages `skimage` and `tetgen` are not installed.
    TypeError
        If ``nifti_mask`` is not a Nifti1Image object or a path-like string to a valid ``.nii`` or
        ``.nii.gz``.
        file.
    ValueError
        If ``closings`` is not a non-negative integer.
    ValueError
        If the generated surface mesh from marching cubes is not closed.
    ValueError
        If the generated surface mesh from marching cubes is not manifold. 
    """
    # Handle optional dependencies EAFP style
    try:
        from skimage.measure import marching_cubes
        from tetgen import TetGen
    except ImportError as e:
        raise ImportError("The `skimage` and `tetgen` packages are required for this function. "
                          "Please ensure they are installed in your environment (e.g., pip install "
                          "neuromodes[vol]).") from e

    # Format / validate arguments
    if isinstance(nifti_mask, (str, Path)):
        nifti_mask = load(nifti_mask)
    elif not isinstance(nifti_mask, Nifti1Image):
        raise TypeError("vol must be a Nifti1Image object or a path-like string to a valid "
                         "`.nii` or `.nii.gz` file.")
    if closings != int(closings) or closings < 0:
        raise ValueError("Parameter `closings` must be a non-negative integer.")
    
    # Get binary ROI from NIFTI
    roi = (nifti_mask.get_fdata() != 0).astype(np.uint8)
    if closings:
        roi = binary_closing(roi, iterations=closings)

    # Marching cubes to extract smoothed surface (replacing mri_mc from FreeSurfer)
    surf_verts, trias, _, _ = marching_cubes(roi, level=0.5, allow_degenerate=False)
    surf_verts = apply_affine(nifti_mask.affine, surf_verts)
    surf = TriaMesh(v=surf_verts, t=trias)
    
    # Handle disconnected components in generated surface
    n_components, labels = surf.connected_components()
    if n_components > 1:
        if discard_components:
            surf.keep_largest_connected_component_()
            print(f"Components discarded: {n_components-1}.")
            unique, counts = np.unique(labels, return_counts=True)
            for comp, count in zip(unique, counts):
                if comp != np.argmax(np.bincount(labels)):
                    print(f"    - {count} vertices.")
        else:
            warn(f"Generated surface has {n_components} connected components; unable to proceed "
                 "with tetrahedral meshing. Surface mesh will be returned with labels array to "
                 "allow visual inspection. Consider using `discard_components` to keep only the "
                 "largest component, or `closings` to fill small holes and merge disconnected "
                 "pieces.")
            return surf, labels

    # Ensure that surface is closed and manifold
    if not surf.is_closed():
        raise ValueError("Generated surface mesh is not closed. Consider using `closings` to fill "
                         "small holes and merge disconnected pieces.")
    if not surf.is_manifold():
        raise ValueError("Generated surface mesh is not manifold: contains edges belonging to more "
                         "than two faces. Consider using `closings` to fill small holes and merge "
                         "disconnected pieces.")
        
    # Append vertices from marching cubes with voxel centers
    vox_coords = np.column_stack(np.nonzero(roi))
    vox_coords = apply_affine(nifti_mask.affine, vox_coords)
    init_verts = np.vstack([surf.v, vox_coords])

    # Generate edges to complete tetrahedral mesh
    tetgen = TetGen(init_verts, surf.t)
    tetgen.tetrahedralize(**tetgen_kwargs)
    verts = tetgen.node
    tetras = tetgen.elem

    # Convert to lapy
    mesh = TetMesh(v=verts.astype(np.float64), t=tetras.astype(np.int32))

    # Ensure the generated volume mesh is valid
    check_vol(mesh)

    return mesh