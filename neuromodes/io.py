"""
Module for reading, validating, and manipulating meshes of brain structures.
"""

from __future__ import annotations
from importlib.resources import files, as_file
import os
from pathlib import Path
from typing import Union, Tuple, cast, TYPE_CHECKING
from joblib import Memory
from lapy import TriaMesh, TetMesh
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike

fs_extensions = ('.white', '.pial', '.inflated', '.orig', '.sphere', '.smoothwm', '.qsphere',
                 '.fsaverage')

def is_vol(
    geometry: Union[TetMesh, TriaMesh, GiftiImage, str, Path, dict]
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

def read_vol(
    vol: Union[str, Path, TetMesh, dict]
) -> TetMesh:
    """
    Load and validate a tetrahedral volume mesh.

    Parameters
    ----------
    vol : str, Path, TetMesh, or dict
        Volume mesh specified as a file path (string or Path) to a VTK (.tetra.vtk) file, an
        instance of `lapy.TetMesh`, or a dictionary with `'vertices'` and `'faces'` keys,
        referencing arrays of shape (n_verts, 3) and (n_tetras, 4), respectively.

    Returns
    -------
    lapy.TetMesh
        Validated volume mesh with vertices and tetrahedra.

    Raises
    ------
    ValueError
        If `vol` is not a path-like string to a valid VTK (`.tetra.vtk`) file, an instance of
        `lapy.TetMesh`, or a dictionary with `'vertices'` and `'faces'` keys.
    """
    if isinstance(vol, TetMesh):
        return vol
    elif isinstance(vol, dict):
        return TetMesh(v=vol['vertices'], t=vol['faces'])
    else:
        vol_str = str(vol)
        if not Path(vol_str).is_file():
            raise ValueError(f"Volume data not found: {vol_str}")
        if vol_str.endswith('.tetra.vtk'):
            # Load with lapy
            return TetMesh.read_vtk(str(vol))
    raise ValueError("`vol` must be a path-like string to a valid VTK (.tetra.vtk) file, an "
                    "instance of `lapy.TetMesh`, or a dictionary with 'vertices' and 'faces' "
                    "keys.")

def read_surf(
    surf: Union[str, Path, GiftiImage, TriaMesh, dict]
) -> TriaMesh:
    """Load a triangular surface mesh.

    Parameters
    ----------
    surf : str, Path, GiftiImage, lapy.TriaMesh, or dict
        Surface mesh specified as a file path (string or Path) to a VTK (.vtk), GIFTI (.gii), or
        FreeSurfer file (.white, .pial, .inflated, .orig, .sphere, .smoothwm, .qsphere, .fsaverage),
        an instance of `nibabel.GiftiImage` or `lapy.TriaMesh`, or a dictionary
        with `'vertices'` and `'faces'` keys, referencing arrays of shapes (n_verts, 3) and
        (n_trias, 3), respectively.

    Returns
    -------
    lapy.TriaMesh
        Surface mesh with vertices and faces.

    Raises
    ------
    ValueError
        If `surf` is a path-like string to an unsupported format.
    ValueError
        If `surf` is a path-like string to a file that does not exist.
    """
    if isinstance(surf, TriaMesh):
        return surf
    elif isinstance(surf, GiftiImage):
        vertices=surf.darrays[0].data
        faces=surf.darrays[1].data
    elif isinstance(surf, dict):
        vertices=surf['vertices']
        faces=surf['faces']
    else:
        surf_str = str(surf)
        # check that file exists
        if not Path(surf_str).is_file():
            raise ValueError(f'File not found: {surf_str}')
        # Handle different file types
        if surf_str.endswith('.vtk'):
            return TriaMesh.read_vtk(surf_str)
        elif surf_str.endswith(fs_extensions):
            return TriaMesh.read_fssurf(surf_str)
        elif surf_str.endswith('.gii'):
            surf_data = cast(GiftiImage, load(surf_str)).darrays
            vertices=surf_data[0].data
            faces=surf_data[1].data
        else:
            raise ValueError(
                '`surf` must be a path-like string to a valid VTK (.vtk), GIFTI (.gii), or '
                f'FreeSurfer file {fs_extensions}, an instance of `nibabel.GiftiImage` or '
                '`lapy.TriaMesh`, or a dictionary of `faces` and `vertices` with shapes '
                '(n_verts, 3) and (n_trias, 3), respectively.'
                )
        
    return TriaMesh(v=vertices, t=faces)

def mask_mesh(
    geometry: Union[TriaMesh, TetMesh],
    mask: ArrayLike
) -> Union[TriaMesh, TetMesh]:
    """
    Remove specified vertices and corresponding elements from a triangular surface or tetrahedral
    volume mesh. Returns a `lapy.TriaMesh` or `lapy.TetMesh` object.

    Parameters
    ----------
    geometry : lapy.TriaMesh or lapy.TetMesh
        The input surface or volume mesh.
    mask : array-like
        A boolean array indicating which vertices to keep (`True`) or remove (`False`).

    Returns
    -------
    lapy.TriaMesh or lapy.TetMesh
        The masked surface or volume mesh.

    Raises
    ------
    ValueError
        If `mask` does not have a length matching the number of vertices in `geometry`.
    """
    # Format / validate arguments
    mask = np.asarray_chkfinite(mask, dtype=bool)
    if mask.shape != (geometry.v.shape[0],):
        raise ValueError(f"`mask` must have shape (n_verts,) = ({geometry.v.shape[0]},).")

    # Remove vertices not in mask
    v_masked = geometry.v[mask]

    # Update vertex indices of elements (-1 represents removed vertices)
    v_map = np.full(len(mask), -1, dtype=int)
    v_map[mask] = np.arange(np.sum(mask))
    t_remapped = v_map[geometry.t]
    
    # Keep only elements where all vertices are in the mask
    elem_mask = np.all(t_remapped != -1, axis=1)
    t_masked = t_remapped[elem_mask]

    # Create a new TriaMesh or TetMesh with the masked vertices and elements
    return geometry.__class__(v=v_masked, t=t_masked)

def normalize_vol(
    geometry: TetMesh
) -> TetMesh:
    """
    Translate the mesh centroid to the origin and rescale to unit volume.

    Parameters
    ----------
    geometry : lapy.TetMesh
        The input volume mesh.

    Returns
    -------
    lapy.TetMesh
        The normalized volume mesh.
    """
    # Translate centroid to origin
    geometry.v -= geometry.v.mean(axis=0)

    # Rescale to unit volume
    geometry.v /= geometry.boundary_tria().volume() ** (1/3)

    return geometry

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

def fetch_vol(
    structure: str,
    species: str = 'human',
    hemi: str = 'L',
    template: str = 'MNI152',
) -> TetMesh:
    """
    Load a tetrahedral volume mesh from neuromodes data directory. For a list of available volumes,
    see https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv.

    Parameters
    ----------
    structure : str
        Brain structure to load. Options include `'thalamus'`, `'striatum'`, and `'hippocampus'`.
    species : str, optional
        Species of the volume mesh. Currently only supports `'human'`. Default is `'human'`.
    hemi : str, optional
        Hemisphere of the volume mesh. Options are `'L'` and `'R'`. Default is `'L'`.
    template : str, optional
        Template of the volume mesh. Currently only supports `'MNI152'`. Default is `'MNI152'`.

    Returns
    -------
    lapy.TetMesh
        The loaded volume mesh.
    """
    data_dir = files('neuromodes.data')
    file_name = f'sp-{species}_tpl-{template}_hemi-{hemi}_{structure}.tetra.vtk'

    # TODO: make this infinitely less ugly after cleaning up all file names
    if structure == 'cortex' and species == 'mouse' and template == 'AMBA':
        file_name = f'sp-{species}_tpl-{template}_res-200um_hemi-{hemi}_315.tetra.vtk'

    try:
        with as_file(data_dir / file_name) as fpath:
            return read_vol(fpath)
    except Exception:
        raise ValueError(
            f"Volume data {file_name} not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv for a"
            " list of available volumes."
            )

def fetch_surf(
    species: str = 'human',
    density: str = '32k',
    hemi: str = 'L',
    surf_type: str = 'midthickness',
    template: str = 'fsLR'
) -> Tuple[TriaMesh, NDArray]:
    """
    Load a cortical triangular surface mesh and medial wall mask from neuromodes data directory. For
    a list of available surfaces, see
    https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv.

    Parameters
    ----------
    species : str, optional
        Species of the surface mesh. Options include `'human'`, `'macaque'`, and `'marmoset'`.
        Default is `'human'`.
    density : str, optional
        Density of the surface mesh. Options include `'32k'` for all species, and `'4k'` for human.
        Default is `'32k'`.
    hemi : str, optional
        Hemisphere of the surface mesh. Options are `'L'` for all species, and `'R'` for human.
        Default is `'L'`.
    surf_type : str, optional
        Surface type to load. Currently only supports `'midthickness'`. Default is `'midthickness'`.
    template : str, optional
        Template of the surface mesh. Currently only supports `'fsLR'`. Default is `'fsLR'`.
    
    Returns
    -------
    surf : lapy.TriaMesh
        The loaded surface mesh.
    medmask : np.ndarray
        The medial wall mask as a boolean array.

    Raises
    ------
    ValueError
        If the specified surface data is not found in the `neuromodes/data` directory.
    """
    data_dir = files('neuromodes.data')
    surf_name = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_{surf_type}.surf.gii'
    mask_name = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_medmask.label.gii'

    try:
        with as_file(data_dir / surf_name) as fpath:
            surf = read_surf(fpath)
        with as_file(data_dir / mask_name) as fpath:
            medmask = cast(GiftiImage, load(fpath)).darrays[0].data.astype(bool)
        
        return surf, medmask
    except Exception:
        raise ValueError(
            f"Surface data {surf_name} not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv for a"
            " list of available surfaces."
            )

def fetch_map(
    data: str,
    species: str = 'human',
    density: str = '32k',
    hemi: str = 'L',
    template: str = 'fsLR'
) -> NDArray:
    """
    Load cortical surface data from neuromodes data directory. For a list of available maps, see
    https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv.

    Parameters
    ----------
    data : str
        Cortical map to load. Options include `'fcgradient1'`, `'myelinmap'`, `'ndi'`, `'odi'`, and
        `'thickness'`.
    species : str, optional
        Species of the surface mesh. Currently only supports `'human'`. Default is `'human'`.
    density : str, optional
        Density of the surface mesh. Currently only supports `'32k'`. Default is `'32k'`.
    hemi : str, optional
        Hemisphere of the surface mesh. Currently only supports `'L'`. Default is `'L'`.
    template : str, optional
        Template of the surface mesh. Currently only supports `'fsLR'`. Default is `'fsLR'`.

    Returns
    -------
    np.ndarray
        The loaded cortical map data.

    Raises
    ------
    ValueError
        If the specified map data is not found in the `neuromodes/data` directory.
    """
    data_dir = files('neuromodes.data')
    filename = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_{data}.func.gii'

    try:
        with as_file(data_dir / filename) as fpath:
            return cast(GiftiImage, load(fpath)).darrays[0].data
    
    except Exception:
        raise ValueError(
            f"Map '{filename}' not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv for a"
            " list of available data files."
        )

def _set_cache():
    """
    Set up joblib memory caching based. Uses the directory specified by the `CACHE_DIR` 
    environment variable, or defaults to `~/.neuromodes_cache` if not set.
    
    Returns
    -------
    joblib.Memory
        Configured joblib Memory object for caching.

    Raises
    ------
    ImportError
        If `joblib` is not installed.
    """
    CACHE_DIR = os.getenv("CACHE_DIR")
    if CACHE_DIR is None:
        CACHE_DIR = Path.home() / ".neuromodes_cache"
        print(f"Using default cache directory at {CACHE_DIR}. To cache elsewhere, set the CACHE_DIR"
              " environment variable beforehand.")
    else:
        CACHE_DIR = Path(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return Memory(CACHE_DIR, verbose=0)