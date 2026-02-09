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
from nibabel.freesurfer.io import read_geometry
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load
import numpy as np
from trimesh import Trimesh

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike

fs_extensions = ('.white', '.pial', '.inflated', '.orig', '.sphere', '.smoothwm', '.qsphere',
                 '.fsaverage')

def is_vol(geometry) -> bool:
    return True if (
        isinstance(geometry, TetMesh)
        or (isinstance(geometry, (str, Path)) and str(geometry).endswith(('.tetra.vtk')))
        or (isinstance(geometry, dict) and 'tetras' in geometry)
        ) else False

def is_surf(geometry) -> bool:
    return True if (
        isinstance(geometry, (TriaMesh, Trimesh, GiftiImage))
        or (isinstance(geometry, (str, Path))
        and str(geometry).endswith(('.vtk', '.gii') + fs_extensions))
        or (isinstance(geometry, dict) and 'faces' in geometry)
        ) else False

def read_vol(
    vol: Union[str, Path, TetMesh, dict]
) -> TetMesh:
    """
    Load and validate a tetrahedral volume mesh.

    Parameters
    ----------
    vol : str, Path, TetMesh, or dict
        Volume mesh specified as a file path (string or Path) to a VTK (.tetra.vtk) file, an
        instance of `lapy.TetMesh`, or a dictionary with `'vertices'` and `'tetras'` keys,
        referencing arrays of shape (n_verts, 3) and (n_tets, 4), respectively.

    Returns
    -------
    lapy.TetMesh
        Validated volume mesh with vertices and tetrahedra.

    Raises
    ------
    ValueError
        If `vol` is not a path-like string to a valid VTK (`.tetra.vtk`) file, an instance of
        `lapy.TetMesh`, or a dictionary with `'vertices'` and `'tetras'` keys.
    """
    if isinstance(vol, TetMesh):
        return vol
    elif isinstance(vol, dict):
        _check_mesh_dict(vol)
        return TetMesh(v=vol['vertices'], t=vol['tetras'])
    else:
        vol_str = str(vol)
        if not Path(vol_str).is_file():
            raise ValueError(f"Volume data not found: {vol_str}")
        if vol_str.endswith('.tetra.vtk'):
            # Load with lapy
            return TetMesh.read_vtk(str(vol))
    raise ValueError("`vol` must be a path-like string to a valid VTK (.tetra.vtk) file, an "
                    "instance of `lapy.TetMesh`, or a dictionary with 'vertices' and 'tetras' "
                    "keys.")

def read_surf(
    surf: Union[str, Path, GiftiImage, Trimesh, TriaMesh, dict]
) -> Trimesh:
    """Load a triangular surface mesh.

    Parameters
    ----------
    surf : str, Path, GiftiImage, trimesh.Trimesh, lapy.TriaMesh, or dict
        Surface mesh specified as a file path (string or Path) to a VTK (.vtk), GIFTI (.gii), or
        FreeSurfer file (.white, .pial, .inflated, .orig, .sphere, .smoothwm, .qsphere, .fsaverage),
        an instance of `nibabel.GiftiImage`, `trimesh.Trimesh`, or `lapy.TriaMesh`, or a dictionary
        with `'vertices'` and `'faces'` keys, referencing arrays of shapes (n_verts, 3) and
        (n_faces, 3), respectively.

    Returns
    -------
    trimesh.Trimesh
        Surface mesh with vertices and faces.

    Raises
    ------
    ValueError
        If `surf` is a path-like string to an unsupported format.
    ValueError
        If `surf` is a path-like string to a file that does not exist.
    """
    if isinstance(surf, Trimesh):
        return surf
    elif isinstance(surf, TriaMesh):
        vertices=surf.v
        faces=surf.t
    elif isinstance(surf, GiftiImage):
        vertices=surf.darrays[0].data
        faces=surf.darrays[1].data
    elif isinstance(surf, dict):
        _check_mesh_dict(surf)
        vertices=surf['vertices']
        faces=surf['faces']
    else:
        surf_str = str(surf)
        # check that file exists
        if not Path(surf_str).is_file():
            raise ValueError(f'File not found: {surf_str}')
        # Handle different file types
        if surf_str.endswith('.vtk'):
            surf_lapy = TriaMesh.read_vtk(surf_str)
            vertices=surf_lapy.v
            faces=surf_lapy.t
        elif surf_str.endswith('.gii'):
            surf_data = cast(GiftiImage, load(surf_str)).darrays
            vertices=surf_data[0].data
            faces=surf_data[1].data
        elif surf_str.endswith(fs_extensions):
            vertices, faces = read_geometry(
                surf_str, read_metadata=False, read_stamp=False
                ) # will only return two outputs now # type: ignore
        else:
            raise ValueError(
                '`surf` must be a path-like string to a valid VTK (.vtk), GIFTI (.gii), or '
                f'FreeSurfer file {fs_extensions}, an instance of `trimesh.Trimesh` or '
                '`lapy.TriaMesh`, or a dictionary of `faces` and `vertices`.'
                )
        
    return Trimesh(vertices=vertices, faces=faces, process=False)

def mask_vol(
    vol: TetMesh,
    mask: ArrayLike
) -> TetMesh:
    """
    Remove specified vertices and corresponding tetrahedra from the volume mesh. Returns a
    `lapy.TetMesh` object.

    Parameters
    ----------
    vol : lapy.TetMesh
        The input volume mesh.
    mask : array-like
        A boolean array indicating which vertices to keep (`True`) or remove (`False`).

    Returns
    -------
    lapy.TetMesh
        The masked volume mesh.

    Raises
    ------
    ValueError
        If `mask` does not have a length matching the number of vertices in `vol`.
    """
    mask = np.asarray_chkfinite(mask, dtype=bool)

    if mask.shape != (vol.v.shape[0],):
        raise ValueError(f"`mask` must have shape (n_verts,) = ({(vol.v.shape[0],)},).")
    
    # Remove vertices not in mask
    vol.v = vol.v[mask]

    # Update vertex indices of tetrahedra (-1 for removed vertices)
    v_map = np.zeros(mask.shape[0]) - 1
    v_map[mask] = np.arange(np.sum(mask))
    vol.t = v_map[vol.t]
    
    # Keep only tetrahedra where all vertices are in the mask
    tet_mask = np.all(vol.t != -1, axis=1)
    vol.t = vol.t[tet_mask]

    return vol

def mask_surf(
    surf: Trimesh,
    mask: ArrayLike
) -> Trimesh:
    """
    Remove specified vertices and corresponding faces from the surface mesh. Returns a 
    `trimesh.Trimesh` object.

    Parameters
    ----------
    surf : trimesh.Trimesh
        The input surface mesh.
    mask : array-like
        A boolean array indicating which vertices to keep (`True`) or remove (`False`).

    Returns
    -------
    trimesh.Trimesh
        The masked surface mesh.

    Raises
    ------
    ValueError
        If `mask` does not have a length matching the number of vertices in `surf`.

    Notes
    -----
    In `Trimesh.submesh`, `repair=False` is used to avoid an unnecessary dependency on 
    `networkx`. Mesh validation is handled separately in `check_surf` in `EigenSolver`.
    """
    mask = np.asarray_chkfinite(mask, dtype=bool)

    if mask.shape != (surf.vertices.shape[0],):
        raise ValueError(f"`mask` must have shape (n_verts,) = ({surf.vertices.shape[0]},).")
    
    # Keep only faces where all vertices are in the mask
    face_mask = np.all(mask[surf.faces], axis=1)
    return surf.submesh([face_mask], repair=False)[0] #type: ignore; submesh returns list by default

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
        If the surface boundary of the volume mesh is not contiguous.
    """
    if vol.has_free_vertices():
        raise ValueError('Volume mesh contains unreferenced vertices (i.e., not part of any '
                         'tetrahedron).')

    # Check that surface boundary of volume is contiguous
    vol_surf = vol.boundary_tria()
    vol_surf.orient_()

    try:
        check_surf(Trimesh(vertices=vol_surf.v, faces=vol_surf.t))
    except ValueError as e:
        # Adjust error message to specify that the issue is with the surface boundary of the volume
        first_word, rest = str(e).split(' ', 1)
        raise ValueError(
            f'{first_word} boundary of the volume {rest}.'
            ) from e

def check_surf(
    surf: Trimesh
) -> None:
    """
    Check if the surface mesh is contiguous with no unreferenced vertices.
    
    Parameters
    ----------
    surf : trimesh.Trimesh
        The surface mesh to check.

    Raises
    ------
    ValueError
        If the surface mesh contains unreferenced vertices.
    ValueError
        If the surface mesh is not contiguous.
    """
    # Check for unreferenced vertices
    referenced = np.zeros(len(surf.vertices), dtype=bool)
    referenced[surf.faces] = True
    if not np.all(referenced):
        raise ValueError(f'Surface mesh contains {np.sum(~referenced)} unreferenced '
                         'vertices (i.e., not part of any face).')

    # Ensure surface is contiguous
    n_components = surf.body_count
    if n_components != 1:
        raise ValueError(f'Surface mesh is not contiguous: {n_components} connected components '
                         'found.')

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

    try:
        with as_file(data_dir / file_name) as fpath:
            vol = read_vol(fpath)
            return vol
    except Exception as e:
        raise ValueError(
            f"Volume data not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv for a"
            " list of available volumes."
            ) from e

def fetch_surf(
    species: str = 'human',
    density: str = '32k',
    hemi: str = 'L',
    surf_type: str = 'midthickness',
    template: str = 'fsLR'
) -> Tuple[Trimesh, NDArray]:
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
    surf : trimesh.Trimesh
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
    except Exception as e:
        raise ValueError(
            f"Surface data not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv for a"
            " list of available surfaces."
            ) from e

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
    
    except Exception as e:
        raise ValueError(
            f"Map '{filename}' not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv for a"
            " list of available data files."
        ) from e
    
def _check_mesh_dict(
    mesh_dict: dict
) -> None:
    """
    Check that a dictionary has the required keys and value shapes for a surface or volume mesh.

    Parameters
    ----------
    mesh_dict : dict
        The mesh dictionary to check.

    Raises
    ------
    ValueError
        If the dictionary does not have keys 'vertices' and either 'faces' or 'tetras'.
    ValueError
        If the 'vertices' key does not reference an array of shape (n_verts, 3).
    ValueError
        If the 'faces' key (for surfaces) does not reference an array of shape (n_faces, 3), or if
        the 'tetras' key (for volumes) does not reference an array of shape (n_tets, 4).
    """
    # Check for required keys
    if 'vertices' not in mesh_dict or ('faces' not in mesh_dict and 'tetras' not in mesh_dict):
        raise ValueError("Mesh dictionary must contain keys 'vertices' and either 'faces' or "
                         "'tetras' for surface and volumes meshes, respectively.")
    
    # Check shapes of vertices
    verts = np.asarray_chkfinite(mesh_dict['vertices'])
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("Mesh dictionary key 'vertices' must reference an array-like with shape "
                         f"(n_verts, 3), received {verts.shape}.")

    # Infer surface vs volume
    if 'faces' in mesh_dict:
        elems_type = 'faces'
        elems_verts = 3
    else:  # 'tetras'
        elems_type = 'tetras'
        elems_verts = 4

    # Check shapes of faces/tetras
    elems = np.asarray_chkfinite(mesh_dict[elems_type])
    if elems.ndim != 2 or elems.shape[1] != elems_verts:
        raise ValueError(f"Mesh dictionary key '{elems_type}' must reference an array-like with "
                         f"shape (n_{elems_type}, {elems_verts}), received {elems.shape}.")

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