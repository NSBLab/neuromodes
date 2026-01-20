"""
Module for reading, validating, and manipulating surface meshes.
"""

from __future__ import annotations
from importlib.resources import files, as_file
import os
from pathlib import Path
from typing import Union, Tuple, cast, TYPE_CHECKING
from warnings import warn
import gmsh
from lapy import TriaMesh, TetMesh
from nibabel.affines import apply_affine
from nibabel.freesurfer.io import read_geometry
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load
import numpy as np
from trimesh import Trimesh
from trimesh.voxel.ops import matrix_to_marching_cubes

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from nibabel import Nifti1Image

def is_vol(geometry) -> bool:
    return True if (isinstance(geometry, (str, Path))
                    and str(geometry).endswith(('.nii', '.nii.gz', '.tetra.vtk'))
        or isinstance(geometry, dict) and 'tetras' in geometry
        or isinstance(geometry, TetMesh)) else False

def read_vol(
    geometry: Union[str, Path, TetMesh, dict]
) -> TetMesh:
    """
    Tetrahedral meshing using Gmsh's python API.
    Returns a lapy.TetMesh object.
    """
    if isinstance(geometry, dict):
        tetras = geometry['tetras']
        verts = geometry['vertices']
        mesh = TetMesh(v=verts.astype(np.float64), t=tetras.astype(np.int32))
    elif isinstance(geometry, (str, Path)) and (geometry.endswith('.nii') or geometry.endswith('.nii.gz')):
        # Load binary NIFTI with ROI
        nifti = load(geometry)
        warn("Provided geometry is a NIFTI file. Generating tetrahedral mesh using Gmsh and "
             "marching cubes algorithm.")
        verts, tetras = make_tetra_mesh(nifti)
        mesh = TetMesh(v=verts.astype(np.float64), t=tetras.astype(np.int32))
    elif isinstance(geometry, (str, Path)) and (geometry.endswith('.vtk')):
        # Load with lapy
        mesh = TetMesh.read_vtk(str(geometry))
    elif isinstance(geometry, TetMesh):
        mesh = geometry
    else:
        raise ValueError("Unsupported volume geometry format. Provide a NIFTI (.nii, .nii.gz) or "
                         "VTK (.vtk) file path, or a dictionary with 'vertices' and 'tetras' keys.")

    return mesh

def make_tetra_mesh(
    nifti: Nifti1Image
) -> Tuple[NDArray, NDArray]:
    """
    Tetrahedral meshing using Gmsh's python API and marching cubes algorithm.
    Returns vertices and tetrahedra arrays.
    """
    # Get ROI from NIFTI
    vol = nifti.get_fdata()
    vol = (vol > 0).astype(np.uint8)

    # Marching cubes to extract surface (replacing mri_mc from FreeSurfer)
    surface = matrix_to_marching_cubes(vol)
    verts = apply_affine(nifti.affine, surface.vertices)
    faces = surface.faces

    # Gmsh tetrahedral meshing (replacing terminal commands to gmsh)
    gmsh.initialize()
    gmsh.model.add("brain")

    # Add points to gmsh
    point_tags = []
    for v in verts:
        tag = gmsh.model.geo.addPoint(v[0], v[1], v[2])
        point_tags.append(tag)

    # Add triangular faces
    triangle_tags = []
    for f in faces:
        l1 = gmsh.model.geo.addLine(point_tags[f[0]], point_tags[f[1]])
        l2 = gmsh.model.geo.addLine(point_tags[f[1]], point_tags[f[2]])
        l3 = gmsh.model.geo.addLine(point_tags[f[2]], point_tags[f[0]])

        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3])
        s = gmsh.model.geo.addPlaneSurface([cl])
        triangle_tags.append(s)

    # Create surface loop and volume
    sl = gmsh.model.geo.addSurfaceLoop(triangle_tags)
    gmsh.model.geo.addVolume([sl])

    gmsh.model.geo.synchronize()

    # Match James' Gmsh settings
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    
    gmsh.model.mesh.generate(3)

    # Get mesh data
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElements()

    # Extract tetrahedra
    elem_types, _, elem_nodes = elements

    tetra_nodes = None

    for etype, nodes in zip(elem_types, elem_nodes):
        if etype == 4:  # Gmsh tetrahedron element type
            tetra_nodes = nodes.reshape(-1, 4)

    verts = node_coords.reshape(-1, 3)
    tetras = tetra_nodes - 1   # gmsh uses 1-based indexing

    return verts, tetras

def read_surf(
    mesh: Union[str, Path, Trimesh, TriaMesh, dict]
) -> Trimesh:
    """Load and validate a surface mesh.

    Parameters
    ----------
    mesh : str, Path, trimesh.Trimesh, lapy.TriaMesh, or dict
        Surface mesh specified as a file path (string or Path) to a VTK (.vtk), GIFTI (.gii), or
        FreeSurfer file (.white, .pial, .inflated, .orig, .sphere, .smoothwm, .qsphere, .fsaverage),
        an instance of `trimesh.Trimesh` or `lapy.TriaMesh`, or a dictionary with `vertices` and
        `faces` keys.

    Returns
    -------
    trimesh.Trimesh
        Validated surface mesh with vertices and faces.

    Raises
    ------
    ValueError
        If `mesh` is a path-like string to an unsupported format.
    ValueError
        If `mesh` is a path-like string to a file that does not exist.
    """
    if isinstance(mesh, Trimesh):
        trimesh = mesh
    elif isinstance(mesh, TriaMesh):
        trimesh = Trimesh(vertices=mesh.v, faces=mesh.t)
    elif isinstance(mesh, dict):
        trimesh = Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
    else:
        mesh_str = str(mesh)
        # check that file exists
        if not Path(mesh_str).is_file():
            raise ValueError(f'File not found: {mesh_str}')

        # Handle different file types
        if mesh_str.endswith('.vtk'):
            mesh_lapy = TriaMesh.read_vtk(mesh_str)
            trimesh = Trimesh(vertices=mesh_lapy.v, faces=mesh_lapy.t)
        elif mesh_str.endswith('.gii'):
            mesh_data = cast(GiftiImage, load(mesh_str)).darrays
            trimesh = Trimesh(vertices=mesh_data[0].data, faces=mesh_data[1].data)
        elif mesh_str.endswith(
            ('white', 'pial', 'inflated', 'orig', 'sphere', 'smoothwm', 'qsphere', 'fsaverage')
            ):
            vertices, faces = read_geometry(
                mesh_str, read_metadata=False, read_stamp=False
                ) # will only return two outputs now # type: ignore
            trimesh = Trimesh(vertices=vertices, faces=faces) # type: ignore
        else:
            raise ValueError(
                '`surf` must be a path-like string to a valid VTK (.vtk), GIFTI (.gii), or '
                'FreeSurfer file (.white, .pial, .inflated, .orig, .sphere, .smoothwm, .qsphere, '
                '.fsaverage), an instance of `trimesh.Trimesh` or `lapy.TriaMesh`, or a dictionary '
                'of `faces` and `vertices`.'
                )
    
    # Validate the mesh before returning
    check_surf(trimesh)

    return trimesh

def mask_surf(
    surf: Trimesh,
    mask: ArrayLike
) -> Trimesh:
    """
    Remove specified vertices and corresponding faces from the surface mesh. Returns a validated 
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
    
    
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.shape != (surf.vertices.shape[0],):
        raise ValueError(f"`mask` must have shape {(surf.vertices.shape[0],)} to match the number "
                         "of vertices in the surface mesh.")
    
    # Mask faces where all vertices are in the mask
    face_mask = np.all(mask[surf.faces], axis=1)
    mesh = surf.submesh([face_mask])[0] #type: ignore # submesh returns a list by default

    # Validate the mesh before returning
    check_surf(mesh)
    
    return mesh

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

    # Check if the mesh is contiguous
    n_components = surf.body_count
    if n_components != 1:
        raise ValueError(f'Surface mesh is not contiguous: {n_components} connected components '
                         'found.')

def fetch_surf(
    species: str = 'human',
    density: str = '32k',
    hemi: str = 'L',
    surf: str = 'midthickness',
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
    surf : str, optional
        Surface type to load. Currently only supports `'midthickness'`. Default is `'midthickness'`.
    template : str, optional
        Template of the surface mesh. Currently only supports `'fsLR'`. Default is `'fsLR'`.
    
    Returns
    -------
    mesh : trimesh.Trimesh
        The loaded surface mesh.
    medmask : np.ndarray
        The medial wall mask as a boolean array.

    Raises
    ------
    ValueError
        If the specified surface data is not found in the `neuromodes/data` directory.
    """
    data_dir = files('neuromodes.data')
    meshname = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_{surf}.surf.gii'
    maskname = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_medmask.label.gii'

    try:
        with as_file(data_dir / meshname) as fpath:
            mesh = read_surf(fpath)
        with as_file(data_dir / maskname) as fpath:
            medmask = cast(GiftiImage, load(fpath)).darrays[0].data.astype(bool)
        
        return mesh, medmask
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

def _set_cache():
    """
    Set up joblib memory caching.
    
    Returns
    -------
    joblib.Memory
        Configured joblib Memory object for caching.

    Raises
    ------
    ImportError
        If `joblib` is not installed.
    """
    try:
        from joblib import Memory
    except ImportError:
        raise ImportError(
            "joblib is required for caching. Please install it via 'pip install joblib' (or "
            "install neuromodes with UV via 'uv add \"neuromodes[cache] @ "
            "git+https://github.com/NSBLab/neuromodes.git\")."
        )

    CACHE_DIR = os.getenv("CACHE_DIR")
    if CACHE_DIR is None:
        CACHE_DIR = Path.home() / ".neuromodes_cache"
        print(f"Using default cache directory at {CACHE_DIR}. To cache elsewhere, set the CACHE_DIR"
              " environment variable beforehand.")
    else:
        CACHE_DIR = Path(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return Memory(CACHE_DIR, verbose=0)