"""
Module for loading brain meshes and maps, as well as setting up caching.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from importlib.util import find_spec
from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from lapy import TetMesh, TriaMesh
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

fs_extensions = ('.white', '.pial', '.inflated', '.orig', '.sphere', '.smoothwm', '.qsphere',
                 '.fsaverage')

def read_vol(
    vol: str | Path | TetMesh | dict
) -> TetMesh:
    """
    Load and validate a tetrahedral volume mesh.

    Parameters
    ----------
    vol : str, Path, TetMesh, or dict
        Volume mesh specified as a file path (string or Path) to a VTK (.tetra.vtk) file, an
        instance of `lapy.TetMesh`, or a dictionary with `'vertices'` and `'cells'` keys,
        referencing arrays of shape (n_verts, 3) and (n_tetras, 4), respectively.

    Returns
    -------
    lapy.TetMesh
        Validated volume mesh with vertices and tetrahedra.

    Raises
    ------
    TypeError
        If `vol` is not a path-like string to a valid VTK (`.tetra.vtk`) file, an instance of
        `lapy.TetMesh`, or a dictionary with `'vertices'` and `'cells'` keys.
    """
    if isinstance(vol, TetMesh):
        return vol
    elif isinstance(vol, dict):
        return TetMesh(v=vol['vertices'], t=vol['cells'])
    else:
        vol_str = str(vol)
        if not Path(vol_str).is_file():
            raise FileNotFoundError(f"Volume data not found: {vol_str}")
        if vol_str.endswith('.tetra.vtk'):
            # Load with lapy
            return TetMesh.read_vtk(str(vol))
    raise TypeError("`vol` must be a path-like string to a valid VTK (.tetra.vtk) file, an "
                    "instance of `lapy.TetMesh`, or a dictionary with 'vertices' and 'cells' "
                    "keys.")

def read_surf(
    surf: str | Path | GiftiImage | TriaMesh | dict
) -> TriaMesh:
    """Load a triangular surface mesh.

    Parameters
    ----------
    surf : str, Path, GiftiImage, lapy.TriaMesh, or dict
        Surface mesh specified as one of:
        - a path (``str`` or ``Path``) to a VTK (``.vtk``), GIFTI (``.gii``), or FreeSurfer file
          (``.white``, ``.pial``, ``.inflated``, ``.orig``, ``.sphere``, ``.smoothwm``,``.qsphere``,
          ``.fsaverage``)
        - an instance of either ``nibabel.GiftiImage`` or ``lapy.TriaMesh``
        - a dictionary with ``'vertices'`` and ``'cells'`` keys, referencing arrays of shapes
        ``(n_verts, 3)`` and ``(n_trias, 3)``, respectively.

    Returns
    -------
    lapy.TriaMesh
        Surface mesh with vertices and cells.

    Raises
    ------
    TypeError
        If ``surf`` is not in a supported format.
    TypeError
        If ``surf`` is a path to an unsupported format.
    FileNotFoundError
        If ``surf`` is a path to a file that does not exist.
    """
    if isinstance(surf, TriaMesh):
        return surf
    elif isinstance(surf, GiftiImage):
        vertices=surf.darrays[0].data
        cells=surf.darrays[1].data
    elif isinstance(surf, dict):
        vertices=surf['vertices']
        cells=surf['cells']
    elif isinstance(surf, (str, Path)):
        surf_str = str(surf)
        # check that file exists
        if not Path(surf_str).is_file():
            raise FileNotFoundError(f'File not found: {surf_str}')
        # Handle different file types
        if surf_str.endswith('.vtk'):
            return TriaMesh.read_vtk(surf_str)
        elif surf_str.endswith('.gii'):
            return TriaMesh.read_gifti(surf_str)
        elif surf_str.endswith(fs_extensions):
            return TriaMesh.read_fssurf(surf_str)
        else:
            raise TypeError(
                f'File type not supported: {surf_str}. Supported formats include VTK (.vtk), GIFTI '
                f'(.gii), and FreeSurfer files ({", ".join(fs_extensions)})'
            )
    else:
        raise TypeError(
            'surf must be a path (str or Path) to a valid VTK (.vtk), GIFTI (.gii), or Freesurfer'
            f'file {fs_extensions}, an instance of nibabel.GiftiImage or lapy.TriaMesh, or a '
            "dictionary of 'cells' and 'vertices' with shapes (n_verts, 3) 'and (n_trias, 3), "
            'respectively.'
            )
        
    return TriaMesh(v=vertices, t=cells)

def fetch_example_surf(
    structure: Literal['midthickness'] = 'midthickness',
    species: Literal['human', 'macaque', 'marmoset'] = 'human',
    res: Literal['32k', '4k'] = '32k',
    hemi: Literal['L', 'R'] = 'L',
    template: Literal['fsLR'] = 'fsLR'
) -> tuple[TriaMesh, NDArray[np.floating]]:
    """
    Load a cortical triangular surface mesh and medial wall mask from the included package data. For
    a list of available surfaces, see ``neuromodes/data/included_data.csv`` or
    https://github.com/NSBLab/neuromodes/blob/main/neuromodes/data/included_data.csv.

    Parameters
    ----------
    structure : str, optional
        Brain structure to load. Currently supports ``'midthickness'`` for all species as well as
        ``'sphere'`` for ``'human'``. Default is ``'midthickness'``.
    species : str, optional
        Species of the surface mesh. Options include ``'human'``, ``'macaque'``, and ``'marmoset'``.
        Default is ``'human'``.
    res : str, optional
        Resolution of the surface mesh. Options include ``'32k'`` for all species, as well as ``'4k'`` for
        human. Default is ``'32k'``.
    hemi : str, optional
        Hemisphere of the surface mesh. Options are ``'L'`` for all species, as well as ``'R'`` for human.
        Default is ``'L'``.
    template : str, optional
        Template of the surface mesh. Currently only supports ``'fsLR'``. Default is ``'fsLR'``.
    
    Returns
    -------
    surf : lapy.TriaMesh
        The loaded surface mesh.
    medmask : np.ndarray
        The medial wall mask as a boolean array.

    Raises
    ------
    TypeError
        If the specified surface data is not found in the ``neuromodes/data`` directory.
    """
    data_dir = files('neuromodes.data')
    surf_name = f'sp-{species}_tpl-{template}_den-{res}_hemi-{hemi}_{structure}.surf.gii'
    mask_name = f'sp-{species}_tpl-{template}_den-{res}_hemi-{hemi}_medmask.label.gii'

    try:
        with as_file(data_dir / surf_name) as fpath:
            surf = read_surf(fpath)
        with as_file(data_dir / mask_name) as fpath:
            medmask = cast(GiftiImage, load(fpath)).darrays[0].data.astype(bool)
        
        return surf, medmask
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Surface data {surf_name} not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/blob/main/neuromodes/data/included_data.csv for a"
            " list of available surfaces."
            )

def fetch_example_vol(
    structure: Literal['thalamus', 'striatum', 'hippocampus', 'isocortex', '315'] = 'thalamus',
    species: Literal['human', 'mouse'] = 'human',
    res: Literal['2mm', '200um'] = '2mm',
    hemi: Literal['L', 'R'] = 'L',
    template: Literal['MNI152', 'AMBA'] = 'MNI152',
) -> TetMesh:
    """
    Load a tetrahedral volume mesh from neuromodes data directory. For a list of available volumes,
    see https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv.

    Parameters
    ----------
    structure : {'thalamus', 'striatum', 'hippocampus', 'isocortex', '315'}, optional
        Brain structure to load. Options include ``'thalamus'``, ``'striatum'``, and ``'hippocampus'`` for
        human and ``'isocortex'`` (alias for ``'315'``, the Allen Mouse Brain Atlas ID) for mouse.
    species : {'human', 'mouse'}, optional
        Species of the volume mesh. Currently only supports ``'human'`` and ``'mouse'``. Default is
        ``'human'``.
    res : {'2mm', '200um'}, optional
        Resolution of the volume mesh. Options include ``'2mm'`` for human and ``'200um'`` for mouse.
        Default is ``'2mm'``.
    hemi : {'L', 'R'}, optional
        Hemisphere of the volume mesh. Options are ``'L'`` and ``'R'``. Default is ``'L'``.
    template : {'MNI152', 'AMBA'}, optional
        Template of the volume mesh. Currently only supports ``'MNI152'`` and ``'AMBA'``. Default is
        ``'MNI152'``.

    Returns
    -------
    lapy.TetMesh
        The loaded volume mesh.
    """
    data_dir = files('neuromodes.data')
    if structure == 'isocortex':
        structure = '315'  # alias for Allen Mouse Brain Atlas ID
    file_name = f'sp-{species}_tpl-{template}_res-{res}_hemi-{hemi}_{structure}.tetra.vtk'

    try:
        with as_file(data_dir / file_name) as fpath:
            return read_vol(fpath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Volume data {file_name} not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/tree/main/neuromodes/data/included_data.csv for a"
            " list of available volumes."
            )

def fetch_example_map(
    data: Literal['fcgradient1', 'myelinmap', 'ndi', 'odi', 'thickness'],
    species: Literal['human', 'macaque', 'marmoset'] = 'human',
    res: Literal['32k', '4k'] = '32k',
    hemi: Literal['L', 'R'] = 'L',
    template: Literal['fsLR'] = 'fsLR'
) -> NDArray:
    """
    Load a cortical surface map from the included package data. For a list of available maps, see
    ``neuromodes/data/included_data.csv`` or
    https://github.com/NSBLab/neuromodes/blob/main/neuromodes/data/included_data.csv.

    Parameters
    ----------
    data : {'fcgradient1', 'myelinmap', 'ndi', 'odi', 'thickness'}
        Cortical map to load. Options include ``'fcgradient1'``, ``'myelinmap'``, ``'ndi'``,
        ``'odi'``, and ``'thickness'``.
    species : {'human'}, optional
        Species of the surface mesh. Currently only supports ``'human'``. Default is ``'human'``.
    res : {'32k', '4k'}, optional
        Resolution of the surface mesh. Currently supports ``'32k'`` for all maps as well as 
        ``'4k'`` for ``'myelinmap'``. Default is ``'32k'``.
    hemi : {'L', 'R'}, optional
        Hemisphere of the surface mesh. Options are ``'L'`` and ``'R'``. Default is ``'L'``.
    template : {'fsLR'}, optional
        Template of the surface mesh. Currently only supports ``'fsLR'``. Default is ``'fsLR'``.

    Returns
    -------
    np.ndarray
        The loaded cortical map data.

    Raises
    ------
    FileNotFoundError
        If the specified map data is not found in the ``neuromodes/data`` directory.
    """
    data_dir = files('neuromodes.data')
    filename = f'sp-{species}_tpl-{template}_den-{res}_hemi-{hemi}_{data}.func.gii'

    try:
        with as_file(data_dir / filename) as fpath:
            return cast(GiftiImage, load(fpath)).darrays[0].data
    
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Map '{filename}' not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/neuromodes/blob/main/neuromodes/data/included_data.csv for a"
            " list of available data files."
            )

def _cache_output(
    function: Callable,
    cache_dir: str | Path | None = None
) -> Callable:
    """
    Set up :class:`joblib.Memory` caching for a given function. The cache directory can be specified
    via ``cache_dir``, or by setting the ``CACHE_DIR`` environment variable. If neither is set,
    defaults to ``~/.neuromodes_cache``.
    
    Parameters
    ----------
    function : callable
        The function to be cached.
    cache_dir : str or Path, optional
        The directory to use for caching. If not provided, uses the ``CACHE_DIR`` environment
        variable. If ``CACHE_DIR`` is not set, defaults to ``~/.neuromodes_cache``.

    Returns
    -------
    callable
        The cached version of the input function.

    Raises
    ------
    ImportError
        If ``joblib`` is not installed.

    Raises
    ------
    ImportError
        If ``joblib`` is not installed.
    """
    if find_spec("joblib") is None:
        raise ImportError("joblib is required for caching. Neuromodes can be installed with the "
                          "'cache' extra to include joblib as a dependency (e.g., pip install "
                          "neuromodes[cache]).")
    from joblib import Memory
    
    if cache_dir is None:
        cache_dir = getenv("CACHE_DIR")
        if cache_dir is None:
            cache_dir = Path.home() / ".neuromodes_cache"
        print(f"Using cache directory at {cache_dir}. To cache elsewhere, set cache_dir.")  

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return Memory(cache_dir, verbose=0).cache(function)