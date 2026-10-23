import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from lapy import TetMesh, TriaMesh
from pytest import raises

from neuromodes.io import (
    _cache_output,
    fetch_example_map,
    fetch_example_surf,
    fetch_example_vol,
    read_surf,
    read_vol,
)
from neuromodes.mesh import check_surf, check_vol, is_vol


def test_fetch_example_surf():
    for hemi in ['L', 'R']:
        for species in ['human', 'macaque', 'marmoset']:
            for res in ['4k', '32k']:
                if species != 'human' and res == '4k':
                    with raises(FileNotFoundError, match="Surface data .* not found"):
                        fetch_example_surf(species=species, hemi=hemi, res=res)
                    continue

                surf, medmask = fetch_example_surf(species=species, hemi=hemi, res=res)
                assert surf.v.shape[0] > 0
                assert surf.v.shape[1] == 3
                assert surf.t.shape[0] > 0
                assert surf.t.shape[1] == 3
                assert medmask.dtype == bool
                assert medmask.shape == (surf.v.shape[0],)

                check_surf(surf)  # Should not raise

def test_fetch_invalid_surf():
    with raises(FileNotFoundError, match="Surface data .* not found"):
        fetch_example_surf(structure='makessense')

def test_fetch_gradient():
    grad = fetch_example_map('fcgradient1')
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (32492,)

def test_fetch_invalid_map():
    with raises(FileNotFoundError, match="Map 'sp-human_tpl-fsLR_den-32k_hemi-L_panshifu.func.gii'.*"):
        fetch_example_map('panshifu')

def test_read_surf_dict():
    surf_data = {
        'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        'cells': [[0, 1, 2], [1, 2, 3]]
    }
    surf = read_surf(surf_data)
    assert isinstance(surf, TriaMesh)
    assert surf.v.shape == (4, 3)
    assert surf.t.shape == (2, 3)

def test_read_surf_vtk():
    vtk_surf = read_surf(
        Path(__file__).parent / 'test_data' / 'sp-human_tpl-fsaverage5_den-10k_hemi-L_midthickness.vtk'
        )

    assert isinstance(vtk_surf, TriaMesh)
    assert vtk_surf.v.shape == (10242, 3)
    assert vtk_surf.t.shape == (20480, 3)

def test_read_surf_invalid():
    invalid_path = Path(__file__).parent / 'test_data' / 'civilised_lunch.surf.vtk'
    with raises(FileNotFoundError, match="File not found: .*civilised_lunch.surf.vtk"):
        read_surf(invalid_path)

def test_read_surf_freesurfer():
    for structure in ['inflated', 'orig', 'pial', 'smoothwm', 'sphere', 'white']:
        fs_surf = read_surf(
            Path(__file__).parent / 'test_data' / f'fsaverage-lh.{structure}'
            )
         
        assert isinstance(fs_surf, TriaMesh)
        assert fs_surf.v.shape[0] > 100
        assert fs_surf.t.shape[0] > 100
        assert fs_surf.v.shape[1] == 3
        assert fs_surf.t.shape[1] == 3

def test_fetch_vol():
    # Check that we can load and validate everything
    for hemi in ['L', 'R']:
        for structure in ['thalamus', 'hippocampus', 'striatum']:
            vol = fetch_example_vol(structure=structure, hemi=hemi)
            check_vol(vol)  # Should not raise
        
        mus = fetch_example_vol('isocortex', species='mouse', template='AMBA', res='200um',
                                hemi=hemi)
        check_vol(mus)

def test_fetch_invalid_vol():
    with raises(FileNotFoundError, match="Volume data .* not found."):
        fetch_example_vol('chillybin')

def test_read_vol_dict():
    verts = ([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
    ])

    tets = ([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [1, 3, 4, 5],
    ])

    vol_data = {
        'vertices': verts,
        'cells': tets
    }

    vol = read_vol(vol_data)
    assert isinstance(vol, TetMesh)
    assert vol.v.shape == (6, 3)
    assert vol.t.shape == (3, 4)

def test_read_vol_vtk():
    filename = 'sp-human_tpl-MNI152_res-2mm_hemi-L_thalamus.tetra.vtk'
    vtk_vol = read_vol(Path(__file__).parent.parent / 'neuromodes' / 'data' / filename)

    assert isinstance(vtk_vol, TetMesh)
    assert vtk_vol.v.shape == (1557, 3)
    assert vtk_vol.t.shape == (5755, 4)

def test_read_vol_invalid():
    invalid_path = Path(__file__).parent / 'test_data' / 'fossilised_lunch.tetra.vtk'
    with raises(FileNotFoundError, match="Volume data not found: .*fossilised_lunch.tetra.vtk"):
        read_vol(invalid_path)

# TODO: also test dict reading by just converting other formats to dict
def test_mesh_dict():
    # Volume case
    vol = {
        'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'cells': [[0, 1, 2, 3]]
    }
    assert is_vol(vol), "is_vol should return True for a valid volume dictionary"
    vol_tetmesh = read_vol(vol)
    assert vol_tetmesh.t.shape == (1, 4), \
        "read_vol should return a TetMesh with the correct tetrahedral connectivity"
    check_vol(vol_tetmesh)

    # Missing tetras
    vol_invalid = {
        'vertices': vol['vertices']
    }
    with raises(ValueError, match="Received an invalid dictionary for `geometry`."):
        is_vol(vol_invalid)

    # Wrong shape
    vol_invalid = {
        'vertices': vol['vertices'],
        'cells': [[0, 1, 2]]  # Should have 4 indices for tetras
    }

    with raises(IndexError):
        read_vol(vol_invalid)  # LaPy should raise as this can't become a TetMesh

    # Surface case
    surf = {
        'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'cells': [[0, 1, 2], [0, 2, 3]]
    }

    assert not is_vol(surf)
    surf_triamesh = read_surf(surf)
    assert surf_triamesh.t.shape == (2, 3), \
        "read_surf should return a TriaMesh with the correct triangular connectivity"
    check_surf(surf_triamesh)

    # Missing cells
    surf_invalid = {
        'vertices': surf['vertices']
    }
    with raises(ValueError, match="Received an invalid dictionary for `geometry`."):
        is_vol(surf_invalid)

    # Wrong shape
    geom_invalid = {
        'vertices': surf['vertices'],
        'cells': [[0, 1], [0, 2]]  # Should have 3 or 4 indices for cells
    }
    with raises(ValueError, match="Received an invalid dictionary for `geometry`."):
        is_vol(geom_invalid)

def test_cache_output_with_temp_dir():
    # Test with temporary directory and a simple function
    with TemporaryDirectory() as temp_cache_dir:
        def add_one(x):
            return x + 1
        cached_func = _cache_output(add_one, cache_dir=temp_cache_dir)
        assert callable(cached_func)
        assert cached_func(2) == 3

def test_cache_output_default_dir(capsys):
    # Temporarily unset CACHE_DIR
    cache_dir = os.getenv("CACHE_DIR")
    if "CACHE_DIR" in os.environ:
        del os.environ["CACHE_DIR"]

    try:
        def add_two(x):
            return x + 2
        cached_func = _cache_output(add_two)
        expected_dir = Path.home() / ".neuromodes_cache"
        assert callable(cached_func)
        assert cached_func(2) == 4

        print_log = capsys.readouterr().out
        assert f"Using cache directory at {expected_dir}" in print_log
    finally:
        # Restore original CACHE_DIR
        if cache_dir is not None:
            os.environ["CACHE_DIR"] = cache_dir

def test_cache_output_caches_result(tmp_path):
    calls = []
    def func(x):
        calls.append(x)
        return x * 2

    cached_func = _cache_output(func, cache_dir=tmp_path)
    # First call: should append to calls
    assert cached_func(5) == 10
    assert calls == [5]
    # Second call: should NOT append to calls (uses cache)
    assert cached_func(5) == 10
    assert calls == [5]  # No new call, so still [5]

def test_caching_no_joblib():
    # Mock the import of joblib to raise ImportError
    with (patch.dict('sys.modules', {'joblib': None}),
          raises(ImportError, match="joblib is required for caching")):
        _cache_output(lambda x: x)