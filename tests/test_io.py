import os
from pathlib import Path
from tempfile import TemporaryDirectory
from joblib import Memory
from lapy import TriaMesh, TetMesh
import numpy as np
from pytest import raises
from neuromodes.io import (
    is_vol, read_vol, read_surf, mask_mesh, check_vol, check_surf, fetch_vol, fetch_surf,
    fetch_map, _set_cache
    )

def test_mask_surf():

    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1]
    ])

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [1, 2, 4],
        [1, 3, 5]
    ])

    vol = TriaMesh(v=verts, t=faces)

    mask = np.array([True, True, True, True, False, False])

    masked_vol = mask_mesh(vol, mask)

    # Only the first tetrahedron should remain
    assert masked_vol.v.shape[0] == 4
    assert (masked_vol.t == [[0, 1, 2], [0, 2, 3]]).all()

def test_surf_unreferenced_verts():
    # Create an invalid surface mesh with unreferenced vertices
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]])  # Last vertex unreferenced
    faces = np.array([[0, 1, 2], [0, 2, 3]])  # Only uses first 4 vertices, vertex 4 is unreferenced
    
    invalid_mesh = TriaMesh(v=vertices, t=faces)
   
    # check_surf should raise ValueError due to unreferenced vertex
    with raises(ValueError, match="Surface mesh contains .* unreferenced vertices"):
        check_surf(invalid_mesh)

def test_surf_not_contiguous():
    # Create two separate triangles (disconnected components)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])  # Two separate triangles
    disconnected_mesh = TriaMesh(v=vertices, t=faces)
    
    # check_surf should raise ValueError due to multiple components
    with raises(ValueError, match="Surface mesh is not contiguous.*connected components"):
        check_surf(disconnected_mesh)

def test_fetch_surf():
    for hemi in ['L', 'R']:
        for species in ['human', 'macaque', 'marmoset']:
            for density in ['4k', '32k']:
                if species != 'human' and density == '4k':
                    with raises(ValueError, match="Surface data .* not found"):
                        fetch_surf(species=species, hemi=hemi, density=density)
                    continue

                surf, medmask = fetch_surf(species=species, hemi=hemi, density=density)
                assert surf.v.shape[0] > 0
                assert surf.v.shape[1] == 3
                assert surf.t.shape[0] > 0
                assert surf.t.shape[1] == 3
                assert medmask.dtype == bool
                assert medmask.shape == (surf.v.shape[0],)

                check_surf(surf)  # Should not raise

def test_fetch_invalid_surf():
    with raises(ValueError, match="Surface data .* not found"):
        fetch_surf(surf_type='makessense')

def test_fetch_gradient():
    grad = fetch_map('fcgradient1')
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (32492,)

def test_fetch_invalid_map():
    with raises(ValueError, match="Map 'sp-human_tpl-fsLR_den-32k_hemi-L_panshifu.func.gii'.*"):
        fetch_map('panshifu')

def test_read_surf_dict():
    surf_data = {
        'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        'faces': [[0, 1, 2], [1, 2, 3]]
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
    with raises(ValueError, match="File not found: .*civilised_lunch.surf.vtk"):
        read_surf(invalid_path)

def test_read_surf_freesurfer():
    for surf_type in ['inflated', 'orig', 'pial', 'smoothwm', 'sphere', 'white']:
        fs_surf = read_surf(
            Path(__file__).parent / 'test_data' / f'fsaverage-lh.{surf_type}'
            )
         
        assert isinstance(fs_surf, TriaMesh)
        assert fs_surf.v.shape[0] > 100
        assert fs_surf.t.shape[0] > 100
        assert fs_surf.v.shape[1] == 3
        assert fs_surf.t.shape[1] == 3

def test_mask_vol():
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
    ])

    tets = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [1, 3, 4, 5],
    ])

    vol = TetMesh(v=verts, t=tets)

    mask = np.array([True, True, True, True, False, False])

    masked_vol = mask_mesh(vol, mask)

    # Only the first tetrahedron should remain
    assert masked_vol.v.shape[0] == 4
    assert (masked_vol.t == [0, 1, 2, 3]).all()

def test_vol_unreferenced_verts():
    # Create an invalid volume mesh with unreferenced vertices
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 2, 2]  # Unreferenced vertex
    ])

    tets = np.array([
        [0, 1, 2, 3]
    ])

    vol = TetMesh(v=verts, t=tets)
   
    # check_vol should raise ValueError due to unreferenced vertex
    with raises(ValueError, match="Volume mesh contains unreferenced vertices"):
        check_vol(vol)

def test_vol_boundary_not_contiguous():
    # Create a volume mesh of two disconnected tetrahedra
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 0, 0],
        [3, 0, 0],
        [2, 1, 0],
        [2, 0, 1]
    ])

    tets = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7]
    ])

    vol = TetMesh(v=verts, t=tets)
    
    # check_vol should raise ValueError due to multiple components
    with raises(ValueError,
                match="Surface mesh is not contiguous: 2 connected"):
        check_vol(vol)

def test_vol_nonmanifold():
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1]
    ])

    # Three tetrahedra share the same triangle (0, 1, 2)
    tets = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 2, 5]
    ])

    vol = TetMesh(v=verts, t=tets)
    
    with raises(ValueError, match="Volume mesh is not manifold"):
        check_vol(vol)

def test_fetch_vol():
    # Check that we can load and validate everything
    for hemi in ['L', 'R']:
        for structure in ['thalamus', 'hippocampus', 'striatum']:
            vol = fetch_vol(structure=structure, hemi=hemi)
            check_vol(vol)  # Should not raise
        
        mus = fetch_vol('cortex', species='mouse', template='AMBA')
        check_vol(mus)

def test_fetch_invalid_vol():
    with raises(ValueError, match="Volume data .* not found."):
        fetch_vol('chillybin')

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
        'faces': tets
    }

    vol = read_vol(vol_data)
    assert isinstance(vol, TetMesh)
    assert vol.v.shape == (6, 3)
    assert vol.t.shape == (3, 4)

def test_read_vol_vtk():
    filename = 'sp-human_tpl-MNI152_hemi-L_thalamus.tetra.vtk'
    vtk_vol = read_vol(Path(__file__).parent.parent / 'neuromodes' / 'data' / filename)

    assert isinstance(vtk_vol, TetMesh)
    assert vtk_vol.v.shape == (1557, 3)
    assert vtk_vol.t.shape == (5755, 4)

def test_read_vol_invalid():
    invalid_path = Path(__file__).parent / 'test_data' / 'fossilised_lunch.tetra.vtk'
    with raises(ValueError, match="Volume data not found: .*fossilised_lunch.tetra.vtk"):
        read_vol(invalid_path)

# TODO: also test dict reading by just converting other formats to dict
def test_mesh_dict():
    # Volume case
    vol = {
        'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'faces': [[0, 1, 2, 3]]
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
        'faces': [[0, 1, 2]]  # Should have 4 indices for tetras
    }

    with raises(IndexError):
        read_vol(vol_invalid)  # LaPy should raise as this can't become a TetMesh

    # Surface case
    surf = {
        'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'faces': [[0, 1, 2], [0, 2, 3]]
    }

    assert not is_vol(surf)
    surf_triamesh = read_surf(surf)
    assert surf_triamesh.t.shape == (2, 3), \
        "read_surf should return a TriaMesh with the correct triangular connectivity"
    check_surf(surf_triamesh)

    # Missing faces
    surf_invalid = {
        'vertices': surf['vertices']
    }
    with raises(ValueError, match="Received an invalid dictionary for `geometry`."):
        is_vol(surf_invalid)

    # Wrong shape
    geom_invalid = {
        'vertices': surf['vertices'],
        'faces': [[0, 1], [0, 2]]  # Should have 3 or 4 indices for faces
    }
    with raises(ValueError, match="Received an invalid dictionary for `geometry`."):
        is_vol(geom_invalid)

def test_caching():
    # Get CACHE_DIR
    cache_dir = os.getenv("CACHE_DIR")

    # Test with temporary directory
    with TemporaryDirectory() as temp_cache_dir:
        os.environ["CACHE_DIR"] = temp_cache_dir
        
        memory = _set_cache()
        assert isinstance(memory, Memory)
        assert str(memory.location) == temp_cache_dir
    
    # Restore original CACHE_DIR
    if cache_dir is not None:
        os.environ["CACHE_DIR"] = cache_dir
    elif "CACHE_DIR" in os.environ:
        del os.environ["CACHE_DIR"]

def test_caching_default_dir(capsys):
    # Temporarily unset CACHE_DIR
    cache_dir = os.getenv("CACHE_DIR")
    if "CACHE_DIR" in os.environ:
        del os.environ["CACHE_DIR"]

    try:
        # Invoke _set_cache and check default directory
        memory = _set_cache()
        expected_dir = Path.home() / ".neuromodes_cache"

        assert isinstance(memory, Memory)
        assert memory.location == expected_dir

        print_log = capsys.readouterr().out
        assert f"Using default cache directory at {expected_dir}" in print_log
    finally:
        # Restore original CACHE_DIR
        if cache_dir is not None:
            os.environ["CACHE_DIR"] = cache_dir