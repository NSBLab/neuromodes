import numpy as np
import pytest
from lapy import TetMesh, TriaMesh

from neuromodes.mesh import check_surf, check_vol, mask_mesh


def verts(n, seed=0): return np.random.default_rng(seed).random((n, 3))

def test_mask_surf():

    cells = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [1, 2, 4],
        [1, 3, 5]
    ])

    surf = TriaMesh(v=verts(6), t=cells)

    mask = np.array([True, True, True, True, False, False])

    masked_surf = mask_mesh(surf, mask)

    # Only two triangles should remain
    assert masked_surf.v.shape[0] == 4
    assert (masked_surf.t == [[0, 1, 2], [0, 2, 3]]).all()

def test_surf_unreferenced_verts():
    # Create an invalid surface mesh with unreferenced vertices (the last one)
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    
    invalid_mesh = TriaMesh(v=verts(5), t=cells)
   
    # check_surf should raise ValueError due to unreferenced vertex
    with pytest.raises(ValueError, match="Surface mesh contains .* unreferenced"):
        check_surf(invalid_mesh)

def test_surf_not_contiguous():
    # Create two separate triangles (disconnected components)
    cells = np.array([[0, 1, 2], [3, 4, 5]])  # Two separate triangles
    disconnected_mesh = TriaMesh(v=verts(6), t=cells)
    
    # check_surf should raise ValueError due to multiple components
    with pytest.raises(ValueError, match="Surface mesh is not contiguous.*connected components"):
        check_surf(disconnected_mesh)

def test_mask_vol():

    tets = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [1, 3, 4, 5],
    ])

    vol = TetMesh(v=verts(6), t=tets)

    mask = np.array([True, True, True, True, False, False])

    masked_vol = mask_mesh(vol, mask)

    # Only the first tetrahedron should remain
    assert masked_vol.v.shape[0] == 4
    assert (masked_vol.t == [0, 1, 2, 3]).all()

def test_vol_unreferenced_verts():
    # Create an invalid volume mesh with unreferenced vertices

    tets = np.array([
        [0, 1, 2, 3]
    ])

    vol = TetMesh(v=verts(5), t=tets)
   
    # check_vol should raise ValueError due to unreferenced vertex
    with pytest.raises(ValueError, match="Volume mesh contains unreferenced vertices"):
        check_vol(vol)

def test_vol_boundary_not_contiguous():
    # Create a volume mesh of two disconnected tetrahedra

    tets = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7]
    ])

    vol = TetMesh(v=verts(8), t=tets)
    
    # check_vol should raise ValueError due to multiple components
    with pytest.raises(ValueError, match="Surface mesh is not contiguous: 2 connected"):
        check_vol(vol)

def test_vol_nonmanifold():

    # Three tetrahedra share the same triangle (0, 1, 2)
    tets = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 2, 5]
    ])

    vol = TetMesh(v=verts(6), t=tets)
    
    with pytest.raises(ValueError, match="Volume mesh is not manifold"):
        check_vol(vol)

def test_mask_mesh_reindexes_cells():
    mesh = TriaMesh(
        v=verts(6),
        t=np.array([[0, 1, 2], [3, 4, 5]]),
    )
    masked, mapping = mask_mesh(
        mesh,
        np.array([False, False, False, True, True, True]),
        return_mapping=True,
    )
    assert np.array_equal(mapping, [3, 4, 5])
    assert np.array_equal(masked.t, [[0, 1, 2]])

def test_mask_mesh_returns_clean_mapping():
    mesh = TriaMesh(
        v=verts(4),
        t=np.array([[0, 1, 2]]),
    )
    masked, mapping = mask_mesh(
        mesh, np.ones(len(mesh.v), dtype=bool), return_mapping=True
    )
    assert len(masked.v) == 3
    assert np.array_equal(mapping, [0, 1, 2])
