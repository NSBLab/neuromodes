import numpy as np
from lapy import TetMesh, TriaMesh
from pytest import raises

from neuromodes.mesh import check_surf, check_vol, mask_mesh


def test_mask_surf():

    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1]
    ])

    cells = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [1, 2, 4],
        [1, 3, 5]
    ])

    vol = TriaMesh(v=verts, t=cells)

    mask = np.array([True, True, True, True, False, False])

    masked_vol = mask_mesh(vol, mask)

    # Only the first tetrahedron should remain
    assert masked_vol.v.shape[0] == 4
    assert (masked_vol.t == [[0, 1, 2], [0, 2, 3]]).all()

def test_surf_unreferenced_verts():
    # Create an invalid surface mesh with unreferenced vertices
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]])  # Last vertex unreferenced
    cells = np.array([[0, 1, 2], [0, 2, 3]])  # Only uses first 4 vertices, vertex 4 is unreferenced
    
    invalid_mesh = TriaMesh(v=vertices, t=cells)
   
    # check_surf should raise ValueError due to unreferenced vertex
    with raises(ValueError, match="Surface mesh contains .* unreferenced"):
        check_surf(invalid_mesh)

def test_surf_not_contiguous():
    # Create two separate triangles (disconnected components)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]])
    cells = np.array([[0, 1, 2], [3, 4, 5]])  # Two separate triangles
    disconnected_mesh = TriaMesh(v=vertices, t=cells)
    
    # check_surf should raise ValueError due to multiple components
    with raises(ValueError, match="Surface mesh is not contiguous.*connected components"):
        check_surf(disconnected_mesh)

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