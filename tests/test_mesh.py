import pytest
import numpy as np
from lapy import TriaMesh

from neuromodes.mesh import mask_mesh, check_surf

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

    surf = TriaMesh(v=verts, t=faces)

    mask = np.array([True, True, True, True, False, False])

    masked_surf = mask_mesh(surf, mask)

    # Only the first two triangles should remain
    assert masked_surf.v.shape[0] == 4
    assert (masked_surf.t == [[0, 1, 2], [0, 2, 3]]).all()

def test_surf_unreferenced_verts():
    # Create an invalid surface mesh with unreferenced vertices
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]])  # Last vertex unreferenced
    faces = np.array([[0, 1, 2], [0, 2, 3]])  # Only uses first 4 vertices, vertex 4 is unreferenced
    
    invalid_mesh = TriaMesh(v=vertices, t=faces)
   
    # check_surf should raise ValueError due to unreferenced vertex
    with pytest.raises(ValueError, match="Surface mesh contains .* unreferenced vertices"):
        check_surf(invalid_mesh)

def test_surf_not_contiguous():
    # Create two separate triangles (disconnected components)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])  # Two separate triangles
    disconnected_mesh = TriaMesh(v=vertices, t=faces)
    
    # check_surf should raise ValueError due to multiple components
    with pytest.raises(ValueError, match="Surface mesh is not contiguous.*connected components"):
        check_surf(disconnected_mesh)
