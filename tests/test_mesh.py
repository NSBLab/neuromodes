from pathlib import Path

import numpy as np
import pytest
from lapy import TetMesh, TriaMesh
from nibabel import Nifti1Image, load
from nibabel.affines import apply_affine

from neuromodes.mesh import (
    check_surf,
    check_vol,
    mask_mesh,
    nifti_data_to_tetmesh,
    nifti_mask_to_tetmesh,
    tetmesh_data_to_nifti,
)

DATA_DIR = Path(__file__).parent.parent / "neuromodes" / "data"

@pytest.fixture(scope="module")
def mask_meshes():
    return {
        name: nifti_mask_to_tetmesh(DATA_DIR / name)
        for name in ('hcp_tha-lh_thr25.nii.gz', 'hcp_hippo-lh_thr25.nii.gz')
    }

def _vox_coords(image):
    voxel_indices = np.column_stack(np.nonzero(image.get_fdata()))
    return apply_affine(image.affine, voxel_indices)

def test_nifti_mask_to_tetmesh(mask_meshes):
    for name, mesh in mask_meshes.items():
        image = load(DATA_DIR / name)
        voxel_coords = _vox_coords(image)

        # check that all nonzero voxels are represented in the mesh
        assert all(
            np.any(np.all(np.isclose(mesh.v, voxel_coord), axis=1))
            for voxel_coord in voxel_coords
        )

        # check that all mesh vertices are within one voxel's width of a nonzero voxel
        vox_width = image.header.get_zooms()[0]  # isotropic; 2 (mm) for included meshes
        voxel_coords_index = apply_affine(np.linalg.inv(image.affine), voxel_coords)
        mesh_coords_index = apply_affine(np.linalg.inv(image.affine), mesh.v)
        nearest_distances = np.min(
            np.linalg.norm(
                mesh_coords_index[:, None, :] - voxel_coords_index[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        assert np.max(nearest_distances) <= vox_width

def test_nifti_mask_to_tetmesh_faulty_return():

    with pytest.warns(UserWarning, match="2 connected components"):
        mesh, labels = nifti_mask_to_tetmesh(DATA_DIR / "hcp_striatum-lh_thr25.nii.gz")

    assert isinstance(mesh, TriaMesh)
    assert labels.shape == (mesh.v.shape[0],)
    assert np.unique(labels).size == 2

def test_nifti_mask_to_tetmesh_faulty_repair():
    path = DATA_DIR / "hcp_striatum-lh_thr25.nii.gz"

    # just check that either binary closing or discarding components runs without error
    nifti_mask_to_tetmesh(path, closings=1)
    nifti_mask_to_tetmesh(path, discard_components=True)

def test_nifti_data_to_tetmesh(mask_meshes):
    # Load hippocampus mask
    image = load(DATA_DIR / "hcp_hippo-lh_thr25.nii.gz")
    mask = image.get_fdata() != 0
    voxel_indices = np.column_stack(np.nonzero(mask))

    # Set data to physical-space x-coordinates
    data = np.zeros(image.shape)
    voxel_coords = apply_affine(image.affine, voxel_indices)
    data[mask] = voxel_coords[:, 0]
    data_image = Nifti1Image(data, image.affine, image.header)

    # Interpolate to tetmesh
    mesh = mask_meshes["hcp_hippo-lh_thr25.nii.gz"]
    mesh_data = nifti_data_to_tetmesh(data_image, mesh, image)

    # check that all nonzero voxels are represented in the mesh
    for voxel_coord in _vox_coords(image):
        vertex = np.flatnonzero(np.all(np.isclose(mesh.v, voxel_coord), axis=1))
        assert vertex.size == 1
        assert np.isclose(mesh_data[vertex[0]], voxel_coord[0])

    # check that interpolated data matches physical mesh vertex coordinates
    assert np.allclose(mesh_data, mesh.v[:, 0], atol=1e-6)

def test_nifti_data_to_tetmesh_multiple_maps(mask_meshes):
    # Load hippocampus mask and mesh
    hippo = "hcp_hippo-lh_thr25.nii.gz"
    image = load(DATA_DIR / hippo)
    mesh = mask_meshes[hippo]
    mask = image.get_fdata() != 0

    # Create data with two maps: physical-space x- and y-coordinates
    voxel_indices = np.column_stack(np.nonzero(mask))
    voxel_coords = apply_affine(image.affine, voxel_indices)
    data = np.zeros(image.shape + (2,))
    data[mask, 0] = voxel_coords[:, 0]
    data[mask, 1] = voxel_coords[:, 1]
    data_image = Nifti1Image(data, image.affine, image.header)

    # Interpolate to tetmesh and check that it matches NIFTI data
    mesh_data = nifti_data_to_tetmesh(data_image, mesh, image)
    assert mesh_data.shape == (mesh.v.shape[0], 2)
    assert np.allclose(mesh_data[:, 0], mesh.v[:, 0], atol=1e-6)
    assert np.allclose(mesh_data[:, 1], mesh.v[:, 1], atol=1e-6)

def test_tetmesh_data_to_nifti(mask_meshes):
    # Load hippocampus mask and mesh
    hippo = "hcp_hippo-lh_thr25.nii.gz"
    image = load(DATA_DIR / hippo)
    mesh = mask_meshes[hippo]

    # Create data on mesh and convert back to NIFTI
    data = np.arange(mesh.v.shape[0], dtype=float)
    output = tetmesh_data_to_nifti(data, mesh, image)

    # check that resampled data is a direct sampling of the mesh data
    voxel_indices = np.column_stack(np.nonzero(image.get_fdata()))
    voxel_coords = apply_affine(image.affine, voxel_indices)
    distances = np.linalg.norm(
        mesh.v[:, None, :] - voxel_coords[None, :, :],
        axis=2,
    )
    mesh_vertices = np.argmin(distances, axis=0)

    assert output.shape == image.shape
    assert np.array_equal(output.get_fdata()[tuple(voxel_indices.T)], data[mesh_vertices])

def test_tetmesh_data_to_nifti_multiple_maps(mask_meshes):
    # Load hippocampus mask and mesh
    hippo = "hcp_hippo-lh_thr25.nii.gz"
    image = load(DATA_DIR / hippo)
    mesh = mask_meshes[hippo]

    # Create data on mesh and convert back to NIFTI
    data = np.column_stack((mesh.v[:, 0], mesh.v[:, 1]))
    output = tetmesh_data_to_nifti(data, mesh, image)

    # check that resampled data is a direct sampling of the mesh data
    voxel_indices = np.column_stack(np.nonzero(image.get_fdata()))
    voxel_coords = apply_affine(image.affine, voxel_indices)
    distances = np.linalg.norm(
        mesh.v[:, None, :] - voxel_coords[None, :, :],
        axis=2,
    )
    mesh_vertices = np.argmin(distances, axis=0)

    assert output.shape == image.shape + (2,)
    assert np.allclose(
        output.get_fdata()[tuple(voxel_indices.T)],
        data[mesh_vertices],
    )

def test_tetmesh_data_to_nifti_aniso_affine():
    # Load hippocampus mask
    hippo = "hcp_hippo-lh_thr25.nii.gz"
    image = load(DATA_DIR / hippo)

    # Make affine anisotropic and create new mesh from it
    aniso_affine = image.affine.copy()
    aniso_affine[:3, :3] = aniso_affine[:3, :3] @ np.diag([1.5, 2.0, 0.75])
    aniso_mask = Nifti1Image(image.get_fdata(), aniso_affine, image.header)
    mesh = nifti_mask_to_tetmesh(aniso_mask)

    # Create data on mesh and convert back to NIFTI
    data = np.arange(mesh.v.shape[0], dtype=float)
    output = tetmesh_data_to_nifti(data, mesh, aniso_mask)

    # check that resampled data is a direct sampling of the mesh data
    voxel_indices = np.column_stack(np.nonzero(image.get_fdata()))
    voxel_coords = apply_affine(aniso_affine, voxel_indices)
    distances = np.linalg.norm(
        mesh.v[:, None, :] - voxel_coords[None, :, :],
        axis=2,
    )
    mesh_vertices = np.argmin(distances, axis=0)

    assert np.array_equal(
        output.get_fdata()[tuple(voxel_indices.T)],
        data[mesh_vertices],
    )

def test_tetmesh_data_to_nifti_unmatched_voxels():
    # Create a mask with voxels that do not have corresponding vertices in the mesh
    mask = np.zeros((2, 2, 2))
    mask[0, 0, 0] = 1
    mask[1, 0, 0] = 1
    image = Nifti1Image(mask, np.eye(4))
    mesh = TetMesh(
        v=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float),
        t=np.array([[0, 1, 2, 3]]),
    )

    with pytest.raises(ValueError, match="must correspond to a unique tetmesh vertex"):
        tetmesh_data_to_nifti(np.arange(4), mesh, image)

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
    with pytest.raises(ValueError, match="Surface mesh contains .* unreferenced"):
        check_surf(invalid_mesh)

def test_surf_not_contiguous():
    # Create two separate triangles (disconnected components)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]])
    cells = np.array([[0, 1, 2], [3, 4, 5]])  # Two separate triangles
    disconnected_mesh = TriaMesh(v=vertices, t=cells)
    
    # check_surf should raise ValueError due to multiple components
    with pytest.raises(ValueError, match="Surface mesh is not contiguous.*connected components"):
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
    with pytest.raises(ValueError, match="Volume mesh contains unreferenced vertices"):
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
    with pytest.raises(ValueError,
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
    
    with pytest.raises(ValueError, match="Volume mesh is not manifold"):
        check_vol(vol)