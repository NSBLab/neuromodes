"""Utility functions for mesh generation and visualization."""

import numpy as np
import tetgen
import plotly.graph_objs as go
from lapy import TetMesh
from matplotlib import colormaps
from typing import Union, TYPE_CHECKING
from nibabel import Nifti1Image, load
from scipy.interpolate import griddata
from trimesh.voxel.ops import matrix_to_marching_cubes
import gmsh
from nibabel.affines import apply_affine

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from pathlib import Path

def make_thin_vol(surface_mesh, scaling=0.99, **kwargs):
    """
    Create a thin shell tetrahedral volume mesh from a surface mesh.
    
    Parameters
    ----------
    surface_mesh : trimesh.Trimesh
        The outer surface mesh.
    scaling : float, optional
        Scale factor for the inner surface (default: 0.99).
    **kwargs
        Additional keyword arguments for tetgen.TetGen.tetrahedralize().
    
    Returns
    -------
    tet_mesh : lapy.TetMesh
        Tetrahedral mesh of the shell volume.
    """
    # Create inner mesh
    inner_mesh = surface_mesh.copy()
    inner_mesh.apply_scale(scaling)
    
    # Combine vertices and faces
    vertices = np.vstack([surface_mesh.vertices, inner_mesh.vertices])
    outer_faces = surface_mesh.faces
    inner_faces = inner_mesh.faces + len(surface_mesh.vertices)
    faces = np.vstack([outer_faces, inner_faces])
    
    # Tetrahedralize
    tet = tetgen.TetGen(vertices, faces)
    tet.tetrahedralize(**kwargs)
    
    # Extract tetrahedral mesh
    vol_vertices = tet.grid.points
    vol_tets = tet.grid.cells.reshape(-1, 5)[:, 1:5].astype(int)
    
    return TetMesh(v=vol_vertices, t=vol_tets)

def plot_mesh_data(geometry, data, cmap='seismic_r', cnorm=True, width=700, height=700):
    """
    Plot a colored mesh surface with overlaid edges.
    
    Parameters
    ----------
    geometry : lapy.TriaMesh
        The surface mesh geometry.
    data : ndarray
        Data values (n_vertices,).
    cmap : str, optional
        Matplotlib colormap name (default: 'seismic_r').
    cnorm : bool, optional
        If data is 2D, whether to normalize color scale across frames (default: True).
    width : int, optional
        Figure width in pixels (default: 700).
    height : int, optional
        Figure height in pixels (default: 700).
    """
    # Make colormap for plotly
    cmap_obj = colormaps.get_cmap(cmap)
    vals = np.linspace(0, 1, 256)
    colorscale = [
        [i / (256 - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
        for i, (r, g, b, _) in enumerate(cmap_obj(vals))
    ]

    x, y, z = geometry.v.T
    i, j, k = geometry.t.T

    # Edge coordinates (static, shared across frames)
    edges = []
    for idx_i, idx_j, idx_k in zip(i, j, k):
        edges.append((idx_i, idx_j))
        edges.append((idx_j, idx_k))
        edges.append((idx_k, idx_i))
    edges = list(set(edges))

    edge_x, edge_y, edge_z = [], [], []
    for idx_i, idx_j in edges:
        edge_x.extend([x[idx_i], x[idx_j], None])
        edge_y.extend([y[idx_i], y[idx_j], None])
        edge_z.extend([z[idx_i], z[idx_j], None])

    is_animated = data.ndim == 2
    if is_animated:
        n_vertices, n_frames = data.shape
        if n_vertices != len(x):
            raise ValueError("data shape[0] must match number of vertices.")
        cmin, cmax = (np.nanmin(data), np.nanmax(data)) if cnorm else (np.nanmin(data[:, 0]), np.nanmax(data[:, 0]))

        # Base frame (first timepoint)
        base_mesh = go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=data[:, 0],
            colorscale=colorscale,
            cmin=cmin, cmax=cmax,
            flatshading=False,
            showscale=False,
        )


        frames = [
            go.Frame(
                data=[
                    go.Mesh3d(
                        x=x, y=y, z=z, i=i, j=j, k=k,
                        intensity=data[:, t],
                        colorscale=colorscale,
                        cmin=(cmin if cnorm else np.nanmin(data[:, t])), cmax=(cmax if cnorm else np.nanmax(data[:, t])),
                        flatshading=False,
                        showscale=False,
                    )
                ],
                name=str(t),
            )
            for t in range(n_frames)
        ]

        fig = go.Figure(data=[base_mesh], frames=frames)
    else:
        cmin, cmax = np.nanmin(data), np.nanmax(data)
        fig = go.Figure(data=[
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                intensity=data,
                colorscale=colorscale,
                cmin=cmin, cmax=cmax,
                flatshading=False,
                showscale=False,
            )
        ])

    # Add static edge overlay
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False,
        hoverinfo='skip',
    ))

    # Layout + optional animation controls
    layout_kwargs = dict(
        width=width,
        height=height,
        scene=dict(aspectmode="data"),
    )

    if is_animated:
        slider_steps = [
            dict(method="animate", args=[[str(t)], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0))], label=str(t))
            for t in range(n_frames)
        ]
        sliders = [dict(active=0, pad=dict(t=50), steps=slider_steps)]
        play_pause = [
            dict(label="▶️ Play", method="animate",
                 args=[None, dict(frame=dict(duration=50, redraw=True),
                                  transition=dict(duration=0),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="⏸️ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                    mode="immediate")]),
        ]
        layout_kwargs.update(
            sliders=sliders,
            updatemenus=[dict(type="buttons", showactive=False, buttons=play_pause, x=0, y=0)],
        )

    fig.update_layout(**layout_kwargs)
    fig.show()

def project_tetmesh_data(
    nifti_input_filename: Union[str, Path],
    data: ArrayLike,
    tetmesh: TetMesh
) -> Nifti1Image:
    """
    Project data defined on a tetrahedral mesh to a volumetric NIFTI space. Modified from James
    Pang's original code in the `BrainEigenmodes` repository.
    """
    data = np.asarray_chkfinite(data)
    n_maps = data.shape[1]

    # prepare transformation
    ROI_data = load(nifti_input_filename)
    roi_data = ROI_data.get_fdata()
    inds_all = np.where(roi_data==1)
    xx = inds_all[0]
    yy = inds_all[1]
    zz = inds_all[2]

    points = np.zeros([xx.shape[0],4])
    points[:,0] = xx
    points[:,1] = yy
    points[:,2] = zz
    points[:,3] = 1

    # calculate transformation matrix
    T = _get_tkrvox2ras(ROI_data.shape, ROI_data.header.get_zooms())

    # apply transformation
    points2 = np.matmul(T, np.transpose(points))

    # initialize nifti output array
    new_shape = np.array(roi_data.shape)
    if roi_data.ndim>3:
        new_shape[3] = n_maps
    else:
        new_shape = np.append(new_shape, n_maps)
    new_data = np.zeros(new_shape)

    # perform interpolation of eigenmodes from tetrahedral surface space to volume space
    for map in range(0, n_maps):
        interpolated_data = griddata(tetmesh.v, data[:,map], np.transpose(points2[0:3,:]), method='linear')
        for ind in range(0, len(interpolated_data)):
            new_data[xx[ind],yy[ind],zz[ind],map] = interpolated_data[ind]

    return Nifti1Image(new_data, ROI_data.affine, header=ROI_data.header)

def _get_tkrvox2ras(
    voldim: NDArray,
    voxres: NDArray
) -> NDArray:
    """Generate transformation matrix to switch between tetrahedral and volume space. Modified from
    James Pang's original code in the `BrainEigenmodes` repository.

    Parameters
    ----------
    voldim : array (1x3)
        Dimension of the volume (number of voxels in each of the 3 dimensions)
    voxres : array (!x3)
        Voxel resolution (resolution in each of the 3 dimensions)

    Returns
    ------
    T : array (4x4)
        Transformation matrix
    """
    x_res, y_res, z_res = voxres
    x_dim, y_dim, z_dim = voldim

    return np.array([
        [-x_res, 0,      0,     x_res*x_dim/2 ],
        [0,      0,      z_res, -z_res*z_dim/2],
        [0,      -y_res, 0,     y_res*y_dim/2 ],
        [0,      0,      0,     1             ]
    ])

def make_vol_mesh(
    nifti: Union[str, Path, Nifti1Image]
) -> TetMesh:
    """
    TODO: Validate this implementation / test edge cases.
    Tetrahedral meshing using Gmsh's python API and marching cubes algorithm.
    Returns a lapy.TetMesh object.
    """
    # Get ROI from NIFTI
    if isinstance(nifti, (str, Path)):
        nifti = load(nifti)
    elif not isinstance(nifti, Nifti1Image):
        raise ValueError("nifti must be a Nifti1Image object or a path-like string to a valid "
                         "`.nii` or `.nii.gz` file.")
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

    return TetMesh(v=verts.astype(np.float64), t=tetras.astype(np.int32))