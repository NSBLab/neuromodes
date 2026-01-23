"""Utility functions for mesh generation and visualization."""

import numpy as np
import tetgen
import plotly.graph_objs as go
from lapy import TetMesh
from matplotlib import colormaps

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

def plot_mesh_data(geometry, data, cmap='seismic_r', width=700, height=700):
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
    width : int, optional
        Figure width in pixels (default: 700).
    height : int, optional
        Figure height in pixels (default: 700).
    """
    # Make seismic_r colormap for plotly
    cmap = colormaps.get_cmap(cmap)
    vals = np.linspace(0, 1, 256)
    colorscale = [
        [i / (256 - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
        for i, (r, g, b, _) in enumerate(cmap(vals))
]

    x, y, z = geometry.v.T
    i, j, k = geometry.t.T
    
    # Create colored mesh surface
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=data,
            colorscale=colorscale,
            flatshading=False,
            showscale=False,
        )
    ])
    
    # Extract and add edges
    edges = []
    for idx_i, idx_j, idx_k in zip(i, j, k):
        edges.append((idx_i, idx_j))
        edges.append((idx_j, idx_k))
        edges.append((idx_k, idx_i))
    
    edges = list(set(edges))
    
    edge_x = []
    edge_y = []
    edge_z = []
    for idx_i, idx_j in edges:
        edge_x.extend([x[idx_i], x[idx_j], None])
        edge_y.extend([y[idx_i], y[idx_j], None])
        edge_z.extend([z[idx_i], z[idx_j], None])
    
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    fig.update_layout(
        width=width,
        height=height,
        scene=dict(aspectmode="data")
    )
    
    fig.show()
