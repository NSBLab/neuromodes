import numpy as np
import pytest
from neuromodes.io import fetch_example_surf
from neuromodes import EigenSolver
from neuromodes.network import compute_gem

@pytest.fixture(scope='module')
def solver():
    n_modes = 5
    surf, medmask = fetch_example_surf(density='4k')

    return EigenSolver(surf, mask=~medmask).solve(n_modes)

def test_compute_gem_properties(solver):
    emodes = solver.emodes
    mass = solver.mass
    evals = solver.evals
    conn = compute_gem(emodes, evals, r=1.5, k=emodes.shape[1], mass=mass)

    # shape
    assert conn.shape == (solver.n_verts, solver.n_verts)

    # symmetric
    assert (conn == conn.T).all()

    # diagonal zeros
    assert (np.diag(conn) == 0).all()

    # non-negative
    assert np.all(conn >= 0)

    # normalized maximum is 1
    assert np.isclose(np.max(conn), 1.0, atol=1e-8)

def test_k_changes_result(solver):
    emodes = solver.emodes
    mass = solver.mass
    evals = solver.evals

    conn_k2 = compute_gem(emodes, evals, r=2.0, k=2, mass=mass)
    conn_k5 = compute_gem(emodes, evals, r=2.0, k=5, mass=mass)

    # Both valid outputs
    assert conn_k2.shape == conn_k5.shape == (solver.n_verts, solver.n_verts)
    assert np.isclose(np.max(conn_k2), 1.0)
    assert np.isclose(np.max(conn_k5), 1.0)

    # Different k should generally produce different connectomes
    assert not np.allclose(conn_k2, conn_k5)

def test_r_changes_result(solver):
    emodes = solver.emodes
    mass = solver.mass
    evals = solver.evals

    conn_r1 = compute_gem(emodes, evals, r=1.0, k=3, mass=mass)
    conn_r2 = compute_gem(emodes, evals, r=2.0, k=3, mass=mass)

    # Both valid outputs
    assert conn_r1.shape == conn_r2.shape == (solver.n_verts, solver.n_verts)
    assert np.isclose(np.max(conn_r1), 1.0)
    assert np.isclose(np.max(conn_r2), 1.0)

    # Different r should generally produce different connectomes
    assert not np.allclose(conn_r1, conn_r2)

@pytest.mark.parametrize("bad_r", [0, -1.0])
def test_invalid_r_raises(solver, bad_r):
    emodes = solver.emodes
    evals = solver.evals
    with pytest.raises(ValueError, match="Parameter r must be a positive number."):
        compute_gem(emodes, evals, r=bad_r, k=3)

@pytest.mark.parametrize("bad_k", [0, 100, 2.5])
def test_invalid_k_raises(bad_k, solver):
    emodes = solver.emodes
    evals = solver.evals
    with pytest.raises(ValueError, match="Parameter k must be an integer in the range"):
        compute_gem(emodes, evals, r=1.0, k=bad_k)

def test_numpy_types(solver):
    emodes = solver.emodes
    mass = solver.mass
    evals = solver.evals

    # Test that different types for r and k work without error
    _ = compute_gem(emodes, evals, r=np.float64(1.5), k=np.int32(3), mass=mass) # type: ignore

def test_pinv_changes_results(solver):
    emodes = solver.emodes
    mass = solver.mass
    r = 1.0
    k = 3

    evals = solver.evals

    conn = compute_gem(emodes, evals, r=r, k=k, weighted_pinv=False, mass=mass)
    connw = compute_gem(emodes, evals, r=r, k=k, weighted_pinv=True, mass=mass)
    assert not np.allclose(conn, connw)

def test_nodestrength(solver):
    emodes = solver.emodes
    evals = solver.evals
    mass = solver.mass

    conn = compute_gem(emodes, evals, r=10.0, k=3, mass=mass)

    areas = mass.sum(axis=0).A1
    node_strength = np.dot(areas, conn)

    # not exactly 1 due to zeroing of diagonal and negatives
    assert np.corrcoef(node_strength, areas)[0, 1] > 0.8

    # constant eigenmode only makes relationship exact
    conn_k1 = compute_gem(emodes, evals, r=10.0, k=1, mass=mass)
    node_strength_k1 = np.dot(areas, conn_k1)
    assert np.corrcoef(node_strength_k1, areas)[0, 1] > 0.999