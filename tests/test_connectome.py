import numpy as np
import pytest

from neuromodes.network import compute_gem
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_example_surf

@pytest.fixture(scope='module')
def solver():
    surf, medmask = fetch_example_surf(res='4k')
    solver = EigenSolver(surf, mask=medmask)
    return solver.solve(n_modes=10)

def test_compute_gem_basic(solver):
    # evals = np.linspace(0.0, 10.0, solver.emodes.shape[1])
    conn = compute_gem(solver.emodes, 
                       solver.evals, 
                       mass=solver.mass,
                       r=1.5, 
                       mode_counts=solver.emodes.shape[1])

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

    conn_k2 = solver.compute_gem(mode_counts=2)
    conn_k5 = solver.compute_gem(mode_counts=5)

    # Both valid outputs
    assert np.isclose(np.max(conn_k2), 1.0)
    assert np.isclose(np.max(conn_k5), 1.0)

    # Different k should generally produce different connectomes
    assert not np.allclose(conn_k2, conn_k5)

def test_r_changes_result(solver):
    
    conn_r1 = solver.compute_gem(r=1.0)
    conn_r2 = solver.compute_gem(r=2.0)

    # Both valid outputs
    assert np.isclose(np.max(conn_r1), 1.0)
    assert np.isclose(np.max(conn_r2), 1.0)

    # Different r should generally produce different connectomes
    assert not np.allclose(conn_r1, conn_r2)

@pytest.mark.parametrize("bad_r", [-0.001, -1.0])
def test_invalid_r_raises(solver, bad_r):
    with pytest.raises(ValueError):
        solver.compute_gem(r=bad_r)

def test_massless_warning(): 
    emodes = np.random.rand(10, 5)
    evals = np.random.rand(5)
    with pytest.warns(UserWarning, match="Mass matrix not provided; assuming that area at each vertex is 1"):
        compute_gem(emodes, evals, r=1.0, mode_counts=2)

def test_emodes_must_be_2d():
    emodes = np.array([1.0, 2.0, 3.0])  # 1D
    evals = np.array([1.0, 2.0, 3.0])
    with pytest.raises(IndexError):
        compute_gem(emodes, evals, r=1.0, mode_counts=2)

def test_eval_length_mismatch_raises():
    # construct an emodes array with 3 columns so the eval length
    # mismatch triggers the ValueError
    emodes = np.random.rand(4, 3)
    evals = np.array([1.0, 2.0])  # length doesn't match 3 columns
    with pytest.raises(ValueError):
        compute_gem(emodes, evals, r=1.0, mode_counts=2)

@pytest.mark.parametrize("bad_k", [0, 100, 2.5])
def test_invalid_k_raises(bad_k, solver):
    with pytest.raises((ValueError, TypeError)):
        solver.compute_gem(mode_counts=bad_k)