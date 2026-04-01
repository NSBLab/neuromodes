import numpy as np
import pytest
from neuromodes.connectome import model_connectome

@pytest.fixture(scope='module')
def emodes():
    n_modes = 5
    n_verts = 10

    return np.random.default_rng(0).standard_normal(size=(n_verts, n_modes))

def test_model_connectome_properties(emodes):
    evals = np.linspace(0.0, 10.0, emodes.shape[1])
    conn = model_connectome(emodes, evals, r=1.5, k=emodes.shape[1])

    # shape
    assert conn.shape == (10, 10)

    # symmetric
    assert (conn == conn.T).all()

    # diagonal zeros
    assert (np.diag(conn) == 0).all()

    # non-negative
    assert np.all(conn >= 0)

    # normalized maximum is 1
    assert np.isclose(np.max(conn), 1.0, atol=1e-8)

def test_k_changes_result(emodes):
    evals = np.arange(5).astype(float)

    conn_k2 = model_connectome(emodes, evals, r=2.0, k=2)
    conn_k5 = model_connectome(emodes, evals, r=2.0, k=5)

    # Both valid outputs
    assert conn_k2.shape == conn_k5.shape == (10, 10)
    assert np.isclose(np.max(conn_k2), 1.0)
    assert np.isclose(np.max(conn_k5), 1.0)

    # Different k should generally produce different connectomes
    assert not np.allclose(conn_k2, conn_k5)

def test_r_changes_result(emodes):
    evals = np.arange(5).astype(float)

    conn_r1 = model_connectome(emodes, evals, r=1.0, k=3)
    conn_r2 = model_connectome(emodes, evals, r=2.0, k=3)

    # Both valid outputs
    assert conn_r1.shape == conn_r2.shape == (10, 10)
    assert np.isclose(np.max(conn_r1), 1.0)
    assert np.isclose(np.max(conn_r2), 1.0)

    # Different r should generally produce different connectomes
    assert not np.allclose(conn_r1, conn_r2)

@pytest.mark.parametrize("bad_r", [0, -1.0, "winterjams"])
def test_invalid_r_raises(emodes, bad_r):
    evals = np.arange(5).astype(float)
    with pytest.raises(ValueError):
        model_connectome(emodes, evals, r=bad_r, k=3)

def test_emodes_must_be_2d():
    emodes = np.array([1.0, 2.0, 3.0])  # 1D
    evals = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        model_connectome(emodes, evals, r=1.0, k=2)

def test_eval_length_mismatch_raises():
    # construct an emodes array with 3 columns so the eval length
    # mismatch triggers the ValueError
    rng = np.random.RandomState(1)
    emodes = rng.randn(4, 3)
    evals = np.array([1.0, 2.0])  # length doesn't match 3 columns
    with pytest.raises(ValueError):
        model_connectome(emodes, evals, r=1.0, k=2)

@pytest.mark.parametrize("bad_k", [0, 100, 2.5])
def test_invalid_k_raises(bad_k, emodes):
    evals = np.arange(5).astype(float)
    with pytest.raises(ValueError):
        model_connectome(emodes, evals, r=1.0, k=bad_k)

def test_numpy_types(emodes):
    evals = np.arange(5).astype(float)

    # Test that different types for r and k work without error
    _ = model_connectome(emodes, evals, r=np.float64(1.5), k=np.int32(3)) # type: ignore