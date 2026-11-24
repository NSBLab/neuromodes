import numpy as np
import pytest

from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_example_surf
from neuromodes.nulls import eigenstrap
from neuromodes.stats import meanw, pdistw

# Params and setup
n_nulls = 100
n_maps = 3 # for 2d data

@pytest.fixture(scope='module')
def solver():
    mesh, medmask = fetch_example_surf(res='4k')
    return EigenSolver(mesh, mask=medmask).solve(n_modes=100)

@pytest.fixture(scope='module')
def test_data(solver):
    """Generate 1D test data"""
    rng = np.random.default_rng()
    return rng.normal(loc=1, size=solver.n_verts)  # random normal data, non-zero mean

@pytest.fixture(scope='module')
def nulls(solver, test_data):
    """Generate nulls for 1D test data"""
    return solver.eigenstrap(test_data, n_nulls=n_nulls)

@pytest.mark.parametrize("size", [(), (1,), (2,), (1,1), (1,2), (2,1), (2,2)])
@pytest.mark.parametrize(
    "kwargs",
    [ # Check all these options (one at a time), but don't need to repeat defaults
        # {"rotation_method": "qr"},
        {"rotation_method": "scipy"},
        # {"randomize": False},
        {"randomize": True},
        # {"residual": None},
        {"residual": "add"},
        {"residual": "permute"},
        # {"resample": None},
        {"resample": "exact"},
        {"resample": "range"},
        # {"decomp_method": "project"},
        {"decomp_method": "regress"},
    ]
)
def test_output_shape_and_isfinite(solver, size, kwargs):
    """Test individual parameter options run without errors."""
    test_data = np.random.rand(solver.n_verts, *size)
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, **kwargs)
    assert nulls.shape == (*test_data.shape, n_nulls)
    assert np.isfinite(nulls).all()

@pytest.mark.parametrize("size", [(), (1,), (2,), (1,1), (1,2), (2,1), (2,2)])
@pytest.mark.parametrize("rotation_method", ["qr", "scipy"])
@pytest.mark.parametrize("decomp_method", ["project", "regress"])
@pytest.mark.parametrize("randomize", [True, False])
@pytest.mark.parametrize("residual", [None]) # mean only preserved for residual=None
@pytest.mark.parametrize("resample", [None]) # mean only preserved for resample=None
def test_null_means(solver, size, rotation_method, decomp_method, randomize, residual, resample):
    """Nulls should preserve area-weighted mean of original data"""
    test_data = np.random.rand(solver.n_verts, *size)
    data_mean = meanw(test_data, solver.mass, keepdims=True)
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, 
                                   rotation_method=rotation_method,
                                   decomp_method=decomp_method,
                                   randomize=randomize,
                                   residual=residual,
                                   resample=resample)
    null_means = meanw(nulls, solver.mass, keepdims=True)
    assert np.allclose(null_means, data_mean[..., np.newaxis], atol=1e-10), \
        f"Null means are not close to data mean {data_mean}"

# Test validation of input parameters
def test_invalid_residual_parameter(solver, test_data):
    """Should raise ValueError for invalid residual parameter"""
    with pytest.raises(ValueError, match="Invalid residual method"):
        solver.eigenstrap(test_data, residual='invalid')

def test_shape_mismatch_data_emodes(solver, test_data):
    """`decompose()` should raise error when data length doesn't match emodes rows"""
    wrong_data = test_data[:-100]  # Truncate data
    
    with pytest.raises((ValueError, IndexError)):
        solver.eigenstrap(wrong_data)

def test_shape_mismatch_emodes_evals(solver, test_data):
    """Should raise error when emodes columns doesn't match evals length"""
    wrong_evals = solver.evals[:-10]  # Truncate evals
    
    with pytest.raises(ValueError, match="must have shape"):
        eigenstrap(test_data, solver.emodes, wrong_evals, mass=solver.mass)

def test_non_square_modes(test_data):
    """Should handle non-square n_modes by truncating last eigengroup with warning"""
    # Use 8 modes (not a perfect square)
    mesh, medmask = fetch_example_surf(res='4k')
    non_square_solver = EigenSolver(mesh, mask=medmask).solve(n_modes=8)
    
    # Should complete with a warning about truncating last eigengroup
    with pytest.warns(UserWarning, match="Last 4 modes will be excluded."):
        nulls = non_square_solver.eigenstrap(test_data, n_nulls=n_nulls, residual='add')
    
    assert nulls.shape == (non_square_solver.n_verts, n_nulls)
    
# Test more properties of output nulls (more akin to correctness)
def test_internull_corrs(test_data, solver):
    """Internull correlations should be centered around zero for residual=None"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, residual=None)
    null_corrs = 1 - pdistw(nulls, mass=solver.mass, metric='correlation')
    np.testing.assert_allclose(null_corrs.mean(), 0, atol=0.01)

def test_residual_none(solver, test_data):
    """Nulls should not have any residual added when residual=None"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, residual=None)
    recons = solver.filter_spectrum(nulls)
    residuals = nulls - recons
    
    assert np.allclose(residuals, 0, atol=1e-10), \
        "Nulls should not have any residual added when residual=None"

def test_residual_add(solver, test_data):
    """Nulls should have residual added when residual='add'"""
    data_recons = solver.filter_spectrum(test_data)
    data_residuals = test_data - data_recons
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, residual='add')
    null_recons = solver.filter_spectrum(nulls)
    null_residuals = nulls - null_recons
    assert np.allclose(null_residuals, data_residuals[:,np.newaxis]), \
        "Nulls should have residual added when residual='add'"
    
def test_residual_permute(solver, test_data):
    """Nulls should have permuted residual added when residual='permute'"""
    data_recons = solver.filter_spectrum(test_data)
    data_residuals = test_data - data_recons
    # Have to do it this way as permute residuals will not be orthogonal to modes (ie an approach
    # like the `residual='add'`` approach will not work)
    nulls_base = solver.eigenstrap(test_data, n_nulls=n_nulls, residual=None, seed=365)
    nulls_perm = solver.eigenstrap(test_data, n_nulls=n_nulls, residual='permute', seed=365)
    nulls_residuals = nulls_perm - nulls_base
    assert not np.allclose(nulls_residuals, data_residuals[:, np.newaxis]), \
        "Nulls should not have original residuals added exactly when residual='permute'"
    assert np.allclose(np.sort(nulls_residuals, axis=0), np.sort(data_residuals)[:, np.newaxis]), \
        "Nulls should have permuted residuals added when residual='permute'"

def test_resample_exact(solver, test_data):
    """With resample=True, nulls should have same values as original data"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, resample="exact")
    
    # Check that each null has the exact same values as original data (just reordered)
    for i in range(nulls.shape[1]):
        null_sorted = np.sort(nulls[:, i])
        data_sorted = np.sort(test_data)
        assert np.allclose(null_sorted, data_sorted), \
            f"Null {i} doesn't preserve data distribution"

def test_resample_range(solver, test_data):
    """With resample='range', nulls should have same min and max as original data"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, resample="range")
    
    data_min = np.min(test_data)
    data_max = np.max(test_data)
    
    for i in range(nulls.shape[1]):
        null_min = np.min(nulls[:, i])
        null_max = np.max(nulls[:, i])
        assert np.isclose(null_min, data_min), f"Null {i} min is not close to data min"
        assert np.isclose(null_max, data_max), f"Null {i} max is not close to data max"

@pytest.fixture
def test_data_2d(solver):
    """Generate 2D test data with n_maps maps"""
    rng = np.random.default_rng()
    data_2d = rng.standard_normal(size=(solver.n_verts, n_maps))
    return data_2d

def test_same_map(solver, test_data_2d): 
    """For the same map, nulls should be the same (and different for different maps)"""
    test_data = np.column_stack((
        test_data_2d[:,0],
        test_data_2d[:,0],
        np.random.standard_normal(size=solver.n_verts)
        ))
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls)

    assert np.allclose(nulls[:,1,:], nulls[:,0,:]), \
        "Nulls for map 1 should be the same as nulls for map 0"
    assert not np.allclose(nulls[:,1,:], nulls[:,2,:]), \
        "Nulls for map 1 should be different to nulls for map 2"

def test_different_maps(solver, test_data_2d): 
    """For different maps, each null should be different to each other null"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls)

    for i in range(n_maps): 
        for j in range(i): 
            assert not np.allclose(nulls[:,i,:], nulls[:,j,:]), \
                f"Nulls for map {i} should be different to nulls for map {j}"
    for i in range(n_nulls): 
        for j in range(i): 
            assert not np.allclose(nulls[:,:,i], nulls[:,:,j]), \
                f"Nulls {i} should be different to nulls {j}"

def test_resample_exact_2d(solver, test_data_2d):
    """With resample=True, nulls should have same values as original data"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="exact")
    
    # Check that each null has the exact same values as original data (just reordered)
    # TODO: vectorise if slow?
    for j in range(n_maps):
        for i in range(n_nulls):
            null_sorted = np.sort(nulls[:, j, i])
            data_sorted = np.sort(test_data_2d[:, j])
            assert np.allclose(null_sorted, data_sorted), \
                f"Null {i} doesn't preserve data distribution for map {j}"

def test_resample_range_2d(solver, test_data_2d):
    """With resample='range', nulls should have same min and max as original data"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="range")
    
    for j in range(n_maps):
        data_min = np.min(test_data_2d[:, j])
        data_max = np.max(test_data_2d[:, j])
        for i in range(n_nulls):
            null_min = np.min(nulls[:, j, i])
            null_max = np.max(nulls[:, j, i])
            assert np.isclose(null_min, data_min), f"Null {i} map {j} min is not close to data min"
            assert np.isclose(null_max, data_max), f"Null {i} map {j} max is not close to data max"
