import pytest
import numpy as np
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_example_surf
from neuromodes.nulls import eigenstrap
from neuromodes.stats import meanw, stdw

# Params and setup
n_nulls = 100
n_maps = 3 # for 2d data

@pytest.fixture(scope='module')
def solver():
    mesh, medmask = fetch_example_surf(density='4k')
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

def test_output_shape(solver, nulls):
    """Test that code runs and output shape is correct - should be (n_verts, n_nulls)"""    
    assert nulls.shape == (solver.n_verts, n_nulls), \
        f"Expected shape {(solver.n_verts, n_nulls)}, got {nulls.shape}"

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
    mesh, medmask = fetch_example_surf(density='4k')
    non_square_solver = EigenSolver(mesh, mask=medmask).solve(n_modes=8)
    
    # Should complete with a warning about truncating last eigengroup
    with pytest.warns(UserWarning, match="Last 4 modes will be excluded."):
        nulls = non_square_solver.eigenstrap(test_data, n_nulls=n_nulls, residual='add')
    
    assert nulls.shape == (non_square_solver.n_verts, n_nulls)

@pytest.mark.parametrize("rotation_method", ['scipy', 'qr'])
def test_rotation_options(solver, test_data, rotation_method):
    """Test rotation_method parameter works without errors"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, rotation_method=rotation_method)
    assert nulls.shape == (solver.n_verts, n_nulls)
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for rotation_method={rotation_method}"

@pytest.mark.parametrize("randomize", [True, False])
def test_randomize_options(solver, test_data, randomize):
    """Test randomize parameter works without errors"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, randomize=randomize)
    assert nulls.shape == (solver.n_verts, n_nulls)
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for randomize={randomize}"

@pytest.mark.parametrize("residual", ['add', 'permute', None])
def test_residual_options(solver, test_data, residual):
    """Test different residual methods run without errors"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, residual=residual)
    assert nulls.shape == (solver.n_verts, n_nulls)
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for residual method '{residual}'"

@pytest.mark.parametrize("resample", ['exact', 'affine', 'mean', 'range', None])
def test_resample_options(solver, test_data, resample):
    """Test different resample methods run without errors"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, resample=resample)
    assert nulls.shape == (solver.n_verts, n_nulls)
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for resample method '{resample}'"
    
@pytest.mark.parametrize("decomp_method", ['regress', 'project'])
def test_decomp_options(solver, test_data, decomp_method):
    """Test different decomp methods run without errors"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, decomp_method=decomp_method)
    assert nulls.shape == (solver.n_verts, n_nulls)
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for decomp method '{decomp_method}'"
    
# Test more properties of output nulls (more akin to correctness)
def test_finite(nulls):
    """Output should not contain NaNs or Infs"""    
    assert np.isfinite(nulls).all(), "Nulls contain NaNs or Infs"

def test_internull_corrs(nulls):
    """Internull correlations should be centered around zero"""
    inter_null_corrs = np.corrcoef(nulls.T)
    triu_inds = np.triu_indices_from(inter_null_corrs, k=1)
    mean_corr = inter_null_corrs[triu_inds].mean()
    assert np.abs(mean_corr) < 0.01, \
        f"Mean internull correlation should be close to zero, got {mean_corr:.3f}"
    
def test_residual_none(solver, test_data):
    """Nulls should not have any residual added when residual=None"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, residual=None)
    recons = np.squeeze(solver.reconstruct(nulls, mode_counts=[solver.n_modes]))
    residuals = nulls - recons
    
    assert np.allclose(residuals, 0, atol=1e-10), \
        "Nulls should not have any residual added when residual=None"

def test_residual_add(solver, test_data):
    """Nulls should have residual added when residual='add'"""
    data_recons = np.squeeze(solver.reconstruct(test_data, mode_counts=[solver.n_modes]))
    data_residuals = test_data - data_recons
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, residual='add')
    null_recons = np.squeeze(solver.reconstruct(nulls, mode_counts=[solver.n_modes]))
    null_residuals = nulls - null_recons
    assert np.allclose(null_residuals, data_residuals[:,np.newaxis]), \
        "Nulls should have residual added when residual='add'"
    
def test_residual_permute(solver, test_data):
    """Nulls should have permuted residual added when residual='permute'"""
    data_recons = solver.reconstruct(test_data, mode_counts=solver.n_modes)
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
    
def test_resample_none(test_data, nulls):
    """Nulls should approximately preserve mean of original data even without using resample='mean'"""
    data_mean = np.mean(test_data)
    null_means = np.mean(nulls, axis=0)
    assert np.allclose(null_means, data_mean, atol=0.05), \
        f"Null means are not close to data mean {data_mean}"

def test_resample_exact(solver, test_data):
    """With resample=True, nulls should have same values as original data"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, resample="exact")
    
    # Check that each null has the exact same values as original data (just reordered)
    for i in range(nulls.shape[1]):
        null_sorted = np.sort(nulls[:, i])
        data_sorted = np.sort(test_data)
        assert np.allclose(null_sorted, data_sorted), \
            f"Null {i} doesn't preserve data distribution"

# ignore warning from decompose()
@pytest.mark.filterwarnings("ignore:data contains NaNs")
def test_resample_exact_nan_mask_preserved_1d(solver, test_data):
    """With resample='exact', NaN locations should be preserved and finite values rank-matched."""
    test_data_nan = test_data.copy()
    test_data_nan[::19] = np.nan
    nulls = solver.eigenstrap(test_data_nan, n_nulls=n_nulls, resample="exact", decomp_method='regress')

    nan_mask = np.isnan(test_data_nan)
    valid_mask = ~nan_mask
    expected_sorted = np.sort(test_data_nan[valid_mask])

    assert np.isnan(nulls[nan_mask, :]).all(), "NaN locations should be preserved across nulls"
    assert np.isfinite(nulls[valid_mask, :]).all(), "Valid locations should remain finite"

    for i in range(nulls.shape[1]):
        null_sorted = np.sort(nulls[valid_mask, i])
        assert np.allclose(null_sorted, expected_sorted), \
            f"Null {i} doesn't preserve finite-value distribution with NaNs present"

@pytest.mark.filterwarnings("ignore:data contains NaNs")
def test_resample_exact_nan_mask_preserved_2d(solver, test_data_2d):
    """With resample='exact', each map should preserve its own NaN mask and finite distribution."""
    test_data = test_data_2d.copy()

    # add nans
    test_data[::17, 0] = np.nan
    test_data[5::19, 1] = np.nan
    test_data[5::19, 2] = np.nan  # use same pattern to test efficient sorting

    # add infs
    test_data[::29, 0] = np.inf
    test_data[7::31, 1] = -np.inf

    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, resample="exact", decomp_method='regress')

    for m in range(test_data.shape[1]):
        nan_mask = np.isnan(test_data[:, m])
        inf_mask = np.isinf(test_data[:, m])
        valid_mask = ~(nan_mask | inf_mask)
        expected_sorted = np.sort(test_data[valid_mask, m])

        assert np.isnan(nulls[nan_mask, :, m]).all(), f"NaN locations should be preserved in map {m}"
        assert np.isinf(nulls[inf_mask, :, m]).all(), f"Inf locations should be preserved in map {m}"
        assert np.isfinite(nulls[valid_mask, :, m]).all(), f"Valid locations should remain finite in map {m}"

        for i in range(nulls.shape[1]):
            null_sorted = np.sort(nulls[valid_mask, i, m])
            assert np.allclose(null_sorted, expected_sorted), \
                f"Null {i} in map {m} doesn't preserve finite-value distribution with NaNs/Infs present"

def test_resample_affine(solver, test_data):
    """With resample='affine', nulls should have mean and std that match the data"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, resample="affine")
    
    for i in range(nulls.shape[1]):
        mean = meanw(nulls[:, i], solver.mass)
        std = stdw(nulls[:, i], solver.mass)
        assert np.isclose(mean, meanw(test_data, solver.mass)), f"Null {i} mean is not close to data mean"
        assert np.isclose(std, stdw(test_data, solver.mass)), f"Null {i} std is not close to data std"

def test_resample_mean(solver, test_data):
    """With resample='mean', nulls should have mean equal to original data mean"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, resample="mean")
    
    data_mean = meanw(test_data, solver.mass)

    for i in range(nulls.shape[1]):
        null_mean = meanw(nulls[:, i], solver.mass)
        assert np.isclose(null_mean, data_mean), f"Null {i} mean is not close to data mean"

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

def test_output_shape_2d(solver, test_data_2d):
    """Output shape should be (n_verts, n_nulls, n_maps)"""
    sizes = [(solver.n_verts, 1), (solver.n_verts, 2), (solver.n_verts, n_maps)]
    for size in sizes:
        test_data_2d = np.random.normal(loc=1, size=size)
        nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls)
        expected_shape = (solver.n_verts, n_nulls, size[1])
        assert nulls.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {nulls.shape} for input shape {size}"

@pytest.mark.parametrize("rotation_method", ['scipy', 'qr'])
def test_rotation_options_2d(solver, test_data_2d, rotation_method):
    """Test rotation_method parameter works without errors"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, rotation_method=rotation_method)
    assert nulls.shape == (solver.n_verts, n_nulls, test_data_2d.shape[1])
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for `rotation_method={rotation_method}`"

@pytest.mark.parametrize("randomize", [True, False])
def test_randomize_options_2d(solver, test_data_2d, randomize):
    """Test randomize parameter works without errors"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, randomize=randomize)
    assert nulls.shape == (solver.n_verts, n_nulls, test_data_2d.shape[1])
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for `randomize={randomize}`"

@pytest.mark.parametrize("residual", ['add', 'permute'])
def test_residual_options_2d(solver, test_data_2d, residual):
    """Test different residual methods run without errors"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, residual=residual)
    assert nulls.shape == (solver.n_verts, n_nulls, test_data_2d.shape[1])
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for `residual='{residual}'`"
    
@pytest.mark.parametrize("resample", ['exact', 'affine', 'mean', 'range', None])    
def test_resample_options_2d(solver, test_data_2d, resample):
    """Test different resample methods run without errors"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample=resample)
    assert nulls.shape == (solver.n_verts, n_nulls, test_data_2d.shape[1])
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for `resample='{resample}'`"
    
@pytest.mark.parametrize("decomp_method", ['regress', 'project'])
def test_decomp_options_2d(solver, test_data_2d, decomp_method):
    """Test different decomp methods run without errors"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, decomp_method=decomp_method)
    assert nulls.shape == (solver.n_verts, n_nulls, test_data_2d.shape[1])
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for `decomp_method='{decomp_method}'`"

def test_same_map(solver, test_data_2d): 
    """For the same map, nulls should be the same (and different for different maps)"""
    test_data = np.column_stack((test_data_2d[:,0], test_data_2d[:,0], np.random.standard_normal(size=solver.n_verts)))
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls)

    assert not np.allclose(nulls[:,:,1], nulls[:,:,2]), \
        "Nulls for map 1 should be different to nulls for map 2"
    assert np.allclose(nulls[:,:,1], nulls[:,:,0]), \
        "Nulls for map 1 should be the same as nulls for map 0"

def test_different_maps(solver, test_data_2d): 
    """For different maps, each null should be different to each other null"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls)

    for i in range(n_maps): 
        for j in range(i): 
            assert not np.allclose(nulls[:,:,i], nulls[:,:,j]), \
                f"Nulls for map {i} should be different to nulls for map {j}"
    for i in range(n_nulls): 
        for j in range(i): 
            assert not np.allclose(nulls[:,i,:], nulls[:,j,:]), \
                f"Nulls {i} should be different to nulls {j}"

def test_resample_none_2d(solver, test_data_2d):
    """Nulls should approximately preserve mean of original data even without using resample='mean'"""
    data_means = meanw(test_data_2d, solver.mass)
    null_means = meanw(solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample=None), solver.mass)

    assert np.allclose(null_means, data_means, atol=0.05), \
        f"Null means are not close to data means {data_means}"

def test_resample_exact_2d(solver, test_data_2d):
    """With resample=True, nulls should have same values as original data"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="exact")
    
    # Check that each null has the exact same values as original data (just reordered)
    # TODO: vectorise if slow?
    for j in range(test_data_2d.shape[1]):
        for i in range(nulls.shape[1]):
            null_sorted = np.sort(nulls[:, i, j])
            data_sorted = np.sort(test_data_2d[:, j])
            assert np.allclose(null_sorted, data_sorted), \
                f"Null {i} doesn't preserve data distribution for map {j}"

def test_resample_affine_2d(solver, test_data_2d):
    """With resample='affine', nulls should have mean and std that match the data"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="affine")
    
    for j in range(test_data_2d.shape[1]):
        for i in range(nulls.shape[1]):
            mean = meanw(nulls[:, i, j], solver.mass)
            std = stdw(nulls[:, i, j], solver.mass)
            data_mean = meanw(test_data_2d[:, j], solver.mass)
            data_std = stdw(test_data_2d[:, j], solver.mass)
            assert np.isclose(mean, data_mean), f"Null {i} map {j} mean is not close to data mean"
            assert np.isclose(std, data_std), f"Null {i} map {j} std is not close to data std"

def test_resample_mean_2d(solver, test_data_2d):
    """With resample='mean', nulls should have mean equal to original data mean"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="mean")
    
    for j in range(test_data_2d.shape[1]):
        data_mean = meanw(test_data_2d[:, j], solver.mass)
        for i in range(nulls.shape[1]):
            null_mean = meanw(nulls[:, i, j], solver.mass)
            assert np.isclose(null_mean, data_mean), \
                f"Null {i} map {j} mean is not close to data mean"

def test_resample_range_2d(solver, test_data_2d):
    """With resample='range', nulls should have same min and max as original data"""
    nulls = solver.eigenstrap(test_data_2d, n_nulls=n_nulls, resample="range")
    
    for j in range(test_data_2d.shape[1]):
        data_min = np.min(test_data_2d[:, j])
        data_max = np.max(test_data_2d[:, j])
        for i in range(nulls.shape[1]):
            null_min = np.min(nulls[:, i, j])
            null_max = np.max(nulls[:, i, j])
            assert np.isclose(null_min, data_min), f"Null {i} map {j} min is not close to data min"
            assert np.isclose(null_max, data_max), f"Null {i} map {j} max is not close to data max"
