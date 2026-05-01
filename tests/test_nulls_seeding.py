from pathlib import Path
import pytest
import numpy as np
from scipy.sparse import csc_matrix, eye
from packaging import version
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf, fetch_map
from neuromodes.nulls import eigenstrap

# Params
density = '4k'
n_modes = 100 # should be a square number
n_maps = 3
n_nulls = 20

# Set these parameters as options for the tests below. This determines how thorough the tests are.
# The options can be reduced to speed up testing (only a subset of the correctness will be tested). 
rotation_options = ['qr', 'scipy']
randomize_options = [True, False]
residual_options = [None, 'permute'] # skip add for convenience as it has no randomisation
# Skip resample and decomp method as they are not related to seeding

@pytest.fixture(scope='module')
def solver(seed=None):
    mesh, medmask = fetch_surf(density=density)
    return EigenSolver(mesh, mask=medmask).solve(n_modes=n_modes, seed=seed)

@pytest.fixture(scope='module')
def test_data(solver):
    """Generate test data""" # random normal data, non-zero mean
    return np.random.default_rng(None).normal(loc=1, size=(solver.n_verts, n_maps))  

@pytest.fixture(scope='module')
def nulls_orig():
    """Load original nulls, handling scipy version compatibility."""
    # Hardcoded params matching both tests
    density = '4k'
    hemi = 'L'
    surf_type = 'midthickness'
    
    test_data = Path(__file__).parent / 'test_data'
    
    # Try scipy-versioned file first (for _seed_outside test)
    import scipy
    if version.parse(scipy.__version__) < version.parse('1.16.0'):
        scipy_version = '1.15.3'
    else:
        scipy_version = '1.16.0'
    
    nulls_file = test_data / f"sp-human_tpl-fsLR_den-{density}_hemi-{hemi}_{surf_type}_eigenstrap-nulls-orig_scipy={scipy_version}.npy"
        
    return np.load(nulls_file)

@pytest.mark.parametrize("rotation_method", rotation_options)
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options) 
@pytest.mark.parametrize("seed", [None, 0, 42, np.random.choice(n_nulls, n_nulls, replace=False)])
def test_seed_options_1d(solver, test_data, seed, rotation_method, randomize, residual):
    """Test different seed options run without errors (1D data)"""
    test_data_1d = test_data[:, 0]
    nulls = solver.eigenstrap(test_data_1d, n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, randomize=randomize, residual=residual)
    assert nulls.shape == (solver.n_verts, n_nulls)
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for seed={seed}"

@pytest.mark.parametrize("rotation_method", rotation_options)
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options) 
@pytest.mark.parametrize("seed", [None, 0, 42, np.random.choice(n_nulls, n_nulls, replace=False)])
def test_seed_options(solver, test_data, seed, rotation_method, randomize, residual):
    """Test different seed options run without errors (2D data)"""
    nulls = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, randomize=randomize, residual=residual)
    assert nulls.shape == (solver.n_verts, n_nulls, test_data.shape[1])
    assert np.isfinite(nulls).all(), \
        f"Nulls contain non-finite values for seed={seed}"
    
@pytest.mark.parametrize("rotation_method", rotation_options)
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options) 
@pytest.mark.parametrize("seed", [0, 42, np.random.choice(n_nulls, n_nulls, replace=False)])
def test_same_and_different_seeds(solver, test_data, rotation_method, randomize, residual, seed): 
    """Test the results based on seeding"""
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, randomize=randomize, residual=residual)
    a2 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, randomize=randomize, residual=residual)
    b1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed+1, rotation_method=rotation_method, randomize=randomize, residual=residual)
    
    assert np.allclose(a1, a2), \
        "Nulls generated with the same seed should be the same"
    assert not np.allclose(a1, b1), \
        "Nulls generated with different seeds should be different"
    
@pytest.mark.parametrize("rotation_method", rotation_options)
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options) 
def test_seed_none(solver, test_data, rotation_method, randomize, residual): 
    """Test the results based on seeding"""
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method=rotation_method, randomize=randomize, residual=residual)
    b1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method=rotation_method, randomize=randomize, residual=residual)

    assert not np.allclose(a1, b1), \
        "Nulls generated with None seeds should be different by default"

@pytest.mark.parametrize("rotation_method", rotation_options)
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options) 
def test_randomize_resample(solver, rotation_method, randomize, residual):
    """Test that the rotation matrices are not affected by randomizing/resampling"""
    # Generate beta coeffs at the group level so that they are not affected by randomization
    group_indices = np.floor(np.sqrt(np.arange(solver.n_modes))).astype(int)
    beta0_group = np.random.default_rng(None).normal(loc=1, size=(max(group_indices)+1, n_maps))
    beta0_mode = beta0_group[group_indices, :]  # coeffs will be the same within each group, so not affected by randomization
    data = solver.emodes @ beta0_mode           # data generated from the bases, so there will be no residuals

    a1 = solver.eigenstrap(data, n_nulls=n_nulls, seed=1, rotation_method=rotation_method, randomize=False, residual=None) 
    a2 = solver.eigenstrap(data, n_nulls=n_nulls, seed=1, rotation_method=rotation_method, randomize=randomize, residual=residual)

    assert np.allclose(a1, a2), \
        "These specific nulls should be the same regardless of randomize and residual parameters when seed is fixed"

@pytest.mark.parametrize("rotation_method", rotation_options)
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options) 
def test_seeds_reordered(solver, test_data, rotation_method, randomize, residual):
    """Test that the same seed produces the same nulls even if input data is in different order (eg different number of maps)"""
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=np.arange(n_nulls), rotation_method=rotation_method, randomize=randomize, residual=residual)
    a2 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=np.flip(np.arange(n_nulls)), rotation_method=rotation_method, randomize=randomize, residual=residual)

    assert np.allclose(np.flip(a1,axis=1), a2, atol=1e-10), \
        "Nulls generated with the same seed should be identical regardless of seed position"

@pytest.mark.parametrize("rotation_method", rotation_options)
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options) 
def test_seed_reordered_data(solver, test_data, rotation_method, randomize, residual):
    """Test that the same seed produces the same nulls even if input data is in different order"""
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=1, rotation_method=rotation_method, randomize=randomize, residual=residual)
    a2 = solver.eigenstrap(np.flip(test_data, axis=1), n_nulls=n_nulls, seed=1, rotation_method=rotation_method, randomize=randomize, residual=residual)

    assert np.allclose(np.flip(a1, axis=2), a2, atol=1e-10), \
        "Nulls generated with the same seed should be identical regardless of input data order"

@pytest.mark.parametrize("rotation_method", rotation_options)
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options) 
def test_seed_single_multiple(solver, test_data, rotation_method, randomize, residual):
    """Test that the same seed produces the same nulls regardless of whether seed is passed as single value or array"""
    a1 = solver.eigenstrap(test_data, n_nulls=2, seed=np.arange(2), rotation_method=rotation_method, randomize=randomize, residual=residual)

    for i in range(2):
        # If the input seed is a tuple or array, it should use that seed directly
        a2 = solver.eigenstrap(test_data, n_nulls=1, seed=(i,), rotation_method=rotation_method, randomize=randomize, residual=residual)
        assert np.allclose(a1[:, i:i+1, :], a2, atol=1e-10), \
            f"Nulls generated with seed {i} should be identical"
        
        # If the input seed is a single integer, it should use that to spawn a new set of seed(s)
        b1 = solver.eigenstrap(test_data, n_nulls=1, seed=i, rotation_method=rotation_method, randomize=randomize, residual=residual)
        assert not np.allclose(a1[:, i:i+1, :], b1, atol=1e-10), \
            f"Nulls generated with seed {i} should be different when passed as single integer vs array"

@pytest.mark.parametrize("randomize", [False])  # only rotation_method may be affected by global state
@pytest.mark.parametrize("residual", [None])    # these methods use the new Generators and will not be affected
def test_seed_global_scipy_match_orig(solver, test_data, randomize, residual): 
    """Setting the global seed should produce the same nulls (to maintain compatibility with original implementation)"""
    np.random.seed(1) # set global seed
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method="scipy", randomize=randomize, residual=residual)
    np.random.seed(1) # reset global seed
    a2 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method="scipy", randomize=randomize, residual=residual)
    np.random.seed(2) # change global seed
    b1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method="scipy", randomize=randomize, residual=residual)

    assert np.allclose(a1, a2), \
        "Nulls generated with the same global seed should be identical for seed=None and rotation_method='scipy'"
    assert not np.allclose(a1, b1), \
        "Nulls generated with different global seeds should be different for seed=None and rotation_method='scipy'"

@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options)
def test_seed_global_qr(solver, test_data, randomize, residual): 
    """Setting the global seed should not affect nulls generated with QR rotation"""
    np.random.seed(1)           # set random states in two different ways
    np.random.default_rng(1)    # neither of these change inner state for QR rotation
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method="qr", randomize=randomize, residual=residual)
    np.random.seed(1)
    np.random.default_rng(1)
    b1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=None, rotation_method="qr", randomize=randomize, residual=residual)
    
    assert not np.allclose(a1, b1), \
        "Nulls generated with the same global seed should not be identical for seed=None and rotation_method='qr'"

@pytest.mark.parametrize("rotation_method", rotation_options)
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options)
@pytest.mark.parametrize("seed", [0, 42, np.arange(n_nulls)])
def test_specific_seed_not_affected_by_global_seed(solver, test_data, rotation_method, randomize, residual, seed): 
    """Setting the global seed should not affect nulls generated with a specific seed (to maintain compatibility with original implementation)"""
    np.random.seed(1) # set global seed
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, randomize=randomize, residual=residual)
    np.random.seed(2) # change global seed
    a2 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, randomize=randomize, residual=residual)

    assert np.allclose(a1, a2), \
        f"Nulls generated with the same specific seed (seed={seed}) should be identical regardless of global seed for rotation_method={rotation_method}"

@pytest.mark.parametrize("rotation_method", rotation_options)
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options)
def test_reproducibility_number_nulls(solver, test_data, rotation_method, randomize, residual):
    """Nulls with same seed should be identical, regardless of number of nulls requested"""
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=1, rotation_method=rotation_method, randomize=randomize, residual=residual)
    a2 = solver.eigenstrap(test_data, n_nulls=n_nulls-1, seed=1, rotation_method=rotation_method, randomize=randomize, residual=residual)

    assert np.allclose(a1[:,:-1], a2, atol=1e-10), \
        "Nulls with the same seed should be identical"

@pytest.mark.parametrize("rotation_method", rotation_options)
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", residual_options)
def test_reproducibility_number_data(solver, test_data, rotation_method, randomize, residual):
    """Nulls with same seed should be identical, regardless of number of input maps"""
    a1 = solver.eigenstrap(test_data, n_nulls=n_nulls, seed=1, rotation_method=rotation_method, randomize=randomize, residual=residual)
    a2 = solver.eigenstrap(test_data[:,:-1], n_nulls=n_nulls, seed=1, rotation_method=rotation_method, randomize=randomize, residual=residual)

    assert np.allclose(a1[:,:,:-1], a2, atol=1e-10), \
        "Nulls with the same seed should be identical"

@pytest.mark.parametrize("rotation_method", ['qr']) # only new method supports this
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", [None, 'add']) # can't be permute so we can accurately get back betas
@pytest.mark.parametrize("seed", [0, 42, np.random.choice(n_nulls, n_nulls, replace=False)]) # supported for all seeds except None
def test_reproducibility_number_groups_qr(solver, test_data, rotation_method, randomize, residual, seed):
    """Nulls with same seed should be identical, regardless of number of groups"""
    n_groups2 = int(np.sqrt(solver.n_modes)/2)
    a1 = solver.eigenstrap(test_data[:,:1], n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, 
                           randomize=False, residual=residual, n_groups=int(np.sqrt(solver.n_modes)))
    a2 = solver.eigenstrap(test_data[:,:1], n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, 
                           randomize=False, residual=residual, n_groups=n_groups2)
    # Nulls won't be exactly equal, but betas should be
    beta1 = solver.decompose(np.squeeze(a1, axis=2))
    beta2 = solver.decompose(np.squeeze(a2, axis=2))

    assert np.allclose(beta1[:n_groups2**2,:], beta2[:n_groups2**2,:], atol=1e-10), \
        "Nulls with the same seed should be identical regardless of number of groups"

@pytest.mark.parametrize("rotation_method", ['scipy']) # scipy needs to specify individiual seeds to support this
@pytest.mark.parametrize("randomize", randomize_options)
@pytest.mark.parametrize("residual", [None, 'add']) # can't be permute so we can accurately get back betas
@pytest.mark.parametrize("seed", [np.random.choice(n_nulls, n_nulls, replace=False)]) # supported only for all seeds specified
def test_reproducibility_number_groups_scipy(solver, test_data, rotation_method, randomize, residual, seed):
    """Nulls with same seed should be identical, regardless of number of groups"""
    n_groups2 = int(np.sqrt(solver.n_modes)/2)
    a1 = solver.eigenstrap(test_data[:,:1], n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, 
                           randomize=False, residual=residual, n_groups=int(np.sqrt(solver.n_modes)))
    a2 = solver.eigenstrap(test_data[:,:1], n_nulls=n_nulls, seed=seed, rotation_method=rotation_method, 
                           randomize=False, residual=residual, n_groups=n_groups2)
    # Nulls won't be exactly equal, but betas should be
    beta1 = solver.decompose(np.squeeze(a1, axis=2))
    beta2 = solver.decompose(np.squeeze(a2, axis=2))

    assert np.allclose(beta1[:n_groups2**2,:], beta2[:n_groups2**2,:], atol=1e-10), \
        "Nulls with the same seed should be identical regardless of number of groups"

def test_compared_to_original_seed_outside(nulls_orig): 
    # These parameters are hard coded to match data saved in the repo and should not be changed
    density = '4k'
    hemi = 'L'
    surf_type = 'midthickness'
    n_modes = 10**2
    n_nulls = 100
    seed = 365
    data = 'myelinmap'

    # Load data
    mesh, medmask = fetch_surf(density=density, hemi=hemi, surf_type=surf_type)
    map = fetch_map(data, density=density)[medmask]
    map = (map - np.mean(map)) # to match original implementation which doesn't use the constant mode

    # Compute new nulls
    solver = EigenSolver(mesh, mask=medmask).solve(n_modes, fix_mode1=True)
    np.random.seed(seed)            # matches original seed=seed 
    nulls_neuromodes = eigenstrap(
        emodes=solver.emodes,
        evals=solver.evals,
        data=map,
        n_nulls=n_nulls,
        checks=False,               # skip checks to match original implementation which doesn't have these checks
        mass=csc_matrix(eye(solver.n_verts)),# matches original which doesn't account for vertex areal differences
        residual=None,              # matches original add_res=False and permute=False
        resample="range",           # matches original resample=False
        decomp_method="regress",    # matches original decomp_method='matrix'
        rotation_method="scipy",    # matches original rotations.indirect_method (called by geometry.gen_eigensamples)
    )

    # Compare (on diagonal near 1, off diagonal near 0)
    null_corrs = np.corrcoef(nulls_neuromodes.T, nulls_orig.T)[:n_nulls, n_nulls:]

    diagonal_corrs = np.diagonal(null_corrs)
    print(f"Min: {np.min(diagonal_corrs):.3f}")
    print(f"Max: {np.max(diagonal_corrs):.3f}")
    assert np.allclose(diagonal_corrs, 1.0, atol=0.001), \
        'New nulls should be similar to corresponding old nulls'

    column_mean = np.mean(null_corrs - np.diag(diagonal_corrs), axis=0)
    assert np.allclose(column_mean, 0.0, atol=0.1), \
        'New nulls should not be similar to different old nulls'
    assert np.allclose(np.mean(column_mean), 0.0, atol=0.01), \
        'New nulls should not be similar to different old nulls'

def test_compared_to_original_seed_inside(nulls_orig): 
    # These parameters are hard coded to match data saved in the repo and should not be changed
    density = '4k'
    hemi = 'L'
    surf_type = 'midthickness'
    n_modes = 10**2
    n_nulls = 100
    seed = 365
    data = 'myelinmap'

    # Load data
    mesh, medmask = fetch_surf(density=density, hemi=hemi, surf_type=surf_type)
    map = fetch_map(data, density=density)[medmask]
    map = (map - np.mean(map)) # to match original implementation which doesn't use the constant mode

    # Compute new nulls
    solver = EigenSolver(mesh, mask=medmask).solve(n_modes, fix_mode1=True)
    nulls_neuromodes = eigenstrap(
        emodes=solver.emodes,
        evals=solver.evals,
        data=map,
        n_nulls=n_nulls,
        checks=False,               # skip checks to match original implementation which doesn't have these checks
        mass=csc_matrix(eye(solver.n_verts)), # matches original which doesn't account for vertex areal differences
        seed=seed,                  # matches original seed=seed 
        residual=None,              # matches original add_res=False and permute=False
        resample="range",           # matches original resample=False
        decomp_method="regress",    # matches original decomp_method='matrix'
        rotation_method="scipy",    # matches original rotations.indirect_method (called by geometry.gen_eigensamples)
    )

    # Compare (on diagonal near 1, off diagonal near 0)
    null_corrs = np.corrcoef(nulls_neuromodes.T, nulls_orig.T)[:n_nulls, n_nulls:]

    diagonal_corrs = np.diagonal(null_corrs)
    assert np.allclose(diagonal_corrs, 1.0, atol=0.001), \
        'New nulls should be similar to corresponding old nulls'

    column_mean = np.mean(null_corrs - np.diag(diagonal_corrs), axis=0)
    assert np.allclose(column_mean, 0.0, atol=0.1), \
        'New nulls should not be similar to different old nulls'
    assert np.isclose(np.mean(column_mean), 0.0, atol=0.01), \
        'New nulls should not be similar to different old nulls'
