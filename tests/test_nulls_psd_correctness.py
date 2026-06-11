import pytest
import numpy as np
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_example_surf

# Params
density = '4k'
hemi = 'L'
n_modes = 100 # should be perfect square
n_maps = 3
n_nulls = 20

@pytest.fixture(scope='module')
def solver():
    """Initialise solver and solve for eigenmodes, which will be used for all tests."""
    mesh, _ = fetch_example_surf(density=density, hemi=hemi, surf_type='sphere')
    return EigenSolver(mesh).solve(n_modes=n_modes)

# These are the main parameters which will exactly preserve PSD. For example, `resample='exact'`
# will not.
@pytest.mark.parametrize("rotation_method", ['scipy', 'qr'])
@pytest.mark.parametrize("randomize", [True, False])
@pytest.mark.parametrize("decomp_method", ['project', 'regress'])
@pytest.mark.parametrize("seed", [None, 0, 42, np.arange(n_nulls)])
def test_psd_preservation(solver, rotation_method, randomize, decomp_method, seed):
    """Nulls should preserve eigengroup power spectral density of map on sphere"""    
    # Synthetic data for testing - generated from known psd
    beta0_mode = np.random.default_rng().normal(loc=1, size=(solver.n_modes, n_maps))
    data = solver.emodes @ beta0_mode
    nulls = solver.eigenstrap(data, 
                              n_nulls=n_nulls, 
                              rotation_method=rotation_method, 
                              randomize=randomize, 
                              decomp_method=decomp_method, 
                              seed=seed) # these parameters should not affect PSD preservation

    # Compute group level PSD for original map and nulls
    psd0_mode = beta0_mode**2
    psd1_mode = solver.decompose(nulls.reshape(solver.n_verts, -1), method=decomp_method).reshape(-1, n_nulls, n_maps)**2

    split_indices = np.arange(int(np.ceil(np.sqrt(solver.n_modes))))**2
    psd0_group = np.add.reduceat(psd0_mode, split_indices, axis=0) # (n_groups, n_maps)
    psd1_group = np.add.reduceat(psd1_mode, split_indices, axis=0) # (n_groups, n_nulls, n_maps)

    # Check
    assert np.allclose(psd0_group[:, np.newaxis, :], psd1_group, rtol=0.01), \
        f"Nulls do not preserve eigengroup PSD (rotation_method={rotation_method}, randomize={randomize}, decomp_method={decomp_method}, seed={seed})"

@pytest.mark.parametrize("n_nulls", [10000]) # need to use a large number of nulls to get good estimate of mean
def test_randomize_True(solver, n_nulls): 
    """When `randomize=True`, each mode within a group should have same mean psd"""
    # note that the overall group accuracy is tested in `test_psd_preservation`
    # Synthetic data for testing - generated from known psd
    beta0_mode = np.random.default_rng().normal(loc=1, size=(solver.n_modes, n_maps))
    data = solver.emodes @ beta0_mode
    nulls = solver.eigenstrap(data, n_nulls=n_nulls, randomize=True)

    # Compute mode level PSD for original map and nulls
    psd1_mode = solver.decompose(nulls.reshape(solver.n_verts, -1)).reshape(n_modes, n_nulls, n_maps)**2
    psd1_mode_mean = psd1_mode.mean(axis=1)

    group_idx = np.sqrt(np.arange(n_modes)).astype(int)
    for i in range(group_idx.max() + 1):
        curr = psd1_mode_mean[group_idx == i, :]
        mean = curr.mean(axis=0)
        assert np.allclose(curr, mean, atol=0.1, rtol=0.1), \
            f"When `randomize=True`, modes within group {i} should have same mean PSD, but got {curr} instead of {mean}"

# TODO: consider testing the distributions of the null beta coefficients: 
#   - each null should be chi-square distributed with dof = number of modes in group
#   -  mean = function of group PSD, and original beta of that mode if randomize=False

# TODO: test that default params preserve mass-weighted mean of map
