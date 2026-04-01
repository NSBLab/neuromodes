import numpy as np
import pytest
from neuromodes.basis import (decompose, reconstruct, reconstruct_timeseries, calc_norm_power,
                              calc_vec_fc)
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_surf, fetch_map

@pytest.fixture(scope='module')
def solver():
    surf, medmask = fetch_surf(density='4k')
    hetero = np.random.default_rng(0).standard_normal(size=len(medmask))
    return EigenSolver(surf, mask=medmask, hetero=hetero).solve(n_modes=10, seed=0)

def test_decompose_eigenmodes(solver):
    emodes = solver.emodes

    for i in range(solver.n_modes):
        data = emodes[:, i]  # Use an eigenmode as data
        beta = decompose(data, emodes, mass=solver.mass)

        # The mode should load onto only itself due to orthogonality
        beta_expected = np.zeros((solver.n_modes, 1))
        beta_expected[i, 0] = 1
        assert np.allclose(beta, beta_expected, atol=1e-4), f'Decomposition of mode {i+1} failed.'

def test_decompose_invalid_data_shape(solver):

    with pytest.raises(ValueError, match=r"emodes \(3619\)."):
        decompose(np.ones(4002), solver.emodes, mass=solver.mass)

def test_decompose_nan_inf_mode(solver):
    emodes = solver.emodes
    data = np.ones(solver.n_verts)

    bad_emodes = emodes.copy()

    bad_emodes[0,0] = np.nan
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        decompose(data, bad_emodes, mass=solver.mass)

    bad_emodes[0,0] = np.inf
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        decompose(data, bad_emodes, mass=solver.mass)

def test_decompose_massless(solver):

    with pytest.raises(ValueError, match="do not form an orthonormal basis set in Euclidean space"):
        decompose(np.ones(solver.n_verts), solver.emodes)

def test_decompose_invalid_method(solver):

    with pytest.raises(ValueError,
                       match="Invalid method 'fornitonian'; must be 'project' or 'regress'."):
        decompose(np.ones(solver.n_verts), solver.emodes, method='fornitonian')

@pytest.fixture(scope='module')
def solver_32k():
    # Get modes of fsLR 32k midthickness (data is in 32k)
    mesh, medmask = fetch_surf()
    rng = np.random.default_rng(0)
    hetero = rng.standard_normal(size=len(medmask))
    solver = EigenSolver(mesh, mask=medmask, hetero=hetero)
    solver.solve(n_modes=10)
    return solver

def test_decompose_nans(solver_32k):
    # Decompose some maps
    data = np.stack(
        (fetch_map('fcgradient1')[solver_32k.mask],
         fetch_map('myelinmap')[solver_32k.mask]),
        axis=1
    )
    beta = solver_32k.decompose(data, method='regress')

    # Append data with NaNs and Infs (+100 vertices)
    extraverts = 100
    data_naninfs = np.concatenate([
        data,
        np.full((extraverts // 2, data.shape[1]), np.nan),
        np.full((extraverts // 2, data.shape[1]), np.inf)
    ], axis=0)

    # Add noise to modes and mass to match shapes (+100 vertices)
    noise = np.random.default_rng(0).standard_normal(
         extraverts * solver_32k.n_modes
        ).reshape((extraverts, solver_32k.n_modes))
    modes_noise = np.concatenate([solver_32k.emodes, noise], axis=0)

    # emodes/mass get masked according to the nans/infs in data, leading to original beta values
    with pytest.warns(UserWarning, match="data and emodes"):
        beta_masked = decompose(data_naninfs, modes_noise, method='regress', checks=False)
    assert np.allclose(beta, beta_masked, atol=1e-4), \
        'Beta values for project method are not close when data contains NaNs/Infs'

# TODO: more complicated version of above test, where three maps have two unique patterns of NaNs/Infs

@pytest.fixture(scope='module')
def gen_eigenmap(solver):

    # Use randomly weighted sums of modes to generate maps
    n_maps = 5
    rng = np.random.default_rng(0)
    weights = rng.standard_normal(size=(solver.n_modes, n_maps))
    eigenmaps = solver.emodes @ weights

    return eigenmaps, weights

def test_reconstruct_mode_superposition(solver, gen_eigenmap):
    eigenmaps, weights = gen_eigenmap

    recon, correlation_error, beta = reconstruct(eigenmaps, solver.emodes, mass=solver.mass)

    # Correlation error should decrease from 1 to 0 when using mode 1 only versus all relevant modes
    assert np.allclose(recon[:,-1,:], eigenmaps,
                       atol=1e-5), 'Final reconstructions do not match input maps.'
    assert np.allclose(correlation_error[-1,:], 0,
                       atol=1e-5), 'Correlation error is not close to 0 when using all modes.'

    assert np.allclose(beta[-1], weights, atol=1e-4), \
        'Beta values do not match input mode weights when using all modes.'

    # Euclidean error should be 0 when using all modes
    _, euclidean_error, _ = reconstruct(eigenmaps, solver.emodes, mass=solver.mass, metric='euclidean')
    assert np.allclose(euclidean_error[-1,:], 0,
                       atol=1e-5), 'Euclidean error is not close to 0 when using all modes.'

    # Reconstruct using the first 5 modes, then the first 2 modes
    _, correlation_error_modesq, _ = reconstruct(eigenmaps, solver.emodes, mass=solver.mass, mode_counts=[5,2])
    assert np.allclose(correlation_error_modesq[0,:], correlation_error[4,:]), \
        'Reconstruction scores do not match for 5 modes.'
    assert np.allclose(correlation_error_modesq[1,:], correlation_error[1,:]), \
        'Reconstruction scores do not match for 2 modes.'

def test_reconstruct_regress_method(solver, gen_eigenmap):
    eigenmaps, _ = gen_eigenmap

    _, correlation_error, _ = reconstruct(eigenmaps, solver.emodes, method='regress', checks=False, metric='correlation')
    _, euclidean_error, _ = reconstruct(eigenmaps, solver.emodes, method='regress', checks=False, metric='euclidean')

    # Errors should strictly decrease when adding modes
    assert np.all(np.diff(correlation_error[1:,:], axis=0) < 0), \
        'Correlation error does not strictly decrease when adding modes.'
    assert np.all(np.diff(euclidean_error, axis=0) < 0), \
        'Euclidean error does not strictly decrease when adding modes.'

# When mode_counts contains 1 (e.g. when mode_counts is None, the default), the timeseries is
# reconstructed using only the first (constant) mode. This leads to a simulated timeseries which is
# the same at each vertex (at any timepoint), creating an FC matrix which is 1 everywhere. When
# z-transforming the matrix, this results in a RuntimeWarning (due to division by 0) and an output
# which has inf values. This also creates another warning when using the 'correlation' metric due to
# the prescence of inf values. These behaviours are reasonable, and should be flagged for the user,
# but we can filter these warnings for this test. Due to precision errors, some reconstructed FC
# matrices may have a correlation of 1 which leads to NaN values in the correlation_error output. 
# This is also reasonable. This can be mitigated by using more timepoints in gen_eigenmap, but for
# computaional efficiency we only use 3 timepoints.
@pytest.mark.filterwarnings("ignore:divide by zero encountered in arctanh:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
def test_reconstruct_mode_superposition_timeseries(solver, gen_eigenmap):
    eigenmaps, _ = gen_eigenmap

    eigen_ts = eigenmaps.astype(np.float64) # Prevent memory allocation error
    fc = calc_vec_fc(eigen_ts)

    # Treat eigenmaps as timepoints of activity
    fc_recon, correlation_error, recon, recon_error, beta = reconstruct_timeseries(
        eigen_ts, solver.emodes, method='regress', checks=False, metric='correlation')
    
    # check shapes
    assert fc_recon.shape == (solver.n_verts*(solver.n_verts-1)/2, solver.n_modes), \
        'fc_recon has incorrect shape.'
    assert correlation_error.shape == (solver.n_modes,), \
        'fc_recon_error has incorrect shape.'
    assert recon.shape == (solver.n_verts, solver.n_modes, eigen_ts.shape[1]), \
        'recon has incorrect shape.'
    assert recon_error.shape == (solver.n_modes, eigen_ts.shape[1]), \
        'recon_error has incorrect shape.'
    assert beta[0].shape == (1, eigen_ts.shape[1]), \
        'beta[0] has incorrect shape.'
    assert beta[-1].shape == (solver.n_modes, eigen_ts.shape[1]), \
        'beta[-1] has incorrect shape.'

    # Use another metric for fc recon error
    _, euclidean_error, _, _, _ = reconstruct_timeseries(
        eigen_ts, solver.emodes, method='regress', checks=False, metric='euclidean')
    mse = euclidean_error / fc.size  # Convert to MSE
    
    assert np.allclose(np.tanh(fc_recon[:,-1]), np.tanh(fc), atol=1e-5), \
        'Reconstructed FC does not match original.'
    assert correlation_error[-1] < 1e-6, \
        'FC reconstruction error is not close to 0 when using all modes.'
    assert mse[-1] < 1e-6, 'MSE is not close to 0 when using all modes.'

def test_reconstruct_real_map_32k(solver_32k):
    emodes = solver_32k.emodes

    # Load FC gradient from Margulies 2016 PNAS
    map = fetch_map('fcgradient1')[solver_32k.mask]
    _, recon_score, _ = reconstruct(map, emodes, mass=solver_32k.mass)

    # Correlation error should strictly decrease from 1, but not reach 0
    assert np.all(np.diff(recon_score) < 0), 'Reconstruction error does not strictly decrease.'
    assert not np.isclose(recon_score[-1], 0, atol=1e-6), \
        'Reconstruction error is unexpectedly close to 0 for only 10 modes.'

def test_reconstruct_invalid_map_shape(solver):

    with pytest.raises(ValueError, match=r"emodes \(3619\)."):
        reconstruct(np.ones(4002), solver.emodes, mass=solver.mass)

def test_reconstruct_massless(solver):

    with pytest.raises(ValueError, match="do not form an orthonormal basis set in Euclidean space"):
        reconstruct(np.ones(solver.n_verts), solver.emodes)

def test_calc_norm_power():
    # Dummy coefficients
    beta = np.array([[-3, 4], [1.5, 2], [0, 0.1]])

    norm_power = calc_norm_power(beta)

    # Check that powers are non-negative
    assert np.all(norm_power >= 0), 'Normalized powers contain negative values.'

    # Check that columns sum to 1
    assert np.allclose(np.sum(norm_power, axis=0), 1, atol=1e-8), \
        'Normalized powers do not sum to 1.'