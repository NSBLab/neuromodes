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

def test_decompose_eigenmodes_1d(solver):
    for i in range(solver.n_modes):
        # Use an eigenmode as data
        beta = decompose(solver.emodes[:, i], solver.emodes, mass=solver.mass)

        # The mode should load onto only itself due to orthogonality
        beta_expected = np.zeros((solver.n_modes,))
        beta_expected[i] = 1
        assert np.allclose(beta, beta_expected, atol=1e-4), \
            f'Decomposition of mode {i} failed.'

def test_decompose_eigenmodes_2d(solver):
    emodes = solver.emodes
    beta = decompose(data=emodes, emodes=emodes, mass=solver.mass)
    beta_expected = np.eye(solver.n_modes)
    assert np.allclose(beta, beta_expected, atol=1e-4), \
        f'Decomposition of modes onto themselves failed.'
    
def test_decompose_eigenmodes_3d(solver):
    emodes = solver.emodes
    data = np.stack((emodes, emodes), axis=2)
    beta = decompose(data=data, emodes=emodes, mass=solver.mass)
    beta_expected = np.stack((np.eye(solver.n_modes), np.eye(solver.n_modes)), axis=2)
    assert np.allclose(beta, beta_expected, atol=1e-4), \
        f'Decomposition of modes onto themselves failed for 3D data.'

def test_decompose_invalid_data_shape(solver):

    with pytest.raises(ValueError, match=r"data.*first dimension.*3619"):
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
                       match="Invalid method 'foo'; must be 'project' or 'regress'."):
        decompose(np.ones(solver.n_verts), solver.emodes, method='foo') # type: ignore

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
    noise = np.random.default_rng().standard_normal((extraverts, solver_32k.n_modes))
    modes_noise = np.concatenate([solver_32k.emodes, noise], axis=0)

    # emodes/mass get masked according to the nans/infs in data, leading to original beta values
    with pytest.warns(UserWarning, match="values detected in data"):
        beta_masked = decompose(data_naninfs, modes_noise, method='regress', checks='maps')
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

def test_reconstruct_project(solver):
    recon, _, beta = reconstruct(solver.emodes, solver.emodes, mass=solver.mass)
    assert np.allclose(recon, solver.emodes,
                       atol=1e-5), 'Final reconstructions do not match input modes.'
    assert np.allclose(beta, np.eye(solver.n_modes), atol=1e-5), \
        'Beta values do not match expected identity matrix when reconstructing modes onto themselves.'
    
def test_reconstruct_regress(solver):
    recon, _, beta = reconstruct(solver.emodes, solver.emodes, method='regress')
    assert np.allclose(recon, solver.emodes,
                       atol=1e-5), 'Final reconstructions do not match input modes.'
    assert np.allclose(beta, np.eye(solver.n_modes), atol=1e-5), \
        'Beta values do not match expected identity matrix when reconstructing modes onto themselves.'

def test_reconstruct_mode_superposition(solver, gen_eigenmap):
    eigenmaps, weights = gen_eigenmap

    recon, correlation_error, beta = reconstruct(eigenmaps, solver.emodes, mass=solver.mass, mode_counts=np.arange(solver.n_modes)+1)

    # Correlation error should decrease from 1 to 0 when using mode 1 only versus all relevant modes
    assert np.allclose(recon[:,:,-1], eigenmaps,
                       atol=1e-5), 'Final reconstructions do not match input maps.'
    assert np.allclose(correlation_error[:,-1], 0,
                       atol=1e-5), 'Correlation error is not close to 0 when using all modes.'

    assert np.allclose(beta[-1], weights, atol=1e-4), \
        'Beta values do not match input mode weights when using all modes.'

    # Euclidean error should be 0 when using all modes
    _, euclidean_error, _ = reconstruct(eigenmaps, solver.emodes, mass=solver.mass, mode_counts=np.arange(solver.n_modes)+1, metric='euclidean')
    assert np.allclose(euclidean_error[:,-1], 0,
                       atol=1e-5), 'Euclidean error is not close to 0 when using all modes.'

    # Reconstruct using the first 5 modes, then the first 2 modes
    _, correlation_error_modesq, _ = reconstruct(eigenmaps, solver.emodes, mass=solver.mass, mode_counts=[5,2])
    assert np.allclose(correlation_error_modesq[:,0], correlation_error[:,4]), \
        'Reconstruction scores do not match for 5 modes.'
    assert np.allclose(correlation_error_modesq[:,1], correlation_error[:,1]), \
        'Reconstruction scores do not match for 2 modes.'

def test_reconstruct_regress_method(solver, gen_eigenmap):
    eigenmaps, _ = gen_eigenmap

    _, correlation_error, _ = reconstruct(eigenmaps, solver.emodes, method='regress', checks=False, metric='correlation', mode_counts=np.arange(solver.n_modes)+1)
    _, euclidean_error, _ = reconstruct(eigenmaps, solver.emodes, method='regress', checks=False, metric='euclidean', mode_counts=np.arange(solver.n_modes)+1)

    # Errors should strictly decrease when adding modes
    assert np.all(np.diff(correlation_error, axis=1) < 0), \
        'Correlation error does not strictly decrease when adding modes.'
    assert np.all(np.diff(euclidean_error, axis=1) < 0), \
        'Euclidean error does not strictly decrease when adding modes.'

# When mode_counts contains 1, the timeseries is reconstructed using only the first (constant) mode.
# This leads to a simulated timeseries which is the same at each vertex (at any timepoint), creating
# an FC matrix which is 1 everywhere. When z-transforming the matrix, this results in a
# RuntimeWarning (due to division by 0) and an output which has inf values. This also creates
# another warning when using the 'correlation' metric due to the prescence of inf values. These
# behaviours are reasonable, and should be flagged for the user, but we can filter these warnings
# for this test. Due to precision errors, some reconstructed FC matrices may have a correlation of 1
# which leads to NaN values in the correlation_error output. This is also reasonable. This can be
# mitigated by using more timepoints in gen_eigenmap, but for computaional efficiency we only use 3
# timepoints.
@pytest.mark.filterwarnings("ignore:divide by zero encountered in arctanh:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
def test_reconstruct_mode_superposition_timeseries(solver, gen_eigenmap):
    eigenmaps, _ = gen_eigenmap

    eigen_ts = eigenmaps.astype(np.float64) # Prevent memory allocation error
    fc = calc_vec_fc(eigen_ts)

    # Treat eigenmaps as timepoints of activity
    fc_recon, correlation_error, recon, recon_error, beta = reconstruct_timeseries(
        eigen_ts, solver.emodes, method='regress', checks=False, metric='correlation', mode_counts=np.arange(solver.n_modes)+1)
    
    # check shapes
    assert fc_recon.shape == (solver.n_verts*(solver.n_verts-1)/2, solver.n_modes), \
        'fc_recon has incorrect shape.'
    assert correlation_error.shape == (solver.n_modes,), \
        'fc_recon_error has incorrect shape.'
    assert recon.shape == (solver.n_verts, eigen_ts.shape[1], solver.n_modes), \
        'recon has incorrect shape.'
    assert recon_error.shape == (eigen_ts.shape[1], solver.n_modes), \
        'recon_error has incorrect shape.'
    assert beta[0].shape == (1, eigen_ts.shape[1]), \
        'beta[0] has incorrect shape.'
    assert beta[-1].shape == (solver.n_modes, eigen_ts.shape[1]), \
        'beta[-1] has incorrect shape.'

    # Use another metric for fc recon error
    _, euclidean_error, _, _, _ = reconstruct_timeseries(
        eigen_ts, solver.emodes, method='regress', checks=False, metric='euclidean', mode_counts=np.arange(solver.n_modes)+1)
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
    _, recon_score, _ = reconstruct(map, emodes, mass=solver_32k.mass, mode_counts=np.arange(solver_32k.n_modes)+1)

    # Correlation error should strictly decrease from 1, but not reach 0
    assert np.all(np.diff(recon_score[1:]) < 0), 'Reconstruction error does not strictly decrease.'
    assert not np.isclose(recon_score[-1], 0, atol=1e-6), \
        'Reconstruction error is unexpectedly close to 0 for only 10 modes.'

def test_reconstruct_invalid_map_shape(solver):

    with pytest.raises(ValueError, match=r"data.*first dimension.*3619"):
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

class TestShape: 
    def test_decompose_1d(self, solver):
        for i in range(solver.n_modes):
            beta = decompose(solver.emodes[:, i], solver.emodes, mass=solver.mass)
            assert beta.shape == (solver.n_modes,), \
                'Beta shape is incorrect for 1D data.'

    def test_decompose_1d_trivial(self, solver):
        for i in range(solver.n_modes):
            beta = decompose(solver.emodes[:, i:i+1], solver.emodes, mass=solver.mass)
            assert beta.shape == (solver.n_modes, 1), \
                'Beta shape is incorrect for 2D data with one column.'

    def test_decompose_2d(self, solver):
        beta_decomposed = decompose(solver.emodes, solver.emodes, mass=solver.mass)
        assert beta_decomposed.shape == (solver.n_modes, solver.n_modes), \
            'Beta shape is incorrect for 2D data.'
    
    def test_decompose_3d(self, solver):
        data = np.random.default_rng().standard_normal((solver.n_verts, 5, 3))
        beta_decomposed = decompose(data, solver.emodes, mass=solver.mass, mode_counts=solver.n_modes)
        assert beta_decomposed.shape == (solver.n_modes, 5, 3), \
            'Beta shape is incorrect for 3D data.'
    
    def test_decompose_1d_mode_counts(self, solver): 
        beta = np.random.default_rng().standard_normal(solver.n_modes)
        data = solver.emodes @ beta
        mode_counts = np.random.default_rng().integers(1, solver.n_modes+1, size=10)
        beta_decomposed = decompose(data, solver.emodes, mass=solver.mass, mode_counts=mode_counts)
        assert len(beta_decomposed) == len(mode_counts), \
            'Number of beta outputs does not match number of mode counts.'
        for i in range(len(mode_counts)):
            assert beta_decomposed[i].shape == (mode_counts[i],), \
                f'Beta shape is incorrect for mode count {mode_counts[i]}'

    def test_decompose_1d_mode_ids(self, solver): 
        beta = np.random.default_rng().standard_normal(solver.n_modes)
        data = solver.emodes @ beta
        n_modes = np.random.default_rng().integers(1, solver.n_modes, size=3)
        mode_ids = [np.random.default_rng().choice(solver.n_modes, size=k, replace=False) for k in n_modes]
        beta_decomposed = decompose(data, solver.emodes, mass=solver.mass, mode_ids=mode_ids)
        assert len(beta_decomposed) == len(mode_ids), \
            'Number of beta outputs does not match number of mode ID sets.'
        for i in range(len(mode_ids)):
            assert beta_decomposed[i].shape == (len(mode_ids[i]),), \
                f'Beta shape is incorrect for mode IDs {mode_ids[i]}'

    def test_decompose_2d_mode_counts(self, solver):
        beta = np.random.default_rng().standard_normal((solver.n_modes, 5))
        data = solver.emodes @ beta
        mode_counts = np.random.default_rng().integers(1, solver.n_modes+1, size=10)
        beta_decomposed = decompose(data, solver.emodes, mass=solver.mass, mode_counts=mode_counts)
        assert len(beta_decomposed) == len(mode_counts), \
            'Number of beta outputs does not match number of mode counts.'
        for i in range(len(mode_counts)):
            assert beta_decomposed[i].shape == (mode_counts[i], 5), \
                f'Beta shape is incorrect for mode count {mode_counts[i]}'

    def test_reconstruct_1d(self, solver):
        for i in range(solver.n_modes):
            recon, _, _ = reconstruct(solver.emodes[:,i], solver.emodes, mass=solver.mass)
            assert recon.shape == (solver.n_verts,), \
                'Reconstruction shape does not match number of vertices for 1D data.'

    def test_reconstruct_2d_trivial(self, solver):
        for i in range(solver.n_modes):
            recon, _, _ = reconstruct(solver.emodes[:,i:i+1], solver.emodes, mass=solver.mass)
            assert recon.shape == (solver.n_verts,1), \
                'Reconstruction shape does not match number of vertices for 2D data with one column.'              

    def test_reconstruct_2d(self, solver):
        recon, _, _ = reconstruct(solver.emodes, solver.emodes, mass=solver.mass)
        assert recon.shape == (solver.n_verts, solver.n_modes), \
            'Reconstruction shape does not match input modes for 2D data.'

    def test_reconstruct_2d_random(self, solver): 
        for _ in range(10): 
            ndims = np.random.default_rng().integers(1, 5) # data will be between 2 and 6 dimensional
            shape = np.random.default_rng().integers(1, 10, size=ndims) # actual size 
            data = np.random.default_rng().standard_normal((solver.n_verts, *shape))
            recon, _, _ = reconstruct(data, solver.emodes, mass=solver.mass)
            assert recon.shape == (solver.n_verts, *shape), \
                'Reconstruction shape does not match expected shape for random data.'

    def test_reconstruct_1d_mode_counts(self, solver): 
        mode_counts = np.random.default_rng().integers(1, solver.n_modes+1, size=10)
        recon, _, _ = reconstruct(solver.emodes[:, 0], solver.emodes, mass=solver.mass, mode_counts=mode_counts)
        assert recon.shape == (solver.n_verts, len(mode_counts)), \
            'Reconstruction shape does not match expected shape for 1D data with mode counts.'

    def test_reconstruct_1d_mode_ids(self, solver):
        n_modes = np.random.default_rng().integers(1, solver.n_modes, size=3)
        mode_ids = [np.random.default_rng().choice(solver.n_modes, size=k, replace=False) for k in n_modes]
        recon, _, _ = reconstruct(solver.emodes[:, 0], solver.emodes, mass=solver.mass, mode_ids=mode_ids)
        assert recon.shape == (solver.n_verts, len(mode_ids)), \
            f'Reconstruction shape does not match expected shape for 1D data with mode IDs.'
            
    def test_reconstruct_2d_mode_counts(self, solver):
        mode_counts = np.random.default_rng().integers(1, solver.n_modes+1, size=10)
        recon, _, _ = reconstruct(solver.emodes, solver.emodes, mass=solver.mass, mode_counts=mode_counts)
        assert recon.shape == (solver.n_verts, solver.n_modes, len(mode_counts)), \
            'Reconstruction shape does not match expected shape for 2D data with mode counts.'

    def test_reconstruct_2d_mode_ids(self, solver):
        n_modes = np.random.default_rng().integers(1, solver.n_modes, size=3)
        mode_ids = [np.random.default_rng().choice(solver.n_modes, size=k, replace=False) for k in n_modes]
        recon, _, _ = reconstruct(solver.emodes, solver.emodes, mass=solver.mass, mode_ids=mode_ids)
        assert recon.shape == (solver.n_verts, solver.n_modes, len(mode_ids)), \
            'Reconstruction shape does not match expected shape for 2D data with mode IDs.'
