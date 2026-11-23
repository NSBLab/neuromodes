from functools import partial

import numpy as np
import pytest
from scipy.sparse import csc_matrix, eye

from neuromodes.basis import decompose, calc_recon_error, reconstruct, filter_spectrum
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_example_map, fetch_example_surf
from neuromodes.stats import sigmoid_rescale, zscorew, _multiply_nd

DATA_SHAPES = [(), (1,), (5,), (5,1), (5,3), (5,3,1), (5,3,2)]
ATOL = {'project': 1e-7, 'regress': 1e-2}

@pytest.fixture(scope='module')
def solver():
    surf, medmask = fetch_example_surf(res='4k')
    randmap = np.random.default_rng(0).standard_normal(size=medmask.sum())
    solver = EigenSolver(surf, mask=medmask)
    hetero = sigmoid_rescale(zscorew(randmap, solver.mass), steepness=0.5, upper=2.0)
    return solver.solve(n_modes=10, hetero=hetero)

class TestIdentity: 
    @pytest.mark.parametrize("method", ['project', 'regress'])
    def test_identity_eigenmodes_1d(self, solver, method):
        for i in range(solver.n_modes):
            # Use an eigenmode as data
            coeffs = decompose(solver.emodes[:, i], solver.emodes, mass=solver.mass, method=method)

            # The mode should load onto only itself due to orthogonality
            coeffs_expected = np.zeros((solver.n_modes,))
            coeffs_expected[i] = 1
            assert np.allclose(coeffs, coeffs_expected), \
                f'Decomposition of mode {i} failed.'

            recon = reconstruct(coeffs, solver.emodes)
            assert np.allclose(recon, solver.emodes[:, i]), \
                f'Final reconstruction of mode {i} does not match input mode.'

            filtered = filter_spectrum(solver.emodes[:, i], solver.emodes, solver.evals, mass=solver.mass, method=method)
            assert np.allclose(filtered, solver.emodes[:, i]), \
                f'Filtered mode {i} does not match input mode.'

            error = calc_recon_error(solver.emodes[:, i], filtered, mass=solver.mass, metric='euclidean')
            np.testing.assert_allclose(error, 0, atol=ATOL[method])

    @pytest.mark.parametrize("method", ['project', 'regress'])
    def test_identity_eigenmodes_2d(self, solver, method):
        emodes = solver.emodes

        coeffs = decompose(data=emodes, emodes=emodes, mass=solver.mass, method=method)
        assert np.allclose(coeffs, np.eye(solver.n_modes)), \
            'Decomposition of modes onto themselves failed.'

        recon = reconstruct(coeffs, emodes)
        assert np.allclose(recon, emodes), \
            'Final reconstructions do not match input modes.'

        filtered = filter_spectrum(emodes, emodes, solver.evals, mass=solver.mass, method=method)
        assert np.allclose(filtered, emodes), \
            'Filtered modes do not match input modes.'

        error = calc_recon_error(emodes, filtered, mass=solver.mass, metric='euclidean')
        np.testing.assert_allclose(error, 0, atol=ATOL[method])

    @pytest.mark.parametrize("method", ['project', 'regress'])
    def test_identity_eigenmodes_3d(self, solver, method):
        emodes = solver.emodes
        data = np.stack((emodes, emodes), axis=2)

        coeffs = decompose(data=data, emodes=emodes, mass=solver.mass, method=method)
        coeffs_expected = np.stack((np.eye(solver.n_modes), np.eye(solver.n_modes)), axis=2)
        assert np.allclose(coeffs, coeffs_expected), \
            'Decomposition of modes onto themselves failed for 3D data.'

        recon = reconstruct(coeffs, emodes)
        assert np.allclose(recon, data), \
            'Final reconstructions do not match input modes for 3D data.'

        filtered = filter_spectrum(data, emodes, solver.evals, mass=solver.mass, method=method)
        assert np.allclose(filtered, data), \
            'Filtered modes do not match input modes for 3D data.'

        error = calc_recon_error(data, filtered, mass=solver.mass, metric='euclidean')
        np.testing.assert_allclose(error, 0, atol=ATOL[method])

class TestRandomMaps:
    @pytest.mark.parametrize("method", ['project', 'regress'])
    @pytest.mark.parametrize("data_shape", DATA_SHAPES)
    def test_none(self, solver, data_shape, method):
        weights = np.random.default_rng().standard_normal((solver.n_modes,) + data_shape)
        data = _multiply_nd(solver.emodes, weights)

        coeffs = decompose(data, solver.emodes, mass=solver.mass, method=method)
        assert coeffs.shape == weights.shape, \
            'coeffs shape is incorrect for the given data shape.'
        assert np.allclose(coeffs, weights), \
            'Decomposition of random maps failed.'
        
        recon = reconstruct(coeffs, solver.emodes)
        assert recon.shape == data.shape, \
            'recon shape is incorrect for the given data shape.'
        assert np.allclose(recon, data), \
            'Final reconstructions of random maps do not match input data.'
        
        filtered = filter_spectrum(data, solver.emodes, solver.evals, mass=solver.mass, method=method)
        assert filtered.shape == data.shape, \
            'filtered shape is incorrect for the given data shape.'
        assert np.allclose(filtered, data), \
            'Filtered data does not match input data.'

        error = calc_recon_error(data, filtered, mass=solver.mass)
        assert error.shape == data_shape, \
            'error shape is incorrect for the given data shape.'
        np.testing.assert_allclose(error, 0, atol=ATOL[method])

    @pytest.mark.parametrize("method", ['project', 'regress'])
    @pytest.mark.parametrize("mode_counts", [2, 3, 5, 10])
    @pytest.mark.parametrize("data_shape", DATA_SHAPES)
    def test_mode_counts(self, solver, mode_counts, data_shape, method):
        weights = np.random.default_rng().standard_normal((mode_counts,) + data_shape)
        data = _multiply_nd(solver.emodes[:, :mode_counts], weights)

        coeffs = decompose(data, solver.emodes, mass=solver.mass, mode_counts=mode_counts, method=method)
        assert coeffs.shape == weights.shape, \
            'coeffs shape is incorrect for the given data shape.'
        assert np.allclose(coeffs, weights), \
            'Decomposition of random maps failed.'

        coeffs_all = decompose(data, solver.emodes, mass=solver.mass)
        assert np.allclose(coeffs_all[:mode_counts], weights), \
            'Full decomposition does not match partial decomposition for mode counts.'

        recon = reconstruct(coeffs, solver.emodes, mode_counts=mode_counts)
        assert recon.shape == data.shape, \
            'recon shape is incorrect for the given data shape.'
        assert np.allclose(recon, data), \
            'Final reconstructions of random maps do not match input data.'

        filtered = filter_spectrum(data, solver.emodes, solver.evals, mass=solver.mass, mode_counts=mode_counts, method=method)
        assert filtered.shape == data.shape, \
            'filtered shape is incorrect for the given data shape.'
        assert np.allclose(filtered, data), \
            'Filtered data does not match input data.'

        error = calc_recon_error(data, filtered, mass=solver.mass)
        assert error.shape == data_shape, \
            'error shape is incorrect for the given data shape.'
        np.testing.assert_allclose(error, 0, atol=ATOL[method])

    @pytest.mark.parametrize("method", ['project', 'regress'])
    @pytest.mark.parametrize("mode_counts", [[2,3,4], [5,10]])
    @pytest.mark.parametrize("data_shape", DATA_SHAPES)
    def test_mode_counts_list(self, solver, mode_counts, data_shape, method):
        weights = np.random.default_rng().standard_normal((max(mode_counts),) + data_shape)
        data = _multiply_nd(solver.emodes[:, :max(mode_counts)], weights)

        coeffs = decompose(data, solver.emodes, mass=solver.mass, mode_counts=mode_counts, method=method)
        assert len(coeffs) == len(mode_counts), \
            'Number of coeffs outputs does not match number of mode counts.'
        for i, mc in enumerate(mode_counts):
            assert coeffs[i].shape == (mc,) + data_shape, \
                f'coeffs shape is incorrect for mode count {mc}'
            np.testing.assert_allclose(coeffs[i], weights[:mc], atol=ATOL[method])

        recon = reconstruct(coeffs, solver.emodes, mode_counts=mode_counts)
        assert len(recon) == len(mode_counts), \
            'Number of recon outputs does not match number of mode counts.'
        for i, mc in enumerate(mode_counts):
            assert recon[i].shape == data.shape, \
                f'recon shape is incorrect for mode count {mc}'
            curr = _multiply_nd(solver.emodes[:, :mc], weights[:mc])
            np.testing.assert_allclose(recon[i], curr, atol=ATOL[method])

        filtered = filter_spectrum(data, solver.emodes, solver.evals, mass=solver.mass, mode_counts=mode_counts, method=method)
        assert filtered.shape == data.shape + (len(mode_counts),), \
            'filtered shape is incorrect for the given data shape.'
        for i, mc in enumerate(mode_counts):
            curr = _multiply_nd(solver.emodes[:, :mc], weights[:mc])
            np.testing.assert_allclose(filtered[..., i], curr, atol=ATOL[method])

        error = calc_recon_error(data, filtered, mass=solver.mass)
        assert error.shape == data_shape + (len(mode_counts),), \
            'error shape is incorrect for the given data shape.'
        np.testing.assert_allclose(error[..., mode_counts==max(mode_counts)], 0, atol=ATOL[method])

    @pytest.mark.parametrize("method", ['project', 'regress'])
    @pytest.mark.parametrize("mode_ids", [[0,1,2,3,6,9], range(0,10,2)])
    @pytest.mark.parametrize("data_shape", DATA_SHAPES)
    def test_mode_ids(self, solver, mode_ids, data_shape, method):
        weights = np.random.default_rng().standard_normal((len(mode_ids),) + data_shape)
        data = _multiply_nd(solver.emodes[:, mode_ids], weights)

        coeffs = decompose(data, solver.emodes, mass=solver.mass, mode_ids=mode_ids, method=method)
        assert coeffs.shape == (len(mode_ids),) + data_shape, \
            f'coeffs shape is incorrect for mode IDs {mode_ids}'
        assert np.allclose(coeffs, weights), \
            f'Decomposition of random maps failed for mode IDs {mode_ids}'

        coeffs_all = decompose(data, solver.emodes, mass=solver.mass)
        assert np.allclose(coeffs_all[mode_ids], weights), \
            f'Full decomposition does not match partial decomposition for mode IDs {mode_ids}'

        recon = reconstruct(coeffs, solver.emodes, mode_ids=mode_ids)
        assert recon.shape == data.shape, \
            f'recon shape is incorrect for mode IDs {mode_ids}'
        assert np.allclose(recon, data), \
            f'Final reconstructions of random maps do not match input data for mode IDs {mode_ids}'

        filtered = filter_spectrum(data, solver.emodes, solver.evals, mass=solver.mass, mode_ids=mode_ids, method=method)
        assert filtered.shape == data.shape, \
            f'filtered shape is incorrect for mode IDs {mode_ids}'
        assert np.allclose(filtered, data), \
            f'Filtered data does not match input data for mode IDs {mode_ids}'

        error = calc_recon_error(data, filtered, mass=solver.mass)
        assert error.shape == data_shape, \
            f'error shape is incorrect for mode IDs {mode_ids}'
        np.testing.assert_allclose(error, 0, atol=ATOL[method])

    @pytest.mark.parametrize("method", ['project', 'regress'])
    @pytest.mark.parametrize("mode_ids", [[[1,2,3],[4,5,6],[7,8,9]],[range(0,10,2), range(0,10,3)]])
    @pytest.mark.parametrize("data_shape", DATA_SHAPES)
    def test_mode_ids_list(self, solver, mode_ids, data_shape, method):
        weights = np.random.default_rng().standard_normal((solver.n_modes,) + data_shape)
        data = _multiply_nd(solver.emodes, weights)

        coeffs = decompose(data, solver.emodes, mass=solver.mass, mode_ids=mode_ids, method=method)
        assert len(coeffs) == len(mode_ids), \
            'Number of coeffs outputs does not match number of mode ID sets.'
        for i, mids in enumerate(mode_ids):
            assert coeffs[i].shape == (len(mids),) + data_shape, \
                f'coeffs shape is incorrect for mode IDs {mids}'
            np.testing.assert_allclose(coeffs[i], weights[mids], atol=ATOL[method])

        recon = reconstruct(coeffs, solver.emodes, mode_ids=mode_ids)
        assert len(recon) == len(mode_ids), \
            'Number of recon outputs does not match number of mode ID sets.'
        for i, mids in enumerate(mode_ids):
            assert recon[i].shape == data.shape, \
                f'recon shape is incorrect for mode IDs {mids}'
            curr = _multiply_nd(solver.emodes[:, mids], weights[mids])
            np.testing.assert_allclose(recon[i], curr, atol=ATOL[method])            

        filtered = filter_spectrum(data, solver.emodes, solver.evals, mass=solver.mass, mode_ids=mode_ids, method=method)
        assert filtered.shape == data.shape + (len(mode_ids),), \
            'filtered shape is incorrect for the given data shape.'
        for i, mids in enumerate(mode_ids):
            curr = _multiply_nd(solver.emodes[:, mids], weights[mids])
            np.testing.assert_allclose(filtered[..., i], curr, atol=ATOL[method])

        error = calc_recon_error(data, filtered, mass=solver.mass)
        assert error.shape == data_shape + (len(mode_ids),), \
            'error shape is incorrect for the given data shape.'
        # can't do error testing here due to how modes are constructed

class TestInvalid:
    def test_decompose_invalid_data_shape(self, solver):
        with pytest.raises(ValueError, match=r"invalid shape"):
            decompose(np.ones(solver.n_verts-1), solver.emodes, mass=solver.mass)

    def test_decompose_massless(self, solver):
        with pytest.warns(UserWarning, match="Mass"):
            decompose(np.ones(solver.n_verts), solver.emodes)

    def test_decompose_invalid_method(self, solver):
        with pytest.raises(ValueError,
                        match="Invalid method 'foo'; must be 'project' or 'regress'."):
            decompose(np.ones(solver.n_verts), solver.emodes, method='foo', mass=solver.mass) # type: ignore


# TODO: more complicated version of nan test, where three maps have two unique patterns of NaNs

@pytest.mark.parametrize("method", ['project', 'regress'])
@pytest.mark.parametrize("data_shape", DATA_SHAPES)
@pytest.mark.parametrize("transfer_function", [
    lambda evals: np.ones_like(evals),
    lambda evals: np.exp(-evals),
    lambda evals: np.diag(1/(1 + 0.5**2 * evals))
    ])
def test_filters(solver, method, data_shape, transfer_function):
    # Generate random maps
    weights = np.random.default_rng().standard_normal((solver.n_modes,) + data_shape)

    filtered_evals = transfer_function(solver.evals)
    if filtered_evals.ndim == 1:
        filtered_evals = np.diag(filtered_evals)
    expected_weights = _multiply_nd(filtered_evals, weights)
    
    data = _multiply_nd(solver.emodes, weights)
    expected_data = _multiply_nd(solver.emodes, expected_weights)
    filtered = filter_spectrum(data, solver.emodes, solver.evals, mass=solver.mass, transfer_function=transfer_function, method=method)

    np.testing.assert_allclose(filtered, expected_data)
    np.testing.assert_allclose(solver.decompose(filtered), expected_weights)

@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.parametrize("method", ['project', 'regress'])
def test_error_vs_nmodes(solver, method):
    weights = np.random.default_rng().standard_normal((solver.n_modes, 5))
    eigenmaps = solver.emodes @ weights

    recon = filter_spectrum(eigenmaps, solver.emodes, solver.evals, mass=solver.mass, method=method, mode_counts=range(1, solver.n_modes+1))
    cor_error = calc_recon_error(eigenmaps, recon, mass=solver.mass, metric='correlation')
    euc_error = calc_recon_error(eigenmaps, recon, mass=solver.mass, metric='euclidean')

    # Errors should strictly decrease when adding modes
    assert np.all(np.diff(cor_error[:, 1:], axis=1) < 0), \
        'Correlation error does not strictly decrease when adding modes.'  # nan error for constant mode (col 0), so ignore
    assert np.all(np.diff(euc_error, axis=1) < 0), \
        'Euclidean error does not strictly decrease when adding modes.'
    np.testing.assert_allclose(np.nan_to_num(cor_error[:,0], nan=1.0), 1.0)
    np.testing.assert_allclose(cor_error[:,-1], 0, atol=1e-6)
    np.testing.assert_allclose(euc_error[:,-1], 0, atol=1e-6)

