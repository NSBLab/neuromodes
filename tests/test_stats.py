import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.distance import cdist, pdist
from scipy.stats import zscore

from neuromodes import EigenSolver
from neuromodes.io import fetch_example_surf
from neuromodes.stats import (
    _mult_by_cholesky,
    cdistw,
    covw,
    demeanw,
    dotw,
    gramw,
    lstsqw,
    meanw,
    parcellate,
    pdistw,
    sigmoid_rescale,
    ssqw,
    stdw,
    varw,
    vecnormw,
    zscorew,
)


@pytest.fixture(scope='module')
def random_data():
    """Generate random data for testing."""
    rng = np.random.default_rng(0)
    n_verts = 10
    n_maps = 3

    X = rng.standard_normal(size=(n_verts, n_maps))
    Y = rng.standard_normal(size=(n_verts, n_maps))

    eye = sparse.eye(n_verts)  # identity mass for testing
    noneye = sparse.diags(np.arange(1, n_verts + 1), dtype=np.float64)
    return X, Y, eye, noneye

class TestEye:
    def test_gramw_eye(self, random_data):
        X, Y, eye, noneye = random_data
        gram_w = gramw(X, Y, mass=eye)
        gram_u = X.T @ Y
        assert np.allclose(gram_w, gram_u), "Gram matrix with identity mass should be the same " \
                                            "as unweighted Gram matrix."
        gram_wn = gramw(X, Y, mass=noneye)
        assert not np.allclose(gram_wn, gram_w), "Gram matrix with non-identity mass should not " \
                                                 "be the same " \

    def test_dotw_eye(self, random_data):
        X, Y, eye, noneye = random_data
        dot_w = dotw(X, Y, mass=eye)
        dot_u = np.einsum('ij,ij->j', X, Y)
        assert np.allclose(dot_w, dot_u), "Dot product with identity mass should be the same as " \
                                          "unweighted dot product."
        
        dot_wn = dotw(X, Y, mass=noneye)
        assert not np.allclose(dot_wn, dot_w), "Dot product with non-identity mass should not " \
                                               "be the same as unweighted dot product."
    def test_ssqw_eye(self, random_data):
        X, _, eye, noneye = random_data
        ssq_w = ssqw(X, mass=eye)
        ssq_u = np.sum(X**2, axis=0)
        assert np.allclose(ssq_w, ssq_u), "Sum of squares with identity mass should be the same " \
                                          "as unweighted sum of squares."
        
        ssq_wn = ssqw(X, mass=noneye)
        assert not np.allclose(ssq_wn, ssq_w), "Sum of squares with non-identity mass should not " \
                                               "be the same as unweighted sum of squares."
        
    def test_meanw_eye(self, random_data):
        X, _, eye, noneye = random_data
        mean_w = meanw(X, mass=eye)
        mean_u = np.mean(X, axis=0)
        assert np.allclose(mean_w, mean_u), "Mean with identity mass should be the same as " \
                                            "unweighted mean."
        
        mean_wn = meanw(X, mass=noneye)
        assert not np.allclose(mean_wn, mean_w), "Mean with non-identity mass should not be the " \
                                                 "same as unweighted mean."
        
    def test_varw_eye(self, random_data):
        X, _, eye, noneye = random_data
        var_w = varw(X, mass=eye)
        var_u = np.var(X, axis=0, ddof=0)
        assert np.allclose(var_w, var_u), "Variance with identity mass should be the same as " \
                                          "unweighted variance."
        
        var_wn = varw(X, mass=noneye)
        assert not np.allclose(var_wn, var_w), "Variance with non-identity mass should not be " \
                                               "the same as unweighted variance."
        
    def test_stdw_eye(self, random_data):
        X, _, eye, noneye = random_data
        std_w = stdw(X, mass=eye)
        std_u = np.std(X, axis=0)
        assert np.allclose(std_w, std_u), "Standard deviation with identity mass should be the " \
                                          "same as unweighted standard deviation."
        
        std_wn = stdw(X, mass=noneye)
        assert not np.allclose(std_wn, std_w), "Standard deviation with non-identity mass should " \
                                               "not be the same as unweighted standard deviation."
        
    def test_zscorew_eye(self, random_data):
        X, _, eye, noneye = random_data
        zscore_w = zscorew(X, mass=eye)
        zscore_u = zscore(X)
        assert np.allclose(zscore_w, zscore_u), "Z-score with identity mass should be the same " \
                                                "as unweighted z-score."
        
        zscore_wn = zscorew(X, mass=noneye)
        assert not np.allclose(zscore_wn, zscore_w), "Z-score with non-identity mass should not " \
                                                     "be the same as unweighted z-score."

    def test_covw_eye(self, random_data):
        X, Y, eye, noneye = random_data
        cov_w = covw(X, Y, mass=eye)
        cov_u = np.cov(X, Y, rowvar=False, bias=True)[:X.shape[1], X.shape[1]:]
        assert np.allclose(cov_w, cov_u), "Covariance with identity mass should be the same as " \
                                          "unweighted covariance (without Bessel correction)."
        
        cov_wn = covw(X, Y, mass=noneye)
        assert not np.allclose(cov_wn, cov_w), "Covariance with non-identity mass should not be " \
                                               "the same as unweighted covariance."

    def test_vecnormw_eye(self, random_data):
        X, _, eye, noneye = random_data
        norm_w = vecnormw(X, mass=eye)
        norm_u = np.linalg.norm(X, axis=0)
        assert np.allclose(norm_w, norm_u), "Vector norm with identity mass should be the same " \
                                            "as unweighted vector norm."
        
        norm_wn = vecnormw(X, mass=noneye)
        assert not np.allclose(norm_wn, norm_w), "Vector norm with non-identity mass should not " \
                                                 "be the same as unweighted vector norm."

    def test_demeanw_eye(self, random_data):
        X, _, eye, noneye = random_data
        demean_w = demeanw(X, mass=eye)
        demean_u = X - np.mean(X, axis=0)
        assert np.allclose(demean_w, demean_u), "Demean with identity mass should be the same as " \
                                                "unweighted demean."
        
        demean_wn = demeanw(X, mass=noneye)
        assert not np.allclose(demean_wn, demean_w), "Demean with non-identity mass should not " \
                                                     "be the same as unweighted demean."
        
    def test_lstsqw_eye(self, random_data):
        X, Y, eye, noneye = random_data
        lstsq_w = lstsqw(X, Y, mass=eye)[0]
        lstsq_u = np.linalg.lstsq(X, Y, rcond=None)[0]
        assert np.allclose(lstsq_w, lstsq_u), "Least squares with identity mass should be the " \
                                              "same as unweighted least squares."
        
        lstsq_wn = lstsqw(X, Y, mass=noneye)[0]
        assert not np.allclose(lstsq_wn, lstsq_w), "Least squares with non-identity mass should " \
                                                   "not be the same as unweighted least squares."
        
    def test_cdistw_eye(self, random_data):
        X, Y, eye, noneye = random_data
        for metric in ['euclidean', 'sqeuclidean', 'cosine', 'correlation']:
            cdist_w = cdistw(X, Y, mass=eye, metric=metric)
            cdist_u = cdist(X.T, Y.T, metric=metric)
            assert np.allclose(cdist_w, cdist_u), "Pairwise distances with identity mass should " \
                                                  "be the same as unweighted pairwise distances " \
                                                  f"for metric '{metric}'."
            
            cdist_wn = cdistw(X, Y, mass=noneye, metric=metric)
            assert not np.allclose(cdist_wn, cdist_w), (
                "Pairwise distances with non-identity mass should not be the same as unweighted "
                f"pairwise distances for metric '{metric}'."
            )

    def test_pdistw_eye(self, random_data):
        X, _, eye, noneye = random_data

        for metric in ['euclidean', 'sqeuclidean', 'cosine', 'correlation']:
            pdist_w = pdistw(X, mass=eye, metric=metric)
            pdist_u = pdist(X.T, metric=metric)
            assert np.allclose(pdist_w, pdist_u), "Pairwise distances with identity mass should " \
                                                  "be the same as unweighted pairwise distances " \
                                                  f"for metric '{metric}'."
            
            pdist_wn = pdistw(X, mass=noneye, metric=metric)
            assert not np.allclose(pdist_wn, pdist_w), (
                "Pairwise distances with non-identity mass should not be the same as unweighted "
                f"pairwise distances for metric '{metric}'."
            )

class Test1D:
    def test_dotw_1d(self, random_data):
        X, Y, eye, _ = random_data
        x_1d = X[:, 0]
        y_1d = Y[:, 0]
        assert np.allclose(dotw(x_1d, y_1d, mass=eye), np.dot(x_1d, y_1d))
    
    def test_gramw_1d(self, random_data):
        X, Y, eye, _ = random_data
        x_1d = X[:, 0]
        y_1d = Y[:, 0]
        gram_w = gramw(x_1d, y_1d, mass=eye)
        gram_u = np.dot(x_1d, y_1d)
        assert np.allclose(gram_w, gram_u), "Gramw with 1D input should match dot product."
    
    def test_ssqw_1d(self, random_data):
        X, _, eye, _ = random_data
        x_1d = X[:, 0]
        ssq_w = ssqw(x_1d, mass=eye)
        ssq_u = np.sum(x_1d**2)
        assert np.allclose(ssq_w, ssq_u), "Ssqw with 1D input should match sum of squares."
    
    def test_meanw_1d(self, random_data):
        X, _, eye, _ = random_data
        x_1d = X[:, 0]
        mean_w = meanw(x_1d, mass=eye)
        mean_u = np.mean(x_1d)
        assert np.allclose(mean_w, mean_u), "Meanw with 1D input should match unweighted mean."
    
    def test_demeanw_1d(self, random_data):
        X, _, eye, _ = random_data
        x_1d = X[:, 0]
        demean_w = demeanw(x_1d, mass=eye)
        demean_u = x_1d - np.mean(x_1d)
        assert np.allclose(demean_w, demean_u), "Demeanw with 1D input should match unweighted demean."
    
    def test_varw_1d(self, random_data):
        X, _, eye, _ = random_data
        x_1d = X[:, 0]
        var_w = varw(x_1d, mass=eye)
        var_u = np.var(x_1d, ddof=0)
        assert np.allclose(var_w, var_u), "Varw with 1D input should match unweighted variance."
    
    def test_stdw_1d(self, random_data):
        X, _, eye, _ = random_data
        x_1d = X[:, 0]
        std_w = stdw(x_1d, mass=eye)
        std_u = np.std(x_1d)
        assert np.allclose(std_w, std_u), "Stdw with 1D input should match unweighted std dev."
    
    def test_zscorew_1d(self, random_data):
        X, _, eye, _ = random_data
        x_1d = X[:, 0]
        zscore_w = zscorew(x_1d, mass=eye)
        zscore_u = zscore(x_1d)
        assert np.allclose(zscore_w, zscore_u), "Zscorew with 1D input should match unweighted z-score."
    
    def test_covw_1d(self, random_data):
        X, Y, eye, _ = random_data
        x_1d = X[:, 0]
        y_1d = Y[:, 0]
        cov_w = covw(x_1d, y_1d, mass=eye)
        cov_u = np.cov(x_1d, y_1d, bias=True)[0, 1]
        assert np.allclose(cov_w, cov_u), "Covw with 1D input should match unweighted covariance."
    
    def test_vecnormw_1d(self, random_data):
        X, _, eye, _ = random_data
        x_1d = X[:, 0]
        norm_w = vecnormw(x_1d, mass=eye)
        norm_u = np.linalg.norm(x_1d)
        assert np.allclose(norm_w, norm_u), "Vecnormw with 1D input should match unweighted norm."
    
    def test_lstsqw_1d(self, random_data):
        X, Y, eye, _ = random_data
        x_1d = X[:, 0]
        y_1d = Y[:, 0]
        # lstsqw expects 2D, so reshape 1D to (n, 1)
        lstsq_w = lstsqw(x_1d[:, np.newaxis], y_1d[:, np.newaxis], mass=eye)[0]
        lstsq_u = np.linalg.lstsq(x_1d[:, np.newaxis], y_1d[:, np.newaxis], rcond=None)[0]
        assert np.allclose(lstsq_w, lstsq_u), "Lstsqw with 1D input should match unweighted lstsq."

    # TODO: add test to check that lstsqw output for consistent mass matches np.linalg.solve(a.T @
    # mass @ a, a.T @ mass @ b)

    # TODO: add test to check that lstsq output is different for lumped vs consistent mass and
    # matches some expectation?
    
    def test_cdistw_1d(self, random_data):
        X, Y, eye, _ = random_data
        x_1d = X[:, 0]
        y_1d = Y[:, 0]
        # cdistw expects column vectors
        cdist_w = cdistw(x_1d, y_1d, mass=eye)
        cdist_u = cdist(x_1d[:, np.newaxis].T, y_1d[:, np.newaxis].T)
        assert np.allclose(cdist_w, cdist_u), "Cdistw with 1D input should match unweighted cdist."

def test_sigmoid_rescale():
    size = 1000
    randmap = np.random.default_rng().normal(loc=1e6, scale=10, size=size)

    # check that sigmoid_rescaled map is all 2s due to high mean
    hetero_sig = sigmoid_rescale(randmap, upper=2)
    assert np.allclose(hetero_sig, 2.0), 'Sigmoid rescaled map with high mean should be all 2s.'

    # Check that sigmoid-scaled z-scored map is within (0, 2)
    heteroz_sig = sigmoid_rescale(zscorew(randmap, sparse.eye(size)), steepness=100, upper=2)
    assert np.all((heteroz_sig >= 0) & (heteroz_sig <= 2)), \
        'Sigmoid rescaled z-scored map should be within (0, 2).'
    
    # TODO: check that low values are close to 0 and high values are close to 2
    # TODO: check that steepness controls the range of values in the sigmoid rescaled map
    # TODO: check that lower and upper bounds are respected
    # TODO: check that negative steepness reverses rank order exactly

class TestParcellate:
    @pytest.fixture(scope='class')
    def parcellation(self, random_data):
        n_verts = random_data[0].shape[0]
        parc = np.zeros(n_verts, dtype=int)
        parc[n_verts//3:] = 1
        parc[2*n_verts//3:] = 2
        return parc

    def test_parcellate_sum_eye(self, random_data, parcellation):
        X, _, eye, _ = random_data
        n_verts = X.shape[0]

        # identity mass and method='sum' should sum values within parcels
        parc_sum = parcellate(X, parcellation, mass=eye, method='sum')
        expected_sum = np.array([
            X[:n_verts//3].sum(axis=0),
            X[n_verts//3:2*n_verts//3].sum(axis=0),
            X[2*n_verts//3:].sum(axis=0)
            ])
        assert np.allclose(parc_sum, expected_sum), \
            "Parcellate with identity mass and method='sum' should sum vertex values."
    
    def test_parcellate_mean_eye(self, random_data, parcellation):
        X, _, eye, _ = random_data
        n_verts = X.shape[0]

        # identity mass and method='mean' should average values within parcels
        parc_mean = parcellate(X, parcellation, mass=eye, method='mean')
        expected_mean = np.array([
            X[:n_verts//3].mean(axis=0),
            X[n_verts//3:2*n_verts//3].mean(axis=0),
            X[2*n_verts//3:].mean(axis=0)
            ])
        assert np.allclose(parc_mean, expected_mean), \
            "Parcellate with identity mass and method='mean' should average vertex values."

    def test_parcellate_sum_vs_mean(self, random_data, parcellation):
        X, _, eye, _ = random_data
        X_abs = np.abs(X)

        parc_sum = parcellate(X_abs, parcellation, mass=eye, method='sum')
        parc_mean = parcellate(X_abs, parcellation, mass=eye, method='mean')

        # Sum should be larger than mean for positive values (as in random data)
        assert np.all(parc_sum > parc_mean), \
            "Parcellate sum should be > mean for positive data."
    
    def test_parcellate_1d_data(self, random_data, parcellation):
        X, _, eye, _ = random_data
        X_1d = X[:, 0]  # Get first column as 1D

        # Test with 1D data should return 1D result
        parc_result = parcellate(X_1d, parcellation, mass=eye, method='mean')
        assert parc_result.ndim == 1, "Parcellate with 1D data should return 1D result."
        assert parc_result.shape[0] == 3, "Result should have 3 parcels."
    
    def test_parcellate_nonidentity_mass(self, random_data, parcellation):
        X, _, eye, noneye = random_data
        
        parc_eye = parcellate(X, parcellation, mass=eye, method='mean')
        parc_noneye = parcellate(X, parcellation, mass=noneye, method='mean')
        
        # Non-identity mass should produce different results
        assert not np.allclose(parc_eye, parc_noneye), \
            "Parcellate with non-identity mass should produce different results."
    
    def test_parcellate_invalid_method(self, random_data, parcellation):
        X, _, eye, _ = random_data
        
        with pytest.raises(ValueError, match="method must be 'mean' or 'sum'"):
            parcellate(X, parcellation, mass=eye, method='how2say')
    
    def test_parcellate_invalid_data_dim(self, random_data, parcellation):
        X, _, eye, _ = random_data
        X_3d = X[:, :, np.newaxis]
        
        with pytest.raises(ValueError, match="data must be 1D or 2D"):
            parcellate(X_3d, parcellation, mass=eye)
    
    def test_parcellate_invalid_parc_dim(self, random_data, parcellation):
        X, _, eye, _ = random_data
        parc_2d = np.stack([parcellation, parcellation], axis=-1)
        
        with pytest.raises(ValueError, match="Parcellation map must be 1D"):
            parcellate(X, parc_2d, mass=eye)

    # TODO: add something similar to MGH's example where a simple function is irregularly sampled
    # but accounted for by the mass matrix, and check that parcellation recovers the expected values
    # in each parcel.

@pytest.fixture(scope='module')
def solver():
    surf, medmask = fetch_example_surf(density='4k')
    return EigenSolver(surf, medmask)

def test_mult_by_cholesky(solver):
    n_maps = 3

    # generate random maps
    maps = np.random.default_rng(0).standard_normal(size=(solver.n_verts, n_maps))

    # mass-weight maps
    maps_M = solver.mass @ maps

    # mass-weight maps via Cholesky
    maps_LT = _mult_by_cholesky(maps, solver.mass, transpose=True)
    maps_LLT = _mult_by_cholesky(maps_LT, solver.mass, transpose=False)

    assert np.allclose(maps_LLT, maps_M, atol=1e-20), \
        "Mult by Cholesky should satisfy L @ L.T @ maps = mass @ maps"

    # verify that consistent mass != L^T L
    maps_L = _mult_by_cholesky(maps, solver.mass, transpose=False)
    maps_LTL = _mult_by_cholesky(maps_L, solver.mass, transpose=True)

    assert not np.allclose(maps_LTL, maps_M, atol=1e-3), \
        "Mult by Cholesky should not satisfy L.T @ L @ maps = mass @ maps for consistent mass"

    # use lumped mass
    solver.compute_lbo(lump=True)
    maps_M_l = solver.mass @ maps
    maps_LTL_l = _mult_by_cholesky(
        _mult_by_cholesky(maps, solver.mass, transpose=False),
        solver.mass,
        transpose=True
        )

    # verify that lumped mass = L^T L = L L^T
    assert np.allclose(maps_LTL_l, maps_M_l, atol=1e-20), \
        "Mult by Cholesky should satisfy L.T @ L @ maps = mass @ maps for lumped mass"

def test_lstsqw_consistent_mass(solver):
    n_maps = 3
    # ensure that the solver has consistent mass
    solver.compute_lbo(lump=False)

    # generate random maps
    maps = np.random.default_rng(0).standard_normal(size=(solver.n_verts, n_maps))

    # generate random target maps
    targets = np.random.default_rng(1).standard_normal(size=(solver.n_verts, n_maps))

    # solve least squares with consistent mass
    coeffs = lstsqw(maps, targets, mass=solver.mass)[0]

    # check that it solves the normal equations: a.T @ mass @ a @ x = a.T @ mass @ b
    lhs = maps.T @ solver.mass @ maps @ coeffs
    rhs = maps.T @ solver.mass @ targets

    assert np.allclose(lhs, rhs, atol=1e-20), \
        "Lstsqw with consistent mass should solve the normal equations: a.T @ mass @ a @ x = a.T @ mass @ b"

    # check that coeffs differs when mass is lumped
    solver.compute_lbo(lump=True)
    coeffs_lumped = lstsqw(maps, targets, mass=solver.mass)[0]
    assert not np.allclose(coeffs, coeffs_lumped, atol=1e-3), \
        "Lstsqw with lumped mass should produce different coefficients than consistent mass."