import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.distance import cdist, pdist
from scipy.stats import zscore
from neuromodes.stats import (gramw, dotw, ssqw, lstsqw, solvew, cdistw, pdistw, meanw, demeanw,
                              varw, stdw, zscorew, covw, vecnormw)

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

    def test_solvew_eye(self, random_data):
        X, Y, eye, noneye = random_data

        solve_w = solvew(X, Y, mass=eye)
        solve_u = np.linalg.solve(X.T @ X, X.T @ Y)
        assert np.allclose(solve_w, solve_u), "Solve with identity mass should be the same as " \
                                              "unweighted solve."
        
        solve_wn = solvew(X, Y, mass=noneye)
        assert not np.allclose(solve_wn, solve_w), "Solve with non-identity mass should not be " \
                                                   "the same as unweighted solve."
        
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

class TestShapes:
    def test_dotw_ssqw_3d(self, random_data):
        X, Y, eye, noneye = random_data
        X3 = np.stack([X, X + 1.0], axis=-1)
        Y3 = np.stack([Y, Y - 1.0], axis=-1)

        dot_w = dotw(X3, Y3, mass=eye)
        dot_u = np.einsum('vij,vij->ij', X3, Y3)
        assert dot_w.shape == X3.shape[1:]
        assert np.allclose(dot_w, dot_u)

        dot_wn = dotw(X3, Y3, mass=noneye)
        assert not np.allclose(dot_wn, dot_w)

        ssq_w = ssqw(X3, mass=eye)
        ssq_u = np.sum(X3**2, axis=0)
        assert ssq_w.shape == X3.shape[1:]
        assert np.allclose(ssq_w, ssq_u)