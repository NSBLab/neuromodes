from functools import partial

import numpy as np
from scipy.stats import genpareto, kstest, lognorm
import pytest

from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_example_surf
from neuromodes.mbm import (permute_design, compute_stat_map, generate_null_stat_maps, 
                            mbm_calc_p_vals, _fit_and_ad_exceedances)

class TestPermutations:
    @pytest.mark.parametrize("method", ['flip', 'shuffle', 'flip_shuffle'])
    @pytest.mark.parametrize("seed", [None, 42])
    @pytest.mark.parametrize("n_subjects", [1, 5])
    @pytest.mark.parametrize("n_observations", [1, 6])
    @pytest.mark.parametrize("n_permutations", [1, 7])
    def test_permutation_shape(self, method, seed, n_subjects, n_observations, n_permutations):
        """Check that the permutation matrix has the correct shape."""
        design = np.ones((n_subjects, n_observations))  # Dummy design matrix
        perm = permute_design(
            design=design, 
            columns=np.arange(n_observations),
            method=method,
            seed=seed,
            n_permutations=n_permutations
        )

        assert perm.shape == (n_subjects, n_observations, n_permutations)

    @pytest.mark.parametrize("seed", [None, 42])
    @pytest.mark.parametrize("n_subjects", [1, 5])
    @pytest.mark.parametrize("n_observations", [1, 6])
    @pytest.mark.parametrize("n_permutations", [1, 7])
    def test_shuffle_contents(self, seed, n_subjects, n_observations, n_permutations):
        """Check that the permutation matrix has the correct contents."""
        design = np.repeat(np.arange(n_subjects)[:, None], n_observations, axis=1)
        perm = permute_design(
            design=design,
            columns=np.arange(n_observations),
            n_permutations=n_permutations,
            method='shuffle',
            seed=seed,
        )

        assert np.all(
            np.sort(perm, axis=0) == np.sort(design, axis=0)[..., None]
        )

    @pytest.mark.parametrize("seed", [None, 42])
    @pytest.mark.parametrize("n_subjects", [1, 5])
    @pytest.mark.parametrize("n_observations", [1, 6])
    @pytest.mark.parametrize("n_permutations", [1, 7])
    def test_flip_contents(self, seed, n_subjects, n_observations, n_permutations):
        """Check that the permutation matrix has the correct contents."""
        design = np.ones((n_subjects, n_observations))
        perm = permute_design(
            design=design,
            columns=np.arange(n_observations),
            n_permutations=n_permutations,
            method='flip',
            seed=seed,
        )

        assert np.isin(perm, [-1, 1]).all()

    @pytest.mark.parametrize("seed", [None, 42])
    @pytest.mark.parametrize("n_subjects", [1, 5])
    @pytest.mark.parametrize("n_observations", [1, 6])
    @pytest.mark.parametrize("n_permutations", [1, 7])
    def test_flip_shuffle_contents(self, seed, n_subjects, n_observations, n_permutations):
        """Check that the permutation matrix has the correct contents."""
        design = np.repeat(np.arange(n_subjects)[:, None], n_observations, axis=1)
        perm = permute_design(
            design=design,
            columns=np.arange(n_observations),
            n_permutations=n_permutations,
            method='flip_shuffle',
            seed=seed,
        )

        assert np.all(
            np.sort(np.abs(perm), axis=0) == np.sort(design, axis=0)[..., None]
        )
        assert np.isin(perm, np.arange(-n_subjects, n_subjects + 1)).any()

    @pytest.mark.parametrize("method", ['flip', 'shuffle', 'flip_shuffle'])
    @pytest.mark.parametrize("seed", [1, 42])
    @pytest.mark.parametrize("n_subjects", [1, 5])
    @pytest.mark.parametrize("n_observations", [1, 6])
    @pytest.mark.parametrize("n_permutations", [2, 7])
    def test_method_n_permutations_stable(self, method, seed, n_subjects, n_observations, n_permutations):
        """Check that the first n_permutations of a larger permutation set are the same as a smaller permutation set."""
        design = np.repeat(np.arange(n_subjects)[:, None], n_observations, axis=1)
        
        first = permute_design(
            design=design,
            columns=np.arange(n_observations),
            n_permutations=n_permutations,
            method=method,
            seed=seed,
        )

        second = permute_design(
            design=design,
            columns=np.arange(n_observations),
            n_permutations=n_permutations-1,
            method=method,
            seed=seed,
        )

        assert np.array_equal(first[:, :, :-1], second)

    @pytest.mark.parametrize("seed", [1, 42])
    @pytest.mark.parametrize("n_subjects", [2, 5])
    @pytest.mark.parametrize("n_observations", [1, 6])
    @pytest.mark.parametrize("n_permutations", [5, 10])
    def test_flip_n_subjects_stable(self, seed, n_subjects, n_observations, n_permutations):
        """Check that the first n_subjects of a larger flip-sign permutation set are the same as a smaller flip-sign permutation set."""
        first = permute_design(
            design=np.ones((n_subjects, n_observations)), 
            columns=np.arange(n_observations),
            n_permutations=n_permutations,
            method='flip',
            seed=seed,
        )

        second = permute_design(
            design=np.ones((n_subjects-1, n_observations)),
            columns=np.arange(n_observations),
            n_permutations=n_permutations,
            method='flip',
            seed=seed,
        )

        assert np.array_equal(first[:-1], second)

    @pytest.mark.parametrize("method", ['flip', 'shuffle', 'flip_shuffle'])
    @pytest.mark.parametrize("seed", [1, 42])
    @pytest.mark.parametrize("n_subjects", [3, 5])
    @pytest.mark.parametrize("n_observations", [1, 6])
    @pytest.mark.parametrize("n_permutations", [3, 7])
    def test_permutation_different_seeds(self, method, seed, n_subjects, n_observations, n_permutations):
        """Check that different seeds produce different permutation matrices."""
        perm = partial(
            permute_design,
            design=np.repeat(np.arange(n_subjects)[:, None], n_observations, axis=1),
            columns=np.arange(n_observations),
            n_permutations=n_permutations,
            method=method,
        )

        perm1 = perm(seed=seed)
        perm2 = perm(seed=seed+1)
        assert not np.array_equal(perm1, perm2)

    @pytest.mark.parametrize("seed", range(3)) # has to be a fixed set which is known to be within error bounds (up to 50 tested)
    @pytest.mark.parametrize("method", ['shuffle', 'flip_shuffle'])
    @pytest.mark.parametrize("n_subjects", [2, 5])
    @pytest.mark.parametrize("n_permutations", [2500])
    def test_shuffle_distribution(self, seed, method, n_subjects, n_permutations):
        """Check that the distribution of shuffle-row permutations is approximately uniform."""
        design = np.arange(n_subjects)[:, None]
        perm = permute_design(
            design=design,
            columns=np.arange(1),
            n_permutations=n_permutations,
            method=method,
            seed=seed,
        )

        expected_count = n_permutations / n_subjects
        rtol = 4 * np.sqrt((n_subjects - 1) / n_permutations)  # ~4 SDs of a binomial distribution

        for ii in range(n_subjects):
            counts = np.bincount(np.abs(perm[ii, 0, :]), minlength=n_subjects)
            np.testing.assert_allclose(counts, expected_count, rtol=rtol)

    @pytest.mark.parametrize("seed", range(3)) # has to be a fixed set which is known to be within error bounds (up to 50 tested)
    @pytest.mark.parametrize("method", ['flip', 'flip_shuffle'])
    @pytest.mark.parametrize("n_subjects", [2, 5])
    @pytest.mark.parametrize("n_permutations", [2500])
    def test_flip_distribution(self, seed, method, n_subjects, n_permutations):
        """Check that the distribution of flip-sign permutations is approximately uniform."""
        design = np.ones((n_subjects, 1))
        perm = permute_design(
            design=design,
            columns=np.arange(1),
            n_permutations=n_permutations,
            method=method,
            seed=seed,
        )

        np.testing.assert_allclose(np.mean(perm, axis=-1), 0, atol=0.1)  # check each subject (row)
        np.testing.assert_allclose(np.mean(perm), 0, atol=0.05)          # check whole matrix

class TestTStatMapsKnown(): 
    @pytest.fixture(scope="class")
    def solver(self): 
        density = '4k'
        n_modes = 25
        surf, medmask = fetch_example_surf(density=density)
        solver = EigenSolver(surf, mask=medmask).solve(n_modes)
        return solver

    @pytest.mark.parametrize("n_modes", [10, 25])       # n_subjects * 2 > n_modes        
    @pytest.mark.parametrize("n_subjects", [20, 50])    # just to synthesise data for this test
    @pytest.mark.parametrize("seed", [None, 1, 42])
    def test_t_map(self, solver, n_modes, n_subjects, seed): 
        statistic = 't'
        df = 2 * (n_subjects - 1)

        beta = np.arange(n_modes) / n_modes
        emodes = solver.emodes[:, :n_modes]
        spatial_pattern = (emodes @ beta[:,None]).flatten()

        ## Generate synthetic data with known signal and noise structure
        # 1. Zero-group-mean Gaussian noise across subjects: shape (n_verts, 2 * n_subjects)
        Z = np.random.default_rng(seed=seed).normal(size=(solver.n_verts, 2, n_subjects))
        Z = (Z - Z.mean(axis=2, keepdims=True)).reshape(solver.n_verts, 2 * n_subjects)

        # 2. Orthonormal bases for mode space (Q_a) and mode-projected subjects (Q_c)
        Q_a, _ = np.linalg.qr(solver.mass @ emodes) # shape (n_verts, n_modes)
        Q_c, _ = np.linalg.qr(Z.T @ Q_a)            # shape (2 * n_subjects, n_modes)

        # 3. Project residual orthogonal to Q_c (subjects) and Q_a (vertices), then balance norms
        r_norm = np.sqrt(df * (1.0 - np.sum(Q_a ** 2, axis=1, keepdims=True)))
        R = Z - (Z @ Q_c) @ Q_c.T
        for _ in range(10):
            R -= Q_a @ (Q_a.T @ R)
            R *= r_norm / np.linalg.norm(R, axis=1, keepdims=True)

        # 4. Split into groups and add spatial pattern
        U = np.sqrt(df) * (Q_a @ Q_c.T)
        data_a, data_b = np.hsplit(U + R, 2)
        data_b += spatial_pattern[:, np.newaxis]

        ## Set up data
        response = np.vstack((data_a.T, data_b.T))
        design = np.kron(np.array([[1,-1], [1,1]]), np.ones((n_subjects, 1)))
        contrast = np.array([[0, 1]])

        t_expected_vertex = spatial_pattern * np.sqrt(n_subjects / 2)
        t_expected_mode = beta * np.sqrt(n_subjects / 2)

        ## SBM (get t-stat at each vertex)
        t_observed_vertex = compute_stat_map(design, response, contrast, statistic)
        d_observed_vertex = t_observed_vertex / np.sqrt(n_subjects / 2)

        np.testing.assert_allclose(t_observed_vertex, t_expected_vertex)
        np.testing.assert_allclose(d_observed_vertex, spatial_pattern)

        ## MBM (get betas of the t-maps)
        t_observed_mode = solver.decompose(t_observed_vertex, mode_counts=n_modes)
        d_observed_mode = (emodes @ t_observed_mode[:, np.newaxis]).flatten() / np.sqrt(n_subjects / 2)

        np.testing.assert_allclose(t_observed_mode, t_expected_mode, atol=1e-12)
        np.testing.assert_allclose(d_observed_mode, spatial_pattern)

class TestStatMapsRandom(): 
    @pytest.fixture(scope="class")
    def glm_data(self):
        """Generates a reproducible 2-predictor GLM setup with a clear signal."""
        rng = np.random.default_rng(42)
        n_subjects = 50
        n_data = 30

        # Continuous target predictor + continuous nuisance predictor
        x_target = rng.standard_normal((n_subjects, 1))
        x_nuisance = rng.standard_normal((n_subjects, 1))
        design = np.hstack([x_target, x_nuisance])

        # Response with strong effect on predictor 0
        signal = x_target * 2.5
        noise = rng.standard_normal((n_subjects, n_data))
        response = signal + noise

        contrast_1row = np.array([[1.0, 0.0]])
        contrast_2row = np.array([[1.0, 0.0], [0.0, 1.0]])

        return {
            "design": design,
            "response": response,
            "contrast_1row": contrast_1row,
            "contrast_2row": contrast_2row,
            "n_subjects": n_subjects,
            "n_data": n_data,
        }

    @pytest.mark.parametrize("statistic", ["t", "z", "f"])
    def test_empirical_shapes_and_types(self, glm_data, statistic):
        stat_map = compute_stat_map(
            glm_data["design"],
            glm_data["response"],
            glm_data["contrast_1row"],
            statistic,
        )

        assert isinstance(stat_map, np.ndarray)
        assert np.issubdtype(stat_map.dtype, np.floating)
        assert stat_map.shape == (glm_data["n_data"],)
        assert not np.isnan(stat_map).any()

    @pytest.mark.parametrize("statistic", ["t", "z", "f"])
    @pytest.mark.parametrize("method", ["flip", "shuffle", "flip_shuffle"])
    @pytest.mark.parametrize("n_permutations", [5, 10])
    def test_nulls_1row_shapes_and_types(self, glm_data, statistic, method, n_permutations):
        null_maps = generate_null_stat_maps(
            glm_data["design"],
            glm_data["response"],
            glm_data["contrast_1row"],
            statistic=statistic,
            method=method,
            n_permutations=n_permutations,
            seed=42,
        )

        assert isinstance(null_maps, np.ndarray)
        assert np.issubdtype(null_maps.dtype, np.floating)
        assert null_maps.shape == (glm_data["n_data"], n_permutations)
        assert not np.isnan(null_maps).any()

    @pytest.mark.parametrize("statistic", ["f"]) # multirow only for F-test
    @pytest.mark.parametrize("method", ["flip", "shuffle", "flip_shuffle"])
    @pytest.mark.parametrize("n_permutations", [5, 10])
    def test_nulls_2row_shapes_and_types(self, glm_data, statistic, method, n_permutations):
        null_maps = generate_null_stat_maps(
            glm_data["design"],
            glm_data["response"],
            glm_data["contrast_2row"],
            statistic=statistic,
            method=method,
            n_permutations=n_permutations,
            seed=42,
        )

        assert isinstance(null_maps, np.ndarray)
        assert np.issubdtype(null_maps.dtype, np.floating)
        assert null_maps.shape == (glm_data["n_data"], n_permutations)
        assert not np.isnan(null_maps).any()


    @pytest.mark.parametrize("statistic", ["t", "z", "f"])
    @pytest.mark.parametrize("method", ["flip", "shuffle", "flip_shuffle"])
    @pytest.mark.parametrize("n_permutations", [1000]) # has to be large
    @pytest.mark.parametrize("seed", range(3)) # has to be a fixed set which is known to be within error bounds
    def test_permutation_p_values_recover_signal(self, glm_data, statistic, method, n_permutations, seed):
        """Verifies that permutation tests yield significant p-values for true signals
        and non-significant p-values for noise/nuisance.
        """
        design = glm_data["design"]
        response = glm_data["response"]

        # --- 1. Test True Signal (Predictor 0) ---
        contrast_signal = np.array([[1.0, 0.0]])
        obs_stat_signal = compute_stat_map(
            design, 
            response, 
            contrast_signal, 
            statistic=statistic
        )
        null_stats_signal = generate_null_stat_maps(
            design,
            response,
            contrast_signal,
            statistic=statistic,
            method=method,
            n_permutations=n_permutations,
            seed=seed,
        )

        # The mean of the null t-statistics across permutations 
        # should be close to 0 for t and z (F-test is not centred at 0)
        if statistic in ["t", "z"]:
            null_means = np.mean(null_stats_signal, axis=1)
            np.testing.assert_allclose(null_means, 0.0, atol=0.25)
            
        # Empirical p-values: proportion of null stats >= observed stat
        # Signal was strong (2.5), so p-values should be near minimal (1 / (n_perm + 1))
        p_vals_signal = np.mean(
            null_stats_signal >= obs_stat_signal[:, None], axis=1
        )
        assert np.all(p_vals_signal < 0.05)


    @pytest.mark.parametrize("statistic", ["t", "z", "f"])
    @pytest.mark.parametrize("method", ["flip", "shuffle", "flip_shuffle"])
    @pytest.mark.parametrize("n_permutations", [100])
    @pytest.mark.parametrize("seed", range(3)) # has to be a fixed set which is known to be within error bounds
    def test_permutation_p_values_recover_nuisance(self, glm_data, statistic, method, n_permutations, seed):
        """Verifies that permutation tests yield significant p-values for true signals
        and non-significant p-values for noise/nuisance.
        """
        design = glm_data["design"]
        response = glm_data["response"]
        # --- 2. Test Pure Nuisance (Predictor 1, no injected effect) ---
        contrast_nuisance = np.array([[0.0, 1.0]])
        obs_stat_nuisance = compute_stat_map(
            design, 
            response, 
            contrast_nuisance,  
            statistic=statistic
        )
        null_stats_nuisance = generate_null_stat_maps(
            design,
            response,
            contrast_nuisance,
            statistic=statistic,
            method=method,
            n_permutations=n_permutations,
            seed=seed,
        )

        p_vals_nuisance = np.mean(
            null_stats_nuisance >= obs_stat_nuisance[:, None], axis=1
        )

        # Nuisance predictor has no signal -> p-values should follow ~Uniform(0, 1)
        assert np.mean(p_vals_nuisance) > 0.25
        assert np.mean(kstest(p_vals_nuisance, 'uniform').pvalue) > 0.1  # Kolmogorov-Smirnov test for uniformity

class TestFitAndAD:
    """
    Test suite for _fit_and_ad_exceedances. 
    Covers basic input handling, exact quantile recovery, and random sample behavior.
    """

    @pytest.fixture
    def basic_exceedances(self):
        # Scale basic data to keep estimated shape parameter k within [-0.9, 0.5]
        return genpareto.ppf((np.arange(10)+0.5) / 10, c=0.1)

    def test_output_shape_and_gof_none(self, basic_exceedances):
        fit, adp = _fit_and_ad_exceedances(basic_exceedances, gof_acceptance_level=None)
        assert len(fit) == 3
        assert fit[1] == 0.0  # floc is strictly fixed to 0.0
        assert np.isnan(adp)

    def test_output_shape_and_gof_active(self, basic_exceedances):
        fit, adp = _fit_and_ad_exceedances(basic_exceedances, gof_acceptance_level=0.05)
        assert len(fit) == 3
        assert isinstance(adp, float)
        assert 0.0 <= adp <= 1.0

    def test_nan_and_inf_handling(self):
        # SciPy's fit or the AD calculation should raise on invalid numeric inputs
        bad_data = np.array([0.1, 0.5, np.nan, np.inf])
        with pytest.raises(ValueError):
            _fit_and_ad_exceedances(bad_data, gof_acceptance_level=0.05)

    @pytest.mark.parametrize("c", [-0.45, 0, 0.85])
    @pytest.mark.parametrize("scale", [1.0])
    @pytest.mark.parametrize("n_samples", [1000, 10000])
    def test_exact_distribution_fit(self, c, scale, n_samples):
        # Generate exact sample grid using ppf to ensure perfect fit
        p_grid = (np.arange(n_samples)+0.5) / n_samples
        exact_data = genpareto.ppf(p_grid, c, loc=0.0, scale=scale)
        
        fit, adp = _fit_and_ad_exceedances(exact_data, gof_acceptance_level=0.05)
        fit_c, fit_loc, fit_scale = fit
        
        assert fit_loc == 0.0
        np.testing.assert_allclose(fit_c, c, atol=1e-2)
        np.testing.assert_allclose(fit_scale, scale, atol=1e-2)
        
        # A near-perfect theoretical fit should yield a high Anderson-Darling p-value
        assert 0.99 <= adp <= 1.0

    @pytest.mark.parametrize("c", [-0.45, 0, 0.8])
    @pytest.mark.parametrize("scale", [1.0])
    @pytest.mark.parametrize("n_samples", [10000])
    @pytest.mark.parametrize("n_simulations", [20])
    @pytest.mark.parametrize("seed", [42])  
    def test_random_gpd_fit(self, c, scale, n_samples, n_simulations, seed):

        # If data do indded come from GPD, p-values should be uniformly distributed
        p_values = np.empty(n_simulations)
        for ii in range(n_simulations): 
            random_data = np.sort(
                genpareto.rvs(c, loc=0.0, scale=scale, size=n_samples, random_state=seed+ii)
            )
        
            fit, p_values[ii] = _fit_and_ad_exceedances(random_data, gof_acceptance_level=0.05)
            fit_c, fit_loc, fit_scale = fit
            
            assert fit_loc == 0.0
            np.testing.assert_allclose(fit_c, c, atol=0.2)
            np.testing.assert_allclose(fit_scale, scale, atol=0.2)

        assert np.mean(p_values < 0.05) < 0.20, "GPD model failed to accept true GPD data"

        _, ks_p = kstest(p_values, 'uniform', args=(0, 1))
        assert ks_p > 0.01, f"P-values fail uniform test (KS p-val = {ks_p})"


    @pytest.mark.parametrize("s", [0.1, 0.8])
    @pytest.mark.parametrize("scale", [1.0])
    @pytest.mark.parametrize("n_samples", [1000])
    @pytest.mark.parametrize("n_simulations", [20])
    @pytest.mark.parametrize("seed", [42])
    @pytest.mark.filterwarnings("ignore:.*table bounds.*:UserWarning")
    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    def test_random_non_gpd_fit_rejection(self, s, scale, n_samples, n_simulations, seed):
        """
        If data come from a non-GPD distribution (e.g., Lognormal), the goodness-of-fit 
        test should reject H0, yielding p-values clustered near 0.0 rather than uniform.
        """
        p_values = np.empty(n_simulations)
        for ii in range(n_simulations):
            # Generate lognormal data (heavy-tailed, but not GPD)
            random_data = np.sort(
                lognorm.rvs(s=s, loc=0.0, scale=scale, size=n_samples, random_state=seed+ii)
            )
            _, p_values[ii] = _fit_and_ad_exceedances(random_data, gof_acceptance_level=0.05)

        # 1. Most individual p-values should directly fail the GOF threshold (< 0.05)
        assert np.mean(p_values < 0.05) > 0.80, "GPD model failed to reject non-GPD data"

        # 2. The distribution of p-values must fail the Kolmogorov-Smirnov test for uniformity
        _, ks_p = kstest(p_values[~np.isnan(p_values)], 'uniform', args=(0, 1))
        print(p_values)
        assert ks_p < 0.01, f"Expected non-uniform p-values, but KS test passed (KS p-val = {ks_p})"

class TestPVals(): 
    @pytest.mark.parametrize("n_data", [10, 50])
    @pytest.mark.parametrize("n_permutations", [100, 1000])
    def test_no_gpd_1d(self, n_data, n_permutations):
        """Test size/shape and simple calculation, 1 observation per row."""
        observed = np.random.randn(n_data)
        nulls = np.random.randn(n_data, n_permutations)

        p_geq = np.mean(nulls >= observed[:, None], axis=1)
        p_leq = np.mean(nulls <= observed[:, None], axis=1)
        p_two = 2.0 * np.minimum(p_geq, p_leq)

        # Test the different alternatives
        tests = {
            'greater': p_geq,
            'less': p_leq,
            'two-sided': p_two
        }
        
        for alternative, expected in tests.items():
            p_act, tail = mbm_calc_p_vals(observed, nulls, alternative,   # type: ignore
                                   n_for_pval_recalculation=None, # turn off fitting
                                   return_tail=True
                                   )
            
            np.testing.assert_array_equal(expected, p_act)
            np.testing.assert_array_equal(tail.shape, observed.shape)

    @pytest.mark.parametrize("n_data", [10, 50])
    @pytest.mark.parametrize("n_observations", [1, 10])
    @pytest.mark.parametrize("n_permutations", [100, 1000])
    def test_no_gpd_2d(self, n_data, n_observations, n_permutations):
        """Test size/shape and simple calculation, multiple observations per row."""
        observed = np.random.randn(n_data, n_observations)
        nulls = np.random.randn(n_data, n_permutations)

        p_geq = np.mean(nulls[:,:,None] >= observed[:, None, :], axis=1)
        p_leq = np.mean(nulls[:,:,None] <= observed[:, None, :], axis=1)
        # p_two = 2.0 * np.minimum(p_geq, p_leq)

        # Test the different alternatives
        tests = {
            'greater': p_geq,
            'less': p_leq,
            # 'two-sided': p_two
        }

        for alternative, expected in tests.items():
            p_act, tail = mbm_calc_p_vals(observed, nulls, alternative,   # type: ignore
                                   n_for_pval_recalculation=None, # turn off fitting
                                   return_tail=True
                                   )

            np.testing.assert_array_equal(expected, p_act)
            np.testing.assert_array_equal(tail.shape, observed.shape)

    @pytest.mark.parametrize("c", [0.1, 0.5, 0.85])
    @pytest.mark.parametrize("scale", [1.0, 2.0])
    @pytest.mark.parametrize("loc", [0.0, 1.0])
    @pytest.mark.parametrize("n_permutations", [1000, 10000])
    @pytest.mark.parametrize("n_for_tail_estimation", [range(1000, 1, -10)])
    def test_gpd_fit_near_exact(self, c, scale, loc, n_permutations, n_for_tail_estimation):
        """Use exact GPD distribution to generate tails and verify the extrapolation."""
        
        # 1. Uniformly spaced quantile midpoints over N=n_permutations
        # Midpoints ensure no point maps to exact CDF boundary 0.0 or 1.0
        p_grid = (np.arange(n_permutations)+0.5) / (n_permutations)
        p_target = np.array([[0.25, 0.20, 0.15, 0.10, 0.05]]) / (n_permutations)
    
        # 2. Generate exact sample grid
        statMapNull = np.reshape(genpareto.ppf(p_grid, c, loc=loc, scale=scale), (1, -1))
        observedStatMap = genpareto.ppf(1 - p_target, c, loc=loc, scale=scale)
    
        # 3. Check naive empirical p-value
        p_naive = mbm_calc_p_vals(
            observedStatMap=observedStatMap,
            statMapNull=statMapNull,
            alternative="greater",
            n_for_pval_recalculation=None
        )
        assert np.all(p_naive == 0.0)
    
        # 4. Call function with correct range direction
        p_fit = mbm_calc_p_vals(
            observedStatMap=observedStatMap,
            statMapNull=statMapNull,
            alternative="greater",
            n_for_pval_recalculation=10,
            n_for_tail_estimation=n_for_tail_estimation,
            gof_acceptance_level=0.9, # could even be 1 as the data are from the exact distribution
            return_tail=False,
        )
    
        # 5. Verify accuracy against target_p
        np.testing.assert_allclose(
            p_fit,
            p_target,
            rtol=1e-1,
            err_msg=f"Expected interpolated p-value to be close to {p_target}, got {p_fit}",
        )

    # Test fitting procedure but not GOF results (set to 0 here)
    @pytest.mark.parametrize("n_null", [10000])
    @pytest.mark.parametrize("n_tail", [1000]) # needs to always be less than n_null
    @pytest.mark.parametrize("c", [-0.45, 0, 0.85])
    @pytest.mark.parametrize("loc", [12.0])
    @pytest.mark.parametrize("scale", [2.0])
    @pytest.mark.parametrize("p", [np.array([[1e-4, 1e-5, 1e-6]])])
    @pytest.mark.parametrize("alternative, factor", [("greater", 1.0), ("less", -1.0)])
    @pytest.mark.parametrize("seed", [42])
    def test_all_gpd_one_sided(self, n_null, n_tail, c, loc, scale, p, alternative, factor, seed):
        """Test that the fitting finds a more accurate p-value. Note that the GOF thresholding is not tested here."""
        obs = factor * genpareto.ppf(1-p, c, loc=loc, scale=scale)
        nulls = factor *  np.sort(genpareto.rvs(c, loc=loc, scale=scale, size=(1,n_null), random_state=seed))

        floc = np.mean(factor * nulls[:,-n_tail-1:-n_tail+1])
        fit = genpareto.fit(factor * nulls[:,-n_tail:] - floc, floc = 0.0)
        p_exp = genpareto.sf(factor * obs - floc, *fit) * n_tail/n_null

        p_test = mbm_calc_p_vals(obs, 
                             nulls,
                             alternative=alternative, 
                             n_for_pval_recalculation=nulls.size, 
                             n_for_tail_estimation=n_tail,
                             gof_acceptance_level=None,
                             )

        p_naive = mbm_calc_p_vals(obs, 
                             nulls,
                             alternative=alternative, 
                             n_for_pval_recalculation=None, # no recalculation
                             )
     
        print(f"p: {p}, p_exp: {p_exp}, p_test: {p_test}, p_naive: {p_naive}")
        np.testing.assert_allclose(p_test, p, rtol=1, atol=0) # has to be pretty loose here
        np.testing.assert_allclose(p_test, p_exp, rtol=1e-6, atol=1e-6)
        assert np.all((p_naive!=p_test) | (p_test==0)), "P-values with and without fitting should be different."

    @pytest.mark.parametrize("n_tail", [50, 100])
    @pytest.mark.parametrize("p", [np.array([[1e-4, 1e-5, 1e-6]])])
    @pytest.mark.parametrize("seed", range(10)) # has to be a fixed set which is known to be within error bounds (false negs of AD test)
    def test_gpd_tail_fallback_selection(self, n_tail, p,seed):
        """Verify that a large invalid tail fails GoF and falls back to a valid smaller tail."""
        
        # Bulk data (Normal) + Extreme Tail (GPD), 99% of data is bulk
        bulk = 3.0 - np.random.default_rng(seed).exponential(scale=1.0, size=(1, n_tail * 99))
        tail = genpareto.rvs(c=0.2, loc=3.0, scale=1.0, size=(1, n_tail), random_state=seed)
        nulls = np.sort(np.hstack([bulk, tail]), axis=1)

        obs = genpareto.ppf(1-p, c=0.2, loc=3.0, scale=1.0)

        # Call 1
        p_test = mbm_calc_p_vals(
            obs, nulls,
            alternative='greater',
            n_for_tail_estimation=(10*n_tail, n_tail),
            n_for_pval_recalculation=nulls.size, # force recalculation
            gof_acceptance_level=0.05 # check that first is rejected
        )

        # Call 2
        p_good = mbm_calc_p_vals(
            obs, nulls,
            alternative='greater',
            n_for_tail_estimation=(n_tail,),
            n_for_pval_recalculation=nulls.size,
            gof_acceptance_level=0.05
        )

        # Call 3
        p_bad = mbm_calc_p_vals(
            obs, nulls,
            alternative='greater',
            n_for_tail_estimation=(10*n_tail,),
            n_for_pval_recalculation=nulls.size,
            gof_acceptance_level=None # force this to be accepted here so that we can get a value
        )

        p_naive = mbm_calc_p_vals(
            obs, nulls,
            alternative='greater',
            n_for_pval_recalculation=None, # no recalculation
        )

        print(f"p: {p}, p_test: {p_test}, p_good: {p_good}, p_bad: {p_bad}, p_naive: {p_naive}")

        # If fallback succeeded
        np.testing.assert_allclose(p_test, p_good, rtol=1e-6, atol=1e-6)
        
        # Confirm that nte=1000 yields a different p-value than nte=10
        assert not np.allclose(p_test, p_bad, rtol=1e-3, atol=0.0) or np.allclose(p_good, 0.0, atol=1e-6) 
        assert not np.allclose(p_test, p_naive, rtol=1e-3, atol=0.0) or np.allclose(p_good, 0.0, atol=1e-6)

