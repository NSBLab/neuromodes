import csv
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from lapy import TriaMesh
from lapy.shapedna import normalize_ev
import pytest

from neuromodes.eigen import (EigenSolver, is_orthonormal_basis, scale_hetero, get_eigengroup_inds, 
                              convert_to_eigenvalue, convert_from_eigenvalue, mode_to_group, group_to_mode, 
                              estimate_fwhm, truncate_emodes)
from neuromodes.io import fetch_surf, fetch_map
from neuromodes.mesh import mask_mesh

@pytest.fixture(scope="module")
def surf_medmask_hetero():
    mesh, medmask = fetch_surf(density='4k')
    hetero = fetch_map(data="myelinmap", density="4k")
    return mesh, medmask, hetero

def test_init_params(surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    _ = EigenSolver(surf, mask=medmask, hetero=hetero, alpha=0.5,
                         scaling='exponential')
    
def test_premasked_surf(surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    masked_surf = mask_mesh(surf, medmask)
    _ = EigenSolver(masked_surf, hetero=hetero[medmask])

def test_no_medmask(surf_medmask_hetero):
    surf, _, hetero = surf_medmask_hetero
    EigenSolver(surf, hetero=hetero)

def test_invalid_mask_shape(surf_medmask_hetero):
    surf, _, _ = surf_medmask_hetero
    bad_mask = np.ones(10, dtype=bool)
    with pytest.raises(ValueError, match=r"mask must have shape \(4002,\)"):
        EigenSolver(surf, mask=bad_mask)

def test_no_hetero(surf_medmask_hetero):
    surf, medmask, _ = surf_medmask_hetero
    homogenous_solver = EigenSolver(surf, mask=medmask, hetero=None)
    homogenous_solver.solve(10) # hardcoded to 10 to match the saved prior_modes

    # Load homogeneous eigenmodes/eigenvalues for comparison
    test_data = Path(__file__).parent / 'test_data'
    prior_modes = np.load(test_data / 'sp-human_tpl-fsLR_den-4k_hemi-L_midthickness-emodes.npy')
    prior_evals = np.load(test_data / 'sp-human_tpl-fsLR_den-4k_hemi-L_midthickness-evals.npy')

    for i in range(1, homogenous_solver.n_modes):
        assert np.abs(np.corrcoef(homogenous_solver.emodes[:, i], prior_modes[:, i])[0, 1]) > 0.99, \
            f'Eigenmode {i} does not match the previously computed homogeneous result.'
        assert np.allclose(homogenous_solver.evals[i], prior_evals[i], rtol=0.1), \
            f'Eigenvalue {i} does not match the previously computed homogeneous result.'
        
def test_no_hetero_alpha_scaling(surf_medmask_hetero):
    surf, medmask, _ = surf_medmask_hetero
    with pytest.warns(UserWarning, match="alpha is ignored"):
        EigenSolver(surf, mask=medmask, hetero=None, alpha=0.5)
    with pytest.warns(UserWarning, match="scaling is ignored"):
        EigenSolver(surf, mask=medmask, hetero=None, scaling='exponential')

def test_invalid_hetero_shape(surf_medmask_hetero):
    surf, _, _ = surf_medmask_hetero
    bad_hetero = np.ones(10)
    with pytest.raises(ValueError, match=r"vertices in the provided mesh \(4002\)."):
        EigenSolver(surf, hetero=bad_hetero)

def test_nan_inf_hetero(surf_medmask_hetero):
    surf, _, hetero = surf_medmask_hetero
    bad_hetero = hetero.copy()
    bad_hetero[0] = np.nan
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        EigenSolver(surf, hetero=bad_hetero)

    bad_hetero[0] = np.inf
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        EigenSolver(surf, hetero=bad_hetero)

def test_constant_hetero(surf_medmask_hetero):
    surf = surf_medmask_hetero[0]

    hetero = np.ones(surf.v.shape[0])

    with pytest.warns(UserWarning, match="Provided hetero is constant"):
        EigenSolver(surf, hetero=hetero)

def test_nan_inf_hetero_medmask(surf_medmask_hetero):
    # Inject NaN/Inf at a cortical vertex (should raise error)
    surf, medmask, hetero = surf_medmask_hetero
    cortical_vertex = np.where(medmask)[0][0]
    bad_hetero = hetero.copy()
    bad_hetero[cortical_vertex] = np.nan
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        EigenSolver(surf, hetero=bad_hetero)
    bad_hetero[cortical_vertex] = np.inf
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        EigenSolver(surf, hetero=bad_hetero)

def test_nan_inf_hetero_medmask_ignored(surf_medmask_hetero):
    # Inject NaN/Inf at a medial vertex (should be ignored)
    surf, medmask, hetero = surf_medmask_hetero
    medial_vertex = np.where(~medmask)[0][0]
    print(medial_vertex)
    bad_hetero = hetero.copy()
    bad_hetero[medial_vertex] = np.nan
    EigenSolver(surf, mask=medmask, hetero=bad_hetero)
    bad_hetero[medial_vertex] = np.inf  
    EigenSolver(surf, mask=medmask, hetero=bad_hetero)

def test_hetero_ones(surf_medmask_hetero):
    surf, medmask, _ = surf_medmask_hetero
    hetero = np.ones(sum(medmask))

    homo_solver = EigenSolver(surf, mask=medmask).solve(20, seed=0)
    with pytest.warns(UserWarning, match="Provided hetero is constant"):
        het_solver = EigenSolver(surf, mask=medmask, hetero=hetero).solve(20, seed=0)

    assert np.allclose(het_solver.evals, homo_solver.evals), \
        'Eigenvalues with hetero=ones do not match homogeneous eigenvalues.'
    for i in range(homo_solver.n_modes):
        assert np.allclose(het_solver.emodes[:, i], homo_solver.emodes[:, i], atol=1e-4), \
            f'Eigenmode {i+1} with hetero=ones does not match its homogeneous equivalent.'

def test_real_heteromaps(surf_medmask_hetero):
    mesh, medmask = fetch_surf() # 32k density to match real maps
    for map in ['fcgradient1', 'myelinmap', 'ndi', 'odi', 'thickness']:
        hetero = fetch_map(map)
        EigenSolver(mesh, mask=medmask, hetero=hetero) # just test that it initializes without error

@pytest.fixture(scope="module")
def presolver(surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    presolver = EigenSolver(surf, mask=medmask, hetero=hetero)
    presolver.compute_lbo()
    return presolver

def test_symmetric_mass(presolver):
    diff = presolver.mass - presolver.mass.transpose()
    assert abs(diff).max() == 0, 'Mass matrix is not symmetric.'

def test_symmetric_stiffness(presolver):
    diff = presolver.stiffness - presolver.stiffness.transpose()
    assert abs(diff).max() == 0, 'Stiffness matrix is not symmetric.'

def test_stiffness_rowsums(presolver):
    assert abs(presolver.stiffness.sum(axis=1)).max() < 2e-6
        
def test_seeded_modes(presolver):
    presolver.solve(16, standardize=False, fix_mode1=False, seed=36)
    emodes1 = presolver.emodes
    evals1 = presolver.evals

    presolver.solve(presolver.n_modes, standardize=False, fix_mode1=False, seed=36)
    emodes2 = presolver.emodes
    evals2 = presolver.evals

    assert (emodes1 == emodes2).all(), 'Modes from same seed are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed are not identical.'

    presolver.solve(presolver.n_modes, standardize=False, fix_mode1=False, seed=37)
    emodes3 = presolver.emodes
    evals3 = presolver.evals

    assert not (emodes1 == emodes3).all(), 'Modes from different seeds should not be identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seeds should not be identical.'

def test_generator_seeded_modes(presolver):
    rng = np.random.default_rng(0)
    presolver.solve(16, standardize=False, fix_mode1=False, seed=rng)
    emodes1 = presolver.emodes
    evals1 = presolver.evals

    # Reset the generator to ensure the same sequence of random numbers
    rng = np.random.default_rng(0)
    presolver.solve(presolver.n_modes, standardize=False, fix_mode1=False, seed=rng)
    emodes2 = presolver.emodes
    evals2 = presolver.evals
    assert (emodes1 == emodes2).all(), 'Modes from same seed generator are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed generator are not identical.'

    rng = np.random.default_rng(1)
    presolver.solve(presolver.n_modes, standardize=False, fix_mode1=False, seed=rng)
    emodes3 = presolver.emodes
    evals3 = presolver.evals
    assert not (emodes1 == emodes3).all(), 'Modes from different seed generators are identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seed generators are identical.'

def test_vector_seeded_modes(presolver):
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(size=presolver.n_verts)
    presolver.solve(16, standardize=False, fix_mode1=False, v0=v0)
    emodes1 = presolver.emodes
    evals1 = presolver.evals

    # Reuse the same seed vector
    presolver.solve(presolver.n_modes, standardize=False, fix_mode1=False, v0=v0)
    emodes2 = presolver.emodes
    evals2 = presolver.evals

    assert (emodes1 == emodes2).all(), 'Modes from same seed vector are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed vector are not identical.'

    rng = np.random.default_rng(1)
    v0_diff = rng.standard_normal(size=presolver.n_verts)

    presolver.solve(presolver.n_modes, standardize=False, fix_mode1=False, v0=v0_diff)
    emodes3 = presolver.emodes
    evals3 = presolver.evals

    assert not (emodes1 == emodes3).all(), 'Modes from different seed vectors are identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seed vectors are identical.'

def test_invalid_vector_seed(presolver):
    with pytest.raises(ValueError,
                       match=r"v0 must have shape \(n_verts,\) = \(3619,\)."):
        presolver.solve(16, v0=np.ones(10))

@pytest.fixture(scope="module")
def solver(presolver):
    presolver.solve(n_modes=16, seed=0)
    return presolver

def test_nonstandard_modes(solver, surf_medmask_hetero):
    emodes = solver.emodes
    surf, medmask, hetero = surf_medmask_hetero
    solver_nonstd = EigenSolver(surf, mask=medmask, hetero=hetero)
    emodes_nonstd = solver_nonstd.solve(solver.n_modes, standardize=False, seed=0).emodes
    
    assert not np.all(emodes_nonstd[0, :] >= 0), \
        'Non-standardized first vertex should have both positive and negative values.'
    assert np.all(emodes[0, :] >= 0), 'Standardized first vertex has negative values.'
    assert (abs(emodes) == abs(emodes_nonstd)).all(), \
    'Non-standardized modes do not match standardized modes.'

def test_solve_lumped_mass(solver, surf_medmask_hetero):
    emodes = solver.emodes
    surf, medmask, hetero = surf_medmask_hetero

    # Get modes after solving with lumped mass matrix
    solver_lump = EigenSolver(surf, mask=medmask, hetero=hetero).solve(emodes.shape[1], lump=True)
    emodes_lumped = solver_lump.emodes

    for i in range(1, solver_lump.n_modes):
        mse = np.mean((emodes[:, i] - emodes_lumped[:, i])**2)
        assert mse < 1e-4, f'Mode {i} has MSE {mse:.2e} between lumped and consistent mass ' \
            'solutions, which is above the threshold of 1e-4.'

def test_solutions(solver):
    emodes = solver.emodes
    evals = solver.evals

    assert emodes.shape == (solver.n_verts,
                           solver.n_modes), (f'Eigenmodes have shape {emodes.shape}, should be '
                                            f'{(solver.n_verts, solver.n_modes)}.')
    assert len(evals) == solver.n_modes, (f'Eigenvalues has length {len(evals)}, should be '
                                         f'{solver.n_modes}.')
    assert np.all(np.diff(evals) > 0), 'Eigenvalues are not sorted in descending order.'

def test_n_modes_consistency(solver, surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero

    # Solve for more modes and check that the first 16 modes are approximately the same
    # TODO: may as well use 100 modes in the fixture and instead solve for fewer here?
    solver_more_modes = EigenSolver(surf, mask=medmask, hetero=hetero).solve(100, seed=0)
    assert np.allclose(solver.emodes, solver_more_modes.emodes[:, :16], atol=1e-4), \
        'Modes differ when solving for different n_modes.'
    
def test_normalized_surf(surf_medmask_hetero, solver):
    surf, medmask, hetero = surf_medmask_hetero

    # Use LaPy to normalize evals
    evals_lapy = normalize_ev(solver.geometry, solver.evals)

    # Normalize mesh within EigenSolver
    solver = EigenSolver(surf, mask=medmask, hetero=hetero, normalize=True).solve(16, seed=0)

    # Check that evals match between the two normalization approaches
    assert np.allclose(evals_lapy, solver.evals, atol=1e-20), \
    'Evals from LaPy normalization do not match evals from EigenSolver normalization.'

def test_constant_mode1(solver, surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    emode1 = solver.emodes[:, 0]

    solver_unfixed = EigenSolver(surf, mask=medmask, hetero=hetero).solve(2, fix_mode1=False)
    emode1_unfixed = solver_unfixed.emodes[:, 0]
    eval1_unfixed = solver_unfixed.evals[0]

    assert (emode1 == emode1[0]).all(), 'Fixed first mode is not exactly constant.'
    assert np.allclose(emode1_unfixed, emode1[0],
                       atol=1e-4), 'Unfixed first mode is not approximately constant.'
    assert np.isclose(np.mean(emode1_unfixed), emode1[0],
                      atol=1e-6), 'Mean of unfixed first mode is not close to fixed value.'
    assert eval1_unfixed < 1e-6, 'First eigenvalue of unfixed first mode is not close to 0.'

def test_positive_sigma(solver, surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    emodes = solver.emodes
    evals = solver.evals

    with pytest.warns(UserWarning, match="Mode 1 will not be fixed"):
        solver_pos_sigma = EigenSolver(surf, mask=medmask, hetero=hetero)
        solver_pos_sigma.solve(solver.n_modes, sigma=1e-4, seed=0)
    emodes_pos_sigma = solver_pos_sigma.emodes
    evals_pos_sigma = solver_pos_sigma.evals

    assert np.allclose(evals, evals_pos_sigma, atol=1e-4), \
        'Eigenvalues differ with positive sigma.'
    assert np.allclose(emodes, emodes_pos_sigma, atol=1e-4), \
        'Eigenmodes differ with positive sigma.'

def test_check_orthonorm(solver):
    emodes = solver.emodes

    # Check that modes are not orthonormal in Euclidean space
    assert not is_orthonormal_basis(emodes)

    emodes[:, 0] += 0.1 # Destroy mass-orthonormality by changing first mode's value

    assert not is_orthonormal_basis(emodes, solver.mass)

def test_check_euclidean_orthonorm():
    # Create orthonormal vectors in Euclidean space
    vecs = np.eye(5)[:, [2, 0, 4, 1]]

    assert is_orthonormal_basis(vecs)
    assert is_orthonormal_basis(vecs, mass=np.eye(5)) # type: ignore
    assert not is_orthonormal_basis(vecs, mass=np.ones((5, 5))) # type: ignore

def test_scale_hetero(surf_medmask_hetero):
    _, _, hetero = surf_medmask_hetero

    # Check that sigmoid-scaled hetero is within (0, 2)
    hetero_sig = scale_hetero(hetero)
    assert np.all((hetero_sig > 0) & (hetero_sig < 2))

    # Check that exponential-scaled hetero is all positive
    hetero_exp = scale_hetero(hetero, scaling='exponential')
    assert np.all(hetero_exp > 0)

def test_invalid_scale_hetero(surf_medmask_hetero):
    _, _, hetero = surf_medmask_hetero

    with pytest.raises(ValueError, match="Invalid scaling 'plantasia'"):
        scale_hetero(hetero, scaling='plantasia')

def test_get_eigengroup_inds(solver):
    # Test that function returns correct groups for 8 modes
    groups = get_eigengroup_inds(8)
    expected_groups = [np.array([0]), np.array([1, 2, 3]), np.array([4, 5, 6, 7])]
    for g, expected in zip(groups, expected_groups):
        assert np.array_equal(g, expected), f'Expected group {expected}, got {g}.'

    # Check on solver
    groups = get_eigengroup_inds(solver.n_modes)
    last_mode = groups[-1][-1] + 1
    assert solver.emodes[:, :last_mode].shape == solver.emodes.shape, \
        'Last eigengroup indices do not match number of modes.'
    
class TestEigenvalueConversion:
    # Tests with manually specified values
    def test_convert_to_eigenvalue_exact_values(self):
        """Tests that the mathematical formulas are applied correctly."""
        area = 4 * np.pi # sphere of radius 1
        assert_allclose(convert_to_eigenvalue(2.0, 'eigenvalue'),       2.0)
        assert_allclose(convert_to_eigenvalue(2.0, 'wavelength'),       np.pi**2)
        assert_allclose(convert_to_eigenvalue(2.0, 'fwhm'),             2 * np.log(2))
        assert_allclose(convert_to_eigenvalue(2.0, 'group', area=area), 6.0)
        assert_allclose(convert_to_eigenvalue(2.0, 'mode', area=area),  2.0)

    def test_convert_from_eigenvalue_exact_values(self):
        """Tests the inverse mathematical formulas."""
        area = 4 * np.pi # sphere of radius 1
        assert_allclose(convert_from_eigenvalue(2.0,       'eigenvalue'), 2.0)
        assert_allclose(convert_from_eigenvalue(np.pi**2,  'wavelength'), 2.0)
        assert_allclose(convert_from_eigenvalue(2 * np.log(2),   'fwhm'), 2.0)
        assert_allclose(convert_from_eigenvalue(6.0, 'group', area=area), 2.0)
        assert_allclose(convert_from_eigenvalue(2.0,  'mode', area=area), 2.0)

    # Test inversions
    @pytest.mark.parametrize('ctype', ['wavelength', 'fwhm', 'group', 'mode'])
    def test_inversions(self, ctype):
        """Tests that converting to and from an eigenvalue returns the original array."""
        area = 100.0
        values1 = np.array([0.5, 1.0, 2.5, 10.0, 100.0])
        evals = convert_to_eigenvalue(values1, input=ctype, area=area)
        values2 = convert_from_eigenvalue(evals, output=ctype, area=area)
        assert_allclose(values1, values2)

    # Exception and validation Tests
    @pytest.mark.parametrize('direction', [convert_to_eigenvalue, convert_from_eigenvalue])
    @pytest.mark.parametrize('ctype', ['group', 'mode'])
    def test_missing_area_errors(self, direction, ctype):
        """Tests that missing area arguments trigger ValueErrors."""
        with pytest.raises(ValueError, match="Area must be provided"):
            direction(5.0, ctype) 

    @pytest.mark.parametrize('direction', [convert_to_eigenvalue, convert_from_eigenvalue])
    def test_invalid_type_errors(self, direction):
        """Tests that unrecognized string types trigger ValueErrors."""
        with pytest.raises(ValueError, match=r"Incorrect .*put specified"):
            direction(5.0, 'frequency')

    # Zero and inf
    @pytest.mark.parametrize('direction', [convert_to_eigenvalue, convert_from_eigenvalue])
    @pytest.mark.parametrize('ctype', ['wavelength', 'fwhm'])
    def test_zero_division_handling(self, direction, ctype):
        """Tests that the context managers successfully handle division by zero."""
        with pytest.warns(RuntimeWarning, match="divide by zero encountered in .*"):
            # direction(0.0, ctype)
            evals = np.array([0.0, 1.0, 1.0, 2.0])
            out = direction(evals, ctype)
            assert np.isinf(out[0])
            assert not np.isinf(out[1:]).any()

class TestModeGroupConversion:
    # First test perfect squares
    @pytest.mark.parametrize("method", ['ceil', 'floor', 'round', 'raw'])
    def test_mode_to_group_squares(self, method): 
        n = np.arange(1,22)**2-1
        assert np.all(mode_to_group(n, method=method) == np.sqrt(n+1)-1), \
            f"Failed for method {method} with perfect squares."

    # Then test non-perfect squares with manually calculated expected values
    def test_mode_to_group(self):
        idx = np.arange(22)
        assert np.all(mode_to_group(idx, method='ceil') 
            == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4])
        assert np.all(mode_to_group(idx, method='floor') 
            == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
        assert np.all(mode_to_group(idx, method='round') 
            == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4])
        assert np.allclose(mode_to_group(idx, method='raw'), 
            np.sqrt(np.arange(1,23)) - 1)

    # First test perfect squares
    @pytest.mark.parametrize("method", ['ceil', 'floor', 'round', 'raw'])
    def test_group_to_mode_squares(self, method): 
        n = np.arange(22)
        assert np.all(group_to_mode(n, method=method) == (n + 1)**2 - 1), \
            f"Failed for method {method} with perfect squares."

    # Then test non-perfect squares with manually calculated expected values
    def test_group_to_mode(self):
        idx = mode_to_group(np.arange(22), method='raw')
        assert np.all(group_to_mode(idx, method='ceil') 
            == [0, 3, 3, 3, 8, 8, 8, 8, 8, 15, 15, 15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24])
        assert np.all(group_to_mode(idx, method='floor') 
            == [0, 0, 0, 3, 3, 3, 3, 3, 8,  8,  8,  8,  8,  8,  8, 15, 15, 15, 15, 15, 15, 15])
        assert np.all(group_to_mode(idx, method='round') 
            == [0, 0, 3, 3, 3, 3, 8, 8, 8,  8,  8,  8, 15, 15, 15, 15, 15, 15, 15, 15, 24, 24])
        assert np.allclose(group_to_mode(idx, method='raw'), 
            np.arange(22))

    # Mode -> Group -> Mode
    def test_mode_to_mode(self):
        """Tests that the 'raw' methods are perfectly invertible bijections."""
        a = np.arange(0, 100)
        b = group_to_mode(mode_to_group(a, method='raw'), method='raw')
        np.testing.assert_array_almost_equal(a, b)

    # Group -> Mode -> Group
    def test_group_to_group(self):
        """Tests that the 'raw' methods are perfectly invertible bijections."""
        a = np.linspace(0, 20, 100)
        with pytest.warns(UserWarning, match="mode_id should be an integer or array of integers."):
            b = mode_to_group(group_to_mode(a, method='raw'), method='raw') # type: ignore
        np.testing.assert_array_almost_equal(a, b)

@pytest.fixture(scope="module")
def sphere(): 
    mesh, _ = fetch_surf(surf_type='sphere', density='4k')
    return EigenSolver(mesh).solve(n_modes=100)

class TestFWHM:
    # Test that (i) both methods run on 2d data; (ii) 'fem' is very close to theory; (iii) 'wb'
    # approximately equals theory
    @pytest.mark.parametrize("method_and_rtol", [('wb', 1e-1), ('fem', 1e-6)])
    def test_fwhm(self, sphere, method_and_rtol):
        estimated_value = estimate_fwhm(sphere.geometry, sphere.emodes[:,1:], method=method_and_rtol[0])
        theoretical_value = np.sqrt(8 * np.log(2) / sphere.evals[1:])
        assert np.allclose(estimated_value, theoretical_value, rtol=method_and_rtol[1]), \
            f"Estimated value does not match theoretical value for method {method_and_rtol[0]}"

    # Test that 'wb' is very close to values previously computed using wb_command
    def test_fwhm_wb(self):
        ATOL = 1e-6
        # Do some CSV parsing without using pandas
        filename = Path(__file__).parent / 'test_data' / 'mesh_estimate_fwhm_results.csv'
        with open(filename, mode='r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            data_list = list(csv_reader)
        header = data_list[0]  # first row is the header

        # Compare with saved values
        for row in data_list[1:]:
            species = row[header.index('species')]
            template = row[header.index('template')]
            density = row[header.index('density')]
            hemi = row[header.index('hemi')]
            surf_type = row[header.index('surf_type')]
            data = row[header.index('data')]
            expected_fwhm = row[header.index('wbcommand_fwhm')]

            mesh, _ = fetch_surf(surf_type=surf_type, template=template, species=species, density=density, hemi=hemi)
            geometry = TriaMesh(mesh.v, mesh.t)
            vfunc = fetch_map(data=data, template=template, species=species, density=density, hemi=hemi)
            estimated_fwhm = estimate_fwhm(geometry, vfunc, method='wb')

            assert np.allclose(estimated_fwhm, float(expected_fwhm), atol=ATOL), \
                f"Estimated FWHM ({estimated_fwhm}) does not match expected FWHM ({float(expected_fwhm)}) for {data}"

    # Test that roi_mask runs without errors and produces different results than global FWHM (but
    # doesn't test actual correctness)
    # TODO : add correctness tests for method fem (what is ground truth)? 
    @pytest.mark.parametrize("method", ['wb', 'fem'])
    def test_roi_mask_execution(self, sphere, method):
        """Tests that the roi_mask successfully evaluates without shape errors."""
        # Create a dummy mask (e.g., just the first half of the vertices)
        num_vertices = sphere.geometry.v.shape[0]
        mask = np.zeros(num_vertices, dtype=bool)
        mask[:num_vertices // 2] = True
        
        # Test 1D array
        vfunc_1d = sphere.emodes[:, 1]
        fwhm_1d = estimate_fwhm(sphere, vfunc_1d, method=method, roi_mask=mask)
        assert isinstance(fwhm_1d, float)
        assert not np.isnan(fwhm_1d)
        
        # Verify FWHM is actually different than the global FWHM
        global_fwhm = estimate_fwhm(sphere, vfunc_1d, method=method, roi_mask=None)
        assert fwhm_1d != global_fwhm

        # Test 2D array
        vfunc_2d = sphere.emodes[:, 1:]
        fwhm_2d = estimate_fwhm(sphere, vfunc_2d, method=method, roi_mask=mask)
        assert isinstance(fwhm_2d, np.ndarray)
        assert fwhm_2d.shape == (sphere.n_modes - 1,)
        assert not np.isnan(fwhm_2d).any()
        assert not np.isinf(fwhm_2d).any()

class TestTruncateEmodes:
    # Test correctness of decompose
    def test_truncate_emodes_decompose(self, sphere):
        betas = np.random.default_rng().standard_normal((sphere.n_modes, 1))
        betas[0] = 0
        vfunc = sphere.emodes @ betas 

        power = np.cumsum(betas**2)
        power = power / power[-1]
        
        kmin = 3
        thresholds = 1 - (power[kmin-1:-1] + power[kmin:]) / 2

        for i in range(kmin, sphere.n_modes - 1):
            out = truncate_emodes(
                geometry=sphere.geometry, 
                vfunc=vfunc, 
                emodes=sphere.emodes, 
                mass=sphere.mass, 
                threshold_method='decompose', 
                threshold=thresholds[i - kmin], 
                output='mode'
            )
            assert out == i + 1, f"Expected {i + 1} modes, but got {out}."

    # Test correctness of reconstruct
    def test_truncate_emodes_reconstruct(self, sphere):
        betas = np.random.default_rng().standard_normal((sphere.n_modes,))
        betas[0] = 0
        vfunc = sphere.emodes @ betas

        # TODO: change to new reconstruct
        _, errors, _ = sphere.reconstruct(data=vfunc, mode_counts=np.arange(1, sphere.n_modes+1))

        kmin = 3
        thresholds = (errors[kmin-1:-1] + errors[kmin:]) / 2

        for i in range(kmin, sphere.n_modes - 1):
            out = truncate_emodes(
                geometry=sphere.geometry, 
                vfunc=vfunc, 
                emodes=sphere.emodes, 
                mass=sphere.mass, 
                threshold_method='reconstruct', 
                threshold=thresholds[i - kmin], 
                output='mode'
            )
            assert out == i + 1, f"Expected {i + 1} modes, but got {out}."
    
    # Test correctness of physical methods
    @pytest.mark.parametrize("threshold_method", ['eigenvalue', 'fwhm', 'wavelength'])
    def test_truncate_emodes_physical(self, sphere, threshold_method):
        kmin = 3
        thresholds = (sphere.evals[kmin-1:-1] + sphere.evals[kmin:]) / 2
        thresholds = convert_from_eigenvalue(thresholds, output=threshold_method)

        for i in range(kmin, sphere.n_modes - 1):
            out = truncate_emodes(
                geometry=sphere.geometry, 
                vfunc=sphere.emodes[:, i],
                evals=sphere.evals, 
                threshold_method=threshold_method, 
                threshold=thresholds[i - kmin], 
                output='mode'
            )
            assert out == i, f"Expected {i} modes, but got {out}."

    # Test auto-thresholding for physical methods
    @pytest.mark.parametrize("threshold_method", ['eigenvalue', 'fwhm', 'wavelength'])
    def test_truncate_emodes_auto_threshold(self, sphere, threshold_method):
        """Tests that passing threshold=None successfully triggers the FWHM estimator."""
        idx = np.random.default_rng().integers(1, sphere.n_modes-1)
        vfunc = sphere.emodes[:, idx] # Use a specific mode as the map
        
        # It should run without raising errors
        out_mode = truncate_emodes(
            geometry=sphere.geometry, 
            vfunc=vfunc, 
            threshold=None, # Trigger auto-fallback
            evals=sphere.evals, 
            threshold_method=threshold_method, 
            output='mode'
        )
        assert isinstance(out_mode, (int, np.integer))
        assert 0 < out_mode <= idx + 1

    # Test correctness of statistical thresholding using maps with some betas=0 
    @pytest.mark.parametrize("threshold_method", ['decompose', 'reconstruct'])
    def test_truncate_emodes_2d_statistical(self, sphere, threshold_method):
        betas = np.random.default_rng().integers(1, 2, size=(sphere.n_modes, sphere.n_modes-3))
        filt = 1 - np.tri(*betas.shape, k=-3).astype(int) 
        vfunc = sphere.emodes @ (betas * filt)
        expected_modes = filt.sum(axis=0)
        out = truncate_emodes(geometry=sphere.geometry, vfunc=vfunc, 
            emodes=sphere.emodes, evals=sphere.evals, mass=sphere.mass, 
            threshold_method=threshold_method, threshold=1e-8, output='mode')
        assert np.array_equal(out, expected_modes)

    # Test correctness of physical thresholding using maps with some betas=0
    @pytest.mark.parametrize("threshold_method", ['eigenvalue', 'fwhm', 'wavelength'])
    def test_truncate_emodes_2d_physical(self, sphere, threshold_method):
        betas = np.random.default_rng().integers(1, 2, size=(sphere.n_modes, sphere.n_modes-3))
        filt = 1 - np.tri(*betas.shape, k=-3).astype(int) 
        vfunc = sphere.emodes @ (betas * filt)
        thresholds = [sphere.evals[i-1] for i in filt.sum(axis=0)]
        thresholds = convert_from_eigenvalue(thresholds, output=threshold_method)
        expected_modes = filt.sum(axis=0)
        out = truncate_emodes(geometry=sphere.geometry, vfunc=vfunc, 
            emodes=sphere.emodes, evals=sphere.evals, mass=sphere.mass, 
            threshold_method=threshold_method, threshold=thresholds, output='mode')
        assert np.array_equal(out, expected_modes)

    # Test some exceptions
    def test_truncate_emodes_exceptions(self, sphere):
        """Tests that missing requirements trigger the correct errors."""
        vfunc = sphere.emodes[:, 1]
        
        # Missing emodes/mass for decompose
        with pytest.raises(ValueError, match="emodes and mass must be provided"):
            truncate_emodes(geometry=sphere.geometry, vfunc=vfunc, threshold=0.05, 
                            threshold_method='decompose')
            
        # Missing evals for physical properties
        with pytest.raises(ValueError, match="evals must be provided"):
            truncate_emodes(geometry=sphere.geometry, vfunc=vfunc, threshold=10.0, 
                            threshold_method='wavelength')
            
        # Mismatched threshold array length
        vfunc_2d = sphere.emodes[:, 1:3] # 2 maps
        with pytest.raises(ValueError, match="Length of threshold array must match"):
            truncate_emodes(geometry=sphere.geometry, vfunc=vfunc_2d, threshold=[0.05], # Only 1 threshold
                            emodes=sphere.emodes, mass=sphere.mass)
