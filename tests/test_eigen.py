from pathlib import Path
from lapy.shapedna import normalize_ev
import numpy as np
import pytest
from scipy.stats import zscore  # TODO: replace with stats.zscorew
from neuromodes.eigen import EigenSolver, is_orthonormal_basis, sigmoid_rescale, get_eigengroup_inds
from neuromodes.io import fetch_example_surf, fetch_example_map
from neuromodes.mesh import mask_mesh

@pytest.fixture(scope="module")
def surf_medmask_hetero():
    mesh, medmask = fetch_example_surf(density='4k')
    hetero = fetch_map(data="myelinmap", density="4k")
    hetero = sigmoid_rescale(zscore(hetero), steepness=0.5, upper=2.0)
    return mesh, medmask, hetero

def test_init_params(surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    _ = EigenSolver(surf, mask=medmask, hetero=hetero)
    
def test_premasked_surf(surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    masked_surf = mask_mesh(surf, medmask)
    _ = EigenSolver(masked_surf, hetero=hetero[medmask])

def test_no_medmask(surf_medmask_hetero):
    surf, _, hetero = surf_medmask_hetero
    EigenSolver(surf, hetero=hetero)

def test_invalid_mask_shape(surf_medmask_hetero):
    surf, _, _ = surf_medmask_hetero
    bad_mask = np.ones(10)
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
    mesh, medmask = fetch_example_surf() # 32k density to match real maps
    for map in ['fcgradient1', 'myelinmap', 'ndi', 'odi', 'thickness']:
        hetero = fetch_example_map(map)
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

def test_sigmoid_rescale():
    hetero = np.random.default_rng().normal(loc=100, scale=10, size=1000)

    # Check that sigmoid-scaled hetero is within (0, 2)
    hetero_sig = sigmoid_rescale(zscore(hetero), steepness=0.5, upper=2.0)
    assert np.all((hetero_sig > 0) & (hetero_sig < 2))

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
