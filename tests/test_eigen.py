from pathlib import Path

import numpy as np
import pytest
from lapy.shapedna import normalize_ev

from neuromodes.eigen import EigenSolver, get_eigengroup_inds, is_orthonormal_basis
from neuromodes.io import fetch_example_map, fetch_example_surf
from neuromodes.stats import sigmoid_rescale, zscorew


@pytest.fixture(scope="module")
def surf_medmask():
    return fetch_example_surf(density='4k')

@pytest.fixture(scope="module")
def solver(surf_medmask):
    surf, medmask = surf_medmask
    myelinmap = fetch_example_map(data="myelinmap", density="4k")[medmask]
    solver = EigenSolver(surf, mask=medmask) # TODO: just use surf_medmask?
    hetero = sigmoid_rescale(zscorew(myelinmap, solver.mass), steepness=0.5, upper=2.0)
    return solver.compute_lbo(hetero=hetero)

@pytest.fixture(scope="module")
def hetero(solver):
    return solver.hetero

def test_invalid_mask_shape(surf_medmask):
    surf, _ = surf_medmask
    bad_mask = np.ones(10)
    with pytest.raises(ValueError, match=r"mask must have shape \(4002,\)"):
        EigenSolver(surf, mask=bad_mask)

def test_no_hetero(surf_medmask):
    surf, medmask = surf_medmask
    homo_solver = EigenSolver(surf, mask=medmask)
    homo_solver.solve(10, decomp='cholesky') # hardcoded to 10 to match the saved prior_modes

    # Load homogeneous eigenmodes/eigenvalues for comparison
    test_data = Path(__file__).parent / 'test_data'
    prior_modes = np.load(test_data / 'sp-human_tpl-fsLR_den-4k_hemi-L_midthickness-emodes.npy')
    prior_evals = np.load(test_data / 'sp-human_tpl-fsLR_den-4k_hemi-L_midthickness-evals.npy')

    for i in range(1, homo_solver.n_modes):  # TODO: use allclose(np.abs())
        assert np.abs(np.corrcoef(homo_solver.emodes[:, i], prior_modes[:, i])[0, 1]) > 0.99, \
            f'Eigenmode {i} does not match the previously computed homogeneous result.'
        assert np.allclose(homo_solver.evals[i], prior_evals[i], rtol=0.1), \
            f'Eigenvalue {i} does not match the previously computed homogeneous result.'

def test_invalid_hetero_shape(surf_medmask):
    surf, medmask = surf_medmask
    bad_hetero = np.ones(10)
    with pytest.raises(ValueError, match=r"shape \(n_verts,\) = \(3619,\)"):
        EigenSolver(surf, mask=medmask).compute_lbo(hetero=bad_hetero)

def test_nan_hetero(surf_medmask):
    surf, medmask = surf_medmask
    bad_hetero = np.ones(medmask.sum())
    bad_hetero[0] = np.nan
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        EigenSolver(surf, mask=medmask).compute_lbo(hetero=bad_hetero)

    bad_hetero[0] = np.inf
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        EigenSolver(surf, mask=medmask).compute_lbo(hetero=bad_hetero)

def test_hetero_ones(surf_medmask):
    surf, medmask = surf_medmask
    hetero = np.ones(sum(medmask))

    # If hetero is all ones, this should give the same stiffness matrix
    homo_solver = EigenSolver(surf, mask=medmask).solve(20, decomp='cholesky')
    hetero_solver = EigenSolver(surf, mask=medmask).solve(20, hetero=hetero, decomp='cholesky')

    assert np.allclose(hetero_solver.evals, homo_solver.evals), \
        'Eigenvalues with hetero=ones do not match homogeneous eigenvalues.'
    for i in range(homo_solver.n_modes):
        assert np.allclose(hetero_solver.emodes[:, i], homo_solver.emodes[:, i], atol=1e-4), \
            f'Eigenmode {i+1} with hetero=ones does not match its homogeneous equivalent.'

def test_real_heteromaps():
    mesh, medmask = fetch_example_surf() # 32k density to match included maps
    for map in ['fcgradient1', 'myelinmap', 'ndi', 'odi', 'thickness']:
        hetero = fetch_example_map(map)[medmask]
        # just test that LBO can be computed without error
        EigenSolver(mesh, mask=medmask).compute_lbo(hetero=hetero)

def test_symmetric_mass(solver):
    diff = solver.mass - solver.mass.transpose()
    assert abs(diff).max() == 0, 'Mass matrix is not symmetric.'

# TODO: test that lumped mass is the same as summed consistent mass

def test_symmetric_stiffness(solver):
    diff = solver.stiffness - solver.stiffness.transpose()
    assert abs(diff).max() == 0, 'Stiffness matrix is not symmetric.'

def test_stiffness_rowsums(solver):
    assert abs(solver.stiffness.sum(axis=1)).max() < 2e-6
        
def test_seeded_modes(solver):
    n_modes = 16
    solver.solve(n_modes, hetero=solver.hetero, align_emodes=False, set_emode1=False, seed=36,
                 decomp='cholesky')
    emodes1 = solver.emodes.copy()
    evals1 = solver.evals.copy()

    solver.solve(n_modes, hetero=solver.hetero, align_emodes=False, set_emode1=False, seed=36,
                    decomp='cholesky')
    emodes2 = solver.emodes.copy()
    evals2 = solver.evals.copy()

    assert (emodes1 == emodes2).all(), 'Modes from same seed are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed are not identical.'

    solver.solve(n_modes, hetero=solver.hetero, align_emodes=False, set_emode1=False, seed=37,
                 decomp='cholesky')
    emodes3 = solver.emodes.copy()
    evals3 = solver.evals.copy()

    assert not (emodes1 == emodes3).all(), 'Modes from different seeds should not be identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seeds should not be identical.'

def test_generator_seeded_modes(solver):
    n_modes = 16
    rng = np.random.default_rng(0)
    solver.solve(n_modes, hetero=solver.hetero, align_emodes=False, set_emode1=False,
                 decomp='cholesky', seed=rng)
    emodes1 = solver.emodes.copy()
    evals1 = solver.evals.copy()

    # Reset the generator to ensure the same sequence of random numbers
    rng = np.random.default_rng(0)
    solver.solve(n_modes, hetero=solver.hetero, align_emodes=False, set_emode1=False,
                 decomp='cholesky', seed=rng)
    emodes2 = solver.emodes.copy()
    evals2 = solver.evals.copy()
    assert (emodes1 == emodes2).all(), 'Modes from same seed generator are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed generator are not identical.'

    rng = np.random.default_rng(1)
    solver.solve(n_modes, hetero=solver.hetero, align_emodes=False, set_emode1=False,
                 decomp='cholesky', seed=rng)
    emodes3 = solver.emodes.copy()
    evals3 = solver.evals.copy()
    assert not (emodes1 == emodes3).all(), 'Modes from different seed generators are identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seed generators are identical.'

def test_vector_seeded_modes(solver):
    n_modes = 16
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(size=solver.n_verts)

    solver.solve(n_modes, hetero=solver.hetero, align_emodes=False, set_emode1=False, v0=v0,
                 decomp='cholesky')
    emodes1 = solver.emodes.copy()
    evals1 = solver.evals.copy()

    # Reuse the same seed vector
    solver.solve(n_modes, hetero=solver.hetero, align_emodes=False, set_emode1=False, v0=v0,
                 decomp='cholesky')
    emodes2 = solver.emodes.copy()
    evals2 = solver.evals.copy()

    assert (emodes1 == emodes2).all(), 'Modes from same seed vector are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed vector are not identical.'

    v0_diff = rng.standard_normal(size=solver.n_verts)

    solver.solve(n_modes, hetero=solver.hetero, align_emodes=False, set_emode1=False,
                 v0=v0_diff, decomp='cholesky')
    emodes3 = solver.emodes.copy()
    evals3 = solver.evals.copy()

    assert not (emodes1 == emodes3).all(), 'Modes from different seed vectors are identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seed vectors are identical.'

def test_invalid_vector_seed(solver):
    with pytest.raises(ValueError,
                       match=r"v0 must have shape \(n_verts,\) = \(3619,\)."):
        solver.solve(2400, v0=np.ones(10))

def test_cholesky_vs_lu(solver):
    solver.solve(16, hetero=solver.hetero, decomp='lu', set_emode1=False)
    emodes_lu = solver.emodes.copy()
    evals_lu = solver.evals.copy()

    # NOTE: this is the last in-place modification of the solver object, so the emodes and evals
    # computed here are used for the rest of the tests
    solver.solve(16, hetero=solver.hetero, decomp='cholesky')

    assert np.allclose(emodes_lu, solver.emodes), \
        'Eigenmodes from LU and Cholesky decompositions do not match.'
    assert np.allclose(evals_lu, solver.evals), \
        'Eigenvalues from LU and Cholesky decompositions do not match.'

def test_unaligned_modes(solver, surf_medmask):
    emodes = solver.emodes
    surf, medmask = surf_medmask
    emodes_unalign = EigenSolver(surf, mask=medmask).solve(
        solver.n_modes, hetero=solver.hetero, align_emodes=False, decomp='cholesky'
        ).emodes
    
    assert not np.all(emodes_unalign[0, :] >= 0), \
        'Unaligned first vertex should have both positive and negative values.'
    assert np.all(emodes[0, :] >= 0), 'Aligned first vertex has negative values.'
    assert (abs(emodes) == abs(emodes_unalign)).all(), \
    'Absolute values of unaligned modes do not match those of aligned modes.'

def test_solve_lumped_mass(solver, surf_medmask):
    surf, medmask = surf_medmask

    # Get modes after solving with lumped mass matrix
    emodes_lump = EigenSolver(surf, mask=medmask).solve(solver.n_modes, hetero=solver.hetero,
                                                        lump=True, decomp='cholesky').emodes

    for i in range(1, solver.n_modes):
        mse = np.mean((solver.emodes[:, i] - emodes_lump[:, i])**2)
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

def test_n_modes_consistency(solver, surf_medmask):
    surf, medmask = surf_medmask

    # Solve for more modes and check that the first 16 modes are approximately the same
    # TODO: may as well use 100 modes in the fixture and instead solve for fewer here?
    solver_more_modes = EigenSolver(surf, mask=medmask).solve(100, hetero=solver.hetero,
                                                              decomp='cholesky')
    assert np.allclose(solver.emodes, solver_more_modes.emodes[:, :16], atol=1e-4), \
        'Modes differ when solving for different n_modes.'
    
def test_normalized_surf(solver):
    surf = solver.geometry

    # Use LaPy to normalize evals
    evals_lapy = normalize_ev(surf, solver.evals)

    # Normalize mesh before EigenSolver
    surf_norm = surf.__class__(surf.v, surf.t)  # Avoid in-place modification
    surf_norm.normalize_()
    solver_norm = EigenSolver(surf_norm).solve(16, hetero=solver.hetero, decomp='cholesky')

    # Check that evals match between the two normalization approaches
    assert np.allclose(evals_lapy, solver_norm.evals, atol=1e-20), \
    'Evals from LaPy normalization do not match evals from EigenSolver normalization.'

def test_constant_mode1(solver, surf_medmask):
    surf, medmask = surf_medmask
    emode1 = solver.emodes[:, 0]
    
    solver_unset = EigenSolver(surf, mask=medmask).solve(2, hetero=solver.hetero, set_emode1=False,
                                                         decomp='cholesky')
    emode1_unfixed = solver_unset.emodes[:, 0]
    eval1_unfixed = solver_unset.evals[0]

    assert (emode1 == emode1[0]).all(), 'Fixed first mode is not exactly constant.'
    assert not (emode1_unfixed == emode1_unfixed[0]).all(), \
        'Unfixed first mode is exactly constant.'
    assert np.allclose(emode1_unfixed, emode1[0],
                       atol=1e-4), 'Unfixed first mode is not approximately constant.'
    assert np.isclose(np.mean(emode1_unfixed), emode1[0],
                      atol=1e-6), 'Mean of unfixed first mode is not close to fixed value.'
    assert eval1_unfixed < 1e-6, 'First eigenvalue of unfixed first mode is not close to 0.'

def test_positive_sigma(solver, surf_medmask):
    surf, medmask = surf_medmask
    emodes = solver.emodes
    evals = solver.evals

    # check that Cholesky is rejected
    with pytest.raises(ValueError, match=r"sigma must be negative for Cholesky decomposition"):
        solver_pos_sigma = EigenSolver(surf, mask=medmask).solve(
            solver.n_modes, hetero=solver.hetero, sigma=0, decomp='cholesky'
            )

    with pytest.warns(UserWarning, match=r"emodes\[:, 0\] will not be set"):
        solver_pos_sigma = EigenSolver(surf, mask=medmask).solve(
            solver.n_modes, hetero=solver.hetero, sigma=1e-4, decomp='lu'
            )
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

# TODO: cache validation tests