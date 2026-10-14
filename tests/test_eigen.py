from pathlib import Path

import numpy as np
import pytest
from lapy.shapedna import normalize_ev

from neuromodes.eigen import EigenSolver, get_eigengroup_inds, is_orthonormal_basis
from neuromodes.io import fetch_example_map, fetch_example_surf, fetch_example_vol
from neuromodes.mesh import normalize_vol
from neuromodes.stats import sigmoid_rescale, zscorew


def test_vol_modes():
    for hemi in ['L', 'R']:
        for structure in ['thalamus', 'hippocampus', 'striatum', 'isocortex']:
            if structure == 'isocortex':
                vol = fetch_example_vol(structure=structure, species='mouse', template='AMBA',
                                        res='200um', hemi=hemi)
            else:
                vol = fetch_example_vol(structure=structure, hemi=hemi)

            # test hetero while we're at it
            hetero = np.abs(np.random.default_rng(0).standard_normal(vol.v.shape[0]))
            solver = EigenSolver(vol).solve(10, hetero=hetero, seed=0)

            # check that evals are in ascending order
            assert np.all(np.diff(solver.evals) > 0), \
                f"Eigenvalues are not sorted in ascending order for {structure} {hemi}."

            # check that modes are mass-orthonormal
            assert is_orthonormal_basis(solver.emodes, mass=solver.mass), \
                f"Eigenmodes are not mass-orthonormal for {structure} {hemi}."

@pytest.fixture(scope="module")
def surf_medmask():
    return fetch_example_surf(res='4k')

@pytest.fixture(scope="module")
def presolver(surf_medmask):
    surf, medmask = surf_medmask
    myelinmap = fetch_example_map(data="myelinmap", res="4k")[medmask]
    solver = EigenSolver(surf, mask=medmask) # TODO: just use surf_medmask?
    hetero = sigmoid_rescale(zscorew(myelinmap, solver.mass), steepness=0.5, upper=2.0)
    return solver.compute_lbo(hetero=hetero)

@pytest.fixture(scope="module")
def hetero(presolver):
    return presolver.hetero

def test_invalid_mask_shape(surf_medmask):
    surf, _ = surf_medmask
    bad_mask = np.ones(10)
    with pytest.raises(ValueError, match=r"mask must have shape \(4002,\)"):
        EigenSolver(surf, mask=bad_mask)

def test_no_hetero(surf_medmask):
    surf, medmask = surf_medmask
    homo_solver = EigenSolver(surf, mask=medmask)
    homo_solver.solve(10) # hardcoded to 10 to match the saved prior_modes

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
    homo_solver = EigenSolver(surf, mask=medmask).solve(20)
    hetero_solver = EigenSolver(surf, mask=medmask).solve(20, hetero=hetero)

    assert np.allclose(hetero_solver.evals, homo_solver.evals), \
        'Eigenvalues with hetero=ones do not match homogeneous eigenvalues.'
    for i in range(homo_solver.n_modes):
        assert np.allclose(hetero_solver.emodes[:, i], homo_solver.emodes[:, i], atol=1e-4), \
            f'Eigenmode {i+1} with hetero=ones does not match its homogeneous equivalent.'

def test_real_heteromaps():
    mesh, medmask = fetch_example_surf() # 32k to match included maps
    solver = EigenSolver(mesh, mask=medmask)

    map_names = ['fcgradient1', 'myelinmap', 'ndi', 'odi', 'thickness']
    heteros = np.stack([fetch_example_map(m)[medmask] for m in map_names], axis=1)
    heteros_z = zscorew(heteros, solver.mass)
    heteros_zsig = sigmoid_rescale(heteros_z, upper=2)

    # just test that LBO can be computed without error
    for i in range(len(map_names)):
        solver.compute_lbo(hetero=heteros_zsig[:, i])

def test_symmetric_mass(presolver):
    diff = presolver.mass - presolver.mass.transpose()
    assert abs(diff).max() == 0, 'Mass matrix is not symmetric.'

def test_mass_rowsums(presolver):
    mass_rowsums = presolver.mass.sum(axis=1).A1
    mass_lumped = EigenSolver(presolver.geometry).compute_lbo(lump=True).mass.data
    np.testing.assert_allclose(mass_rowsums, mass_lumped,
                                  err_msg='Mass matrix row sums do not match lumped mass diagonal.')

def test_symmetric_stiffness(presolver):
    diff = presolver.stiffness - presolver.stiffness.transpose()
    assert abs(diff).max() == 0, 'Stiffness matrix is not symmetric.'

def test_stiffness_rowsums(presolver):
    assert abs(presolver.stiffness.sum(axis=1)).max() < 2e-6
        
def test_seeded_modes(presolver):
    n_modes = 16
    presolver.solve(n_modes, hetero=presolver.hetero, align_emodes=False, set_emode1=False, seed=36)
    emodes1 = presolver.emodes.copy()
    evals1 = presolver.evals.copy()

    presolver.solve(n_modes, hetero=presolver.hetero, align_emodes=False, set_emode1=False, seed=36)
    emodes2 = presolver.emodes.copy()
    evals2 = presolver.evals.copy()

    assert (emodes1 == emodes2).all(), 'Modes from same seed are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed are not identical.'

    presolver.solve(n_modes, hetero=presolver.hetero, align_emodes=False, set_emode1=False, seed=37)
    emodes3 = presolver.emodes.copy()
    evals3 = presolver.evals.copy()

    assert not (emodes1 == emodes3).all(), 'Modes from different seeds should not be identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seeds should not be identical.'

def test_generator_seeded_modes(presolver):
    n_modes = 16
    rng = np.random.default_rng(0)
    presolver.solve(n_modes, hetero=presolver.hetero, align_emodes=False, set_emode1=False,
                    seed=rng)
    emodes1 = presolver.emodes.copy()
    evals1 = presolver.evals.copy()

    # Reset the generator to ensure the same sequence of random numbers
    rng = np.random.default_rng(0)
    presolver.solve(n_modes, hetero=presolver.hetero, align_emodes=False, set_emode1=False,
                    seed=rng)
    emodes2 = presolver.emodes.copy()
    evals2 = presolver.evals.copy()
    assert (emodes1 == emodes2).all(), 'Modes from same seed generator are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed generator are not identical.'

    rng = np.random.default_rng(1)
    presolver.solve(n_modes, hetero=presolver.hetero, align_emodes=False, set_emode1=False,
                    seed=rng)
    emodes3 = presolver.emodes.copy()
    evals3 = presolver.evals.copy()
    assert not (emodes1 == emodes3).all(), 'Modes from different seed generators are identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seed generators are identical.'

def test_vector_seeded_modes(presolver):
    n_modes = 16
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(size=presolver.n_verts)

    presolver.solve(n_modes, hetero=presolver.hetero, align_emodes=False, set_emode1=False, v0=v0)
    emodes1 = presolver.emodes.copy()
    evals1 = presolver.evals.copy()

    # Reuse the same seed vector
    presolver.solve(n_modes, hetero=presolver.hetero, align_emodes=False, set_emode1=False, v0=v0)
    emodes2 = presolver.emodes.copy()
    evals2 = presolver.evals.copy()

    assert (emodes1 == emodes2).all(), 'Modes from same seed vector are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed vector are not identical.'

    v0_diff = rng.standard_normal(size=presolver.n_verts)

    presolver.solve(n_modes, hetero=presolver.hetero, align_emodes=False, set_emode1=False,
                    v0=v0_diff)
    emodes3 = presolver.emodes.copy()
    evals3 = presolver.evals.copy()

    assert not (emodes1 == emodes3).all(), 'Modes from different seed vectors are identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seed vectors are identical.'

def test_invalid_vector_seed(presolver):
    with pytest.raises(ValueError,
                       match=r"v0 must have shape \(n_verts,\) = \(3619,\)."):
        presolver.solve(2400, v0=np.ones(10))

# TODO: this seems a bit redundant
@pytest.fixture(scope="module")
def solver(presolver):
    return presolver.solve(16, hetero=presolver.hetero)

def test_unaligned_modes(solver, surf_medmask):
    emodes = solver.emodes
    surf, medmask = surf_medmask
    emodes_unalign = EigenSolver(surf, mask=medmask).solve(solver.n_modes, hetero=solver.hetero,
                                                           align_emodes=False).emodes
    
    assert not np.all(emodes_unalign[0, :] >= 0), \
        'Unaligned first vertex should have both positive and negative values.'
    assert np.all(emodes[0, :] >= 0), 'Aligned first vertex has negative values.'
    assert (abs(emodes) == abs(emodes_unalign)).all(), \
    'Absolute values of unaligned modes do not match those of aligned modes.'

def test_solve_lumped_mass(solver, surf_medmask):
    surf, medmask = surf_medmask

    # Get modes after solving with lumped mass matrix
    emodes_lump = EigenSolver(surf, mask=medmask).solve(solver.n_modes, hetero=solver.hetero,
                                                        lump=True).emodes

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
    solver_more_modes = EigenSolver(surf, mask=medmask).solve(100, hetero=solver.hetero)
    assert np.allclose(solver.emodes, solver_more_modes.emodes[:, :16], atol=1e-4), \
        'Modes differ when solving for different n_modes.'
    
def test_normalized_surf(solver):
    surf = solver.geometry

    # Use LaPy to normalize evals
    evals_lapy = normalize_ev(surf, solver.evals)

    # Normalize mesh before EigenSolver
    surf_norm = surf.__class__(surf.v, surf.t)  # Avoid in-place modification
    surf_norm.normalize_()
    solver_norm = EigenSolver(surf_norm).solve(16, hetero=solver.hetero)

    # Check that evals match between the two normalization approaches
    assert np.allclose(evals_lapy, solver_norm.evals, atol=1e-20), \
    'Evals from LaPy normalization do not match evals from EigenSolver normalization.'

def test_normalized_vol():
    # Above test but for volumes
    hippo = fetch_example_vol('hippocampus')

    volser = EigenSolver(hippo).solve(16, seed=0)
    evals_lapy = normalize_ev(volser.geometry, volser.evals)
    volser_norm = EigenSolver(normalize_vol(hippo)).solve(16, seed=0)

    assert np.allclose(evals_lapy, volser_norm.evals, atol=1e-20), \
    'Evals from LaPy normalization do not match evals from EigenSolver normalization.'

def test_constant_mode1(solver, surf_medmask):
    surf, medmask = surf_medmask
    emode1 = solver.emodes[:, 0]
    
    solver_unset = EigenSolver(surf, mask=medmask).solve(2, hetero=solver.hetero, set_emode1=False)
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

    with pytest.warns(UserWarning, match=r"emodes\[:, 0\] will not be set"):
        solver_pos_sigma = EigenSolver(surf, mask=medmask).solve(solver.n_modes,
                                                                 hetero=solver.hetero, sigma=1e-4)
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