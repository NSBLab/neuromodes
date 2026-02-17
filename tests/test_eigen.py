from pathlib import Path
from lapy.shapedna import normalize_ev
import numpy as np
import pytest
from neuromodes.eigen import EigenSolver, is_orthonormal_basis, scale_hetero
from neuromodes.io import fetch_vol, fetch_surf, fetch_map, mask_mesh

def test_vol_modes():
    for hemi in ['L', 'R']:
        for structure in ['thalamus', 'hippocampus', 'striatum', 'cortex']:
            if structure == 'cortex':
                vol = fetch_vol(structure=structure, species='mouse', template='AMBA', hemi=hemi)
            else:
                vol = fetch_vol(structure=structure, hemi=hemi)
            solver = EigenSolver(vol).solve(10, seed=0)

            assert solver.emodes.shape == (vol.v.shape[0], 10), (
                f'emodes has shape {solver.emodes.shape}, should be {(vol.v.shape[0], 10)}.')
            assert len(solver.evals) == 10, (f'evals has length {len(solver.evals)}, should be 10.')

@pytest.fixture
def surf_medmask_hetero():
    mesh, medmask = fetch_surf(density='4k')
    rng = np.random.default_rng(0)
    hetero = rng.standard_normal(size=len(medmask))
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
    bad_mask = np.ones(10)
    with pytest.raises(ValueError, match=r"`mask` must have shape \(n_verts,\) = \(4002,\)"):
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
    with pytest.warns(UserWarning, match="`alpha` is ignored"):
        EigenSolver(surf, mask=medmask, hetero=None, alpha=0.5)
    with pytest.warns(UserWarning, match="`scaling` is ignored"):
        EigenSolver(surf, mask=medmask, hetero=None, scaling='exponential')

def test_invalid_hetero_shape(surf_medmask_hetero):
    surf, _, _ = surf_medmask_hetero
    bad_hetero = np.ones(10)
    with pytest.raises(ValueError, match=r"vertices in the provided mesh \(4002\)."):
        EigenSolver(surf, hetero=bad_hetero)

def test_nan_inf_hetero(surf_medmask_hetero):
    surf, _, hetero = surf_medmask_hetero
    hetero[0] = np.nan
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        EigenSolver(surf, hetero=hetero)

    hetero[0] = np.inf
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        EigenSolver(surf, hetero=hetero)

def test_constant_hetero(surf_medmask_hetero):
    surf, _, hetero = surf_medmask_hetero
    hetero[:] = 2.0
    with pytest.warns(UserWarning, match="Provided `hetero` is constant"):
        EigenSolver(surf, hetero=hetero)

def test_nan_inf_hetero_medmask(surf_medmask_hetero):
    # Inject NaN/Inf at a cortical vertex (should raise error)
    surf, medmask, hetero = surf_medmask_hetero
    cortical_vertex = np.where(medmask)[0][0]
    print(cortical_vertex)
    hetero[cortical_vertex] = np.nan
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        EigenSolver(surf, hetero=hetero)
    hetero[cortical_vertex] = np.inf
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        EigenSolver(surf, hetero=hetero)

def test_nan_inf_hetero_medmask_ignored(surf_medmask_hetero):
    # Inject NaN/Inf at a medial vertex (should be ignored)
    surf, medmask, hetero = surf_medmask_hetero
    medial_vertex = np.where(~medmask)[0][0]
    print(medial_vertex)
    hetero[medial_vertex] = np.nan
    EigenSolver(surf, mask=medmask, hetero=hetero)
    hetero[medial_vertex] = np.inf  
    EigenSolver(surf, mask=medmask, hetero=hetero)

def test_hetero_ones(surf_medmask_hetero):
    surf, medmask, _ = surf_medmask_hetero
    hetero = np.ones(sum(medmask))

    homo_solver = EigenSolver(surf, mask=medmask).solve(20, seed=0)
    with pytest.warns(UserWarning, match="Provided `hetero` is constant"):
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

@pytest.fixture
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

def test_vector_seeded_modes(presolver):
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(size=presolver.n_verts)
    presolver.solve(16, standardize=False, fix_mode1=False, seed=v0)
    emodes1 = presolver.emodes
    evals1 = presolver.evals

    # Reuse the same seed vector
    presolver.solve(presolver.n_modes, standardize=False, fix_mode1=False, seed=v0)
    emodes2 = presolver.emodes
    evals2 = presolver.evals

    assert (emodes1 == emodes2).all(), 'Modes from same seed vector are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed vector are not identical.'

    rng = np.random.default_rng(1)
    v0_diff = rng.standard_normal(size=presolver.n_verts)

    presolver.solve(presolver.n_modes, standardize=False, fix_mode1=False, seed=v0_diff)
    emodes3 = presolver.emodes
    evals3 = presolver.evals

    assert not (emodes1 == emodes3).all(), 'Modes from different seed vectors are identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seed vectors are identical.'

def test_invalid_vector_seed(presolver):
    with pytest.raises(ValueError,
                       match=r"of shape \(n_verts,\) = \(3636,\)."):
        presolver.solve(16, seed=np.ones(10))

@pytest.fixture
def solver(presolver):
    presolver.solve(n_modes=16, seed=0)
    return presolver

def test_nonstandard_modes(solver):
    emodes = solver.emodes
    emodes_nonstd = solver.solve(solver.n_modes, standardize=False, seed=0).emodes
    
    assert not np.all(emodes_nonstd[0, :] >= 0), \
        'Non-standardized first vertex should have both positive and negative values.'
    assert np.all(emodes[0, :] >= 0), 'Standardized first vertex has negative values.'
    assert (abs(emodes) == abs(emodes_nonstd)).all(), \
    'Non-standardized modes do not match standardized modes.'

def test_solve_lumped_mass(solver, surf_medmask_hetero):
    emodes = solver.emodes
    surf, medmask, hetero = surf_medmask_hetero

    # Get modes after solving with lumped mass matrix
    solver = EigenSolver(surf, mask=medmask, hetero=hetero)
    solver.solve(emodes.shape[1], lump=True)
    emodes_lumped = solver.emodes

    assert np.allclose(abs(emodes), abs(emodes_lumped), atol=1e-3), \
        'Lumped mass modes do not approximately match original modes.'
    for i in range(1, solver.n_modes):
        assert np.corrcoef(emodes[:, i], emodes_lumped[:, i])[0, 1] > 0.99, \
            'Lumped mass modes do not match original modes.'

def test_solutions(solver):
    emodes = solver.emodes
    evals = solver.evals

    assert emodes.shape == (solver.n_verts,
                           solver.n_modes), (f'Eigenmodes have shape {emodes.shape}, should be '
                                            f'{(solver.n_verts, solver.n_modes)}.')
    assert len(evals) == solver.n_modes, (f'Eigenvalues has length {len(evals)}, should be '
                                         f'{solver.n_modes}.')
    assert np.all(np.diff(evals) > 0), 'Eigenvalues are not sorted in descending order.'

def test_n_modes_consistency(solver):
    emode_set = solver.emodes

    # Solve for more modes and check that the first 16 modes are approximately the same
    emode_set_larger_n_modes = solver.solve(100, seed=0).emodes[:, :16]
    assert np.allclose(emode_set, emode_set_larger_n_modes, atol=1e-4), \
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

def test_normalized_vol():
    # Above test but for volumes
    hippo = fetch_vol('hippocampus')
    hetero = np.random.default_rng(0).standard_normal(hippo.v.shape[0])

    volser = EigenSolver(hippo, hetero=hetero).solve(16, seed=0)
    evals_lapy = normalize_ev(volser.geometry, volser.evals)
    volser_norm = EigenSolver(hippo, hetero=hetero, normalize=True).solve(16, seed=0)

    assert np.allclose(evals_lapy, volser_norm.evals, atol=1e-20), \
    'Evals from LaPy normalization do not match evals from EigenSolver normalization.'

def test_constant_mode1(solver):
    emode1 = solver.emodes[:, 0]

    solver.solve(2, fix_mode1=False)
    emode1_unfixed = solver.emodes[:, 0]
    eval1_unfixed = solver.evals[0]

    assert (emode1 == emode1[0]).all(), 'Fixed first mode is not exactly constant.'
    assert np.allclose(emode1_unfixed, emode1[0],
                       atol=1e-4), 'Unfixed first mode is not approximately constant.'
    assert np.isclose(np.mean(emode1_unfixed), emode1[0],
                      atol=1e-6), 'Mean of unfixed first mode is not close to fixed value.'
    assert eval1_unfixed < 1e-6, 'First eigenvalue of unfixed first mode is not close to 0.'

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
    assert is_orthonormal_basis(vecs, mass=np.eye(5))
    assert not is_orthonormal_basis(vecs, mass=np.ones((5, 5)))

def test_scale_hetero(surf_medmask_hetero):
    _, _, hetero = surf_medmask_hetero

    # Check that sigmoid-scaled hetero has mean ~1 and be within (0, 2)
    hetero_sig = scale_hetero(hetero)
    assert np.isclose(np.mean(hetero_sig), 1.0, atol=1e-3)
    assert np.all((hetero_sig > 0) & (hetero_sig < 2))

    # Check that exponential-scaled hetero is all positive
    hetero_exp = scale_hetero(hetero, scaling='exponential')
    assert np.all(hetero_exp > 0)

def test_invalid_scale_hetero(surf_medmask_hetero):
    _, _, hetero = surf_medmask_hetero

    with pytest.raises(ValueError, match="Invalid scaling 'plantasia'"):
        scale_hetero(hetero, scaling='plantasia')