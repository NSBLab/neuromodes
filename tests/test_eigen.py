from pathlib import Path
import pytest
import numpy as np
from lapy import TriaMesh
from nsbtools.io import fetch_surf, fetch_map, mask_surf
from nsbtools.eigen import EigenSolver, decompose, reconstruct, calc_norm_power, is_mass_orthonormal_modes

@pytest.fixture
def surf_medmask_hetero():
    mesh, medmask = fetch_surf(density='4k')
    rng = np.random.default_rng(0)
    hetero = rng.standard_normal(size=len(medmask))
    return mesh, medmask, hetero

def test_init_params(surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    _ = EigenSolver(surf, mask=medmask, hetero=hetero, n_modes=2, alpha=0.5, beta_old=3,
                         scaling='exponential')
    
def test_premasked_surf(surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    masked_surf = mask_surf(surf, medmask)
    _ = EigenSolver(masked_surf, hetero=hetero[medmask])
    
def test_triamesh_surf(surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    mesh = TriaMesh(surf.vertices, surf.faces)
    _ = EigenSolver(mesh, mask=medmask, hetero=hetero)

def test_no_medmask(surf_medmask_hetero):
    surf, _, hetero = surf_medmask_hetero
    EigenSolver(surf, hetero=hetero)

def test_invalid_mask_shape(surf_medmask_hetero):
    surf, _, _ = surf_medmask_hetero
    bad_mask = np.ones(10)
    with pytest.raises(ValueError, match=r"`mask` \(10\) must match .* mesh \(4002\)."):
        EigenSolver(surf, mask=bad_mask)

def test_invalid_hetero_shape(surf_medmask_hetero):
    surf, _, _ = surf_medmask_hetero
    bad_hetero = np.ones(10)
    with pytest.raises(ValueError, match=r"`hetero` \(10\) must match .* mesh \(4002\)."):
        EigenSolver(surf, hetero=bad_hetero)

def test_nan_inf_hetero(surf_medmask_hetero):
    surf, _, hetero = surf_medmask_hetero
    hetero[0] = np.nan
    with pytest.raises(ValueError, match="`hetero` must not contain NaNs or Infs."):
        EigenSolver(surf, hetero=hetero)

    hetero[0] = np.inf
    with pytest.raises(ValueError, match="`hetero` must not contain NaNs or Infs."):
        EigenSolver(surf, hetero=hetero)

def test_nan_inf_hetero_medmask(surf_medmask_hetero):
    # Inject NaN/Inf at a cortical vertex (should raise error)
    surf, medmask, hetero = surf_medmask_hetero
    cortical_vertex = np.where(medmask)[0][0]
    print(cortical_vertex)
    hetero[cortical_vertex] = np.nan
    with pytest.raises(ValueError, match="`hetero` must not contain NaNs or Infs."):
        EigenSolver(surf, hetero=hetero)
    hetero[cortical_vertex] = np.inf
    with pytest.raises(ValueError, match="`hetero` must not contain NaNs or Infs."):
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

def test_init_invalid_wave_speed(surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    solver = EigenSolver(surf, mask=medmask, hetero=hetero, beta_old=11)

    # This alpha scaling should result in implausibly fast waves at some vertices
    solver.alpha = 2
    with pytest.raises(ValueError, match='.*non-physiological wave speeds.*'):
        solver.hetero = hetero

def test_init_invalid_scaling(surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    with pytest.raises(ValueError, match="Invalid scaling 'plantasia'.*"):
        EigenSolver(surf, mask=medmask, hetero=hetero, scaling='plantasia')

@pytest.fixture
def presolver(surf_medmask_hetero):
    surf, medmask, hetero = surf_medmask_hetero
    presolver = EigenSolver(surf, mask=medmask, hetero=hetero, n_modes=10)
    return presolver

def test_symmetric_mass(presolver):
    diff = presolver.mass - presolver.mass.transpose()
    assert abs(diff).max() == 0, 'Mass matrix is not symmetric.'

def test_symmetric_stiffness(presolver):
    diff = presolver.stiffness - presolver.stiffness.transpose()
    assert abs(diff).max() == 0, 'Stiffness matrix is not symmetric.'

def test_stiffness_rowsums(presolver):
    assert abs(presolver.stiffness.sum(axis=1)).max() < 2e-6

def test_no_hetero(presolver):

    with pytest.warns(UserWarning, match="Setting `alpha` and `beta` to 0.*"):
        presolver.hetero = None
    emodes, evals = presolver.solve()

    test_data = Path(__file__).parent / 'test_data'

    # Load homogeneous eigenmodes/eigenvalues for comparison
    prior_modes = np.load(test_data / 'sp-human_tpl-fsLR_den-4k_hemi-L_midthickness-emodes.npy')
    prior_evals = np.load(test_data / 'sp-human_tpl-fsLR_den-4k_hemi-L_midthickness-evals.npy')

    for i in range(1, 10):
        assert np.abs(np.corrcoef(emodes[:, i], prior_modes[:, i])[0, 1]) > 0.99, \
            f'Eigenmode {i} does not match the previously computed homogeneous result.'
        assert np.allclose(evals[i], prior_evals[i], rtol=0.1), \
            f'Eigenvalue {i} does not match the previously computed homogeneous result.'
        
def test_seeded_modes(presolver):
    emodes1, evals1 = presolver.solve(standardize=False, fix_mode1=False, seed=36)
    emodes2, evals2 = presolver.solve(standardize=False, fix_mode1=False, seed=36)

    assert (emodes1 == emodes2).all(), 'Modes from same seed are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed are not identical.'

    emodes3, evals3 = presolver.solve(standardize=False, fix_mode1=False, seed=37)

    assert not (emodes1 == emodes3).all(), 'Modes from different seeds should not be identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seeds should not be identical.'

def test_vector_seeded_modes(presolver):
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(size=presolver.n_verts)
    emodes1, evals1 = presolver.solve(standardize=False, fix_mode1=False, seed=v0)

    # Reuse the same seed vector
    emodes2, evals2 = presolver.solve(standardize=False, fix_mode1=False, seed=v0)

    assert (emodes1 == emodes2).all(), 'Modes from same seed vector are not identical.'
    assert (evals1 == evals2).all(), 'Eigenvalues from same seed vector are not identical.'

    rng = np.random.default_rng(1)
    v0_diff = rng.standard_normal(size=presolver.n_verts)

    emodes3, evals3 = presolver.solve(standardize=False, fix_mode1=False, seed=v0_diff)

    assert not (emodes1 == emodes3).all(), 'Modes from different seed vectors are identical.'
    assert not (evals1 == evals3).all(), 'Eigenvalues from different seed vectors are identical.'

def test_invalid_vector_seed(presolver):
    with pytest.raises(ValueError,
                       match=r"of shape \((3636,)\)."):
        presolver.solve(seed=np.ones(10))

@pytest.fixture
def solver(presolver):
    _, _ = presolver.solve()
    return presolver

def test_nonstandard_modes(solver):
    emodes = solver.emodes
    emodes_nonstd, _ = solver.solve(standardize=False)
    
    assert np.all(emodes[0, :] >= 0), 'Standardized first element has negative values.'
    assert not np.all(emodes_nonstd[0, :] >= 0), \
        'Non-standardized first element should have both positive and negative values.'
    assert np.allclose(abs(emodes), abs(emodes_nonstd),
                       atol=1e-6), 'Non-standardized modes do not match standardized modes.'

def test_solve_lumped_mass(solver, surf_medmask_hetero):
    emodes = solver.emodes
    surf, medmask, hetero = surf_medmask_hetero

    # Get first 10 modes after solving with lumped mass matrix
    solver = EigenSolver(surf, mask=medmask, hetero=hetero, lump=True, n_modes=10)
    solver.solve()
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

def test_constant_mode1(solver):
    emode1 = solver.emodes[:, 0]

    solver.solve(fix_mode1=False)
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
    assert not is_mass_orthonormal_modes(emodes)

    emodes[:, 0] += 0.1 # Destroy mass-orthonormality by changing first mode's value

    assert not is_mass_orthonormal_modes(emodes, solver.mass)

def test_check_euclidean_orthonorm():
    # Create orthonormal vectors in Euclidean space
    vecs = np.eye(5)

    assert is_mass_orthonormal_modes(vecs)
    assert is_mass_orthonormal_modes(vecs, mass=np.eye(5))
    assert not is_mass_orthonormal_modes(vecs, mass=np.zeros((5, 5)))

def test_decompose_eigenmodes(solver):
    emodes = solver.emodes

    for i in range(solver.n_modes):
        data = emodes[:, i]  # Use an eigenmode as data
        beta = decompose(data, emodes, mass=solver.mass)

        # The mode should load onto only itself due to orthogonality
        beta_expected = np.zeros((solver.n_modes, 1))
        beta_expected[i, 0] = 1
        assert np.allclose(beta, beta_expected, atol=1e-4), f'Decomposition of mode {i+1} failed.'

def test_decompose_invalid_data_shape(solver):

    with pytest.raises(ValueError, match=r".*`data` \(4002\) must match .* `emodes` \(3636\)."):
        decompose(np.ones(4002), solver.emodes, mass=solver.mass)

def test_decompose_nan_inf_mode(solver):
    emodes = solver.emodes
    data = np.ones(solver.n_verts)

    emodes[0,0] = np.nan
    with pytest.raises(ValueError, match="`emodes` contains NaNs or Infs."):
        decompose(data, emodes, mass=solver.mass)

    emodes[0,0] = np.inf
    with pytest.raises(ValueError, match="`emodes` contains NaNs or Infs."):
        decompose(data, emodes, mass=solver.mass)

def test_decompose_massless(solver):

    with pytest.raises(ValueError, match=r"Mass matrix of shape \(3636, 3636\) must be provided .*"):
        decompose(np.ones(solver.n_verts), solver.emodes)

def test_decompose_invalid_method(solver):

    with pytest.raises(ValueError, match="Invalid decomposition method 'fornitonian'.*"):
        decompose(np.ones(solver.n_verts), solver.emodes, method='fornitonian')

@pytest.fixture
def gen_eigenmap(solver):

    # Use randomly weighted sums of modes to generate maps
    n_maps = 3
    rng = np.random.default_rng(0)
    weights = rng.standard_normal(size=(solver.n_modes, n_maps))
    eigenmaps = solver.emodes @ weights

    return weights, eigenmaps

def test_reconstruct_mode_superposition(solver, gen_eigenmap):
    weights, eigenmaps = gen_eigenmap

    beta, recon, pearsonr = reconstruct(eigenmaps, solver.emodes, mass=solver.mass)

    # Pearson's r should increase from 0 to 1 when using mode 1 only versus all relevant modes
    assert np.allclose(recon[:,-1,:], eigenmaps,
                       atol=1e-6), 'Final reconstructions do not match input maps.'
    assert (pearsonr[0,:] == 0).all(), 'Pearson r is not 0 when using only mode 1.'
    assert np.allclose(pearsonr[-1,:], 1,
                       atol=1e-10), 'Pearson r is not close to 1 when using all modes.'

    assert np.allclose(beta[-1], weights, atol=1e-4), \
        'Beta values do not match input mode weights when using all modes.'

    # MSE should be 0 when using all modes
    _, _, mse = reconstruct(eigenmaps, solver.emodes, mass=solver.mass, metric='mse')
    assert np.allclose(mse[-1,:], 0,
                       atol=1e-10), 'MSE is not close to 0 when using all modes.'

    # Reconstruct using the first 5 modes, then the first 2 modes
    _, _, pearsonr_modesq = reconstruct(eigenmaps, solver.emodes, mass=solver.mass, mode_seq=[5,2])
    assert (pearsonr_modesq[0,:] == pearsonr[4,:]).all(), \
        'Reconstruction scores do not match for 5 modes.'
    assert (pearsonr_modesq[1,:] == pearsonr[1,:]).all(), \
        'Reconstruction scores do not match for 2 modes.'

def test_reconstruct_regress_method(solver, gen_eigenmap):
    _, eigenmaps = gen_eigenmap

    _, _, pearsonr = reconstruct(eigenmaps, solver.emodes, method='regress')
    _, _, mse = reconstruct(eigenmaps, solver.emodes, method='regress', metric='mse')

    # Pearson's r should strictly increase and MSE should strictly decrease when adding modes
    assert np.all(np.diff(pearsonr, axis=0) > 0), \
        'Pearson r does not strictly increase when adding modes.'
    assert np.all(np.diff(mse, axis=0) < 0), \
        'MSE does not strictly decrease when adding modes.'

def test_reconstruct_mode_superposition_timeseries(solver, gen_eigenmap):
    _, eigenmaps = gen_eigenmap

    eigen_ts = eigenmaps.astype(np.float32) # Prevent memory allocation error
    fc = np.corrcoef(eigen_ts)

    # Treat eigenmaps as timepoints of activity
    _, _, fc_recon, pearsonr = reconstruct(eigen_ts, solver.emodes, mass=solver.mass,
                                                        timeseries=True)

    assert np.allclose(fc_recon[:,:,-1], fc, atol=1e-3), 'Reconstructed FC does not match original.'
    assert pearsonr[0] == 0, \
        'FC reconstruction score is not close to 0 when using only the constant mode.'
    assert pearsonr[-1] > 0.999, 'FC reconstruction score is not close to 1 when using all modes.'
    
    _, _, _, mse = reconstruct(eigen_ts, solver.emodes, mass=solver.mass, timeseries=True,
                                      metric='mse')
    assert mse[-1] < 1e-9, 'MSE is not close to 0 when using all modes.'

def test_reconstruct_real_map_32k():
    # Get modes of fsLR 32k midthickness (data is in 32k)
    mesh, medmask = fetch_surf()
    rng = np.random.default_rng(0)
    hetero = rng.standard_normal(size=len(medmask))
    solver = EigenSolver(mesh, mask=medmask, hetero=hetero, n_modes=10)
    solver.solve()
    emodes = solver.emodes

    # Load FC gradient from Margulies 2016 PNAS
    map = fetch_map('fcgradient1')[medmask]
    _, _, recon_score = reconstruct(map, emodes, mass=solver.mass)

    # Pearson's r should strictly increase from 0, but not reach 1
    assert np.all(np.diff(recon_score) > 1e-6), \
        r'Reconstruction score \(r\) does not strictly increase.'
    assert np.isclose(recon_score[0], 0, atol=1e-6), \
        r'Reconstruction score \(r\) is non-zero when using only the constant mode.'
    assert not np.isclose(recon_score[-1], 1, atol=1e-6), \
        r'Reconstruction score \(r\) is unexpectedly close to 1 for only 10 modes.'

def test_reconstruct_invalid_map_shape(solver):

    with pytest.raises(ValueError, match=r".*`data` \(4002\) must match .* `emodes` \(3636\)."):
        reconstruct(np.ones(4002), solver.emodes, mass=solver.mass)

def test_reconstruct_invalid_metric(solver):

    with pytest.raises(ValueError, match="Invalid metric 'barnesv'.*"):
        reconstruct(np.ones(solver.n_verts), solver.emodes, mass=solver.mass, metric='barnesv')

def test_reconstruct_massless(solver):

    with pytest.raises(ValueError, match=r"Mass matrix of shape \(3636, 3636\) must be provided .*"):
        reconstruct(np.ones(solver.n_verts), solver.emodes)

def test_calc_norm_power():
    # Dummy coefficients
    beta = np.array([[-3, 4], [1.5, 2], [0, 0.1]])

    norm_power = calc_norm_power(beta)

    # Check that powers are non-negative
    assert np.all(norm_power >= 0), 'Normalized powers contain negative values.'

    # Check that columns sum to 1
    assert np.allclose(np.sum(norm_power, axis=0), 1, atol=1e-8), 'Normalized powers do not sum to 1.'