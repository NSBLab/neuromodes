import os
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pytest
from neuromodes.io import fetch_surf, fetch_map
from neuromodes.eigen import EigenSolver
from neuromodes.waves import (simulate_waves, calc_wave_speed, get_balloon_params, _gen_noise,
                              _simulate_waves_fem, _analytical_fc)

@pytest.fixture(scope="module")
def solver():
    mesh, medmask = fetch_surf(density='4k')
    hetero = fetch_map(data="myelinmap", density="4k")[medmask]
    return EigenSolver(mesh, mask=medmask, hetero=hetero).solve(n_modes=100, seed=0)

def test_unusual_wave_speed(solver):
    with pytest.warns(UserWarning, match=r'range of 0-150 m/s \(calculated 46.2-162.5 m/s\).'):
        solver.simulate_waves(r=1000, nt=10)

def test_unusual_wave_speed_no_hetero(solver):
    with pytest.warns(UserWarning, match=r'range of 0-115 m/s \(calculated 116.0 m/s\).'):
        simulate_waves(
            solver.emodes,
            solver.evals,
            mass=solver.mass,
            r=1000,
            speed_limits=(0, 115),
            nt=100
            )

def test_single_speed_limit(solver):
    with pytest.raises(ValueError, match="speed_limits must be a tuple"):
        solver.simulate_waves(nt=10, r=18.0, speed_limits=150)

def test_reversed_speed_limits(solver):
    with pytest.raises(ValueError, match="speed_limits must be a tuple"):
        solver.simulate_waves(nt=10, r=18.0, speed_limits=(150, 0))

def test_simulate_waves_impulse(solver):

    # Simulate timeseries with a 10ms impulse of white noise to the cortex
    dt = 1e-3
    nt = 200
    i_start = 10 # ms
    i_stop = 20 # ms
    rng = np.random.default_rng(0)
    impulse = rng.standard_normal(size=solver.n_verts)
    ext_input = np.zeros((solver.n_verts, nt))
    ext_input[:, i_start:i_stop] = impulse[:, np.newaxis]

    fourier_ts = solver.simulate_waves(ext_input=ext_input, dt=dt)
    ode_ts = solver.simulate_waves(ext_input=ext_input, dt=dt, pde_method='ode')

    # Check output shapes
    assert fourier_ts.shape == (solver.n_verts, nt), 'Fourier output shape is incorrect.'
    assert ode_ts.shape == (solver.n_verts, nt), 'ODE output shape is incorrect.'

    # Check that activity is negligible before impulse
    assert np.allclose(fourier_ts[:, :i_start], 0, atol=1e-3), \
        'Fourier activity is not negligible before impulse.'
    assert np.allclose(ode_ts[:, :i_start], 0, atol=1e-10), \
        'ODE activity is not negligible before impulse.'

    # Check that activity returns to negligible by 200ms
    assert np.allclose(fourier_ts[:, -1], 0, atol=1e-4), \
        'Fourier activity is not negligible after 200ms.'
    assert np.allclose(ode_ts[:, -1], 0, atol=1e-8), \
        'ODE activity is not negligible after 200ms.'

def test_simulate_waves_methods(solver):

    nt = 100
    dt = 1e-4
    seed = 0

    # Check that Fourier and ODE methods produce similar neural activity at selected timepoints
    fourier_ts = solver.simulate_waves(nt=nt, dt=dt, seed=seed)
    ode_ts = solver.simulate_waves(nt=nt, dt=dt, seed=seed, pde_method='ode')

    for t in range(50, nt):
        assert np.corrcoef(fourier_ts[:, t], ode_ts[:, t])[0, 1] > 0.85, \
            f'Fourier and ODE solutions are not correlated at r>.85 at t={t}.'

def test_simulate_waves_methods_bold(solver):

    nt = 100
    dt = 1e-2
    seed = 0

    # Check that Fourier and ODE methods produce similar BOLD signal at selected timepoints
    activity_fourier = solver.simulate_waves(nt=nt, dt=dt, seed=seed)
    activity_ode = solver.simulate_waves(nt=nt, dt=dt, seed=seed, pde_method='ode')
    bold_fourier = solver.bold_transform(activity_fourier, dt=dt)
    bold_ode = solver.bold_transform(activity_ode, dt=dt, pde_method='ode')

    # Methods converge to r=.98 by t=500, but this takes too long to run, so just anchor the test
    # to a lower value to catch if the alignment ever drops (TODO: add to validation?)
    for t in range(75, nt):
        assert np.corrcoef(bold_fourier[:, t], bold_ode[:, t])[0, 1] > 0.6, \
            f'Fourier and ODE BOLD solutions are not correlated at r>.6 at t={t}.'
        
def test_gen_noise_reproducibility():
    seed = 0
    noise1 = _gen_noise(5, 10, seed=seed)
    noise2 = _gen_noise(5, 20, seed=seed)
    assert (noise1 == noise2[:, :10]).all(), \
        "Noise generated with the same seed does not match across different nt."

def test_simulate_waves_reproducibility_fourier(solver):
    
    nt = 100
    dt = 1e-2
    seed = 0
    
    # Same seed should be reproducible across different nt
    ts0 = solver.simulate_waves(nt=nt, dt=dt, seed=seed)
    ts1 = solver.simulate_waves(nt=2*nt, dt=dt, seed=seed)

    # Different seed should produce different results
    ts2 = solver.simulate_waves(nt=nt, dt=dt, seed=seed+1)

    mse01 = np.mean((ts0 - ts1[:, :nt])**2)
    mse02 = np.mean((ts0 - ts2)**2)
    assert mse01 < 1e-5, \
        f"Simulated timeseries with the same seed do not match (MSE={mse01:.4e})."
    assert mse02 > 1e-3, \
        f"Simulated timeseries with different seeds match unexpectedly (MSE={mse02:.4f})."

def test_simulate_waves_invalid_input_shape(solver):

    with pytest.raises(ValueError, match=r"data.*first dimension.*3619"):
        solver.simulate_waves(ext_input=np.ones((4002, 1000)))

def test_simulate_waves_invalid_pde_method(solver):

    with pytest.raises(ValueError, match="Invalid PDE method 'zote'"):
        solver.simulate_waves(nt=10, pde_method='zote')

@pytest.mark.filterwarnings("ignore:overflow encountered in scalar power:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in dot:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar subtract:RuntimeWarning")
def test_simulate_waves_ode_balloon_overflow(solver):

    # Large dt can cause overflow errors in the dqdt expression for the ODE balloon model, so
    # test that our error is raised
    dt = 1

    with pytest.raises(RuntimeError, match="message: Required step size is less than spacing"):
        activity = solver.simulate_waves(dt=dt, nt=10, pde_method='ode')
        solver.bold_transform(activity, pde_method='ode', dt=dt)

def test_simulate_waves_cached(solver):
    # Get CACHE_DIR
    cache_dir = os.getenv("CACHE_DIR")

    # Test with temporary directory
    try:
        with TemporaryDirectory() as temp_cache_dir:
            os.environ["CACHE_DIR"] = temp_cache_dir
            _ = solver.simulate_waves(nt=10, cache_input=True, seed=0)

            # Check that the temp_cache_dir/neuromodes/waves subdirectory exists
            cache_dir_waves = os.path.join(
                temp_cache_dir,
                "neuromodes",
                "waves"
            )
            assert os.path.exists(cache_dir_waves), "Waves cache directory was not created."
    finally:
        # Restore original CACHE_DIR
        if cache_dir is not None:
            os.environ["CACHE_DIR"] = cache_dir
        else:
            del os.environ["CACHE_DIR"]

def test_simulate_waves_balloon_param(solver):
    nt = 100
    dt = 1e-2

    activity = solver.simulate_waves(nt=nt, dt=dt)
    bold_default = solver.bold_transform(activity, dt=dt)
    bold_custom = solver.bold_transform(activity, dt=dt, rho=0.5)

    assert not np.allclose(bold_default, bold_custom), \
        "BOLD signals with different balloon model parameters match unexpectedly."
    
def test_get_balloon_params():
    # Check an invalid override
    with pytest.raises(ValueError, match="'rho' must be positive."):
        _ = get_balloon_params(rho=0)

def test_calc_wave_speed(solver):

    # Homogeneous case
    speed = calc_wave_speed(r=18.0, gamma=116)
    assert isinstance(speed, float), "Output type is not float for hetero=None."

    # Heterogeneous case
    speed = calc_wave_speed(r=18.0, gamma=116, scaled_hetero=solver.hetero)
    assert np.all(speed > 0), "Output contains non-positive wave speeds when using scaled_hetero."
    assert speed.shape == (solver.n_verts,), "Output shape is incorrect when using scaled_hetero." # type: ignore

def test_analytical_fc(solver):
    sim_ts = solver.simulate_waves(nt=1000, dt=0.1, seed=0)
    # Check that simulated FC from waves aligns with the analytical FC
    ana_fc = _analytical_fc(solver.emodes, solver.evals, r=17.4, gamma=116)
    sim_fc = np.corrcoef(sim_ts)
    mse = np.mean((ana_fc - sim_fc)**2)
    assert mse < 0.01, f"Analytical FC does not align with simulated FC (MSE={mse:.4f})."

def test_fem_alignment(solver):
    # Check that modal approximation aligns with FEM solution
    nt=50
    dt=0.1
    seed=0

    fourier_ts = solver.simulate_waves(nt=nt, dt=dt, seed=seed)

    # Get lumped mass and run FEM simulation
    solver.compute_lbo(lump=True)
    fem_ts = _simulate_waves_fem(solver.mass, solver.stiffness, nt=nt, dt=dt, seed=seed, n_jobs=-1)

    # Reset mass attribute
    solver.compute_lbo(lump=False)

    # Assess
    for t in range(10, nt):
        assert np.corrcoef(fourier_ts[:, t], fem_ts[:, t])[0, 1] > 0.8, \
            f'Modal and FEM solutions are not correlated at r>.8 at t={t}.'

def test_fem_no_joblib(solver):
    # Check that FEM simulation runs without joblib installed
    nt=2
    dt=0.1
    seed=0

    with patch.dict('sys.modules', {'joblib': None}):
        # Get lumped mass and run FEM simulation
        solver.compute_lbo(lump=True)
        with pytest.warns(UserWarning, match="joblib is not installed"):
            fem_ts = _simulate_waves_fem(solver.mass, solver.stiffness, nt=nt, dt=dt, seed=seed,
                                         n_jobs=-1)

        # Reset mass attribute
        solver.compute_lbo(lump=False)

        assert fem_ts.shape == (solver.n_verts, nt), \
            "FEM output shape is incorrect when joblib is not installed."