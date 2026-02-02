import pytest
import os
from tempfile import TemporaryDirectory
import numpy as np
from neuromodes.io import fetch_surf
from neuromodes.eigen import EigenSolver
from neuromodes.waves import (simulate_waves, calc_wave_speed, get_balloon_params,
                              _simulate_waves_fem)

@pytest.fixture
def solver():
    rng = np.random.default_rng(0)
    mesh, medmask = fetch_surf(density='4k')
    hetero = rng.standard_normal(size=sum(medmask))
    return EigenSolver(mesh, mask=medmask, hetero=hetero).solve(n_modes=100, seed=0)

def test_unusual_wave_speed(solver):
    with pytest.warns(UserWarning, match=r'range of 0-150 m/s \(calculated 23.3-160.4 m/s\).'):
        solver.simulate_waves(r=1000, nt=100)

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
    with pytest.raises(ValueError, match="`speed_limits` must be a tuple"):
        simulate_waves(
            solver.emodes,
            solver.evals,
            mass=solver.mass,
            r=18.0,
            speed_limits=150,
        )

def test_reversed_speed_limits(solver):
    with pytest.raises(ValueError, match="`speed_limits` must be a tuple"):
        simulate_waves(
            solver.emodes,
            solver.evals,
            mass=solver.mass,
            r=18.0,
            speed_limits=(150, 0),
        )

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

    fourier_ts = simulate_waves(
        solver.emodes,
        solver.evals,
        ext_input=ext_input,
        mass=solver.mass,
        dt=dt,
        checks=False
    )
    ode_ts = simulate_waves(
        solver.emodes,
        solver.evals,
        ext_input=ext_input,
        mass=solver.mass,
        dt=dt,
        pde_method='ode',
        checks=False
    )

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

    fourier_ts = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=seed,
        checks=False
    )
    ode_ts = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=seed,
        pde_method='ode',
        checks=False
    )

    for t in range(50, nt):
        assert np.corrcoef(fourier_ts[:, t], ode_ts[:, t])[0, 1] > 0.85, \
            f'Fourier and ODE solutions are not correlated at r>.85 at t={t}.'

def test_simulate_waves_methods_bold(solver):

    nt = 100
    dt = 1e-2
    seed = 0

    # Check that Fourier and ODE methods produce similar BOLD signal at selected timepoints

    bold_fourier = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        bold_out=True,
        seed=seed,
        checks=False
    )
    bold_ode = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        bold_out=True,
        seed=seed,
        pde_method='ode',
        checks=False
    )

    # Methods converge to r=.98 by t=500, but this takes too long to run, so just anchor the test
    # to a lower value to catch if the alignment ever drops
    for t in range(75, nt):
        assert np.corrcoef(bold_fourier[:, t], bold_ode[:, t])[0, 1] > 0.6, \
            f'Fourier and ODE BOLD solutions are not correlated at r>.6 at t={t}.'

def test_simulate_waves_seed_bold_reproducibility_fourier(solver):
    
    nt = 100
    dt = 1e-1
    seed = 36

    ts1 = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=seed,
        bold_out=True,
        checks=False
    )
    ts2 = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=seed,
        bold_out=True,
        checks=False
    )
    ts3 = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        seed=seed+1,
        bold_out=True,
        checks=False
    )

    assert np.allclose(ts1, ts2), "Simulations with the same seed do not match."
    assert not np.allclose(ts1, ts3), "Simulations with different seeds match unexpectedly."

def test_simulate_waves_invalid_input_shape(solver):

    with pytest.raises(ValueError, match=r"n_verts is the number of rows in `emodes` \(3636\)."):
        simulate_waves(
            solver.emodes,
            solver.evals,
            ext_input=np.ones((4002, 1000)),
            mass=solver.mass
        )

def test_simulate_waves_invalid_pde_method(solver):

    with pytest.raises(ValueError, match="Invalid PDE method 'zote'"):
        simulate_waves(
            solver.emodes,
            solver.evals,
            mass=solver.mass,
            pde_method='zote'
        )

@pytest.mark.filterwarnings("ignore:overflow encountered in scalar power:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in dot:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar subtract:RuntimeWarning")
def test_simulate_waves_ode_balloon_overflow(solver):

    # Large `dt` can cause overflow errors in the `dqdt` expression for the ODE balloon model, so
    # test that our error is raised

    with pytest.raises(RuntimeError, match="message: Required step size is less than spacing"):
        simulate_waves(
            solver.emodes,
            solver.evals,
            mass=solver.mass,
            dt=1,
            nt=10,
            pde_method='ode',
            bold_out=True,
            checks=False
        )

def test_simulate_waves_cached(solver):
    # Get CACHE_DIR
    cache_dir = os.getenv("CACHE_DIR")

    # Test with temporary directory
    try:
        with TemporaryDirectory() as temp_cache_dir:
            os.environ["CACHE_DIR"] = temp_cache_dir
            _ = simulate_waves(
                solver.emodes,
                solver.evals,
                mass=solver.mass,
                nt=10,
                cache_input=True,
                checks=False,
                seed=0
            )

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

    ts_default = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        bold_out=True,
        checks=False
    )

    ts_custom = simulate_waves(
        solver.emodes,
        solver.evals,
        mass=solver.mass,
        nt=nt,
        dt=dt,
        bold_out=True,
        rho = 0.5,
        checks=False
    )

    assert not np.allclose(ts_default, ts_custom), \
        "BOLD signals with different balloon model parameters match unexpectedly."
    
def test_get_balloon_params():

    # Check a default
    params = get_balloon_params()
    assert params['rho'] == 0.34, "Default parameter 'rho' is incorrect."

    # Check an override
    params = get_balloon_params(rho=0.5)
    assert params['rho'] == 0.5, "Overridden parameter 'rho' is incorrect."

    # Check an invalid override
    with pytest.raises(ValueError, match=r"\(received rho=0\)."):
        _ = get_balloon_params(rho=0)

    # Check an invalid parameter name
    with pytest.raises(ValueError, match="Invalid Balloon model parameter 'yoyoyo'."):
        _ = get_balloon_params(yoyoyo=1.0)

def test_calc_wave_speed(solver):

    # Homogeneous case
    speed = calc_wave_speed(r=18.0, gamma=116)
    assert isinstance(speed, float), "Output type is not float for `hetero=None`."

    # Heterogeneous case
    speed = calc_wave_speed(r=18.0, gamma=116, scaled_hetero=solver.hetero)
    assert np.all(speed > 0), "Output contains non-positive wave speeds when using `scaled_hetero`."
    assert speed.shape == (solver.n_verts,), "Output shape is incorrect when using `scaled_hetero`."

def test_fem_alignment(solver):
    # Check that modal approximation aligns with FEM solution
    nt=50
    dt=0.01

    modal_ts = solver.simulate_waves(nt=nt, dt=dt, seed=0)
    fem_ts = _simulate_waves_fem(solver.mass, solver.stiffness, nt=nt, dt=dt, seed=0)
    for t in range(10, nt):
        assert np.corrcoef(modal_ts[:, t], fem_ts[:, t])[0, 1] > 0.8, \
            f'Modal and FEM solutions are not correlated at r>.8 at t={t}.'