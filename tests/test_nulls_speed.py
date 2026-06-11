import pytest
import time
import numpy as np
from neuromodes.eigen import EigenSolver
from neuromodes.io import fetch_example_surf

# This test should be run with the `-s` flag to print timing results, e.g. `pytest -s tests/test_nulls_speed.py`

# Params
# These will be used for the bulk of the profiling, in the `test_all` function. 
# A smaller set of tests on the 32k mesh is below. 
density = '4k'
n_groups = (10,15,20)
n_maps = (1,10,100)
n_nulls = (1,10,100)

@pytest.fixture(scope='module')
def solver():
    """Initialise solver and solve for eigenmodes, which will be used for all tests."""
    mesh, medmask = fetch_example_surf(density=density)
    print(f"\nInitilising mesh with {density} vertices and {max(n_groups)**2} modes.")
    tic = time.time()
    s = EigenSolver(mesh, mask=medmask).solve(n_modes=max(n_groups)**2)
    print(f"Time to solve eigenmodes: {time.time() - tic:.5f} seconds.\n")
    return s

@pytest.fixture(scope='module')
def test_data(solver):
    """Generate 2D test data"""
    rng = np.random.default_rng()
    return rng.normal(loc=1, size=(solver.n_verts, max(n_maps)))  # random normal data, non-zero mean

def test_min(solver, test_data):
    """Quick test to ensure everything works"""
    n_null = min(n_nulls)
    n_map = min(n_maps)
    n_group = min(n_groups)
    tic = time.time()
    solver.eigenstrap(test_data[:,:n_map], n_nulls=n_null, n_groups=n_group)
    toc = time.time()
    print(f"Time to generate {n_null} nulls for {n_map} maps with {n_group**2} modes: {toc - tic:.5f} seconds.")

@pytest.mark.parametrize("rotation_method", ['qr', 'scipy'])
def test_supermax_nulls(solver, test_data, rotation_method):
    """Run one test (profile with `pytest tests/test_nulls_speed.py::test_supermax_nulls --profile`)."""
    n_null = 10 * max(n_nulls)
    n_map = min(n_maps)
    n_group = min(n_groups)

    tic = time.time()
    solver.eigenstrap(test_data[:,:n_map], n_nulls=n_null, n_groups=n_group, rotation_method=rotation_method)
    toc = time.time()
    print(f"Time to generate {n_null} nulls for {n_map} maps with {n_group**2} modes using {rotation_method}: {toc - tic:.5f} seconds.")

@pytest.mark.parametrize("rotation_method", ['qr', 'scipy'])
def test_supermax_maps(solver, test_data, rotation_method):
    """Run one test (profile with `pytest tests/test_nulls_speed.py::test_supermax_maps --profile`)."""
    n_null = min(n_nulls)
    n_map = 10 * max(n_maps)
    n_group = min(n_groups)

    tic = time.time()
    solver.eigenstrap(test_data[:,:n_map], n_nulls=n_null, n_groups=n_group, rotation_method=rotation_method)
    toc = time.time()
    print(f"Time to generate {n_null} nulls for {n_map} maps with {n_group**2} modes using {rotation_method}: {toc - tic:.5f} seconds.")

def test_all(solver, test_data):
    """Test all combinations of n_maps/n_groups/n_nulls"""
    s = 12 # for spacing only
    for n_group in n_groups:
        print(f"\n{n_group**2} modes:")         # all these print statements are formatting output
        print(f"{'Maps | Nulls'}", end="")
        for n_null in n_nulls:
            print(f"{n_null:>{s}}", end="")
        print()
        for n_map in n_maps:
            print(f"{n_map:<{s}}", end="")
            for n_null in n_nulls:
                tic = time.time()               # this is where the computation (and timing) is
                solver.eigenstrap(test_data[:,:n_map], n_nulls=n_null, n_groups=n_group)
                toc = time.time()
                print(f"{toc - tic:>{s}.6f}", end="")
            print()
              
def test_32k():
    """Test on 32k mesh with specific number of nulls/maps/modes."""
    density='32k'
    n_modes = 100 # should be square number
    n_maps = 10
    n_nulls = 1000

    print(f"\nInitialising mesh with {density} vertices and {n_modes} modes.")
    tic = time.time()
    solver = EigenSolver(fetch_example_surf(density=density)[0]).solve(n_modes=n_modes)
    print(f"Time to solve eigenmodes: {time.time() - tic:.5f} seconds.")

    test_data = np.random.default_rng().normal(size=(solver.n_verts, n_maps))

    for rotation_method in ['qr', 'scipy']:
        tic = time.time()
        solver.eigenstrap(test_data, n_nulls=n_nulls, rotation_method=rotation_method)
        print(f"Nulls generated for {n_maps} maps with {n_nulls} nulls using {rotation_method}: {time.time() - tic:.5f} seconds.")
