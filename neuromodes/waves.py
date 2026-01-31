"""
Module for using neural field theory to simulate neural activity and BOLD signals on cortical 
surfaces.
"""

from __future__ import annotations
from typing import Union, TYPE_CHECKING
from warnings import warn
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import spmatrix, linalg
from neuromodes.basis import decompose

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike

def simulate_waves(
    emodes: ArrayLike,
    evals: ArrayLike,
    nt: int = None,
    ext_input: Union[ArrayLike, None] = None,
    bold_out: bool = False,
    dt: float = 1e-4,
    r: float = 17.4,
    gamma: float = 116.0,
    pde_method: str = "fourier",
    decomp_method: str = "project",
    mass: Union[spmatrix, ArrayLike, None] = None,
    speed_limits: Union[tuple[float, float], None] = (0, 150),
    scaled_hetero: Union[ArrayLike, None] = None,
    checks: bool = True,
    seed: Union[int, None] = None,
    cache_input: bool = False,
    **balloon_params
) -> NDArray:
    """
    Simulate neural activity or BOLD signals on the surface mesh using the eigenmode decomposition.
    The simulation uses a Neural Field Theory wave model and optionally the Balloon-Windkessel model
    for BOLD signal generation.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_verts is the number of vertices
        and n_modes is the number of eigenmodes.
    evals : array-like
        The eigenvalues array of shape (n_modes,).
    nt : int, optional
        Number of time points to simulate under white noise input. Note that either `nt` or
        `ext_input` must be provided. Default is `None`.
    bold_out : bool, optional
        If `True`, simulate BOLD signal using the balloon model. If `False`, simulate neural
        activity. Default is `False`.
    ext_input : array-like, optional
        External input array of shape (n_verts, n_timepoints). If `None`, white noise input is
        generated to simulate `nt` time points. Default is `None`.
    dt : float, optional
        Time step for simulation in seconds. Default is `1e-4`.
    r : float, optional
        Spatial length scale of wave propagation in millimeters. Default is `17.4`.
    gamma : float, optional
        Damping rate of wave propagation in seconds^-1. Default is `116.0`.
    pde_method : str, optional
        Method for solving the wave and balloon PDEs. Either `'fourier'` or `'ode'`. Default is
        `'fourier'`.
    decomp_method : str, optional
        The method used to eigendecompose `ext_input`, either `'project'` to project data into a
        mass-orthonormal space or `'regress'` for least-squares fitting. Note that the beta values
        from `'regress'` tend towards those from `'project'` when more modes are provided. Default
        is `'project'`.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is
        `'project'`. If using `EigenSolver`, provide its `self.mass`. Default is `None`.
    speed_limits : tuple, optional
        If any wave speeds are outside this range (in m/s), a warning is raised. If `None`, no
        warning is raised. Default is `(0, 150)`.
    scaled_hetero : array-like, optional
        Scaled heterogeneity map of shape (n_verts,), used only to check wave speeds (see
        `speed_limits` above). If not provided, wave speed is assumed to be spatially uniform. To
        scale a heterogeneity map, use the `eigen.scale_hetero` function.
        Default is `None`.
    checks : bool, optional
        Whether to check if `emodes` are mass-orthonormal before using the `'project'` method for
        decomposition. Default is `True`.
    seed : int, optional
        Random seed for generating external input. Default is `None`.
    cache_input : bool, optional
        If `True` and `ext_input` is `None`, cache the generated random input to avoid
        recomputation for the same values of `nt`, `seed`, and number of rows (vertices) in
        `emodes`. Inputs are cached in the directory specified by the `CACHE_DIR` environment
        variable. If not set, the user's home directory is chosen. This requires the `joblib`
        package to be installed. Default is `False`.
    **balloon_params
        Optional balloon model parameters to override defaults (e.g., `rho`, `k1`). See
        `get_balloon_params()` for available parameters. Only used when `bold_out=True`.

    Returns
    -------
    np.ndarray
        Simulated neural activity or BOLD signal of shape (n_verts, n_timepoints).

    Raises
    ------
    ValueError
        If `emodes` does not have shape (n_verts, n_modes), where n_verts ≥ n_modes.
    ValueError
        If `evals` does not have shape (n_modes,).
    ValueError
        If `r`, `gamma`, or `dt` is not positive.
    ValueError
        If `nt` is not a positive integer.
    ValueError
        If `speed_limits` is not a tuple (min_speed, max_speed), where 0 ≤ min_speed < max_speed.
    ValueError
        If `nt` is not provided when `ext_input` is `None`.
    ValueError
        If `pde_method` is not `'fourier'` or `'ode'`.
    RuntimeError
        If the ODE solver fails when using `pde_method='ode'` and `bold_out=True`.

    Notes
    -----
    Since the simulation begins at rest, consider discarding the first ~50 seconds to allow the
    system to reach a steady state.
    """
    # Format / validate arguments
    r = float(r)
    gamma = float(gamma)
    
    if checks:
        emodes = np.asarray(emodes) # chkfinite in decompose
        evals = np.asarray_chkfinite(evals)
        if emodes.ndim != 2 or emodes.shape[0] < emodes.shape[1]:
            raise ValueError("`emodes` must have shape (n_verts, n_modes), where n_verts ≥ n_modes.")
    n_verts, n_modes = emodes.shape
    if checks and evals.shape != (n_modes,):
        raise ValueError("`evals` must have shape (n_modes,), matching the number of columns in "
                         f"`emodes` ({n_modes}).")
    if r <= 0:
        raise ValueError("Parameter `r` must be positive.")
    if gamma <= 0:
        raise ValueError("Parameter `gamma` must be positive.")
    if dt <= 0:
        raise ValueError("`dt` must be positive.")
    if nt is not None and (not isinstance(nt, int) or nt <= 0):
        raise ValueError("`nt` must be `None` or a positive integer.")
    if speed_limits is not None:
        if (not isinstance(speed_limits, tuple) or not len(speed_limits) == 2
            or speed_limits[0] < 0 or speed_limits[0] >= speed_limits[1]):
            raise ValueError("`speed_limits` must be a tuple of (min_speed, max_speed), where "
                             "0 ≤ min_speed < max_speed.")
        speed = calc_wave_speed(r, gamma, scaled_hetero=scaled_hetero)
        min_speed, max_speed = np.min(speed), np.max(speed)
        if min_speed < speed_limits[0] or max_speed > speed_limits[1]:
            calc_str = min_speed if min_speed == max_speed else f"{min_speed:.1f}-{max_speed:.1f}"
            warn("The combination of `r`, `gamma`, and `scaled_hetero` leads to wave speeds "
                 f"outside the range of {speed_limits[0]}-{speed_limits[1]} m/s (calculated "
                 f"{calc_str} m/s). Consider changing these parameters to ensure physiologically "
                 "plausible wave speeds, or adjust `speed_limits`.")
    if len(balloon_params) > 0 and not bold_out:
        warn("Balloon model parameters will be ignored as `bold_out` is `False`.")
    if pde_method not in ['fourier', 'ode']:
        raise ValueError(f"Invalid PDE method '{pde_method}'; must be 'fourier' or 'ode'.")

    if ext_input is not None:
        ext_input = np.asarray_chkfinite(ext_input)
        if nt is not None:
            warn("`nt` is ignored when `ext_input` is provided.")
        if seed is not None:
            warn("`seed` is ignored when `ext_input` is provided.")
        if cache_input:
            warn("`cache_input` is ignored when `ext_input` is provided.")
        nt = ext_input.shape[1]
    else:
        if nt is None:
            raise ValueError("`nt` must be provided when `ext_input` is `None`.")
        if cache_input:
            if seed is None:
                warn("`cache_input` is ignored when `seed` is None.")
            else:
                from neuromodes.io import _set_cache

                memory = _set_cache()
                gen_input = memory.cache(_gen_noise)
        else:
            gen_input = _gen_noise
        
        ext_input = gen_input((n_verts, nt), seed)

    # Eigendecompose external input to get modal coefficients over time
    input_coeffs = decompose(ext_input, emodes, method=decomp_method, mass=mass, checks=checks)

    # Compute activity timeseries for each mode
    mode_coeffs = (_model_wave_fourier(input_coeffs, dt, r, gamma, evals)
                   if pde_method == "fourier" else
                   _model_wave_ode(input_coeffs, dt, r, gamma, evals))

    if bold_out:
        # Get parameters for Balloon-Windkessel model
        all_balloon_params = get_balloon_params(**balloon_params)

        # Apply model to each mode's activity timeseries
        mode_coeffs = (_model_balloon_fourier(mode_coeffs, dt, all_balloon_params)
                       if pde_method == "fourier" else
                       _model_balloon_ode(mode_coeffs, dt, all_balloon_params))

    # Transform timeseries from modal coefficients back to vertex space
    return emodes @ mode_coeffs

def calc_wave_speed(
    r: float,
    gamma: float,
    scaled_hetero: Union[ArrayLike, None] = None
) -> Union[float, NDArray]:
    """
    Calculate wave speed based on the two parameters of the wave model. If a scaled
    heterogeneity map is provided, wave speeds are calculated for each cortical vertex.
    
    Parameters
    ----------
    r : float
        Axonal length scale for wave propagation in millimeters.
    gamma : float
        Damping parameter for wave propagation in seconds^-1.
    scaled_hetero : array-like, optional
        Scaled heterogeneity map of shape (n_verts,). If `None`, wave speed is assumed to be
        spatially uniform. To scale a heterogeneity map, use the `eigen.scale_hetero` function.
        Default is `None`.
    
    Returns
    -------
    float or np.ndarray
        Wave speed across the whole surface, or at each cortical vertex if `scaled_hetero` is
        provided.
    """
    speed = (r / 1000) * gamma # Convert r to meters
    if scaled_hetero is not None:
        speed *= np.sqrt(scaled_hetero)

    return speed

def _gen_noise(size, seed):
    return np.random.default_rng(seed).standard_normal(size=size)

def _model_wave_fourier(
    input_coeffs: NDArray,
    dt: float,
    r: float,
    gamma: float,
    evals: NDArray
) -> NDArray:
    """
    Simulates the time evolution of wave models for all modes using a frequency-domain approach.
    This function applies a Fourier transform to the input mode coefficients, computes the system's
    frequency response, and then applies an inverse Fourier transform to obtain the time-domain
    response of each mode.

    Parameters
    ----------
    input_coeffs : np.ndarray
        Array of mode coefficients at each time representing the input signals to the model, with
        shape (n_modes, nt).
    dt : float
        Time step for the simulation in seconds.
    r : float
        Spatial length scale of wave propagation in millimeters.
    gamma : float
        Damping rate of wave propagation in seconds^-1.
    evals : np.ndarray
        The eigenvalues associated with each mode, with shape (n_modes,).

    Returns
    -------
    out : ndarray
        The real part of the time-domain response of all modes at the specified time points, with
        shape (n_modes, nt).
    
    Notes
    -----
    This function uses a frequency-domain method to simulate the damped wave response of a causal
    input. To ensure causality (i.e., the input is zero for t < 0), the input is zero-padded on the
    negative time axis and transformed using `ifft`, which mimics the forward Fourier transform of a
    causal signal. The system's frequency response (transfer function) is then applied, and `fft` is
    used to return to the time domain. This approach is standard for simulating linear
    time-invariant causal systems and is equivalent to convolution with a Green's function.

    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take ifft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use fft to return to the time domain (with appropriate shifts)
    """
    n_modes, nt = input_coeffs.shape

    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    input_coeffs_padded = np.concatenate([np.zeros((n_modes, nt)), input_coeffs], axis=1)

    # Apply inverse Fourier transform to get frequency-domain representation of the causal signal.
    input_coeffs_f = np.fft.fftshift(np.fft.ifft(input_coeffs_padded, axis=1), axes=1)

    # Frequencies for full signal
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(2*nt, d=dt))

    # Compute transfer function and apply it to frequency-domain input
    H = gamma**2 / (-omega**2 - 2j * omega * gamma + gamma**2 * (1 + r**2 * evals[:, np.newaxis]))
    out_fft = H * input_coeffs_f

    # Inverse transform to time domain, implemented as forward FFT for causality
    out_full = np.real(np.fft.fft(np.fft.ifftshift(out_fft, axes=1), axis=1))

    # Return only the non-negative time part (t >= 0)
    return out_full[:, nt:]

def _model_wave_ode(
    input_coeffs: NDArray,
    dt: float,
    r: float,
    gamma: float,
    evals: NDArray
) -> NDArray:
    """
    Solves the damped wave ODE for all eigenmodes.

    Parameters
    ----------
    input_coeffs : np.ndarray
        Input drive to the system with shape (n_modes, nt) (written as `qj` in equation below).
    dt : float
        Time step for the simulation in seconds.
    gamma : float
        Damping coefficient seconds^-1.
    r : float
        Spatial length scale in millimeters.
    evals : np.ndarray
        Eigenvalues for each mode with shape (n_modes,) (written as `lambdaj` in equation below).

    Returns
    -------
    np.ndarray
        Time evolution of phi_j(t), solution to the wave equation, with shape (n_modes, nt).
    
    Notes
    -----
    The equation is derived from the damped wave equation:
    d^2 phi_j / dt^2 + 2 * gamma * d phi_j / dt + gamma^2 * (1 + r^2 * lambdaj) * phi_j = gamma^2 * qj
    
    Rearranging gives us the first-order system
        dx1/dt = x2
        dx2/dt = -2 * gamma * x2 - gamma^2 * (1 + r^2 * lambdaj) * x1 + gamma^2 * qval
    """
    n_modes, nt = input_coeffs.shape
    t = np.linspace(0, dt * (nt - 1), nt)
    
    # Simulate wave equation for each mode
    mode_coeffs = np.zeros((n_modes, nt))
    for j in range(n_modes):
        def wave_odes_j(t_, y):
            """Returns the wave ODEs for mode j."""
            x1, x2 = y

            # Interpolate input coefficient at time t_
            qval = np.interp(t_, t, input_coeffs[j, :])
            if isinstance(qval, np.ndarray):
                qval = qval.item()

            # Set expressions for time derivatives
            dx1dt = x2
            dx2dt = -2 * gamma * x2 - gamma**2 * (1 + r**2 * evals[j]) * x1 + gamma**2 * qval
            return [dx1dt, dx2dt]

        # Call ODE solver
        sol = solve_ivp(
            wave_odes_j,
            t_span=(t[0], t[-1]),
            y0=[0.0, 0.0],  # Initial condition: phi_j(0) = 0, dphi_j/dt(0) = 0
            t_eval=t,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )

        mode_coeffs[j, :] = sol.y[0]  # Store phi_j(t)

    return mode_coeffs

def get_balloon_params(**overrides) -> dict:
    """
    Return balloon model parameters with optional overrides.
    
    Parameters
    ----------
    **overrides
        Balloon model parameters to override default values. Must be positive.
        
    Returns
    -------
    dict
        Balloon model parameters.
        - `kappa`: Signal decay rate [s^-1]. Default is `0.65`.
        - `gamma_h`: Rate of elimination [s^-1]. Default is `0.41`.
        - `tau`: Hemodynamic transit time [s]. Default is `0.98`.
        - `alpha`: Grubb's exponent [unitless]. Default is `0.32`.
        - `rho`: Resting oxygen extraction fraction [unitless]. Default is `0.34`.
        - `V_0`: Resting blood volume fraction [unitless]. Default is `0.02`.
        - `w_f`: Frequency of blood flow response [rad/s]. Default is `0.56`.
        - `k1`, `k2`, `k3`: Coefficients for BOLD signal equation [unitless]. Defaults are `3.72`,
        `0.527`, and `0.48`, respectively.
    
    Raises
    ------
    ValueError
        If any provided balloon model parameter name is invalid.
    ValueError
        If any provided balloon model parameter is non-positive or non-finite.
    """
    
    # Get default values
    params = {
        'kappa': 0.65,
        'gamma_h': 0.41,
        'tau': 0.98,
        'alpha': 0.32,
        'rho': 0.34,
        'V_0': 0.02,
        'w_f': 0.56,
        'k1': 3.72,
        'k2': 0.527,
        'k3': 0.48
    }

    # Validate and apply overrides
    for param, value in overrides.items():
        if param not in params:
            raise ValueError(f"Invalid Balloon model parameter '{param}'.")
        if value <= 0 or np.isnan(value) or np.isinf(value):
            raise ValueError("All Balloon model parameters must be positive and finite (received "
                             f"{param}={value}).")

    params.update(overrides)

    return params

def _model_balloon_fourier(
    activity_coeffs: NDArray,
    dt: float,
    params: dict,
) -> NDArray:
    """
    Simulates the hemodynamic response of all modes using the balloon model in the frequency domain. 
    This function computes the balloon model's frequency response and applies it to the input mode 
    coefficients via Fourier transforms, returning the modeled hemodynamic response over time.

    Parameters
    ----------
    activity_coeffs : np.ndarray
        Array of mode coefficients representing the input signals to the model, with shape (n_modes,
        nt).
    dt : float
        Time step in seconds.
    params : dict
        Balloon model parameters. See the `get_balloon_params` function for default parameters.

    Returns
    -------
    np.ndarray
        The real part of the time-domain response of all modes at the specified time points, with
        shape (n_modes, nt).

    Notes
    -----
    This function uses a frequency-domain method to simulate the damped wave response of a causal 
    input. To ensure causality (i.e., the input is zero for t < 0), the input is zero-padded on the 
    negative time axis and transformed using `ifft`, which mimics the forward Fourier transform of a 
    causal signal. The system's frequency response (transfer function) is then applied, and `fft` is 
    used to return to the time domain. This approach is standard for simulating linear 
    time-invariant causal systems and is equivalent to convolution with a Green's function.

    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take ifft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use fft to return to the time domain (with appropriate shifts)
    """
    # Extract parameters
    kappa = params['kappa']
    tau = params['tau']
    alpha = params['alpha']
    w_f = params['w_f']
    V_0 = params['V_0']
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    rho = params['rho']

    n_modes, nt = activity_coeffs.shape

    # Calculate balloon model frequency response
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(2*nt, d=dt))
    beta = (rho + (1 - rho) * np.log(1 - rho)) / rho
    phi_hat_Fz = 1 / (-(omega + 1j * 0.5 * kappa) ** 2 + w_f ** 2)
    phi_hat_yF = V_0 * (alpha * (k2 + k3) * (1 - 1j * tau * omega) 
                                - (k1 + k2) * (alpha + beta - 1 - 1j * tau * alpha * beta * omega)
                                ) / ((1 - 1j * tau * omega) * (1 - 1j * tau * alpha * omega))
    balloon_freq_response = phi_hat_yF * phi_hat_Fz

    # Zero-pad input at t < 0 for causality
    activity_coeffs_padded = np.concatenate([np.zeros((n_modes, nt)), activity_coeffs], axis=1)

    # Apply Fourier transform (implemented as inverse FFT for causality)
    activity_coeffs_f = np.fft.fftshift(np.fft.ifft(activity_coeffs_padded, axis=1), axes=1)

    # Apply frequency response (broadcast along time axis)
    out_fft = balloon_freq_response[np.newaxis, :] * activity_coeffs_f

    # Inverse transform back to timeseries (implemented as forward FFT for causality)
    out_full = np.real(np.fft.fft(np.fft.ifftshift(out_fft, axes=1), axis=1))

    # Remove zero padding
    return out_full[:, nt:]

def _model_balloon_ode(
    activity_coeffs: NDArray,
    dt: float,
    params: dict
) -> NDArray:
    """
    Simulates the hemodynamic response of all modes using the balloon model in the time domain (ODE 
    approach). This function numerically integrates the balloon model ODEs for each input mode 
    time course.

    Parameters
    ----------
    activity_coeffs : np.ndarray
        Array of mode coefficients representing the input signals to the model, with shape (n_modes,
        nt).
    dt : float
        Time step for the simulation in seconds.
    params: dict
        Balloon model parameters. See the `get_balloon_params` function for default parameters.

    Returns
    -------
    np.ndarray
        The BOLD signal time course for all modes at the specified time points, with shape (n_modes,
        nt).

    Raises
    ------
    RuntimeError
        If the ODE solver fails.
    """    
    # Extract base parameters
    kappa = params['kappa']
    gamma_h = params['gamma_h']
    tau = params['tau']
    alpha = params['alpha']
    V_0 = params['V_0']
    rho = params['rho']
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']

    n_modes, nt = activity_coeffs.shape
    t = np.linspace(0, dt * (nt - 1), nt)

    # Simulate balloon model for each mode
    bold_coeffs = np.zeros((n_modes, nt))
    for j in range(n_modes):
        def balloon_odes_j(t_, y):
            """Returns the balloon model ODEs for mode j."""
            z, f, v, q = y

            # Interpolate input coefficient at time t_
            N = np.interp(t_, t, activity_coeffs[j])

            # Set expressions for time derivatives
            dzdt = N - kappa * z - gamma_h * (f - 1)
            dfdt = z
            dvdt = (f - v ** (1 / alpha)) / tau
            dqdt = (f * (1 - (1 - rho) ** (1 / f)) / rho - q * v ** (1 / alpha - 1)) / tau
            return [dzdt, dfdt, dvdt, dqdt]

        # Call ODE solver
        sol = solve_ivp(
            balloon_odes_j,
            t_span=(t[0], t[-1]),
            y0=[0.0, 1.0, 1.0, 1.0], # Initial condition for [z, f, v, q]
            t_eval=t,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )

        if not sol.success:
            raise RuntimeError("Balloon model ODE solver failed. Try using `pde_method='fourier'` "
                               "or a smaller `dt` timestep without altering balloon model "
                               f"parameters. `scipy.integrate.solve_ivp` message: {sol.message}")

        # Apply standard BOLD signal equation
        _, _, v, q = sol.y
        bold_coeffs[j, :] = V_0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))

    return bold_coeffs

def _simulate_waves_fem(
    mass: spmatrix,
    stiffness: spmatrix,
    nt: int = None,
    ext_input: ArrayLike = None,
    dt: float = 1e-4,
    r: float = 17.4,
    gamma: float = 116.0,
    speed_limits: Union[tuple[float, float], None] = (0, 150),
    scaled_hetero: Union[ArrayLike, None] = None,
    seed: int = None,
    cache_input: bool = False
) -> NDArray:
    """
    Full FEM version of `simulate_waves(..., bold_out=False)`, for validating the eigenmode
    expansion approach.
    """
    # Format / validate arguments
    r = float(r)
    gamma = float(gamma)
    
    if not isinstance(mass, spmatrix) or not isinstance(stiffness, spmatrix):
        raise ValueError("`mass` and `stiffness` must be scipy sparse matrices.")
    n_verts = mass.get_shape()[0]
    if mass.get_shape() != (n_verts, n_verts) or stiffness.get_shape() != (n_verts, n_verts):
        raise ValueError("`mass` and `stiffness` must have shape (n_verts, n_verts).")
    if r <= 0:
        raise ValueError("Parameter `r` must be positive.")
    if gamma <= 0:
        raise ValueError("Parameter `gamma` must be positive.")
    if dt <= 0:
        raise ValueError("`dt` must be positive.")
    if nt is not None and (not isinstance(nt, int) or nt <= 0):
        raise ValueError("`nt` must be `None` or a positive integer.")
    if speed_limits is not None:
        if (not isinstance(speed_limits, tuple) or not len(speed_limits) == 2
            or speed_limits[0] < 0 or speed_limits[0] >= speed_limits[1]):
            raise ValueError("`speed_limits` must be a tuple of (min_speed, max_speed), where "
                             "0 ≤ min_speed < max_speed.")
        speed = calc_wave_speed(r, gamma, scaled_hetero=scaled_hetero)
        min_speed, max_speed = np.min(speed), np.max(speed)
        if min_speed < speed_limits[0] or max_speed > speed_limits[1]:
            calc_str = min_speed if min_speed == max_speed else f"{min_speed:.1f}-{max_speed:.1f}"
            warn("The combination of `r`, `gamma`, and `scaled_hetero` leads to wave speeds "
                 f"outside the range of {speed_limits[0]}-{speed_limits[1]} m/s (calculated "
                 f"{calc_str} m/s). Consider changing these parameters to ensure physiologically "
                 "plausible wave speeds, or adjust `speed_limits`.")

    if ext_input is not None:
        ext_input = np.asarray_chkfinite(ext_input)
        if nt is not None:
            warn("`nt` is ignored when `ext_input` is provided.")
        if seed is not None:
            warn("`seed` is ignored when `ext_input` is provided.")
        if cache_input:
            warn("`cache_input` is ignored when `ext_input` is provided.")
        nt = ext_input.shape[1]
    else:
        if nt is None:
            raise ValueError("`nt` must be provided when `ext_input` is `None`.")
        if cache_input:
            if seed is None:
                warn("`cache_input` is ignored when `seed` is None.")
            else:
                from neuromodes.io import _set_cache

                memory = _set_cache()
                gen_input = memory.cache(_gen_noise)
        else:
            gen_input = _gen_noise
        
        ext_input = gen_input((n_verts, nt), seed)

    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    ext_input_padded = np.concatenate([np.zeros((n_verts, nt)), ext_input], axis=1)

    # Apply inverse Fourier transform to get frequency-domain representation of the causal signal.
    ext_input_freqs = np.fft.fftshift(np.fft.ifft(ext_input_padded, axis=1), axes=1)
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(2 * nt, dt))

    # Treat noise input as a continuous field
    mass_ext_input_freqs = mass @ ext_input_freqs

    # Compute response at each frequency
    phi_freqs = np.zeros_like(ext_input_freqs, dtype=complex)
    for k, w in enumerate(omega):
        operator = mass * (-w**2 / gamma**2 - 2j * w / gamma + 1) + stiffness * r**2

        # Sparse LU decomposition and solve PDE
        phi_freqs[:, k] = linalg.splu(operator).solve(mass_ext_input_freqs[:, k])

    # Inverse transform to time domain, implemented as forward FFT for causality
    phi = np.real(np.fft.fft(np.fft.ifftshift(phi_freqs, axes=1), axis=1))

    # Return only the non-negative time part (t >= 0)
    return phi[:, nt:]