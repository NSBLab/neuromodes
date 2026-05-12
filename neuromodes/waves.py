"""
Module for using neural field theory to simulate neural activity and BOLD signals on cortical 
surfaces.
"""

from __future__ import annotations
from typing import Literal, TYPE_CHECKING, cast
from warnings import warn
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.sparse import linalg
from neuromodes.eigen import EigenData
from neuromodes.basis import decompose

if TYPE_CHECKING:
    from typing import Literal
    from numpy import floating
    from scipy.sparse import csc_matrix
    from neuromodes.eigen import _CheckKind
    from neuromodes.basis import _DecompositionKind
    _PDEKind = Literal["fourier", "ode", "fem"]

def sim_nft_waves(
    emodes: NDArray,
    evals: NDArray,
    nt: int | None = None,
    ext_input: NDArray | None = None,
    dt: float = 1e-4,
    r: float = 17.4,
    gamma: float = 116.0,
    pde_method: _PDEKind = "fourier",
    mass: csc_matrix | None = None,
    stiffness: csc_matrix | None = None, # only used for FEM
    speed_limits: tuple[float, float] | None = (0, 150),
    scaled_hetero: NDArray | None = None,
    checks: _CheckKind = True,
    seed: int | None = None,
    cache_input: bool = False,
    n_jobs: int = 1, # only used for FEM
    verbose: int = 0 # only used for FEM
) -> NDArray:
    """
    Simulate neural activity using a Neural Field Theory wave model [1]_ [2]_ [3]_.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape ``(n_verts, n_modes)``, where ``n_verts`` is the number of
        vertices and ``n_modes`` is the number of eigenmodes.
    evals : array-like
        The eigenvalues array of shape ``(n_modes,)``.
    nt : int, optional
        Number of time points to simulate under white noise input. Note that either ``nt`` or
        ``ext_input`` must be provided. Default is ``None``.
    ext_input : array-like, optional
        External input array of shape ``(n_verts, n_timepoints)``. If ``None``, white noise input is
        generated to simulate ``nt`` time points. Default is ``None``.
    dt : float, optional
        Time step for simulation in seconds. Default is ``1e-4``.
    r : float, optional
        Spatial length scale of wave propagation in millimeters. Default is ``17.4``.
    gamma : float, optional
        Damping rate of wave propagation in seconds^(-1). Default is ``116.0``.
    pde_method : str, optional
        Method for solving the wave PDEs. Either ``'fourier'`` or ``'ode'``. Default is
        ``'fourier'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)`` used for the decomposition when method is
        ``'project'``. Default is ``None``.
    speed_limits : tuple, optional
        If any wave speeds are outside this range (in meters per second), a warning is raised. If
        ``None``, no warning is raised. Default is ``(0, 150)``.
    scaled_hetero : array-like, optional
        Scaled heterogeneity map of shape ``(n_verts,)``, used only to check wave speeds (see
        ``speed_limits`` above). If not provided, wave speed is assumed to be spatially uniform. To
        scale a heterogeneity map, use :func:`eigen.scale_hetero`.
        Default is ``None``.
    checks : bool, optional
        Whether to check if ``emodes`` are mass-orthonormal before using the ``'project'`` method
        for decomposition. Default is ``True``.
    seed : int, optional
        Random seed for generating external input. Default is ``None``.
    cache_input : bool, optional
        If ``True`` and ``ext_input`` is ``None``, cache the generated random input to avoid
        recomputation for the same values of ``nt``, ``seed``, and number of rows (vertices) in
        ``emodes``. Inputs are cached in the directory specified by the ``CACHE_DIR`` environment
        variable. If not set, the user's home directory is chosen. This requires the ``joblib``
        package to be installed. Default is ``False``.

    Returns
    -------
    np.ndarray
        Simulated neural activity or BOLD signal of shape ``(n_verts, n_timepoints)``.

    Raises
    ------
    ValueError
        If ``r``, ``gamma``, or ``dt`` is not positive.
    ValueError
        If ``nt`` is not a positive integer.
    ValueError
        If ``speed_limits`` is not a tuple ``(min_speed, max_speed)``, where ``0 ≤ min_speed <
        max_speed``.
    ValueError
        If neither ``nt`` nor ``ext_input`` are provided.
    ValueError
        If ``pde_method`` is not ``'fourier'`` or ``'ode'``.

    Notes
    -----
    Prior works have treated ``r`` as a free parameter to fit empirical data [1]_ [2]_, with the
    default value reflecting an optimal fit to human resting-state functional MRI data [2]_.
    Consider adjusting this parameter, as its optimum can vary across analyses (e.g., different
    surfaces, heterogeneous modes, parcellated timeseries, empirical data, fitting metrics, etc.).

    Since the simulation begins at rest, consider discarding the first ~50 seconds to allow the
    system to reach a steady state.

    While the wave model can be run using non-cortical modes, users should consider whether this is
    theoretically sensible and physiologically plausible.

    References
    ----------
    ..  [1] Barnes, V., et al. (2026). Regional heterogeneity shapes macroscopic wave dynamics of
        the human and non-human primate cortex. bioRxiv. https://doi.org/10.64898/2026.01.22.701178
    ..  [2] Pang, J. C., et al. (2023). Geometric constraints on human brain function. Nature.
        https://doi.org/10.1038/s41586-023-06098-1
    ..  [3] Robinson, P. A., et al. (1997). Propagation and stability of waves of electrical
        activity in the cerebral cortex. Physical Review E. https://doi.org/10.1103/physreve.56.826
    """
    # Format / validate arguments
    if checks is not False:
        ved = EigenData(
            emodes=emodes, evals=evals, mass=mass, stiffness=stiffness, 
            scaled_hetero=scaled_hetero, data=ext_input, checks=checks
            )
        emodes, evals, mass, stiffness, ext_input, scaled_hetero = \
            ved.emodes, ved.evals, ved.mass, ved.stiffness, ved.data, ved.scaled_hetero
        
    if emodes is not None: 
        n_verts = emodes.shape[0]
    elif stiffness is not None:
        n_verts = stiffness.shape[0]
    elif pde_method == 'fem':
        raise ValueError(f"mass and stiffness matrices must be provided for {pde_method} method.")
    else: 
        raise ValueError(f"emodes must be provided for {pde_method} method.")
        
    r = float(r)
    gamma = float(gamma)
    if r <= 0:
        raise ValueError("Parameter r must be positive.")
    if gamma <= 0:
        raise ValueError("Parameter gamma must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if nt is not None and (not isinstance(nt, int) or nt <= 0):
        raise ValueError("nt must be None or a positive integer.")
    if speed_limits is not None:
        if (not isinstance(speed_limits, tuple) or not len(speed_limits) == 2
            or speed_limits[0] < 0 or speed_limits[0] >= speed_limits[1]):
            raise ValueError("speed_limits must be a tuple of (min_speed, max_speed), where "
                             "0 ≤ min_speed < max_speed.")
        speed = calc_wave_speed(r, gamma, scaled_hetero=scaled_hetero)
        min_speed, max_speed = np.min(speed), np.max(speed)
        if min_speed < speed_limits[0] or max_speed > speed_limits[1]:
            calc_str = min_speed if min_speed == max_speed else f"{min_speed:.1f}-{max_speed:.1f}"
            warn("The combination of r, gamma, and scaled_hetero leads to wave speeds "
                 f"outside the range of {speed_limits[0]}-{speed_limits[1]} m/s (calculated "
                 f"{calc_str} m/s). Consider changing these parameters to ensure physiologically "
                 "plausible wave speeds, or adjust speed_limits.")
    if pde_method not in ['fourier', 'ode', 'fem']:
        raise ValueError(f"Invalid PDE method '{pde_method}'; must be 'fourier', 'ode', or 'fem'.")

    if ext_input is not None:
        if nt is not None:
            warn("nt is ignored when ext_input is provided.")
        if seed is not None:
            warn("seed is ignored when ext_input is provided.")
        if cache_input:
            warn("cache_input is ignored when ext_input is provided.")
        if np.isnan(ext_input).any():
            raise ValueError("ext_input contains NaN values, which are not allowed.")
        nt = ext_input.shape[1]
    elif nt is not None:
        if cache_input and seed is not None:
            from neuromodes.io import _cache_output
            noise_func = _cache_output(_gen_noise)
        else:
            if cache_input and seed is None:
                warn("cache_input is ignored when seed is None.")
            noise_func = _gen_noise
        ext_input = np.asarray(noise_func(n_verts, nt, seed=seed))
    else: # not the nicest, but it makes pyright the happiest
        raise ValueError("Either nt or ext_input must be provided.")

    # Non-modal FEM implementation
    if pde_method == 'fem':
        if mass is None or stiffness is None:
            raise ValueError("Mass and stiffness matrices must be provided for FEM method.")
        return _model_wave_fem(ext_input, dt=dt, r=r, gamma=gamma, mass=mass, stiffness=stiffness,
                               n_jobs=n_jobs, verbose=verbose)
    
    # Standard modal implementation: decompose input and reconstruct output
    input_coeffs = decompose(ext_input, emodes, mass=mass, checks=False)
    _model_wave = _model_wave_fourier if pde_method == 'fourier' else _model_wave_ode
    activity_coeffs = _model_wave(input_coeffs, dt, r, gamma, evals)

    return emodes @ activity_coeffs

def balloon_model(
    activity: NDArray[floating],
    dt: float,
    emodes: NDArray[floating],
    pde_method: _PDEKind = "fourier",
    mass: csc_matrix | None = None,
    checks: _CheckKind = True,
    **params
) -> NDArray[floating]:
    """
    Transform simulated activity to blood oxygen level-dependent (BOLD) signal using the
    Balloon-Windkessel model [1]_ [2]_.
    
    Parameters
    ----------
    activity : array-like
        Simulated neural activity in vertex space of shape ``(n_verts, n_timepoints)``.
    emodes : array-like
        The eigenmodes array of shape ``(n_verts, n_modes)``, where ``n_verts`` is the number of
        vertices and ``n_modes`` is the number of eigenmodes.
    dt : float, optional
        Time step of simulated activity in seconds.
    pde_method : str, optional
        Method for solving the balloon PDEs. Either ``'fourier'`` or ``'ode'``. Default is
        ``'fourier'``.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is
        ``'project'``. Default is ``None``.
    checks : bool, optional
        Whether to perform checks on the input arrays. Default is ``True``.
    **params
        Optional balloon model parameters to override defaults (e.g., ``rho``, ``k1``). See
        :func:`_model_balloon_fourier` or :func:`_model_balloon_ode` for available parameters.

    Returns
    -------
    ndarray
        Simulated BOLD signal in vertex space of shape ``(n_verts, n_timepoints)``.

    Raises
    ------
    ValueError
        If ``dt`` is not positive.
    ValueError
        If ``pde_method`` is not ``'fourier'`` or ``'ode'``.

    References
    ----------
    ..  [1] Buxton, R. B., et al. (1998). Dynamics of blood flow and oxygenation changes during
        brain activation: The balloon model. Magnetic Resonance in Med.
        https://doi.org/10.1002/mrm.1910390602
    ..  [2] Stephan, K. E., et al. (2007). Comparing hemodynamic models with DCM. NeuroImage.
        https://doi.org/10.1016/j.neuroimage.2007.07.040
    """
    # Format / validate arguments
    if checks is not False:
        ved = EigenData(emodes=emodes, mass=mass, data=activity, checks=checks)
        emodes, mass, activity = ved.emodes, ved.mass, ved.data

    if np.isnan(activity).any():
        raise ValueError("activity contains NaN values, which are not allowed.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if pde_method not in ['fourier', 'ode']:
        raise ValueError(f"Invalid PDE method '{pde_method}'; must be 'fourier' or 'ode'.")
    for param_name, param_value in params.items():
        if not isinstance(param_value, (int, float)) or param_value <= 0:
            raise ValueError(f"Parameter '{param_name}' must be a positive number.")

    # Eigendecompose activity to get modal coefficients over time
    activity_coeffs = decompose(activity, emodes, mass=mass, checks=False)

    # Apply model to each mode's activity timeseries
    _model_balloon = _model_balloon_fourier if pde_method == 'fourier' else _model_balloon_ode
    bold_coeffs = _model_balloon(activity_coeffs, dt, **params)

    # Transform timeseries from modal coefficients back to vertex space
    return emodes @ bold_coeffs

def calc_wave_speed(
    r: float,
    gamma: float,
    scaled_hetero: NDArray[floating] | None = None
) -> float | NDArray[floating]:
    """
    Calculate wave speed (m/s) based on the two parameters of the wave model. If a scaled
    heterogeneity map is provided, wave speeds are calculated for each cortical vertex.
    
    Parameters
    ----------
    r : float
        Axonal length scale for wave propagation in millimeters.
    gamma : float
        Damping parameter for wave propagation in seconds^-1.
    scaled_hetero : array-like, optional
        Scaled heterogeneity map of shape (n_verts,). If ``None``, wave speed is assumed to be
        spatially uniform. To scale a heterogeneity map, use :func:eigen.scale_hetero. Default is
        ``None``.
    
    Returns
    -------
    float or np.ndarray
        Wave speed across the whole cortex in meters per second, or at each vertex if
        ``scaled_hetero`` is provided.
    """
    speed = (r / 1000) * gamma # Convert r to meters
    if scaled_hetero is not None:
        speed *= np.sqrt(scaled_hetero)

    return speed

def _gen_noise(
    n_verts: int,
    nt: int,
    seed: int | None
) -> NDArray[floating]:
    """
    Generate reproducible white noise of shape ``(n_verts, nt)`` for a given ``seed``, derived from
    a standard normal distribution. The output is reproducible across nt (i.e.,
    ``_gen_noise(n_verts, nt, seed) == _gen_noise(n_verts, nt+k, seed)[:, :nt]``).

    Parameters
    ----------
    n_verts : int
        Number of vertices (rows) in the output noise array.
    nt : int
        Number of time points (columns) in the output noise array.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Gaussian white noise array of shape ``(n_verts, nt)``.
    """
    rng = np.random.default_rng(seed)
    # Generate in column-major order to ensure reproducibility across nt, then transpose
    return rng.standard_normal((nt, n_verts)).T

def _model_wave_fourier(
    input_coeffs: NDArray[floating],
    dt: float,
    r: float,
    gamma: float,
    evals: NDArray[floating]
) -> NDArray[floating]:
    """
    Simulates the time evolution of wave models for all modes using a frequency-domain approach.
    This function applies a Fourier transform to the input mode coefficients, computes the system's
    frequency response, and then applies an inverse Fourier transform to obtain the time-domain
    response of each mode.

    Parameters
    ----------
    input_coeffs : np.ndarray
        Array of mode coefficients at each time representing the input signals to the model, with
        shape ``(n_modes, nt)``.
    dt : float
        Time step for the simulation in seconds.
    r : float
        Spatial length scale of wave propagation in millimeters.
    gamma : float
        Damping rate of wave propagation in seconds^(-1).
    evals : np.ndarray
        The eigenvalues associated with each mode, with shape ``(n_modes,)``.

    Returns
    -------
    out : ndarray
        The real part of the time-domain response of all modes at the specified time points, with
        shape ``(n_modes, nt)``.
    
    Notes
    -----
    This function uses a frequency-domain method to simulate the damped wave response of a causal
    input. To ensure causality (i.e., the input is zero for t < 0), the input is zero-padded on the
    negative time axis and transformed using ``np.fft.ifft``, which mimics the forward Fourier
    transform of a causal signal. The system's frequency response (transfer function) is then
    applied, and ``np.fft.fft`` is used to return to the time domain. This approach is standard for
    simulating linear time-invariant causal systems and is equivalent to convolution with a Green's
    function.

    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take ifft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use fft to return to the time domain (with appropriate shifts)
    """
    nt = input_coeffs.shape[1]

    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    input_coeffs_padded = np.concatenate([np.zeros_like(input_coeffs), input_coeffs], axis=1)

    # Frequency-domain representation of the causal signal
    # Faster to use `rfft` here than `fftshift(ifft)` (original implementation)
    input_coeffs_f = np.fft.rfft(input_coeffs_padded, axis=1)

    # Frequencies for full signal
    omega = -2 * np.pi * np.fft.rfftfreq(2*nt, d=dt) # keep consistent with _model_balloon_fourier

    # Compute transfer function and apply it to frequency-domain input
    H = gamma**2 / ((-omega**2 - 2j * omega * gamma) + gamma**2 * (1 + r**2 * evals[:, np.newaxis]))
    out_fft = H * input_coeffs_f

    # Inverse transform to time domain (irfft is fast)
    out_full = np.fft.irfft(out_fft, n=2*nt, axis=1)

    # Return only the non-negative time part (t >= 0)
    return out_full[:, nt:]

def _model_wave_ode(
    input_coeffs: NDArray[floating],
    dt: float,
    r: float,
    gamma: float,
    evals: NDArray[floating]
) -> NDArray[floating]:
    """
    Solves the damped wave ODE for all eigenmodes.

    Parameters
    ----------
    input_coeffs : np.ndarray
        Input drive to the system with shape ``(n_modes, nt)`` (written as qj in equation below).
    dt : float
        Time step for the simulation in seconds.
    gamma : float
        Damping coefficient seconds^-1.
    r : float
        Spatial length scale in millimeters.
    evals : np.ndarray
        Eigenvalues for each mode with shape ``(n_modes,)`` (written as lambdaj in equation below).

    Returns
    -------
    np.ndarray
        Time evolution of phi_j(t), solution to the wave equation, with shape ``(n_modes, nt)``.
    
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
    mode_coeffs = np.empty_like(input_coeffs)
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

def _model_balloon_fourier(
    activity_coeffs: NDArray[floating],
    dt: float,
    kappa: float = 0.65,
    tau: float = 0.98,
    alpha: float = 0.32,
    rho: float = 0.34,
    V_0: float = 0.02,
    w_f: float = 0.56,
    k1: float = 3.72,
    k2: float = 0.527,
    k3: float = 0.48
) -> NDArray[floating]:
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
    kappa : float, optional
        Signal decay rate in seconds^-1. Default is ``0.65``.
    tau : float, optional
        Hemodynamic transit time in seconds. Default is ``0.98``.
    alpha : float, optional
        Grubb's exponent (unitless). Default is ``0.32``.
    rho : float, optional
        Resting oxygen extraction fraction (unitless). Default is ``0.34``.
    V_0 : float, optional
        Resting blood volume fraction (unitless). Default is ``0.02``.
    w_f : float, optional
        Frequency of blood flow response in radians per second. Default is ``0.56``.
    k1 : float, optional
        First coefficient in BOLD signal equation (unitless). Default is ``3.72``
    k2 : float, optional
        Second coefficient in BOLD signal equation (unitless). Default is ``0.527``.
    k3 : float, optional
        Third coefficient in BOLD signal equation (unitless). Default is ``0.48``.

    Returns
    -------
    np.ndarray
        The real part of the time-domain response of all modes at the specified time points, with
        shape (n_modes, nt).

    Notes
    -----
    This function uses a frequency-domain method to simulate the damped wave response of a causal 
    input. To ensure causality (i.e., the input is zero for t < 0), the input is zero-padded on the 
    negative time axis and transformed using ``np.fft.ifft``, which mimics the forward Fourier
    transform of a causal signal. The system's frequency response (transfer function) is then
    applied, and ``np.fft.fft`` is used to return to the time domain. This approach is standard for
    simulating linear time-invariant causal systems and is equivalent to convolution with a Green's
    function.

    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take ifft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use fft to return to the time domain (with appropriate shifts)
    """
    nt = activity_coeffs.shape[1]

    # Calculate balloon model frequency response
    omega = -2 * np.pi * np.fft.rfftfreq(2*nt, d=dt)
    beta = (rho + (1 - rho) * np.log(1 - rho)) / rho
    phi_hat_Fz = 1 / (-(omega + 1j * 0.5 * kappa) ** 2 + w_f ** 2)
    phi_hat_yF = V_0 * (alpha * (k2 + k3) * (1 - 1j * tau * omega) 
                                - (k1 + k2) * (alpha + beta - 1 - 1j * tau * alpha * beta * omega)
                                ) / ((1 - 1j * tau * omega) * (1 - 1j * tau * alpha * omega))
    balloon_freq_response = phi_hat_yF * phi_hat_Fz

    # Zero-pad input at t < 0 for causality
    activity_coeffs_padded = np.concatenate([np.zeros_like(activity_coeffs), activity_coeffs],
                                            axis=1)

    # Apply Fourier transform (implemented as rfft for speed)
    activity_coeffs_f = np.fft.rfft(activity_coeffs_padded, axis=1)

    # Apply frequency response (broadcast along time axis)
    out_fft = balloon_freq_response[np.newaxis, :] * activity_coeffs_f

    # Inverse transform back to timeseries (inverse of previous transform)
    out_full = np.fft.irfft(out_fft, n=2*nt, axis=1)

    # Remove zero padding
    return out_full[:, nt:]

def _model_balloon_ode(
    activity_coeffs: NDArray[floating],
    dt: float,
    kappa: float = 0.65,
    tau: float = 0.98,
    alpha: float = 0.32,
    rho: float = 0.34,
    V_0: float = 0.02,
    gamma_h: float = 0.41,
    k1: float = 3.72,
    k2: float = 0.527,
    k3: float = 0.48
) -> NDArray[floating]:
    """
    Simulates the hemodynamic response of all modes using the balloon model in the time domain (ODE 
    approach). This function numerically integrates the balloon model ODEs for each input mode 
    time course.

    Parameters
    ----------
    activity_coeffs : np.ndarray
        Array of mode coefficients representing the input signals to the model, with shape
        ``(n_modes, nt)``.
    dt : float
        Time step for the simulation in seconds.
    kappa : float, optional
        Signal decay rate in seconds^-1. Default is ``0.65``.
    tau : float, optional
        Hemodynamic transit time in seconds. Default is ``0.98``.
    alpha : float, optional
        Grubb's exponent (unitless). Default is ``0.32``.
    V_0 : float, optional
        Resting blood volume fraction (unitless). Default is ``0.02``.
    gamma_h : float, optional
        Hemodynamic gain (unitless). Default is ``0.41``.
    k1 : float, optional
        First coefficient in BOLD signal equation (unitless). Default is ``3.72``.
    k2 : float, optional
        Second coefficient in BOLD signal equation (unitless). Default is ``0.527``.
    k3 : float, optional
        Third coefficient in BOLD signal equation (unitless). Default is ``0.48``.

    Returns
    -------
    np.ndarray
        The BOLD signal time course for all modes at the specified time points, with shape
        ``(n_modes, nt)``.

    Raises
    ------
    RuntimeError
        If the ODE solver fails.
    """
    n_modes, nt = activity_coeffs.shape
    t = np.linspace(0, dt * (nt - 1), nt)

    # Simulate balloon model for each mode
    bold_coeffs = np.empty_like(activity_coeffs)
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
            raise RuntimeError("Balloon model ODE solver failed. Try using pde_method='fourier' or "
                               "a smaller timestep (dt) without altering balloon model parameters. "
                               f"scipy.integrate.solve_ivp message: {sol.message}")

        # Apply standard BOLD signal equation
        _, _, v, q = sol.y
        bold_coeffs[j, :] = V_0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))

    return bold_coeffs

def _model_wave_fem(
    ext_input: NDArray,
    mass: csc_matrix,
    stiffness: csc_matrix,
    dt: float = 1e-4,
    r: float = 17.4,
    gamma: float = 116.0,
    n_jobs: int = 1,
    verbose: int = 0 # for Parallel only (consider making **Parallel_kwargs)
) -> NDArray[np.floating]:
    """
    Full FEM version of ``sim_nft_waves()``, for validating the eigenmode expansion approach.
    """
    # Format / validate arguments
    nt = ext_input.shape[1]

    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    ext_input_padded = np.concatenate([np.zeros_like(ext_input), ext_input], axis=1)

    # Apply Fourier transform to get frequency-domain representation of the causal signal.
    ext_input_padded_freqs = np.fft.rfft(mass @ ext_input_padded, axis=1)

    # Compute components of NFT operator
    spatial = r**2 * stiffness
    omega = -2 * np.pi * np.fft.rfftfreq(2*nt, dt)
    temporal = -omega**2 / gamma**2 - 2j * omega / gamma + 1

    # Main computation
    # Compute activity at each frequency
    eqns = (
        (spatial + temporal[k] * mass, ext_input_padded_freqs[:, k])
        for k in range(len(temporal))
    )

    phi_freqs = None # stops it being unbound and keeps pyright happy
    if n_jobs > 1 or n_jobs == -1:
        try:
            from joblib import Parallel, delayed
            phi_freqs = Parallel(n_jobs=n_jobs, verbose=verbose, prefer="threads")(
                delayed(_solve_fem_freq)(op, inp) for op, inp in eqns
            ) 
        except ImportError:
            warn("joblib is not installed; parallel computation of frequencies will be disabled. "
                "Neuromodes can be installed with the 'cache' extra to include joblib as a "
                "dependency (e.g., pip install neuromodes[cache]).")
    if phi_freqs is None: # supposed to be serial, or if parallel failed
        phi_freqs = [_solve_fem_freq(op, inp) for op, inp in eqns]
    
    phi_freqs = np.stack(cast(list[NDArray[np.complex128]], phi_freqs), axis=1)

    # Inverse transform to time domain
    phi = np.fft.irfft(phi_freqs, axis=1, n=2*nt)

    # Return only the non-negative time part (t >= 0)
    return phi[:, nt:]

# TODO
def _model_balloon_fem(): 
    raise NotImplementedError("A FEM version of the balloon model is not yet implemented.")

def _solve_fem_freq(
    operator: csc_matrix,
    input: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Helper function for parallel frequency solves."""
    return linalg.splu(operator).solve(input)

def _analytical_fc(
    emodes: NDArray[floating],
    evals: NDArray[floating],
    r: float
) -> NDArray[floating]:
    """
    Calculate the analytical FC for the wave model under white noise input.

    Parameters
    ----------
    emodes : np.ndarray
        Eigenmodes of shape ``(n_verts, n_modes)``.
    evals : np.ndarray
        Eigenvalues corresponding to the modes, with shape ``(n_modes,)``.
    r : float
        Spatial length scale of wave propagation in millimeters.

    Returns
    -------
    np.ndarray
        Analytical FC matrix of shape ``(n_verts, n_verts)``.
    """
    ved = EigenData(emodes=emodes, evals=evals, checks=False)
    emodes, evals = ved.emodes, ved.evals
    mode_vars = 1.0 / (1 + r**2 * evals)
    cov = emodes @ (mode_vars[:, np.newaxis] * emodes.T)
    diag = np.sqrt(np.diag(cov))
    return cov / diag[:, np.newaxis] / diag[np.newaxis, :]