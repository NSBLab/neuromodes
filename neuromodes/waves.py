"""
Module for using neural field theory to simulate neural activity and BOLD signals on cortical 
surfaces.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal, cast
from warnings import warn

import numpy as np
import scipy.fft
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.sparse.linalg import eigsh, splu

from neuromodes.basis import decompose
from neuromodes.eigen import EigenData
from neuromodes.stats import _mult_by_cholesky

if TYPE_CHECKING:
    from typing import Literal

    from scipy.sparse import csc_matrix

    from neuromodes.eigen import _CheckKind
    _PDEKind = Literal["fourier", "ode", "fem"]

def sim_nft_waves(
    emodes: NDArray[np.floating] | None,
    evals: NDArray[np.floating] | None,
    nt: int | None = None,
    ext_input: NDArray[np.floating] | None = None,
    dt: float = 1e-4,
    r: float = 17.4,
    gamma: float = 116.0,
    method: _PDEKind = "fourier",
    mass: csc_matrix | None = None,
    stiffness: csc_matrix | None = None, # only used for FEM
    speed_limits: tuple[float, float] | None = (0, 150),
    hetero: NDArray[np.floating] | None = None,
    padding_tol: np.floating | Literal['nt'] = 1e-5,
    checks: _CheckKind = True,
    seed: int | None = None,
    cache_input: bool = False,
    n_jobs: int = 1,
    verbose: int = 0
) -> NDArray[np.floating]:
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
    method : str, optional
        Method for solving the wave PDEs. Either ``'fourier'``, ``'ode'``, or ``fem``. If ``fem``
        (no modal approximation), ``mass`` and ``stiffness`` must be provided while ``emodes`` and
        ``evals`` can be ``None``. Default is ``'fourier'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)`` used for decomposition of ``ext_input`` if
        provided and ``method`` is not ``'fem'``. Default is ``None``.
    speed_limits : tuple, optional
        If any wave speeds are outside this range (in meters per second), a warning is raised. If
        ``None``, no warning is raised. Default is ``(0, 150)``.
    hetero : array-like, optional
        Heterogeneity map of shape ``(n_verts,)``, to be provided when ``emodes`` are heterogeneous.
        This is used only to check wave speeds (see ``speed_limits`` above). If not provided, wave
        speed is assumed to be spatially uniform. Default is ``None``.
    padding_tol : float, optional
        Tolerance for Fourier wrap-around artifacts when ``method`` is ``'fourier'`` or ``'fem'``,
        between ``0`` and ``1``. Lower values increase the amount of zero-padding and thus the
        simulation time and memory usage. The number of padding timepoints is given by
        -ln(``padding_tol``)/(``gamma``*``dt``), meaning that a value of ``1`` at t=0 wraps around
        to a value of ``padding_tol`` at t=nt-1. Default is ``1e-5``.
    checks : bool, optional
        Whether to validate the input arguments. Default is ``True``.
    seed : int, optional
        Random seed for generating external input. Default is ``None``.
    cache_input : bool, optional
        If ``True`` and ``ext_input`` is ``None``, cache the generated random input to avoid
        recomputation for the same values of ``nt``, ``seed``, and number of rows (vertices) in
        ``emodes`` (see :func:`~neuromodes.io._cache_output` for details). Default is ``False``.
    n_jobs : int, optional
        Number of parallel jobs to run, for speeding up computation. Ignored if ``method`` is
        ``'ode'``. If not ``1`` and ``method`` is ``'fem'``, ``joblib`` must be installed. Default
        is ``1``. 
    verbose : int, optional
        ``joblib`` verbosity level for parallel execution when ``method`` is ``'fem'`` and ``n_jobs
        > 1``. Default is ``0``.

    Returns
    -------
    np.ndarray
        Simulated neural activity of shape ``(n_verts, nt)``.

    Raises
    ------
    ValueError
        If any of ``r``, ``gamma``, or ``dt`` is not positive (or zero in the case of ``r``).
    ValueError
        If ``nt`` is not ``None`` nor a positive integer.
    ValueError
        If ``speed_limits`` is not ``None`` nor a tuple ``(min_speed, max_speed)``, where ``0 ≤
        min_speed < max_speed``.
    ValueError
        If neither ``nt`` nor ``ext_input`` are provided.
    ValueError
        If ``method`` is not ``'fourier'``, ``'ode'``, nor ``'fem'``.
    ValueError
        If ``method='fem'`` and either of ``mass`` or ``stiffness`` is not provided.
    ValueError
        If ``method`` is ``'fourier'`` or ``'ode'`` and either of ``emodes`` or ``evals`` is not
        provided.
    ValueError
        If ``ext_input`` is provided and contains any NaN values.
    ValueError
        If ``n_jobs`` is not ``1``, ``method`` is ``'fem'``, and ``joblib`` is not installed.
    ValueError
        If ``padding_tol`` is not between ``0`` and ``1``.

    Notes
    -----
    Prior works have treated ``r`` as a free parameter to fit empirical data [1]_ [2]_, with the
    default value reflecting an optimal fit to human resting-state functional MRI data [2]_.
    Consider adjusting this parameter, as its optimum can vary across analyses (e.g., different
    surfaces, heterogeneous modes, parcellated timeseries, empirical data, fitting metrics, etc.).

    Since the simulation begins at rest, consider discarding the first timepoints of activity to
    allow the system to reach a steady state.

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
            hetero=hetero, data=ext_input, checks=checks
            )
        emodes, evals, mass, stiffness, ext_input, hetero = \
            ved.emodes, ved.evals, ved.mass, ved.stiffness, ved.data, ved.hetero
        
    if emodes is not None: 
        n_verts, n_modes = emodes.shape
    elif stiffness is not None:
        n_verts = stiffness.shape[0]
    elif method == 'fem':
        raise ValueError(f"mass and stiffness matrices must be provided for {method} method.")
    else: 
        raise ValueError(f"emodes must be provided for {method} method.")
        
    r = float(r)
    gamma = float(gamma)
    if r < 0:
        raise ValueError("Parameter r must be non-negative.")
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
        speed = calc_nft_wave_speed(r, gamma, hetero=hetero)
        min_speed, max_speed = np.min(speed), np.max(speed)
        if min_speed < speed_limits[0] or max_speed > speed_limits[1]:
            calc_str = min_speed if min_speed == max_speed else f"{min_speed:.1f}-{max_speed:.1f}"
            warn("The combination of r, gamma, and hetero leads to wave speeds "
                 f"outside the range of {speed_limits[0]}-{speed_limits[1]} m/s (calculated "
                 f"{calc_str} m/s). Consider changing these parameters to ensure physiologically "
                 "plausible wave speeds, or adjust speed_limits.")
    if method not in ['fourier', 'ode', 'fem']:
        raise ValueError(f"Invalid PDE method '{method}'; must be 'fourier', 'ode', or 'fem'.")
    if padding_tol != 'nt' and (not isinstance(padding_tol, (int, float))
                                or padding_tol <= 0 or padding_tol > 1):
        raise ValueError("padding_tol must be between 0 and 1, or 'nt'.")
    if n_jobs != 1:
        if method == 'ode':
            warn("n_jobs is ignored when method is 'ode'.")
        elif method == 'fem' and find_spec("joblib") is None:
            raise ImportError("When method is 'fem', joblib must be installed to use n_jobs != "
                              "1 for parallel execution. Neuromodes can be installed with the "
                              "'cache' extra to include joblib as a dependency (e.g., pip install "
                              "neuromodes[cache]). ")

    # Check that dt is small enough to capture highest frequency, per Nyquist-Shannon theorem
    # Highest frequency comes from the mode with the highest eigenvalue
    # This needs to be computed for the FEM method
    max_eval = eigsh(stiffness, M=mass, k=1, which='LM', return_eigenvectors=False)[0] \
        if method == 'fem' else evals[-1]

    max_freq = calc_nft_mode_freqs(max_eval, r, gamma)

    # Check if highest-frequency mode is above Nyquist limit
    if max_freq >= 1 / (2 * dt):
        dt_nyquist = 1 / (2 * max_freq)
        warn(f"dt={dt} is too large to capture frequencies produced by the highest-frequency mode "
             f"({max_freq:.4f} Hz), per the Nyquist-Shannon sampling theorem. To reduce aliasing, "
             f"consider reducing dt to below {dt_nyquist:.4f}.")

    # Process or generate external input
    if ext_input is not None:
        if np.isnan(ext_input).any():
            raise ValueError("ext_input contains NaN values, which are not allowed.")
        if nt is not None:
            warn("nt is ignored when ext_input is provided.")
        if seed is not None:
            warn("seed is ignored when ext_input is provided.")
        if cache_input:
            warn("cache_input is ignored when ext_input is provided.")
        nt = ext_input.shape[1]
        if method == 'fem':
            input_w = mass @ ext_input
        else:
            input_coeffs = decompose(ext_input, emodes, mass=mass, checks=False)
    elif nt is not None:
        if cache_input and seed is not None:
            from neuromodes.io import _cache_output
            gen_noise = _cache_output(_gen_noise)
        else:
            if cache_input and seed is None:
                warn("cache_input is ignored when seed is None.")
            gen_noise = _gen_noise
        if method == 'fem':
            n_verts = stiffness.shape[0]
            # White noise in vertex space, pre-weighted by mass for weak form PDE
            input_w = np.asarray(gen_noise(n_verts, nt, seed=seed, sample='vertices', mass=mass))
        else:
            # Generate white noise in modal space for computational efficiency
            input_coeffs = np.asarray(gen_noise(n_modes, nt, seed=seed))
    else: # not the nicest, but it makes pyright the happiest
        raise ValueError("Either nt or ext_input must be provided.")

    # Non-modal FEM implementation
    if method == 'fem':
        if mass is None or stiffness is None:
            raise ValueError("Mass and stiffness matrices must be provided when method is 'fem'.")
        return _model_wave_fem(input_w, mass, stiffness, dt, r, gamma, padding_tol, n_jobs, verbose)
    
    # Standard modal implementation: pass decomposed input, then output reconstruction
    activity_coeffs = (_model_wave_fourier(input_coeffs, dt, r, gamma, evals, padding_tol, n_jobs)
                       if method == 'fourier' else
                       _model_wave_ode(input_coeffs, dt, r, gamma, evals))

    return emodes @ activity_coeffs

def calc_nft_wave_speed(
    r: float,
    gamma: float,
    hetero: NDArray[np.floating] | None = None
) -> float | NDArray[np.floating]:
    """
    Calculate wave speed (m/s) based on the two parameters of the wave model. If a scaled
    heterogeneity map is provided, wave speeds are calculated for each cortical vertex (i.e., each
    entry of ``hetero``).
    
    Parameters
    ----------
    r : float
        Axonal length scale for wave propagation in millimeters.
    gamma : float
        Damping parameter for wave propagation in seconds^-1.
    hetero : array-like, optional
        Non-negative heterogeneity map of shape (n_verts,). If ``None`` (or all ones), speed is
        taken as spatially uniform. In line with prior work [1]_, it is recommended that ``hetero``
        is in the range [0, 2] (see rescaling functions :func:`~neuromodes.stats.zscorew` and
        :func:`~neuromodes.stats.sigmoid_rescale`). Default is ``None``.
    
    Returns
    -------
    float or np.ndarray
        Wave speed across the whole cortex in meters per second, or at each vertex if ``hetero`` is
        provided.

    Raises
    ------
    ValueError
        If ``r`` is negative or ``gamma`` is non-positive.
    ValueError
        If ``hetero`` is provided and contains negative values.

    References
    ----------
    ..  [1] Barnes, V., et al. (2026). Regional heterogeneity shapes macroscopic wave dynamics
        of the human and non-human primate cortex, bioRxiv.
        https://doi.org/10.64898/2026.01.22.701178
    """
    if r < 0:
        raise ValueError("Parameter r must be non-negative.")
    if gamma <= 0:
        raise ValueError("Parameter gamma must be positive.")
    
    speed = (r / 1000) * gamma # Convert r to meters

    if hetero is not None:
        hetero = np.asarray(hetero)
        if np.any(hetero < 0):
            raise ValueError("Heterogeneity map must contain only non-negative values.")
        
        speed *= np.sqrt(hetero)

    return speed

def calc_nft_mode_freqs(
    evals: NDArray[np.floating],
    r: float,
    gamma: float
) -> NDArray[np.floating]:
    """
    Calculate the damped frequencies (Hz) produced by each mode of the wave model. Note that all
    modes are underdamped, except the first which has an eigenvalue of 0 and thus is critically
    damped (see Notes).

    Parameters
    ----------
    evals : np.ndarray
        Eigenvalues corresponding to the modes, with shape ``(n_modes,)``.
    r : float
        Spatial length scale of wave propagation in millimeters.
    gamma : float
        Damping rate of wave propagation in seconds^(-1).

    Returns
    -------
    np.ndarray
        Damped frequencies of shape ``(n_modes,)``.

    Raises
    ------
    ValueError
        If ``r`` is negative or ``gamma`` is non-positive.

    Notes
    -----
    The NFT wave equation for mode j is given by
    
        [1/γ² d²/dt² + 2/γ d/dt + (1 + r² λⱼ)] ϕⱼ(t) = qⱼ(t),

    where ϕⱼ(t) is the activity of mode j, λⱼ is the eigenvalue of mode j, and qⱼ(t) is the external
    input. Mapping these to the classical driven harmonic oscillator equation, we have:

        m d²/dt² ϕⱼ(t) = - k ϕⱼ(t) - c d/dt ϕⱼ(t) + qⱼ(t)
        m = 1/γ², c = 2/γ, k = 1 + r²λⱼ

    The damped angular frequency of mode j (ω_dⱼ) is then derived from the undamped angular
    frequency (ω_uⱼ) and damping ratio (ζⱼ) as:
    
        ω_dⱼ = ω_uⱼ √(1 - ζⱼ²)
             = √(k/m) √(1 - (c/(2√(k/m)))²)
             = √(k/m - c²/(4m²))
             = √((1 + r²λⱼ) γ² - (2/γ)² / (4(1/γ²)²)
             = √((1 + r²λⱼ) γ² - γ²)
             = rγ √λⱼ

    Angular frequency is converted to Hz by dividing by 2π. Also worth noting is that the damping
    ratio ζⱼ = c/(2√(k/m)) = 1/√(1 + r²λⱼ) < 1 for λⱼ > 0, meaning all modes are underdamped except
    the first mode, which is critically damped (ζ₀ = 1).
    """
    # Validate parameters
    if r < 0:
        raise ValueError("Parameter r must be non-negative.")
    if gamma <= 0:
        raise ValueError("Parameter gamma must be positive.")

    # Calculate and convert from rad/s to Hz
    return r * gamma * np.sqrt(evals) / (2 * np.pi)

def calc_nft_fc(
    emodes: NDArray[np.floating],
    evals: NDArray[np.floating],
    r: float
) -> NDArray[np.floating]:
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

    Notes
    -----
    Parseval's identity tells us that the variance of a signal in the time domain is equal to the
    integral of its power spectral density in the frequency domain. For the wave model under a white
    noise input q(t), it follows that the j-th mode's variance is given by

        var(ϕⱼ(t)) = ∫ |Φⱼ(ω)|² dω
                   = ∫ |Qⱼ(ω) Hⱼ(ω)|² dω
                   = ∫ |Qⱼ(ω)|² |Hⱼ(ω)|² dω
                   = ∫ |Hⱼ(ω)|² dω
                   = ∫ 1 / |-(1/γ²) ω² - (2/γ) iω + (1 + r²λⱼ)|² dω
                   = 1 / (2γ (1 + r²λⱼ)).

    The variance-covariance matrix of the wave model can then be reconstructed in vertex space as

        cov(ϕ(t)) = (1/n) ϕ(t) ϕ(t)ᵀ
                  = (1/n) (Ψ ϕ_β(t)) (Ψ ϕ_β(t))ᵀ
                  = Ψ (1/n) ϕ_β(t) ϕ_β(t)ᵀ Ψᵀ
                  = Ψ cov(ϕ_β(t)) Ψᵀ
                  = Ψ diag(var(ϕ_β(t))) Ψᵀ,

    where Ψ is the eigenmodes matrix and ϕ_β(t) is the activity in modal space.

    Finally, the correlation matrix is obtained by normalising the rows and columns of the
    variance-covariance matrix by their respective standard deviations. This removes the scaling
    factor of 1/(2γ) from the correlation matrix, meaning that the damping parameter γ has no effect
    on the model FC.
    """
    ved = EigenData(emodes=emodes, evals=evals, checks=False)
    emodes, evals = ved.emodes, ved.evals

    # compute mode variances, ignoring the 1/(2*gamma) factor
    mode_vars = 1.0 / (1 + r**2 * evals)

    # reconstruct (change from modal to vertex basis)
    cov = emodes @ (mode_vars[:, None] * emodes.T)

    # Normalise variance-covariance matrix to get correlation
    stds = np.sqrt(np.diag(cov))
    return cov / stds[:, None] / stds[None, :]

def balloon_model(
    activity: NDArray[np.floating],
    emodes: NDArray[np.floating],
    dt: float,
    kappa: float = 0.65,
    tau: float = 0.98,
    alpha: float = 0.32,
    rho: float = 0.34,
    gamma_h: float = 0.41,
    V_0: float = 0.02,
    k1: float = 3.72,
    k2: float = 0.527,
    k3: float = 0.48,
    method: _PDEKind = "fourier",
    padding_tol: np.floating | Literal['nt'] = 1e-5,
    mass: csc_matrix | None = None,
    checks: _CheckKind = True,
    n_jobs: int = 1
) -> NDArray[np.floating]:
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
    kappa : float, optional
        Signal decay rate in seconds^-1.
    tau : float, optional
        Hemodynamic transit time in seconds.
    alpha : float, optional
        Grubb's exponent (unitless).
    rho : float, optional
        Resting oxygen extraction fraction (unitless).
    gamma_h : float, optional
        Hemodynamic gain (unitless).
    V_0 : float, optional
        Resting blood volume fraction (unitless).
    k1 : float, optional
        First coefficient in BOLD signal equation (unitless).
    k2 : float, optional
        Second coefficient in BOLD signal equation (unitless).
    k3 : float, optional
        Third coefficient in BOLD signal equation (unitless).
    method : str, optional
        Method for solving the balloon PDEs. Either ``'fourier'`` or ``'ode'``. Note that
        ``'fourier'`` relies on a first-order approximation of the nonlinear ODE system. Default is
        ``'fourier'``.
    padding_tol : float, optional
        Tolerance for Fourier wrap-around artifacts, between ``0`` and ``1``. Lower values increase
        the amount of zero-padding and thus the simulation time and memory usage. The number of
        padding timepoints is given by -ln(``padding_tol``)/(``dt`` * min(``kappa``/2, 1/``tau``,
        1/(``tau`` * ``alpha``))), meaning that a value of ``1`` at t=0 wraps around to a value of
        approximately ``padding_tol`` at t=``nt``-1. Default is ``1e-5``.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts), used for the decomposition of activity. Default
        is ``None``, which uses the identity matrix.
    checks : bool, optional
        Whether to perform checks on the input arrays. Default is ``True``.
    n_jobs : int, optional
        Number of threads to use for speeding up computation when ``method`` is ``'fourier'``.
        Default is ``1``.

    Returns
    -------
    ndarray
        Simulated BOLD signal in vertex space of shape ``(n_verts, n_timepoints)``.

    Raises
    ------
    ValueError
        If ``dt`` is not positive.
    ValueError
        If ``method`` is not ``'fourier'`` or ``'ode'``.

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
    if method not in ['fourier', 'ode']:
        raise ValueError(f"Invalid PDE method '{method}'; must be 'fourier' or 'ode'.")
    params = {"kappa": kappa, "tau": tau, "alpha": alpha, "rho": rho, "V_0": V_0,
              "gamma_h": gamma_h, "k1": k1, "k2": k2, "k3": k3}
    for param_name, param_value in params.items():
        if not isinstance(param_value, (int, float)) or param_value <= 0:
            raise ValueError(f"Parameter '{param_name}' must be a positive number.")

    # check that dt is sufficiently small to capture frequency of blood flow response
    w_f = np.sqrt(gamma_h - kappa**2/4) / (2 * np.pi)  # Hz
    if w_f >= 1 / (2 * dt):
        dt_nyquist = 1 / (2 * w_f)
        warn(f"dt={dt} is too large to capture the linearised blood flow response frequency "
             f"({w_f:.4f} Hz), per the Nyquist-Shannon sampling theorem. To reduce aliasing, "
             f"consider reducing dt to below {dt_nyquist:.4f}.")

    # Eigendecompose activity to get modal coefficients over time
    # TODO: consider re-adding sim_nft_waves(..., return_bold=False) to avoid redundant
    # decomposition/padding/fft/ifft/slicing/reconstruction
    activity_coeffs = decompose(activity, emodes, mass=mass, checks=False)

    # Apply model to each mode's activity timeseries
    bold_coeffs = (_model_balloon_fourier(activity_coeffs, dt, padding_tol, n_jobs, kappa, tau,
                                          alpha, rho, V_0, w_f, k1, k2, k3)
                   if method == 'fourier' else
                   _model_balloon_ode(activity_coeffs, dt, kappa, tau, alpha, rho, V_0, gamma_h, k1,
                                      k2, k3))

    # Transform timeseries from modal coefficients back to vertex space
    return emodes @ bold_coeffs

def _gen_noise(
    n_samples: int,
    nt: int,
    mass: csc_matrix | None = None,
    seed: int | None = None
) -> NDArray[np.floating]:
    """
    Generate reproducible white noise of shape ``(n_samples, nt)`` for a given ``seed``, derived
    from a standard normal distribution. The output is reproducible across nt (i.e.,
    ``_gen_noise(n_samples, nt, seed) == _gen_noise(n_samples, nt+k, seed)[:, :nt]``). If ``mass``
    is provided, the noise is pre-weighted by the mass matrix for use in the weak form PDE of the
    FEM solver (see Notes).

    Parameters
    ----------
    n_samples : int
        Number of samples (rows) in the output noise array. Depending on ``sample``, each row
        represents either an eigenmode or a vertex.
    nt : int
        Number of time points (columns) in the output noise array.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)`` used for combined normalization /
        mass-weighting (for weak form PDE). If ``None``, no corrections are applied (e.g., modal
        noise). Default is ``None``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Gaussian white noise array of shape ``(n_samples, nt)``.

    Notes
    -----
    A vector of standard normal noise (f) has an expected Euclidean mean and variance of 0 and 1,
    respectively. However, in vertex space, a map's mean and variance are weighted by the mass
    matrix. As such, spatial noise must account for this mass-weighting. This is achieved by
    leveraging the Cholesky decomposition of the symmetric positive definite mass matrix: M = LLᵀ,
    where L is lower triangular. By defining the mass-normalized noise as w = L⁻ᵀf, the expected
    mean and variance remain independent of the mass matrix:

    E[μ(w)] = E[sum(Mw) / sum(M)]
            = E[1ᵀ(LLᵀ)(L⁻ᵀf) / (1ᵀM1)]
            = (1ᵀLE[f]) / (1ᵀM1)
            = (1ᵀL0) / (1ᵀM1)
            = 0

    E[σ²(w)] = E[(w-μ(w))ᵀM(w-μ(w))]
             = E[wᵀMw - wᵀMμ(w) - μ(w)ᵀMw + μ(w)ᵀMμ(w)]
             = E[wᵀMw] - E[wᵀMμ(w)] - E[μ(w)ᵀMw] + E[μ(w)ᵀMμ(w)]
             = E[(L⁻ᵀf)ᵀ(LLᵀ)(L⁻ᵀf)] - E[wᵀ]ME[μ(w)] - E[μ(w)]ᵀME[w] + E[μ(w)ᵀ]ME[μ(w)]
             = E[fᵀ(L⁻¹LLᵀL⁻ᵀ)f] - 0 - 0 + 0 
             = E[fᵀf]
             = E[∑ᵢⁿ(fᵢ²)]
             = nE[fᵢ²]
             = n
    
    However, since the weak form PDE of the NFT wave model's FEM solver needs mass-weighting, we can
    pre-factor this into our noise generation:
    
    Mw = ML⁻ᵀf
       = LLᵀL⁻ᵀf
       = Lf

    This lets us use a simple sparse multiplication instead of obtaining w = L⁻ᵀf via the slower and
    less numerically stable ``scipy.sparse.linalg.spsolve_triangular(L.T, f)``. Note that for lumped
    (diagonal) mass, L = Lᵀ = √M (element-wise).
    """
    # Draw samples from N(0, 1) in column-major order to ensure reproducibility across nt, then
    # transpose to desired shape (n_samples, nt)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((nt, n_samples)).T

    if mass is None:  # e.g., modes along rows
        return noise

    # vertex space, pre-weighted by Cholesky factor
    return _mult_by_cholesky(noise, mass, transpose=False)

def _model_wave_fourier(
    input_coeffs: NDArray[np.floating],
    dt: float,
    r: float,
    gamma: float,
    evals: NDArray[np.floating],
    padding_tol: float | Literal['nt'],
    n_jobs: int
) -> NDArray[np.floating]:
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
    padding_tol : float
        Tolerance for Fourier wrap-around artifacts, between ``0`` and ``1``. Lower values increase
        the amount of zero-padding and thus the simulation time and memory usage. The number of
        padding timepoints is given by -ln(``padding_tol``)/(``gamma``*``dt``), meaning that a value
        of ``1`` at t=0 wraps around to a value of ``padding_tol`` at t=``nt``-1. For more details,
        see
        https://github.com/NSBLab/neuromodes/blob/main/docs/validation/waves_fourier_padding.ipynb.
    n_jobs : int, optional
        Number of threads to use for speeding up computation.

    Returns
    -------
    out : ndarray
        The real part of the time-domain response of all modes at the specified time points, with
        shape ``(n_modes, nt)``.    
    """
    nt = input_coeffs.shape[1]

    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    # NFT transfer function has a temporal envelope of exp(-gamma * t), so we can pad until this is
    # below the padding tolerance.
    if padding_tol == 'nt':
        n_pad = nt
    else:
        t_pad = -np.log(padding_tol) / gamma
        n_pad = int(np.ceil(t_pad / dt))
    input_coeffs_padded = np.pad(input_coeffs, ((0, 0), (n_pad, 0)), constant_values=0)

    # Get frequencies (rad/s)
    # Negative sign in front of omega matches the physics convention of e^(iwt)
    omega = -2 * np.pi * scipy.fft.rfftfreq(n_pad + nt, d=dt)

    # Compute NFT transfer function
    tfunc = gamma**2 / (-omega**2 - 2j * omega * gamma + gamma**2 * (1 + r**2 * evals[:, None]))

    # Fourier transform, apply transfer function, and inverse transform to time domain
    # Faster to use `rfft` here than `fftshift(ifft)` (original implementation)
    input_coeffs_f = scipy.fft.rfft(input_coeffs_padded, axis=1, overwrite_x=True, workers=n_jobs)

    out_fft = tfunc * input_coeffs_f

    out_full = scipy.fft.irfft(out_fft, n=n_pad+nt, axis=1, overwrite_x=True, workers=n_jobs)

    # Return only the non-negative time part (t >= 0)
    return out_full[:, n_pad:]

def _model_wave_ode(
    input_coeffs: np.ndarray,
    dt: float,
    r: float,
    gamma: float,
    evals: np.ndarray
) -> np.ndarray:
    """
    Solves the damped wave ODE for all eigenmodes.

    Parameters
    ----------
    input_coeffs : np.ndarray
        Input drive to the system with shape ``(n_modes, nt)`` (written as qⱼ in equation below).
    dt : float
        Time step for the simulation in seconds.
    gamma : float
        Damping coefficient in seconds^-1.
    r : float
        Spatial length scale in millimeters.
    evals : np.ndarray
        Eigenvalues for each mode with shape ``(n_modes,)``.

    Returns
    -------
    np.ndarray
        Time evolution of ϕⱼ(t), solution to the wave equation, with shape ``(n_modes, nt)``.
    
    Notes
    -----
    The equation is derived from the damped wave equation for each mode j:
    
       (1/γ²) d²(ϕⱼ)/dt² + (2/γ) d(ϕⱼ)/dt + (1 + r²λⱼ) ϕⱼ = qⱼ
    
    Rearranging gives us the first-order system
        d(x₁)/dt = x₂
        d(x₂)/dt = -2γ x₂ - γ²(1 + r²λⱼ) x₁ + γ²qⱼ
    """
    n_modes, nt = input_coeffs.shape
    t_vec = np.linspace(0, dt * (nt - 1), nt)
    
    # Create interpolator for input, as solver may need intermediate timepoints.
    # Linear interpolation creates sharp corners, leading to high-frequency artefacts and numerical
    # instabilities in the solver.
    # Cubic splining is better but can overshoot/undershoot and create phantom preceding inputs.
    # Piecewise Cubic Hermite Interpolating Polynomials (PCHIPs) are smooth and monotonic.
    # Another option is sinc interpolation, which bandlimits and thus matches the Fourier method,
    # but this is demanding on time/memory and not implemented by numpy/scipy.
    input_interp = PchipInterpolator(t_vec, input_coeffs, axis=1, extrapolate=False)

    # Define ODE system, needed for solve_ivp
    def wave_odes(t, y):
        # Unpack y as a state vector of modes' activity (x1) and their derivatives (x2)
        x1, x2 = np.split(y, 2)
        
        # Get inputs for all modes at time t
        # If t is among the timepoints of interest (t_vec), this is simply input_coeffs[:, idx]
        input = input_interp(t)
        
        # Calculate derivatives using the NFT wave equation
        dx1dt = x2
        dx2dt = -2 * gamma * x2 - gamma**2 * (1 + r**2 * evals) * x1 + gamma**2 * input
        
        # Return state vector derivative
        return np.concatenate([dx1dt, dx2dt])
        
    # Set initial conditions: all modes begin at rest
    y0 = np.zeros(2 * n_modes)

    # Solve ODE system
    sol = solve_ivp(
        wave_odes,
        t_span=(0, t_vec[-1]),
        y0=y0,
        t_eval=t_vec,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    # Return activity without derivatives
    return sol.y[:n_modes, :]

def _model_wave_fem(
    input_w: NDArray[np.floating],
    mass: csc_matrix,
    stiffness: csc_matrix,
    dt: float,
    r: float,
    gamma: float,
    padding_tol: float | Literal['nt'],
    n_jobs: int,
    verbose: int # for Parallel only (consider making **Parallel_kwargs)
) -> NDArray[np.floating]:
    """
    Simulates the time evolution of wave models for all vertices using a finite element method (FEM)
    approach. This function applies a Fourier transform to the input, computes the system's
    frequency response, and then applies an inverse Fourier transform to obtain the time-domain
    response of each vertex.

    Parameters
    ----------
    input_w : np.ndarray
        Array of mass-weighted external input at each vertex over time, with shape ``(n_verts,
        nt)``.
    mass : scipy.sparse.csc_matrix
        The mass matrix of shape ``(n_verts, n_verts)``.
    stiffness : scipy.sparse.csc_matrix
        The stiffness matrix of shape ``(n_verts, n_verts)``.
    dt : float
        Time step for the simulation in seconds.
    r : float
        Spatial length scale of wave propagation in millimeters.
    gamma : float
        Damping rate of wave propagation in seconds^(-1).
    padding_tol : float
        Tolerance for Fourier wrap-around artifacts, between ``0`` and ``1``. Lower values increase
        the amount of zero-padding and thus the simulation time and memory usage. The number of
        padding timepoints is given by -ln(``padding_tol``)/(``gamma``*``dt``), meaning that a value
        of ``1`` at t=0 wraps around to a value of ``padding_tol`` at t=nt-1.
    n_jobs : int
        Number of parallel jobs to run. If not ``1``, ``joblib`` must be installed.
    verbose : int
        Verbosity level for parallel execution.
    """
    nt = input_w.shape[1]

    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    if padding_tol == 'nt':
        n_pad = nt
    else:
        t_pad = -np.log(padding_tol) / gamma
        n_pad = int(np.ceil(t_pad / dt))
    input_padded = np.pad(input_w, ((0, 0), (n_pad, 0)), constant_values=0)

    # Get frequencies
    # Negative sign in front of omega matches the physics convention of e^(iwt)
    omega = -2 * np.pi * scipy.fft.rfftfreq(nt+n_pad, dt)

    # Compute components of NFT operator
    spatial = r**2 * stiffness
    temporal = -omega**2 / gamma**2 - 2j * omega / gamma + 1

    # Apply Fourier transform to get frequency-domain representation of the causal signal.
    input_padded_freqs = scipy.fft.rfft(input_padded, axis=1, overwrite_x=True, workers=n_jobs)

    # Compute activity at each frequency (TODO: consider memory usage, if fine then vectorise)
    eqns = (
        (spatial + temporal[k] * mass, input_padded_freqs[:, k])
        for k in range(len(temporal))
    )

    # Define helper function for parallelization
    def _solve_fem_freq(operator, input):
        return splu(operator).solve(input)  # cannot use Cholesky decomposition is not always SPD

    phi_freqs = None # stops it being unbound and keeps pyright happy
    if n_jobs > 1 or n_jobs == -1:
        from joblib import Parallel, delayed
        phi_freqs = Parallel(n_jobs=n_jobs, verbose=verbose, prefer="threads")(
            delayed(_solve_fem_freq)(op, inp) for op, inp in eqns
        )
    else:
        phi_freqs = [_solve_fem_freq(op, inp) for op, inp in eqns]
    
    phi_freqs = np.stack(cast(list[NDArray[np.complex128]], phi_freqs), axis=1)

    # Inverse transform to time domain
    phi = scipy.fft.irfft(phi_freqs, axis=1, n=nt+n_pad, overwrite_x=True, workers=n_jobs)

    # Return only the non-negative time part (t >= 0)
    return phi[:, n_pad:]

def _model_balloon_fourier(
    activity_coeffs: NDArray[np.floating],
    dt: float,
    padding_tol: float | Literal['nt'],
    n_jobs: int,
    kappa: float,
    tau: float,
    alpha: float,
    rho: float,
    V_0: float,
    w_f: float,
    k1: float,
    k2: float,
    k3: float
) -> NDArray[np.floating]:
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
    padding_tol : float
        Tolerance for Fourier wrap-around artifacts, between ``0`` and ``1``. Lower values increase
        the amount of zero-padding and thus the simulation time and memory usage. The number of
        padding timepoints is given by -ln(``padding_tol``)/(``dt`` * min(``kappa``/2, 1/``tau``,
        1/(``tau`` * ``alpha``))), meaning that a value of ``1`` at t=0 wraps around to a value of
        approximately ``padding_tol`` at t=``nt``-1. For more details, see
        https://github.com/NSBLab/neuromodes/blob/main/docs/validation/waves_fourier_padding.ipynb.
    n_jobs : int
        Number of threads to use for speeding up computation.
    kappa : float
        Signal decay rate in seconds^-1.
    tau : float
        Hemodynamic transit time in seconds.
    alpha : float
        Grubb's exponent (unitless).
    rho : float
        Resting oxygen extraction fraction (unitless).
    V_0 : float
        Resting blood volume fraction (unitless).
    w_f : float
        Frequency of blood flow response in radians per second.
    k1 : float
        First coefficient in BOLD signal equation (unitless).
    k2 : float
        Second coefficient in BOLD signal equation (unitless).
    k3 : float
        Third coefficient in BOLD signal equation (unitless).

    Returns
    -------
    np.ndarray
        The real part of the time-domain response of all modes at the specified time points, with
        shape (n_modes, nt).
    """
    nt = activity_coeffs.shape[1]
    
    # Zero-pad input at t < 0 for causality
    # Identify poles of transfer function by finding roots of its denominator
    # Flow response transfer function (phi_hat_Fz) has a pole at s = -kappa/2
    # BOLD response transfer function (phi_hat_yF) has poles at s = -1/tau and s = -1/(alpha*tau)
    # Padding should compensate for the slowest-decaying of these poles, each given by e^(s*t)
    if padding_tol == 'nt':
        n_pad = nt
    else:
        decay_eff = np.min((kappa / 2, 1 / tau, 1 / (alpha * tau)))
        t_pad = -np.log(padding_tol) / decay_eff
        n_pad = int(np.ceil(t_pad / dt))
    activity_coeffs_padded = np.pad(activity_coeffs, ((0, 0), (n_pad, 0)), constant_values=0)

    # Get frequencies
    omega = -2 * np.pi * scipy.fft.rfftfreq(nt+n_pad, d=dt)

    # Calculate balloon model frequency response (Pang et al. 2016)
    # Negative sign in front of omega matches the physics convention of e^(iwt)
    beta = (rho + (1 - rho) * np.log(1 - rho)) / rho
    phi_hat_Fz = 1 / (-(omega + 1j * 0.5 * kappa) ** 2 + w_f ** 2)
    phi_hat_yF = V_0 * (
        alpha * (k2 + k3) * (1 - 1j * tau * omega) 
        - (k1 + k2) * (alpha + beta - 1 - 1j * tau * alpha * beta * omega)
        ) / ((1 - 1j * tau * omega) * (1 - 1j * tau * alpha * omega))
    balloon_freq_response = phi_hat_yF * phi_hat_Fz

    # Fourier transform, apply transfer function, and inverse transform back to time domain
    activity_coeffs_f = scipy.fft.rfft(activity_coeffs_padded, axis=1, overwrite_x=True,
                                       workers=n_jobs)

    out_fft = balloon_freq_response[None, :] * activity_coeffs_f

    out_full = scipy.fft.irfft(out_fft, n=nt+n_pad, axis=1, overwrite_x=True, workers=n_jobs)

    # Remove zero padding
    return out_full[:, n_pad:]

def _model_balloon_ode(
    activity_coeffs: NDArray[np.floating],
    dt: float,
    kappa: float,
    tau: float,
    alpha: float,
    rho: float,
    V_0: float,
    gamma_h: float,
    k1: float,
    k2: float,
    k3: float
) -> NDArray[np.floating]:
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
    kappa : float
        Signal decay rate in seconds^-1.
    tau : float
        Hemodynamic transit time in seconds.
    alpha : float
        Grubb's exponent (unitless).
    rho : float
        Resting oxygen extraction fraction (unitless).
    V_0 : float
        Resting blood volume fraction (unitless).
    gamma_h : float
        Hemodynamic gain (unitless).
    k1 : float
        First coefficient in BOLD signal equation (unitless).
    k2 : float
        Second coefficient in BOLD signal equation (unitless).
    k3 : float
        Third coefficient in BOLD signal equation (unitless).

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
    t_vec = np.linspace(0, dt * (nt - 1), nt)

    # Create interpolator for activity, as solver may need intermediate timepoints
    activity_interp = PchipInterpolator(t_vec, activity_coeffs, axis=1, extrapolate=False)

    # Define ODE system, needed for solve_ivp
    def balloon_odes(t, y):
        # Unpack y as a state vector of four components for each mode: [z, f, v, q]
        z, f, v, q = np.split(y, 4)

        # Get activity for all modes at time t
        # If t is among the timepoints of interest (t_vec), this is simply activity_coeffs[:, idx]
        activity = activity_interp(t)
        
        # Calculate derivatives
        dzdt = activity - kappa * z - gamma_h * (f - 1)
        dfdt = z
        dvdt = (f - v ** (1 / alpha)) / tau
        dqdt = (f * (1 - (1 - rho) ** (1 / f)) / rho - q * v ** (1 / alpha - 1)) / tau

        # Return state vector derivative
        return np.concatenate([dzdt, dfdt, dvdt, dqdt])
        
    # Set initial conditions
    y0 = np.zeros(4 * n_modes)
    y0[n_modes:] = 1.0  # f, v, q start at 1

    # Solve ODE system
    sol = solve_ivp(
        balloon_odes,
        t_span=(0, t_vec[-1]),
        y0=y0,
        t_eval=t_vec,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    if not sol.success:
        raise RuntimeError("Balloon model ODE solver failed. Try using method='fourier' or a "
                           "smaller timestep (dt) without altering balloon model parameters. "
                           f"scipy.integrate.solve_ivp message: {sol.message}")

    # Compute BOLD
    _, _, v, q = np.split(sol.y, 4)
    return V_0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))