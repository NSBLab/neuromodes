"""
Module for computing geometric eigenmodes of brain structures from surface meshes.
"""

from __future__ import annotations
from typing import Any, Literal, TYPE_CHECKING
from warnings import warn
from dataclasses import dataclass
from lapy import TriaMesh, Solver
import numpy as np
from neuromodes.io import read_surf
from neuromodes.mesh import mask_mesh, check_surf

if TYPE_CHECKING:
    from pathlib import Path
    # from lapy import TriaMesh
    from nibabel.gifti.gifti import GiftiImage
    from numpy import floating, signedinteger, bool_
    from numpy.random import Generator
    from numpy.typing import NDArray
    from scipy.sparse import csc_matrix
    from neuromodes.basis import _ReconSingle, _ReconList, _ReconTSSingle, _ReconTSList

class EigenSolver(Solver):
    """
    Class for computing and using eigenmodes and eigenvalues of a brain structure mesh [1]_ via the
    Finite Element Method, which discretizes the Laplace-Beltrami eigenvalue problem using mass and
    stiffness matrices [2]_ [3]_. Spatial heterogeneity can be optionally incorporated, modifying
    the Laplace-Beltrami operator via an isotropic diffusion tensor [4]_. After calling
    :meth:`solve` to compute modes, a range of analysis methods can be called (:meth:`decompose`,
    :meth:`reconstruct`, :meth:`reconstruct_timeseries`, :meth:`sim_nft_waves`,
    :meth:`balloon_model`, :meth:`compute_gem`).

    Parameters
    ----------
    geometry : str, pathlib.Path, lapy.TriaMesh, or dict
        The surface mesh of a brain structure. Can be:
        - A path to one of the following file formats: ``.gii``, ``.vtk``, ``.white``, ``.pial``,
        ``.inflated``, ``.orig``, ``.sphere``, ``.smoothwm``, ``.qsphere``, ``.fsaverage``
        - An instance of ``GiftiImage``, ``lapy.TriaMesh``
        - A dictionary with keys ``'vertices'`` and ``'faces'``
    mask : array-like, optional
        A boolean mask to exclude certain vertices (e.g., medial wall) from the mesh. Default is
        ``None``.
    normalize : bool, optional
        Whether to normalize the mesh to have unit surface area and centroid at origin. Note that
        this will rescale the computed eigenmodes, eigenvalues, and mass matrix. Default is
        ``False``.
    hetero : array-like, optional
        A heterogeneity map to scale the Laplace-Beltrami operator. Default is ``None``.
    alpha : float, optional
        Scaling parameter for the heterogeneity map. If a heterogenity map is specified, the default
        is ``1.0``. Otherwise, this value is ignored (and is set to ``None``).
    scaling : str, optional
        Scaling function to apply to the heterogeneity map. Must be ``'sigmoid'`` or
        ``'exponential'``. If a heterogenity map is specified, the default is ``'sigmoid'``.
        Otherwise, this value is ignored (and is set to ``None``). 

    Raises
    ------
    ValueError
        If ``hetero`` length does not match the number of vertices (masked or unmasked).

    Notes
    -----
    The coordinates of vertices in ``geometry`` are assumed to be in millimetres, following the
    convention of common neuroimaging software.

    References
    ----------
    ..  [1] Pang, J. C., et al. (2023). Geometric constraints on human brain function. Nature.
        https://doi.org/10.1038/s41586-023-06098-1
    ..  [2] Reuter, M., et al. (2006). Laplace-Beltrami spectra as 'Shape-DNA' of surfaces and
        solids, Computer-Aided Design. https://doi.org/10.1016/j.cad.2005.10.011
    ..  [3] Wachinger, C., et al. (2015). BrainPrint: a discriminative characterization of brain
        morphology, Neuroimage. https://doi.org/10.1016/j.neuroimage.2015.01.032
    ..  [4] Barnes, V., et al. (2026). Regional heterogeneity shapes macroscopic wave dynamics of
        the human and non-human primate cortex. bioRxiv. https://doi.org/10.64898/2026.01.22.701178
    """
    def __init__(
        self,
        geometry: str | Path | GiftiImage | TriaMesh | dict,
        mask: NDArray[bool_] | None = None,
        normalize: bool = False,
        hetero: NDArray[floating] | None = None,
        alpha: float | None = None, # default to 1.0 if hetero given (and remains None)
        scaling: Literal['sigmoid', 'exponential'] | None = None  # default to "sigmoid" if hetero given (and remains None)
    ):
        # Read in surface mesh
        geometry = read_surf(geometry)

        # Optionally mask
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)  # chkfinite in mask_mesh
            geometry = mask_mesh(geometry, mask)

        # Optionally normalize
        if normalize:
            geometry.normalize_()  # LaPy method
        
        # Validate mesh
        check_surf(geometry)

        # Hetero inputs
        if hetero is None:
            if scaling is not None:
                warn("scaling is ignored as hetero is None.")
            if alpha is not None:
                warn("alpha is ignored as hetero is None.")
        else:
            hetero = np.asarray(hetero)  # chkfinite in scale_hetero
            alpha = 1.0 if alpha is None else float(alpha)
            scaling = "sigmoid" if scaling is None else scaling

            # Ensure hetero has correct length (masked or unmasked)
            if mask is not None and hetero.shape == (len(mask),):
                hetero = hetero[mask]
            elif hetero.shape != (geometry.v.shape[0],):
                err_str = f"the number of vertices in the provided mesh ({geometry.v.shape[0]})"
                if mask is not None:
                    err_str += f" or the masked mesh ({mask.sum()})"
                raise ValueError(f"hetero must be a 1D array with length matching {err_str}.")

            # Scale the heterogeneity map
            hetero = scale_hetero(hetero, alpha=alpha, scaling=scaling)

        # Assign attributes
        self.geometry = geometry
        self.n_verts = geometry.v.shape[0]  # Nicety
        self.mask = mask
        self.hetero = hetero
        self._scaling = scaling    
        self._alpha = alpha
        self.use_cholmod = False  # Permit lapy.eigs()

    def __str__(self) -> str:
        """String representation of the ``EigenSolver`` object."""
        # Prepare mesh info
        geom_type = "Surface"
        elem_type = "triangles"

        # Construct output
        str_out = (
            'EigenSolver\n'
            '-----------\n'
            f'{geom_type} mesh: {self.n_verts} vertices'
            )
        if self.mask is not None:
            str_out += f' ({np.sum(self.mask == 0)} others masked out)'
        str_out += f', {self.geometry.t.shape[0]} {elem_type}'

        if self.hetero is not None:
            str_out += f'\nHeterogeneity map scaling: {self._scaling} (alpha={self._alpha})'

        str_out += f'\n{self.n_modes if hasattr(self, "emodes") else "No"} eigenmodes computed'

        return str_out

    def compute_lbo(
        self, 
        lump: bool = False
    ) -> EigenSolver:
        """
        This method computes the Laplace-Beltrami operator using finite element methods on a
        triangular or tetrahedral mesh, optionally incorporating spatial heterogeneity.
        The resulting ``stiffness`` and ``mass`` matrices are stored as attributes.

        Parameters
        ----------
        lump : bool, optional
            Whether to compute a lumped (i.e., diagonal) mass matrix. Default is ``False``.

        Returns
        -------
        EigenSolver
            The ``EigenSolver`` instance.
        """
        if self.hetero is None:
            stiffness, mass = self._fem_tria(self.geometry, lump)
        else:
            # Get principal curvatures to define direction of anisotropy
            # Note: change of basis into (u1, u2) is not strictly needed for our isotropic
            # diffusion tensor, but _fem_tria_aniso performs it
            u1, u2, _, _ = self.geometry.curvature_tria()

            # Map hetero from vertices to triangles by averaging
            hetero_tria = self.geometry.map_vfunc_to_tfunc(self.hetero)

            # Construct isotropic diffusion tensor by using hetero for both u1 and u2 directions
            hetero_mat = np.stack((hetero_tria, hetero_tria), axis=1)

            # Compute FEM matrices under heterogeneous LBO
            stiffness, mass = self._fem_tria_aniso(self.geometry, u1, u2, hetero_mat, lump)
                
        # Assign attributes and return instance to allow chaining
        self.stiffness = stiffness
        self.mass = mass
        return self

    def solve(
        self,
        n_modes: int, 
        standardize: bool = True,
        fix_mode1: bool = True,
        lump: bool = False,
        atol: float = 1e-3,
        rtol: float = 1e-5,
        sigma: float = -0.01, # EASIEST way is to hard-code this to LaPy default (2026/03)
        seed: int | Generator | None = 0, 
        v0: NDArray[floating] | None = None
    ) -> EigenSolver:
        """
        Solves the generalized eigenvalue problem for the Laplace-Beltrami operator and compute
        eigenvalues and eigenmodes, which are stored as attributes (``emodes`` and ``evals``).

        Parameters
        ----------
        n_modes : int
            Number of eigenmodes to compute. Must be a positive integer less than the number of
            vertices.
        standardize : bool, optional
            If ``True``, standardizes the sign of the eigenmodes so the first element is positive.
            Default is ``False``.
        fix_mode1 : bool, optional
            If ``True``, sets the first eigenmode to a constant value and the first eigenvalue to
            zero, as is expected analytically. Default is ``True``. See the ``is_orthonormal_basis``
            function for details.
        lump: bool, optional
            Whether to use a lumped mass matrix for the Laplace-Beltrami operator. Default is
            ``False``.
        atol : float, optional
            Absolute tolerance for mass-orthonormality validation. Default is ``1e-3``.
        rtol : float, optional
            Relative tolerance for mass-orthonormality validation. Default is ``1e-5``.
        sigma : float, optional
            Shift-invert parameter to speed up the computation of eigenvalues close to this value.
            Default is ``-0.01``.
        seed : int or numpy.random.Generator, optional
            Random seed for the generation of the initialization vector (see below). If ``None``,
            computed eigenmodes and eigenvalues will not be exactly identical across runs. Most
            notably, the (arbitrary) signs of modes may flip. Default is ``None``.
        v0 : array-like, optional
            Initialization vector of shape ``(n_verts,)`` for the iterative solver. If ``None``, a
            vector sampled uniformly over [-1, 1] will be generated using the specified ``seed``.
            This parameter takes priority over ``seed``. Default is ``None``.

        Returns
        -------
        EigenSolver
            The ``EigenSolver`` instance.

        Raises
        ------
        ValueError
            If ``n_modes`` is not a positive integer less than ``n_verts``.
        ValueError
            If ``v0`` is provided but does not have shape ``(n_verts,)``.
        """
        # Validate arguments
        if n_modes != int(n_modes) or n_modes <= 0 or n_modes >= self.n_verts:
            raise ValueError("n_modes must be a positive integer less than the number of vertices"
                             f" ({self.n_verts}).")

        # Compute the Laplace-Beltrami operator / set stiffness and mass matrices
        self.compute_lbo(lump)
        
        # Setup intitialization vector
        if v0 is not None:
            v0 = np.asarray_chkfinite(v0)
            if v0.shape != (self.n_verts,):
                raise ValueError(f"v0 must have shape (n_verts,) = {(self.n_verts,)}.")

        # Solve the eigenvalue problem
        # can't pass sigma = None to LaPy
        evals, emodes = self.eigs(k=n_modes, sigma=sigma, v0=v0, rng=seed)

        # Validate results
        if not is_orthonormal_basis(emodes, self.mass, atol=atol, rtol=rtol, checks=False):
            warn(f"Computed eigenmodes are not mass-orthonormal (atol={atol}, rtol={rtol}).")

        ## Post-process
        if fix_mode1:
            if sigma >= 0:
                warn("Mode 1 will not be fixed to a constant when sigma >= 0, as the constant mode "
                     "may not be among the computed modes.")
            else:
                # Value given by mass-orthonormality condition
                emodes[:, 0] = self.mass.sum()**(-0.5)
                evals[0] = 0.0

        if standardize:
            emodes = standardize_emodes(emodes, checks=False)

        # Store results
        self.n_modes = n_modes  # Nicety
        self.evals = evals
        self.emodes = emodes
        return self

    def _check_for_emodes(self) -> None:
        if not hasattr(self, 'emodes'):
            raise ValueError("Eigenmodes not found. Please run the solve() method first.")
    
    def decompose(
        self,
        data: NDArray[floating],
        **kwargs
    ) -> NDArray[floating]:
        """
        This is a wrapper for :func:`~neuromodes.basis.decompose`. Note that ``emodes``, ``mass``,
        and ``checks`` are passed automatically by the ``EigenSolver`` instance.
        """
        from neuromodes.basis import decompose

        self._check_for_emodes()
    
        return decompose(
            data=data,
            emodes=self.emodes,
            mass=self.mass,
            checks='maps',
            **kwargs
        )
    
    def reconstruct(
        self,
        data: NDArray[floating],
        **kwargs
    ) -> _ReconSingle | _ReconList:
        """
        This is a wrapper for :func:`~neuromodes.basis.reconstruct`. Note that ``emodes``, ``mass``,
        and ``checks`` are passed automatically by the ``EigenSolver`` instance.
        """
        from neuromodes.basis import reconstruct
        
        self._check_for_emodes()
            
        return reconstruct(
            data=data,
            emodes=self.emodes,
            mass=self.mass,
            checks='maps',
            **kwargs
        )
    
    def reconstruct_timeseries(
        self,
        timeseries: NDArray[floating],
        **kwargs
    ) -> _ReconTSSingle | _ReconTSList:
        """
        This is a wrapper for :func:`~neuromodes.basis.reconstruct_timeseries`. Note that
        ``emodes``, ``mass``, and ``checks`` are passed automatically by the ``EigenSolver``
        instance.
        """
        from neuromodes.basis import reconstruct_timeseries

        self._check_for_emodes()
            
        return reconstruct_timeseries(
            timeseries=timeseries,
            emodes=self.emodes,
            mass=self.mass,
            checks='maps',
            **kwargs
        )
    
    def compute_gem(
        self,
        **kwargs
    ) -> NDArray[floating]:
        """
        This is a wrapper for :func:`~neuromodes.network.compute_gem`. Note that ``emodes``,
        ``evals``, and ``checks`` are passed automatically by the ``EigenSolver`` instance.
        """
        from neuromodes.network import compute_gem

        self._check_for_emodes()

        return compute_gem(
            emodes=self.emodes,
            evals=self.evals,
            checks=False,
            **kwargs
        )
    
    def sim_nft_waves(
        self,
        **kwargs
    ) -> NDArray[floating]:
        """
        This is a wrapper for :func:`~neuromodes.waves.sim_nft_waves`. Note that ``emodes``,
        ``evals``, ``mass``, ``scaled_hetero``, and ``checks`` are passed automatically by the
        ``EigenSolver`` instance.
        """
        from neuromodes.waves import sim_nft_waves

        self._check_for_emodes()

        return sim_nft_waves(
            emodes=self.emodes,
            evals=self.evals,
            mass=self.mass,
            scaled_hetero=self.hetero,
            checks='maps',
            **kwargs
        )
    
    def balloon_model(
        self,
        activity: NDArray[floating],
        dt: float,
        **kwargs
    ) -> NDArray[floating]:
        """
        This is a wrapper for :func:`~neuromodes.waves.balloon_model`. Note that ``emodes``,
        ``mass``, and ``checks`` are passed automatically by the ``EigenSolver`` instance.
        """
        from neuromodes.waves import balloon_model

        self._check_for_emodes()

        return balloon_model(
            activity=activity,
            dt=dt,
            emodes=self.emodes,
            mass=self.mass,
            checks='maps',
            **kwargs
        )
    
    def unmask_data(
        self,
        data: NDArray[floating],
        **kwargs
    ) -> NDArray[floating]:
        """
        This is a wrapper for :func:`~neuromodes.mesh.unmask_data`. Note that ``mask`` is passed
        automatically by the ``EigenSolver`` instance.
        """
        from neuromodes.mesh import unmask_data

        if self.mask is None:
            raise ValueError("No mask found. This method is only applicable for masked meshes.")

        return unmask_data(
            data=data,
            mask=self.mask,
            **kwargs
        )
    
    def eigenstrap(
        self,
        data: NDArray[floating],
        **kwargs
    ) -> NDArray[floating]:
        """
        This is a wrapper for :func:`~neuromodes.nulls.eigenstrap`. Note that `emodes`, `evals`,
        `mass`, and `checks` are passed automatically by the `EigenSolver` instance.
        """
        from neuromodes.nulls import eigenstrap

        self._check_for_emodes()

        return eigenstrap(
            data=data,
            emodes=self.emodes,
            evals=self.evals,
            mass=self.mass,
            checks='maps',
            **kwargs
        )

def scale_hetero(
    hetero: NDArray[floating],
    alpha: float = 1.0,
    scaling: Literal['sigmoid', 'exponential'] = "sigmoid"
) -> NDArray[floating]:
    """
    Scales a heterogeneity map using specified normalization and scaling functions.
    
    Parameters
    ----------
    hetero : array-like
        The heterogeneity map to be scaled.
    alpha : float, optional
        Scaling parameter controlling the strength of the transformation. Default is ``1.0``.
    scaling : str, optional
        The scaling function to apply to the heterogeneity map, either ``'sigmoid'`` or
        ``'exponential'``. Default is ``'sigmoid'``.
    
    Returns
    -------
    ndarray
        The scaled heterogeneity map.

    Raises
    ------
    ValueError
        If ``hetero`` is not a 1D array.
    ValueError
        If ``scaling`` is not ``'exponential'`` or ``'sigmoid'``.
    """
    # Format / validate arguments
    hetero = np.asarray_chkfinite(hetero)
    alpha = float(alpha)
    if hetero.ndim != 1:
        raise ValueError("hetero must be a 1D array.")
    if scaling not in ["exponential", "sigmoid"]:
        raise ValueError(f"Invalid scaling '{scaling}'. Must be 'exponential' or 'sigmoid'.")
    if alpha == 0:
        warn("alpha is set to 0, meaning hetero will have no effect.")
    std = np.std(hetero)
    if std == 0:
        warn("Provided hetero is constant; scaling hetero to a vector of ones.")
        hetero_scaled = np.ones_like(hetero)
    else:
        # Scale the heterogeneity map
        hetero_z = (hetero - np.mean(hetero)) / std
        hetero_scaled = (2 / (1 + np.exp(-alpha * hetero_z))
                         if scaling == 'sigmoid' else np.exp(alpha * hetero_z))
    
    return hetero_scaled

def standardize_emodes(
    emodes: NDArray[floating],
    checks: bool = True
) -> NDArray[floating]:
    """
    Flips the modes' signs such that the first element of each eigenmode has positive amplitude. 
    Note that the sign of each mode is arbitrary--standardisation is only helpful to compare sets of
    eigenmodes.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape ``(n_verts, n_modes)``, where n_modes is the number of
        eigenmodes.
    checks : bool, optional
        Whether to validate the shape and type of ``emodes``. Default is ``True``.

    Returns
    -------
    numpy.ndarray
        The standardized eigenmodes array of shape ``(n_verts, n_modes)``, with the first element of
        each mode set to be positive.
    """
    if checks:
        emodes = EigenData(emodes=emodes).emodes
   
    # Flip modes where the first element is negative
    return emodes * np.copysign(1, np.sign(np.asarray(emodes)[0, :]))

def is_orthonormal_basis(
    emodes: NDArray[floating],
    mass: csc_matrix | None = None,
    atol: float = 1e-03,
    rtol: float = 1e-05,
    checks: bool | str = 'shape'
) -> bool:
    """
    Check if a set of vectors is orthonormal in Euclidean space (i.e., ``emodes.T @ emodes == I``,
    where ``I`` is an identity matrix) or with respect to a mass matrix (i.e., ``emodes.T @ mass @
    emodes == I``). Mass-orthonormality is expected for the geometric eigenmodes (see notes).

    Parameters
    ----------
    emodes : array-like
        The vectors array of shape ``(n_verts, n_modes)``, where n_modes is the number of vectors.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)``. If ``None``, Euclidean orthonormality is
        checked. Default is ``None``.
    atol : float, optional
        Absolute tolerance for the orthonormality check. Default is ``1e-3``.
    rtol : float, optional
        Relative tolerance for the orthonormality check. Default is ``1e-5``.
    checks : bool | str, optional
        Whether to validate the shape and type of ``emodes`` and ``mass``. Default is ``True``.

    Returns
    -------
    bool
        ``True`` if the set of vectors is orthonormal (Euclidean or mass-orthonormal), ``False``
        otherwise.

    Notes
    -----
    Under discretization, the set of solutions for any generalized eigenvalue problem ``stiffness @
    emodes = - evals * mass @ emodes`` is expected to be mass-orthonormal, rather than orthonormal
    with respect to the standard Euclidean inner (dot) product. It follows that the first mode is
    expected to be a specific constant, but precision error during computation can introduce
    spurious spatial heterogeneity. Since many eigenmode analyses rely on mass-orthonormality (e.g.,
    decomposition, wave simulation), this function serves to ensure the validity of any calculated
    or provided eigenmodes.
    """
    # Format / validate arguments
    if checks is not False: 
        ved = EigenData(emodes=emodes, mass=mass, checks=checks)
        emodes, mass = ved.emodes, ved.mass

    # Check Euclidean or mass-orthonormality
    prod = emodes.T @ emodes if mass is None else emodes.T @ (mass @ emodes)
    identity = np.eye(emodes.shape[1])
    return np.allclose(prod, identity, rtol=rtol, atol=atol, equal_nan=False)

def truncate_emodes(geometry: TriaMesh, vfunc, threshold, 
                    evals=None, emodes=None, mass=None, threshold_method='decompose',
                    threshold_kwargs=None, output='group'): 

    from neuromodes.basis import decompose, reconstruct#, reconstruction_error

    # Prelims
    vf = np.asarray(vfunc, copy=True)
    vf = vf if vf.ndim > 1 else vf[:, np.newaxis] # ensure 2d for consistent processing
    n_maps = vf.shape[1]
    
    physical_threshold = threshold_method in ['eigenvalue', 'wavelength', 'fwhm']

    if threshold is None:
        if physical_threshold:
            thresholds = _convert_spatial_scale(estimate_fwhm(geometry, vf), input='fwhm', output=threshold_method)
        else: 
            raise ValueError(f"Threshold must be provided for threshold_method={threshold_method}.")
    elif np.isscalar(threshold):
        thresholds = np.full(n_maps, threshold)
    elif len(threshold) != n_maps:
        raise ValueError("Length of threshold array must match number of maps.")
    else: 
        thresholds = np.asarray(threshold)

    if threshold_kwargs is None: # have to set this outside the function declaration as {} is mutable
        threshold_kwargs = {}

    # Get data to threshold against
    if threshold_method == 'decompose':
        if emodes is None or mass is None:
            raise ValueError(f"emodes and mass must be provided when using threshold_method='{threshold_method}'.")
        betas = decompose(vf, emodes=emodes, mass=mass, **threshold_kwargs)
        ascending_data = np.cumsum(betas**2, axis=0) # user needs to demean if desired
        total_power = np.sum(vf * (mass @ vf), axis=0) # = np.diag(vf.T @ mass @ vf)
        thresholds = total_power * (1 - thresholds)
    elif threshold_method == 'reconstruct':
        if emodes is None or mass is None:
            raise ValueError(f"emodes and mass must be provided when using threshold_method='{threshold_method}'.")
        # TODO : change to new reconstruct
        # recons = reconstruct(data=vf, emodes=emodes, mass=mass, mode_counts=np.arange(1, emodes.shape[1]+1))
        # errors = reconstruction_error(vf, recon=recons, mass=mass, **threshold_kwargs)
        _, errors, _ = reconstruct(data=vf, emodes=emodes, mass=mass, mode_counts=np.arange(1, emodes.shape[1]+1), **threshold_kwargs)
        ascending_data = -errors.T # make this (n_modes, n_maps)
        thresholds = -thresholds
    elif threshold_method == 'eigenvalue':
        if evals is None:
            raise ValueError(f"evals must be provided when using threshold_method='{threshold_method}'.")
        ascending_data = np.broadcast_to(evals[:, np.newaxis], (len(evals), n_maps))
    elif threshold_method == 'fwhm' or threshold_method == 'wavelength': 
        if evals is None:
            raise ValueError(f"evals must be provided when using threshold_method='{threshold_method}'.")
        data = np.full(len(evals), np.inf) # default to inf for zero evals to avoid divide-by-zero issues
        data[1:] = convert_from_eigenvalue(evals[1:], output=threshold_method)
        ascending_data = -np.broadcast_to(data[:, np.newaxis], (len(evals), n_maps))
        thresholds = -thresholds
    else: 
        raise ValueError("Incorrect threshold_method specified.")

    # Find the required mode for each map
    # If the threshold is physical, keep only the modes that are strictly below the threshold
    # It the threshold is statistical, use the first mode that crosses the threshold 
    side = 'right' if physical_threshold else 'left' 
    n_mode = [np.searchsorted(ascending_data[:,i], thresholds[i], side=side) for i in range(n_maps)]
    n_mode = np.array(n_mode)
    if (failed_indices := np.where(n_mode == ascending_data.shape[0])[0]).size > 0:
        warn(f"Threshold not met for map(s) [{', '.join(map(str, failed_indices))}]. "
             f"All available modes were used. Consider providing more modes or loosening the threshold.")
    n_mode += not physical_threshold # this is the number of modes to use i.e. the first excluded mode

    # Return
    if output == 'mode': 
        result = n_mode
    elif output == 'group':
        result = mode_to_group(n_mode-1, method='ceil')+1 # number of groups (of last included mode)
    else: 
        raise ValueError("Output must be either 'mode' or 'group'.")
    
    return result if vfunc.ndim > 1 else result.item() # type: ignore # result will only be scalar if vfunc.ndim=1

def estimate_fwhm(surf: TriaMesh | Solver | EigenSolver, vfunc, method='fem', roi_mask=None):
    if method == 'wb':
        if isinstance(surf, TriaMesh):
            return _estimate_fwhm_wb(surf, vfunc, roi_mask=roi_mask)
        elif isinstance(surf, EigenSolver): 
            return _estimate_fwhm_wb(surf.geometry, vfunc, roi_mask=roi_mask) 
        elif isinstance(surf, Solver):
            raise TypeError("Solver instances cannot be used as they do not have a geometry attribute.")
        else:
            raise TypeError("For 'wb' method, geometry must be a TriaMesh or EigenSolver instance.")
        
    elif method == 'fem':
        if isinstance(surf, (Solver, EigenSolver)):
            stiffness, mass = surf.stiffness, surf.mass
        elif isinstance(surf, TriaMesh):
            stiffness, mass = Solver._fem_tria(surf) # doesn't matter if lumped or not
        else:
            raise TypeError("For 'fem' method, geometry must be a Solver/EigenSolver/TriaMesh instance.")
        return _estimate_fwhm_fem(stiffness, mass, vfunc, roi_mask=roi_mask)
    
    else:
        raise ValueError("Method must be either 'wb' or 'fem'.")

def _estimate_fwhm_wb(geometry: TriaMesh, vfunc, roi_mask=None):
    """Equivalent to wb_command -metric-estimate-fwhm"""
    # Prelims
    rois = np.ones(len(geometry.v), dtype=bool) if roi_mask is None else np.asarray(roi_mask, dtype=bool)
    vf = vfunc[rois, ...]

    # Get edges
    adj = geometry.adj_sym[rois, :][:, rois]
    rows, cols = adj.nonzero()
    rows, cols = rows[rows>cols], cols[rows>cols]  # Keep only upper triangle indices

    # Main computation
    vg = np.var(vf, axis=0, ddof=0)                           # global variance
    vl = np.mean((vf[rows, ...] - vf[cols, ...])**2, axis=0)  # local variance
    
    # In accordance with wb_command, use whole mesh mean edge length (not just mask)
    lambda_eff = 4 * -np.log(1 - vl / (2 * vg)) / geometry.avg_edge_length()**2
    return convert_from_eigenvalue(lambda_eff, output='fwhm') # alternate expression, but same as wb_command

def _estimate_fwhm_fem(stiffness, mass, vfunc, roi_mask=None):
    if roi_mask is not None: # subset stiffness, mass, and vfunc to roi (set diags to correct values)
        idx = np.asarray(roi_mask, dtype=bool)
        vf = vfunc[idx, ...]
        
        S = stiffness[idx, :][:, idx]
        S.setdiag(0)
        S.setdiag(-np.asarray(S.sum(axis=0)).ravel())

        M = mass[idx, :][:, idx]
        m_sum = np.asarray(M.sum(axis=0)).ravel()
        if M.nnz > M.shape[0]: # consistent
            target_mass = np.asarray(mass[:, idx].sum(axis=0)).ravel()
            M.setdiag(M.diagonal() + (target_mass - m_sum))
        else: # lumped
            M.setdiag(m_sum)

    else: 
        S, M, vf = stiffness, mass, vfunc

    # Vm = \Sigma_{i=1}^N \beta_i^2 (where \beta_i is the coefficient of mode i in the decomposition of vfunc)
    # Vs = \Sigma_{i=1}^N \beta_i^2 \lambda_i (where \lambda_i is the eigenvalue of mode i)
    # Vs/Vm = \lambda_{eff} where lambda is the effective eigenvalue of the map (weighted average)
    # If the input is a mode i, then this reduces to \lambda_i for that mode
    # TODO: change to demeanw / stats.py etc
    vf -= np.average(vf, weights=M.diagonal(), axis=0) # set (mass-weighted) mean to 0
    Vm = vf * (M @ vf)
    Vs = vf * (S @ vf) - 0.5 * (S @ vf**2) # keep second term for correction on open meshes
    lambda_eff = np.sum(Vs, axis=0) / np.sum(Vm, axis=0)
    return convert_from_eigenvalue(lambda_eff, output='fwhm')

def _estimate_fwhm_fem_local(stiffness, mass, vfunc):
    # TODO: change to demeanw / stats.py etc
    vf = vfunc - np.average(vfunc, weights=mass.diagonal(), axis=0) # set (mass-weighted) mean to 0
    Vm = vf * (mass @ vf)
    Vs = vf * (stiffness @ vf) - 0.5 * (stiffness @ vf**2)
    lambda_eff = Vs / Vm
    return convert_from_eigenvalue(lambda_eff, output='fwhm')
      

def get_eigengroup_inds(
    n_modes: int,
    ) -> list[NDArray[signedinteger]]:
    """
    Identify eigengroups based on ordering of spherical harmonics. Each eigengroup 
    contains the next 2k+1 modes, where k is the eigengroup number (starting from 0). If
    n_modes does not include a complete eigengroup, the final group will contain the 
    remaining modes.
    
    Parameters
    ----------
    n_modes : int
        The number of eigenmodes, which determines the grouping.
    
    Returns
    -------
    list of list of int
        A list where each element is a list of indices corresponding to the modes in that 
        eigengroup.
    """
    i = np.arange(n_modes)
    g = np.floor(np.sqrt(i)).astype(int)
    idx = [i[g == k] for k in np.unique(g)]

    return idx

# TODO : clarify that for inputs of the form n^2-1, all methods return n-1 
# TODO : add overloads for int vs array inputs and int/float outputs
def mode_to_group(mode_id: int | NDArray, method: str = 'ceil') -> int | float | NDArray:
    """
    Translates a linear mode index to its spherical harmonic group index.
    
    Parameters
    ----------
    mode_id : int or array_like
        The 0-indexed position of the eigenmode(s).
    method : {'round', 'floor', 'ceil', 'raw'}, optional
        How to resolve halfway points inside a degenerate group.
        'raw' returns the exact continuous fractional position.
    """
    # For method, get the (function, outputtype)
    opts = {
        'ceil':  (np.ceil, int),        # includes current group if mode_id is anywhere in it
        'floor': (np.floor, int),       # includes current group only if it is complete
        'round': (np.round, int),       # rounds to nearest group (if half, includes current group)
        'raw':   (lambda x: x, float)   # gives non-integer group index
    }

    # Setup
    if method not in opts:
        raise ValueError(f"Method must be one of {list(opts.keys())}")
    func, outtype = opts[method]
    mode_id_arr = np.asarray(mode_id)
    if not np.issubdtype(mode_id_arr.dtype, np.integer):
        warn("mode_id should be an integer or array of integers.")
    
    # Calcs
    result = func(np.sqrt(mode_id_arr + 1)).astype(outtype) - 1
    return result.item() if np.isscalar(mode_id) else result

# TODO : clarify that for inputs of the form n, all methods return (n+1)^2-1=n(n+2)
# TODO : add overloads for int vs array inputs and int/float outputs
def group_to_mode(group_id: int | float | NDArray, method: str = 'ceil') -> int | float | NDArray:
    """
    Translates a spherical harmonic group index back to a linear mode index.
    
    Parameters
    ----------
    group_id : int, float, or array_like
        The index of the harmonic group(s).
    method : {'ceil', 'floor', 'round', 'raw'}, optional
        How to resolve fractional modes. Default is 'ceil', which maps 
        an integer group ID to its terminal (final) mode index.
        'raw' mathematically inverts a fractional group back to its exact mode.
    """
    # For method, get the (function, outputtype)
    opts = {
        'ceil':  (np.ceil, int),        # modes up to and including all of the current group
        'floor': (np.floor, int),       # includes current group only if it is complete
        'round': (np.round, int),       # rounds to nearest complete group
        'raw':   (lambda x: x, float)   # gives (possibly) non-integer mode index
    }

    # Setup
    if method not in opts:
        raise ValueError(f"Method must be one of {list(opts.keys())}")
    func, outtype = opts[method]
    
    # Calcs
    result = func(np.asarray(group_id) + 1).astype(outtype)**2 - 1
    return result.item() if np.isscalar(group_id) else result

def convert_to_eigenvalue(value, input, area=None):
    if input == 'eigenvalue': 
        return value
    elif input == 'wavelength':
        return (2 * np.pi / value)**2
    elif input == 'fwhm':
        return 8 * np.log(2) / value**2
    elif input == 'group':
        if area is None:
            raise ValueError("Area must be provided when input is 'group'.")
        return value * (value + 1) * 4 * np.pi / area
    elif input == 'mode':
        if area is None:
            raise ValueError("Area must be provided when input is 'mode'.")
        return 4 * np.pi * value / area 
    else:
        raise ValueError("Incorrect input specified")

def convert_from_eigenvalue(eigenvalue, output, area=None):
    if output == 'eigenvalue': 
        return eigenvalue
    elif output == 'wavelength':
        return 2 * np.pi / np.sqrt(eigenvalue)
    elif output == 'fwhm':
        return np.sqrt(8 * np.log(2) / eigenvalue)
    elif output == 'group':
        if area is None:
            raise ValueError("Area must be provided when output is 'group'.")
        return (np.sqrt(eigenvalue * area / np.pi + 1) - 1) / 2
    elif output == 'mode':
        if area is None:
            raise ValueError("Area must be provided when output is 'mode'.")
        return eigenvalue * area / (4 * np.pi)
    else:
        raise ValueError("Incorrect output specified")
    
# TODO : consider making public
def _convert_spatial_scale(data, input, output, area=None):
    """Convenience function to convert between spatial scale representations without needing to manually convert to eigenvalues."""
    eigenvalue = convert_to_eigenvalue(data, input=input, area=area)
    return convert_from_eigenvalue(eigenvalue, output=output, area=area)

_MISSING = object()  
@dataclass(frozen=True, init=False)
class EigenData:
    emodes: NDArray[floating]
    evals: NDArray[floating] 
    mass: csc_matrix
    stiffness: csc_matrix
    scaled_hetero: NDArray[floating]
    data: NDArray[floating]

    def __init__(
        self,
        emodes: NDArray[floating] | None = _MISSING, # type: ignore[assignment]
        evals: NDArray[floating] | None = _MISSING, # type: ignore[assignment] 
        mass: csc_matrix | None = _MISSING, # type: ignore[assignment]
        stiffness: csc_matrix | None = _MISSING, # type: ignore[assignment]
        scaled_hetero: NDArray[floating] | None = _MISSING, # type: ignore[assignment]
        data: NDArray[floating] | None = _MISSING, # type: ignore[assignment]
        checks: bool | str = True
    ):  # TODO: add mask?

        # Local helper to bypass 'frozen' restriction during initialization
        def _set(name, val):
            object.__setattr__(self, name, val)

        check_shape = checks is True or checks == 'shape' or checks == 'maps' # need to get first dim when checking maps
        check_maps = checks is True or checks == 'maps'
        check_ortho = checks is True or checks == 'ortho'
        check_evals = checks is True or checks == 'evals'

        all_inputs = []

        # Cast types and check shapes
        if emodes is not _MISSING:
            all_inputs.append('emodes')
            if emodes is not None:
                emodes = np.asarray_chkfinite(emodes)
                if check_shape:
                    if emodes.ndim != 2: 
                        raise ValueError("emodes must be a 2D array.")
                    if emodes.shape[0] <= emodes.shape[1]:
                        raise ValueError("emodes must have shape (n_verts, n_modes), where n_verts "
                                         "> n_modes.")
            _set('emodes', emodes)

        if evals is not _MISSING:
            if evals is not None:
                evals = np.asarray_chkfinite(evals)
                if check_shape: 
                    if emodes is not None and evals.shape != (emodes.shape[1],):
                        raise ValueError(f"evals must have shape (n_modes,) = ({emodes.shape[1]},).")
                if check_evals:
                    if (evals[1:] <= 0).any():
                        warn("Non-positive eigenvalues detected (beyond first eigenvalue). This "
                             "may indicate an issue with the computation.")
                    # Allow first eval to be slightly negative due to precision error
                    if np.abs(evals[0]) > 1e-6:
                        warn("The first eigenvalue is expected to be close to zero, received "
                             f"{evals[0]}.")
            _set('evals', evals)

        # TODO : add lump input and parameter (confirm that mass is diagonal if lump=True)
        if mass is not _MISSING:
            all_inputs.append('mass')
            if mass is not None and check_shape:
                if mass.ndim != 2 or mass.shape[0] != mass.shape[1]: # type: ignore[union-attr]
                    raise ValueError("mass must be a square matrix.")
            _set('mass', mass)

        if stiffness is not _MISSING:
            all_inputs.append('stiffness')
            if stiffness is not None and check_shape:
                if stiffness.ndim != 2 or stiffness.shape[0] != stiffness.shape[1]: # type: ignore[union-attr]
                    raise ValueError("stiffness must be a square matrix.")
            _set('stiffness', stiffness)

        if scaled_hetero is not _MISSING:
            all_inputs.append('scaled_hetero')
            if scaled_hetero is not None:
                scaled_hetero = np.asarray_chkfinite(scaled_hetero)
                if check_shape:
                    if scaled_hetero.ndim != 1:
                        raise ValueError("scaled_hetero must have shape (n_verts,).")
            _set('scaled_hetero', scaled_hetero)

        n_verts = None
        # Check first dimension of each map at the same time (after self.name is set)
        if check_shape:
            for name in all_inputs:
                val = getattr(self, name)
                if val is None or val is _MISSING:
                    continue
                
                first_dim = val.shape[0] # Sparse matrices and NDArrays both have .shape

                if n_verts is None:
                    # Establish the ground truth from the first available data source
                    n_verts = first_dim
                elif first_dim != n_verts:
                    raise ValueError(
                        f"Dimension mismatch in '{name}': "
                        f"expected first dimension {n_verts}, but got {first_dim}."
                    )
            
        if data is not _MISSING:
            if check_maps and data is not None: # if check_maps is True, always check the shape
                data = np.asarray(data)
                if np.isnan(data).any(): 
                    warn("NaN values detected in data, which may cause issues with computations.")
                if np.isinf(data).any():
                    warn("Inf values detected in data, which may cause issues with computations.")
                if n_verts is not None and data.shape[0] != n_verts:
                    raise ValueError(f"data must have first dimension {n_verts} to match the other "
                                     "variables.")
            _set('data', data)

        # Check mass-orthonormality
        if check_ortho and emodes is not _MISSING and emodes is not None:
            m = mass if mass is not _MISSING else None
            if not is_orthonormal_basis(emodes, m, checks=False):
                err_str = "in Euclidean space" if m is None else "with the provided mass matrix"
                raise ValueError(
                    f"The columns of emodes do not form an orthonormal basis set {err_str}. Either "
                    "provide a suitable mass matrix such that emodes.T @ mass @ emodes = I, use "
                    "the 'regress' method for decomposition, or set checks=False."
                )

    def __getattribute__(self, name: str) -> Any:
        val = super().__getattribute__(name)
        if val is _MISSING:
            raise AttributeError(f"'{name}' was not provided to this EigenData instance.")
        return val
