"""
Module for computing geometric eigenmodes of brain structures from surface meshes.
"""

from __future__ import annotations
from typing import Any, Literal, TYPE_CHECKING
from warnings import warn
from dataclasses import dataclass
from lapy import Solver
import numpy as np
from neuromodes.io import read_surf
from neuromodes.mesh import mask_mesh, check_surf

if TYPE_CHECKING:
    from pathlib import Path
    from lapy import TriaMesh
    from nibabel.gifti.gifti import GiftiImage
    from numpy import floating, integer, bool_
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
            stiffness=self.stiffness,
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

def get_eigengroup_inds(
    n_modes: int,
    ) -> list[NDArray[integer]]:
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
                    raise ValueError("Inf values detected in data, which are not allowed.")
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
