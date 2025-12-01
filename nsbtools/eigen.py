"""
Module for computing geometric eigenmodes on cortical surface meshes and decomposing/reconstructing 
cortical maps.
"""

from pathlib import Path
from warnings import warn
from typing import Optional, Union, Any, Tuple
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import sparse
from scipy.stats import zscore
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from trimesh import Trimesh
from lapy import Solver, TriaMesh
from nsbtools.io import read_surf, mask_surf

class EigenSolver(Solver):
    """
    EigenSolver class for spectral analysis and simulation on surface meshes.

    This class computes the Laplace-Beltrami operator on a triangular mesh via the Finite Element
    Method, which discretizes the eigenvalue problem according to mass and stiffness matrices.
    Spatial heterogeneity and various normalization/scaling options are supported.
    """

    def __init__(
        self,
        surf: Union[str, Path, Trimesh, TriaMesh, dict],
        mask: Optional[ArrayLike] = None,
        hetero: Optional[ArrayLike] = None,
        aniso: Optional[ArrayLike] = None,
        n_modes: int = 200,
        alpha: float = 1.0,
        beta: float = 1.0,
        beta_old: float = 1.0,
        r: float = 28.9,
        gamma: float = 0.116,
        scaling: str = "sigmoid",
        lump: bool = False,
        smoothit: int = 10,
        normalize: bool = False
    ):
        """
        Initialize the EigenSolver class.

        Parameters
        ----------
        surf : str, pathlib.Path, trimesh.Trimesh, lapy.TriaMesh, or dict
            The surface mesh to be used. Can be a file path to a supported format 
            (see io.read_surf), a supported mesh object, or a dictionary with 'vertices' and 'faces'
            keys.
        mask : array-like, optional
            A boolean mask to exclude certain points (e.g., medial wall) from the surface mesh.
            Default is None.
        hetero : array-like, optional
            A heterogeneity map to scale the Laplace-Beltrami operator. Default is None.
        aniso : array-like, optional
            Anisotropy map to define direction-dependent diffusion properties on the surface mesh.
        n_modes : int, optional
            Number of eigenmodes to compute. Default is 100.
        alpha_hetero : float, optional
            Scaling factor for the heterogeneity map. Default is 1.0.
        alpha_aniso : float, optional
            Scaling factor for anisotropic diffusion strength. Default is 1.0.
        beta : float, optional
            Exponent for the sigmoid scaling of the heterogeneity map. Default is 1.0.
        r : float, optional
            Axonal length scale for wave propagation, used to ensure plausible wave speeds for 
            activity simulation when hetero is provided. Default is 28.9.
        gamma : float, optional
            Damping parameter for wave propagation, used to ensure plausible wave speeds for
            activity simulation when hetero is provided. Default is 0.116.
        scaling : str, optional
            Scaling function to apply to the heterogeneity map. Must be "sigmoid" or "exponential". 
            Default is "sigmoid".
        lump : bool, optional
            Whether to use lumped mass matrix for the Laplace-Beltrami operator. Default is False.
        smoothit : int, optional
            Number of smoothing iterations for curvature calculation. Default is 10.
        normalize : bool, optional
            Whether to normalize the surface mesh. Default is False.

        Raises
        -------
        ValueError
            Raised if any input parameter is invalid, such as negative or non-numeric values for  
            `r`, `gamma`, or `beta`, or if `n_modes` is not a positive integer.
        """
        r = float(r)
        gamma = float(gamma)
        beta_old = float(beta_old)

        if r <= 0:
            raise ValueError("`r` must be positive.")
        if gamma <= 0:
            raise ValueError("`gamma` must be positive.")
        if beta < 0:
            raise ValueError("`alpha_aniso` must be non-negative.")
        if beta_old < 0:
            raise ValueError("`beta` must be non-negative.")
        if smoothit < 0 or not isinstance(smoothit, int):
            raise ValueError("`smoothit` must be a non-negative integer.")
        if n_modes <= 0 or not isinstance(n_modes, int):
            raise ValueError("`n_modes` must be a positive integer.")
        if not isinstance(lump, bool):
            raise ValueError("`lump` must be a boolean value.")
        
        self.n_modes = n_modes
        self._r = r
        self._gamma = gamma
        self.alpha_hetero = alpha if hetero is not None else 0
        self.alpha_aniso = beta if aniso is not None else 0
        self.beta = beta_old if hetero is not None else 0
        self.scaling = scaling
        self.smoothit = smoothit
        self.lump = lump

        # Initialize surface and convert to TriaMesh object
        surf = read_surf(surf)
        if mask is not None:
            self.mask = np.asarray(mask, dtype=bool)
            surf = mask_surf(surf, self.mask)
        else:
            self.mask = None
        self.geometry = TriaMesh(surf.vertices, surf.faces)
        if normalize:
            self.geometry.normalize_()
        self.n_verts = surf.vertices.shape[0]
        self.hetero = hetero
        self.aniso = aniso

        # Calculate the two matrices of the Laplace-Beltrami operator
        self.compute_lbo()

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, r: float) -> None:
        if not is_valid_hetero(hetero=self.hetero, r=r, gamma=self.gamma):
            raise ValueError(f"alpha_hetero value results in non-physiological wave speeds above 150 m/s. "
                             "Try using a smaller alpha_hetero value.")
        self._r = r

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float) -> None:
        if not is_valid_hetero(hetero=self.hetero, r=self.r, gamma=gamma):
            raise ValueError(f"alpha_hetero value results in non-physiological wave speeds above 150 m/s. "
                             "Try using a smaller alpha_hetero value.")
        self._gamma = gamma

    @property
    def hetero(self) -> NDArray:
        return self._hetero

    @hetero.setter
    def hetero(self, hetero: Optional[ArrayLike]) -> None:
        # Handle None case by setting to ones
        if hetero is None:
            if self.alpha_hetero != 0 or self.beta != 0:
                warn('Setting `alpha_hetero` and `beta` to 0 because `hetero` is None.')
                self.alpha_hetero = 0
                self.beta = 0
            self._hetero = np.ones(self.n_verts)
        else:
            hetero = np.asarray(hetero)

            # Ensure hetero has correct length
            n_expected = len(self.mask) if self.mask is not None else self.n_verts
            if len(hetero) != n_expected:
                raise ValueError(f"The number of elements in `hetero` ({len(hetero)}) must match "
                                 f"the number of vertices in the surface mesh ({n_expected}).")
                         
            if self.mask is not None:
                hetero = hetero[self.mask]
            # Check for NaN/Inf values
            if np.isnan(hetero).any() or np.isinf(hetero).any():
                raise ValueError("`hetero` must not contain NaNs or Infs.")
                
            # Scale the heterogeneity map
            hetero = scale_hetero(
                hetero=hetero, 
                alpha=self.alpha_hetero, 
                beta=self.beta,
                scaling=self.scaling
            )

            if not is_valid_hetero(hetero=hetero, r=self.r, gamma=self.gamma):
                raise ValueError(f"Alpha value results in non-physiological wave speeds above "
                                 "150 m/s. Try using a smaller alpha value.")

            # Assign to private attribute
            self._hetero = hetero

    @property
    def aniso(self) -> Optional[NDArray]:
        return self._aniso

    @aniso.setter
    def aniso(self, aniso: Optional[ArrayLike]) -> None:
        # Handle None case
        if aniso is None:
            self._aniso = None
        else:
            aniso = np.asarray(aniso)

            # Ensure aniso has correct length
            n_expected = len(self.mask) if self.mask is not None else self.n_verts
            if len(aniso) != n_expected:
                raise ValueError(f"The number of elements in `aniso` ({len(aniso)}) must match "
                                 f"the number of vertices in the surface mesh ({n_expected}).")
                         
            if self.mask is not None:
                aniso = aniso[self.mask]
            
            # Check for NaN/Inf values
            if np.isnan(aniso).any() or np.isinf(aniso).any():
                raise ValueError("`aniso` must not contain NaNs or Infs.")

            # Z-score the anisotropy map for consistent scaling
            aniso = zscore(aniso)

            # Assign to private attribute
            self._aniso = aniso

    def compute_lbo(self) -> None:   
        """
        This method computes the Laplace-Beltrami operator using finite element methods on a
        triangular mesh, optionally incorporating spatial heterogeneity and anisotropy.
        The resulting stiffness and mass matrices are stored as attributes.
        
        Notes
        -----
        When anisotropy is applied, the gradient components are cross-coupled with the tangent
        basis directions due to the FEM formulation. Specifically, the gradient component in the 
        u1 direction scales diffusion in the u2 direction, and vice versa. This corresponds to 
        the tensor transformation R^T D R where R = [0,-1; 1,0], a standard rotation required 
        by the finite element discretization. This ensures that positive alpha_aniso enhances 
        diffusion parallel to the gradient direction as intended.
        """
        from lapy.diffgeo import tria_compute_gradient
        
        hetero_tri = self.geometry.map_vfunc_to_tfunc(self.hetero)
        u1, u2, _, _ = self.geometry.curvature_tria(smoothit=self.smoothit)
        
        if self.aniso is not None:
            # Compute 3D gradient at each triangle
            grad_3d = tria_compute_gradient(self.geometry, self.aniso)
            
            # Project gradient onto local (u1, u2) coordinates
            grad_u1 = np.sum(grad_3d * u1, axis=1)  # component in u1 direction
            grad_u2 = np.sum(grad_3d * u2, axis=1)  # component in u2 direction
            
            # Compute gradient magnitude and normalize to [0, 1] range
            grad_mag = np.sqrt(grad_u1**2 + grad_u2**2)
            grad_mag_norm = grad_mag / np.maximum(grad_mag.max(), 1e-10)
            
            # Compute unit gradient directions (avoid division by zero)
            grad_u1_unit = grad_u1 / np.maximum(grad_mag, 1e-10)
            grad_u2_unit = grad_u2 / np.maximum(grad_mag, 1e-10)
            
            # Create anisotropic diffusion tensor with FEM-required cross-coupling:
            # The FEM formulation requires swapping gradient components with basis directions.
            # grad_u2_unit**2 (perpendicular to u2) scales diffusion in u1 direction
            # grad_u1_unit**2 (perpendicular to u1) scales diffusion in u2 direction
            # This implements the tensor transformation R^T D R where R = [0,-1; 1,0]
            aniso_u1 = 1 + (self.alpha_aniso * grad_mag_norm) * grad_u2_unit**2  
            aniso_u2 = 1 + (self.alpha_aniso * grad_mag_norm) * grad_u1_unit**2
            
            # Scale by heterogeneity to control overall diffusion speed
            aniso_mat = np.column_stack([
                hetero_tri * aniso_u1,  # when alpha_aniso=0, this becomes hetero_tri * 1
                hetero_tri * aniso_u2   # when alpha_aniso=0, this becomes hetero_tri * 1
            ])
        else:
            # Isotropic case (current implementation)
            aniso_mat = np.tile(hetero_tri[:, np.newaxis], (1, 2))
        
        self.stiffness, self.mass = self._fem_tria_aniso(self.geometry, u1, u2,
                                                         aniso_mat, self.lump)

    def solve(
        self,
        standardize: bool = True,
        fix_mode1: bool = True,
        atol: float = 1e-3,
        seed: Optional[Union[int, ArrayLike]] = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Solve the generalized eigenvalue problem for the Laplace-Beltrami operator and compute 
        eigenvalues and eigenmodes.

        Parameters
        ----------
        standardize : bool, optional
            If True, standardizes the sign of the eigenmodes so the first element is positive. 
            Default is False.
        fix_mode1 : bool, optional
            If True, sets the first eigenmode to a constant value and the first eigenvalue to zero, 
            as is expected analytically. Default is True. See the is_mass_orthonormal_modes function
            for details.
        atol : float, optional
            Absolute tolerance for mass-orthonormality validation. Default is 1e-3.
        seed : int or array-like, optional
            Random seed for reproducibility. Default is None.

        Returns
        -------
        numpy.ndarray
            The computed eigenmodes.
        numpy.ndarray
            The computed eigenvalues.

        Raises
        ------
        ValueError
            If `seed` is an array but does not have the correct shape of (n_verts,).
        AssertionError
            If any computed eigenvalues are NaN.
        """
        
        sigma = -0.01
        lu = splu(self.stiffness - sigma * self.mass)
        op_inv = LinearOperator(
            matvec=lu.solve,
            shape=self.stiffness.shape,
            dtype=self.stiffness.dtype,
        )

        if seed is None or isinstance(seed, int):
            rng = np.random.default_rng(seed)
            v0 = rng.random(self.n_verts)
        else:
            v0 = np.asarray(seed)
            if v0.shape != (self.n_verts,):
                raise ValueError("`seed` must be either an integer or an array-like of shape "
                                f"({self.n_verts},).")

        # Solve the eigenvalue problem
        self.evals, self.emodes = eigsh(
            self.stiffness,
            k=self.n_modes,
            M=self.mass,
            sigma=sigma,
            OPinv=op_inv,
            v0=v0
        )

        assert not np.isnan(self.evals).any(), "Eigenvalues contain NaNs."
        if self.evals[0] / self.evals[1] >= 0.01:
            warn("Unfixed first eigenvalue (analytically expected to be 0) is not at least"
                          " 100 times smaller than the second.")

        if not is_mass_orthonormal_modes(self.emodes, self.mass, atol=atol):
            warn(f"Computed eigenmodes are not mass-orthonormal (atol={atol}).")

        if fix_mode1:
            self.emodes[:, 0] = np.full(self.n_verts, 1 / np.sqrt(self.mass.sum()))
            self.evals[0] = 0.0
        if standardize:
            self.emodes = standardize_modes(self.emodes)

        return self.emodes, self.evals
    
    def decompose(
        self,
        data: ArrayLike,
        method: str = 'project',
        **kwargs
    ) -> NDArray:
        """
        Calculate the decomposition of the given data onto the geometric eigenmodes.

        Parameters
        ----------
        data : array-like
            The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices 
            and n_maps is the number of brain maps.
        method : str, optional
            The method used for the decomposition, either 'project' to project data into a 
            mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
            from 'regress' tend towards those from 'project' when EigenSolver is initialised with a 
            higher `n_modes`. For a non-orthonormal basis set, 'regress' must be used. Default is
            'project'.
        **kwargs
            Additional keyword arguments to be passed to the solve() method (standardize, fix_mode1,
            atol, seed).

        Returns
        -------
        numpy.ndarray
            The beta coefficients array of shape (n_modes, n_maps), obtained from the decomposition.
        
        Raises
        ------
        ValueError
            If the number of vertices in `data` and `self.emodes` do not match, or if an invalid 
            method is specified.
        """
        if not hasattr(self, 'emodes'):
            _ = self.solve(**kwargs)
            print("Solved Laplace-Beltrami eigenvalue problem, "
                  "stored in self.emodes and self.evals.")
    
        return decompose(
            data,
            self.emodes,
            method=method,
            mass=self.mass
        )
    
    def reconstruct(
        self,
        data: ArrayLike,
        method: str = 'project',
        mode_seq: Optional[ArrayLike] = None,
        timeseries: bool = False,
        metric: str = "pearsonr",
        return_all: bool = False,
        **kwargs
    ) -> Any:
        """
        Calculate and score the reconstruction of the given data using the geometric eigenmodes.

        Parameters
        ----------
        data : array-like
            The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices 
            and n_maps is the number of brain maps.
        method : str, optional
            The method used for the decomposition, either 'project' to project data into a 
            mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
            from 'regress' tend towards those from 'project' when EigenSolver is initialised with a 
            higher `n_modes`. For a non-orthonormal basis set, 'regress' must be used. Default is
            'project'.
        mode_seq : array-like, optional
            The sequence of modes to be used for reconstruction. Default is None, which 
            uses all modes provided.
        timeseries : bool, optional
            Whether to treat brain maps as a time series of activity and reconstruct the functional
            coupling matrix. Default is False.
        metric : str, optional
            The metric used for calculating reconstruction accuracy, either "pearsonr" or "mse".
            Default is "pearsonr".
        return_all : bool, optional
            Whether to return the reconstructed timepoints when timeseries is True. Default is
            False.
        **kwargs
            Additional keyword arguments to be passed to the solve() method (standardize, fix_mode1,
            atol, seed).
        
        Returns
        -------
        list of numpy.ndarray
            A list of beta coefficients calculated for each mode.
        numpy.ndarray
            The reconstructed data array of shape (n_verts, nq, n_maps).
        numpy.ndarray
            The correlation coefficients array of shape (nq, n_maps).
        numpy.ndarray, optional
            The functional connectivity reconstructed data array of shape (n_verts, n_verts, nq). 
            Returned only if data_type is "timeseries".
        numpy.ndarray, optional
            The functional connectivity correlation coefficients array of shape (nq,). Returned only
            if data_type is "timeseries".
        
        Raises
        ------
        ValueError
            If the number of vertices in `data` and `self.emodes` do not match, or if an invalid 
            method is specified.
        """
        if not hasattr(self, 'emodes'):
            _ = self.solve(**kwargs)
            print("Solved Laplace-Beltrami eigenvalue problem, "
                  "stored in self.emodes and self.evals.")
            
        return reconstruct(
            data,
            self.emodes,
            method=method,
            mass=self.mass,
            mode_seq=mode_seq,
            timeseries=timeseries,
            metric=metric,
            return_all=return_all
        )
    
    def generate_connectome(
        self,
        r: float = 9.53,
        k: int = 108,
        **kwargs
    ) -> NDArray:
        """
        Generate a vertex-wise structural connectivity matrix using the Green's function approach
        described in Normand et al., 2025.

        Parameters
        ----------
        r : float, optional
            Spatial scale parameter for the Green's function. Default is 9.53.
        k : int, optional
            Number of eigenmodes to use. Default is 108.
        **kwargs
            Additional keyword arguments to be passed to the solve() method (standardize, fix_mode1,
            atol, seed).

        Returns
        -------
        numpy.ndarray
            The generated vertex-wise structural connectivity matrix.

        Raises
        ------
        ValueError
            If any input parameter is invalid, such as negative or non-numeric values for  
            `r`, or if `k` is not a positive integer within the valid range.

        Notes
        -----
        If comparing this model to empirical connectomes, consider thresholding the generated 
        connectome to match the density of the empirical data.
        """
        from nsbtools.connectome import generate_connectome

        if not hasattr(self, 'emodes'):
            _ = self.solve(**kwargs)
            print("Solved Laplace-Beltrami eigenvalue problem, "
                  "stored in self.emodes and self.evals.")

        return generate_connectome(
            emodes=self.emodes,
            evals=self.evals,
            r=r,
            k=k
        )
    
    def simulate_waves(
        self,
        ext_input: Optional[ArrayLike] = None,
        dt: float = 0.1,
        nt: int = 1000,
        bold_out: bool = False,
        decomp_method: str = "project",
        pde_method: str = "fourier",
        seed: Optional[int] = None,
        **kwargs
    ) -> NDArray:
        """
        Simulate neural activity or BOLD signals on the surface mesh using the eigenmode 
        decomposition. The simulation uses a Neural Field Theory wave model and optionally the
        Balloon-Windkessel model for BOLD signal generation. 

        Parameters
        ----------
        ext_input : array-like, optional
            External input array of shape (n_verts, n_timepoints). If None, random input is 
            generated.
        dt : float, optional
            Time step for simulation in milliseconds. Default is 0.1.
        nt : int, optional
            Number of time points to simulate Default is 1000.
        bold_out : bool, optional
            If True, simulate BOLD signal using the balloon model. If False, simulate neural 
            activity. Default is False.
        decomp_method : str, optional
            The method used for the eigendecomposition, either 'project' to project data into a 
            mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
            from 'regress' tend towards those from 'project' when EigenSolver is initialised with a 
            higher `n_modes`. For a non-orthonormal basis set, 'regress' must be used. Default is
            'project'.
        pde_method : str, optional
            Method for solving the wave PDE. Either "fourier" or "ode". Default is "fourier".
        seed : int, optional
            Random seed for generating external input. Default is None.
        **kwargs
            Additional keyword arguments to be passed to the solve() method (standardize, fix_mode1,
            atol, seed).

        Returns
        -------
        numpy.ndarray
            Simulated neural or BOLD activity of shape (n_verts, n_timepoints).

        Raises
        ------
        ValueError
            If the shape of ext_input does not match (n_verts, n_timepoints), or if either the
            eigen-decomposition or PDE method is invalid.

        Notes
        -----
        Since the simulation begins at rest, consider discarding the first 50 timepoints to allow 
        the system to reach a steady state.
        """
        from nsbtools.waves import simulate_waves

        if not hasattr(self, 'emodes'):
            _ = self.solve(**kwargs)
            print("Solved Laplace-Beltrami eigenvalue problem, "
                  "stored in self.emodes and self.evals.")

        return simulate_waves(
            emodes=self.emodes,
            evals=self.evals,
            r=self.r,
            gamma=self.gamma,
            ext_input=ext_input,
            dt=dt,
            nt=nt,
            mass=self.mass,
            bold_out=bold_out,
            decomp_method=decomp_method,
            pde_method=pde_method,
            seed=seed
        )

def is_valid_hetero(
    hetero: ArrayLike,
    r: float,
    gamma: float
) -> bool:
    """
    Check if the heterogeneity map values result in physiologically plausible wave speeds.
    
    Parameters
    ----------
    hetero : array_like
        Heterogeneity map values.
    r : float
        Axonal length scale for wave propagation.
    gamma : float
        Damping parameter for wave propagation.
    
    Raises
    ------
    ValueError
        If the computed wave speed exceeds 150 m/s, indicating non-physiological values.
    """
    hetero = np.asarray(hetero)
    max_speed = np.max(r * gamma * np.sqrt(hetero))

    return max_speed <= 150

def scale_hetero(
    hetero: ArrayLike,
    alpha: float = 1.0,
    beta: float = 1.0,
    scaling: str = "sigmoid"
) -> NDArray:
    """
    Scales a heterogeneity map using specified normalization and scaling functions.
    
    Parameters
    ----------
    hetero : array-like
        The heterogeneity map to be scaled.
    alpha : float, optional
        Scaling parameter controlling the strength of the transformation. Default is 1.0.
    scaling : str, optional
        The scaling function to apply to the heterogeneity map, either "sigmoid" or "exponential".
        Default is "sigmoid".
    
    Returns
    -------
    ndarray
        The scaled heterogeneity map.

    Raises
    ------
    ValueError
        If the scaling parameter is not a supported function.
    """
    hetero = np.asarray(hetero)

    # Z-score the heterogeneity map
    hetero_z = zscore(hetero)

    # Scale the heterogeneity map
    if scaling == "exponential":
        hetero_scaled = np.exp(alpha * hetero_z)
    elif scaling == "sigmoid":
        hetero_scaled = (2 / (1 + np.exp(-alpha * hetero_z)))**beta
    else:
        raise ValueError(f"Invalid scaling '{scaling}'. Must be 'exponential' or 'sigmoid'.")

    return hetero_scaled

def is_mass_orthonormal_modes(
    emodes: ArrayLike,
    mass: Optional[ArrayLike] = None,
    atol: float = 1e-3
) -> bool:
    """
    Check if a set of vectors is approximately mass-orthonormal 
    (i.e., `emodes.T @ mass @ emodes = I`).

    Parameters
    ----------
    emodes : array-like
        The vectors array of shape (n_verts, n_vecs), where n_vecs is the number of vectors.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts). If using EigenSolver, provide its self.mass. If 
        None, an identity matrix will be used, corresponding to Euclidean orthonormality. Default is 
        None.
    atol : float, optional
        Absolute tolerance for the orthonormality check. Default is 1e-3.

    Notes
    -----
    Under discretization, the set of solutions for the generalized eigenvalue problem is expected to
    be mass-orthogonal (mode_i^T * mass matrix * mode_j = 0 for i ≠ j), rather than orthogonal with
    respect to the standard Euclidean inner (dot) product (mode_i^T * mode_j = 0 for i ≠ j). 
    Eigenmodes are also expected to be mass-normal (mode_i^T * mass matrix * mode_i = 1). It follows
    that the first mode is expected to be a specific constant, but precision error during
    computation can introduce spurious spatial heterogeneity. Since many eigenmode analyses rely on
    mass-orthonormality (e.g., decomposition, wave simulation), this function serves to ensure the
    validity of any calculated or provided eigenmodes.
    """
    emodes = np.asarray(emodes)

    prod = emodes.T @ emodes if mass is None else emodes.T @ sparse.csc_matrix(mass) @ emodes

    return np.allclose(prod, np.eye(prod.shape[0]), atol=atol)

def standardize_modes(
    emodes: ArrayLike
) -> NDArray:
    """
    Perform standardisation by flipping the modes such that the first element of each eigenmode is 
    positive. This is helpful when visualising eigenmodes.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape (n_verts, n_modes), where n_modes is the number of eigenmodes.

    Returns
    -------
    numpy.ndarray
        The standardized eigenmodes array of shape (n_verts, n_modes), with the first element of
        each mode set to be positive.
    """
    emodes = np.asarray(emodes)

    # Find the sign of the first non-zero element in each column
    signs = np.sign(emodes[np.argmax(emodes != 0, axis=0), np.arange(emodes.shape[1])])
    
    # Apply the sign to the modes
    standardized_modes = emodes * signs
    
    return standardized_modes

def calc_norm_power(
    beta: ArrayLike
) -> NDArray:
    """
    Transform beta coefficients from a decomposition into normalised power.

    Parameters
    ----------
    beta : array-like
        The beta coefficients array of shape (n_vects, n_maps), where n_vects is the number of 
        orthogonal vectors and n_maps is the number of brain maps.

    Returns
    -------
    numpy.ndarray
        The normalized power array of shape (n_vects, n_maps), where each element represents the 
        proportion of power contributed by the corresponding orthogonal vector to each brain map.
    """
    beta_sq = np.asarray(beta)**2
    total_power = np.sum(beta_sq, axis=0)

    return beta_sq / total_power

def decompose(
    data: ArrayLike,
    emodes: ArrayLike,
    method: str = 'project',
    mass: ArrayLike = None
) -> NDArray:
    """
    Calculate the decomposition of the given data onto a basis set.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices
        and n_maps is the number of brain maps.
    emodes : array-like
        The vectors array of shape (n_verts, n_vecs), where n_vecs is the number of basis vectors.
    method : str, optional
        The method used for the decomposition, either 'project' to project data into a 
        mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
        from 'regress' tend towards those from 'project' when more basis vectors are provided. For a 
        non-orthonormal basis set, 'regress' must be used. Default is 'project'.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method
        is 'project'. If using EigenSolver, provide its self.mass. Default is None.

    Returns
    -------
    numpy.ndarray
        The beta coefficients array of shape (n_vecs, n_maps), obtained from the decomposition.
    
    Raises
    ------
    ValueError
        If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs,
        or if an invalid method is specified.
    """
    data = np.asarray(data)
    emodes = np.asarray(emodes)
    if mass is not None:
        mass = sparse.csc_matrix(mass)
    
    n_verts = emodes.shape[0]

    if data.shape[0] != n_verts:
        raise ValueError(f"The number of elements in `data` ({data.shape[0]}) must match "
                            f"the number of vertices in `emodes` ({n_verts}).")
    if np.isnan(emodes).any() or np.isinf(emodes).any():
        raise ValueError("`emodes` contains NaNs or Infs.")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)

    if method == 'project':
        if is_mass_orthonormal_modes(emodes):
            warn("Provided `emodes` are orthonormal in Euclidean space; ignoring mass matrix.")
            beta = emodes.T @ data
        else:
            if mass is None or mass.get_shape() != (n_verts, n_verts):
                raise ValueError(f"Mass matrix of shape ({n_verts}, {n_verts}) must be provided "
                                 "when method is 'project' and `emodes` is not an orthonormal basis"
                                 " set in Euclidean space.")
            beta = emodes.T @ mass @ data
    elif method == 'regress':
        beta = np.linalg.lstsq(emodes, data)[0]
    else:
        raise ValueError(f"Invalid decomposition method '{method}'; must be 'project' "
                            "or 'regress'.")

    return beta

def reconstruct(
    data: ArrayLike,
    emodes: ArrayLike,
    method: str = 'project',
    mass: Optional[ArrayLike] = None,
    mode_seq: Optional[ArrayLike] = None,
    timeseries: bool = False,
    metric: str = "pearsonr",
    return_all: bool = False
) -> Any:
    """
    Calculate and score the reconstruction of the given data using the provided orthogonal vectors.

    Parameters
    ----------
    data : array-like
        The input data array of shape (n_verts, n_maps), where n_verts is the number of vertices 
        and n_maps is the number of brain maps.
    emodes : array-like
        The vectors array of shape (n_verts, n_vects), where n_vects is the number of orthogonal 
        vectors.
    method : str, optional
        The method used for the decomposition, either 'project' to project data into a 
        mass-orthonormal space or 'regress' for least-squares fitting. Note that the beta values
        from 'regress' tend towards those from 'project' when more basis vectors are provided. For a 
        non-orthonormal basis set, 'regress' must be used. Default is 'project'.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is 
        'project'. If using EigenSolver, provide its self.mass. Default is None.
    mode_seq : array-like, optional
        The sequence of vectors to be used for reconstruction. Default is None, which uses all 
        vectors provided.
    timeseries : bool, optional
        Whether to treat brain maps as a time series of activity and reconstruct the functional
        coupling matrix. Default is False.
    metric : str, optional
        The metric used for calculating reconstruction accuracy, either "pearsonr" or "mse".
        Default is "pearsonr".
    return_all : bool, optional
        Whether to return the reconstructed timepoints when timeseries is True. Default is
        False.

    Returns
    -------
    list of numpy.ndarray
        A list of beta coefficients calculated for each mode.
    numpy.ndarray
        The reconstructed data array of shape (n_verts, nq, n_maps).
    numpy.ndarray
        The correlation coefficients array of shape (nq, n_maps).
    numpy.ndarray, optional
        The functional connectivity reconstructed data array of shape (n_verts, n_verts, nq). 
        Returned only if data_type is "timeseries".
    numpy.ndarray, optional
        The functional connectivity correlation coefficients array of shape (nq,). Returned only
        if data_type is "timeseries".
    
    Raises
    ------
    ValueError
        If the number of vertices in `data` and `emodes` do not match, if `emodes` contain NaNs,
        or if an invalid method is specified.
    """
    data = np.asarray(data)
    emodes = np.asarray(emodes)
    if mass is not None:
        mass = sparse.csc_matrix(mass)

    if metric not in ["pearsonr", "mse"]:
        raise ValueError(f"Invalid metric '{metric}'; must be 'pearsonr' or 'mse'.")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)

    # Use all modes if not specified (except the first constant mode)
    mode_seq = np.arange(1, np.shape(emodes)[1] + 1) if mode_seq is None else np.asarray(mode_seq)
    nq = len(mode_seq)

    n_verts, n_maps = data.shape

    # If data is timeseries, calculate the FC of the original data and initialize output arrays
    if timeseries:
        triu_inds = np.triu_indices(n_verts, k=1)
        fc_recon = np.empty((n_verts, n_verts, nq), dtype=data.dtype)
        fc_recon_score = np.empty((nq,), dtype=data.dtype)
        fc_orig = np.corrcoef(data)[triu_inds]
        if metric == "pearsonr":
            # Clip FC to exclude 1s and -1s, then Fisher r-to-z transform
            eps = np.finfo(fc_orig.dtype).eps
            fc_orig_z = np.arctanh(np.clip(
                fc_orig, -1+eps, 1-eps, dtype=fc_orig.dtype
                ), dtype=fc_orig.dtype)

    # Decompose the data to get beta coefficients
    if method == 'project':
        tmp = decompose(data, emodes[:, :np.max(mode_seq)], mass=mass)
        beta = [tmp[:mq] for mq in mode_seq]
    else:
        beta = [
            decompose(data, emodes[:, :mode_seq[i]], method=method)
            for i in range(nq)
        ]

    # Initialize the output arrays
    recon = np.empty((n_verts, nq, n_maps), dtype=data.dtype)
    recon_score = np.empty((nq, n_maps), dtype=data.dtype)
    for i in range(nq):
        # Reconstruct the data using the beta coefficients
        recon[:, i, :] = emodes[:, :mode_seq[i]] @ beta[i]

        # Score reconstruction
        if return_all or timeseries is False:
            if metric == "pearsonr":
                recon_score[i, :] = [
                    0 if mode_seq[i] == 1 else np.corrcoef(data[:, j],
                                                            np.squeeze(recon[:, i, j]))[0, 1]
                    for j in range(n_maps)
                ]
            elif metric == "mse":
                recon_score[i, :] = [
                    np.mean((data[:, j] - np.squeeze(recon[:, i, j]))**2)
                    for j in range(n_maps)
                ]

        if timeseries:
            # Calculate FC from the reconstruction
            fc_recon[:, :, i] = 0 if mode_seq[i] == 1 else np.corrcoef(recon[:, i, :])

            # Clip FC at (-1, 1), then Fisher r-to-z transform
            eps = np.finfo(fc_recon.dtype).eps
            fc_recon_z = np.arctanh(np.clip(
                fc_recon[:, :, i][triu_inds], -1+eps, 1-eps, dtype=fc_recon.dtype
                ), dtype=fc_recon.dtype)

            # Score reconstruction of FC
            if metric == "pearsonr":
                fc_recon_score[i] = 0 if mode_seq[i] == 1 else np.corrcoef(
                    fc_orig_z, fc_recon_z  
                )[0, 1]
            elif metric == "mse":
                fc_recon_score[i] = np.mean(
                    (fc_orig - np.squeeze(fc_recon[:, :, i][triu_inds]))**2
                    )
                
    beta = [beta[i].squeeze() for i in range(nq)]
    recon = recon.squeeze()
    recon_score = recon_score.squeeze()

    if timeseries:
        if return_all:
            return beta, recon, recon_score, fc_recon, fc_recon_score
        else:
            return beta, recon, fc_recon, fc_recon_score
    else:
        return beta, recon, recon_score