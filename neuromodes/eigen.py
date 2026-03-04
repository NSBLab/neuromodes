"""
Module for computing geometric eigenmodes of brain structures from surface meshes.
"""

from __future__ import annotations
from typing import Union, Tuple, TYPE_CHECKING
from warnings import warn
from lapy import Solver
import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from neuromodes.io import read_surf
from neuromodes.mesh import mask_mesh, check_surf

if TYPE_CHECKING:
    from pathlib import Path
    from lapy import TriaMesh
    from nibabel.gifti.gifti import GiftiImage
    from numpy import floating
    from numpy.typing import NDArray, ArrayLike

class EigenSolver(Solver):
    """
    Class for computing eigenmodes and eigenvalues from a brain structure mesh [1] via the Finite
    Element Method, which discretizes the Laplace-Beltrami eigenvalue problem using mass and
    stiffness matrices [2,3]. Spatial heterogeneity can be optionally incorporated, modifying the
    Laplace-Beltrami operator via a symmetric diffusion tensor [4]. After calling the `solve`
    method, a range of mode-based methods can be called (`decompose`, `reconstruct`,
    `reconstruct_timeseries`, `simulate_waves`, and `model_connectome`).

    Parameters
    ----------
    geometry : str, pathlib.Path, lapy.TriaMesh, or dict
        The surface mesh of a brain structure. Can be:
        - A path to one of the following file formats: `.gii`, `.vtk`, `.white`, `.pial`,
        `.inflated`, `.orig`, `.sphere`, `.smoothwm`, `.qsphere`, `.fsaverage`
        - An instance of `GiftiImage`, `lapy.TriaMesh`
        - A dictionary with keys `'vertices'` and `'faces'`
    mask : array-like, optional
        A boolean mask to exclude certain vertices (e.g., medial wall) from the mesh. Default is
        `None`.
    hetero : array-like, optional
        A heterogeneity map to scale the Laplace-Beltrami operator. Default is `None`.
    alpha : float, optional
        Scaling parameter for the heterogeneity map. If a heterogenity map is specified, the
        default is `1.0`. Otherwise, this value is ignored (and is set to `None`).
    scaling : str, optional
        Scaling function to apply to the heterogeneity map. Must be `'sigmoid'` or
        `'exponential'`. If a heterogenity map is specified, the default is `'sigmoid'`.
        Otherwise, this value is ignored (and is set to `None`).

    Raises
    ------
    ValueError
        If `hetero` length does not match the number of vertices (masked or unmasked).

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
        geometry: Union[str, Path, GiftiImage, TriaMesh, dict],
        mask: Union[ArrayLike, None] = None,
        normalize: bool = False,
        hetero: Union[ArrayLike, None] = None,
        alpha: Union[float, None] = None, # default to 1.0 if hetero given (and remains None)
        scaling: Union[str, None] = None  # default to "sigmoid" if hetero given (and remains None)
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
                warn("`scaling` is ignored as `hetero` is None.")
            if alpha is not None:
                warn("`alpha` is ignored as `hetero` is None.")
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
                raise ValueError(f"`hetero` must be a 1D array with length matching {err_str}.")

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
        """String representation of the EigenSolver object."""
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
        The resulting `stiffness` and `mass` matrices are stored as attributes.

        Parameters
        ----------
        lump : bool, optional
            Whether to use lumped mass matrix for the Laplace-Beltrami operator. Default is `False`.

        Returns
        -------
        EigenSolver
            The EigenSolver instance.
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

            # Construct symmetric (isotropic) diffusion tensor
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
        atol: float = 1e-3,
        rtol: float = 1e-5,
        sigma: Union[float, None] = -0.01,
        seed: Union[int, ArrayLike, None] = None, 
        lump: bool = False,
    ) -> EigenSolver:
        """
        Solves the generalized eigenvalue problem for the Laplace-Beltrami operator and compute
        eigenvalues and eigenmodes, which are stored as attributes (`emodes` and `evals`).

        Parameters
        ----------
        n_modes : int
            Number of eigenmodes to compute. Must be a positive integer less than the number of
            vertices.
        standardize : bool, optional
            If `True`, standardizes the sign of the eigenmodes so the first element is positive.
            Default is `False`.
        fix_mode1 : bool, optional
            If `True`, sets the first eigenmode to a constant value and the first eigenvalue to
            zero, as is expected analytically. Default is `True`. See the `is_orthonormal_basis`
            function for details.
        atol : float, optional
            Absolute tolerance for mass-orthonormality validation. Default is `1e-3`.
        rtol : float, optional
            Relative tolerance for mass-orthonormality validation. Default is `1e-5`.
        sigma : float, optional
            Shift-invert parameter to speed up the computation of eigenvalues close to this value.
            Default is `-0.01`.
        seed : int or array-like, optional
            Random seed for reproducibile generation of eigenvectors (which otherwise use an
            iterative algorithm that starts with a random vector, meaning that repeated generation
            of eigenmodes from the same mesh can have different orientations). Specify as an `int`
            (to set the seed) or a vector with n_verts elements (to directly set the initialisation
            vector). Default is `None` (not reproducible).
        lump: bool, optional
            Whether to use lumped mass matrix for the Laplace-Beltrami operator. Default is `False`.

        Returns
        -------
        EigenSolver
            The EigenSolver instance.

        Raises
        ------
        ValueError
            If `n_modes` is not a positive integer less than the number of vertices.
        ValueError
            If `seed` is an array but does not have shape (n_verts,).
        AssertionError
            If computed eigenvalues or eigenmodes contain NaNs.
        """
        # Validate arguments
        if n_modes != int(n_modes) or n_modes <= 0 or n_modes >= self.n_verts:
            raise ValueError("`n_modes` must be a positive integer less than the number of vertices"
                             f" ({self.n_verts}).")

        # Compute the Laplace-Beltrami operator / set stiffness and mass matrices
        if not hasattr(self, 'stiffness'):
            self.compute_lbo(lump)
        
        # Set intitialization vector (if desired) for reproducibile eigenvectors 
        if seed is None or isinstance(seed, int):
            rng = np.random.default_rng(seed)
            v0 = rng.random(self.n_verts)
        else:
            v0 = np.asarray_chkfinite(seed)
            if v0.shape != (self.n_verts,):
                raise ValueError("`seed` must be either an integer or an array of shape (n_verts,) "
                                 f"= {(self.n_verts,)}.")

        # Solve the eigenvalue problem
        lu = splu(self.stiffness - sigma * self.mass)
        op_inv = LinearOperator( 
            matvec=lu.solve, # type: ignore
            shape=self.stiffness.shape,
            dtype=self.stiffness.dtype,
        )

        evals, emodes = eigsh(
            self.stiffness,
            k=n_modes,
            M=self.mass,
            sigma=sigma,
            OPinv=op_inv,
            v0=v0
        )

        # Validate results
        assert not ((np.isnan(evals).any() or np.isnan(emodes).any())), (
            "Computed eigenvalues or eigenmodes contain NaNs. This may indicate numerical "
            "instability; consider adjusting `sigma` or checking mesh quality.")

        if not is_orthonormal_basis(emodes, self.mass, atol=atol, rtol=rtol):
            warn(f"Computed eigenmodes are not mass-orthonormal (atol={atol}, rtol={rtol}).")

        ## Post-process

        # Sort modes by ascending eigenvalue (should already be sorted for sigma < 0)
        sort_idx = np.argsort(evals)
        evals = evals[sort_idx]
        emodes = emodes[:, sort_idx]

        if fix_mode1:
            if sigma >= 0:
                warn("Mode 1 will not be fixed to a constant when sigma >= 0, as the constant mode "
                     "may not be among the computed modes.")
            else:
                # Value given by mass-orthonormality condition
                emodes[:, 0] = self.mass.sum()**(-0.5)
                evals[0] = 0.0

        if standardize:
            emodes = standardize_modes(emodes)

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
        data: ArrayLike,
        **kwargs
    ) -> NDArray[floating]:
        """
        This is a wrapper for `neuromodes.basis.decompose`, see its documentation for details: 
        https://neuromodes.readthedocs.io/en/latest/generated/neuromodes.basis.decompose.html

        Note that `emodes`, `mass`, and `checks` are passed automatically by the `EigenSolver`
        instance.
        """
        from neuromodes.basis import decompose

        self._check_for_emodes()
    
        return decompose(
            data,
            self.emodes,
            mass=self.mass,
            checks=False,
            **kwargs
        )
    
    def reconstruct(
        self,
        data: ArrayLike,
        **kwargs
    ) -> Tuple[NDArray[floating], NDArray[floating], list[NDArray[floating]]]:
        """
        This is a wrapper for `neuromodes.basis.reconstruct`, see its documentation for details:
        https://neuromodes.readthedocs.io/en/latest/generated/neuromodes.basis.reconstruct.html

        Note that `emodes`, `mass`, and `checks` are passed automatically by the `EigenSolver`
        instance.
        """
        from neuromodes.basis import reconstruct
        
        self._check_for_emodes()
            
        return reconstruct(
            data,
            self.emodes,
            mass=self.mass,
            checks=False,
            **kwargs
        )
    
    def reconstruct_timeseries(
        self,
        data: ArrayLike,
        **kwargs
    ) -> Tuple[NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating],
               list[NDArray[floating]]]:
        """
        This is a wrapper for `neuromodes.basis.reconstruct_timeseries`, see its documentation for
        details:
        https://neuromodes.readthedocs.io/en/latest/generated/neuromodes.basis.reconstruct_timeseries.html

        Note that `emodes`, `mass`, and `checks` are passed automatically by the `EigenSolver`
        instance.
        """
        from neuromodes.basis import reconstruct_timeseries

        self._check_for_emodes()
            
        return reconstruct_timeseries(
            data,
            self.emodes,
            mass=self.mass,
            checks=False,
            **kwargs
        )
    
    def model_connectome(
        self,
        **kwargs
    ) -> NDArray[floating]:
        """
        This is a wrapper for `neuromodes.connectome.model_connectome`, see its documentation for
        details:
        https://neuromodes.readthedocs.io/en/latest/generated/neuromodes.connectome.model_connectome.html

        Note that `emodes`, `evals`, and `checks` are passed automatically by the `EigenSolver`
        instance.
        """
        from neuromodes.connectome import model_connectome

        self._check_for_emodes()

        return model_connectome(
            emodes=self.emodes,
            evals=self.evals,
            checks=False,
            **kwargs
        )
    
    def simulate_waves(
        self,
        **kwargs
    ) -> NDArray[floating]:
        """
        This is a wrapper for `neuromodes.waves.simulate_waves`, see its documentation for details:
        https://neuromodes.readthedocs.io/en/latest/generated/neuromodes.waves.simulate_waves.html

        Note that `emodes`, `evals`, `mass`, `scaled_hetero`, and `checks` are passed automatically
        by the `EigenSolver` instance.
        """
        from neuromodes.waves import simulate_waves

        self._check_for_emodes()

        return simulate_waves(
            emodes=self.emodes,
            evals=self.evals,
            mass=self.mass,
            scaled_hetero=self.hetero,
            checks=False,
            **kwargs
        )
    
    def bold_transform(
        self,
        activity: ArrayLike,
        dt: float,
        **kwargs
    ) -> NDArray[floating]:
        """
        This is a wrapper for `neuromodes.waves.bold_transform`, see its documentation for details:
        https://neuromodes.readthedocs.io/en/latest/generated/neuromodes.waves.bold_transform.html

        Note that `emodes`, `mass`, and `checks` are passed automatically by the `EigenSolver`
        instance.
        """
        from neuromodes.waves import bold_transform

        self._check_for_emodes()

        return bold_transform(
            activity,
            emodes=self.emodes,
            mass=self.mass,
            checks=False,
            **kwargs
        )

def scale_hetero(
    hetero: ArrayLike,
    alpha: float = 1.0,
    scaling: str = "sigmoid"
) -> NDArray[floating]:
    """
    Scales a heterogeneity map using specified normalization and scaling functions.
    
    Parameters
    ----------
    hetero : array-like
        The heterogeneity map to be scaled.
    alpha : float, optional
        Scaling parameter controlling the strength of the transformation. Default is `1.0`.
    scaling : str, optional
        The scaling function to apply to the heterogeneity map, either `'sigmoid'` or
        `'exponential'`. Default is `'sigmoid'`.
    
    Returns
    -------
    ndarray
        The scaled heterogeneity map.

    Raises
    ------
    ValueError
        If `hetero` is not a 1D array.
    ValueError
        If `scaling` is not 'exponential' or 'sigmoid'.
    ValueError
        If `hetero` is constant.
    """
    # Format / validate arguments
    hetero = np.asarray_chkfinite(hetero)
    alpha = float(alpha)
    if hetero.ndim != 1:
        raise ValueError("`hetero` must be a 1D array.")
    if scaling not in ["exponential", "sigmoid"]:
        raise ValueError(f"Invalid scaling '{scaling}'. Must be 'exponential' or 'sigmoid'.")
    if alpha == 0:
        warn("`alpha` is set to 0, meaning heterogeneity map will have no effect.")
    std = np.std(hetero)
    if std == 0:
        warn("Provided `hetero` is constant; scaling `hetero` to a vector of ones.")
        hetero_scaled = np.ones_like(hetero)
    else:
        # Scale the heterogeneity map
        hetero_z = (hetero - np.mean(hetero)) / std
        hetero_scaled = (2 / (1 + np.exp(-alpha * hetero_z))
                         if scaling == 'sigmoid' else np.exp(alpha * hetero_z))
    
    return hetero_scaled

def standardize_modes(
    emodes: ArrayLike
) -> NDArray[floating]:
    """
    Flips the modes' signs such that the first element of each eigenmode has positive amplitude. 
    Note that the sign of each mode is arbitrary--standardisation is only helpful to compare sets of
    eigenmodes.

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
    emodes = np.asarray_chkfinite(emodes)

    # Find the sign of each mode's amplitude at the first vertex
    signs = np.sign(emodes[0, :])
    signs[signs == 0] = 1  # Treat zero as positive (unlikely case)
    
    # Flip modes where the first element is negative
    return emodes * signs

def is_orthonormal_basis(
    emodes: ArrayLike,
    mass: Union[spmatrix, ArrayLike, None] = None,
    atol: float = 1e-03,
    rtol: float = 1e-05
) -> bool:
    """
    Check if a set of vectors is orthonormal in Euclidean space (i.e., `emodes.T @ emodes == I`,
    where `I` is the identity matrix) or with respect to a mass matrix (i.e., `emodes.T @ mass @
    emodes == I`). Mass-orthonormality is expected for the geometric eigenmodes (see notes).

    Parameters
    ----------
    emodes : array-like
        The vectors array of shape (n_verts, n_modes), where n_modes is the number of vectors.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts). If `None`, Euclidean orthonormality is checked.
        Default is `None`.
    atol : float, optional
        Absolute tolerance for the orthonormality check. Default is `1e-3`.
    rtol : float, optional
        Relative tolerance for the orthonormality check. Default is `1e-5`.

    Returns
    -------
    bool
        `True` if the set of vectors is orthonormal (Euclidean or mass-orthonormal), `False`
        otherwise.

    Raises
    ------
    ValueError
        If `emodes` does not have shape (n_verts, n_modes), where n_verts > n_modes.
    ValueError
        If `mass` is provided but does not have shape (n_verts, n_verts).

    Notes
    -----
    Under discretization, the set of solutions for any generalized eigenvalue problem `stiffness @
    emodes = - evals * mass @ emodes` is expected to be mass-orthonormal, rather than orthonormal
    with respect to the standard Euclidean inner (dot) product. It follows that the first mode is
    expected to be a specific constant, but precision error during computation can introduce
    spurious spatial heterogeneity. Since many eigenmode analyses rely on mass-orthonormality (e.g.,
    decomposition, wave simulation), this function serves to ensure the validity of any calculated
    or provided eigenmodes.
    """
    # Format / validate arguments
    emodes = np.asarray_chkfinite(emodes)
    if not isinstance(mass, (spmatrix, type(None))):
        mass = np.asarray_chkfinite(mass)

    if emodes.ndim != 2 or emodes.shape[0] <= emodes.shape[1]:
        raise ValueError("`emodes` must have shape (n_verts, n_modes), where n_verts > n_modes.")
    n_verts, n_modes = emodes.shape
    if mass is not None and (mass.shape != (n_verts, n_verts)):
        raise ValueError(f"`mass` must have shape (n_verts, n_verts) = {(n_verts, n_verts)}.")

    # Check Euclidean or mass-orthonormality
    prod = emodes.T @ emodes if mass is None else emodes.T @ mass @ emodes
    return np.allclose(prod, np.eye(n_modes), rtol=rtol, atol=atol, equal_nan=False)