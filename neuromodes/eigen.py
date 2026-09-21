"""
Module for computing geometric eigenmodes of brain structures from surface and volume meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload
from warnings import warn

import numpy as np
from lapy import Solver
from scipy.sparse import csc_matrix

from neuromodes.io import read_surf, read_vol
from neuromodes.mesh import check_surf, check_vol, is_vol, mask_mesh

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Literal, TypeAlias

    from lapy import TetMesh, TriaMesh
    from nibabel.gifti.gifti import GiftiImage
    from numpy.random import Generator
    from numpy.typing import NDArray

    _CheckKind: TypeAlias = bool | Literal['maps', 'ortho', 'shape', 'evals']
    from neuromodes.basis import _IntSequenceKind, _SeqSequenceKind

class EigenSolver(Solver):
    """
    Class for computing and using eigenmodes and eigenvalues of a brain structure mesh [1]_ via the
    Finite Element Method, which discretizes the Laplace-Beltrami eigenvalue problem using mass and
    stiffness matrices [2]_ [3]_. Spatial heterogeneity can be optionally incorporated, modifying
    the stiffness matrix via an isotropic diffusion tensor [4]_. After calling :meth:`solve` to
    compute modes, a range of analysis methods can be called (:meth:`decompose`,
    :meth:`reconstruct`, :meth:`reconstruct_timeseries`, :meth:`sim_nft_waves`,
    :meth:`balloon_model`, :meth:`compute_gem`).

    Parameters
    ----------
    geometry : str, pathlib.Path, lapy.TriaMesh, lapy.TetMesh, or dict
        The mesh representation of a brain structure. Can be:
        - A path to one of the following file formats: ``.gii``, ``.vtk``, ``.tetra.vtk``,
        ``.white``, ``.pial``, ``.inflated``, ``.orig``, ``.sphere``, ``.smoothwm``, ``.qsphere``,
        ``.fsaverage``
        - An instance of ``GiftiImage``, ``lapy.TriaMesh``, or ``lapy.TetMesh``
        - A dictionary with two keys: ``'vertices'``, referencing a ``(n_verts, 3)``-shape array of
        vertex coordinates, and ``'cells'``, referencing a ``(n_cells, 3)``- or ``(n_cells,
        4)``-shape array of vertex indices in each triangle (surfaces) or tetrahedron (volumes),
        respectively
    mask : array-like, optional
        A boolean mask to exclude certain vertices (e.g., medial wall) from the mesh. Note that some
        masks may be invalid, if they lead to multiple connected components or faceless vertices.
        Default is ``None``.

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
        geometry: str | Path | GiftiImage | TriaMesh | TetMesh | dict,
        mask: NDArray[np.bool_] | None = None
    ):
        # Read in surface or volume mesh
        geometry = read_vol(geometry) if is_vol(geometry) else read_surf(geometry)

        # Optionally mask
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)  # chkfinite in mask_mesh
            geometry = mask_mesh(geometry, mask)
        
        # Validate mesh
        if is_vol(geometry):
            check_vol(geometry)
        else:
            check_surf(geometry)

        # Assign attributes
        self.geometry = geometry
        self.n_verts = geometry.v.shape[0]  # Nicety
        self.mask = mask
        self.use_cholmod = False  # Permit lapy.eigs()

        # Configuration states
        self._lump = False
        self._hetero = None

        # Internal caches
        self._mass = None
        self._stiffness = None
        self._emodes = None
        self._evals = None
        self._n_modes = None

    @property
    def lump(self) -> bool:
        """
        Whether the mass matrix is lumped (i.e., diagonal) or consistent.
        """
        return self._lump
    
    @property
    def hetero(self) -> NDArray[np.floating] | None:
        """
        The spatial heterogeneity map used to modify the stiffness matrix, or ``None`` if not used.
        """
        return self._hetero
    
    @property
    def mass(self) -> csc_matrix:
        """
        The sparse mass matrix of the mesh.
        """
        if self._mass is None:
            # Compute mass (consistent or lumped) for surface or volume mesh
            self._mass = (
                self._fem_tetra(self.geometry, self._lump)[1] if is_vol(self.geometry)
                else self.fem_tria_mass(self.geometry, self._lump)
                )
        return self._mass

    @property
    def stiffness(self) -> csc_matrix:
        """
        The sparse stiffness matrix of the mesh.
        """
        if self._stiffness is None:
            self.compute_lbo(lump=self._lump, hetero=self._hetero)
        return self._stiffness

    @property
    def emodes(self) -> NDArray[np.floating]:
        """
        The geometric (i.e., Laplace-Beltrami) eigenmodes of the mesh.
        """
        if self._emodes is None:
            raise ValueError("Eigenmodes not found. Please run the solve() method first.")
        return self._emodes
    
    @property
    def evals(self) -> NDArray[np.floating]:
        """
        The eigenvalues of the mesh.
        """
        if self._evals is None:
            raise ValueError("Eigenvalues not found. Please run the solve() method first.")
        return self._evals
    
    @property
    def n_modes(self) -> int:
        """
        The number of computed eigenmodes and eigenvalues.
        """
        if self._n_modes is None:
            raise ValueError("Eigenmodes not found. Please run the solve() method first.")
        return self._n_modes

    def __str__(self) -> str:
        """String representation of the ``EigenSolver`` object."""
        # Prepare mesh info
        if is_vol(self.geometry):
            geom_type = "Volume"
            elem_type = "tetrahedra"
        else:
            geom_type = "Surface"
            elem_type = "triangles"

        # Construct base output string
        str_out = (
            'EigenSolver\n'
            '-----------\n'
            f'{geom_type} mesh: {self.n_verts} vertices'
            )
        if self.mask is not None:
            str_out += f' ({np.sum(self.mask == 0)} others masked out)'
        str_out += f', {self.geometry.t.shape[0]} {elem_type}'

        # FEM matrices and modes
        if self._mass is not None:
            str_out += f'\nMass matrix: {"lumped" if self._lump else "consistent"}'
        if self._stiffness is not None:
            str_out += ('\nStiffness matrix: '
                        f'{("heterogeneous" if self._hetero is not None else "homogeneous")}')
        if self._emodes is not None:
            str_out += f'\nEigenmodes and eigenvalues: {self.n_modes} computed'

        return str_out

    def compute_lbo(
        self, 
        lump: bool = False,
        hetero: NDArray[np.floating] | None = None
    ) -> EigenSolver:
        """
        This method computes the Laplace-Beltrami operator via the LaPy package [1]_ [2]_,
        optionally incorporating spatial heterogeneity [3]_ [4]_. The resulting ``mass`` and
        ``stiffness`` matrices are cached as attributes. Note that changing ``lump`` or ``hetero``
        will invalidate any existing ``emodes`` and ``evals`` attributes, and these will be reset to
        ``None``.

        Parameters
        ----------
        lump : bool, optional
            Whether to compute the lumped (i.e., diagonal) mass matrix. Note that the lumped mass
            can be obtained from the consistent mass by simply summing either rows or columns onto
            the diagonal. Default is ``False``.
        hetero : array-like, optional
            A spatial heterogeneity map of shape ``(n_verts,)`` to modify the stiffness matrix via
            an isotropic diffusion tensor. If ``None`` or all ones, the standard homogeneous LBO is
            computed. In line with prior work [3]_, it is recommended that ``hetero`` is in the
            range [0, 2] (see rescaling functions :func:`~neuromodes.stats.zscorew` and
            :func:`~neuromodes.stats.sigmoid_rescale`). Default is ``None``. 

        Returns
        -------
        EigenSolver
            The ``EigenSolver`` instance.

        References
        ----------
        ..  [1] Reuter, M., et al. (2006). Laplace-Beltrami spectra as 'Shape-DNA' of surfaces and
            solids, Computer-Aided Design. https://doi.org/10.1016/j.cad.2005.10.011
        ..  [2] Wachinger, C., et al. (2015). BrainPrint: a discriminative characterization of brain
            morphology, Neuroimage. https://doi.org/10.1016/j.neuroimage.2015.01.032
        ..  [3] Barnes, V., et al. (2026). Regional heterogeneity shapes macroscopic wave dynamics
            of the human and non-human primate cortex, bioRxiv.
            https://doi.org/10.64898/2026.01.22.701178
        ..  [4] Andreux, M., et al. (2015). Anisotropic Laplace-Beltrami Operators for Shape
            Analysis, Computer Vision. https://doi.org/10.1007/978-3-319-16220-1_21
        """
        # For the LBO to be SPSD, hetero must be non-negative
        if hetero is not None and np.any(hetero < 0):
            warn("hetero contains negative values, which may result in negative Laplace-Beltrami "
                 "eigenvalues. It is recommended that heterogeneity maps are first rescaled (e.g., via " \
                 "neuromodes.stats.sigmoid_rescale) to be non-negative.")

        # Cache validation
        if not np.array_equal(hetero, self._hetero):
            self._stiffness = None  # hetero affects stiffness only
            self._emodes = None
            self._evals = None
            self._n_modes = None
            if hetero is not None:
                hetero = np.asarray_chkfinite(hetero)
                if hetero.shape != (self.n_verts,):
                    raise ValueError(f"hetero must have shape (n_verts,) = ({self.n_verts},).")
            self._hetero = hetero
        
        if lump != self._lump:
            self._mass = None  # lump affects mass only
            self._emodes = None
            self._evals = None
            self._n_modes = None
            self._lump = lump

        # Compute LBO as mass and stiffness matrices
        # LaPy has no method to compute only stiffness, so this recomputes mass as well
        if self._stiffness is None:
            if is_vol(self.geometry):
                if self.hetero is None:
                    # Compute FEM matrices under homogeneous LBO
                    self._stiffness, self._mass = self._fem_tetra(self.geometry, lump)
                else:
                    # Isotropic volumetric FEM (LaPy has no Solver._fem_tetra_aniso, so use our own)
                    self._stiffness, self._mass = _fem_tetra_hetero(
                        self.geometry, self.hetero, lump
                    )
            else:  # surface
                if self.hetero is None:
                    self._stiffness, self._mass = self._fem_tria(self.geometry, lump)
                else:
                    # Get principal curvatures to define direction of anisotropy
                    # Note: change of basis into (u1, u2) is not strictly needed for our isotropic
                    # diffusion tensor, but _fem_tria_aniso performs it
                    u1, u2, _, _ = self.geometry.curvature_tria()

                    # Map hetero from vertices to triangles by averaging
                    hetero_tria = self.geometry.map_vfunc_to_tfunc(self.hetero)

                    # Construct isotropic diffusion tensor by using hetero for both u1 and u2
                    hetero_mat = np.stack((hetero_tria, hetero_tria), axis=1)

                    # Compute FEM matrices under heterogeneous LBO
                    self._stiffness, self._mass = self._fem_tria_aniso(
                        self.geometry, u1, u2, hetero_mat, lump
                    )
        return self

    def solve(
        self,
        n_modes: int,
        hetero: NDArray[np.floating] | None = None,
        lump: bool = False,
        set_emode1: bool = True,
        align_emodes: bool = True,
        sigma: float = -0.01, # EASIEST way is to hard-code this to LaPy default (2026/03)
        seed: int | Generator | None = 0, 
        v0: NDArray[np.floating] | None = None
    ) -> EigenSolver:
        """
        Solves the generalized eigenvalue problem for the Laplace-Beltrami operator, optionally
        incorporating spatial heterogeneity. Resulting ``emodes`` and ``evals`` are stored as
        attributes.

        Parameters
        ----------
        n_modes : int
            Number of eigenmodes and eigenvalues to compute. Must be a positive integer less than
            the number of vertices.
        hetero : array-like, optional
            A spatial heterogeneity map of shape ``(n_verts,)`` to modify the stiffness matrix via
            an isotropic diffusion tensor. If ``None`` or all ones, the standard homogeneous LBO is
            computed. In line with prior work [1]_, it is recommended that ``hetero`` is in the
            range [0, 2] (see rescaling functions :func:`~neuromodes.stats.zscorew` and
            :func:`~neuromodes.stats.sigmoid_rescale`). Default is ``None``.
        lump: bool, optional
            Whether to use the lumped (i.e., diagonal) mass matrix. Note that the lumped mass can be
            obtained from the consistent mass by simply summing either rows or columns onto the
            diagonal. Default is ``False``.
        set_emode1 : bool, optional
            Whether to set the first eigenmode to a constant ``1/sqrt(sum(mass))`` and the first
            eigenvalue to ``0``, as is expected analytically for a single connected component.
            Default is ``True``.
        align_emodes : bool, optional
            Whether to ensure that each eigenmode has a positive first element by flipping its sign
            if necessary. Note that since these signs are arbitrary but this can be useful for
            visualization. Default is ``True``.
        sigma : float, optional
            Shift-invert parameter to speed up the computation of eigenvalues close to this value.
            Note that this changes the identity and ordering of returned eigenmodes and eigenvalues.
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

        References
        ----------
        ..  [1] Barnes, V., et al. (2026). Regional heterogeneity shapes macroscopic wave dynamics
            of the human and non-human primate cortex, bioRxiv.
            https://doi.org/10.64898/2026.01.22.701178
        """
        # Validate arguments
        if n_modes != int(n_modes) or n_modes <= 0 or n_modes >= self.n_verts:
            raise ValueError("n_modes must be a positive integer less than the number of vertices"
                             f" ({self.n_verts}).")
        if v0 is not None:
            v0 = np.asarray_chkfinite(v0)
            if v0.shape != (self.n_verts,):
                raise ValueError(f"v0 must have shape (n_verts,) = {(self.n_verts,)}.")
            
        # Ensure LBO is consistent with lump/hetero config
        self.compute_lbo(lump, hetero)

        # Solve the eigenvalue problem
        evals, emodes = self.eigs(k=n_modes, sigma=sigma, v0=v0, rng=seed)

        ## Post-process
        if set_emode1:
            if sigma >= 0:
                warn("emodes[:, 0] will not be set to its analytical constant value when sigma >= "
                     "0, as it may not correspond to the constant eigenmode.")
            else:
                # Value given by mass-orthonormality condition
                emodes[:, 0] = self.mass.sum()**(-0.5)
                evals[0] = 0.0

        if align_emodes:
            emodes = align_basis(emodes, checks=False)

        # Store results
        self._n_modes = n_modes  # Nicety
        self._evals = evals
        self._emodes = emodes
        return self
        
    # 1. mode_counts is None or int -> Single Array 
    @overload
    def decompose(
        self,
        data: NDArray[np.floating],
        *,
        mode_counts: int | None = ...,
        mode_ids: None = ...
    ) -> NDArray[np.floating]: ...

    # 2. mode_counts is Sequence -> List of Arrays
    @overload
    def decompose(
        self,
        data: NDArray[np.floating],
        *,
        mode_counts: _IntSequenceKind,
        mode_ids: None = ...
    ) -> list[NDArray[np.floating]]: ...

    # 3. mode_ids is Sequence -> List of Arrays
    @overload
    def decompose(
        self,
        data: NDArray[np.floating],
        *,
        mode_counts: None = ...,
        mode_ids: _SeqSequenceKind
    ) -> list[NDArray[np.floating]]: ...

    def decompose(
        self,
        data: NDArray[np.floating],
        **kwargs
    ) -> NDArray[np.floating] | list[NDArray[np.floating]]:
        """
        This is a wrapper for :func:`~neuromodes.basis.decompose`. Note that ``emodes``, ``mass``,
        and ``checks`` are passed automatically by the ``EigenSolver`` instance.
        """
        from neuromodes.basis import decompose as _decompose
    
        return _decompose(
            data=data,
            emodes=self.emodes,
            mass=self.mass,
            checks='maps',
            **kwargs
        )
    
    def reconstruct(
        self,
        data: NDArray[np.floating],
        **kwargs
    ) -> NDArray[np.floating]:
        """
        This is a wrapper for :func:`~neuromodes.basis.reconstruct`. Note that ``emodes``, ``mass``,
        and ``checks`` are passed automatically by the ``EigenSolver`` instance.
        """
        from neuromodes.basis import reconstruct as _reconstruct
            
        return _reconstruct(
            data=data,
            emodes=self.emodes,
            mass=self.mass,
            checks='maps',
            **kwargs
        )
    
    def calc_recon_error(
        self,
        data: NDArray[np.floating],
        recon: NDArray[np.floating],
        **kwargs
    ) -> NDArray[np.floating]:
        """
        This is a wrapper for :func:`~neuromodes.basis.calc_recon_error`. Note that ``mass`` and
        ``checks`` are passed automatically by the ``EigenSolver`` instance.
        """
        from neuromodes.basis import calc_recon_error as _calc_recon_error
            
        return _calc_recon_error(
            data=data,
            recon=recon,
            mass=self.mass,
            checks='maps',
            **kwargs
        )
    
    def compute_gem(
        self,
        **kwargs
    ) -> NDArray[np.floating]:
        """
        This is a wrapper for :func:`~neuromodes.network.compute_gem`. Note that ``emodes``,
        ``evals``, and ``checks`` are passed automatically by the ``EigenSolver`` instance.
        """
        from neuromodes.network import compute_gem as _compute_gem

        return _compute_gem(
            emodes=self.emodes,
            evals=self.evals,
            checks=False,
            **kwargs
        )
    
    def sim_nft_waves(
        self,
        **kwargs
    ) -> NDArray[np.floating]:
        """
        This is a wrapper for :func:`~neuromodes.waves.sim_nft_waves`. Note that ``emodes``,
        ``evals``, ``mass``, ``hetero``, and ``checks`` are passed automatically by the
        ``EigenSolver`` instance.
        """
        from neuromodes.waves import sim_nft_waves as _sim_nft_waves

        return _sim_nft_waves(
            emodes=self.emodes,
            evals=self.evals,
            mass=self.mass,
            hetero=self.hetero,
            stiffness=self.stiffness,
            checks='maps',
            **kwargs
        )
    
    def balloon_model(
        self,
        activity: NDArray[np.floating],
        dt: float,
        **kwargs
    ) -> NDArray[np.floating]:
        """
        This is a wrapper for :func:`~neuromodes.waves.balloon_model`. Note that ``emodes``,
        ``mass``, and ``checks`` are passed automatically by the ``EigenSolver`` instance.
        """
        from neuromodes.waves import balloon_model as _balloon_model

        return _balloon_model(
            activity=activity,
            dt=dt,
            emodes=self.emodes,
            mass=self.mass,
            checks='maps',
            **kwargs
        )
    
    def eigenstrap(
        self,
        data: NDArray[np.floating],
        **kwargs
    ) -> NDArray[np.floating]:
        """
        This is a wrapper for :func:`~neuromodes.nulls.eigenstrap`. Note that `emodes`, `evals`,
        `mass`, and `checks` are passed automatically by the `EigenSolver` instance.
        """
        from neuromodes.nulls import eigenstrap as _eigenstrap

        return _eigenstrap(
            data=data,
            emodes=self.emodes,
            evals=self.evals,
            mass=self.mass,
            checks='maps',
            **kwargs
        )

    def unmask_data(
        self,
        data: NDArray[np.floating],
        **kwargs
    ) -> NDArray[np.floating]:
        """
        This is a wrapper for :func:`~neuromodes.mesh.unmask_data`. Note that ``mask`` is passed
        automatically by the ``EigenSolver`` instance.
        """
        from neuromodes.mesh import unmask_data as _unmask_data

        if self.mask is None:
            raise ValueError("No mask found. This method is only applicable for masked meshes.")

        return _unmask_data(
            data=data,
            mask=self.mask,
            **kwargs
        )

def align_basis(
    emodes: NDArray[np.floating],
    checks: bool = True
) -> NDArray[np.floating]:
    """
    Ensures that the first element of each basis vector is positive by flipping its sign if
    necessary. For basis sets with arbitrary signs (e.g., geometric eigenmodes, principal
    components), this can be useful for visualization.

    Parameters
    ----------
    emodes : array-like
        The basis vectors array of shape ``(n_verts, n_modes)``, where n_modes is the number of
        vectors.
    checks : bool, optional
        Whether to validate the shape and type of ``emodes``. Default is ``True``.

    Returns
    -------
    numpy.ndarray
        The aligned basis vectors array of shape ``(n_verts, n_modes)``.
    """
    emodes = EigenData(emodes=emodes, checks=checks).emodes
   
    # Flip modes where the first element is negative
    return emodes * np.copysign(1, np.sign(np.asarray(emodes)[0, :]))

def is_orthonormal_basis(
    emodes: NDArray[np.floating],
    mass: csc_matrix | None = None,
    atol: float = 1e-03,
    rtol: float = 1e-05,
    checks: _CheckKind = 'shape'
) -> bool:
    """
    Check if a basis set is orthonormal with respect to a mass matrix (i.e., ``emodes.T @ mass
    @ emodes == I``, where ``I`` is an identity matrix). ``mass = I`` corresponds to Euclidean
    orthonormality, and an assumption that all vertices in a mesh have equal Voronoi areas/volumes.
    Mass-orthonormality is expected for the geometric eigenmodes (see notes).

    Parameters
    ----------
    emodes : array-like
        The vectors array of shape ``(n_verts, n_modes)``, where n_modes is the number of vectors.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)``. If ``None``, an identity matrix is used
        (Eucliean orthonormality). Default is ``None``.
    atol : float, optional
        Absolute tolerance for the orthonormality check. Default is ``1e-3``.
    rtol : float, optional
        Relative tolerance for the orthonormality check. Default is ``1e-5``.
    checks : bool | str, optional
        Whether to validate the shape and type of ``emodes`` and ``mass``. Default is ``True``.

    Returns
    -------
    bool
        ``True`` if basis vectors are orthonormal, ``False`` otherwise.

    Notes
    -----
    Under discretization, the set of eigenmodes ``emodes`` for any generalized eigenvalue problem ``A @
    emodes = - evals * mass @ emodes`` is expected to be mass-orthonormal, rather than Euclidean
    orthonormal. It follows that the first eigenmode is a constant ``1/sqrt(sum(mass))``, but
    precision error during computation can introduce spurious spatial deviations. Since
    mode-based analyses rely on mass-orthonormality, this function serves to ensure the validity of
    any calculated or provided eigenmodes.
    """
    # Format / validate arguments
    ved = EigenData(emodes=emodes, mass=mass, checks=checks)
    emodes, mass = ved.emodes, ved.mass

    # Check Euclidean or mass-orthonormality
    prod = emodes.T @ emodes if mass is None else emodes.T @ (mass @ emodes)
    identity = np.eye(emodes.shape[1])
    return np.allclose(prod, identity, rtol=rtol, atol=atol, equal_nan=False)

def get_eigengroup_inds(
    n_modes: int,
    ) -> list[NDArray]:
    """
    Identify eigengroups based on ordering of spherical harmonics. Each eigengroup 
    contains the next 2k+1 modes, where k is the eigengroup number (starting from 0). If
    ``n_modes`` does not include a complete eigengroup, the final group will contain the 
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

def _fem_tetra_hetero(
    geometry: TetMesh,
    hetero: NDArray[np.floating],
    lump: bool = False
) -> tuple[csc_matrix, csc_matrix]:
    """
    This function is a copy of ``lapy.solver.Solver._fem_tetra``, modified to incorporate
    heterogeneity. For a ``hetero`` of ones, output is identical to LaPy's ``_fem_tetra`` method.
    """        
    # Compute vertex coordinates and a difference vector for each triangle:
    t1 = geometry.t[:, 0]
    t2 = geometry.t[:, 1]
    t3 = geometry.t[:, 2]
    t4 = geometry.t[:, 3]
    v1 = geometry.v[t1, :]
    v2 = geometry.v[t2, :]
    v3 = geometry.v[t3, :]
    v4 = geometry.v[t4, :]
    e1 = v2 - v1
    e2 = v3 - v2
    e3 = v1 - v3
    e4 = v4 - v1
    e5 = v4 - v2
    e6 = v4 - v3
    # Compute cross product and 6 * vol for each triangle:
    cr = np.cross(e1, e3)
    vol = np.abs(np.sum(e4 * cr, axis=1))
    # zero vol will cause division by zero below, so set to small value:
    vol_mean = 0.0001 * np.mean(vol)
    vol[vol == 0] = vol_mean
    # compute dot products of edge vectors
    e11 = np.sum(e1 * e1, axis=1)
    e22 = np.sum(e2 * e2, axis=1)
    e33 = np.sum(e3 * e3, axis=1)
    e44 = np.sum(e4 * e4, axis=1)
    e55 = np.sum(e5 * e5, axis=1)
    e66 = np.sum(e6 * e6, axis=1)
    e12 = np.sum(e1 * e2, axis=1)
    e13 = np.sum(e1 * e3, axis=1)
    e14 = np.sum(e1 * e4, axis=1)
    e15 = np.sum(e1 * e5, axis=1)
    e23 = np.sum(e2 * e3, axis=1)
    e25 = np.sum(e2 * e5, axis=1)
    e26 = np.sum(e2 * e6, axis=1)
    e34 = np.sum(e3 * e4, axis=1)
    e36 = np.sum(e3 * e6, axis=1)
    # compute entries for A (negations occur when one edge direction is flipped)
    # these can be computed multiple ways
    # basically for ij, take opposing edge (call it Ek) and two edges from the
    # starting point of Ek to point i (=El) and to point j (=Em), then these are of
    # the scheme:   (El * Ek)  (Em * Ek) - (El * Em) (Ek * Ek)
    # where * is vector dot product
    a12 = (-e36 * e26 + e23 * e66) / vol
    a13 = (-e15 * e25 + e12 * e55) / vol
    a14 = (e23 * e26 - e36 * e22) / vol
    a23 = (-e14 * e34 + e13 * e44) / vol
    a24 = (e13 * e34 - e14 * e33) / vol
    a34 = (-e14 * e13 + e11 * e34) / vol
    # compute diagonals (from row sum = 0)
    a11 = -a12 - a13 - a14
    a22 = -a12 - a23 - a24
    a33 = -a13 - a23 - a34
    a44 = -a14 - a24 - a34

    # ----------------------------------- APPLY HETEROGENEITY ---------------------------------
    hetero_tetras = np.sum(hetero[geometry.t], axis=1) / geometry.t.shape[1]  # mean per tetra
    a12 *= hetero_tetras
    a13 *= hetero_tetras
    a14 *= hetero_tetras
    a23 *= hetero_tetras
    a24 *= hetero_tetras
    a34 *= hetero_tetras
    a11 *= hetero_tetras
    a22 *= hetero_tetras
    a33 *= hetero_tetras
    a44 *= hetero_tetras
    # -----------------------------------------------------------------------------------------

    # stack columns to assemble data
    local_a = np.column_stack(
        (
            a12,
            a12,
            a23,
            a23,
            a13,
            a13,
            a14,
            a14,
            a24,
            a24,
            a34,
            a34,
            a11,
            a22,
            a33,
            a44,
        )
    ).reshape(-1)
    i = np.column_stack(
        (t1, t2, t2, t3, t3, t1, t1, t4, t2, t4, t3, t4, t1, t2, t3, t4)
    ).reshape(-1)
    j = np.column_stack(
        (t2, t1, t3, t2, t1, t3, t4, t1, t4, t2, t4, t3, t1, t2, t3, t4)
    ).reshape(-1)
    local_a = local_a / 6.0
    a = csc_matrix((local_a, (i, j)))
    if not lump:
        # create b matrix data (account for that vol is 6 times tet volume)
        bii = vol / 60.0
        bij = vol / 120.0
        local_b = np.column_stack(
            (
                bij,
                bij,
                bij,
                bij,
                bij,
                bij,
                bij,
                bij,
                bij,
                bij,
                bij,
                bij,
                bii,
                bii,
                bii,
                bii,
            )
        ).reshape(-1)
        b = csc_matrix((local_b, (i, j)))
    else:
        # when lumping put all onto diagonal (volume/4 for each vertex)
        bii = vol / 24.0
        local_b = np.column_stack((bii, bii, bii, bii)).reshape(-1)
        i = np.column_stack((t1, t2, t3, t4)).reshape(-1)
        b = csc_matrix((local_b, (i, i)))
    return a, b

_MISSING = object()  
@dataclass(frozen=True, init=False)
class EigenData:
    emodes: NDArray[np.floating]
    evals: NDArray[np.floating] 
    mass: csc_matrix
    stiffness: csc_matrix
    hetero: NDArray[np.floating]
    data: NDArray[np.floating]

    def __init__(
        self,
        emodes: NDArray[np.floating] | None = _MISSING, # type: ignore[assignment]
        evals: NDArray[np.floating] | None = _MISSING, # type: ignore[assignment] 
        mass: csc_matrix | None = _MISSING, # type: ignore[assignment]
        stiffness: csc_matrix | None = _MISSING, # type: ignore[assignment]
        hetero: NDArray[np.floating] | None = _MISSING, # type: ignore[assignment]
        data: NDArray[np.floating] | tuple[NDArray[np.floating]] | list[NDArray[np.floating]] | None = _MISSING, # type: ignore[assignment]
        checks: _CheckKind = True
    ):  # TODO: add mask?
        # TODO: refactor to use helper functions

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
                if check_shape and emodes is not None and evals.shape != (emodes.shape[1],):
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
            if mass is not None and check_shape and (mass.ndim != 2 or mass.shape[0] != mass.shape[1]):
                raise ValueError("mass must be a square matrix.")
            _set('mass', mass)

        if stiffness is not _MISSING:
            all_inputs.append('stiffness')
            if stiffness is not None and check_shape and (stiffness.ndim != 2 or stiffness.shape[0] != stiffness.shape[1]):
                raise ValueError("stiffness must be a square matrix.")
            _set('stiffness', stiffness)

        # TODO: consider removing, data can just accept a list of arrays instead
        if hetero is not _MISSING:
            all_inputs.append('hetero')
            if hetero is not None:
                hetero = np.asarray_chkfinite(hetero)
                if check_shape and hetero.ndim != 1:
                    raise ValueError("hetero must have shape (n_verts,).")
            _set('hetero', hetero)

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
            if check_maps: # if check_maps is True, always check the shape
                # Convert single data array to iterable for consistent processing
                if not isinstance(data, (tuple, list)):
                    data = [data]
                
                # check shape and for NaN/Inf values in each data array
                data_proc = []
                for d in data:
                    if d is not None:
                        d = np.asarray(d)
                        if np.isnan(d).any(): 
                            warn("NaN values detected in data, which may cause issues with computations.")
                        if np.isinf(d).any():
                            warn("Inf values detected in data, which may cause issues with computations.")
                        if n_verts is None:
                            n_verts = d.shape[0]  # Establish the ground truth if not set
                        elif n_verts != d.shape[0]:
                            raise ValueError(f"data must have first dimension n_verts = {n_verts} to "
                                            "match the other arguments.")
                    data_proc.append(d)
                
                # Convert to tuple if needed
                data = tuple(data_proc) if len(data_proc) > 1 else data_proc[0]
                    
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
