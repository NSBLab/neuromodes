"""
Spatial null models via eigenmode rotation.

This module provides functions for generating null brain maps that preserve spatial
autocorrelation structure through random rotation of geometric eigenmodes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from warnings import warn

import numpy as np
from scipy.stats import special_ortho_group

from neuromodes.basis import decompose
from neuromodes.eigen import EigenData, get_eigengroup_inds
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.sparse import csc_matrix

    from neuromodes.basis import _DecompositionKind
    from neuromodes.eigen import _CheckKind

def eigenstrap(
    data: NDArray[np.floating],
    emodes: NDArray[np.floating],
    evals: NDArray[np.floating], 
    n_nulls: int = 1000,
    residual: Literal['add', 'permute'] | None = 'add',
    randomize: bool = False,
    n_groups: int | None = None,
    rotation_method: str = 'qr',
    decomp_method: _DecompositionKind = 'project',
    mass: csc_matrix | None = None,
    seed: int | NDArray | None = None,
    checks: _CheckKind = True,
) -> NDArray[np.floating]:
    """
    Generate spatial null maps via eigenstrapping [1]_.
    
    This function generates spatial null maps that preserve the spatial autocorrelation structure of
    brain maps through random rotation of geometric eigenmodes. The method works by rotating
    eigenmodes within eigengroups (defined by spherical harmonics), then reconstructing null maps
    using the original decomposition coefficients.
    
    Parameters
    ----------
    data : array-like
        Empirical brain map(s) of shape ``(n_verts,)`` or ``(n_verts, n_maps)`` to generate nulls
        from. If ``n_maps > 1``, the same set of randomized rotations is applied to all maps for
        each null (see Notes). 
    emodes : array-like
        The eigenmodes array of shape ``(n_verts, n_modes)``. This function rotates modes within
        eigengroups. By default (if ``n_groups`` is None), if the number of eigenmodes is not a
        perfect square (i.e., the number of modes does not allow for complete eigengroups), then the
        last incomplete eigengroup will be excluded.
    evals : array-like
        The eigenvalues array of shape ``(n_modes,)``.
    n_nulls : int, optional
        Number of null maps to generate per input map. Default is 1000.
    residual : str, optional
        How to handle reconstruction residuals after generating null maps. Either ``None`` to
        exclude residuals, ``'add'`` to add original residuals, or ``'permute'`` to add shuffled
        residuals. Default is ``'add'``. See Notes for details on which option to choose.
    randomize : bool, optional
        Whether to shuffle decomposition coefficients within eigengroups. This increases
        randomization but reduces spatial autocorrelation similarity to empirical data. Default is
        ``False``.
    n_groups : int or None, optional
        Number of eigengroups to use for generating nulls. If ``None``, uses all complete
        eigengroups contained in ``emodes`` (⌊√(n_modes)⌋). Default is ``None``.
    rotation_method : str, optional
        The method used to generate random rotations for each eigengroup. Either ``'qr'`` to
        generate random orthogonal matrices using QR decomposition of random normal matrices, or
        ``'scipy'`` to sample random orthogonal matrices from SO(N) using
        ``scipy.stats.special_ortho_group.rvs``. Default is ``'qr'``. See Notes for details on which
        option to choose.
    decomp_method : str, optional
        The method used for decomposition, either ``'project'`` or ``'regress'`` (see
        :func:`~neuromodes.basis.decompose`). Default is ``'project'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)``. Default is ``None``.
    seed : array-like or int, optional
        Random seed for reproducibility. If an array of shape ``(n_nulls,)`` is provided, it is used
        directly as the seed for each null. Otherwise, if a single integer is provided, it is used
        to generate a master seed that is then used to generate a different seed for each null. If
        ``None``, the global state is used. Default is ``None``.
    checks : bool, optional
        Whether to verify types, shapes, and orthonormality of ``emodes`` and ``mass`` before
        decomposition. Default is ``True``.
    
    Returns
    -------
    np.ndarray
        Generated null maps of shape ``(n_verts, n_nulls)`` if ``data`` has shape ``(n_verts,)``, or
        ``(n_verts, n_nulls, n_maps)`` if ``data`` has shape ``(n_verts, n_maps)``.

    Raises
    ------
    ValueError
        If ``residual`` is not one of ``None``, ``'add'``, or ``'permute'``.
    ValueError
        If ``rotation_method`` is not one of ``'qr'`` or ``'scipy'``.
    ValueError
        If ``n_groups`` is greater than the number of eigengroups than can be formed from the number
        of modes in ``emodes``.

    Notes
    -----
    1. Same transformations for each map. When ``data`` contains multiple maps (n_maps > 1), the
       same set of randomized rotations is applied to all maps for each null. This means that null i
       for map A and null i for map B use identical eigenmode rotations (± coefficient
       randomizations and residual permutations).

    2. ``n_groups`` and ``residual``. The choice of ``n_groups`` and ``residual`` will affect the
       spatial autocorrelation similarity between the nulls and empirical data. See ref [1]_ for a
       heuristic for choosing ``n_groups`` and to see how the choice of ``residual`` affects the
       spatial autocorrelation of the nulls.

    3. ``rotation_method``. ``rotation_method="scipy"`` is largely a legacy option to match the
       original implementation of eigenstrapping, which uses this method. The scipy method is
       recommended only for users who want to exactly match the original implementation of
       eigenstrapping. ``rotation_method="qr"`` is generally faster, especially for larger numbers
       of modes and nulls, and is recommended for most users. See more similarities/differences in
       the notes below.

    4. Resampling nulls. While each null will have a mean matching the original data, the
       variance/range/distribution will not be preserved in general. Users can therefore consider
       adjusting the distribution of the nulls via :func:`~neuromodes.stats.resample`. This should
       be guided by the importance of matching the original distribution of values and ultimately by
       whichever option produces the lowest false discovery rate (FDR). See ref [1]_ for an example
       of how to compute the FDR.

    5. ``seed``. Seeding is handled in one or two stages. First, if ``seed`` is None or a scalar
       integer, it is used to generate an array of size ``(n_nulls,)``. (If ``seed`` is already an
       array of size ``(n_nulls,)``, it is used directly.) Then, each of the ``n_nulls`` seeds are
       used to generate 3 new seeds: for rotations, randomisations, and residual permutations (if
       randomisations/permutations are not requested, these will be generated but not used). This
       ensures that the same seed will produce the same nulls regardless of how many nulls are
       generated (e.g., if n_nulls is increased from 10 to 100, the first 10 nulls will be
       identical).

    6. Comparisons with original implementation. The following notes are in relation to the original
       implementation of eigenstrapping in ref [1]_, which is available `here
       <https://github.com/SNG-Newy/eigenstrapping/tree/c2d8e5a5e7af47649f6358334644a6b49d22cf8e>`__.
       In this function, we have made a few changes to the implementation. These changes simplify
       installation, increase speed, process multiple maps concurrently, adjust for mesh
       irregularity, and facilitate reproducibility. Nonetheless, under a specific configuration,
       the function can exactly match the default usage of the implementation (i.e. generating the
       same nulls when the corresponding seed is set). See `this notebook
       <https://neuromodes.readthedocs.io/en/latest/validation/nulls_eigenstrap_orig.html>`__
       for an example.

       a. Simplified installation. ``eigenstrapping`` is dependent on many packages, including exact
          versions of ``brainspace`` and ``lapy``. In contrast, ``neuromodes`` depends on much fewer
          dependencies. This makes installation easier and reduces the risk of conflicts with other
          packages.

       b. Speed comparison. When generating 1000 nulls for one map on a 4k surface using 100 modes,
          the methods take:

          i. ``SurfaceEigenstrapping(n=1000)``:         3.80 seconds
          ii. ``eigenstrap(rotation_method='scipy')``:  1.12 seconds (exactly the same output as i)
          iii. ``eigenstrap(rotation_method='qr')``:    0.10 seconds

          See `this notebook
          <https://neuromodes.readthedocs.io/en/latest/validation/nulls_eigenstrap_speed.html>`__
          for more comparisons between the ``'scipy'`` and ``'qr'`` methods.

       c. Lack of cholmod. The original implementation uses ``cholmod`` for fast sparse matrix
          operations when computing the original calculation of geometric eigenmodes. In contrast,
          in order to simplify installation, this implementation does not use ``cholmod``. This has
          some impacts on how quickly modes can be computed. However, as they can be computed once
          and then saved, we hope that this will not be a major issue for users (we welcome
          contributions to add support for this!).

       d. Lack of parallelization from ``joblib``. The original implementation uses ``joblib`` to
          parallelize the generation of nulls across multiple CPU cores. Given our generation of
          nulls is faster, we have not implemented that at this stage. We welcome contributions to
          add support for this, but note that it may be difficult to implement this in a way that is
          reproducible across different numbers of nulls/maps/cores/seeding strategies. We note that
          the original implementation is not reproducible when using ``joblib`` due to the order in
          which the seeds are generated/used.

       e. Only match default. We are able to exactly match the default functionality of the original
          implementation, but not all possible configurations. This is because of some changes we
          have made to increase speed, account for mesh irregularity, and facilitate
          reproducibility. In particular, when ``randomize=False`` and ``residual=None``, the output
          of ``eigenstrap(rotation_method='scipy')`` will match the original output (see below for
          more parameters which also need to be specified). However, if ``randomize=True`` or
          ``residual='permute'``, it is not possible to match results between the two
          implementations. This is because of changes we have made to increase speed and account for
          mesh irregularity.

       f. Changes to RNG. Here, we have changed to ``numpy``'s newer ``Generator`` for random number
          generation, which means that the global seed does not affect the output of the function
          when using ``rotation_method='qr'``. This is in contrast to the original implementation,
          which used the legacy ``RandomState`` approach. See the last paragraph in the `'Quick
          Start' section
          <https://numpy.org/doc/stable/reference/random/index.html#random-quick-start>`__ for more
          information about the change. In particular, the original implementation was affected by
          the global seed when using ``seed=None``; this is preserved in ``rotation_method='scipy'``
          for compatibility, but not in ``rotation_method='qr'``.

       g. Use of first mode: This function uses the constant mode (first column of ``emodes``) and
          its corresponding eigenvalue. In contrast, the original implementation excludes the
          constant mode and its eigenvalue. Whereas in the original  implementation users were
          expected to input ``emodes`` and ``evals`` with the constant mode/eigenvalue removed
          (something of the form ``emodes[:, 1:]`` and ``evals[1:]``), here users are expected to
          input ``emodes`` and ``evals`` with the constant mode/eigenvalue included. This has
          the advantage of preserving the original mean of the data without need for resampling.

       h. Concurrent processing of multiple maps. This function can process multiple maps at the
          same time. This was possible in the original implementation, but required users to save
          rotation matrices and reapply them to all maps.

       i. Adding residuals: The original implementation does not add residuals of reconstruction to
          the nulls by default. However, this modal truncation of nulls increases their spatial
          autocorrelation, meaning that statistical inference may become overly conservative. In
          contrast, our default is ``residual='add'``, which better preserves the spatial
          autocorrelation of the nulls. The original default is equivalent to ``residual=None``.

       j. Adding residuals AND resampling: The original implementation supports resampling before
          adding residuals. Here, we offer a separate function for resampling, meaning that the
          order of operations is swapped (i.e., add residuals and then resample). This ensures that
          the resampling correctly matches the final surrogates to the input map(s).

       k. Mass-weighted decomposition. To account for mesh irregularity, the mass matrix is used
          when decomposing input maps into modal coefficients. In contrast, the original
          implementation performs ordinary least-squares regression.

       l. Syntax for exact replication. To exactly match the default version of the original
          implementation of eigenstrapping in ref [1]_, users must:

          - Set ``mass`` to ``None``.
          - Ensure ``data`` has a mean of zero.
          - Set ``seed=seed``.
          - Set ``decomp_method="regress"``.
          - Set ``rotation_method="scipy"``.
          - Set ``residual=None``.
          - Set ``checks='maps'``.
          - Resample the nulls using ``stats.resample(nulls, data, 'range')``.

          All other parameters should be left as their defaults. Note that the original
          implementation (``eigenstrapping.SurfaceEigenstrapping``) must also be run with a
          particular configuration to ensure reproducibility/compatibility:

          - Set the global seed before running ``SurfaceEigenstrapping(...)`` (e.g.,
            ``np.random.seed(seed)``).
          - Additionally, pass this seed into the function call: ``SurfaceEigenstrapping(...,
            seed=seed)``
          - Remember to remove the first eigenmode/eigenvalue from the call to
            ``SurfaceEigenstrapping``

          For an example of how to do this, see `this notebook
          <https://neuromodes.readthedocs.io/en/latest/validation/nulls_eigenstrap_orig.html>`__.
    
    References
    ----------
    ..  [1] Koussis, N. C., et al. (2024). Generation of surrogate brain maps preserving 
        spatial autocorrelation through random rotation of geometric eigenmodes. 
        Imaging Neuroscience. https://doi.org/10.1162/IMAG.a.71
    """
    # Format / validate arguments
    if residual not in (None, 'add', 'permute'):
        raise ValueError(f"Invalid residual method '{residual}'; must be 'add', 'permute', or "
                         "None.")

    if checks is not False:
        ved = EigenData(emodes=emodes, evals=evals, mass=mass, checks=checks)
        emodes, evals, mass = ved.emodes, ved.evals, ved.mass

    data = np.asarray(data)
    if (is_vector_data := data.ndim == 1):
        data = data[:, np.newaxis]
    n_maps = data.shape[1]

    # n_groups : Determine number of eigengroups, and trim emodes and evals to match
    n_cols = emodes.shape[1]
    if n_groups is None:
        n_groups = int(np.sqrt(n_cols))  # floor of root
        if n_groups**2 != n_cols:
              warn("`emodes` contains an incomplete eigengroup (i.e., number of modes is not a "
                 f"perfect square). Last {n_cols - n_groups**2} modes will be excluded.")
    elif n_groups**2 > n_cols:
        raise ValueError(f"`n_groups`={n_groups} implies n_modes={n_groups**2}, which exceeds the "
                         f"number of columns in `emodes` ({n_cols}).")
    n_modes = n_groups**2
    groups = get_eigengroup_inds(n_modes)
    emodes = emodes[:, :n_modes].copy()
    evals = evals[:n_modes].copy()
    
    # seed : Need to ultimately generate n_nulls * 3 Generators to use for each step of the process
    # (randomize, residual, rotate). (i) If `seed` is an array of shape (n_nulls,), we use each seed
    # to generate 3 new seeds; (ii) if `seed` is an int or None, we first generate an array of shape
    # (n_nulls,), then do (i).
    seed_scalar_input = seed is None or isinstance(seed, (int, np.integer))
    if seed_scalar_input: # generate a new array of seeds, one for each null
        seeds_main = np.random.SeedSequence(seed).generate_state(n_nulls, dtype=np.uint64)
    else: # check that this is a valid array of seeds, one for each null
        seeds_main = np.asarray_chkfinite(seed)
        if seeds_main.shape != (n_nulls,):
            raise ValueError(f"If `seed` is an array, it must have shape (n_nulls,) = ({n_nulls},),"
                             f" got {seeds_main.shape}.")
        if not np.issubdtype(seeds_main.dtype, np.integer):
            raise ValueError(f"`seed` must be an integer or array of integers, got dtype "
                             f"{seeds_main.dtype}.")

    # Convert the n_nulls seeds into n_nulls*3 seeds for each step of the process in a reproducible way
    states = np.array([np.random.SeedSequence(s).generate_state(3, dtype=np.uint64) for s in seeds_main])
    seeds_rotate, seeds_randomize, seeds_residual = states.T # randomize and residual only used if needed

    # rotation_method : For legacy behaviour mainly (compatibility with original eigenstrapping).
    # 1. We preserve the legacy behaviour of the original eigenstrapping under default parameters
    # (i.e., randomize=False and residual=None) when using the 'scipy' method. We don't support the
    # other cases due to the various other changes we have made to increase speed and facilitate
    # reproducibility. Therefore, we don't change seeds_randomize or seeds_residual here.  
    # 2. The legacy implementation does not support array inputs, so we can just keep the above
    # behaviour in that case. Moreover, it would not be possible to make the two compatible (given
    # all the other differences between our implementations). 
    if rotation_method == 'qr':
        _rotate_coeffs = _rotate_coeffs_qr
    elif rotation_method == 'scipy':
        _rotate_coeffs = _rotate_coeffs_scipy
        if seed_scalar_input: # legacy mode needs to set seeds_rotate to all Nones (cf. _rotate_coeffs_scipy)
            seeds_rotate = np.full((n_nulls,), None)
            if seed is not None: # allow legacy seeding to set the seed inside or outside the function call
                np.random.seed(seed) 
    else: 
        raise ValueError(f"Invalid rotation method '{rotation_method}'; must be 'qr' or 'scipy'.")

    if evals[0] != 0:
        warn("First eigenvalue is not exactly zero, as is analytically expected for geometric "
             "eigenmodes. However, since the first mode is not rotated, this will not affect the "
             "nulls.")
    if emodes[:, 0].std() > 1e-12:
        warn("First eigenmode is not exactly constant across vertices, as is analytically expected "
             "for geometric eigenmodes. Computed nulls may be biased.")
    evals[0] = 1 # No transform for constant mode (avoids division by zero)

    # Main calculations
    # Precompute transformed modes (ellipsoid -> spheroid for each eigengroup)
    sqrt_evals = np.sqrt(evals)
    norm_emodes = emodes / sqrt_evals[np.newaxis, :] # sqrt_evals behaves like a row vector
    sqrt_evals = sqrt_evals[:, np.newaxis, np.newaxis] # turn it into 3D column vector for below broadcasting

    # Eigendecompose maps (coeffs is n_modes x n_maps)
    coeffs = decompose(data, emodes, method=decomp_method, mass=mass, checks=False)
    # Turn coeffs into inv_coeffs, a 3D array of shape (n_modes, n_nulls, n_maps)
    # This is the precomputed inverse-transformed coefficients (spheroid -> ellipsoid for each eigengroup)
    if randomize:
        inv_coeffs = np.empty((n_modes, n_nulls, n_maps))
        for i, s in enumerate(seeds_randomize):
            rng = np.random.default_rng(s)
            for group in groups:
                inv_coeffs[group, i, :] = rng.permutation(coeffs[group, :], axis=0) 
        inv_coeffs *= sqrt_evals
    else:
        inv_coeffs = np.broadcast_to((coeffs[:, np.newaxis, :] * sqrt_evals), (n_modes, n_nulls, n_maps))

    # Generate nulls using tforms of shape (n_modes, n_nulls, n_maps)
    tforms = _rotate_coeffs(
        inv_coeffs=inv_coeffs,
        groups=groups,
        seeds=seeds_rotate
    )
    # tensordot appears faster than einsum
    nulls = np.tensordot(norm_emodes, tforms, axes=(1, 0)) # (n_verts, n_nulls, n_maps)

    # Optional post-processing steps
    # Add residuals of reconstruction
    if residual is not None:
        residual_data = data - emodes @ coeffs # shape (n_verts, n_maps)
        if residual == 'add':
            nulls += residual_data[:, np.newaxis, :]
        elif residual == 'permute':
            for i, s in enumerate(seeds_residual):
                nulls[:, i, :] += np.random.default_rng(s).permutation(residual_data, axis=0)

    return nulls.squeeze(axis=2) if is_vector_data else nulls

def _rotate_coeffs_scipy(
    inv_coeffs: NDArray[np.floating],
    groups: list[NDArray[np.integer]],
    seeds: NDArray[np.integer]
) -> NDArray[np.floating]:
    """
    Rotate coefficients using ``scipy.stats.special_ortho_group.rvs`` to sample random orthogonal
    matrices from SO(N). This is largely a legacy option to match the original implementation of
    eigenstrapping, which uses this method. The QR method is generally faster, especially for larger
    numbers of modes and nulls, and is recommended for most users. The scipy method is recommended
    only for users who want to exactly match the original implementation of eigenstrapping.

    Parameters
    ----------
    inv_coeffs : array-like
        The inverse-transformed coefficients (spheroid -> ellipsoid) of shape (n_modes, n_nulls,
        n_maps) to rotate.
    groups : list of arrays
        A list of arrays, where each array contains the indices of modes belonging to the same
        eigengroup.
    seeds : array-like
        An array of integer seeds of shape (n_nulls,) to use for reproducibility of the random
        rotations for each null map.

    Returns
    -------
    np.ndarray of shape ``(n_modes, n_nulls, n_maps)``
        The rotated coefficients (spheroid -> ellipsoid) to use for generating null maps.

    Notes
    -----
    We cannot use ``special_ortho_group.rvs(..., size=len(seeds))`` here in order to match the
    original implementation. This is because the rotations have to be generated separately for each
    null. Due to how the random seed progresses when the rotations are generated, it is not possible
    to generate rotations for all nulls at once. For the same reason, the order of the following
    loops cannot be changed (must generate rotations for one null before moving on to the next
    null).
    """
    # Unlike qr method, have to pass None (to keep using global seed) to match original eigenstrapping
    rngs = [None if s is None else np.random.default_rng(s) for s in seeds]

    # Define helper to sample rotation matrices from SO(k)
    def _get_so(k: int, rng: np.random.Generator | None) -> NDArray[np.floating]:
        return special_ortho_group.rvs(dim=k, random_state=rng) if k != 1 else np.array([[1.0]])

    tforms = np.empty_like(inv_coeffs)
    for i, rng in enumerate(rngs):  # Has to be in this order to match original functionality
        for group in groups:        # ie can't switch order of loops (even though it would be better)
            Q = _get_so(len(group), rng)
            # NumPy slicing trickery as second dims of tforms and inv_coeffs are collapsed:
            tforms[group, i, :] = Q @ inv_coeffs[group, i, :] 

    return tforms

def _rotate_coeffs_qr(
    inv_coeffs: NDArray[np.floating],
    groups: list[NDArray[np.integer]],
    seeds: NDArray[np.integer]
) -> NDArray[np.floating]:
    """
    Rotate coefficients using QR decomposition of random Gaussian matrices to generate random
    orthogonal matrices.
    
    Parameters
    ----------
    inv_coeffs : np.ndarray
        The inverse-transformed coefficients (spheroid -> ellipsoid) of shape ``(n_modes, n_nulls,
        n_maps)`` to rotate.
    groups : list of np.ndarrays
        A list of arrays, where each array contains the indices of modes belonging to the same
        eigengroup.
    seeds : np.ndarray
        An array of integer seeds of shape ``(n_nulls,)`` to use for reproducibility of the random
        rotations for each null map.

    Returns
    -------
    np.ndarray of shape ``(n_modes, n_nulls, n_maps)``
        The rotated coefficients (spheroid -> ellipsoid) to use for generating null maps.

    Notes
    -----
    We choose not to use ``special_ortho_group.rvs(..., size=len(seeds))`` here in order to
    facilitate generating rotations with precise seeding. We _could_ use this approach instead of
    defining our own helper, but it would not allow specific seeds to be used for each null.
    Instead, the user would have to pass a single seed which would then generate the rotation
    matrices for all the nulls. This would make it difficult to exactly replicate specific nulls,
    thus impairing replicability. Instead, we unwrap the code in ``special_ortho_group.rvs``: we
    generate random Gaussian matrices ourselves (using specific seeds for each null), and then use
    the same procedure as in that function (also in a vectorised manner).
    """
    # Unlike scipy method, still have to return Generator even if seed is None (for generating random Gaussian matrices)
    rngs = [np.random.default_rng(s) for s in seeds]

    # Define helper to generate random orthogonal matrices via QR decomposition (using scipy/Mezzadri algorithm)
    def _generate_so(k: int, rngs: list[np.random.Generator]) -> NDArray[np.floating]:
        # Generate random gaussian matrices for all nulls
        X = np.stack([rng.standard_normal((k, k)) for rng in rngs], axis=0) # rng progresses over each group
        # Perform QR decomposition
        Q, R = np.linalg.qr(X) # Q has shape (n_nulls, k, k)
        # Make QR decomp unique and uniform by making R's diagonal positive
        r = np.copysign(1.0, np.diagonal(R, axis1=1, axis2=2)) # copysign avoids chance of 0
        Q *= r[:, np.newaxis, :]
        # Ensure det(Q) = 1
        dets = np.linalg.det(Q)
        Q[:, :, 0] *= dets[:, np.newaxis]
        return Q

    tforms = np.empty_like(inv_coeffs)
    for group in groups: # unlike scipy, can loop over groups only (don't also have to loop over nulls)
        Q = _generate_so(len(group), rngs) # as we can generate rotations for all nulls at once
        # Batched matmul (equiv of @) appears faster than einsum or tensordot:
        tforms[group, :, :] = np.matmul(Q, inv_coeffs[group, :, :], axes=[(1, 2), (0, 2), (0, 2)])

    return tforms
