from pathlib import Path
from typing import Any, overload, TypeAlias, Literal
import numpy as np
from lapy import Solver, TriaMesh
from nibabel.gifti.gifti import GiftiImage
from numpy.random import Generator
from numpy.typing import NDArray, ArrayLike
from scipy.sparse import csc_matrix

from neuromodes.basis import (_DecompositionKind, _IntSequenceKind, _SeqSequenceKind, _MetricCallbackKind, 
                              _ReconList, _ReconSingle, 
                              _ReconTSSingle, _ReconTSList)

_CheckKind: TypeAlias = bool | Literal['maps', 'ortho', 'shape', 'data'] | None

# ==========================================
# CLASSES
# ==========================================

class EigenSolver(Solver):
    geometry: str | Path | GiftiImage | TriaMesh | dict  # Can be TriaMesh or similar internally
    n_verts: int
    mask: NDArray[np.bool_] | None
    hetero: NDArray[np.floating] | None
    use_cholmod: bool
    stiffness: csc_matrix
    mass: csc_matrix
    n_modes: int
    evals: NDArray[np.floating]
    emodes: NDArray[np.floating]
    _scaling: str | None
    _alpha: float | None

    def __init__(
        self,
        geometry: str | Path | GiftiImage | TriaMesh | dict,
        mask: ArrayLike | None = ...,
        normalize: bool = ...,
        hetero: ArrayLike | None = ...,
        alpha: float | None = ...,
        scaling: str | None = ...
    ) -> None: ...

    def __str__(self) -> str: ...

    def compute_lbo(self, lump: bool = ...) -> EigenSolver: ...

    def solve(
        self,
        n_modes: int,
        standardize: bool = ...,
        fix_mode1: bool = ...,
        lump: bool = ...,
        atol: float = ...,
        rtol: float = ...,
        sigma: float = ...,
        seed: int | Generator | None = ...,
        v0: ArrayLike | None = ...
    ) -> EigenSolver: ...

    def _check_for_emodes(self) -> None: ...

    # %% DECOMPOSE
    # 1. mode_counts is None or int -> Single Array 
    @overload
    def decompose(
        self,
        data: NDArray,
        method: _DecompositionKind = ...,
        *,
        mode_counts: int | None = ...,
        mode_ids: None = ...,
        checks: _CheckKind = ...
    ) -> NDArray[np.floating]: ...

    # 2. mode_counts is Sequence -> List of Arrays
    @overload
    def decompose(
        self,
        data: NDArray,
        method: _DecompositionKind = ...,
        *,
        mode_counts: _IntSequenceKind,
        mode_ids: None = ...,
        checks: _CheckKind = ...
    ) -> list[NDArray[np.floating]]: ...

    # 3. mode_ids is Sequence -> List of Arrays
    @overload
    def decompose(
        self,
        data: NDArray,
        method: _DecompositionKind = ...,
        *,
        mode_counts: None = ...,
        mode_ids: _SeqSequenceKind,
        checks: _CheckKind = ...
    ) -> list[NDArray[np.floating]]: ...

    # %% RECONSTRUCT
    # 1. mode_counts is None or int -> Tuple with Single Array
    @overload
    def reconstruct(
        self,
        data: NDArray,
        method: _DecompositionKind = ...,
        *,
        mode_counts: int | None = ...,
        mode_ids: None = ...,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconSingle: ...

    # 2. mode_counts is Sequence -> Tuple with List of Arrays
    @overload
    def reconstruct(
        self,
        data: NDArray,
        method: _DecompositionKind = ...,
        *,
        mode_counts: _IntSequenceKind,
        mode_ids: None = ...,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconList: ...

    # 3. Mode IDs -> Tuple with List of Arrays
    @overload
    def reconstruct(
        self,
        data: NDArray,
        method: _DecompositionKind = ...,
        *,
        mode_counts: None = ...,
        mode_ids: _SeqSequenceKind,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconList: ...

    # %% RECONSTRUCT_TIMESERIES
    # 1. mode_counts is None or int -> Tuple with Single Array
    @overload
    def reconstruct_timeseries(
        self,
        timeseries: NDArray,
        method: _DecompositionKind = ...,
        *,
        mode_counts: int | None = ...,
        mode_ids: None = ...,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconTSSingle: ...

    # 2. mode_counts is Sequence -> Tuple with List of Arrays
    @overload
    def reconstruct_timeseries(
        self,
        timeseries: NDArray,
        method: _DecompositionKind = ...,
        *,
        mode_counts: _IntSequenceKind,
        mode_ids: None = ...,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconTSList: ...

    # 3. Mode IDs -> Tuple with List of Arrays
    @overload
    def reconstruct_timeseries(
        self,
        timeseries: NDArray,
        method: _DecompositionKind = ...,
        *,
        mode_counts: None = ...,
        mode_ids: _SeqSequenceKind,
        checks: _CheckKind = ...,
        metric: _MetricCallbackKind = ...,
        **cdist_kwargs
    ) -> _ReconTSList: ...

    # --- OTHER WRAPPERS ---
    def model_connectome(self, **kwargs: Any) -> NDArray[np.floating]: ...
    def simulate_waves(self, **kwargs: Any) -> NDArray[np.floating]: ...
    def bold_transform(self, activity: ArrayLike, dt: float, **kwargs: Any) -> NDArray[np.floating]: ...
    def eigenstrap(self, data: NDArray, **kwargs: Any) -> NDArray: ...


class EigenData:
    emodes: NDArray[np.floating]
    evals: NDArray[np.floating]
    mass: csc_matrix
    stiffness: csc_matrix
    scaled_hetero: NDArray[np.floating]
    data: NDArray[np.floating]

    def __init__(
        self,
        emodes: NDArray[np.floating] | None = ...,
        evals: NDArray[np.floating] | None = ...,
        mass: csc_matrix | None = ...,
        stiffness: csc_matrix | None = ...,
        scaled_hetero: NDArray[np.floating] | None = ...,
        data: NDArray[np.floating] | None = ...,
        checks: _CheckKind = ...
    ) -> None: ...

    def __getattribute__(self, name: str) -> Any: ...


# ==========================================
# FUNCTIONS
# ==========================================

def scale_hetero(
    hetero: ArrayLike,
    alpha: float = ...,
    scaling: Literal["exponential", "sigmoid"] = ...
) -> NDArray[np.floating]: ...

def standardize_emodes(
    emodes: NDArray,
    checks: bool = ...
) -> NDArray: ...

def is_orthonormal_basis(
    emodes: NDArray,
    mass: csc_matrix | None = ...,
    atol: float = ...,
    rtol: float = ...,
    checks: _CheckKind = ...
) -> bool: ...

def get_eigengroup_inds(
    n_modes: int
) -> list[NDArray]: ...