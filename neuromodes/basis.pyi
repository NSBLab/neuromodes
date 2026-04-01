import numpy as np
from typing import overload, TypeAlias, Literal, Any
from collections.abc import Sequence
from numpy.typing import NDArray
from scipy.sparse import csc_matrix
from scipy.spatial.distance import _MetricCallback, _MetricKind 

from neuromodes.eigen import _CheckKind

# %% TYPE ALIASES
# Types for reconstruct (Tuple of 3)
_ReconSingle: TypeAlias = tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
_ReconList: TypeAlias = tuple[NDArray[np.floating], NDArray[np.floating], list[NDArray[np.floating]]]

# Types for reconstruct_timeseries (Tuple of 5)
_ReconTSSingle: TypeAlias = tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
_ReconTSList: TypeAlias = tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], list[NDArray[np.floating]]]

# Generic Types for inputs
_MetricCallbackKind: TypeAlias = _MetricCallback | _MetricKind | None
_IntSequenceKind: TypeAlias = Sequence[int] | NDArray[np.integer]
_SeqSequenceKind: TypeAlias = Sequence[_IntSequenceKind] | NDArray[Any]
_DecompositionKind: TypeAlias = Literal['project', 'regress']

# %% DECOMPOSE
# 1. mode_counts is None or int -> Single Array 
@overload
def decompose(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: int | None = ...,
	mode_ids: None = ...,
	checks: _CheckKind = ...
) -> NDArray[np.floating]: ...

# 2. mode_counts is Sequence -> List of Arrays
@overload
def decompose(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: _IntSequenceKind,
	mode_ids: None = ...,
	checks: _CheckKind = ...
) -> list[NDArray[np.floating]]: ...

# 3. mode_ids is Sequence -> List of Arrays
@overload
def decompose(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: None = ...,
	mode_ids: _IntSequenceKind,
	checks: _CheckKind = ...
) -> list[NDArray[np.floating]]: ...

# %% RECONSTRUCT
# 1. mode_counts is None or int -> Tuple with Single Array
@overload
def reconstruct(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: int | None = ...,
	mode_ids: None = ...,
	checks: _CheckKind = ...,
	metric: _MetricCallbackKind = ...,
	**cdist_kwargs
) -> _ReconSingle: ...

# 2. mode_counts is Sequence -> Tuple with List of Arrays
@overload
def reconstruct(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: _IntSequenceKind,
	mode_ids: None = ...,
	checks: _CheckKind = ...,
	metric: _MetricCallbackKind = ...,
	**cdist_kwargs
) -> _ReconList: ...

# 3. Mode IDs -> Tuple with List of Arrays
@overload
def reconstruct(
    data: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: None = ...,
	mode_ids: _IntSequenceKind,
	checks: _CheckKind = ...,
	metric: _MetricCallbackKind = ...,
	**cdist_kwargs
) -> _ReconList: ...

# %% RECONSTRUCT_TIMESERIES
# 1. mode_counts is None or int -> Tuple with Single Array
@overload
def reconstruct_timeseries(
    timeseries: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: int | None = ...,
	mode_ids: None = ...,
	checks: _CheckKind = ...,
	metric: _MetricCallbackKind = ...,
	**cdist_kwargs
) -> _ReconTSSingle: ...

# 2. mode_counts is Sequence -> Tuple with List of Arrays
@overload
def reconstruct_timeseries(
    timeseries: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: _IntSequenceKind,
	mode_ids: None = ...,
	checks: _CheckKind = ...,
	metric: _MetricCallbackKind = ...,
	**cdist_kwargs
) -> _ReconTSList: ...

# 3. Mode IDs -> Tuple with List of Arrays
@overload
def reconstruct_timeseries(
    timeseries: NDArray[np.floating],
	emodes: NDArray[np.floating],
	method: _DecompositionKind = ...,
	*,
    mass: csc_matrix | None = ...,
	mode_counts: None = ...,
	mode_ids: _IntSequenceKind,
	checks: _CheckKind = ...,
	metric: _MetricCallbackKind = ...,
	**cdist_kwargs
) -> _ReconTSList: ...

# %% OTHERS
def calc_norm_power(beta: NDArray[np.floating]) -> NDArray[np.floating]: ...
def calc_vec_fc(timeseries: NDArray[np.floating]) -> NDArray[np.floating]: ...
