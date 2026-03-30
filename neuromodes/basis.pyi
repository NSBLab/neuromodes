import numpy as np
from typing import overload, TypeAlias, Literal
from numpy import floating
from numpy.typing import NDArray
from scipy.sparse import csc_matrix
from scipy.spatial.distance import _MetricCallback, _MetricKind 

from neuromodes.eigen import _CheckKind

# %% TYPE ALIASES
# Base types for decompose
_FloatArray: TypeAlias = NDArray[floating]
_ListArray: TypeAlias = list[NDArray[floating]]

# Types for reconstruct (Tuple of 3)
_ReconSingle: TypeAlias = tuple[_FloatArray, _FloatArray, _FloatArray]
_ReconList: TypeAlias = tuple[_FloatArray, _FloatArray, _ListArray]

# Types for reconstruct_timeseries (Tuple of 5)
_ReconTSSingle: TypeAlias = tuple[_FloatArray, _FloatArray, _FloatArray, _FloatArray, _FloatArray]
_ReconTSList: TypeAlias = tuple[_FloatArray, _FloatArray, _FloatArray, _FloatArray, _ListArray]

# Generic Types for inputs
Matrix: TypeAlias = csc_matrix | None
_MetricCallbackKind: TypeAlias = _MetricCallback | _MetricKind | None
_IntSequenceKind: TypeAlias = list[int] | tuple[int, ...] | NDArray[np.integer]
_DecompositionKind: TypeAlias = Literal['project', 'regress']

# %% DECOMPOSE
# 1. mode_counts is None or int -> Single Array 
@overload
def decompose(
    data: _FloatArray,
	emodes: _FloatArray,
	method: _DecompositionKind = ...,
	*,
    mass: Matrix = ...,
	mode_counts: int | None = ...,
	mode_ids: None = ...,
	checks: _CheckKind = ...
) -> _FloatArray: ...

# 2. mode_counts is Sequence -> List of Arrays
@overload
def decompose(
    data: _FloatArray,
	emodes: _FloatArray,
	method: _DecompositionKind = ...,
	*,
    mass: Matrix = ...,
	mode_counts: _IntSequenceKind,
	mode_ids: None = ...,
	checks: _CheckKind = ...
) -> _ListArray: ...

# 3. mode_ids is Sequence -> List of Arrays
@overload
def decompose(
    data: _FloatArray,
	emodes: _FloatArray,
	method: _DecompositionKind = ...,
	*,
    mass: Matrix = ...,
	mode_counts: None = ...,
	mode_ids: _IntSequenceKind,
	checks: _CheckKind = ...
) -> _ListArray: ...

# %% RECONSTRUCT
# 1. mode_counts is None or int -> Tuple with Single Array
@overload
def reconstruct(
    data: _FloatArray,
	emodes: _FloatArray,
	method: _DecompositionKind = ...,
	*,
    mass: Matrix = ...,
	mode_counts: int | None = ...,
	mode_ids: None = ...,
	checks: _CheckKind = ...,
	metric: _MetricCallbackKind = ...,
	**cdist_kwargs
) -> _ReconSingle: ...

# 2. mode_counts is Sequence -> Tuple with List of Arrays
@overload
def reconstruct(
    data: _FloatArray,
	emodes: _FloatArray,
	method: _DecompositionKind = ...,
	*,
    mass: Matrix = ...,
	mode_counts: _IntSequenceKind,
	mode_ids: None = ...,
	checks: _CheckKind = ...,
	metric: _MetricCallbackKind = ...,
	**cdist_kwargs
) -> _ReconList: ...

# 3. Mode IDs -> Tuple with List of Arrays
@overload
def reconstruct(
    data: _FloatArray,
	emodes: _FloatArray,
	method: _DecompositionKind = ...,
	*,
    mass: Matrix = ...,
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
    timeseries: _FloatArray,
	emodes: _FloatArray,
	method: _DecompositionKind = ...,
	*,
    mass: Matrix = ...,
	mode_counts: int | None = ...,
	mode_ids: None = ...,
	checks: _CheckKind = ...,
	metric: _MetricCallbackKind = ...,
	**cdist_kwargs
) -> _ReconTSSingle: ...

# 2. mode_counts is Sequence -> Tuple with List of Arrays
@overload
def reconstruct_timeseries(
    timeseries: _FloatArray,
	emodes: _FloatArray,
	method: _DecompositionKind = ...,
	*,
    mass: Matrix = ...,
	mode_counts: _IntSequenceKind,
	mode_ids: None = ...,
	checks: _CheckKind = ...,
	metric: _MetricCallbackKind = ...,
	**cdist_kwargs
) -> _ReconTSList: ...

# 3. Mode IDs -> Tuple with List of Arrays
@overload
def reconstruct_timeseries(
    timeseries: _FloatArray,
	emodes: _FloatArray,
	method: _DecompositionKind = ...,
	*,
    mass: Matrix = ...,
	mode_counts: None = ...,
	mode_ids: _IntSequenceKind,
	checks: _CheckKind = ...,
	metric: _MetricCallbackKind = ...,
	**cdist_kwargs
) -> _ReconTSList: ...

# %% OTHERS
def calc_norm_power(beta: _FloatArray) -> _FloatArray: ...
def calc_vec_fc(timeseries: _FloatArray) -> _FloatArray: ...
