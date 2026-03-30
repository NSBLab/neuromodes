import numpy as np
from typing import overload
from numpy import floating
from numpy.typing import NDArray
from scipy.sparse import csc_matrix
from scipy.spatial.distance import _MetricCallback, _MetricKind 

# %% DECOMPOSE
# 1. mode_ids unspecified, and mode counts is None or int -> Array 
@overload
def decompose(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: None = ...,
    mode_ids: None = ...,
    checks: str | bool = ...
) -> NDArray[floating]: ...

@overload
def decompose(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: int,
    mode_ids: None = ...,
    checks: str | bool = ...
) -> NDArray[floating]: ...

# 2. mode counts is list/tuple -> list of Arrays
@overload
def decompose(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: list | tuple | NDArray,
    mode_ids: None = ...,
    checks: str | bool = ...
) -> list[NDArray[floating]]: ...

# 3. Mode IDs -> list of Arrays
@overload
def decompose(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: None = ...,
    mode_ids: list | tuple | NDArray,
    checks: str | bool = ...
) -> list[NDArray[floating]]: ...


# %% RECONSTRUCT
@overload
def reconstruct(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: None = ...,
    mode_ids: None = ...,
    checks: str | bool = ...,
    metric: _MetricCallback | _MetricKind | None = ...,
    **cdist_kwargs
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating]]: ...

@overload
def reconstruct(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: int,
    mode_ids: None = ...,
    checks: str | bool = ...,
    metric: _MetricCallback | _MetricKind | None = ...,
    **cdist_kwargs
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating]]: ...

# 2. mode counts is list/tuple -> list of Arrays
@overload
def reconstruct(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: list | tuple | NDArray,
    mode_ids: None = ...,
    checks: str | bool = ...,
    metric: _MetricCallback | _MetricKind | None = ...,
    **cdist_kwargs
) -> tuple[NDArray[floating], NDArray[floating], list[NDArray[floating]]]: ...

# 3. Mode IDs -> list of Arrays
@overload
def reconstruct(
    data: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: None = ...,
    mode_ids: list | tuple | NDArray,
    checks: str | bool = ...,
    metric: _MetricCallback | _MetricKind | None = ...,
    **cdist_kwargs
) -> tuple[NDArray[floating], NDArray[floating], list[NDArray[floating]]]: ...

# %% RECONSTRUCT_TIMESERIES
@overload
def reconstruct_timeseries(
    timeseries: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: None = ...,
    mode_ids: None = ...,
    checks: str | bool = ...,
    metric: _MetricCallback | _MetricKind | None = ...,
    **cdist_kwargs
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating]]: ...

@overload
def reconstruct_timeseries(
    timeseries: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: int,
    mode_ids: None = ...,
    checks: str | bool = ...,
    metric: _MetricCallback | _MetricKind | None = ...,
    **cdist_kwargs
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating]]: ...

# 2. mode counts is list/tuple -> list of Arrays
@overload
def reconstruct_timeseries(
    timeseries: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: list | tuple | NDArray,
    mode_ids: None = ...,
    checks: str | bool = ...,
    metric: _MetricCallback | _MetricKind | None = ...,
    **cdist_kwargs
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating], list[NDArray[floating]]]: ...

# 3. Mode IDs -> list of Arrays
@overload
def reconstruct_timeseries(
    timeseries: NDArray,
    emodes: NDArray,
    method: str = ...,
    *, 
    mass: csc_matrix | None = ...,
    mode_counts: None = ...,
    mode_ids: list | tuple | NDArray,
    checks: str | bool = ...,
    metric: _MetricCallback | _MetricKind | None = ...,
    **cdist_kwargs
) -> tuple[NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating], list[NDArray[floating]]]: ...

# %% OTHERS
def calc_norm_power(
    beta: NDArray[floating]
) -> NDArray[floating]: ...

def calc_vec_fc(
    timeseries: NDArray
) -> NDArray[floating]: ...
