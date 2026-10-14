from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index
)

if TYPE_CHECKING:
    from typing import Callable, Optional
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides
    from .tensor_ops import MapProto, TensorOps


#Helper to convert oridinal to index for a specific dimension
@njit(inline="always")
def _get_idx(ordinal, shape, k):
    """Get the index for dimension k from a flat ordinal postion"""
    div = 1
    for s in range(k + 1, len(shape)):
        div *= shape[s]
    return(ordinal // div) % shape[k]



#Now implement tensor_map:
def tensor_map(fn):
    f = fn
    def _map(out, out_shape, out_strides, in_storage,in_shape, in_strides):
        for i in prange(len(out)):
            out_pos = 0
            in_pos = 0
            for k in range(len(out_shape)):
                idx_k = _get_idx(i, out_shape, k)
                out_pos += idx_k * out_strides[k]

                #Handle broadcasting : map output dim to input dim
                in_k = k - (len(out_shape) - len(in_shape))
                if in_k >= 0 and in_shape[in_k] > 1:
                    in_pos += idx_k * in_strides[in_k]

            out[out_pos] = f(in_storage[in_pos])

    return njit(parallel=True)(_map)


#Implement tensor_zip()


def tensor_zip(fn):
    f = fn

    def _zip(
        out, out_shape, out_strides,
        a_storage, a_shape, a_strides,
        b_storage, b_shape, b_strides,
    ):
        for i in prange(len(out)):
            out_pos = 0
            a_pos = 0
            b_pos = 0

            for k in range(len(out_shape)):
                idx_k = _get_idx(i, out_shape, k)

                out_pos += idx_k * out_strides[k]

                # Map output dimension to input A.
                a_k = k - (len(out_shape) - len(a_shape))
                if a_k >= 0 and a_shape[a_k] > 1:
                    a_pos += idx_k * a_strides[a_k]

                # Map output dimension to input B.
                b_k = k - (len(out_shape) - len(b_shape))
                if b_k >= 0 and b_shape[b_k] > 1:
                    b_pos += idx_k * b_strides[b_k]

            out[out_pos] = f(
                a_storage[a_pos],
                b_storage[b_pos],
            )

    return njit(parallel=True)(_zip)




#Implement tensor_reduce(fn)
def tensor_reduce(fn):
    f = fn
    def _reduce(out, out_shape, out_strides, in_storage, in_shape, in_strides, reduce_dim):
        for i in prange(len(out)):
            out_pos = 0
            in_base_pos = 0
            for k in range (len(out_shape)):
                idx_k = _get_idx(i, out_shape, k)
                out_pos += idx_k * out_strides[k]
                in_base_pos += idx_k * in_strides[k]

            accum = out[out_pos]
            for j in range(in_shape[reduce_dim]):
                accum = f(accum, in_storage[in_base_pos + j * in_strides[reduce_dim]])
            out[out_pos] = accum
    return njit(parallel=True)(_reduce)





               


                


