from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from typing import Callable, Optional
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides
    from .tensor_ops import MapProto, TensorOps

class FastOps:
    cuda = False # this backend is not a CUDA backend ,it desined for CPU execution 

    #High lavel implementation 
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret
    
    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """
        Higher-order tensor map function for columns.
        """
        f = FastOps.map(fn)

        def ret(a: Tensor, out: Tensor) -> Tensor:
            return f(a, out)

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        _tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out



#Helper to convert oridinal to index for a specific dimension
@njit(inline="always")
def _get_idx(ordinal, shape, k):
    """Get the index for dimension k from a flat ordinal postion"""
    div = 1
    for s in range(k + 1, len(shape)):
        div *= shape[s]
    return(ordinal // div) % shape[k]




# Low lavel implementation 
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





def _tensor_matrix_multiply(
        out, out_shape, out_strides,
        a_storage, a_shape, a_strides,
        b_storage, b_shape, b_strides
):
    """Numba tesnor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as:
    assert a_shape[-1] == b_shape[-2]


    Optimizations:
    * outer look in parallel
    * no index buffers or function calls 
    * inner loop should have no global writes , 1 multiply
    """

    #handle batch dimension broadcasting
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    batch_size = out_shape[0]
    M = out_shape[1]
    N = out_shape[2]
    K = a_shape[2]

    for n in prange(batch_size):
        for i in range(M):
            for j in range(N):
                out_pos = (
                    n * out_strides[0] + 
                    i * out_strides[1] + 
                    j * out_strides[2]
                )

                acc = 0.0
                for k in range(K):
                    a_pos = (
                        n * a_batch_stride +
                        i * a_strides[1] + 
                        k * a_strides[2]
                    )
                    b_pos = (
                        n * b_batch_stride + 
                        k * b_strides[1] + 
                        j * b_strides[2]
                    )

                    acc += a_storage[a_pos] * b_storage[b_pos]

                out[out_pos] = acc

_tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)






                


