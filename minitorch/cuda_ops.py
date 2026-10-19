"""CUDA tensor operations for MiniTorch.

This implementation assumes TensorData storage is compatible with Numba CUDA
(device arrays when running on a real GPU) and that tensor_data helpers are
CUDA-callable, as required by the rest of the project.
"""

import numba
import numpy as np
from numba import cuda

from .tensor_data import MAX_DIMS, to_index, index_to_position, broadcast_index

THREADS_PER_BLOCK = 32
TILE_SIZE = 16  # Conservative tile size: 16x16 threads and shared-memory tiles.


def _blocks(size, threads=THREADS_PER_BLOCK):
    """Return a nonzero 1D CUDA grid size."""
    return max(1, (size + threads - 1) // threads)


# ---------------------------------------------------------------------------
# 1. MAP
# ---------------------------------------------------------------------------

def tensor_map(fn):
    f = cuda.jit(device=True)(fn)

    @cuda.jit
    def _map(out, out_shape, out_strides, out_size,
             in_storage, in_shape, in_strides):
        i = cuda.grid(1)
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            in_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            out[out_pos] = f(in_storage[in_pos])

    return _map


# ---------------------------------------------------------------------------
# 2. ZIP / ELEMENTWISE BINARY OPERATION
# ---------------------------------------------------------------------------

def tensor_zip(fn):
    f = cuda.jit(device=True)(fn)

    @cuda.jit
    def _zip(out, out_shape, out_strides, out_size,
             a_storage, a_shape, a_strides,
             b_storage, b_shape, b_strides):
        i = cuda.grid(1)
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            b_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out[out_pos] = f(a_storage[a_pos], b_storage[b_pos])

    return _zip


# ---------------------------------------------------------------------------
# 3. REDUCE
# ---------------------------------------------------------------------------

def tensor_reduce(fn, reduce_value):
    f = cuda.jit(device=True)(fn)

    @cuda.jit
    def _reduce(out, out_shape, out_strides, out_size,
                a_storage, a_shape, a_strides, reduce_dim, start):
        i = cuda.grid(1)
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            acc = start
            for s in range(a_shape[reduce_dim]):
                out_index[reduce_dim] = s
                in_pos = index_to_position(out_index, a_strides)
                acc = f(acc, a_storage[in_pos])
            out[out_pos] = acc

    return _reduce


# ---------------------------------------------------------------------------
# 4. SHARED-MEMORY REDUCTION EXERCISE
# ---------------------------------------------------------------------------

@cuda.jit
def _sum_practice(out, in_storage, size):
    cache = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)
    i = cuda.grid(1)
    local_i = cuda.threadIdx.x
    block_id = cuda.blockIdx.x

    if i < size:
        cache[local_i] = in_storage[i]
    else:
        cache[local_i] = 0.0
    cuda.syncthreads()

    offset = THREADS_PER_BLOCK // 2
    while offset > 0:
        if local_i < offset:
            cache[local_i] += cache[local_i + offset]
        cuda.syncthreads()
        offset //= 2

    if local_i == 0:
        out[block_id] = cache[0]


def sum_practice(a):
    """Return one partial sum per block (not a single final sum)."""
    if len(a.shape) != 1:
        raise ValueError("sum_practice expects a 1D tensor")
    (size,) = a.shape
    blocks = _blocks(size)
    out = a.zeros((blocks,))
    _sum_practice[blocks, THREADS_PER_BLOCK](
        out._tensor._storage, a._tensor._storage, size
    )
    return out


# ---------------------------------------------------------------------------
# 5. TILED MATRIX MULTIPLICATION (2D and batched 3D)
# ---------------------------------------------------------------------------

@cuda.jit
def _mm_kernel(out, out_strides,
                a_storage, a_shape, a_strides,
                b_storage, b_shape, b_strides,
                m, k_size, n):
    """Compute (batch, M, K) @ (batch, K, N) using shared-memory tiles.

    Metadata is padded to 3 dimensions. A stride of zero in the batch axis
    allows a 2D input or a broadcast batch dimension to be reused.
    """
    a_tile = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)
    b_tile = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)

    batch = cuda.blockIdx.z
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.x * TILE_SIZE + tx
    col = cuda.blockIdx.y * TILE_SIZE + ty

    acc = 0.0
    for k_start in range(0, k_size, TILE_SIZE):
        a_tile[tx, ty] = 0.0
        b_tile[tx, ty] = 0.0

        a_batch = batch * a_strides[0]
        b_batch = batch * b_strides[0]
        if row < m and k_start + ty < k_size:
            a_tile[tx, ty] = a_storage[
                a_batch + row * a_strides[1] + (k_start + ty) * a_strides[2]
            ]
        if k_start + tx < k_size and col < n:
            b_tile[tx, ty] = b_storage[
                b_batch + (k_start + tx) * b_strides[1] + col * b_strides[2]
            ]
        cuda.syncthreads()

        for kk in range(TILE_SIZE):
            acc += a_tile[tx, kk] * b_tile[kk, ty]
        cuda.syncthreads()

    if row < m and col < n:
        out_pos = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]
        out[out_pos] = acc


def _as_3d_metadata(tensor):
    """Pad 2D tensor metadata with a broadcast batch dimension."""
    shape = tuple(tensor.shape)
    strides = tuple(tensor.strides)
    if len(shape) == 2:
        shape = (1,) + shape
        strides = (0,) + strides
    elif len(shape) != 3:
        raise ValueError("matrix_multiply supports 2D or 3D tensors only")
    return np.asarray(shape, dtype=np.int32), np.asarray(strides, dtype=np.int32)


def matrix_multiply_launcher(out, a, b):
    a_shape, a_strides = _as_3d_metadata(a._tensor)
    b_shape, b_strides = _as_3d_metadata(b._tensor)
    out_shape, out_strides = _as_3d_metadata(out._tensor)

    m, k_size = int(a_shape[1]), int(a_shape[2])
    k_b, n = int(b_shape[1]), int(b_shape[2])
    if k_size != k_b:
        raise ValueError(f"Incompatible matrix dimensions: {k_size} and {k_b}")

    batch = int(out_shape[0])
    grid = (
        max(1, (m + TILE_SIZE - 1) // TILE_SIZE),
        max(1, (n + TILE_SIZE - 1) // TILE_SIZE),
        max(1, batch),
    )
    _mm_kernel[grid, (TILE_SIZE, TILE_SIZE, 1)](
        out._tensor._storage, out_strides,
        a._tensor._storage, a_shape, a_strides,
        b._tensor._storage, b_shape, b_strides,
        m, k_size, n,
    )


# ---------------------------------------------------------------------------
# 6. BACKEND ADAPTER
# ---------------------------------------------------------------------------

class CudaOps:
    cuda = True

    @staticmethod
    def map(fn):
        kernel = tensor_map(fn)

        def ret(a):
            out = a.zeros(a.shape)
            if out.size == 0:
                return out
            kernel[_blocks(out.size), THREADS_PER_BLOCK](
                out._tensor._storage, out._tensor._shape,
                out._tensor._strides, out.size,
                a._tensor._storage, a._tensor._shape,
                a._tensor._strides,
            )
            return out
        return ret

    @staticmethod
    def zip(fn):
        kernel = tensor_zip(fn)

        def ret(a, b):
            out_shape = _broadcast_shape(a.shape, b.shape)
            out = a.zeros(out_shape)
            if out.size == 0:
                return out
            kernel[_blocks(out.size), THREADS_PER_BLOCK](
                out._tensor._storage, out._tensor._shape,
                out._tensor._strides, out.size,
                a._tensor._storage, a._tensor._shape, a._tensor._strides,
                b._tensor._storage, b._tensor._shape, b._tensor._strides,
            )
            return out
        return ret

    @staticmethod
    def reduce(fn, start=0.0):
        kernel = tensor_reduce(fn, start)

        def ret(a, dim):
            if dim < 0:
                dim += len(a.shape)
            if dim < 0 or dim >= len(a.shape):
                raise IndexError("reduction dimension out of range")
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            if out.size == 0:
                return out
            kernel[_blocks(out.size), THREADS_PER_BLOCK](
                out._tensor._storage, out._tensor._shape,
                out._tensor._strides, out.size,
                a._tensor._storage, a._tensor._shape,
                a._tensor._strides, int(dim), start,
            )
            return out
        return ret

    @staticmethod
    def matrix_multiply(a, b):
        if len(a.shape) not in (2, 3) or len(b.shape) not in (2, 3):
            raise ValueError("matrix_multiply supports 2D or 3D tensors only")
        a_batch = a.shape[0] if len(a.shape) == 3 else 1
        b_batch = b.shape[0] if len(b.shape) == 3 else 1
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(
                f"Incompatible matrix dimensions: {a.shape[-1]} and {b.shape[-2]}"
            )
        if a_batch != b_batch and a_batch != 1 and b_batch != 1:
            raise ValueError(f"Batch dimensions {a_batch} and {b_batch} cannot broadcast")
        batch = max(a_batch, b_batch)
        m, n = a.shape[-2], b.shape[-1]
        if len(a.shape) == 2 and len(b.shape) == 2:
            out_shape = (m, n)
        else:
            out_shape = (batch, m, n)
        out = a.zeros(out_shape)
        if out.size:
            matrix_multiply_launcher(out, a, b)
        return out

    cmap = map


def _broadcast_shape(shape1, shape2):
    """Compute standard right-aligned broadcast shape or raise ValueError."""
    result = []
    max_dims = max(len(shape1), len(shape2))
    padded1 = (1,) * (max_dims - len(shape1)) + tuple(shape1)
    padded2 = (1,) * (max_dims - len(shape2)) + tuple(shape2)
    for d1, d2 in zip(padded1, padded2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
        result.append(max(d1, d2))
    return tuple(result)
