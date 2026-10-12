from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)


if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(
        self,
        x: Tensor,
        out: Optional[Tensor] = ...,
        /,
    ) -> Tensor:
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        raise NotImplementedError

    @staticmethod
    def cmap(
        fn: Callable[[float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        raise NotImplementedError

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        raise NotImplementedError

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float],
        start: float = 0.0,
    ) -> Callable[[Tensor, int], Tensor]:
        raise NotImplementedError

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a tensor_ops
        object that implements map, cmap, zip, and reduce.
        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)

        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)

        # Matrix multiplication
        self.matrix_multiply = ops.matrix_multiply

        # Backend type
        self.cuda = ops.cuda


class SimpleOps(TensorOps):

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function.

        Applies fn element-by-element to a tensor.

        Example:

            out = map(fn)(a)
        """

        f = tensor_map(fn)

        def ret(
            a: "Tensor",
            out: Optional["Tensor"] = None,
        ) -> "Tensor":

            if out is None:
                out = a.zeros(a.shape)

            f(
                *out.tuple(),
                *a.tuple(),
            )

            return out

        return ret

    @staticmethod
    def cmap(
        fn: Callable[[float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Contiguous map.

        Applies fn element-by-element from input tensor `a`
        into an already allocated output tensor `out`.
        """

        f = tensor_map(fn)

        def ret(
            a: "Tensor",
            out: "Tensor",
        ) -> "Tensor":

            f(
                *out.tuple(),
                *a.tuple(),
            )

            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function.

        Applies fn element-by-element to two tensors.

        Supports broadcasting.
        """

        f = tensor_zip(fn)

        def ret(
            a: "Tensor",
            b: "Tensor",
        ) -> "Tensor":

            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape

            out = a.zeros(c_shape)

            f(
                *out.tuple(),
                *a.tuple(),
                *b.tuple(),
            )

            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float],
        start: float = 0.0,
    ) -> Callable[["Tensor", int], "Tensor"]:
        """
        Higher-order tensor reduce function.

        Reduces one dimension of a tensor.
        """

        f = tensor_reduce(fn)

        def ret(
            a: "Tensor",
            dim: int,
        ) -> "Tensor":

            out_shape = list(a.shape)
            out_shape[dim] = 1

            out = a.zeros(tuple(out_shape))

            # Initialize output with start value.
            out._tensor._storage[:] = start

            f(
                *out.tuple(),
                *a.tuple(),
                dim,
            )

            return out

        return ret

    
    @staticmethod
    def matrix_multiply(
        a: "Tensor",
        b: "Tensor",
    ) -> "Tensor":
        """Multiply two 2D tensors: (m, n) @ (n, p) -> (m, p)."""

        if len(a.shape) != 2 or len(b.shape) != 2:
            raise NotImplementedError(
                "SimpleOps matrix_multiply currently supports 2D tensors only"
            )

        m, n = a.shape
        n_b, p = b.shape

        if n != n_b:
            raise ValueError(
                f"Incompatible shapes for matrix multiplication: "
                f"{a.shape} and {b.shape}"
            )

        out = a.zeros((m, p))

        for i in range(m):
            for j in range(p):
                total = 0.0

                for k in range(n):
                    total += (
                        a._tensor.get((i, k))
                        * b._tensor.get((k, j))
                    )

                out._tensor.set((i, j), total)

        return out



def tensor_map(fn: Callable[[float], float]):
    """
    Higher-order tensor map.

    Applies fn to every element of the input tensor.
    """

    def _map(
        out,
        out_shape,
        out_strides,
        in_storage,
        in_shape,
        in_strides,
    ):

        out_index = np.zeros(MAX_DIMS, dtype=np.int32)
        in_index = np.zeros(MAX_DIMS, dtype=np.int32)

        for i in range(int(np.prod(out_shape))):

            # Convert ordinal to multidimensional index.
            to_index(
                i,
                out_shape,
                out_index,
            )

            # Convert output index to input index.
            broadcast_index(
                out_index,
                out_shape,
                in_shape,
                in_index,
            )

            # Convert indexes to storage positions.
            out_pos = index_to_position(
                out_index,
                out_strides,
            )

            in_pos = index_to_position(
                in_index[:len(in_shape)],
                in_strides,
            )

            # Apply function.
            out[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_zip(fn: Callable[[float, float], float]):
    """
    Higher-order tensor zip.

    Applies fn to two tensors element-by-element
    with broadcasting support.
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):

        out_index = np.zeros(
            MAX_DIMS,
            dtype=np.int32,
        )

        a_index = np.zeros(
            MAX_DIMS,
            dtype=np.int32,
        )

        b_index = np.zeros(
            MAX_DIMS,
            dtype=np.int32,
        )

        for i in range(int(np.prod(out_shape))):

            # Convert ordinal to output index.
            to_index(
                i,
                out_shape,
                out_index,
            )

            # Broadcast output index into each input.
            broadcast_index(
                out_index,
                out_shape,
                a_shape,
                a_index,
            )

            broadcast_index(
                out_index,
                out_shape,
                b_shape,
                b_index,
            )

            # Convert indexes to storage positions.
            out_pos = index_to_position(
                out_index,
                out_strides,
            )

            a_pos = index_to_position(
                a_index[:len(a_shape)],
                a_strides,
            )

            b_pos = index_to_position(
                b_index[:len(b_shape)],
                b_strides,
            )

            # Apply function.
            out[out_pos] = fn(
                a_storage[a_pos],
                b_storage[b_pos],
            )

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
):
    """
    Higher-order tensor reduce.

    Reduces one dimension of a tensor.
    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        reduce_dim,
    ):

        out_index = np.zeros(
            MAX_DIMS,
            dtype=np.int32,
        )

        a_index = np.zeros(
            MAX_DIMS,
            dtype=np.int32,
        )

        for i in range(int(np.prod(out_shape))):

            # Convert ordinal to output index.
            to_index(
                i,
                out_shape,
                out_index,
            )

            # Find output storage position.
            out_pos = index_to_position(
                out_index,
                out_strides,
            )

            # Copy output index into input index.
            for j in range(len(out_shape)):
                a_index[j] = out_index[j]

            # Iterate through reduced dimension.
            for j in range(a_shape[reduce_dim]):

                a_index[reduce_dim] = j

                a_pos = index_to_position(
                    a_index[:len(a_shape)],
                    a_strides,
                )

                out[out_pos] = fn(
                    out[out_pos],
                    a_storage[a_pos],
                )

    return _reduce



# Create the default CPU backend.
SimpleBackend = TensorBackend(SimpleOps)