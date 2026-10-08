from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import numba
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod


MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Convert a multidimensional tensor index into a single-dimensional
    position in storage based on strides.

    Args:
        index: Index tuple of ints as a NumPy array.
        strides: Tensor strides as a NumPy array.

    Returns:
        Position in storage.
    """
    position = 0

    for i, s in zip(index, strides):
        position += i * s

    return int(position)


def to_index(
    ordinal: int,
    shape: Shape,
    out_index: OutIndex,
) -> None:
    """Convert an ordinal position into a multidimensional tensor index.

    Args:
        ordinal: Ordinal position from 0 to size - 1.
        shape: Tensor shape.
        out_index: Output array to fill with index values.

    Returns:
        None. Modifies out_index in place.
    """
    cur_ord = ordinal

    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = cur_ord % shape[i]
        cur_ord //= shape[i]


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape

    strides: UserStrides
    shape: UserShape
    dims: int
    size: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        # Convert storage to NumPy array if necessary.
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        # Generate contiguous strides if no strides were provided.
        if strides is None:
            strides = strides_from_shape(shape)

        # MiniTorch expects shape and strides to be tuples.
        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"

        if len(strides) != len(shape):
            raise IndexingError(
                f"Strides {strides} and shape {shape} "
                "must be the same length"
            )

        self._strides = array(strides, dtype=np.int32)
        self._shape = array(shape, dtype=np.int32)

        self.strides = strides
        self.shape = shape
        self.dims = len(shape)
        self.size = int(prod(shape))

        if len(self._storage) != self.size:
            raise IndexingError(
                f"Storage size {len(self._storage)} does not match "
                f"tensor size {self.size}."
            )

    def to_cuda_(self) -> None:
        """Move storage to CUDA."""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check whether the tensor layout is contiguous.

        A contiguous tensor has decreasing strides from the outermost
        dimension to the innermost dimension.

        Returns:
            True if the tensor is contiguous, otherwise False.
        """
        last = 1e9

        for stride in self._strides:
            if stride > last:
                return False

            last = stride

        return True

    @staticmethod
    def shape_broadcast(
        shape_a: UserShape,
        shape_b: UserShape,
    ) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Convert a user tensor index into a storage position.

        Args:
            index:
                Either a single integer or a sequence of integers.

        Returns:
            The corresponding position in the underlying storage.
        """

        # Convert integer index to a one-dimensional NumPy array.
        if isinstance(index, int):
            aindex: Index = array([index], dtype=np.int32)
        else:
            aindex = array(index, dtype=np.int32)

        # Check number of dimensions.
        if len(aindex) != self.dims:
            raise IndexingError(
                f"Index {aindex} must be size of {self.shape}."
            )

        # Check every index is inside the valid range.
        for i, ind in enumerate(aindex):
            if ind < 0:
                raise IndexingError(
                    f"Negative indexing for {aindex} not supported."
                )

            if ind >= self.shape[i]:
                raise IndexingError(
                    f"Index {aindex} out of range {self.shape}."
                )

        # Convert multidimensional index to flat storage position.
        return index_to_position(aindex, self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generate every valid index in the tensor."""
        lshape: Shape = array(self.shape, dtype=np.int32)

        out_index: OutIndex = array(
            self.shape,
            dtype=np.int32,
        )

        for i in range(self.size):
            to_index(i, lshape, out_index)

            yield tuple(int(x) for x in out_index)

    def sample(self) -> UserIndex:
        """Generate a random valid tensor index."""
        return tuple(
            random.randint(0, s - 1)
            for s in self.shape
        )

    def get(self, key: UserIndex) -> float:
        """Get a value from the tensor."""
        x: float = self._storage[self.index(key)]

        return float(x)

    def set(
        self,
        key: UserIndex,
        val: float,
    ) -> None:
        """Set a value in the tensor."""
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return storage, shape, and strides."""
        return (
            self._storage,
            self._shape,
            self._strides,
        )

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
            *order:
                A permutation of the tensor dimensions.

        Returns:
            A new TensorData object sharing the same storage but
            with a different shape and stride ordering.
        """

        assert list(sorted(order)) == list(range(len(self.shape))), (
            f"Must give a position to each dimension. "
            f"Shape: {self.shape} Order: {order}"
        )

        new_shape = tuple(
            self.shape[o]
            for o in order
        )

        new_strides = tuple(
            self.strides[o]
            for o in order
        )

        return TensorData(
            self._storage,
            new_shape,
            new_strides,
        )

    def to_string(self) -> str:
        """Return a formatted string representation of the tensor."""
        s = ""

        for index in self.indices():
            l = ""

            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break

            s += l

            v = self.get(index)
            s += f"{v:3.2f}"

            l = ""

            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break

            if l:
                s += l
            else:
                s += " "

        return s




def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new unnion shape"""
    result = []
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)

    for i in range(max_len):
        d1 = shape1[len1 - 1 - i] if i < len1 else 1
        d2 = shape2[len2 - 1 - i] if i < len2 else 1

        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise IndexingError(
                f"Cannot broadcast shapes {shape1} and {shape2}"
            )
    return tuple(reversed(result))



def broadcast_index(
        big_index:Index,
        big_shape:Shape,
        shape:Shape,
        out_index:OutIndex
)-> None:
        """Convert index from broadcasted shape to original shape."""

        offset = len(big_shape) - len(shape)

        for i in range(len(shape)):
            if shape[i] == 1:
                out_index[i] = 0
            else:
                out_index[i] = big_index[i + offset]


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Compute contiguous strides for a given shape.

    Args:
        shape: Tensor shape.
        """
    layout = [1]
    offset = 1

    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset

    return tuple(reversed(layout[:-1]))


