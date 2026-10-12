from .autodiff import central_difference ,topological_sort, backpropagate
from .scalar import Scalar
from minitorch.scalar import Scalar
from .tensor_data import TensorData, IndexingError, shape_broadcast
from .operators import prod
from .tensor import Tensor
from .tensor_functions import tensor , zeros
from .testing import MathTestVariable, grad_check
from .tensor_ops import SimpleBackend
from .module import Module, Parameter




# # a = Scalar(2.0, name="a")
# # b = Scalar(3.0, name="b")

# # print(f"a.data = {a.data}")
# # print(f"a.is_leaf() = {a.is_leaf()}")
# # print(f"a.derivative = {a.derivative}")

# # a.accumulate_derivative(1.0)
# # a.accumulate_derivative(2.0)
# # print(f"After accumulating: a.derivative = {a.derivative}")





# # # test for ,forward computation + computation history
# # a = Scalar(2.0)
# # a.requires_grad_(True)

# # b = Scalar(3.0)
# # b.requires_grad_(True)

# # c = a * b

# # print(f"c.data = {c.data}")
# # print(f"c.is_leaf() = {c.is_leaf()}")
# # print(f"c.history.last_fn = {c.history.last_fn}")



# # from minitorch.scalar import Scalar

# # a = Scalar(2.0)
# # a.requires_grad_(True)
# # b = Scalar(3.0)
# # b.requires_grad_(True)

# # c = a * b     # c = 6
# # d = c + a     # d = 6 + 2 = 8
# # d.backward()

# # print(f"a.derivative = {a.derivative}")  # d/da
# # print(f"b.derivative = {b.derivative}")  # d/db


# from minitorch.scalar import Scalar
# from minitorch.autodiff import backpropagate

# a = Scalar(2.0)
# a.requires_grad_(True)

# b = Scalar(3.0)
# b.requires_grad_(True)

# c = Scalar(44.0)
# c.requires_grad_(True)

# f= (a * b) / c;

# print("Before backward:")
# print(f"a.derivative = {a.derivative}")
# print(f"b.derivative = {b.derivative}")
# print(f"c.derivative = {c.derivative}")
# print(f"c.derivative = {f.derivative}")

# backpropagate(f)

# print("\nAfter backward:")
# print(f"a.derivative = {a.derivative}")
# print(f"b.derivative = {b.derivative}")
# print(f"c.derivative = {c.derivative}")
# print(f"c.derivative = {f.derivative}")



from minitorch.tensor_data import (
    shape_broadcast,
    strides_from_shape,
)

print(shape_broadcast((2, 3), (3,)))
print(shape_broadcast((2, 1, 4), (3, 4)))
print(strides_from_shape((2, 3, 4)))