from __future__ import annotations


import random

from typing import TYPE_CHECKING, Any, List, Tuple


import numpy as np


import minitorch

from . import operators

from .autodiff import Context, History

from .tensor_ops import SimpleBackend, TensorBackend


if TYPE_CHECKING:

    from .tensor import Tensor

    from .tensor_data import UserIndex, UserShape


# ----------------------------------------------------------------------

# Helpers

# ----------------------------------------------------------------------


def wrap_tuple(x):

    """Turn a possible value into a tuple."""

    if isinstance(x, tuple):

        return x

    return (x,)


# ----------------------------------------------------------------------

# Function

# ----------------------------------------------------------------------


class Function:

    @classmethod

    def _backward(

        cls,

        ctx: Context,

        grad_out: Tensor,

    ) -> Tuple[Tensor, ...]:

        return wrap_tuple(cls.backward(ctx, grad_out))


    @classmethod

    def _forward(

        cls,

        ctx: Context,

        *inps: Tensor,

    ) -> Tensor:

        return cls.forward(ctx, *inps)


    @classmethod

    def apply(

        cls,

        *vals: Tensor,

    ) -> Tensor:


        raw_vals = []

        need_grad = False


        for v in vals:

            if v.requires_grad():

                need_grad = True


            raw_vals.append(v.detach())


        # Create context.

        ctx = Context(not need_grad)


        # Forward pass.

        c = cls._forward(ctx, *raw_vals)


        # Create history only when gradients are required.

        back = None


        if need_grad:

            back = History(

                cls,

                ctx,

                vals,

            )


        return minitorch.Tensor(

            c._tensor,

            back,

            backend=c.backend,

        )


# ----------------------------------------------------------------------

# Basic operations

# ----------------------------------------------------------------------


class Neg(Function):


    @staticmethod

    def forward(

        ctx: Context,

        t1: Tensor,

    ) -> Tensor:

        return t1.f.neg_map(t1)


    @staticmethod

    def backward(

        ctx: Context,

        grad_output: Tensor,

    ) -> Tuple[Tensor]:

        return (grad_output.f.neg_map(grad_output),)


class Inv(Function):


    @staticmethod

    def forward(

        ctx: Context,

        t1: Tensor,

    ) -> Tensor:

        ctx.save_for_backward(t1)

        return t1.f.inv_map(t1)


    @staticmethod

    def backward(

        ctx: Context,

        grad_output: Tensor,

    ) -> Tuple[Tensor]:

        (t1,) = ctx.saved_values


        return (

            grad_output.f.inv_back_zip(

                t1,

                grad_output,

            ),

        )


# ----------------------------------------------------------------------

# Broadcasting gradient helper

# ----------------------------------------------------------------------


def grad_reduce_broadcast(

    grad: Tensor,

    original_shape: UserShape,

) -> Tensor:

    """

    Reduce a gradient tensor to match the original shape

    by summing over broadcasted dimensions.

    """


    if grad.shape == original_shape:

        return grad


    # Add leading 1s to original shape.

    ndims_added = len(grad.shape) - len(original_shape)


    padded_shape = (1,) * ndims_added + tuple(original_shape)


    # Sum dimensions that were broadcast.

    for i in range(len(grad.shape)):


        if padded_shape[i] == 1 and grad.shape[i] > 1:

            grad = grad.sum(i)


    # Restore original shape.

    return grad.contiguous().view(*original_shape)


# ----------------------------------------------------------------------

# Add

# ----------------------------------------------------------------------


class Add(Function):


    @staticmethod

    def forward(

        ctx: Context,

        t1: Tensor,

        t2: Tensor,

    ) -> Tensor:


        ctx.save_for_backward(

            t1.shape,

            t2.shape,

        )


        return t1.f.add_zip(t1, t2)


    @staticmethod

    def backward(

        ctx: Context,

        grad_output: Tensor,

    ) -> Tuple[Tensor, Tensor]:


        t1_shape, t2_shape = ctx.saved_values


        return (

            grad_reduce_broadcast(

                grad_output,

                t1_shape,

            ),

            grad_reduce_broadcast(

                grad_output,

                t2_shape,

            ),

        )


# ----------------------------------------------------------------------

# Multiply

# ----------------------------------------------------------------------


class Mul(Function):


    @staticmethod

    def forward(

        ctx: Context,

        a: Tensor,

        b: Tensor,

    ) -> Tensor:


        ctx.save_for_backward(a, b)


        return a.f.mul_zip(a, b)


    @staticmethod

    def backward(

        ctx: Context,

        grad_output: Tensor,

    ) -> Tuple[Tensor, Tensor]:


        a, b = ctx.saved_values


        grad_a = grad_output.f.mul_zip(

            grad_output,

            b,

        )


        grad_b = grad_output.f.mul_zip(

            grad_output,

            a,

        )


        grad_a = grad_reduce_broadcast(

            grad_a,

            a.shape,

        )


        grad_b = grad_reduce_broadcast(

            grad_b,

            b.shape,

        )


        return grad_a, grad_b


# ----------------------------------------------------------------------

# Sigmoid

# ----------------------------------------------------------------------


class Sigmoid(Function):


    @staticmethod

    def forward(

        ctx: Context,

        t1: Tensor,

    ) -> Tensor:


        out = t1.f.sigmoid_map(t1)


        ctx.save_for_backward(out)


        return out


    @staticmethod

    def backward(

        ctx: Context,

        grad_output: Tensor,

    ) -> Tuple[Tensor]:


        (out,) = ctx.saved_values


        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))


        one = out.zeros(out.shape)

        one._tensor._storage[:] = 1.0


        derivative = out.f.mul_zip(

            out,

            out.f.add_zip(

                one,

                out.f.neg_map(out),

            ),

        )


        return (

            grad_output.f.mul_zip(

                grad_output,

                derivative,

            ),

        )


# ----------------------------------------------------------------------

# ReLU

# ----------------------------------------------------------------------


class ReLU(Function):


    @staticmethod

    def forward(

        ctx: Context,

        t1: Tensor,

    ) -> Tensor:


        ctx.save_for_backward(t1)


        return t1.f.relu_map(t1)


    @staticmethod

    def backward(

        ctx: Context,

        grad_output: Tensor,

    ) -> Tuple[Tensor]:


        (t1,) = ctx.saved_values


        return (

            grad_output.f.relu_back_zip(

                t1,

                grad_output,

            ),

        )


# ----------------------------------------------------------------------

# Log

# ----------------------------------------------------------------------


class Log(Function):


    @staticmethod

    def forward(

        ctx: Context,

        t1: Tensor,

    ) -> Tensor:


        ctx.save_for_backward(t1)


        return t1.f.log_map(t1)


    @staticmethod

    def backward(

        ctx: Context,

        grad_output: Tensor,

    ) -> Tuple[Tensor]:


        (t1,) = ctx.saved_values


        return (

            grad_output.f.log_back_zip(

                t1,

                grad_output,

            ),

        )


# ----------------------------------------------------------------------

# Exp

# ----------------------------------------------------------------------


class Exp(Function):

    @staticmethod
    def forward(
        ctx: Context,
        t1: Tensor,
    ) -> Tensor:
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(
        ctx: Context,
        grad_output: Tensor,
    ) -> Tuple[Tensor]:
        (out,) = ctx.saved_values

        return (
            grad_output.f.mul_zip(
                grad_output,
                out,
            ),
        )

# ----------------------------------------------------------------------

# Comparisons

# ----------------------------------------------------------------------


class LT(Function):


    @staticmethod

    def forward(

        ctx: Context,

        a: Tensor,

        b: Tensor,

    ) -> Tensor:


        return a.f.lt_zip(a, b)


class EQ(Function):


    @staticmethod

    def forward(

        ctx: Context,

        a: Tensor,

        b: Tensor,

    ) -> Tensor:


        return a.f.eq_zip(a, b)


class IsClose(Function):


    @staticmethod

    def forward(

        ctx: Context,

        a: Tensor,

        b: Tensor,

    ) -> Tensor:


        return a.f.is_close_zip(a, b)


# ----------------------------------------------------------------------

# Permute

# ----------------------------------------------------------------------


class Permute(Function):


    @staticmethod

    def forward(

        ctx: Context,

        a: Tensor,

        order: Tensor,

    ) -> Tensor:


        ctx.save_for_backward(order)


        order_list = [

            int(order[i])

            for i in range(order.size)

        ]


        return a._new(

            a._tensor.permute(*order_list)

        )


    @staticmethod
    def backward(
        ctx: Context,
        grad_output: Tensor,
    ) -> Tuple[Tensor, float]:
        """Apply the inverse permutation to the incoming gradient."""
        (order,) = ctx.saved_values
        order_list = [int(order[i]) for i in range(order.size)]
        inverse_order = [order_list.index(i) for i in range(len(order_list))]
        return grad_output.permute(*inverse_order), 0.0

# ----------------------------------------------------------------------

# Sum

# ----------------------------------------------------------------------


class Sum(Function):


    @staticmethod

    def forward(

        ctx: Context,

        a: Tensor,

        dim: Tensor,

    ) -> Tensor:


        ctx.save_for_backward(

            a.shape,

            dim,

        )


        return a.f.add_reduce(

            a,

            int(dim.item()),

        )


    @staticmethod

    def backward(

        ctx: Context,

        grad_output: Tensor,

    ) -> Tuple[Tensor, float]:


        a_shape, dim = ctx.saved_values


        # Expand gradient back to original shape.

        return (

            grad_output.zeros(a_shape)

            + grad_output,

            0.0,

        )


# ----------------------------------------------------------------------

# All / Product reduction

# ----------------------------------------------------------------------


class All(Function):


    @staticmethod

    def forward(

        ctx: Context,

        a: Tensor,

        dim: Tensor,

    ) -> Tensor:


        if dim is not None:

            return a.f.mul_reduce(

                a,

                int(dim.item()),

            )


        return a.f.mul_reduce(

            a.contiguous().view(

                int(operators.prod(a.shape))

            ),

            0,

        )


# ----------------------------------------------------------------------

# View

# ----------------------------------------------------------------------


class View(Function):


    @staticmethod

    def forward(

        ctx: Context,

        a: Tensor,

        shape: Tensor,

    ) -> Tensor:


        ctx.save_for_backward(a.shape)


        assert a._tensor.is_contiguous(), "Must be contiguous to view"


        shape2 = [

            int(shape[i])

            for i in range(shape.size)

        ]


        return minitorch.Tensor.make(

            a._tensor._storage,

            tuple(shape2),

            backend=a.backend,

        )


    @staticmethod

    def backward(

        ctx: Context,

        grad_output: Tensor,

    ) -> Tuple[Tensor, float]:


        (original,) = ctx.saved_values


        return (

            minitorch.Tensor.make(

                grad_output.contiguous()._tensor._storage,

                original,

                backend=grad_output.backend,

            ),

            0.0,

        )


# ----------------------------------------------------------------------

# Copy

# ----------------------------------------------------------------------


class Copy(Function):


    @staticmethod

    def forward(

        ctx: Context,

        a: Tensor,

    ) -> Tensor:


        return a.f.id_map(a)


    @staticmethod

    def backward(

        ctx: Context,

        grad_output: Tensor,

    ) -> Tuple[Tensor]:


        return (grad_output,)


# ----------------------------------------------------------------------

# Matrix multiplication

# ----------------------------------------------------------------------


class MatMul(Function):


    @staticmethod

    def forward(

        ctx: Context,

        t1: Tensor,

        t2: Tensor,

    ) -> Tensor:


        ctx.save_for_backward(

            t1,

            t2,

        )


        return t1.f.matrix_multiply(

            t1,

            t2,

        )


    @staticmethod

    def backward(

        ctx: Context,

        grad_output: Tensor,

    ) -> Tuple[Tensor, Tensor]:


        t1, t2 = ctx.saved_values


        # Transpose last two dimensions.


        dims = list(range(len(t1.shape)))

        dims[-2], dims[-1] = (

            dims[-1],

            dims[-2],

        )


        t1_T = t1.permute(*dims)


        dims = list(range(len(t2.shape)))

        dims[-2], dims[-1] = (

            dims[-1],

            dims[-2],

        )


        t2_T = t2.permute(*dims)


        # dL/dA = dL/dC @ B^T

        grad_t1 = grad_output.f.matrix_multiply(

            grad_output,

            t2_T,

        )


        # dL/dB = A^T @ dL/dC

        grad_t2 = grad_output.f.matrix_multiply(

            t1_T,

            grad_output,

        )


        # Handle batch broadcasting.

        if len(t1.shape) > 2:

            if t1.shape[0] == 1 and grad_t1.shape[0] > 1:

                grad_t1 = grad_t1.sum(0)


        if len(t2.shape) > 2:

            if t2.shape[0] == 1 and grad_t2.shape[0] > 1:

                grad_t2 = grad_t2.sum(0)


        return grad_t1, grad_t2


# ----------------------------------------------------------------------

# Tensor constructors

# ----------------------------------------------------------------------


def zeros(

    shape: UserShape,

    backend: TensorBackend = SimpleBackend,

) -> Tensor:

    """Produce a zero tensor of size shape."""


    return minitorch.Tensor.make(

        [0] * int(operators.prod(shape)),

        shape,

        backend=backend,

    )


def rand(

    shape: UserShape,

    backend: TensorBackend = SimpleBackend,

    requires_grad: bool = False,

) -> Tensor:

    """Produce a random tensor of size shape."""


    vals = [

        random.random()

        for _ in range(int(operators.prod(shape)))

    ]


    tensor = minitorch.Tensor.make(

        vals,

        shape,

        backend=backend,

    )


    tensor.requires_grad_(requires_grad)


    return tensor


def _tensor(

    ls: Any,

    shape: UserShape,

    backend: TensorBackend = SimpleBackend,

    requires_grad: bool = False,

) -> Tensor:

    """Produce a tensor with data ls and shape shape."""


    tensor = minitorch.Tensor.make(

        ls,

        shape,

        backend=backend,

    )


    tensor.requires_grad_(requires_grad)


    return tensor


def tensor(

    ls: Any,

    backend: TensorBackend = SimpleBackend,

    requires_grad: bool = False,

) -> Tensor:

    """Produce a tensor with data and shape inferred from ls."""


    def shape(ls: Any) -> List[int]:


        if isinstance(ls, (list, tuple)):

            return [len(ls)] + shape(ls[0])


        return []


    def flatten(ls: Any) -> List[float]:


        if isinstance(ls, (list, tuple)):

            return [

                y

                for x in ls

                for y in flatten(x)

            ]


        return [ls]


    cur = flatten(ls)

    shape2 = shape(ls)


    return _tensor(

        cur,

        tuple(shape2),

        backend=backend,

        requires_grad=requires_grad,

    )


# ----------------------------------------------------------------------

# Gradient checking

# ----------------------------------------------------------------------


def grad_central_difference(

    f: Any,

    *vals: Tensor,

    arg: int = 0,

    epsilon: float = 1e-6,

    ind: UserIndex,

) -> float:


    x = vals[arg]


    up = zeros(x.shape)

    up[ind] = epsilon


    vals1 = [

        x if j != arg else x + up

        for j, x in enumerate(vals)

    ]


    vals2 = [

        x if j != arg else x - up

        for j, x in enumerate(vals)

    ]


    delta: Tensor = (

        f(*vals1).sum()

        - f(*vals2).sum()

    )


    return delta[0] / (2.0 * epsilon)


def grad_check(

    f: Any,

    *vals: Tensor,

) -> None:


    for x in vals:

        x.requires_grad_(True)

        x.zero_grad_()


    random.seed(10)


    out = f(*vals)


    out.sum().backward()


    err_msg = """

    Gradient check error for function %s.


    Input %s


    Received derivative %f for argument %d and index %s,

    but was expecting derivative %f from central difference.

    """


    for i, x in enumerate(vals):


        ind = x._tensor.sample()


        check = grad_central_difference(

            f,

            *vals,

            arg=i,

            ind=ind,

        )


        assert x.grad is not None


        np.testing.assert_allclose(

            x.grad[ind],

            check,

            1e-2,

            1e-2,

            err_msg=err_msg % (

                f,

                vals,

                x.grad[ind],

                i,

                ind,

                check,

            ),

        )



@staticmethod
def backward(ctx: Context, grad_output: Tensor):
    (order,) = ctx.saved_values

    order_list = [
        int(order[i])
        for i in range(order.size)
    ]

    inverse_order = [
        order_list.index(i)
        for i in range(len(order_list))
    ]

    return (grad_output.permute(*inverse_order), 0.0)