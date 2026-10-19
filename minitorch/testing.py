from __future__ import annotations

from typing import Callable

import numpy as np


class MathTestVariable:
    """
    Collection of mathematical operations used by the MiniTorch tests.

    Each operation contains:
        - a name
        - a Python/scalar implementation
        - the corresponding Tensor implementation
    """

    @staticmethod
    def _comp_testing():
        """Return the collections of mathematical operations."""

        import minitorch

        one_arg = [
            (
                "neg",
                lambda x: -x,
                lambda x: x.neg(),
            ),
            (
                "sigmoid",
                lambda x: 1.0 / (1.0 + np.exp(-x)),
                lambda x: x.sigmoid(),
            ),
            (
                "relu",
                lambda x: max(0.0, x),
                lambda x: x.relu(),
            ),
            (
                "exp",
                lambda x: np.exp(x),
                lambda x: x.exp(),
            ),
            (
                "log",
                lambda x: np.log(x),
                lambda x: x.log(),
            ),
        ]

        two_arg = [
            (
                "add",
                lambda x, y: x + y,
                lambda x, y: x + y,
            ),
            (
                "mul",
                lambda x, y: x * y,
                lambda x, y: x * y,
            ),
        ]

        red_arg = [
            (
                "sum",
                lambda xs: sum(xs),
                lambda x: x.sum(),
            ),
        ]

        return one_arg, two_arg, red_arg


def grad_check(
    f: Callable,
    *vals,
) -> None:
    """
    Compare automatic differentiation with a numerical
    central-difference approximation.
    """

    import minitorch

    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()

    # Forward pass and automatic differentiation.
    output = f(*vals)
    output.sum().backward()

    epsilon = 1e-6

    for arg, x in enumerate(vals):
        ind = x._tensor.sample()

        # Perturb one element.
        up = minitorch.zeros(x.shape)
        up[ind] = epsilon

        vals_plus = [
            v if i != arg else v + up
            for i, v in enumerate(vals)
        ]

        vals_minus = [
            v if i != arg else v - up
            for i, v in enumerate(vals)
        ]

        # Numerical central difference.
        output_plus = f(*vals_plus).sum()
        output_minus = f(*vals_minus).sum()

        numerical = (
            output_plus[0] - output_minus[0]
        ) / (2.0 * epsilon)

        assert x.grad is not None

        analytical = x.grad[ind]

        # Diagnostic information if the gradients disagree.
        if not np.isclose(
            analytical,
            numerical,
            rtol=1e-2,
            atol=1e-2,
        ):
            print(
                f"\nGradient check diagnostic:"
                f"\n  Function: {getattr(f, '__name__', f)}"
                f"\n  Input value: {x[ind]:.17g}"
                f"\n  Analytical gradient: {analytical}"
                f"\n  Numerical gradient: {numerical}"
                f"\n  Index: {ind}"
            )

        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=1e-2,
            atol=1e-2,
            err_msg=(
                f"Gradient check failed for argument {arg} "
                f"at index {ind}. "
                f"Received derivative {analytical}, "
                f"expected {numerical}."
            ),
        )




def sum_practice(a):
    """Wrapper around CUDA sum_practice that returns the inner TensorData."""
    from .cuda_ops import sum_practice as _cuda_sum_practice
    return _cuda_sum_practice(a)._tensor


def mm_practice(a, b):
    """Wrapper around matrix_multiply that returns the inner TensorData."""
    return a.f.matrix_multiply(a, b)._tensor
