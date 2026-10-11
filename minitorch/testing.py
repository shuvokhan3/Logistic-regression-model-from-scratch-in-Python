from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

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
        """
        Return the collections of operations used by the tests.

        Returns:
            one_arg:
                Unary operations.

            two_arg:
                Binary operations.

            red_arg:
                Reduction operations.
        """

        # Import lazily to avoid circular imports while minitorch
        # itself is being initialized.
        import minitorch

        Tensor = minitorch.Tensor

        # ----------------------------------------------------------
        # One argument operations
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Two argument operations
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Reduction operations
        # ----------------------------------------------------------

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
    Check Tensor gradients using the central-difference method.

    This compares the gradient produced by automatic differentiation
    against a numerical gradient.
    """

    import minitorch

    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()

    # Forward pass.
    random_out = f(*vals)

    # Reduce the output to a scalar and backpropagate.
    random_out.sum().backward()

    epsilon = 1e-6

    for arg, x in enumerate(vals):

        # Pick one random element of the tensor.
        ind = x._tensor.sample()

        # Create perturbation tensor.
        up = minitorch.zeros(x.shape)

        up[ind] = epsilon

        # f(x + epsilon)
        vals1 = [
            v if i != arg else v + up
            for i, v in enumerate(vals)
        ]

        # f(x - epsilon)
        vals2 = [
            v if i != arg else v - up
            for i, v in enumerate(vals)
        ]

        # Central difference:
        #
        # f(x + eps) - f(x - eps)
        # -----------------------
        #          2 eps
        #
        delta = (
            f(*vals1).sum()
            - f(*vals2).sum()
        )

        numerical = delta[0] / (2.0 * epsilon)

        assert x.grad is not None

        np.testing.assert_allclose(
            x.grad[ind],
            numerical,
            rtol=1e-2,
            atol=1e-2,
            err_msg=(
                f"Gradient check failed for argument {arg} "
                f"at index {ind}. "
                f"Received {x.grad[ind]}, "
                f"expected {numerical}."
            ),
        )