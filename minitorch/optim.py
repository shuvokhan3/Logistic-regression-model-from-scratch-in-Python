
from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = list(parameters)


class SGD(Optimizer):
    def __init__(
        self,
        parameters: Sequence[Parameter],
        lr: float = 1.0,
    ):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue

            # Scalar autodifferentiation
            if hasattr(p.value, "derivative"):
                p.value.derivative = None

            # Tensor autodifferentiation
            if hasattr(p.value, "grad"):
                p.value.grad = None

    def step(self) -> None:
        for p in self.parameters:
            value = p.value

            if value is None:
                continue

            # Case 1: Scalar parameter
            if isinstance(value, Scalar):
                if value.derivative is not None:
                    new_value = Scalar(
                        value.data - self.lr * value.derivative
                    )
                    p.update(new_value)

            # Case 2: Tensor parameter
            elif hasattr(value, "grad"):
                grad = value.grad

                if grad is not None:
                    new_value = (
                        value - self.lr * grad
                    ).detach()

                    new_value.requires_grad_(True)
                    p.update(new_value)
