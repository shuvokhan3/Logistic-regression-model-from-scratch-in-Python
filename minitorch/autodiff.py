"""
Automatic differentiation utilities for MiniTorch.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple


def central_difference(
    f: Callable[..., float],
    *vals: float,
    arg: int = 0,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute the numerical derivative of f with respect to one argument
    using the central difference method.
    """
    vals_list = list(vals)

    vals_plus = vals_list.copy()
    vals_plus[arg] += epsilon

    vals_minus = vals_list.copy()
    vals_minus[arg] -= epsilon

    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    return (f_plus - f_minus) / (2.0 * epsilon)


@dataclass
class Variable:
    """
    Base class for values that participate in automatic differentiation.
    """

    history: Optional["History"] = None
    derivative: Optional[Any] = None
    name: Optional[str] = None
    requires_grad: bool = False

    def is_leaf(self) -> bool:
        """Return True if this variable was not created by an operation."""
        return self.history is None

    def is_constant(self) -> bool:
        """Return True if this variable does not require gradients."""
        return not self.requires_grad

    def requires_grad_(self, requires_grad: bool = True) -> "Variable":
        """Set whether this variable requires gradients."""
        self.requires_grad = requires_grad
        return self


@dataclass
class History:
    """
    Record the operation that created a variable and its input variables.
    """

    last_fn: Optional[type] = None
    ctx: Optional["Context"] = None
    inputs: Sequence[Variable] = ()


def topological_sort(variable: Variable) -> List[Variable]:
    """
    Return variables in topological order.

    Every input variable appears before the variable that depends on it.
    """
    order: List[Variable] = []
    visited: Set[int] = set()

    def visit(var: Variable) -> None:
        var_id = id(var)

        if var_id in visited:
            return

        visited.add(var_id)

        history = var.history

        if history is not None:
            for input_var in history.inputs:
                visit(input_var)

        order.append(var)

    visit(variable)
    return order


def _add_gradients(existing_grad: Any, new_grad: Any) -> Any:
    """
    Add two gradients.

    This supports both ordinary numeric gradients and tensor gradients.
    """
    if existing_grad is None:
        return new_grad

    return existing_grad + new_grad


def backpropagate(variable: Variable, deriv: Any = 1.0) -> None:
    """
    Backpropagate gradients from an output variable to its inputs.

    Gradients are stored in a local dictionary during backpropagation.
    Only leaf variables receive gradients through accumulate_derivative().
    """

    # Each entry stores a variable and its accumulated gradient.
    gradients: dict[int, Tuple[Variable, Any]] = {
        id(variable): (variable, deriv)
    }

    # Process the output before its inputs.
    reverse_order = reversed(topological_sort(variable))

    for var in reverse_order:
        entry = gradients.get(id(var))

        if entry is None:
            continue

        _, grad = entry

        # Leaf variables store their final accumulated gradients.
        if var.is_leaf():
            if var.requires_grad:
                var.accumulate_derivative(grad)
            continue

        history = var.history

        if history is None or history.last_fn is None:
            continue

        # Use the variable's chain_rule when available.
        chain_rule = getattr(var, "chain_rule", None)

        if callable(chain_rule):
            input_grads = chain_rule(grad)
        else:
            # Fallback for variables without chain_rule().
            backward_grads = history.last_fn.backward(
                history.ctx, grad
            )
            input_grads = zip(history.inputs, backward_grads)

        # Propagate and accumulate gradients for each input.
        for input_var, input_grad in input_grads:
            if input_grad is None:
                continue

            if not input_var.requires_grad:
                continue

            input_id = id(input_var)
            existing = gradients.get(input_id)

            if existing is None:
                gradients[input_id] = (input_var, input_grad)
            else:
                _, existing_grad = existing
                gradients[input_id] = (
                    input_var,
                    existing_grad + input_grad,
                )



@dataclass
class Context:
    """
    Store values needed during an operation's backward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Save values for use during backpropagation."""
        if self.no_grad:
            return

        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return values saved for the backward pass."""
        return self.saved_values