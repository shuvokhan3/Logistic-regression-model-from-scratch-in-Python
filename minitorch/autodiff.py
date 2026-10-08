"""
Automatic differentiation utilities for MiniTorch.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence ,List, Set, Tuple, Any


def central_difference(
    f: Callable[..., float],
    *vals: float,
    arg: int = 0,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute the numerical derivative of f with respect to argument `arg`.

    Uses the central difference formula:

        (f(x + h) - f(x - h)) / (2h)

    Args:
        f: Function to differentiate.
        *vals: Input values to f.
        arg: Which argument to differentiate with respect to (0-indexed).
        epsilon: Step size for numerical differentiation.

    Returns:
        Approximate derivative.

    Example:
        >>> def mul(x, y):
        ...     return x * y
        >>> central_difference(mul, 3, 4, arg=0)
        4.0
        >>> central_difference(mul, 3, 4, arg=1)
        3.0
    """

    vals_list = list(vals)

    # Create values with the selected argument increased by epsilon.
    vals_plus = vals_list.copy()
    vals_plus[arg] += epsilon

    # Create values with the selected argument decreased by epsilon.
    vals_minus = vals_list.copy()
    vals_minus[arg] -= epsilon

    # Compute function values.
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    # Central difference.
    return (f_plus - f_minus) / (2 * epsilon)


@dataclass
class Variable:
    """
    A node in the computation graph.

    Attributes:
        history:
            Record of the operation that created this variable.

        derivative:
            Accumulated gradient during the backward pass.

        name:
            Optional name used for debugging.

        requires_grad:
            Whether this variable should receive gradients.
    """

    history: Optional["History"] = None
    derivative: Optional[float] = None
    name: Optional[str] = None
    requires_grad: bool = False

    def is_leaf(self) -> bool:
        """
        Return True if this variable was not created by an operation.
        """
        return self.history is None

    def is_constant(self) -> bool:
        """
        Return True if this variable does not require gradients.
        """
        return not self.requires_grad

    def requires_grad_(self, requires_grad: bool = True) -> "Variable":
        """
        Set whether this variable should track gradients.

        Args:
            requires_grad:
                Whether gradients should be tracked.

        Returns:
            This variable.
        """
        self.requires_grad = requires_grad
        return self


@dataclass
class History:
    """
    Record the operation that created a variable.

    Attributes:
        last_fn:
            The function class that created this variable.

        ctx:
            Context object storing values needed during backward.

        inputs:
            Input variables to the operation.
    """

    last_fn: Optional[type] = None
    ctx: Optional["Context"] = None
    inputs: Sequence[Variable] = ()




def topological_sort(variable: Variable) -> List[Variable]:

    """
    return variable in topological order (childeren before parents)

    For backpropagation, a variable must be processed AFTER all variables that depend on it have been processed. 


    Args: varible : The output variable (e.g , loss)

    returns: list of variable in topological sort

    """

    order: List [Variable] = []
    visited: Set[int] = set()

    def visit(var: Variable) -> None:
        #Use id() to handle variables that might compare equal
        var_id = id(var)

        if var_id in visited:
            return
        visited.add(var_id)

        #Visit children first (variables that one depentds on )
        if var.history is not None and var.history.inputs:
            for input_var in var.history.inputs:
                visit(input_var)

        #Add this variable After its children 
        order.append(var)

    visit(variable)
    return order 


def backpropagate(variable: Variable, deriv: float = 1.0) -> None:
    """
    Run backpropagation starting from variable.

    Computes gradients for all variables in the computation graph
    that require gradients.
    """

    # Get variables in topological order.
    sorted_vars = topological_sort(variable)

    # Process output first, then move toward the leaves.
    sorted_vars.reverse()

    # Gradient of output with respect to itself.
    variable.derivative = deriv

    for var in sorted_vars:

        # Nothing to propagate if this variable has no derivative.
        if var.derivative is None:
            continue

        # Leaf variables have no function/history to propagate through.
        if var.is_leaf():
            continue

        history = var.history

        if history is None or history.last_fn is None:
            continue

        # Compute gradients with respect to inputs.
        input_grads = history.last_fn.backward(
            history.ctx,
            var.derivative,
        )

        # Pass gradients to inputs.
        for input_var, grad in zip(history.inputs, input_grads):

            if grad is None:
                continue

            if input_var.requires_grad:
                input_var.accumulate_derivative(grad)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values





        
    