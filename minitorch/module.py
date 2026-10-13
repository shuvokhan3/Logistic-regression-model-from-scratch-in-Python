##The module system provides a structured way to represent a model and organize its trainable parameters and sub-models.

from typing import Dict, List, Tuple, Any, Sequence

class Parameter: ##it represent trainable value
    """A trainable parameter in a neural network"""
    def __init__(self, value):
        self.value = value

    @property
    def shape(self):
        """return parameter shape if avaiable"""
        if hasattr(self.value, "shape"):
            return self.value.shape
        return ()

    def __repr__(self):
        return f"Parameter(value={self.value})"

    def update(self, value:Any) -> None:
        """Update the parameter value"""
        self.value = value;


    


class Module: ## it represents a model/layer
    """base class for all neural network modules"""

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence["Module"]:
        """return all sub-modules recurisvley (depth-first)"""
        results = []
        for module in self._modules.values():
            results.append(module)
            results.extend(module.modules())
        return results

    def train(self):
        """set training mode for this module and all sub-modules"""
        self.training = True
        for module in self.modules():
            module.training = True

    def eval(self):
        """Set evaluation mode for this module and all sub-modules"""
        self.training = False
        for module in self.modules():
            module.training = False

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """return all parameters with their full paths"""
        results = []

        #Add direct parameters
        for name, param in self._parameters.items():
            results.append((name, param))

        #Add parametrs from sub-modules with prefixed names
        for module_name, module in self._modules.items():
            for param_name, param in module.named_parameters():
                full_name = f"{module_name}.{param_name}"
                results.append((full_name,param))

        return results

    def parameters(self) -> Sequence[Parameter]:
        """return all parameters in the module tree"""
        return [param for _, param in self.named_parameters()]

    def add_parameter(self, name:str, value:Any) ->Parameter:
        """Add a parameter to this module"""
        if isinstance(value, Parameter):
            param = value
        else:
            param = Parameter(value)

        self._parameters[name] = param
        return param

    def __setattr__(self, key: str, value: Any):
        """Automatically register modules and parameters."""
        if isinstance(value, Parameter):
            self._parameters[key] = value

        elif isinstance(value, Module):
            self._modules[key] = value

        super().__setattr__(key, value)


    def zero_grad_(self):
        for parameter in self.parameters():
            parameter.value.grad = None




