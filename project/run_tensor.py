
import minitorch
from minitorch import tensor, Tensor

from minitorch.fast_ops import FastOps
from minitorch.cuda_ops import CudaOps

cpu_backend = minitorch.TensorBackend(FastOps)
gpu_backend = minitorch.TensorBackend(CudaOps)



# --------------------------------------------------
# Parameter initialization
# --------------------------------------------------

def small_weights(shape):
    """
    Initialize weights in the range [-0.1, 0.1].
    Uses the existing MiniTorch tensor storage.
    """
    w = minitorch.rand(shape, requires_grad=True)

    w._tensor._storage[:] = (
        w._tensor._storage - 0.5
    ) * 0.2

    return w


def zero_bias(shape):
    """Initialize biases to zero."""
    b = minitorch.rand(shape, requires_grad=True)
    b._tensor._storage[:] = 0.0

    return b


# --------------------------------------------------
# Neural network: 2 -> hidden -> hidden -> 1
# --------------------------------------------------

class TensorNetwork(minitorch.Module):

    def __init__(self, hidden_size: int = 10):
        super().__init__()

        # Layer 1: input (2) -> hidden
        self.layer1_weights = self.add_parameter(
            "layer1_weights",
            small_weights((2, hidden_size)),
        )

        self.layer1_bias = self.add_parameter(
            "layer1_bias",
            zero_bias((hidden_size,)),
        )

        # Layer 2: hidden -> hidden
        self.layer2_weights = self.add_parameter(
            "layer2_weights",
            small_weights((hidden_size, hidden_size)),
        )

        self.layer2_bias = self.add_parameter(
            "layer2_bias",
            zero_bias((hidden_size,)),
        )

        # Layer 3: hidden -> output (1)
        self.layer3_weights = self.add_parameter(
            "layer3_weights",
            small_weights((hidden_size, 1)),
        )

        self.layer3_bias = self.add_parameter(
            "layer3_bias",
            zero_bias((1,)),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: (batch_size, 2)

        # Layer 1: Linear + ReLU
        h1 = (
            x @ self.layer1_weights.value
            + self.layer1_bias.value
        ).relu()

        # Layer 2: Linear + ReLU
        h2 = (
            h1 @ self.layer2_weights.value
            + self.layer2_bias.value
        ).relu()

        # Layer 3: Linear + Sigmoid
        out = (
            h2 @ self.layer3_weights.value
            + self.layer3_bias.value
        ).sigmoid()

        return out


# --------------------------------------------------
# Training loop
# --------------------------------------------------

def train(
    dataset,
    epochs=500,
    learning_rate=0.05,
    hidden_size=10,
):
    model = TensorNetwork(hidden_size)

    # Dataset returns (points, labels)
    points, labels = dataset

    # Convert data into MiniTorch tensors
    X = tensor([
        [p[0], p[1]]
        for p in points
    ])

    y = tensor([
        [label]
        for label in labels
    ])

    for epoch in range(epochs):

        # Forward pass
        model.train()
        out = model.forward(X)

        # Mean squared error
        diff = out - y
        loss = (diff * diff).sum() / len(points)

        # Backward pass
        loss.backward()

        # Print initial predictions and gradients
        if epoch == 0:
            print("\nInitial predictions:")
            print(out.to_numpy()[:10])

            print("\nFirst epoch gradient magnitudes:")

            for i, p in enumerate(model.parameters()):
                grad = p.value.grad

                if grad is None:
                    print(f"Parameter {i}: gradient is None")
                else:
                    grad_values = grad.to_numpy()
                    print(
                        f"Parameter {i}: "
                        f"max_abs_grad="
                        f"{abs(grad_values).max():.8e}"
                    )

        # Gradient descent update
        for p in model.parameters():
            if p.value.grad is not None:
                p.update(
                    p.value - learning_rate * p.value.grad
                )

        # Clear gradients before the next iteration
        model.zero_grad_()

        # Report training progress
        if epoch % 50 == 0:
            print(
                f"Epoch {epoch}, "
                f"Loss: {loss.item():.8f}"
            )

    return model


# --------------------------------------------------
# Run training
# --------------------------------------------------

if __name__ == "__main__":
    from minitorch.datasets import simple

    dataset = simple(100)

    print("Starting neural network training...")

    model = train(
        dataset,
        epochs=500,
        learning_rate=0.05,
        hidden_size=10,
    )

    print("\nTraining completed!")
