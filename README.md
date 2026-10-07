# MiniTorch — Automatic Differentiation from Scratch

> *"Don't just use `loss.backward()` — understand what it does."*

MiniTorch is a tiny PyTorch-like framework built **entirely from scratch** in pure Python.
No NumPy, no PyTorch, no JAX — just `math`, `dataclasses`, and first principles.

The goal is to demystify **automatic differentiation** (autograd) by implementing every piece yourself: scalar values that track their own computation history, differentiable functions with hand-derived gradients, topological sorting of the computation graph, and backpropagation via the chain rule.

By the end you will have a working system where you can write:

```python
a = Scalar(2.0);  a.requires_grad_(True)
b = Scalar(3.0);  b.requires_grad_(True)

loss = (a * b + a).sigmoid().log()
loss.backward()

print(a.derivative)   # ∂loss/∂a  ✓
print(b.derivative)   # ∂loss/∂b  ✓
```

…and **every gradient is computed automatically**, exactly like PyTorch.

---

## 📑 Table of Contents

1. [Why Build This?](#-why-build-this)
2. [Architecture Overview](#-architecture-overview)
3. [Project Structure](#-project-structure)
4. [Module-by-Module Deep Dive](#-module-by-module-deep-dive)
   - [operators.py — Mathematical Foundations](#1-operatorspy--mathematical-foundations)
   - [module.py — Neural Network Module System](#2-modulepy--neural-network-module-system)
   - [autodiff.py — The Autograd Engine](#3-autodiffpy--the-autograd-engine)
   - [scalar.py — Differentiable Scalar Values](#4-scalarpy--differentiable-scalar-values)
   - [scalar_functions.py — Forward & Backward Rules](#5-scalar_functionspy--forward--backward-rules)
   - [datasets.py — Toy Classification Datasets](#6-datasetspy--toy-classification-datasets)
   - [training_example.py — End-to-End Training](#7-training_examplepy--end-to-end-training)
5. [How Automatic Differentiation Works](#-how-automatic-differentiation-works)
6. [Step-by-Step Reimplementation Guide](#-step-by-step-reimplementation-guide)
7. [Getting Started](#-getting-started)
8. [Running Tests](#-running-tests)
10. [Example Output](#-example-output)
11. [Mathematical Reference](#-mathematical-reference)
12. [What's Next (v2 Roadmap)](#-whats-next-v2-roadmap)
13. [Resources & Inspiration](#-resources--inspiration)
14. [License](#-license)

---

## Why Build This?

Most ML engineers call `loss.backward()` hundreds of times a day but never look inside.
Building MiniTorch answers the questions that interviews, debugging sessions, and research papers assume you already know:

| Question | Where You'll Find the Answer |
|----------|------------------------------|
| What *is* a computation graph? | `autodiff.py` — `Variable`, `History` |
| How does `requires_grad` work? | `scalar.py` — gradient tracking flag |
| What does `.backward()` actually do? | `autodiff.py` — `backpropagate()` |
| Why do we need topological sort? | `autodiff.py` — `topological_sort()` |
| How are gradients accumulated? | `scalar.py` — `accumulate_derivative()` |
| What is the chain rule in code? | `scalar_functions.py` — every `backward()` method |
| How does SGD update parameters? | `training_example.py` — the training loop |

---

##  Architecture Overview

```
                 MiniTorch Autograd
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
      scalar.py   scalar_functions.py autodiff.py
          │             │             │
          │             │             │
       "What is       "What is       "How do
        a Scalar?"     the math?"      gradients
                                      flow?"
          │             │             │
          ▼             ▼             ▼
       Objects       Local rules    Chain rule
```

**Data flows forward** through `ScalarFunction.apply()` which builds a computation graph.
**Gradients flow backward** through `backpropagate()` which walks the graph in reverse topological order.

### Dependency Graph

```
operators.py          ← Pure math, no dependencies
    ↑
module.py             ← Parameter & Module containers
    ↑
autodiff.py           ← Variable, History, topological_sort, backpropagate
    ↑
scalar.py             ← Scalar (extends Variable), Context
    ↑
scalar_functions.py   ← ScalarFunction subclasses (Add, Mul, Sigmoid, …)
    ↑
datasets.py           ← Toy data generators
    ↑
training_example.py   ← Puts it all together
```

---

## Project Structure

```
MiniTorch/
│
├── minitorch/                    # Core library
│   ├── __init__.py               # Package exports + quick demo code
│   ├── operators.py              # Pure math operators + higher-order functions
│   ├── module.py                 # Parameter & Module (like torch.nn.Module)
│   ├── autodiff.py               # Variable, History, topological_sort, backpropagate
│   ├── scalar.py                 # Scalar class + Context
│   ├── scalar_functions.py       # Differentiable ops: Add, Mul, Sigmoid, ReLU, …
│   ├── datasets.py               # Simple, Diagonal, Split, XOR dataset generators
│   └── testing.py                # assert_close utility
│
├── tests/                        # Test suite
│   ├── conftest.py               # Pytest markers (task0_x, task1_x)
│   ├── test_operators.py         # Unit + property tests for operators
│   ├── test_module.py            # Tests for Parameter & Module
│   ├── test_autodiff.py          # Tests for central_difference, topo sort, backprop
│   └── test_scalar.py            # Tests for Scalar creation, arithmetic, backward
│
├── images/                       # Generated visualizations (for README)
│   ├── dataset_overview.png      # Raw dataset with ground truth boundary
│   ├── before_after_training.png # Side-by-side before/after comparison
│   ├── training_progress.png     # Loss & accuracy curves
│   └── sigmoid_evolution.png     # How the sigmoid transforms during training
│
├── training_example.py           # Logistic regression training script
├── generate_visualizations.py    # Script to regenerate all plots
├── .gitignore
└── README.md                     # ← You are here
```

---

## 🔬 Module-by-Module Deep Dive

### 1. `operators.py` — Mathematical Foundations

> **Purpose:** Define pure, side-effect-free math functions that every other module builds on.

This is your **mathematical building-block layer**. Every function takes plain `float` values and returns plain `float` values — no autograd, no classes, just math.

#### Basic Arithmetic

| Function | Signature | What It Computes |
|----------|-----------|-----------------|
| `mul(x, y)` | `float × float → float` | `x × y` |
| `add(x, y)` | `float × float → float` | `x + y` |
| `neg(x)` | `float → float` | `−x` |
| `id(x)` | `float → float` | `x` (identity) |
| `lt(x, y)` | `float × float → float` | `1.0` if `x < y`, else `0.0` |
| `eq(x, y)` | `float × float → float` | `1.0` if `x == y`, else `0.0` |
| `max(x, y)` | `float × float → float` | larger of `x`, `y` |
| `is_close(x, y)` | `float × float → float` | `1.0` if `\|x − y\| < 0.01` |

#### Activation & Transcendental Functions

| Function | Formula | Notes |
|----------|---------|-------|
| `sigmoid(x)` | `1 / (1 + e⁻ˣ)` | Numerically stable — uses `e^x / (1 + e^x)` for `x < 0`. Clamps output away from exact 0 and 1. |
| `relu(x)` | `max(0, x)` | |
| `log(x)` | `ln(x)` | |
| `exp(x)` | `eˣ` | |
| `inv(x)` | `1/x` | |

#### Backward Helpers

These compute `∂f/∂x · grad` for use during backpropagation:

| Function | Derivative Rule |
|----------|----------------|
| `log_back(x, grad)` | `grad / x` — because `d(ln x)/dx = 1/x` |
| `inv_back(x, grad)` | `−grad / x²` — because `d(1/x)/dx = −1/x²` |
| `relu_back(x, grad)` | `grad` if `x > 0` else `0.0` |

#### Higher-Order Functions

These are functional programming primitives that operate on lists:

```python
map(fn)         # Returns a function that applies fn to every element
zipWith(fn)     # Returns a function that combines two lists element-wise
reduce(fn, init) # Returns a function that folds a list to a single value
```

**Composed utilities** built from the above:

```python
negList(ls)        →  map(neg)(ls)          # Negate every element
addLists(ls1, ls2) →  zipWith(add)(ls1,ls2) # Element-wise addition
sum(ls)            →  reduce(add, 0.0)(ls)  # Sum all elements
prod(ls)           →  reduce(mul, 1.0)(ls)  # Product of all elements
```

---

### 2. `module.py` — Neural Network Module System

> **Purpose:** Provide `torch.nn.Module`-like infrastructure for organizing parameters and sub-models.

#### `Parameter`

A wrapper around any value that marks it as **trainable**:

```python
p = Parameter(5.0)
p.value          # 5.0
p.update(10.0)   # Change the value
p.shape          # () for scalars, or .shape if the value has one
```

#### `Module`

Base class for all "layers" or "models". Key features:

| Method | What It Does |
|--------|-------------|
| `__setattr__` | **Auto-registers** child `Module`s and `Parameter`s when you assign them as attributes |
| `modules()` | Returns all sub-modules recursively (depth-first) |
| `named_parameters()` | Returns `[(name, param), ...]` with dotted paths like `"child.weight"` |
| `parameters()` | Returns a flat list of all `Parameter` objects in the tree |
| `train()` / `eval()` | Sets `self.training` flag on this module and all descendants |

**Example:**

```python
class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(0.5)   # Auto-registered!
        self.bias   = Parameter(0.0)   # Auto-registered!
```

---

### 3. `autodiff.py` — The Autograd Engine

> **Purpose:** The core differentiation machinery — computation graph representation, numerical derivatives, topological ordering, and backpropagation.

This is the **heart of MiniTorch**.

#### `central_difference(f, *vals, arg=0, epsilon=1e-6)`

Computes a **numerical approximation** of a partial derivative using the central difference formula:

```
∂f/∂xᵢ ≈ [f(x + εeᵢ) − f(x − εeᵢ)] / 2ε
```

Used for **testing** that your analytical gradients are correct:

```python
def mul(x, y): return x * y

central_difference(mul, 3.0, 4.0, arg=0)  # ≈ 4.0  (∂(xy)/∂x = y)
central_difference(mul, 3.0, 4.0, arg=1)  # ≈ 3.0  (∂(xy)/∂y = x)
```

#### `Variable` (dataclass)

A **node in the computation graph**:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `history` | `History \| None` | How this variable was created |
| `derivative` | `float \| None` | Accumulated gradient |
| `name` | `str \| None` | For debugging |
| `requires_grad` | `bool` | Should this node receive gradients? |

Key methods:
- `is_leaf()` → `True` if no history (user-created, not from an operation)
- `is_constant()` → `True` if `requires_grad` is `False`
- `requires_grad_(flag)` → Set the gradient tracking flag

#### `History` (dataclass)

Records **what operation** created a `Variable`:

| Field | Meaning |
|-------|---------|
| `last_fn` | The `ScalarFunction` class (e.g., `Mul`, `Add`) |
| `ctx` | `Context` object with saved values for backward |
| `inputs` | The input `Variable`s to the operation |

#### `topological_sort(variable) → List[Variable]`

Returns all variables in the computation graph in **topological order** (children before parents).

**Why is this needed?** During backpropagation, a variable must be processed **after** all variables that depend on it. Topological order guarantees this.

**Algorithm:** Depth-first search using `id()` to handle variables that might compare equal:

```
visit(var):
    if already visited: return
    mark visited
    for each input in var.history.inputs:
        visit(input)          # Visit children first
    append(var)               # Then add this node
```

#### `backpropagate(variable, deriv=1.0)`

The **main backward pass**. Walks the computation graph in **reverse topological order** and applies the chain rule at each node:

```
1. sorted_vars = topological_sort(variable)
2. Reverse the list (process output → leaves)
3. Set variable.derivative = deriv  (∂L/∂L = 1)
4. For each var in sorted_vars:
     a. Skip if no derivative or is a leaf
     b. Call var.history.last_fn.backward(ctx, var.derivative)
        → returns gradients for each input
     c. Accumulate gradients into each input variable
```

---

### 4. `scalar.py` — Differentiable Scalar Values

> **Purpose:** A concrete `Variable` that wraps a single `float` and supports all arithmetic operators through the autograd system.

#### `ScalarHistory`

Like `History` but specialized for `Scalar`:

```python
@dataclass
class ScalarHistory:
    last_fn: type          # e.g., Mul, Add
    ctx: Context           # Saved values for backward
    inputs: Sequence[Scalar]  # Input scalars
```

#### `Scalar` (extends `Variable`)

The **user-facing** differentiable value:

```python
a = Scalar(2.0)
a.requires_grad_(True)
a.data           # 2.0
a.derivative     # None (before backward), then the gradient
a.is_leaf()      # True
a.history        # None (for leaf nodes)
```

**Operator overloading** — every operation builds the computation graph:

| Python Syntax | Delegates To | Notes |
|---------------|-------------|-------|
| `a + b` | `Add.apply(a, b)` | |
| `a * b` | `Mul.apply(a, b)` | |
| `-a` | `Neg.apply(a)` | |
| `a - b` | `Add.apply(a, Neg.apply(b))` | Subtraction = add + negate |
| `a / b` | `Mul.apply(a, Inv.apply(b))` | Division = multiply by inverse |
| `a.log()` | `Log.apply(a)` | |
| `a.exp()` | `Exp.apply(a)` | |
| `a.sigmoid()` | `Sigmoid.apply(a)` | |
| `a.relu()` | `ReLU.apply(a)` | |

**Gradient methods:**

```python
a.backward(deriv=1.0)          # Run backpropagation from this scalar
a.accumulate_derivative(grad)  # Add gradient (handles None → init)
a.zero_grad_()                 # Reset gradient to None
```

#### `Context`

A simple container to **save values during forward** that are needed during backward:

```python
ctx = Context()
ctx.save_for_backward(x, y)    # Store x, y
x, y = ctx.saved_values        # Retrieve during backward
```

---

### 5. `scalar_functions.py` — Forward & Backward Rules

> **Purpose:** Define every differentiable operation as a `ScalarFunction` subclass with an explicit `forward` and `backward`.

#### `ScalarFunction` Base Class

The `apply()` classmethod is the **bridge between Python operators and the autograd system**:

```
apply(raw_vals):
    1. Convert any plain floats to Scalar(float_val)
    2. Extract .data from each Scalar
    3. Create a Context
    4. Call cls.forward(ctx, *raw_data)  → result float
    5. If any input requires_grad:
         Record history (last_fn, ctx, inputs)
    6. Return Scalar(result, history=..., requires_grad=...)
```

#### Implemented Operations

Every subclass implements `forward` and `backward`:

| Class | Forward: `z = ?` | Backward: `∂z/∂inputs` | Saved for Backward |
|-------|-----------|-------------------------|-------------------|
| **Add** | `x + y` | `(d_out, d_out)` | Nothing |
| **Mul** | `x × y` | `(d_out × y, d_out × x)` | `x, y` |
| **Neg** | `−x` | `(−d_out,)` | Nothing |
| **Inv** | `1/x` | `(−d_out / x²,)` | `x` |
| **Log** | `ln(x)` | `(d_out / x,)` | `x` |
| **Exp** | `eˣ` | `(d_out × eˣ,)` | `result` (= eˣ) |
| **Square** | `x²` | `(d_out × 2x,)` | `x` |
| **Sigmoid** | `σ(x)` | `(d_out × σ × (1−σ),)` | `result` (= σ(x)) |
| **ReLU** | `max(0,x)` | `(d_out if x>0 else 0,)` | `x` |

> **Key insight:** Each `backward` returns a **tuple** — one gradient per input. This is the local application of the chain rule.

---

### 6. `datasets.py` — Toy Classification Datasets

> **Purpose:** Generate 2D point-cloud datasets for binary classification experiments.

All generators return `(points, labels)` where points are `(x, y)` tuples in `[0, 1]²`.

| Dataset | Decision Boundary | Difficulty |
|---------|-------------------|------------|
| **`simple(N)`** | `x ≥ 0.5` → class 1 | Easy — vertical line |
| **`diag(N)`** | `x + y ≥ 1.0` → class 1 | Easy — diagonal line |
| **`split(N)`** | `0.2 ≤ x ≤ 0.8` → class 0 | Medium — two vertical lines |
| **`xor(N)`** | Same quadrant → class 0 | Hard — requires non-linearity |

The `Dataview` wrapper provides a `.X`, `.y`, `.N` interface for convenience.

---

### 7. `training_example.py` — End-to-End Training

> **Purpose:** Prove that MiniTorch works by training a logistic regression model.

#### Model

```python
class SimpleModel:
    # y_hat = sigmoid(w * x + b)
    def forward(self, x):
        return (self.w * x + self.b).sigmoid()
```

Two trainable parameters: `w` (weight) and `b` (bias).

#### Loss Function

Binary Cross-Entropy (BCE):

```
L = −[y · log(p) + (1−y) · log(1−p)]
```

#### Training Loop

```
For each epoch:
    For each sample (x, label):
        1. Zero gradients           ← model.zero_grad()
        2. Forward pass             ← pred = model.forward(x)
        3. Compute loss             ← loss = BCE(pred, label)
        4. Backward pass            ← loss.backward()
        5. SGD update               ← param.data -= lr * param.derivative
        6. Track accuracy
    Print statistics every 10 epochs
```

This is **identical to the PyTorch training loop** — the only difference is that every piece is written by you.

#### The Full Pipeline Visualized

```
        DATASET
           │
           ▼
    ┌─────────────┐
    │   x, y       │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   wx + b     │    ← Forward pass
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   sigmoid    │
    └──────┬──────┘
           │
           ▼
      prediction
           │
           ▼
    ┌─────────────┐
    │     BCE      │    ← Loss computation
    └──────┬──────┘
           │
           ▼
         loss
           │
           ▼
    loss.backward()     ← Backward pass (autograd!)
           │
    ┌──────┴──────┐
    ▼             ▼
  dL/dw         dL/db   ← Gradients
    │             │
    └──────┬──────┘
           ▼
     SGD update          ← Optimizer step
           │
           ▼
      new w, b
```

---

##  How Automatic Differentiation Works

### The Big Picture

1. **Forward pass** — You compute `loss = f(a, b, c, …)`. Each operation (`+`, `*`, `sigmoid`, …) creates a new `Scalar` and records *which function* produced it and *from which inputs* (`ScalarHistory`).

2. **Build the graph** — The chain of `ScalarHistory` objects forms a **Directed Acyclic Graph (DAG)** — the *computation graph*.

3. **Backward pass** — Starting from `loss`, walk the graph in **reverse topological order**. At each node, call the function's `backward()` to compute local gradients, then pass them to the inputs using the **chain rule**.

### Concrete Example

```python
a = Scalar(2.0);  a.requires_grad_(True)
b = Scalar(3.0);  b.requires_grad_(True)
c = a * b        # c = 6.0, recorded: Mul(a, b)
d = c + a        # d = 8.0, recorded: Add(c, a)
d.backward()
```

**Graph:**
```
a(2.0)──┬──→ Mul ──→ c(6.0) ──→ Add ──→ d(8.0)
b(3.0)──┘                        ↑
a(2.0)────────────────────────────┘
```

**Backward walk (reverse topological order):**

| Step | Node | `∂d/∂node` | Rule |
|------|------|-----------|------|
| 1 | `d` | `1.0` | Seed |
| 2 | `c` | `1.0` | `Add.backward` → `(1.0, 1.0)`, so `∂d/∂c = 1.0` |
| 3 | `a` | `1.0` (from Add) + `3.0` (from Mul) = **`4.0`** | Accumulated |
| 4 | `b` | `2.0` | `Mul.backward` → `(d_out × b, d_out × a)` = `(3.0, 2.0)` |

**Result:** `a.derivative = 4.0`, `b.derivative = 2.0` ✓

You can verify: `d = a*b + a = ab + a`, so `∂d/∂a = b + 1 = 3 + 1 = 4` and `∂d/∂b = a = 2`.

---

## 🗺 Step-by-Step Reimplementation Guide

Want to rebuild this yourself? Follow this exact order. Each step builds on the previous one.

### Step 0: Operators (Foundation)

**File:** `operators.py`

1. Implement basic arithmetic: `mul`, `add`, `neg`, `id`, `lt`, `eq`, `max`, `is_close`
2. Implement activations: `sigmoid` (with numerical stability!), `relu`
3. Implement transcendentals: `log`, `exp`, `inv`
4. Implement backward helpers: `log_back`, `inv_back`, `relu_back`
5. Implement higher-order functions: `map`, `zipWith`, `reduce`
6. Build composed utilities: `negList`, `addLists`, `sum`, `prod`

**Test:** `pytest tests/test_operators.py`

> **Tip:** The sigmoid must handle large negative inputs without overflow. Use `e^x / (1 + e^x)` when `x < 0`.

---

### Step 1: Module System

**File:** `module.py`

1. Create `Parameter` class — wraps a value, provides `.update()` and `.shape`
2. Create `Module` class with:
   - `_modules` dict and `_parameters` dict
   - `__setattr__` override for auto-registration
   - `modules()` — recursive depth-first traversal
   - `named_parameters()` — returns `(dotted.name, param)` pairs
   - `parameters()` — flat list of all params
   - `train()` / `eval()` — toggle `self.training` recursively

**Test:** `pytest tests/test_module.py`

---

### Step 2: Numerical Derivatives

**File:** `autodiff.py` (first part)

1. Implement `central_difference(f, *vals, arg=0, epsilon=1e-6)`
   - Perturb argument `arg` by `±ε`
   - Return `(f(x+ε) − f(x−ε)) / 2ε`

**Test:** `pytest tests/test_autodiff.py -k "central_difference"`

> **Why this matters:** You'll use `central_difference` to *verify* that your hand-coded gradients in Step 4 are correct.

---

### Step 3: Computation Graph Infrastructure

**File:** `autodiff.py` (Variable, History) + `scalar.py` (Scalar, ScalarHistory, Context)

1. Define `Variable` dataclass — `history`, `derivative`, `name`, `requires_grad`
2. Define `History` dataclass — `last_fn`, `ctx`, `inputs`
3. Define `Context` class — `save_for_backward()`, `saved_values` property
4. Define `ScalarHistory` — same as History but typed for Scalar
5. Define `Scalar(Variable)`:
   - Store `.data` as a float
   - `is_leaf()`, `requires_grad_()`, `accumulate_derivative()`, `zero_grad_()`
   - Comparison operators (`__lt__`, `__gt__`, `__eq__`)

**Test:** `pytest tests/test_scalar.py -k "creation"`

---

### Step 4: Differentiable Scalar Functions

**File:** `scalar_functions.py`

1. Build the `ScalarFunction` base class with `apply()`:
   - Convert floats → Scalars
   - Call `forward(ctx, *data)`
   - If any input needs grad → create `ScalarHistory`
   - Return new `Scalar` with history
2. Implement subclasses one at a time:
   - `Add` → easiest (gradient is just 1, 1)
   - `Mul` → save `x, y`; gradient is `(y, x)`
   - `Neg` → gradient is `−1`
   - `Inv` → gradient is `−1/x²`
   - `Log` → gradient is `1/x`
   - `Exp` → gradient is `e^x` (save the result!)
   - `Sigmoid` → gradient is `σ(1−σ)` (save the result!)
   - `ReLU` → gradient is `1` if `x > 0` else `0`
3. Wire up `Scalar` operators (`__add__`, `__mul__`, etc.) to call `.apply()`

**Test:** `pytest tests/test_scalar.py -k "arithmetic or sigmoid or relu"`

> **Gotcha:** Circular imports! Import `ScalarFunction` subclasses at the **bottom** of `scalar.py`.

---

### Step 5: Topological Sort & Backpropagation

**File:** `autodiff.py` (second part)

1. Implement `topological_sort(variable)`:
   - DFS using `id(var)` for visited tracking
   - Visit children (inputs) first, then append parent
2. Implement `backpropagate(variable, deriv=1.0)`:
   - Get topological order, then **reverse** it
   - Seed: `variable.derivative = deriv`
   - Walk through, calling `backward()` and accumulating gradients
3. Add `Scalar.backward()` method that calls `backpropagate(self)`

**Test:** `pytest tests/test_autodiff.py`

---

### Step 6: Datasets & Training

**Files:** `datasets.py`, `training_example.py`

1. Implement dataset generators (`simple`, `diag`, `split`, `xor`)
2. Build `SimpleModel` with `w` and `b` as `Scalar` parameters
3. Implement `binary_cross_entropy(pred, target)`
4. Write the training loop: zero_grad → forward → loss → backward → SGD update

**Test:** `python training_example.py`

If accuracy reaches ~95%+ on the `simple` dataset, **your autograd engine works!** 🎉

---

##  Getting Started

### Prerequisites

- **Python 3.8+** (no external dependencies for the core library)
- **pytest** and **hypothesis** (for running tests)

### Installation

```bash
# Clone the repository
git clone https://github.com/shuvokhan3/Logistic-regression-model-from-scratch-in-Python.git
cd Logistic-regression-model-from-scratch-in-Python

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install test dependencies
pip install pytest hypothesis
```

### Quick Demo

```python
from minitorch.scalar import Scalar
from minitorch.autodiff import backpropagate

a = Scalar(2.0);  a.requires_grad_(True)
b = Scalar(3.0);  b.requires_grad_(True)

f = (a * b) / Scalar(4.0)   # f = 6/4 = 1.5
backpropagate(f)

print(f"∂f/∂a = {a.derivative}")  # 3/4 = 0.75
print(f"∂f/∂b = {b.derivative}")  # 2/4 = 0.50
```

---

##  Running Tests

Tests are organized by task/milestone:

```bash
# Run all tests
pytest tests/ -v

# Run by module
pytest tests/test_operators.py -v      # Step 0: operators
pytest tests/test_module.py -v         # Step 1: module system
pytest tests/test_autodiff.py -v       # Steps 2 & 5: autodiff
pytest tests/test_scalar.py -v         # Steps 3 & 4: scalar + functions

# Run by task marker
pytest -m task1_1 -v    # Numerical derivatives
pytest -m task1_2 -v    # Scalar forward
pytest -m task1_3 -v    # Chain rule / scalar functions
pytest -m task1_4 -v    # Backpropagation

# Run training example
python training_example.py
```

The test suite includes both **unit tests** and **property-based tests** (via Hypothesis) that verify mathematical invariants like:
- `sigmoid(-x) ≈ 1 - sigmoid(x)` (symmetry)
- `log(exp(x)) ≈ x` (inverse)
- `inv(inv(x)) ≈ x` (involution)
- `exp(a + b) ≈ exp(a) × exp(b)` (homomorphism)

---

##  Example Output

```
Initial parameters: w=0.4821, b=-0.3127
Epoch   0 | Loss=0.8932 | Accuracy=49.00%
Epoch  10 | Loss=0.5641 | Accuracy=72.00%
Epoch  20 | Loss=0.3847 | Accuracy=86.00%
Epoch  30 | Loss=0.2891 | Accuracy=92.00%
Epoch  40 | Loss=0.2303 | Accuracy=95.00%
Epoch  50 | Loss=0.1912 | Accuracy=96.00%
Epoch  60 | Loss=0.1634 | Accuracy=97.00%
Epoch  70 | Loss=0.1425 | Accuracy=98.00%
Epoch  80 | Loss=0.1262 | Accuracy=98.00%
Epoch  90 | Loss=0.1131 | Accuracy=99.00%
Epoch  99 | Loss=0.1037 | Accuracy=99.00%

Training finished.
Final parameters: w=8.4523, b=-4.2261
Expected decision boundary: x = 0.5
Learned decision boundary: x = 0.5001
```

The model learns that the boundary is at `x = 0.5` — exactly matching the dataset rule. The large weight `w ≈ 8.5` makes the sigmoid sharp at the boundary, and `b ≈ -w/2` centers it.

---

## 📐 Mathematical Reference

### Chain Rule (Single Variable)

If `y = f(g(x))`, then:

```
dy/dx = df/dg · dg/dx
```

### Chain Rule (Multivariate — what backprop actually implements)

If `L` is the loss and `z = f(x, y)`, then:

```
∂L/∂x = ∂L/∂z · ∂z/∂x
```

In code, `d_output` is `∂L/∂z` (the incoming gradient), and `backward()` computes the local derivatives `∂z/∂x`, `∂z/∂y`.

### Derivative Reference Table

| Function `f(x)` | Derivative `f'(x)` |
|------------------|---------------------|
| `x + y` | `∂/∂x = 1`, `∂/∂y = 1` |
| `x × y` | `∂/∂x = y`, `∂/∂y = x` |
| `−x` | `−1` |
| `1/x` | `−1/x²` |
| `ln(x)` | `1/x` |
| `eˣ` | `eˣ` |
| `x²` | `2x` |
| `σ(x) = 1/(1+e⁻ˣ)` | `σ(x)(1 − σ(x))` |
| `relu(x)` | `1 if x > 0 else 0` |

### Binary Cross-Entropy

```
L = −[y · ln(p) + (1 − y) · ln(1 − p)]
```

### SGD Update Rule

```
θ(t+1) = θ(t) − η · ∂L/∂θ(t)
```

where `η` is the learning rate.

---

##  What's Next (v2 Roadmap)

This is **v1** — scalar-only autograd. Here's what future versions could add:

| Version | Feature | Description |
|---------|---------|-------------|
| **v2** | **Tensor support** | Extend `Scalar` to `Tensor` — multi-dimensional arrays with broadcasting |
| **v2** | **More optimizers** | Adam, RMSProp, momentum SGD |
| **v2** | **nn.Linear** | A proper linear layer: `y = Wx + b` |
| **v3** | **GPU backend** | CUDA kernels for parallel tensor operations |
| **v3** | **Conv2d** | 2D convolutions for image processing |
| **v3** | **Autograd for tensors** | Jacobian-vector products, batched backprop |
| **v4** | **MNIST training** | Train a full neural network on real data |

---

##  Resources & Inspiration

-  [MiniTorch — Sasha Rush (Cornell)](https://minitorch.github.io/) — The original course that inspired this project
-  [Autodiff from scratch (Andrej Karpathy's micrograd)](https://github.com/karpathy/micrograd)
-  [Calculus on Computational Graphs (Chris Olah)](https://colah.github.io/posts/2015-08-Backprop/)
-  [PyTorch Autograd Internals](https://pytorch.org/docs/stable/notes/autograd.html)
-  [Deep Learning (Goodfellow et al.) — Chapter 6.5: Back-Propagation](https://www.deeplearningbook.org/)

---

##  License

This project is open source and available for educational purposes.

---

<p align="center">
  <b>Built with ❤️ and pure Python — no frameworks, no shortcuts, just math.</b>
  <br><br>
  <i>"What I cannot create, I do not understand."</i> — Richard Feynman
</p>
