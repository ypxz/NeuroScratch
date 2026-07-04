# NeuroScratch

NeuroScratch is a from-scratch neural network project built to prove the mechanics of backpropagation, matrix calculus, and inference-time export without hiding behind a framework `.fit()` call.

## Architecture

- `neuroscratch/model/`: NumPy-only layers, activations, loss, optimizer, and the sequential network container.
- `neuroscratch/training/`: MNIST loading, training loop, checkpointing, and metrics logging.
- `neuroscratch/reference/`: PyTorch reference implementation for cross-validation only.
- `neuroscratch/export/`: weight export from Python into browser-ready JSON.
- `web/`: static browser demo, canvas pad, JS inference engine, and UI.
- `tests/`: gradient checks, cross-validation, and JS/Python parity tests.

### Canonical model

- Input: 784 floats from a 28x28 grayscale image, flattened row-major and normalized by dividing by 255.
- Dense 784 -> 256 -> ReLU
- Dense 256 -> 128 -> ReLU
- Dense 128 -> 10 -> Softmax
- Loss: categorical cross-entropy (mean over batch)

## Results

<!-- TODO: fill after real runs -->

| Metric | Value |
| --- | --- |
| Test accuracy | <!-- TODO --> |
| Gradient-check tolerance | <!-- TODO --> |
| From-scratch vs PyTorch delta | <!-- TODO --> |
| JS-vs-Python parity rate | <!-- TODO --> |

## Reproduce training

<!-- TODO: fill after real runs -->

1. Create the virtual environment.
2. Install the package in editable mode.
3. Run the training entrypoint.
4. Export weights and launch the browser demo.

## Demo

<!-- TODO: fill after real runs -->

- GIF/screenshot placeholder.

## What I'd extend next

<!-- TODO: fill after real runs -->

- Better optimization experiments.
- Regularization and augmentation.
- A richer browser UI and interactive digit pad.
