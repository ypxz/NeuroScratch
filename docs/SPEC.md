# NeuroScratch Shared Contract

**This is a shared contract. Do not change architecture, math conventions, hyperparameters, or the JSON schemas without coordinating, because multiple components depend on exact agreement.**

## Canonical model architecture

All implementations must match exactly:

- Input: 784 floats = 28x28 grayscale flattened row-major, normalized to [0,1] by dividing by 255.
- Layer 1: Dense 784 -> 256, then ReLU
- Layer 2: Dense 256 -> 128, then ReLU
- Layer 3: Dense 128 -> 10, then Softmax
- Loss: categorical cross-entropy (mean over batch)

## Forward-pass math convention

Python and JavaScript must use the same convention.

- A dense layer computes `z = x @ W + b` where `x` is `[batch, in]`, `W` is `[in, out]`, `b` is `[out]`.
- ReLU: `max(0, z)`.
- Softmax: numerically stable, subtract the row max before exponentiation.
- Prediction = `argmax` of softmax output.

## Weight initialization

- He/Kaiming normal for ReLU-preceding layers: `W ~ N(0, sqrt(2/fan_in))`.
- Final layer W: `W ~ N(0, sqrt(1/fan_in))`.
- Biases = 0.

## Canonical hyperparameters

A and B must use these identically:

- optimizer: Adam
- lr=1e-3
- beta1=0.9
- beta2=0.999
- eps=1e-8
- batch_size=64
- epochs=15
- global seed=42
- Data split: MNIST 60k training set split into 54k train / 6k validation (deterministic, seed=42); official 10k test set used only for final test accuracy. Both A and B use the SAME split.

## Weight export JSON schema

File: `web/weights.json`

```json
{
  "format_version": 1,
  "input": {"shape": [28, 28], "normalization": "divide_by_255"},
  "architecture": [
    {"type": "dense", "in": 784, "out": 256, "activation": "relu"},
    {"type": "dense", "in": 256, "out": 128, "activation": "relu"},
    {"type": "dense", "in": 128, "out": 10,  "activation": "softmax"}
  ],
  "layers": [
    {"W": [[...in x out row-major...]], "b": [...out...]},
    {"W": [[...]], "b": [...]},
    {"W": [[...]], "b": [...]}
  ],
  "meta": {"test_accuracy": 0.0, "trained_at": "ISO8601", "framework": "neuroscratch-from-scratch"}
}
```

Document explicitly: `layers[i].W` has shape `[architecture[i].in, architecture[i].out]` (row = input unit, col = output unit), matching `z = x @ W + b`. Floats are serialized as JSON numbers.

## Metrics logging schema

File: `training/runs/<run>/metrics.json` (directory is gitignored)

```json
{
  "hyperparams": {"optimizer":"adam","lr":1e-3,"beta1":0.9,"beta2":0.999,"eps":1e-8,"batch_size":64,"epochs":15,"seed":42,"architecture":"784-256-128-10"},
  "history": [{"epoch":1,"train_loss":0.0,"train_acc":0.0,"val_loss":0.0,"val_acc":0.0}],
  "test_accuracy": 0.0,
  "gradient_check": {"max_relative_error": 0.0, "tolerance": 1e-6}
}
```

This schema is the interface B's cross-validation report reads to compare against the from-scratch run.

## Module interface expectations

Keep the interface loose but named so the parallel sessions align.

- `neuroscratch.model` exposes: `Dense`, `ReLU`, `Softmax`, `CrossEntropyLoss`, `SGD`, `Adam`, and a `Sequential`/`Network` container with `.forward(x)`, `.backward(dout)`, `.params()`/`.grads()`.
- `neuroscratch.training` exposes a `load_mnist()` returning the canonical splits, and a `train()` entrypoint / `python -m neuroscratch.training.train` CLI.
- `neuroscratch.export` exposes `export_weights(network, path)` writing the schema above.
