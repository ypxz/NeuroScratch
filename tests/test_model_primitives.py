from __future__ import annotations

import numpy as np

from neuroscratch.model import Adam, Dense, ReLU, Softmax, SoftmaxCrossEntropy


def test_dense_forward_and_backward_match_matrix_calculus() -> None:
    """For `z = xW + b`, the forward and backward matrix formulas should agree."""
    dense = Dense(2, 2, std=1.0, rng=np.random.default_rng(0))
    dense.W[...] = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    dense.b[...] = np.array([0.25, -0.75], dtype=np.float64)
    x = np.array([[2.0, -1.0], [0.0, 4.0]], dtype=np.float64)
    dout = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float64)

    forward = dense.forward(x)
    expected_forward = x @ dense.W + dense.b
    assert np.allclose(forward, expected_forward)

    dx = dense.backward(dout)
    assert np.allclose(dx, dout @ dense.W.T)
    assert np.allclose(dense.dW, x.T @ dout)
    assert np.allclose(dense.db, dout.sum(axis=0))


def test_relu_forward_and_backward_use_positive_mask() -> None:
    """ReLU should keep positive activations and zero the rest in backprop."""
    relu = ReLU()
    x = np.array([[-2.0, 0.0, 3.0], [1.5, -4.0, 2.0]], dtype=np.float64)
    dout = np.ones_like(x)

    forward = relu.forward(x)
    expected_forward = np.array([[0.0, 0.0, 3.0], [1.5, 0.0, 2.0]], dtype=np.float64)
    assert np.array_equal(forward, expected_forward)
    assert np.array_equal(relu.backward(dout), (x > 0.0).astype(np.float64))


def test_softmax_rows_sum_to_one_and_remain_stable() -> None:
    """Subtracting the row max should keep softmax finite for large logits."""
    softmax = Softmax()
    logits = np.array([[1000.0, 1001.0, 1002.0], [10000.0, 9999.0, 9998.0]], dtype=np.float64)
    probs = softmax.forward(logits)

    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_softmax_cross_entropy_gradient_matches_closed_form() -> None:
    """The softmax-CE backward pass should equal `(softmax(logits) - y)/N`."""
    logits = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]], dtype=np.float64)
    targets = np.array([2, 0], dtype=np.int64)
    loss = SoftmaxCrossEntropy()
    loss.forward(logits, targets)
    grad = loss.backward()

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    expected = probs.copy()
    expected[np.arange(targets.shape[0]), targets] -= 1.0
    expected /= targets.shape[0]
    assert np.allclose(grad, expected)


def test_adam_first_step_matches_closed_form_update() -> None:
    """The first Adam step should use the bias-corrected `m_hat` and `v_hat`."""
    params = [np.array([1.0, -1.0], dtype=np.float64)]
    grads = [np.array([0.5, -0.25], dtype=np.float64)]
    optimizer = Adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)

    optimizer.step(params, grads)

    expected = np.array([
        1.0 - 1e-3 * 0.5 / (np.sqrt(0.25) + 1e-8),
        -1.0 - 1e-3 * (-0.25) / (np.sqrt(0.0625) + 1e-8),
    ], dtype=np.float64)
    assert np.allclose(params[0], expected)
