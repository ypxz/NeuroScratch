/** Compute an affine transform `x @ W + b`. */
export function dense(x, W, b) {
  const out = new Array(b.length).fill(0);
  for (let j = 0; j < b.length; j += 1) {
    let sum = b[j];
    for (let i = 0; i < x.length; i += 1) {
      sum += x[i] * W[i][j];
    }
    out[j] = sum;
  }
  return out;
}

/** Apply ReLU elementwise. */
export function relu(z) {
  return z.map((value) => Math.max(0, value));
}

/** Compute a numerically stable softmax. */
export function softmax(z) {
  const maxValue = Math.max(...z);
  const expValues = z.map((value) => Math.exp(value - maxValue));
  const total = expValues.reduce((sum, value) => sum + value, 0);
  return expValues.map((value) => value / total);
}

/** Compute the numerically stable log-sum-exp of a score vector. */
export function logSumExp(logits) {
  const maxValue = Math.max(...logits);
  const expValues = logits.map((value) => Math.exp(value - maxValue));
  return maxValue + Math.log(expValues.reduce((sum, value) => sum + value, 0));
}

/**
 * Compute an energy-based OOD score, where lower values are more in-distribution.
 * The score is `-logsumexp(logits)` with a max-shift for numerical stability.
 */
export function energyScore(logits) {
  return -logSumExp(logits);
}

/** Return the lowest index of the maximum element. */
export function argmax(arr) {
  let bestIndex = 0;
  let bestValue = arr[0];
  for (let index = 1; index < arr.length; index += 1) {
    const value = arr[index];
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return bestIndex;
}

/**
 * OOD energy threshold calibrated on the 200 MNIST parity fixtures.
 * Values above this threshold are treated as "not a clear digit".
 * Measured false-positive rate on the fixtures: 4%.
 */
export const OOD_ENERGY_THRESHOLD = -10.00893166160672;

/**
 * Run the canonical three-layer forward pass.
 * @param {{layers: Array<{W: number[][], b: number[]}>}} weights
 * @param {number[]} input784
 */
export function forward(weights, input784) {
  const [layer0, layer1, layer2] = weights.layers;
  const hidden0 = relu(dense(input784, layer0.W, layer0.b));
  const hidden1 = relu(dense(hidden0, layer1.W, layer1.b));
  const logits = dense(hidden1, layer2.W, layer2.b);
  const probabilities = softmax(logits);
  return {
    logits,
    probabilities,
    prediction: argmax(probabilities),
  };
}

/** Load exported weights from a URL in the browser. */
export async function loadWeights(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to load weights from ${url}: ${response.status}`);
  }
  return response.json();
}
