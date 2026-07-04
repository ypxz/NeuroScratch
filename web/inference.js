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
 * Run the canonical three-layer forward pass.
 * @param {{layers: Array<{W: number[][], b: number[]}>}} weights
 * @param {number[]} input784
 */
export function forward(weights, input784) {
  const [layer0, layer1, layer2] = weights.layers;
  const hidden0 = relu(dense(input784, layer0.W, layer0.b));
  const hidden1 = relu(dense(hidden0, layer1.W, layer1.b));
  const probabilities = softmax(dense(hidden1, layer2.W, layer2.b));
  return {
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
