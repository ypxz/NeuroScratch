import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { forward } from "../inference.js";

const weights = JSON.parse(await readFile(new URL("../weights.json", import.meta.url)));
const fixtures = JSON.parse(await readFile(new URL("./fixtures.json", import.meta.url)));

test("browser inference matches the parity fixture", () => {
  let argmaxMatches = 0;
  let maxProbabilityDiff = 0;
  for (const sample of fixtures.samples) {
    const input = sample.pixels.map((pixel) => pixel / 255);
    const actual = forward(weights, input);
    assert.strictEqual(actual.prediction, sample.prediction);
    argmaxMatches += 1;
    for (let index = 0; index < actual.probabilities.length; index += 1) {
      const diff = Math.abs(actual.probabilities[index] - sample.probabilities[index]);
      maxProbabilityDiff = Math.max(maxProbabilityDiff, diff);
      assert.ok(diff <= 1e-5, `probability mismatch at index ${index}: ${diff}`);
    }
  }
  assert.strictEqual(argmaxMatches, fixtures.num_samples);
  console.log(
    `parity samples=${fixtures.num_samples} argmax_matches=${argmaxMatches} max_prob_diff=${maxProbabilityDiff}`,
  );
});
