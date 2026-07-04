import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { OOD_ENERGY_THRESHOLD, energyScore, forward } from "../inference.js";

const weights = JSON.parse(await readFile(new URL("../weights.json", import.meta.url)));
const fixtures = JSON.parse(await readFile(new URL("./fixtures.json", import.meta.url)));

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function predictEnergy(input) {
  return energyScore(forward(weights, input).logits);
}

test("real MNIST fixtures mostly stay below the OOD threshold", () => {
  const energies = fixtures.samples.map((sample) => {
    const input = sample.pixels.map((pixel) => pixel / 255);
    return predictEnergy(input);
  });
  const flagged = energies.filter((energy) => energy > OOD_ENERGY_THRESHOLD).length;
  const flagRate = flagged / energies.length;

  assert.ok(flagRate <= 0.05, `expected <=5% flagged, got ${(flagRate * 100).toFixed(2)}%`);
  console.log(
    `ood real_samples=${energies.length} flagged=${flagged} flag_rate=${flagRate.toFixed(
      4,
    )} threshold=${OOD_ENERGY_THRESHOLD}`,
  );
});

test("sparse random scribbles are flagged as OOD", () => {
  const random = seededRandom(12345);
  const sampleCount = 100;
  let flagged = 0;

  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    const input = new Array(784).fill(0);
    const coverage = 0.1 + random() * 0.1;
    const activeCount = Math.max(1, Math.round(input.length * coverage));
    for (let index = 0; index < activeCount; index += 1) {
      const pixelIndex = Math.floor(random() * input.length);
      input[pixelIndex] = 0.35 + random() * 0.65;
    }

    const energy = predictEnergy(input);
    if (energy > OOD_ENERGY_THRESHOLD) {
      flagged += 1;
    }
  }

  const flagRate = flagged / sampleCount;
  assert.ok(flagRate >= 0.95, `expected >=95% flagged, got ${(flagRate * 100).toFixed(2)}%`);
  console.log(
    `ood synthetic_samples=${sampleCount} flagged=${flagged} flag_rate=${flagRate.toFixed(
      4,
    )} threshold=${OOD_ENERGY_THRESHOLD}`,
  );
});
