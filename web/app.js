import { forward, loadWeights } from "./inference.js";

const drawCanvas = document.getElementById("draw-canvas");
const drawCtx = drawCanvas.getContext("2d", { willReadFrequently: true });
const inputPreviewCanvas = document.getElementById("input-preview");
const inputPreviewCtx = inputPreviewCanvas.getContext("2d");
const perturbedPreviewCanvas = document.getElementById("perturbed-preview");
const perturbedPreviewCtx = perturbedPreviewCanvas.getContext("2d");
const clearButton = document.getElementById("clear-btn");
const modelStatus = document.getElementById("model-status");
const predictionValue = document.getElementById("prediction-value");
const predictionMeta = document.getElementById("prediction-meta");
const perturbedValue = document.getElementById("perturbed-value");
const perturbedMeta = document.getElementById("perturbed-meta");
const probabilityList = document.getElementById("probability-list");
const perturbedProbabilityList = document.getElementById("perturbed-probability-list");
const weightsGrid = document.getElementById("weights-grid");
const noiseSlider = document.getElementById("noise-slider");
const noiseValue = document.getElementById("noise-value");
const noiseSeedInput = document.getElementById("noise-seed");
const regenerateNoiseButton = document.getElementById("regenerate-noise-btn");

const MAIN_SIZE = 280;
const PREVIEW_SIZE = 28;
const DIGIT_BOX = 20;
const BRUSH_RADIUS = 10;
const DEBOUNCE_MS = 120;

const state = {
  weights: null,
  currentInput: null,
  currentPrediction: null,
  currentProbabilities: null,
  perturbedPrediction: null,
  perturbedProbabilities: null,
  drawing: false,
  lastPoint: null,
  debounceTimer: null,
};

const offscreen = {
  scaled: document.createElement("canvas"),
  centered: document.createElement("canvas"),
  perturbed: document.createElement("canvas"),
};

offscreen.scaled.width = DIGIT_BOX;
offscreen.scaled.height = DIGIT_BOX;
offscreen.centered.width = PREVIEW_SIZE;
offscreen.centered.height = PREVIEW_SIZE;
offscreen.perturbed.width = PREVIEW_SIZE;
offscreen.perturbed.height = PREVIEW_SIZE;

const scaledCtx = offscreen.scaled.getContext("2d", { willReadFrequently: true });
const centeredCtx = offscreen.centered.getContext("2d", { willReadFrequently: true });
const perturbedCtx = offscreen.perturbed.getContext("2d", { willReadFrequently: true });

function clearCanvas(ctx, width, height) {
  ctx.save();
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);
  ctx.restore();
}

function drawBackground(ctx) {
  clearCanvas(ctx, MAIN_SIZE, MAIN_SIZE);
}

function setCanvasPixelated(canvas) {
  canvas.style.imageRendering = "pixelated";
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function getCanvasPoint(event) {
  const rect = drawCanvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * drawCanvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * drawCanvas.height;
  return {
    x: clamp(x, 0, drawCanvas.width - 1),
    y: clamp(y, 0, drawCanvas.height - 1),
  };
}

function drawBrushLine(from, to) {
  drawCtx.save();
  drawCtx.strokeStyle = "#fff";
  drawCtx.lineWidth = BRUSH_RADIUS * 2;
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.beginPath();
  drawCtx.moveTo(from.x, from.y);
  drawCtx.lineTo(to.x, to.y);
  drawCtx.stroke();
  drawCtx.restore();
}

function drawBrushDot(point) {
  drawCtx.save();
  drawCtx.fillStyle = "#fff";
  drawCtx.beginPath();
  drawCtx.arc(point.x, point.y, BRUSH_RADIUS, 0, Math.PI * 2);
  drawCtx.fill();
  drawCtx.restore();
}

function scheduleUpdate(immediate = false) {
  if (state.debounceTimer) {
    window.clearTimeout(state.debounceTimer);
    state.debounceTimer = null;
  }
  if (immediate) {
    updatePredictions();
    return;
  }
  state.debounceTimer = window.setTimeout(() => {
    state.debounceTimer = null;
    updatePredictions();
  }, DEBOUNCE_MS);
}

function getBoundingBox(imageData) {
  const { data, width, height } = imageData;
  let left = width;
  let top = height;
  let right = -1;
  let bottom = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      const brightness = data[index];
      if (brightness > 10) {
        left = Math.min(left, x);
        top = Math.min(top, y);
        right = Math.max(right, x);
        bottom = Math.max(bottom, y);
      }
    }
  }

  if (right < left || bottom < top) {
    return null;
  }

  return {
    left,
    top,
    width: right - left + 1,
    height: bottom - top + 1,
  };
}

function computeCenterOfMass(imageData) {
  const { data, width, height } = imageData;
  let total = 0;
  let sumX = 0;
  let sumY = 0;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      const value = data[index] / 255;
      if (value <= 0) {
        continue;
      }
      total += value;
      sumX += value * (x + 0.5);
      sumY += value * (y + 0.5);
    }
  }

  if (total === 0) {
    return { x: width / 2, y: height / 2 };
  }

  return {
    x: sumX / total,
    y: sumY / total,
  };
}

function preprocessCanvas() {
  const fullImage = drawCtx.getImageData(0, 0, MAIN_SIZE, MAIN_SIZE);
  const bbox = getBoundingBox(fullImage);
  if (!bbox) {
    return null;
  }

  scaledCtx.clearRect(0, 0, DIGIT_BOX, DIGIT_BOX);
  const scale = DIGIT_BOX / Math.max(bbox.width, bbox.height);
  const scaledWidth = Math.max(1, Math.round(bbox.width * scale));
  const scaledHeight = Math.max(1, Math.round(bbox.height * scale));
  const offsetX = Math.floor((DIGIT_BOX - scaledWidth) / 2);
  const offsetY = Math.floor((DIGIT_BOX - scaledHeight) / 2);
  scaledCtx.imageSmoothingEnabled = true;
  scaledCtx.drawImage(
    drawCanvas,
    bbox.left,
    bbox.top,
    bbox.width,
    bbox.height,
    offsetX,
    offsetY,
    scaledWidth,
    scaledHeight,
  );

  const scaledImage = scaledCtx.getImageData(0, 0, DIGIT_BOX, DIGIT_BOX);
  const mass = computeCenterOfMass(scaledImage);

  centeredCtx.clearRect(0, 0, PREVIEW_SIZE, PREVIEW_SIZE);
  const translateX = Math.round(PREVIEW_SIZE / 2 - mass.x);
  const translateY = Math.round(PREVIEW_SIZE / 2 - mass.y);
  centeredCtx.drawImage(offscreen.scaled, translateX, translateY);

  const centeredImage = centeredCtx.getImageData(0, 0, PREVIEW_SIZE, PREVIEW_SIZE);
  const input = new Array(784);
  const { data } = centeredImage;
  let pointer = 0;
  for (let y = 0; y < PREVIEW_SIZE; y += 1) {
    for (let x = 0; x < PREVIEW_SIZE; x += 1) {
      const index = (y * PREVIEW_SIZE + x) * 4;
      input[pointer] = data[index] / 255;
      pointer += 1;
    }
  }

  return {
    input,
  };
}

function renderPreviewFromInput(input, ctx) {
  const imageData = ctx.createImageData(PREVIEW_SIZE, PREVIEW_SIZE);
  for (let index = 0; index < input.length; index += 1) {
    const value = Math.round(clamp(input[index], 0, 1) * 255);
    const offset = index * 4;
    imageData.data[offset] = value;
    imageData.data[offset + 1] = value;
    imageData.data[offset + 2] = value;
    imageData.data[offset + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function renderWeightUnit(canvas, values) {
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(PREVIEW_SIZE, PREVIEW_SIZE);
  let maxAbs = 0;
  for (const value of values) {
    maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  const scale = maxAbs === 0 ? 0 : 1 / maxAbs;
  for (let index = 0; index < values.length; index += 1) {
    const normalized = values[index] * scale;
    let r = 0;
    let g = 0;
    let b = 0;
    const magnitude = Math.min(1, Math.abs(normalized));
    const channel = Math.round(magnitude * 255);
    if (normalized > 0) {
      r = channel;
    } else if (normalized < 0) {
      b = channel;
    }
    const offset = index * 4;
    imageData.data[offset] = r;
    imageData.data[offset + 1] = g;
    imageData.data[offset + 2] = b;
    imageData.data[offset + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function createProbabilityRow(label) {
  const row = document.createElement("div");
  row.className = "probability-row";

  const labelNode = document.createElement("div");
  labelNode.className = "probability-label";
  labelNode.textContent = label;

  const track = document.createElement("div");
  track.className = "probability-track";

  const fill = document.createElement("div");
  fill.className = "probability-fill";
  track.append(fill);

  const valueNode = document.createElement("div");
  valueNode.className = "probability-value";
  valueNode.textContent = "0.0%";

  row.append(labelNode, track, valueNode);

  return { row, fill, valueNode };
}

function resetProbabilityRows(rows) {
  for (const row of rows) {
    row.fill.style.width = "0%";
    row.valueNode.textContent = "0.0%";
    row.row.classList.remove("is-top");
  }
}

const originalProbabilityRows = [];
const perturbedProbabilityRows = [];
for (let digit = 0; digit <= 9; digit += 1) {
  const original = createProbabilityRow(String(digit));
  originalProbabilityRows.push(original);
  probabilityList.append(original.row);

  const perturbed = createProbabilityRow(String(digit));
  perturbedProbabilityRows.push(perturbed);
  perturbedProbabilityList.append(perturbed.row);
}

function updateProbabilityRows(rows, probabilities, prediction) {
  for (let digit = 0; digit < rows.length; digit += 1) {
    const probability = probabilities[digit];
    const row = rows[digit];
    row.fill.style.width = `${(probability * 100).toFixed(1)}%`;
    row.valueNode.textContent = `${(probability * 100).toFixed(1)}%`;
    row.row.classList.toggle("is-top", digit === prediction);
  }
}

function setPredictionElements(valueNode, metaNode, prediction, probabilities) {
  if (!probabilities) {
    valueNode.textContent = "—";
    metaNode.textContent = "Draw a digit to begin.";
    return;
  }
  valueNode.textContent = String(prediction);
  metaNode.textContent = `Top class: ${prediction}, confidence ${(probabilities[prediction] * 100).toFixed(
    1,
  )}%`;
}

function seededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generatePerturbedInput(input, amount, seed) {
  const random = seededRandom(seed);
  return input.map((value) => clamp(value + (random() * 2 - 1) * amount, 0, 1));
}

function updateNoiseLabel() {
  noiseValue.textContent = Number.parseFloat(noiseSlider.value).toFixed(2);
}

function renderPerturbedState(input) {
  const amount = Number.parseFloat(noiseSlider.value);
  const seed = Number.parseInt(noiseSeedInput.value, 10) || 0;
  const perturbedInput = generatePerturbedInput(input, amount, seed);
  renderPreviewFromInput(perturbedInput, perturbedCtx);
  perturbedPreviewCtx.clearRect(0, 0, PREVIEW_SIZE, PREVIEW_SIZE);
  perturbedPreviewCtx.drawImage(offscreen.perturbed, 0, 0);

  const perturbedResult = forward(state.weights, perturbedInput);
  state.perturbedPrediction = perturbedResult.prediction;
  state.perturbedProbabilities = perturbedResult.probabilities;
  setPredictionElements(perturbedValue, perturbedMeta, perturbedResult.prediction, perturbedResult.probabilities);
  updateProbabilityRows(perturbedProbabilityRows, perturbedResult.probabilities, perturbedResult.prediction);
}

function updatePredictions() {
  if (!state.weights) {
    return;
  }
  const processed = preprocessCanvas();
  if (!processed) {
    state.currentInput = null;
    state.currentPrediction = null;
    state.currentProbabilities = null;
    state.perturbedPrediction = null;
    state.perturbedProbabilities = null;
    setPredictionElements(predictionValue, predictionMeta, null, null);
    setPredictionElements(perturbedValue, perturbedMeta, null, null);
    resetProbabilityRows(originalProbabilityRows);
    resetProbabilityRows(perturbedProbabilityRows);
    clearCanvas(inputPreviewCtx, PREVIEW_SIZE, PREVIEW_SIZE);
    clearCanvas(perturbedPreviewCtx, PREVIEW_SIZE, PREVIEW_SIZE);
    return;
  }

  state.currentInput = processed.input;
  inputPreviewCtx.clearRect(0, 0, PREVIEW_SIZE, PREVIEW_SIZE);
  inputPreviewCtx.drawImage(offscreen.centered, 0, 0);

  const result = forward(state.weights, processed.input);
  state.currentPrediction = result.prediction;
  state.currentProbabilities = result.probabilities;

  setPredictionElements(predictionValue, predictionMeta, result.prediction, result.probabilities);
  updateProbabilityRows(originalProbabilityRows, result.probabilities, result.prediction);

  renderPerturbedState(processed.input);
}

function initializeDrawingCanvas() {
  drawBackground(drawCtx);
  drawCanvas.addEventListener("pointerdown", (event) => {
    drawCanvas.setPointerCapture(event.pointerId);
    state.drawing = true;
    const point = getCanvasPoint(event);
    drawBrushDot(point);
    state.lastPoint = point;
    scheduleUpdate();
  });

  drawCanvas.addEventListener("pointermove", (event) => {
    if (!state.drawing || state.lastPoint === null) {
      return;
    }
    const point = getCanvasPoint(event);
    drawBrushLine(state.lastPoint, point);
    state.lastPoint = point;
    scheduleUpdate();
  });

  const finishStroke = () => {
    if (!state.drawing) {
      return;
    }
    state.drawing = false;
    state.lastPoint = null;
    scheduleUpdate(true);
  };

  drawCanvas.addEventListener("pointerup", finishStroke);
  drawCanvas.addEventListener("pointercancel", finishStroke);
  drawCanvas.addEventListener("lostpointercapture", finishStroke);
}

function clearDrawing() {
  drawBackground(drawCtx);
  scheduleUpdate(true);
}

async function initializeWeights(weights) {
  state.weights = weights;
  modelStatus.textContent = "Ready";

  const firstLayer = weights.layers[0];
  for (let unit = 0; unit < 64; unit += 1) {
    const unitWrap = document.createElement("div");
    unitWrap.className = "weight-unit";

    const canvas = document.createElement("canvas");
    canvas.width = PREVIEW_SIZE;
    canvas.height = PREVIEW_SIZE;
    setCanvasPixelated(canvas);

    const values = firstLayer.W.map((row) => row[unit]);
    renderWeightUnit(canvas, values);

    const label = document.createElement("div");
    label.className = "weight-index";
    label.textContent = `Unit ${unit}`;

    unitWrap.append(canvas, label);
    weightsGrid.append(unitWrap);
  }

  updatePredictions();
}

function initializeNoiseControls() {
  updateNoiseLabel();
  noiseSlider.addEventListener("input", () => {
    updateNoiseLabel();
    if (state.currentInput) {
      renderPerturbedState(state.currentInput);
    }
  });
  noiseSeedInput.addEventListener("input", () => {
    if (state.currentInput) {
      renderPerturbedState(state.currentInput);
    }
  });
  regenerateNoiseButton.addEventListener("click", () => {
    noiseSeedInput.value = String(Math.floor(Math.random() * 1_000_000));
    if (state.currentInput) {
      renderPerturbedState(state.currentInput);
    }
  });
}

async function main() {
  initializeDrawingCanvas();
  clearButton.addEventListener("click", clearDrawing);
  initializeNoiseControls();
  drawBackground(drawCtx);

  try {
    const weights = await loadWeights("./weights.json");
    await initializeWeights(weights);
  } catch (error) {
    modelStatus.textContent = "Failed to load weights";
    predictionMeta.textContent = error instanceof Error ? error.message : String(error);
  }
}

main();
