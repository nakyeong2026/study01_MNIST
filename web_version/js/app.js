// 화면 연결: 캔버스 그리기(마우스/터치), 이미지 업로드, 인식 결과 표시

import { loadWeights, predict } from "./model.js";
import { preprocessDrawing, preprocessPhoto, decodeImage, IMAGE_SIZE } from "./preprocess.js";

const PEN_WIDTH = 18; // desktop_version/app.py와 같은 펜 굵기 (280px 캔버스 기준)

const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const clearButton = document.getElementById("clear-button");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const digitEl = document.getElementById("result-digit");
const confidenceEl = document.getElementById("result-confidence");
const barsEl = document.getElementById("prob-bars");
const previewCanvas = document.getElementById("input-preview");
const previewCtx = previewCanvas.getContext("2d");

let weights = null;
let drawing = false;
let lastPoint = null;
const barFills = [];
const barValues = [];

function buildBars() {
  for (let digit = 0; digit < 10; digit++) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `<span>${digit}</span><div class="bar-track"><div class="bar-fill"></div></div><span class="bar-value">0%</span>`;
    barFills.push(row.querySelector(".bar-fill"));
    barValues.push(row.querySelector(".bar-value"));
    barsEl.appendChild(row);
  }
}

function clearCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function resetResult() {
  digitEl.textContent = "?";
  confidenceEl.textContent = "숫자를 그려 보세요";
  for (let i = 0; i < 10; i++) {
    barFills[i].style.width = "0%";
    barValues[i].textContent = "0%";
  }
  previewCtx.fillStyle = "black";
  previewCtx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
}

/** 모델에 실제로 들어간 28x28 입력을 미리보기 캔버스에 그림 */
function renderPreview(pixels) {
  const image = previewCtx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
  for (let i = 0; i < pixels.length; i++) {
    const v = Math.max(0, Math.min(255, Math.round(pixels[i])));
    image.data.set([v, v, v, 255], i * 4);
  }
  previewCtx.putImageData(image, 0, 0);
}

function showResult({ pixels, input }) {
  const { digit, confidence, probabilities } = predict(weights, input);
  digitEl.textContent = String(digit);
  confidenceEl.textContent = `신뢰도 ${(confidence * 100).toFixed(1)}%`;
  for (let i = 0; i < 10; i++) {
    const percent = probabilities[i] * 100;
    barFills[i].style.width = `${percent}%`;
    barValues[i].textContent = `${percent.toFixed(1)}%`;
  }
  renderPreview(pixels);
}

// 화면에 보이는 크기와 캔버스 실제 크기(280)가 다를 수 있으므로 좌표를 환산
function toCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * canvas.width) / rect.width,
    y: ((event.clientY - rect.top) * canvas.height) / rect.height,
  };
}

function drawLine(from, to) {
  ctx.strokeStyle = "white";
  ctx.lineWidth = PEN_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}

function recognizeDrawing() {
  if (!weights) return;
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  showResult(preprocessDrawing(data, width, height));
}

canvas.addEventListener("pointerdown", (event) => {
  drawing = true;
  canvas.setPointerCapture(event.pointerId);
  lastPoint = toCanvasPoint(event);
  drawLine(lastPoint, lastPoint); // 클릭만 해도 점이 찍히도록
});

canvas.addEventListener("pointermove", (event) => {
  if (!drawing) return;
  const point = toCanvasPoint(event);
  drawLine(lastPoint, point);
  lastPoint = point;
});

// 한 획을 다 그으면 바로 인식
function endStroke() {
  if (!drawing) return;
  drawing = false;
  recognizeDrawing();
}
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);

clearButton.addEventListener("click", () => {
  clearCanvas();
  resetResult();
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  fileInput.value = ""; // 같은 파일을 다시 골라도 change 이벤트가 나도록
  if (!file || !weights) return;
  try {
    const image = await decodeImage(file);
    showResult(preprocessPhoto(image.data, image.width, image.height));
    statusEl.textContent = `"${file.name}" 인식 결과`;
  } catch (err) {
    console.error(err);
    statusEl.textContent = "이미지를 읽을 수 없습니다. 다른 파일을 골라 주세요.";
  }
});

async function init() {
  buildBars();
  clearCanvas();
  resetResult();
  try {
    weights = await loadWeights(new URL("../model/", import.meta.url));
    fileInput.disabled = false;
    statusEl.textContent = "숫자를 그리거나 이미지를 올려 보세요";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "모델을 불러오지 못했습니다. 페이지를 http(s) 주소로 열었는지 확인해 주세요.";
  }
}

init();
