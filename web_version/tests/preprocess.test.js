import { test, assert, assertClose } from "./harness.js";
import {
  toGrayscale, resizeArea, padAndResize, normalize,
  preprocessDrawing, preprocessPhoto, decodeImage,
} from "../js/preprocess.js";

test("toGrayscale는 PIL convert('L')과 같은 값을 낸다", () => {
  const rgba = Uint8ClampedArray.from([
    255, 0, 0, 255, // 빨강 -> 76
    0, 255, 0, 255, // 초록 -> 150
    0, 0, 255, 255, // 파랑 -> 29
    255, 255, 255, 0, // 흰색(알파 무시) -> 255
  ]);
  assertClose(toGrayscale(rgba, 4, 1), [76, 150, 29, 255]);
});

test("resizeArea는 정수배 축소에서 블록 평균을 낸다", () => {
  const src = Float32Array.from({ length: 16 }, (_, i) => i); // 4x4
  assertClose(resizeArea(src, 4, 4, 2, 2), [2.5, 4.5, 10.5, 12.5]);
});

test("resizeArea는 정수배가 아닌 축소에서 겹치는 면적으로 가중 평균한다", () => {
  // 가로 3칸 -> 2칸: [0, 3, 6] -> [(0 + 3*0.5)/1.5, (3*0.5 + 6)/1.5]
  assertClose(resizeArea(Float32Array.from([0, 3, 6]), 3, 1, 2, 1), [1, 5], 1e-5);
});

test("resizeArea는 크기가 같으면 값을 그대로 둔다", () => {
  const src = Float32Array.from([1, 2, 3, 4, 5, 6]);
  assertClose(resizeArea(src, 3, 2, 3, 2), src);
});

test("padAndResize는 비율을 유지하고 남는 곳을 검정으로 채워 가운데 둔다", () => {
  const gray = new Float32Array(8).fill(255); // 가로 4 x 세로 2
  assertClose(padAndResize(gray, 4, 2, 4), [
    0, 0, 0, 0,
    255, 255, 255, 255,
    255, 255, 255, 255,
    0, 0, 0, 0,
  ]);
});

test("padAndResize는 PIL round()(round-half-to-even)와 같은 크기/위치를 계산한다", () => {
  // 20x30 -> size 28: newW = round(20/30*28) = round(18.667) = 19,
  // offX = round((28-19)*0.5) = round(4.5) = 4 (half-even; JS Math.round라면 5가 됨)
  const gray = new Float32Array(20 * 30).fill(255);
  const out = padAndResize(gray, 20, 30, 28);
  const midRow = 14; // newH = 28이라 세로 오프셋은 0, 가운데 행 아무거나 확인
  assertClose(out[midRow * 28 + 3], 0, 1e-6, "패딩 왼쪽 바깥(3열)은 0이어야 함");
  for (let x = 4; x <= 22; x++) {
    assert(out[midRow * 28 + x] > 0, `이미지 안쪽(${x}열)은 0보다 커야 함`);
  }
  assertClose(out[midRow * 28 + 23], 0, 1e-6, "패딩 오른쪽 바깥(23열)은 0이어야 함");
});

test("normalize는 MNIST 평균/표준편차로 정규화한다", () => {
  assertClose(normalize(Float32Array.from([0, 255])), [-0.1307 / 0.3081, (1 - 0.1307) / 0.3081], 1e-6);
});

test("preprocessPhoto는 밝은 배경 사진을 반전한다", () => {
  // 흰 배경(255)에 검은 점 하나 -> 반전하면 검은 배경에 흰 점 하나
  const rgba = new Uint8ClampedArray(28 * 28 * 4).fill(255);
  rgba.set([0, 0, 0, 255], 0);
  const { pixels } = preprocessPhoto(rgba, 28, 28);
  assertClose(pixels[0], 255);
  assertClose(pixels.slice(1), new Float32Array(783));
});

test("preprocessDrawing은 280x280 캔버스를 28x28로 줄인다", () => {
  // 검은 캔버스의 왼쪽 위 10x10만 흰색 -> 28x28의 첫 픽셀만 255
  const rgba = new Uint8ClampedArray(280 * 280 * 4);
  for (let y = 0; y < 280; y++) {
    for (let x = 0; x < 280; x++) {
      const v = x < 10 && y < 10 ? 255 : 0;
      rgba.set([v, v, v, 255], (y * 280 + x) * 4);
    }
  }
  const { pixels, input } = preprocessDrawing(rgba, 280, 280);
  assertClose(pixels[0], 255, 1e-4);
  assertClose(pixels.slice(1), new Float32Array(783), 1e-4);
  assertClose(input[1], -0.1307 / 0.3081, 1e-5);
});

test("샘플 PNG를 전처리한 결과가 PyTorch 입력과 같다", async () => {
  const reference = await (await fetch(new URL("./fixtures/reference.json", import.meta.url))).json();
  for (const s of reference.samples) {
    const blob = await (await fetch(new URL(`./fixtures/${s.file}`, import.meta.url))).blob();
    const image = await decodeImage(blob);
    const { input } = preprocessPhoto(image.data, image.width, image.height);
    assertClose(input, s.input, 1e-5, s.file);
  }
});
