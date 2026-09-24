// 캔버스 그림이나 사진을 모델 입력(정규화된 28x28 = 784개 값)으로 바꾸는 전처리
// desktop_version의 app.py(그리기)와 predict.py(이미지 파일) 흐름을 따르되, 크기 조정은 PIL의
// LANCZOS(app.py)/BICUBIC(predict.py의 ImageOps.pad 기본값) 대신 면적 평균으로 근사하므로, 실제로
// 크기가 바뀌는 입력에서는 데스크톱 버전과 값이 비슷하지만 완전히 같지는 않다

export const IMAGE_SIZE = 28;
const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;

/** RGBA 픽셀을 PIL convert("L")과 같은 정수 공식으로 흑백(0~255)으로 변환 (알파는 무시) */
export function toGrayscale(rgba, width, height) {
  const gray = new Float32Array(width * height);
  for (let i = 0; i < gray.length; i++) {
    const r = rgba[4 * i];
    const g = rgba[4 * i + 1];
    const b = rgba[4 * i + 2];
    gray[i] = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
  }
  return gray;
}

/** 1차원 면적 평균 가중치: 출력 칸마다 [입력 위치, 가중치] 목록 */
function areaWeights(srcLen, dstLen) {
  const scale = srcLen / dstLen;
  const result = [];
  for (let d = 0; d < dstLen; d++) {
    const start = d * scale;
    const end = (d + 1) * scale;
    const taps = [];
    for (let s = Math.floor(start); s < Math.min(Math.ceil(end), srcLen); s++) {
      const overlap = Math.min(end, s + 1) - Math.max(start, s);
      if (overlap > 0) taps.push([s, overlap / scale]);
    }
    result.push(taps);
  }
  return result;
}

/** 면적 평균으로 크기 변경 (가로 방향 후 세로 방향) — PIL LANCZOS/BICUBIC 대신 쓰는 근사 */
export function resizeArea(src, srcW, srcH, dstW, dstH) {
  const xTaps = areaWeights(srcW, dstW);
  const yTaps = areaWeights(srcH, dstH);

  const tmp = new Float32Array(srcH * dstW);
  for (let y = 0; y < srcH; y++) {
    for (let x = 0; x < dstW; x++) {
      let sum = 0;
      for (const [s, weight] of xTaps[x]) sum += src[y * srcW + s] * weight;
      tmp[y * dstW + x] = sum;
    }
  }

  const out = new Float32Array(dstW * dstH);
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      let sum = 0;
      for (const [s, weight] of yTaps[y]) sum += tmp[s * dstW + x] * weight;
      out[y * dstW + x] = sum;
    }
  }
  return out;
}

/** 비율을 유지한 채 size x size 안에 맞게 줄이고 남는 곳은 검정(0)으로 채워 가운데 배치 (PIL ImageOps.pad와 같은 방식) */
export function padAndResize(gray, width, height, size) {
  const scale = Math.min(size / width, size / height);
  const newW = Math.max(1, Math.round(width * scale));
  const newH = Math.max(1, Math.round(height * scale));
  const resized = resizeArea(gray, width, height, newW, newH);

  const out = new Float32Array(size * size);
  const offX = Math.round((size - newW) / 2);
  const offY = Math.round((size - newH) / 2);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      out[(offY + y) * size + offX + x] = resized[y * newW + x];
    }
  }
  return out;
}

/** 0~255 값을 transforms.ToTensor() + Normalize((0.1307,), (0.3081,))와 같게 변환 */
export function normalize(pixels) {
  return pixels.map((v) => (v / 255 - MNIST_MEAN) / MNIST_STD);
}

/** 그리기 캔버스(검은 배경에 흰 글씨) -> 28x28로 축소 (app.py와 같은 흐름) */
export function preprocessDrawing(rgba, width, height) {
  const gray = toGrayscale(rgba, width, height);
  const pixels = resizeArea(gray, width, height, IMAGE_SIZE, IMAGE_SIZE);
  return { pixels, input: normalize(pixels) };
}

/** 이미지 파일 -> 밝은 배경이면 반전 -> 정사각형 패딩 후 28x28 (predict.py와 같은 흐름) */
export function preprocessPhoto(rgba, width, height) {
  let gray = toGrayscale(rgba, width, height);

  // MNIST는 '검은 배경 + 흰 글씨'이므로, 평균 밝기가 밝으면(흰 종이 사진) 색을 반전
  const mean = gray.reduce((a, b) => a + b, 0) / gray.length;
  if (mean > 127) gray = gray.map((v) => 255 - v);

  const pixels = padAndResize(gray, width, height, IMAGE_SIZE);
  return { pixels, input: normalize(pixels) };
}

/** 이미지 파일(Blob)을 원래 크기의 RGBA 픽셀(ImageData)로 디코딩 */
export async function decodeImage(blob) {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}
