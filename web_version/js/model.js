// desktop_version/model.py의 MnistCNN을 순수 자바스크립트로 옮긴 추론 코드
// 층 구성이 바뀌면 이 파일도 같이 고쳐야 함

import { conv2d3x3, relu, maxPool2x2, linear, softmax, argmax } from "./ops.js";

/**
 * weights.json 매니페스트와 weights.bin 버퍼를 레이어 이름별 텐서로 나눔
 * (weights.bin은 리틀엔디언 float32이며, 브라우저가 도는 거의 모든 기기가 리틀엔디언임)
 */
export function parseWeights(manifest, buffer) {
  const all = new Float32Array(buffer);
  const weights = {};
  for (const { name, shape, offset, length } of manifest.tensors) {
    if (offset + length > all.length) {
      throw new Error(`가중치 파일이 손상되었습니다: ${name} 범위가 파일 크기를 넘습니다`);
    }
    weights[name] = { shape, data: all.subarray(offset, offset + length) };
  }
  return weights;
}

/** baseUrl 폴더에서 weights.json과 weights.bin을 내려받아 파싱 */
export async function loadWeights(baseUrl) {
  const [manifestRes, binRes] = await Promise.all([
    fetch(new URL("weights.json", baseUrl)),
    fetch(new URL("weights.bin", baseUrl)),
  ]);
  if (!manifestRes.ok || !binRes.ok) {
    throw new Error(`가중치 파일을 불러오지 못했습니다 (HTTP ${manifestRes.status}/${binRes.status})`);
  }
  return parseWeights(await manifestRes.json(), await binRes.arrayBuffer());
}

/** 정규화된 28x28 입력(784개)을 받아 클래스별 점수(logit) 10개를 계산 */
export function forward(weights, input) {
  let x = conv2d3x3(input, 1, 28, 28, weights["conv1.weight"].data, weights["conv1.bias"].data, 32);
  x = maxPool2x2(relu(x), 32, 28, 28); // 28x28 -> 14x14

  x = conv2d3x3(x, 32, 14, 14, weights["conv2.weight"].data, weights["conv2.bias"].data, 64);
  x = maxPool2x2(relu(x), 64, 14, 14); // 14x14 -> 7x7

  // 드롭아웃은 추론할 때 아무 일도 하지 않으므로 생략
  // [64, 7, 7] 배열은 이미 PyTorch flatten(1)과 같은 순서임
  x = relu(linear(x, weights["fc1.weight"].data, weights["fc1.bias"].data));
  return linear(x, weights["fc2.weight"].data, weights["fc2.bias"].data);
}

export function predict(weights, input) {
  const probabilities = softmax(forward(weights, input));
  const digit = argmax(probabilities);
  return { digit, confidence: probabilities[digit], probabilities };
}
