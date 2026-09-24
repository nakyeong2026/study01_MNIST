import { test, assert, assertEqual, assertClose } from "./harness.js";
import { parseWeights, forward, predict } from "../js/model.js";

// 실제 모델과 같은 모양의 0 가중치 (fc2.bias만 채워 넣어 순전파 연결을 확인)
function makeZeroWeights() {
  const shapes = {
    "conv1.weight": [32, 1, 3, 3], "conv1.bias": [32],
    "conv2.weight": [64, 32, 3, 3], "conv2.bias": [64],
    "fc1.weight": [128, 3136], "fc1.bias": [128],
    "fc2.weight": [10, 128], "fc2.bias": [10],
  };
  const weights = {};
  for (const [name, shape] of Object.entries(shapes)) {
    weights[name] = { shape, data: new Float32Array(shape.reduce((a, b) => a * b, 1)) };
  }
  weights["fc2.bias"].data.set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  return weights;
}

test("parseWeights는 버퍼를 매니페스트대로 나눈다", () => {
  const manifest = { tensors: [
    { name: "a", shape: [2], offset: 0, length: 2 },
    { name: "b", shape: [3], offset: 2, length: 3 },
  ] };
  const w = parseWeights(manifest, Float32Array.from([1, 2, 3, 4, 5]).buffer);
  assert(w.a.data instanceof Float32Array, "Float32Array가 아님");
  assertClose(w.a.data, [1, 2]);
  assertClose(w.b.data, [3, 4, 5]);
  assertEqual(w.b.shape.join(","), "3");
});

test("parseWeights는 버퍼보다 긴 텐서를 거부한다", () => {
  const manifest = { tensors: [{ name: "a", shape: [4], offset: 0, length: 4 }] };
  let threw = false;
  try {
    parseWeights(manifest, new Float32Array(3).buffer);
  } catch {
    threw = true;
  }
  assert(threw, "오류가 나지 않음");
});

test("forward는 모든 층을 거쳐 10개의 logit을 낸다", () => {
  // 가중치가 모두 0이면 fc2.bias가 그대로 나와야 함
  const logits = forward(makeZeroWeights(), new Float32Array(784).fill(0.5));
  assertClose(logits, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
});

test("predict는 가장 확률이 높은 숫자와 신뢰도를 돌려준다", () => {
  const result = predict(makeZeroWeights(), new Float32Array(784));
  assertEqual(result.digit, 9);
  let sum = 0;
  for (let k = 0; k <= 9; k++) sum += Math.exp(k);
  assertClose(result.confidence, Math.exp(9) / sum, 1e-6);
  assertEqual(result.probabilities.length, 10);
});
