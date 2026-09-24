// desktop_version/export_weights.py가 PyTorch로 계산해 둔 기준값과 JS 추론 결과를 비교
import { test, assert, assertEqual, assertClose } from "./harness.js";
import { loadWeights, forward, predict } from "../js/model.js";

let cache = null;

async function loadFixtures() {
  if (!cache) {
    const res = await fetch(new URL("./fixtures/reference.json", import.meta.url));
    if (!res.ok) {
      throw new Error(`reference.json이 없습니다 (HTTP ${res.status}). desktop_version/export_weights.py를 먼저 실행하세요`);
    }
    cache = {
      reference: await res.json(),
      weights: await loadWeights(new URL("../model/", import.meta.url)),
    };
  }
  return cache;
}

test("JS 순전파 결과가 PyTorch logits와 같다", async () => {
  const { reference, weights } = await loadFixtures();
  assert(reference.samples.length > 0, "샘플이 없음");
  for (const s of reference.samples) {
    assertClose(forward(weights, Float32Array.from(s.input)), s.logits, 1e-3, s.file);
  }
});

test("샘플 이미지의 정답 숫자를 맞힌다", async () => {
  const { reference, weights } = await loadFixtures();
  for (const s of reference.samples) {
    assertEqual(predict(weights, Float32Array.from(s.input)).digit, s.label, s.file);
  }
});
