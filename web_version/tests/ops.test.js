import { test, assertEqual, assertClose } from "./harness.js";
import { conv2d3x3, relu, maxPool2x2, linear, softmax, argmax } from "../js/ops.js";

test("relu는 음수를 0으로 만든다", () => {
  assertClose(relu(Float32Array.from([-1, 0, 2])), [0, 0, 2]);
});

test("maxPool2x2는 2x2 구역의 최댓값을 고른다", () => {
  // 1채널 2x4: [[1,2,5,6],[3,4,7,8]] -> [[4,8]]
  const input = Float32Array.from([1, 2, 5, 6, 3, 4, 7, 8]);
  assertClose(maxPool2x2(input, 1, 2, 4), [4, 8]);
});

test("conv2d3x3는 padding 1로 커널 위치만큼 입력을 옮긴다", () => {
  // 커널의 왼쪽 위 칸만 1이면 out[y][x] = in[y-1][x-1] (밖은 0)
  const input = Float32Array.from([1, 2, 3, 4]); // 1채널 2x2
  const weight = Float32Array.from([1, 0, 0, 0, 0, 0, 0, 0, 0]);
  const out = conv2d3x3(input, 1, 2, 2, weight, Float32Array.from([0]), 1);
  assertClose(out, [0, 0, 0, 1]);
});

test("conv2d3x3는 입력 채널을 합하고 출력 채널별 bias를 더한다", () => {
  const input = Float32Array.from([1, 2, 3, 4, 10, 20, 30, 40]); // 2채널 2x2
  const center = (v) => [0, 0, 0, 0, v, 0, 0, 0, 0];
  const weight = Float32Array.from([
    ...center(1), ...center(1), // 출력 0: 채널0 + 채널1
    ...center(0), ...center(2), // 출력 1: 채널1 x 2
  ]);
  const out = conv2d3x3(input, 2, 2, 2, weight, Float32Array.from([0, -1]), 2);
  assertClose(out, [11, 22, 33, 44, 19, 39, 59, 79]);
});

test("linear는 W·x + b를 계산한다", () => {
  const weight = Float32Array.from([1, 2, 3, 4, 5, 6]); // [3, 2]
  const out = linear(Float32Array.from([1, 1]), weight, Float32Array.from([0.5, 0, -1]));
  assertClose(out, [3.5, 7, 10]);
});

test("softmax는 합이 1인 확률을 만든다", () => {
  assertClose(softmax(Float32Array.from([0, 0])), [0.5, 0.5]);
  const p = softmax(Float32Array.from([1, 2, 3]));
  assertClose(p[0] + p[1] + p[2], 1, 1e-6);
});

test("softmax는 큰 값에서도 NaN이 나지 않는다", () => {
  assertClose(softmax(Float32Array.from([1000, 1000])), [0.5, 0.5]);
});

test("argmax는 가장 큰 값의 위치를 돌려준다", () => {
  assertEqual(argmax(Float32Array.from([0.1, 0.7, 0.2])), 1);
});
