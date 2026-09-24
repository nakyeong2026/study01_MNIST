// CNN 추론에 필요한 기본 연산 (외부 라이브러리 없이 Float32Array로 구현)
// 텐서는 모두 PyTorch와 같은 행 우선(row-major) 순서로 펼친 1차원 배열로 다룸

/**
 * 3x3 합성곱 (stride 1, padding 1) — nn.Conv2d(kernel_size=3, padding=1)과 같음
 * input: [inC, H, W], weight: [outC, inC, 3, 3], bias: [outC] -> 반환: [outC, H, W]
 */
export function conv2d3x3(input, inC, h, w, weight, bias, outC) {
  const out = new Float32Array(outC * h * w);
  for (let oc = 0; oc < outC; oc++) {
    const outBase = oc * h * w;
    out.fill(bias[oc], outBase, outBase + h * w);
    for (let ic = 0; ic < inC; ic++) {
      const inBase = ic * h * w;
      const wBase = (oc * inC + ic) * 9;
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const k = weight[wBase + ky * 3 + kx];
          for (let y = 0; y < h; y++) {
            const iy = y + ky - 1;
            if (iy < 0 || iy >= h) continue; // 패딩 영역(0)은 건너뜀
            for (let x = 0; x < w; x++) {
              const ix = x + kx - 1;
              if (ix < 0 || ix >= w) continue;
              out[outBase + y * w + x] += k * input[inBase + iy * w + ix];
            }
          }
        }
      }
    }
  }
  return out;
}

export function relu(x) {
  return x.map((v) => (v > 0 ? v : 0));
}

/** 2x2 최대 풀링 (stride 2): [C, H, W] -> [C, H/2, W/2] */
export function maxPool2x2(input, c, h, w) {
  const oh = Math.floor(h / 2);
  const ow = Math.floor(w / 2);
  const out = new Float32Array(c * oh * ow);
  for (let ch = 0; ch < c; ch++) {
    const inBase = ch * h * w;
    for (let y = 0; y < oh; y++) {
      for (let x = 0; x < ow; x++) {
        const i = inBase + 2 * y * w + 2 * x;
        out[(ch * oh + y) * ow + x] = Math.max(input[i], input[i + 1], input[i + w], input[i + w + 1]);
      }
    }
  }
  return out;
}

/** 완전연결층 — nn.Linear와 같음. weight: [out, in], bias: [out] */
export function linear(input, weight, bias) {
  const inF = input.length;
  const out = new Float32Array(bias.length);
  for (let o = 0; o < bias.length; o++) {
    let sum = bias[o];
    const base = o * inF;
    for (let i = 0; i < inF; i++) sum += weight[base + i] * input[i];
    out[o] = sum;
  }
  return out;
}

/** 점수(logit)를 확률로 변환. 최댓값을 빼서 exp가 넘치지 않게 함 */
export function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return Float32Array.from(exps, (v) => v / sum);
}

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}
