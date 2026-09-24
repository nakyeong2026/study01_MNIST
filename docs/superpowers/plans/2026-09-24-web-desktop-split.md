# 손글씨 인식기 웹/데스크톱 분리 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 PyTorch 코드를 `desktop_version/`으로 옮기고, 같은 가중치로 순수 자바스크립트 추론을 하는 정적 웹 앱 `web_version/`을 새로 만들어 GitHub Pages로 배포한다.

**Architecture:** `desktop_version/export_weights.py`가 `mnist_cnn.pt`를 float32 바이너리와 JSON 매니페스트로 내보낸다. 웹 앱은 ES 모듈 4개로 나뉜다: 연산(`ops.js`), 모델(`model.js`), 전처리(`preprocess.js`), UI(`app.js`). 이 가중치를 `fetch`로 읽어 CNN 순전파를 직접 계산한다. 테스트는 라이브러리 없는 브라우저 테스트 러너에서 돌리고, PyTorch가 만든 기준값(`reference.json`)과 비교한다.

**Tech Stack:** Python 3.12 + PyTorch/torchvision/Pillow/numpy(데스크톱), 바닐라 JS ES 모듈 + Canvas 2D + Pointer Events(웹), GitHub Actions Pages 배포.

**Spec:** `docs/superpowers/specs/2026-09-24-web-desktop-split-design.md`

## Global Constraints

- 모든 코드 주석, UI 문구, 사용자에게 보이는 메시지는 한국어로 쓴다. `CLAUDE.md` 파일은 기존처럼 영어로 쓴다.
- `web_version/`에는 외부 라이브러리, CDN, npm, 번들러, 빌드 단계를 두지 않는다.
- 웹 코드는 `<script type="module">`과 `fetch`를 쓰므로 반드시 HTTP로 연다: `"$PY" -m http.server 8000 --directory web_version` (저장소 루트에서 실행).
- Python 경로(PATH에 없음). Git Bash에서는 `PY="/c/Users/$USERNAME/AppData/Local/Programs/Python/Python312/python.exe"`로 쓴다.
- MNIST 정규화 상수: mean `0.1307`, std `0.3081`. 입력 크기 28×28. 그리기 캔버스 280×280, 펜 굵기 18.
- 모델 구조는 바꾸지 않는다: conv1(1→32, 3×3, pad 1) → ReLU → MaxPool2 → conv2(32→64, 3×3, pad 1) → ReLU → MaxPool2 → flatten(3136) → fc1(128) → ReLU → fc2(10).
- 가중치 텐서 이름: `conv1.weight [32,1,3,3]`, `conv1.bias [32]`, `conv2.weight [64,32,3,3]`, `conv2.bias [64]`, `fc1.weight [128,3136]`, `fc1.bias [128]`, `fc2.weight [10,128]`, `fc2.bias [10]`.
- 브라우저 테스트가 통과하면 `tests/test.html`의 `document.title`이 `PASS n/n`이 된다. 결과가 이전 것 그대로 보이면 캐시 때문이니 강력 새로고침(Ctrl+Shift+R)한다.
- 기본 브랜치는 `master`이고 원격은 `origin`(https://github.com/nakyeong2026/study01_MNIST.git)이다. 푸시는 사용자가 요청할 때만 한다.

## 파일 구조

| 경로 | 작업 | 책임 |
|---|---|---|
| `desktop_version/{model,train,predict,app}.py`, `app.bat`, `mnist_cnn.pt`, `sample_*.png` | 이동(`git mv`) | 기존 데스크톱 앱 |
| `desktop_version/{train,predict,app}.py` | 수정 | 경로를 스크립트 위치 기준으로 바꿈 |
| `desktop_version/export_weights.py` | 생성 | `.pt` → `web_version/model/` + 테스트 픽스처 |
| `desktop_version/CLAUDE.md` | 생성 | 데스크톱 안내 |
| `web_version/js/ops.js` | 생성 | 신경망 기본 연산 (순수 함수) |
| `web_version/js/model.js` | 생성 | 가중치 파싱/로딩, 순전파, 예측 |
| `web_version/js/preprocess.js` | 생성 | 픽셀 → 정규화된 784개 입력 |
| `web_version/js/app.js`, `index.html`, `style.css` | 생성 | UI |
| `web_version/model/weights.{bin,json}` | 생성(스크립트 출력) | 학습된 가중치 |
| `web_version/tests/{test.html,harness.js,run-tests.js,*.test.js}` | 생성 | 브라우저 테스트 |
| `web_version/tests/fixtures/*` | 생성(스크립트 출력) | PyTorch 기준값과 샘플 PNG |
| `web_version/CLAUDE.md` | 생성 | 웹 안내 |
| `.github/workflows/pages.yml` | 생성 | Pages 배포 |
| `CLAUDE.md` (루트) | 다시 씀 | 두 버전 개요 |

---

### Task 1: 데스크톱 코드를 `desktop_version/`으로 이동

**Files:**
- Move: `model.py`, `train.py`, `predict.py`, `app.py`, `app.bat`, `mnist_cnn.pt`, `sample_*.png` → `desktop_version/`
- Modify: `desktop_version/train.py`, `desktop_version/predict.py`, `desktop_version/app.py`
- Create: `desktop_version/CLAUDE.md`

**Interfaces:**
- Produces: `desktop_version/predict.py`의 `preprocess_image(image_path) -> torch.Tensor (1,1,28,28)`와 `desktop_version/model.py`의 `MnistCNN`(Task 4가 import). 각 스크립트는 모듈 상수 `WEIGHTS_PATH`(`pathlib.Path`)를 가진다.

- [ ] **Step 1: 파일 이동**

```bash
cd /c/logistex/study01_MNIST
mkdir desktop_version
git mv model.py train.py predict.py app.py app.bat mnist_cnn.pt sample_0_label_7.png sample_1_label_2.png sample_2_label_1.png sample_3_label_0.png sample_4_label_4.png desktop_version/
mv data desktop_version/data
rm -rf __pycache__
```

- [ ] **Step 2: 실패 확인 — 루트에서 실행하면 가중치 파일을 못 찾는다**

```bash
PY="/c/Users/$USERNAME/AppData/Local/Programs/Python/Python312/python.exe"
"$PY" desktop_version/predict.py desktop_version/sample_0_label_7.png
```
Expected: `FileNotFoundError` ... `mnist_cnn.pt`

- [ ] **Step 3: `predict.py` 경로 수정**

`import sys` 아래 import 목록에 `from pathlib import Path`를 추가하고, `WEIGHTS_PATH = "mnist_cnn.pt"` 줄을 다음으로 바꾼다:

```python
# 스크립트 파일 위치 기준 경로 (어느 폴더에서 실행해도 동작하도록)
WEIGHTS_PATH = Path(__file__).resolve().parent / "mnist_cnn.pt"
```

- [ ] **Step 4: `train.py` 경로 수정**

import 목록 맨 위에 `from pathlib import Path`를 추가하고, `WEIGHTS_PATH = "mnist_cnn.pt"` 줄을 다음으로 바꾼다:

```python
# 스크립트 파일 위치 기준 경로 (어느 폴더에서 실행해도 동작하도록)
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "mnist_cnn.pt"
DATA_DIR = BASE_DIR / "data"
```

`get_data_loaders()` 안의 `root="./data"` 두 곳을 모두 `root=DATA_DIR`로 바꾼다.

- [ ] **Step 5: `app.py` 경로 수정**

`import sys` 아래에 `from pathlib import Path`를 추가하고, `WEIGHTS_PATH = "mnist_cnn.pt"` 줄을 다음으로 바꾼다 (`os.path.exists(WEIGHTS_PATH)`는 `Path`도 받으므로 그대로 둔다):

```python
# 스크립트 파일 위치 기준 경로 (app.bat이나 다른 폴더에서 실행해도 동작하도록)
WEIGHTS_PATH = Path(__file__).resolve().parent / "mnist_cnn.pt"
```

- [ ] **Step 6: 통과 확인**

```bash
"$PY" desktop_version/predict.py desktop_version/sample_0_label_7.png
```
Expected: 첫 줄이 `예측된 숫자: 7`

```bash
"$PY" -c "import sys; sys.path.insert(0, 'desktop_version'); import train, app; print(train.DATA_DIR.exists(), app.WEIGHTS_PATH.exists()); train.get_data_loaders(); print('ok')"
```
Expected: `True True` 다음에 `ok`가 나오고, 그 사이에 다운로드 메시지가 없음 (옮긴 `data/`를 그대로 사용한다는 뜻)

- [ ] **Step 7: `desktop_version/CLAUDE.md` 작성**

````markdown
# CLAUDE.md — desktop_version

Guidance for Claude Code when working in `desktop_version/`. See the root [CLAUDE.md](../CLAUDE.md) for how this relates to `web_version/`.

## What this is

The original PyTorch project: trains a CNN on MNIST and lets a user try it two ways — by pointing at an image file (`predict.py`) or by drawing with the mouse in a Tkinter GUI (`app.py`). It is also the source of the weights the web version uses (`export_weights.py`). All code and comments are written in Korean.

## Environment

- No Python interpreter is on `PATH`. Use `C:\Users\<username>\AppData\Local\Programs\Python\Python312\python.exe` (installed via `winget install -e --id Python.Python.3.12 --scope user`).
- Dependencies (pip, no `requirements.txt`): `torch`, `torchvision` (CPU build, from `https://download.pytorch.org/whl/cpu`), `pillow`, `numpy`.

## Commands

All scripts resolve `mnist_cnn.pt` and `data/` relative to their own file, so they can be run from any working directory.

```
# Train and (re)write mnist_cnn.pt — downloads MNIST into desktop_version/data on first run
python train.py

# Recognize a digit from an image file
python predict.py <이미지_경로>

# Launch the mouse-drawing GUI (blocks in a Tk mainloop)
python app.py

# Export mnist_cnn.pt for the web version (weights + test fixtures) — rerun after every retrain
python export_weights.py
```

`app.bat` double-click-launches `app.py` via `pythonw.exe` with no console window. It resolves the interpreter via `%USERNAME%` and hardcodes the `Python312` install path.

There is no lint config or test suite; verify changes by running `predict.py` on the `sample_*_label_N.png` files (the filename holds the correct digit).

## Architecture

- `model.py` — `MnistCNN`, the only model definition. `mnist_cnn.pt` is a raw `state_dict` with no architecture metadata, so changing layer shapes invalidates it and requires retraining. The web version reimplements this architecture by hand in `../web_version/js/model.js` — keep both in sync.
- `train.py` — downloads MNIST into `data/`, trains 5 epochs, overwrites `mnist_cnn.pt`.
- `predict.py` — CLI. Preprocessing auto-inverts images whose mean pixel is bright (>127), since MNIST is white-on-black; then pads/resizes to 28×28 and normalizes (mean 0.1307, std 0.3081).
- `app.py` — Tkinter GUI. Draws to the visible `Canvas` and, in parallel, to an in-memory `PIL.Image` at every drag event, so the 28×28 input is built without screen capture. Exits with a message if `mnist_cnn.pt` doesn't exist.
- `export_weights.py` — writes `../web_version/model/weights.bin` (all tensors as little-endian float32, concatenated in `state_dict` order) and `weights.json` (name, shape, offset, length in floats), plus `../web_version/tests/fixtures/` (sample PNGs and `reference.json` with PyTorch inputs/logits the web tests compare against).
- `train.py`, `predict.py`, `app.py` each build the same `Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])`, and `../web_version/js/preprocess.js` mirrors it — keep all of them in sync if preprocessing changes.
````

- [ ] **Step 8: 커밋**

```bash
git add -A desktop_version
git commit -m "refactor: move desktop app into desktop_version/ and resolve paths from script location"
```
(루트 `CLAUDE.md`는 Task 7에서 다시 쓴다.)

---

### Task 2: 브라우저 테스트 러너와 `ops.js`

**Files:**
- Create: `web_version/tests/test.html`, `web_version/tests/harness.js`, `web_version/tests/run-tests.js`, `web_version/tests/ops.test.js`, `web_version/js/ops.js`
- Create (gitignore 대상, 커밋 안 함): `.claude/launch.json`

**Interfaces:**
- Produces (`harness.js`): `test(name: string, fn: () => void|Promise<void>)`, `assert(cond, msg?)`, `assertEqual(actual, expected, msg?)`, `assertClose(actual: number|ArrayLike<number>, expected: number|ArrayLike<number>, tol = 1e-6, label = "")`, `runAll(listEl: HTMLElement): Promise<void>`
- Produces (`ops.js`, 모든 텐서는 행 우선 `Float32Array`):
  - `conv2d3x3(input [inC,H,W], inC, h, w, weight [outC,inC,3,3], bias [outC], outC) -> Float32Array [outC,H,W]` (stride 1, padding 1)
  - `relu(x) -> Float32Array`
  - `maxPool2x2(input [C,H,W], c, h, w) -> Float32Array [C,floor(H/2),floor(W/2)]`
  - `linear(input [in], weight [out,in], bias [out]) -> Float32Array [out]`
  - `softmax(logits) -> Float32Array`
  - `argmax(values) -> number`

- [ ] **Step 1: 개발 서버 설정 파일 작성** (`.claude/launch.json`, 브라우저 미리보기용)

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "web_version",
      "runtimeExecutable": "C:\\Users\\안나경\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
      "runtimeArgs": ["-m", "http.server", "8000", "--directory", "web_version"],
      "port": 8000
    }
  ]
}
```

- [ ] **Step 2: 테스트 하네스 작성** — `web_version/tests/harness.js`

```js
// 외부 라이브러리 없이 브라우저에서 돌리는 아주 작은 테스트 하네스

const tests = [];

/** 테스트 등록 (비동기 함수도 가능) */
export function test(name, fn) {
  tests.push({ name, fn });
}

export function assert(condition, message = "assert 실패") {
  if (!condition) throw new Error(message);
}

export function assertEqual(actual, expected, message = "") {
  if (actual !== expected) {
    throw new Error(`${message} 기대값 ${expected}, 실제값 ${actual}`);
  }
}

/** 숫자 또는 숫자 배열이 허용 오차 안에서 같은지 확인 */
export function assertClose(actual, expected, tol = 1e-6, label = "") {
  const a = typeof actual === "number" ? [actual] : Array.from(actual);
  const e = typeof expected === "number" ? [expected] : Array.from(expected);
  if (a.length !== e.length) {
    throw new Error(`${label} 길이가 다름: ${a.length} != ${e.length}`);
  }
  for (let i = 0; i < a.length; i++) {
    // NaN도 실패로 잡기 위해 !(차이 <= 오차) 형태로 비교
    if (!(Math.abs(a[i] - e[i]) <= tol)) {
      throw new Error(`${label} [${i}] ${a[i]} != ${e[i]} (허용 오차 ${tol})`);
    }
  }
}

/** 등록된 테스트를 모두 실행하고 결과를 목록과 문서 제목에 표시 */
export async function runAll(listEl) {
  let passed = 0;
  for (const { name, fn } of tests) {
    const item = document.createElement("li");
    try {
      await fn();
      passed++;
      item.textContent = `PASS  ${name}`;
      item.className = "pass";
    } catch (err) {
      item.textContent = `FAIL  ${name} — ${err.message}`;
      item.className = "fail";
      console.error(name, err);
    }
    listEl.appendChild(item);
  }
  const summary = `${passed === tests.length ? "PASS" : "FAIL"} ${passed}/${tests.length}`;
  document.title = summary;
  document.getElementById("summary").textContent = summary;
}
```

- [ ] **Step 3: 테스트 페이지와 실행 스크립트 작성**

`web_version/tests/test.html`:

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>테스트 실행 중…</title>
  <style>
    body { font-family: ui-monospace, Consolas, monospace; margin: 16px; }
    .pass { color: #1a7f37; }
    .fail { color: #cf222e; }
  </style>
</head>
<body>
  <h1>web_version 테스트</h1>
  <p id="summary">실행 중…</p>
  <ul id="results"></ul>
  <script type="module" src="./run-tests.js"></script>
</body>
</html>
```

`web_version/tests/run-tests.js`:

```js
// 테스트 파일을 모두 불러온 뒤(import 시 test()로 등록됨) 한꺼번에 실행
import { runAll } from "./harness.js";
import "./ops.test.js";

runAll(document.getElementById("results"));
```

- [ ] **Step 4: 실패하는 연산 테스트 작성** — `web_version/tests/ops.test.js`

```js
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
```

- [ ] **Step 5: 실패 확인**

서버를 켠다 (백그라운드로 두고 이후 Task에서도 계속 사용):
```bash
"$PY" -m http.server 8000 --directory web_version
```
브라우저로 `http://localhost:8000/tests/test.html`을 연다.
Expected: 제목이 `테스트 실행 중…`에서 바뀌지 않고, 콘솔에 `js/ops.js` 404 오류 (모듈을 불러오지 못해 실패)

- [ ] **Step 6: `web_version/js/ops.js` 구현**

```js
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
```

- [ ] **Step 7: 통과 확인**

`http://localhost:8000/tests/test.html`을 새로고침한다.
Expected: 제목 `PASS 8/8`

- [ ] **Step 8: 커밋**

```bash
git add web_version/tests web_version/js/ops.js
git commit -m "feat(web): add browser test harness and pure-JS CNN ops"
```

---

### Task 3: `model.js` — 가중치 파싱과 순전파

**Files:**
- Create: `web_version/js/model.js`, `web_version/tests/model.test.js`
- Modify: `web_version/tests/run-tests.js`

**Interfaces:**
- Consumes: `ops.js`의 `conv2d3x3`, `relu`, `maxPool2x2`, `linear`, `softmax`, `argmax` (Task 2)
- Produces:
  - `Weights` = `{ [name: string]: { shape: number[], data: Float32Array } }`
  - `parseWeights(manifest: {tensors: {name, shape, offset, length}[]}, buffer: ArrayBuffer) -> Weights` (`offset`/`length`는 float 개수 단위). 범위를 넘으면 `Error`
  - `loadWeights(baseUrl: URL|string) -> Promise<Weights>` (`baseUrl`는 `/`로 끝나는 폴더 URL이고, 그 안의 `weights.json`/`weights.bin`을 읽음. HTTP 오류면 `Error`)
  - `forward(weights, input: Float32Array(784)) -> Float32Array(10)` (logits)
  - `predict(weights, input) -> { digit: number, confidence: number, probabilities: Float32Array(10) }`

- [ ] **Step 1: 실패하는 테스트 작성** — `web_version/tests/model.test.js`

```js
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
```

`web_version/tests/run-tests.js`의 `import "./ops.test.js";` 아래에 추가한다:

```js
import "./model.test.js";
```

- [ ] **Step 2: 실패 확인**

`http://localhost:8000/tests/test.html`을 새로고침한다.
Expected: 제목이 `테스트 실행 중…` 그대로이고, 콘솔에 `js/model.js` 404

- [ ] **Step 3: `web_version/js/model.js` 구현**

```js
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
```

- [ ] **Step 4: 통과 확인**

새로고침한다.
Expected: 제목 `PASS 12/12`

- [ ] **Step 5: 커밋**

```bash
git add web_version/js/model.js web_version/tests/model.test.js web_version/tests/run-tests.js
git commit -m "feat(web): add weight parsing and MnistCNN forward pass in JS"
```

---

### Task 4: 가중치 내보내기와 PyTorch 일치 테스트

**Files:**
- Create: `desktop_version/export_weights.py`, `web_version/tests/parity.test.js`
- Create (스크립트 출력): `web_version/model/weights.bin`, `web_version/model/weights.json`, `web_version/tests/fixtures/reference.json`, `web_version/tests/fixtures/sample_*.png`
- Modify: `web_version/tests/run-tests.js`

**Interfaces:**
- Consumes: `desktop_version/model.py` `MnistCNN`, `desktop_version/predict.py` `preprocess_image` (Task 1). `model.js`의 `loadWeights`, `forward`, `predict` (Task 3)
- Produces:
  - `weights.json`: `{"dtype": "float32", "byteOrder": "little", "tensors": [{"name", "shape", "offset", "length"}, ...]}`
  - `reference.json`: `{"samples": [{"file": "sample_0_label_7.png", "label": 7, "input": [784 floats], "logits": [10 floats]}, ...]}` (Task 5도 사용)

- [ ] **Step 1: 실패하는 일치 테스트 작성** — `web_version/tests/parity.test.js`

```js
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
```

`run-tests.js`에 `import "./model.test.js";` 아래로 추가한다:

```js
import "./parity.test.js";
```

- [ ] **Step 2: 실패 확인**

새로고침한다.
Expected: 제목 `FAIL 12/14`이고, 두 테스트가 `reference.json이 없습니다 (HTTP 404)`로 실패

- [ ] **Step 3: `desktop_version/export_weights.py` 구현**

```python
# -*- coding: utf-8 -*-
"""학습된 mnist_cnn.pt 가중치를 웹 버전(web_version)이 읽을 수 있는 형식으로 내보내는 스크립트

사용법:
    python export_weights.py
    (train.py로 다시 학습했다면 매번 다시 실행해야 웹 버전에 반영됨)

만드는 파일:
    ../web_version/model/weights.bin      모든 텐서를 리틀엔디언 float32로 이어붙인 바이너리
    ../web_version/model/weights.json     텐서별 이름, 모양(shape), 시작 위치(offset), 길이(float 개수)
    ../web_version/tests/fixtures/        샘플 PNG와 PyTorch 계산 결과(reference.json) — 웹 테스트의 기준값
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from model import MnistCNN
from predict import preprocess_image

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "mnist_cnn.pt"
WEB_DIR = BASE_DIR.parent / "web_version"
MODEL_OUT_DIR = WEB_DIR / "model"
FIXTURE_OUT_DIR = WEB_DIR / "tests" / "fixtures"


def export_weights(state_dict):
    """state_dict를 weights.bin + weights.json으로 저장"""

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    tensors = []
    chunks = []
    offset = 0
    for name, tensor in state_dict.items():
        array = tensor.detach().cpu().numpy().astype("<f4").ravel()
        tensors.append({
            "name": name,
            "shape": list(tensor.shape),
            "offset": offset,
            "length": int(array.size),
        })
        chunks.append(array)
        offset += array.size

    (MODEL_OUT_DIR / "weights.bin").write_bytes(np.concatenate(chunks).tobytes())
    manifest = {"dtype": "float32", "byteOrder": "little", "tensors": tensors}
    (MODEL_OUT_DIR / "weights.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"가중치 {offset}개를 '{MODEL_OUT_DIR}'에 저장했습니다.")


def export_fixtures(model):
    """샘플 이미지마다 PyTorch 전처리 입력과 logits를 기록 (웹 테스트가 비교할 기준값)"""

    FIXTURE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = []
    for image_path in sorted(BASE_DIR.glob("sample_*_label_*.png")):
        label = int(image_path.stem.split("_label_")[1])
        input_tensor = preprocess_image(image_path)
        with torch.no_grad():
            logits = model(input_tensor).squeeze(0)

        samples.append({
            "file": image_path.name,
            "label": label,
            "input": input_tensor.flatten().tolist(),
            "logits": logits.tolist(),
        })
        shutil.copy(image_path, FIXTURE_OUT_DIR / image_path.name)

    (FIXTURE_OUT_DIR / "reference.json").write_text(json.dumps({"samples": samples}), encoding="utf-8")
    print(f"테스트 기준값 {len(samples)}개를 '{FIXTURE_OUT_DIR}'에 저장했습니다.")


def main():
    model = MnistCNN()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    export_weights(model.state_dict())
    export_fixtures(model)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 스크립트 실행**

```bash
"$PY" desktop_version/export_weights.py
ls -l web_version/model web_version/tests/fixtures
```
Expected: `가중치 421642개를 ...`와 `테스트 기준값 5개를 ...`가 출력됨. `weights.bin`은 1686568바이트(421642 × 4), 픽스처 폴더에 PNG 5개와 `reference.json`

- [ ] **Step 5: 통과 확인**

새로고침한다.
Expected: 제목 `PASS 14/14`

- [ ] **Step 6: 커밋**

```bash
git add desktop_version/export_weights.py web_version/model web_version/tests
git commit -m "feat: export PyTorch weights for the web version and verify JS parity"
```

---

### Task 5: `preprocess.js` — 그림과 사진을 모델 입력으로 변환

**Files:**
- Create: `web_version/js/preprocess.js`, `web_version/tests/preprocess.test.js`
- Modify: `web_version/tests/run-tests.js`

**Interfaces:**
- Consumes: `tests/fixtures/reference.json`과 샘플 PNG (Task 4)
- Produces:
  - `IMAGE_SIZE = 28`
  - `toGrayscale(rgba: Uint8ClampedArray|ArrayLike, width, height) -> Float32Array` (0~255, PIL `convert("L")` 공식, 알파 무시)
  - `resizeArea(src: Float32Array, srcW, srcH, dstW, dstH) -> Float32Array` (면적 평균)
  - `padAndResize(gray, width, height, size) -> Float32Array(size*size)` (`ImageOps.pad`와 같은 방식)
  - `normalize(pixels0to255) -> Float32Array`
  - `preprocessDrawing(rgba, width, height) -> { pixels: Float32Array(784) /* 0~255 */, input: Float32Array(784) /* 정규화됨 */ }`
  - `preprocessPhoto(rgba, width, height) -> { pixels, input }` (같은 모양)
  - `decodeImage(blob: Blob) -> Promise<ImageData>` (읽을 수 없는 파일이면 reject)

- [ ] **Step 1: 실패하는 테스트 작성** — `web_version/tests/preprocess.test.js`

```js
import { test, assertClose } from "./harness.js";
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
```

`run-tests.js`에 `import "./parity.test.js";` 아래로 추가한다:

```js
import "./preprocess.test.js";
```

- [ ] **Step 2: 실패 확인**

새로고침한다.
Expected: 제목이 `테스트 실행 중…` 그대로이고, 콘솔에 `js/preprocess.js` 404

- [ ] **Step 3: `web_version/js/preprocess.js` 구현**

```js
// 캔버스 그림이나 사진을 모델 입력(정규화된 28x28 = 784개 값)으로 바꾸는 전처리
// desktop_version의 app.py(그리기)와 predict.py(이미지 파일) 전처리를 따름

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

/** 면적 평균으로 크기 변경 (가로 방향 후 세로 방향) */
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
```

- [ ] **Step 4: 통과 확인**

새로고침한다.
Expected: 제목 `PASS 23/23`

- [ ] **Step 5: 커밋**

```bash
git add web_version/js/preprocess.js web_version/tests/preprocess.test.js web_version/tests/run-tests.js
git commit -m "feat(web): add drawing/photo preprocessing matching the desktop pipeline"
```

---

### Task 6: 웹 UI (`index.html`, `style.css`, `app.js`)

**Files:**
- Create: `web_version/index.html`, `web_version/style.css`, `web_version/js/app.js`

**Interfaces:**
- Consumes: `model.js`의 `loadWeights`, `predict` (Task 3). `preprocess.js`의 `preprocessDrawing`, `preprocessPhoto`, `decodeImage`, `IMAGE_SIZE` (Task 5)
- Produces: 없음 (최종 화면)

UI는 DOM 연결 코드라 단위 테스트 대신 브라우저에서 직접 조작해 확인한다 (Step 4).

- [ ] **Step 1: `web_version/index.html` 작성**

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>손글씨 숫자 인식기</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main>
    <h1>손글씨 숫자 인식기</h1>
    <p id="status" class="status">모델을 불러오는 중…</p>

    <div class="layout">
      <section class="draw-panel">
        <canvas id="draw-canvas" width="280" height="280" aria-label="숫자를 그리는 캔버스"></canvas>
        <div class="controls">
          <button id="clear-button" type="button">지우기</button>
          <label class="file-button">
            이미지 올리기
            <input id="file-input" type="file" accept="image/*" disabled>
          </label>
        </div>
      </section>

      <section class="result-panel">
        <div class="result">
          <span id="result-digit" class="digit">?</span>
          <span id="result-confidence" class="confidence">숫자를 그려 보세요</span>
        </div>
        <div id="prob-bars" class="bars"></div>
        <figure class="preview">
          <canvas id="input-preview" width="28" height="28"></canvas>
          <figcaption>모델 입력 (28×28)</figcaption>
        </figure>
      </section>
    </div>

    <p class="note">모든 계산은 브라우저 안에서 순수 자바스크립트로 이루어지며, 그림이나 이미지는 서버로 보내지 않습니다.</p>
  </main>
  <script type="module" src="js/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: `web_version/style.css` 작성**

```css
:root {
  --bg: #f6f7f9;
  --surface: #ffffff;
  --text: #1f2328;
  --muted: #656d76;
  --accent: #2f6feb;
  --bar-bg: #e6e8eb;
  --border: #d0d7de;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --text: #e6edf3;
    --muted: #8d96a0;
    --accent: #4c8dff;
    --bar-bg: #262c36;
    --border: #30363d;
  }
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
}

main {
  max-width: 720px;
  margin: 0 auto;
  padding: 24px 16px;
}

h1 { margin: 0 0 4px; font-size: 1.6rem; }

.status { margin: 0 0 20px; color: var(--muted); }

.layout {
  display: flex;
  flex-wrap: wrap;
  gap: 24px;
}

.draw-panel { flex: 0 1 280px; }

#draw-canvas {
  display: block;
  width: 100%;
  max-width: 280px;
  aspect-ratio: 1;
  background: #000;
  border-radius: 8px;
  cursor: crosshair;
  touch-action: none; /* 터치로 그릴 때 화면이 스크롤되지 않도록 */
}

.controls {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

button,
.file-button {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  font: inherit;
  text-align: center;
  cursor: pointer;
}

.file-button input { display: none; }

.file-button:has(input:disabled) { opacity: 0.5; cursor: not-allowed; }

.result-panel { flex: 1 1 260px; }

.result {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 12px;
}

.digit { font-size: 3.5rem; font-weight: 700; line-height: 1; }

.confidence { color: var(--muted); }

.bars { display: grid; gap: 4px; }

.bar-row {
  display: grid;
  grid-template-columns: 1.2em 1fr 3.5em;
  align-items: center;
  gap: 8px;
  font-variant-numeric: tabular-nums;
}

.bar-track {
  height: 10px;
  background: var(--bar-bg);
  border-radius: 5px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  width: 0;
  background: var(--accent);
  transition: width 0.15s;
}

.bar-value { text-align: right; color: var(--muted); font-size: 0.85rem; }

.preview { margin: 16px 0 0; }

#input-preview {
  width: 84px;
  height: 84px;
  image-rendering: pixelated;
  border: 1px solid var(--border);
  background: #000;
}

figcaption { color: var(--muted); font-size: 0.85rem; }

.note { margin-top: 24px; color: var(--muted); font-size: 0.85rem; }
```

- [ ] **Step 3: `web_version/js/app.js` 작성**

```js
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
```

- [ ] **Step 4: 브라우저에서 직접 확인**

`http://localhost:8000/`을 열고 다음을 확인한다:
1. 상태 문구가 `숫자를 그리거나 이미지를 올려 보세요`로 바뀐다 (가중치 로드 성공). 콘솔 오류가 없다.
2. 캔버스에 마우스로 세로선(숫자 1)을 긋고 손을 떼면 큰 숫자가 `1`이 되고, 확률 막대와 28×28 미리보기가 갱신된다.
3. `지우기`를 누르면 캔버스가 검게 되고 결과가 `?`로 돌아간다.
4. `이미지 올리기`로 `desktop_version/sample_0_label_7.png`를 고르면 `7`로 인식되고 상태 문구가 `"sample_0_label_7.png" 인식 결과`가 된다. (자동화 도구로 파일 선택이 어려우면 콘솔에서 `fetch('tests/fixtures/sample_0_label_7.png')`로 받은 Blob을 `DataTransfer`에 넣어 `fileInput.files`에 설정하고 `change` 이벤트를 발생시킨다.)
5. 창 너비를 375px로 줄여도 가로 스크롤 없이 캔버스와 결과가 세로로 쌓인다.
6. `tests/test.html`이 여전히 `PASS 23/23`이다.

- [ ] **Step 5: 커밋**

```bash
git add web_version/index.html web_version/style.css web_version/js/app.js
git commit -m "feat(web): add drawing/upload UI for in-browser digit recognition"
```

---

### Task 7: GitHub Pages 배포 워크플로와 CLAUDE.md 정리

**Files:**
- Create: `.github/workflows/pages.yml`, `web_version/CLAUDE.md`
- Modify (다시 씀): `CLAUDE.md`

**Interfaces:**
- Consumes: 완성된 `web_version/` (Task 2~6)
- Produces: `master` 브랜치에 `web_version/`의 변경이 푸시되면 Pages로 배포되는 워크플로

- [ ] **Step 1: `.github/workflows/pages.yml` 작성**

```yaml
# web_version/ 폴더만 GitHub Pages로 배포 (저장소 Settings > Pages > Source를 "GitHub Actions"로 설정해야 함)
name: Deploy web_version to GitHub Pages

on:
  push:
    branches: [master]
    paths:
      - "web_version/**"
      - ".github/workflows/pages.yml"
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: web_version
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: 워크플로 YAML 문법 확인**

```bash
"$PY" -c "import yaml" 2>/dev/null && "$PY" -c "import yaml; d = yaml.safe_load(open('.github/workflows/pages.yml', encoding='utf-8')); print(d['jobs']['deploy']['steps'][2]['with']['path'])" || echo "pyyaml 없음 - 눈으로 들여쓰기 확인"
```
Expected: `web_version` (또는 pyyaml이 없다는 안내. 이 경우 들여쓰기를 직접 확인)

- [ ] **Step 3: `web_version/CLAUDE.md` 작성**

````markdown
# CLAUDE.md — web_version

Guidance for Claude Code when working in `web_version/`. See the root [CLAUDE.md](../CLAUDE.md) for how this relates to `desktop_version/`.

## What this is

A static web page that recognizes handwritten digits entirely in the browser: the user draws on a canvas (mouse or touch) or uploads an image, and a hand-written pure-JavaScript implementation of the same CNN as `../desktop_version/model.py` runs inference. No libraries, CDNs, npm, bundler, or build step — every file here is served as-is. Deployed to GitHub Pages by `../.github/workflows/pages.yml` (repo Settings → Pages → Source must be "GitHub Actions"). All code comments and UI text are in Korean.

## Commands

The page uses ES modules and `fetch`, so it does not work from `file://`. Serve it over HTTP from the repo root (Python is not on `PATH`; see `../desktop_version/CLAUDE.md` for the interpreter path):

```
python -m http.server 8000 --directory web_version
```

- App: `http://localhost:8000/`
- Tests: `http://localhost:8000/tests/test.html` — the page title becomes `PASS n/n` or `FAIL k/n`. Hard-reload (Ctrl+Shift+R) if results look stale.

## Architecture

- `js/ops.js` — pure tensor ops on row-major `Float32Array`s: `conv2d3x3` (stride 1, padding 1), `relu`, `maxPool2x2`, `linear`, `softmax`, `argmax`.
- `js/model.js` — `parseWeights`/`loadWeights` read `model/weights.json` + `model/weights.bin`; `forward` hard-codes the `MnistCNN` layer sequence (dropout omitted, it is a no-op at inference); `predict` returns `{digit, confidence, probabilities}`. If `../desktop_version/model.py` changes, this file must change with it.
- `js/preprocess.js` — mirrors the desktop pipelines: `preprocessDrawing` (280×280 canvas → area-average to 28×28, like `app.py`) and `preprocessPhoto` (PIL-`convert("L")` grayscale → invert if mean > 127 → aspect-preserving pad to 28×28, like `predict.py`), both then normalized with mean 0.1307 / std 0.3081.
- `js/app.js` — DOM wiring: Pointer Events drawing (pen width 18, recognizes on stroke end), file upload, probability bars, 28×28 input preview.
- `model/` — **generated, do not edit by hand.** Regenerate with `python ../desktop_version/export_weights.py` after every retrain; it also rewrites `tests/fixtures/`.
- `tests/` — tiny in-house harness (`harness.js`); `run-tests.js` imports every `*.test.js`, so add an import there for a new test file. `parity.test.js` checks JS logits against PyTorch's (`fixtures/reference.json`) and `preprocess.test.js` checks decoded sample PNGs against PyTorch's preprocessed inputs — these are the guards that the JS port matches the Python model.
````

- [ ] **Step 4: 루트 `CLAUDE.md` 다시 쓰기**

````markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A handwritten digit (MNIST) recognizer, split into two independent versions that share one trained model:

- [desktop_version/](desktop_version/) — the original PyTorch project: training (`train.py`), image-file CLI (`predict.py`), Tkinter drawing GUI (`app.py`), and `export_weights.py`, which converts the trained `mnist_cnn.pt` for the web version. See [desktop_version/CLAUDE.md](desktop_version/CLAUDE.md).
- [web_version/](web_version/) — a static page that runs the same CNN in pure JavaScript (no libraries, no build step), deployed to GitHub Pages. See [web_version/CLAUDE.md](web_version/CLAUDE.md).

All code, comments, and UI text are written in Korean.

## How the two versions connect

The web version never runs Python. Its weights (`web_version/model/`) and test fixtures (`web_version/tests/fixtures/`) are generated files committed to git:

```
desktop_version/train.py          -> desktop_version/mnist_cnn.pt
desktop_version/export_weights.py -> web_version/model/weights.{bin,json}
                                   + web_version/tests/fixtures/{reference.json, sample_*.png}
```

So a retrain must be followed by `export_weights.py` and a run of the web tests. The model architecture (`desktop_version/model.py` ↔ `web_version/js/model.js`) and the preprocessing (`train.py`/`predict.py`/`app.py` ↔ `web_version/js/preprocess.js`) are duplicated across languages and must be kept in sync by hand.

## Environment

- No Python interpreter is on `PATH`; use `C:\Users\<username>\AppData\Local\Programs\Python\Python312\python.exe`. Dependencies (no `requirements.txt`): `torch`, `torchvision` (CPU build), `pillow`, `numpy`.
- No Node.js; web tests run in a browser (`web_version/tests/test.html`).

## Deployment

`.github/workflows/pages.yml` deploys the `web_version/` folder to GitHub Pages on pushes to `master` that touch it (or via manual dispatch). The repository's Pages source must be set to "GitHub Actions".
````

- [ ] **Step 5: 최종 확인**

```bash
git status --short
"$PY" desktop_version/predict.py desktop_version/sample_3_label_0.png
```
Expected: `git status`에는 `.github/`, `web_version/CLAUDE.md`, `CLAUDE.md`만 보임. predict 결과 첫 줄은 `예측된 숫자: 0`
브라우저에서 `http://localhost:8000/tests/test.html`이 `PASS 23/23`

- [ ] **Step 6: 커밋**

```bash
git add .github/workflows/pages.yml web_version/CLAUDE.md CLAUDE.md
git commit -m "docs: add Pages deploy workflow and per-version CLAUDE.md files"
```

(푸시와 Pages 설정은 사용자에게 맡긴다: `git push origin master` 후 저장소 Settings > Pages > Source = GitHub Actions)
