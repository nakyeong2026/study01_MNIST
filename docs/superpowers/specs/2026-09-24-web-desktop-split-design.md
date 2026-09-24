# 손글씨 인식기 웹/데스크톱 분리 설계

## 목표

기존 PyTorch 손글씨 숫자 인식기를 두 버전으로 나눈다.

- `desktop_version/`: 기존 코드(학습, CLI 예측, Tkinter GUI)를 그대로 옮긴 것
- `web_version/`: 외부 라이브러리 없이 순수 자바스크립트로 추론하는 정적 웹 앱. GitHub Pages에 배포

각 폴더에는 해당 버전 전용 `CLAUDE.md`를 두고, 루트 `CLAUDE.md`는 두 버전을 안내한다.

## 결정 사항

| 항목 | 결정 |
|---|---|
| 배포 | GitHub Actions 워크플로(`.github/workflows/pages.yml`)가 `web_version/` 폴더만 Pages로 배포 |
| 웹 입력 방식 | 캔버스 그리기(마우스+터치, Pointer Events)와 이미지 파일 업로드 둘 다 |
| 가중치 전달 | `desktop_version/export_weights.py`가 `mnist_cnn.pt`를 `web_version/model/weights.bin`(float32 리틀엔디언을 이어붙인 것)과 `weights.json`(이름, shape, offset, length)으로 내보냄. 결과물은 git에 커밋 |
| 로컬 실행 | ES 모듈과 `fetch`를 쓰므로 `file://`로는 동작하지 않음. `python -m http.server`로 연다 |
| 외부 의존성 | 웹 버전에는 없음 (라이브러리, 번들러, npm, 빌드 단계 모두 없음) |

## 데스크톱 버전

- 루트의 `model.py`, `train.py`, `predict.py`, `app.py`, `app.bat`, `mnist_cnn.pt`, `sample_*.png`를 `git mv`로 `desktop_version/`에 옮긴다. gitignore된 `data/`도 함께 옮긴다.
- 스크립트의 `mnist_cnn.pt`, `./data` 경로를 스크립트 파일 위치(`Path(__file__).resolve().parent`) 기준으로 바꿔 어느 디렉터리에서 실행해도 동작하게 한다. 모델과 로직은 바꾸지 않는다.
- `export_weights.py`를 새로 만든다. 가중치와 함께 웹 테스트용 기준 데이터(`web_version/tests/fixtures/reference.json`: 샘플별 정답, PyTorch 전처리 입력 784개, PyTorch logits 10개)를 쓰고, 샘플 PNG를 픽스처 폴더에 복사한다.

## 웹 버전 구조

```
web_version/
├── index.html          # UI 마크업
├── style.css
├── js/
│   ├── ops.js          # conv2d3x3, relu, maxPool2x2, linear, softmax, argmax (순수 함수)
│   ├── model.js        # parseWeights, loadWeights, forward, predict
│   ├── preprocess.js   # 흑백 변환, 면적 평균 리사이즈, 패딩, 정규화, decodeImage
│   └── app.js          # DOM 연결 (그리기, 업로드, 결과 표시)
├── model/weights.json, weights.bin
└── tests/
    ├── test.html       # 브라우저에서 여는 테스트 러너
    ├── harness.js      # test / assert / assertClose / runAll
    ├── *.test.js
    └── fixtures/       # reference.json + 샘플 PNG
```

- 텐서는 PyTorch와 같은 행 우선(CHW) 순서의 `Float32Array`로 다룬다. `[64,7,7]` 출력은 그대로 `flatten(1)` 순서와 같다. Dropout은 추론 때 아무 일도 하지 않으므로 생략한다.
- 전처리는 데스크톱과 같게 맞춘다.
  - 그리기(`app.py`): 280×280, 검은 배경에 흰 펜(굵기 18) → 28×28로 축소 → MNIST 정규화 (mean 0.1307, std 0.3081). PIL LANCZOS 대신 면적 평균을 쓴다(10배 축소라 결과가 비슷함).
  - 이미지 파일(`predict.py`): PIL `convert("L")` 공식으로 흑백 변환 → 평균 밝기가 127보다 크면 반전 → 비율을 유지해 28×28에 맞추고 남는 곳은 검정으로 채워 가운데 배치(`ImageOps.pad`) → 정규화.
- UI: 선을 다 그으면(pointerup) 자동으로 인식한다. 예측 숫자, 신뢰도, 10개 클래스 확률 막대, 모델 입력 28×28 미리보기를 보여 주고 "지우기" 버튼을 둔다. 가중치를 불러오지 못하면 상태 문구로 알린다.

## 테스트

- 브라우저 테스트 러너(`tests/test.html`)가 결과를 목록으로 보여 주고 `document.title`을 `PASS n/n` 또는 `FAIL k/n`으로 바꾼다.
- 연산과 전처리 단위 테스트는 손으로 계산한 작은 입력을 쓴다.
- 일치 테스트:
  - (1) PyTorch 입력을 넣었을 때 JS logits가 PyTorch logits와 1e-3 이내로 같고, 예측이 정답과 같다.
  - (2) 샘플 PNG를 JS로 디코딩·전처리한 결과가 PyTorch 입력과 1e-5 이내로 같다.
- 데스크톱은 테스트 스위트가 없으므로 루트에서 `predict.py`를 실행해 샘플을 맞히는지 확인한다.
