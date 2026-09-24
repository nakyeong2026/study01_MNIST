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
