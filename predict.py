# -*- coding: utf-8 -*-
"""저장된 mnist_cnn.pt 가중치를 불러와 손글씨 숫자 이미지를 인식하는 스크립트

사용법:
    python predict.py <이미지_경로>

이미지는 숫자 하나가 크게 보이는 사진/스캔 파일이면 됩니다.
(예: 흰 종이에 검정 펜으로 쓴 숫자 사진)
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from model import MnistCNN

WEIGHTS_PATH = "mnist_cnn.pt"


def load_model(device):
    """저장된 가중치 파일을 불러와 모델을 준비"""

    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path):
    """입력 이미지를 MNIST 학습 데이터와 같은 형태(28x28 흑백)로 변환"""

    image = Image.open(image_path).convert("L")  # 흑백(그레이스케일)으로 변환

    # MNIST는 '검은 배경 + 흰 글씨' 형태이므로,
    # 평균 밝기가 밝다면(흰 종이에 쓴 사진) 색을 반전시켜 맞춰줌
    pixel_mean = float(np.array(image).mean())
    if pixel_mean > 127:
        image = ImageOps.invert(image)

    # MNIST 숫자 주변 여백과 비슷하게 정사각형으로 맞춘 뒤 28x28로 축소
    image = ImageOps.pad(image, (28, 28), color=0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    tensor = transform(image)
    return tensor.unsqueeze(0)  # 배치 차원 추가: (1, 1, 28, 28)


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    input_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).squeeze(0)
        predicted_digit = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted_digit].item())

    print(f"예측된 숫자: {predicted_digit} (신뢰도: {confidence * 100:.2f}%)")
    print("클래스별 확률:")
    for digit, prob in enumerate(probabilities.tolist()):
        print(f"  {digit}: {prob * 100:5.2f}%")

    return predicted_digit


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python predict.py <이미지_경로>")
        sys.exit(1)

    predict(sys.argv[1])
