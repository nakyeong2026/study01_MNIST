# -*- coding: utf-8 -*-
"""마우스로 숫자를 직접 그리면 학습된 CNN 모델이 바로 인식해주는 데스크톱 앱

사용법:
    python app.py
    (실행 전에 train.py로 학습해 mnist_cnn.pt가 먼저 만들어져 있어야 함)
"""

import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from model import MnistCNN

# 스크립트 파일 위치 기준 경로 (app.bat이나 다른 폴더에서 실행해도 동작하도록)
WEIGHTS_PATH = Path(__file__).resolve().parent / "mnist_cnn.pt"
CANVAS_SIZE = 280   # 화면에 보이는 캔버스 크기(픽셀)
IMAGE_SIZE = 28     # 모델에 입력하는 실제 이미지 크기
PEN_WIDTH = 18      # 그리기 펜 굵기


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("손글씨 숫자 인식기")
        self.root.resizable(False, False)

        # 추론에 사용할 장치와 모델 준비
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        # 사용자가 마우스로 그리는 화면 캔버스 (검은 배경 + 흰 글씨, MNIST와 동일한 배색)
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # 화면 캔버스와 동시에 그려지는 메모리 상의 이미지 (화면 캡처 없이 모델 입력을 만들기 위함)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self._on_draw)

        result_font = tkfont.Font(size=28, weight="bold")
        self.result_label = tk.Label(root, text="숫자를 그려보세요", font=result_font)
        self.result_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        clear_button = tk.Button(root, text="지우기", command=self._clear_canvas, width=12)
        clear_button.grid(row=2, column=0, pady=(0, 10))

        predict_button = tk.Button(root, text="인식하기", command=self._predict, width=12)
        predict_button.grid(row=2, column=1, pady=(0, 10))

    def _load_model(self):
        if not os.path.exists(WEIGHTS_PATH):
            print(f"'{WEIGHTS_PATH}' 파일이 없습니다. 먼저 'python train.py'를 실행해 학습해 주세요.")
            sys.exit(1)

        model = MnistCNN().to(self.device)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
        model.eval()
        return model

    def _on_draw(self, event):
        x, y = event.x, event.y
        r = PEN_WIDTH // 2
        # 화면 캔버스에 흰색 원을 그림
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        # 같은 위치를 메모리 상의 이미지에도 그림 (모델 입력용)
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_label.config(text="숫자를 그려보세요")

    def _predict(self):
        # MNIST 학습 데이터와 같은 크기(28x28)로 축소
        small_image = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        tensor = self.transform(small_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1).squeeze(0)
            predicted_digit = int(probabilities.argmax().item())
            confidence = float(probabilities[predicted_digit].item())

        self.result_label.config(text=f"예측: {predicted_digit}  ({confidence * 100:.1f}%)")


def main():
    root = tk.Tk()
    DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
