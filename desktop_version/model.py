# -*- coding: utf-8 -*-
"""MNIST 손글씨 숫자 인식을 위한 CNN 모델 정의"""

import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """0~9 손글씨 숫자(28x28 흑백 이미지)를 분류하는 합성곱 신경망"""

    def __init__(self):
        super().__init__()

        # 첫 번째 합성곱 블록: 1채널(흑백) -> 32채널
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # 두 번째 합성곱 블록: 32채널 -> 64채널
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # 2x2 최대 풀링: 특징 맵 크기를 절반으로 줄임
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 과적합 방지를 위한 드롭아웃
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # 두 번의 풀링을 거치면 28x28 -> 14x14 -> 7x7 크기가 됨
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 숫자 0~9, 총 10개 클래스

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 28x28 -> 14x14

        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 14x14 -> 7x7

        x = self.dropout1(x)
        x = x.flatten(1)  # 완전연결층에 넣기 위해 1차원으로 펼침

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)  # 클래스별 점수(logit) 출력

        return x
