# -*- coding: utf-8 -*-
"""MNIST 데이터셋으로 CNN 모델을 학습시키고 가중치를 mnist_cnn.pt로 저장하는 스크립트"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistCNN

# 학습 하이퍼파라미터
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
# 스크립트 파일 위치 기준 경로 (어느 폴더에서 실행해도 동작하도록)
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "mnist_cnn.pt"
DATA_DIR = BASE_DIR / "data"


def get_data_loaders():
    """MNIST 학습/테스트 데이터셋을 내려받고 DataLoader로 반환"""

    # MNIST의 평균(mean)과 표준편차(std)로 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """한 에폭(epoch) 동안 모델을 학습"""

    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 200 == 0:
            print(f"  에폭 {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)}] 손실: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"에폭 {epoch} 평균 손실: {avg_loss:.4f}")


def evaluate(model, device, test_loader, criterion):
    """테스트 데이터셋으로 모델 정확도를 평가"""

    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"테스트 손실: {test_loss:.4f}, 정확도: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")

    return accuracy


def main():
    # GPU를 사용할 수 있으면 GPU를, 아니면 CPU를 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    train_loader, test_loader = get_data_loaders()

    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, device, train_loader, optimizer, criterion, epoch)
        evaluate(model, device, test_loader, criterion)

    # 학습된 가중치를 파일로 저장
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"학습된 가중치를 '{WEIGHTS_PATH}' 파일에 저장했습니다.")


if __name__ == "__main__":
    main()
