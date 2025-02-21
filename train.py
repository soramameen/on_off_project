import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models import get_resnet18_model
from utils import SwitchDataset


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # トレーニングと検証フェーズをそれぞれ実施
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 学習モード
            else:
                model.eval()   # 評価モード

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # フォワード計算（検証フェーズは勾配計算無効）
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model


def main():
    # 学習用データのディレクトリ（例: "./data/train"）
    data_dir = "./data/train"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = SwitchDataset(data_dir, transform=transform)

    # データセットを80%の訓練用と20%の検証用に分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18_model(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    model = train_model(model, dataloaders, criterion,
                        optimizer, device, num_epochs=num_epochs)

    # チェックポイントディレクトリがなければ作成
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/switch_classifier.pth")
    print("Model saved as checkpoints/switch_classifier.pth")


if __name__ == '__main__':
    main()
