import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import get_resnet18_model
from utils import SwitchDataset


def evaluate_model(model, dataloader, device):
    model.eval()
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
    accuracy = running_corrects.double() / total
    return accuracy.item()


def main():
    # テスト用データのディレクトリ（例: "./data/test"）
    data_dir = "./data/test"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = SwitchDataset(data_dir, transform=transform)
    dataloader = DataLoader(test_dataset, batch_size=16,
                            shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18_model(num_classes=2)

    checkpoint_path = "checkpoints/switch_classifier.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    acc = evaluate_model(model, dataloader, device)
    print(f"Test Accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()
