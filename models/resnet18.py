import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18_model(num_classes=2):
    """
    事前学習済みのResNet18を読み込み、最終層を再定義して2クラス分類用に調整する。
    """
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
