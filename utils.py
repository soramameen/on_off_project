import os
from PIL import Image
from torch.utils.data import Dataset


class SwitchDataset(Dataset):
    """
    固定カメラで撮影した画像データセット。
    画像ディレクトリ内に 'on' フォルダと 'off' フォルダがあり、それぞれの画像をラベルとして読み込む。
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 画像ディレクトリのパス（例: "./data/train" または "./data/test"）
            transform: torchvision.transforms を用いた前処理
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # 'on' フォルダをラベル 0、'off' フォルダをラベル 1 とする例
        for label, subfolder in enumerate(["on", "off"]):
            folder_path = os.path.join(root_dir, subfolder)
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
