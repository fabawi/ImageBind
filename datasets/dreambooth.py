import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from models.imagebind_model import ModalityType
import data


class DreamBoothDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.8, random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.paths = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for filename in os.listdir(cls_dir):
                if filename.endswith('.jpg'):
                    self.paths.append((os.path.join(cls_dir, filename), cls))

        # Split dataset
        train_paths, test_paths = train_test_split(self.paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_paths
        elif split == 'test':
            self.paths = test_paths
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path, class_text = self.paths[index]
        images = data.load_and_transform_vision_data([img_path], self.device, to_tensor=False)

        if self.transform is not None:
            image = images[0]
            images = self.transform(image)

        texts = data.load_and_transform_text([class_text], self.device)

        return images, ModalityType.VISION, texts, ModalityType.TEXT, self.class_to_idx[class_text]
