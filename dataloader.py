import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader


class PixelDataLoader:
    def __init__(self, x_path, y_path, img_path):
        self.x_path = x_path
        self.y_path = y_path
        self.img_path = img_path

        x_data = np.load(x_path)
        y_data = np.load(y_path)
        img_data = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)
        rgb_data = img_data[x_data, y_data]

        self.data = np.stack([x_data, y_data] + [rgb_data[:, i] for i in range(3)], axis=1).astype(np.float32)
        self.data = self.data[np.random.permutation(self.data.shape[0])]
    
    def normalize(self):
        self.data = self.data / np.array([300, 300, 255, 255, 255])

    def denormalize(self, data):
        return data * np.array([300, 300, 255, 255, 255])

    def standardize(self):
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.data = (self.data - self.mean) / self.std

    def destandardize(self, data):
        return data * self.std + self.mean

    def get_dataloader(self, batch_size=64, split_ratio=0.8):
        if not (0 < split_ratio <= 1):
            raise ValueError("split_ratio must be between 0 and 1")
        if split_ratio == 1:
            return DataLoader(torch.tensor(self.data, dtype=torch.float32), batch_size=batch_size, shuffle=True), None
        
        train_size = int(split_ratio * len(self.data))
        valid_size = len(self.data) - train_size

        self.data = torch.tensor(self.data, dtype=torch.float32)
        train_data, valid_data = torch.split(self.data, [train_size, valid_size])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader