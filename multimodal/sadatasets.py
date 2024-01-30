import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'data')
        self.labels = pd.read_csv(os.path.join(root_dir, 'train.txt')).to_dict()
        self.len = len(self.labels['guid'])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        guid = self.labels['guid'][idx]
        image_path = os.path.join(self.data_dir, f'{guid}.jpg')
        text_path = os.path.join(self.data_dir, f'{guid}.txt')

        image = Image.open(image_path)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # Change dimension from HWC to CHW

        with open(text_path, 'r', errors='ignore') as f:
            text = f.read()

        label = self.labels['tag'][idx]
        if label == 'positive':
            label = 0
        elif label == 'neutral':
            label = 1
        else:
            label = 2
        sample = {'guid': guid, 'text': text, 'image': image, 'label': label}

        return sample

class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, 'data')
        self.labels = pd.read_csv(os.path.join(root_dir, 'test_without_label.txt')).to_dict()
        self.len = len(self.labels['guid'])
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        guid = self.labels['guid'][idx]
        image_path = os.path.join(self.data_dir, f'{guid}.jpg')
        text_path = os.path.join(self.data_dir, f'{guid}.txt')

        image = Image.open(image_path)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # Change dimension from HWC to CHW

        with open(text_path, 'r', errors='ignore') as f:
            text = f.read()

        label = self.labels['tag'][idx]
        sample = {'guid': guid, 'text': text, 'image': image, 'label': label}

        return sample

if __name__ == '__main__':
    train_dataset = TrainDataset('../datasets')
    test_dataset = TestDataset('../datasets')
    
    print(len(train_dataset))
    print(len(test_dataset))
    