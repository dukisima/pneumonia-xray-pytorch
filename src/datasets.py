import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from src import transforms #we can do this because we have __init__ in src so it looks at src as module

class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item]
    @property
    def classes(self):
        return self.data.classes
    @property
    def target(self):
        return self.data.targets #list of labels
    @property
    def samples(self):
        return self.data.samples

#../ in data_dir si to sy go one step above bc this script is in src so it doesn't see data
train_directory = "../data/chest_xray/train"
val_directory = "../data/chest_xray/val"
test_directory = "../data/chest_xray/test"

train_dataset = PneumoniaDataset(data_dir=train_directory, transform=transforms.train_transforms)
val_dataset = PneumoniaDataset(val_directory,  transform=transforms.test_val_transforms)
test_dataset = PneumoniaDataset(test_directory, transform=transforms.test_val_transforms)

#Data loaders (small batch size due to weak hardware)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset,  batch_size=8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

for images, labels in train_loader:
    break
# image, label = iter(train_loader)
print(images.shape, labels.shape)