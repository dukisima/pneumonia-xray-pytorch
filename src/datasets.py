import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from src import transforms #we can do this because we have __init__ in src so it looks at src as module
from collections import Counter

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
    def targets(self):
        return self.data.targets #list of labels
    @property
    def samples(self):
        return self.data.samples

class ListImageDataset(Dataset):
    def __init__(self,sample, class_names,transform=True ):
        self.sample = sample #list (path, label)
        self.class_names = class_names #class names: ['NORMAL', 'PNEUMOINA']
        self.transform = transform
    def __len__(self):
        return len(self.sample)
    def __getitem__(self, item):
        path, label = self.sample[item]
        image = Image.open(path).convert("RGB") #if some x-ray are gray scale
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    @property
    def classes(self):
        return self.class_names
    @property
    def samples(self):
        return self.sample
    @property
    def targets(self):
        return [lbl for _, lbl in self.sample]



#../ in data_dir si to sy go one step above bc this script is in src so it doesn't see data
train_directory = "../data/chest_xray/train"
val_directory = "../data/chest_xray/val"
test_directory = "../data/chest_xray/test"

old_train = PneumoniaDataset(data_dir=train_directory, transform=transforms.train_transforms)
old_val = PneumoniaDataset(data_dir=val_directory, transform=transforms.test_val_transforms)

assert old_train.classes == old_val.classes, "Train and Val labels not the same !" #they are the same
class_names = old_train.classes

pooled_samples = old_train.samples + old_val.samples
y = [lbl for _,lbl in pooled_samples] #labels for the stratification split

# ----------- Stratification split ------------

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(sss.split(pooled_samples,y))#splits samples and labels into two datasets

new_train_samples = [pooled_samples[i] for i in train_idx]
new_val_samples = [pooled_samples[i] for i in val_idx]
train_counts = Counter([lbl for _, lbl in new_train_samples])
val_counts   = Counter([lbl for _, lbl in new_val_samples])
print("New TRAIN class counts:", train_counts)
print("New VAL   class counts:", val_counts)

# NEW datasets and dataloaders

train_dataset = ListImageDataset(new_train_samples, class_names=class_names, transform=transforms.train_transforms)
val_dataset = ListImageDataset(new_val_samples, class_names=class_names, transform=transforms.test_val_transforms)
test_dataset = PneumoniaDataset(data_dir=test_directory, transform=transforms.test_val_transforms) #this one stays the same

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)


images, labels = next(iter(train_loader))
print(images.shape, labels.shape)  # očekuješ [B, 3, 224, 224], [B]





# train_dataset = PneumoniaDataset(data_dir=train_directory, transform=transforms.train_transforms)
# val_dataset = PneumoniaDataset(val_directory,  transform=transforms.test_val_transforms)
# test_dataset = PneumoniaDataset(test_directory, transform=transforms.test_val_transforms)
#
# #Data loaders (small batch size due to weak hardware)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_dataset,  batch_size=8, shuffle=False, num_workers=0)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
#
# # for images, labels in train_loader:
# #     break
# # # image, label = iter(train_loader)
# # print(images.shape, labels.shape)