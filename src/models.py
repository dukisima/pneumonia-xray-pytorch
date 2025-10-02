import torch
import torch.nn as nn
from torchvision import models
import timm #this includes all the models I'm going to be using
# from src import datasets
from collections import Counter

#nn.Module is a base for all models
class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__() #initializes an object with everything from a parent class (nn.Module)
        self.baseModel = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)

        if freeze_backbone:
            for p in self.baseModel.parameters():
                p.requires_grad = False #we freeze all the parameters
            for p in self.baseModel.get_classifier().parameters(): #.get_classifier() returns only the last linear layer
                p.requires_grad = True #we unfreeze only the parameters in the head
            for p in self.baseModel.layer4.parameters():
                p.requires_grad = True #we unfrezze layer4 (second before last)
    def forward(self, x):
        output = self.baseModel(x)
        return output

    @property
    def num_classes(self):
        return self.baseModel.num_classes


class PneumoniaClassifierMobileNet(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        # Load pretrained MobileNetV2
        self.baseModel = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=num_classes)

        if freeze_backbone:
            for p in self.baseModel.parameters():
                p.requires_grad = False  # freeze backbone

            # unlock classifier (last linear layer)
            for p in self.baseModel.get_classifier().parameters():
                p.requires_grad = True

            # unlock the last block (features[-1])
            for p in self.baseModel.blocks[-1].parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.baseModel(x)

    @property
    def num_classes(self):
        return self.baseModel.num_classes

model = PneumoniaClassifier(num_classes=2, freeze_backbone=True)
x = torch.rand(8,3,224,224) #4 img, RGB, 224x224
y = model(x)

#Optimizers (make sure to add filter bc we froze backbone)
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-2)

# print(criterion(y, datasets.labels))
# trainable = sum(p.requires_grad for p in model.parameters())
# print("Trainable params:", trainable)