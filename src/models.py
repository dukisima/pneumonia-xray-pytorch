import torch
import torch.nn as nn
from torchvision import models
import timm #this includes all the models I'm going to be using
import datasets

#nn.Mudule is a base for all models
class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2, freez_backbone=True ) :
        super().__init__() #initializes an object with everything from a parent class
        self.baseModel = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)

        if freez_backbone:
            for p in self.baseModel.parameters():
                p.requires_grad = False #we freeze all the parameters
            for p in self.baseModel.get_classifier().parameters(): #.get_classifier() returns only the last linear layer
                p.requires_grad = True #we unfreeze only the parameters in the head
    def forward(self, x):
        output = self.baseModel(x)
        return output


model = PneumoniaClassifier(num_classes=2, freez_backbone=True)
x = torch.rand(8,3,224,224) #4 img, RGB, 224x224
y = model(x)
print(y.shape) #should be [batch_size, num_classes]

#Loss functions
criterion = nn.CrossEntropyLoss()
#Optimizers (make sure to add filter bc we froze backbone)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

print(criterion(y, datasets.labels))
