import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import models
import datasets

#Datasets and dataloaders
train_dataset = datasets.train_dataset
train_loader = datasets.train_loader

val_dataset = datasets.val_dataset
val_loader = datasets.val_loader

#Model/loss function/optimiser
model = models.model
optimizer = models.optimizer
criterion = models.criterion


train_losses, val_losses = [], [] #losses for the epoch
correct, total = 0,0
num_epochs = 5

#Making sure it is running on GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device) #moving a model to a devide(GPU)

for i in range(num_epochs):
    #Train phase
    model.train()
    running_loss = 0.0
    for images,labels in tqdm(train_loader, desc=f"Train {i+1}/{num_epochs}"):
        images = images.to(device)#moving images to GPU
        labels = labels.to(device)#moving labels to GPU
        optimizer.zero_grad() #sets all the gradietns (atributs) to 0 sot that they don't accumulate
        outputs = model(images)
        loss = criterion(outputs,labels)#here we calculate loss
        loss.backward() #backprobagation (recalebrate parametrs)
        optimizer.step() #uses gradinets calculated in backpropagation to update weights
        running_loss += loss.item() * images.size(0) #batch loss * batch size
        #------Accuracy------
        _, preds = torch.max(outputs, 1)  # indeks klase sa max logitom
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader.dataset) #this is a loss for a whole epoch
    train_losses.append(train_loss) #u videu nije definisao train_losses ali njemu ne crveni da koai nije definisano
    train_acc = correct / total

    #Validation phase
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad(): #this is so that the parameters (weights) are not changed
        for images, labels in tqdm(val_loader, desc=f"Val {i+1}/{num_epochs}"):
            images = images.to(device)  # moving images to GPU
            labels = labels.to(device)  # moving labels to GPU
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            #------Acuracy------
            _, preds = torch.max(outputs, 1)  # indeks klase sa max logitom
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_acc = correct/total

    tqdm.write(f"Epoch {i+1}/{num_epochs} - Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | Val loss: {val_loss:.4f}, acc: {val_acc:.3f}")

# --------- Visualisation ---------

plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Val loss")
plt.legend()
plt.title("Loss function per epoch")
plt.show()
