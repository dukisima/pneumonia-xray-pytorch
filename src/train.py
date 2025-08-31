import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import os

import models          # model/optimizer live here
import datasets        # loaders live here

# -----------------------------
# Data (datasets & dataloaders)
# -----------------------------
train_dataset = datasets.train_dataset
val_dataset   = datasets.val_dataset

train_loader  = datasets.train_loader
val_loader    = datasets.val_loader

# -----------------------------
# Model & Device (making sure it runs on GPU)
# -----------------------------
model  = models.model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)   # move model to GPU (MPS) or CPU

# -----------------------------
# Class weights (for imbalance)
# -----------------------------
class_counts  = Counter(train_dataset.targets)     # works with ListImageDataset.targets
num_classes   = len(train_dataset.classes)

counts_tensor = torch.tensor([class_counts.get(i, 0) for i in range(num_classes)],dtype=torch.float)
counts_tensor = torch.clamp(counts_tensor, min=1.0)

weights  = 1.0 / counts_tensor
weights  = weights / weights.sum()

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = models.optimizer   # optimiser from the setting

# -----------------------------
# Tracking containers & epochs
# -----------------------------
train_losses, val_losses = [], []   # epoch-level losses
num_epochs = 15

# -----------------------------
# Early stopping and best model weights tracking
# -----------------------------
best_val = float("inf") # track best validation loss
patience = 4            # epochs to wait without improvement
stall = 0               # how many epochs without improvement has there been

os.makedirs("checkpoints", exist_ok=True) #just in case it desnt exist

# -----------------------------
# Scheduler (auto adjusting of learning rate(lr))
# -----------------------------
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",     # we are minimising val_loss
    factor=0.5,     # when activated -> lr = lr * 0.5
    patience=2,     # wait 2 epoch before adjusting
)

for i in range(num_epochs):
    # -------- TRAIN PHASE --------
    model.train()
    running_loss = 0.0
    correct, total = 0, 0  # reset per-phase (train)

    for images, labels in tqdm(train_loader, desc=f"Train {i+1}/{num_epochs}"):
        # move batch to device
        images = images.to(device)
        labels = labels.to(device)

        # forward + loss
        optimizer.zero_grad()              # clear stale gradients
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward + update
        loss.backward()                    # backprop through the graph
        optimizer.step()                   # update weights using gradients

        # accumulate epoch loss (sum over samples)
        running_loss += loss.item() * images.size(0)

        # accuracy for this batch
        _, preds = torch.max(outputs, 1)   # class index with max logit
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    # epoch-level train metrics
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_acc = correct / total

    # -------- VALIDATION PHASE --------
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0  # reset per-phase (val)

    with torch.no_grad():  # no gradient updates during validation
        for images, labels in tqdm(val_loader, desc=f"Val {i+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # accuracy for this batch
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    # epoch-level val metrics
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_acc = correct / total

    # -------- Schedulers --------
    scheduler.step(val_loss)
    # get current LR (from first param group)
    current_lr = optimizer.param_groups[0]['lr']

    tqdm.write(
        f"Epoch {i+1}/{num_epochs} - "
        f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
        f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f} |"
        f"Current learning rate: {current_lr:.6f}"
    )

    # -------- Save-best and early stopping by Val loss --------
    if val_loss < best_val:
        best_val = val_loss
        stall = 0
        torch.save({
            "model_state": model.state_dict(), # current model weights
            "val_loss": val_loss,
            "val_acc": val_acc
        }, "../checkpoints/best_model.pt")
        tqdm.write(" ✅ Saved  new  best ")
    else:
        stall += 1
        if stall >= patience:
            tqdm.write("No improvement | Early stopping triggered ")
            break



# -------- Visualization --------
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses,   label="Val loss")
plt.title("Loss per epoch")
plt.legend()
plt.show()
