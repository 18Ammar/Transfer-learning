# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from henet import HENet
# %%
# Config
NUM_CLASSES = 200
BATCH_SIZE = 64
IMAGE_SIZE = 224
EPOCHS = 50
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "henet_res_model.pth"
TRAIN_DIR = "D:\\Harf\\splited_datasets-20250425T042643Z-001\\splited_datasets\\train"
VAL_DIR = "D:\\Harf\\splited_datasets-20250425T042643Z-001\\splited_datasets\\test"
# %%
# Transforms
train_transforms = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transforms = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset & Dataloader
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# %%
# Loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, pred, target):
        logprobs = nn.functional.log_softmax(pred, dim=-1)
        nll = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -logprobs.mean(dim=-1)
        return (self.confidence * nll + self.smoothing * smooth).mean()


# %%
# Model
model = HENet(backbone='resnet18', n_class=NUM_CLASSES, beta=1.5).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training Loop
best_loss = float('inf')
patience = 5
early_counter = 0

# %%
for _, labels in train_loader:
    print(labels.min(), labels.max())  
    break

# %%
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"Train Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    print(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    scheduler.step()

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print("✅ Model saved")
        early_counter = 0
    else:
        early_counter += 1
        if early_counter >= patience:
            print("⏹️ Early stopping")
            break

# %%
