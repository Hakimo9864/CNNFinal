import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# REAL CNN STARTER CODE
#
# Goal:
# Train a real CNN to classify circles and squares.
#
# Expected folder structure:
#
# data/
#     train/
#         circles/
#         squares/
#     val/
#         circles/
#         squares/
#     test/
#         circles/
#         squares/
# ============================================================

# ============================================================
# SECTION 1: SETTINGS
#
# TASK:
# Choose values for the training settings below.
# Suggested starting values are included.
# ============================================================

DATA_DIR = "data"
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SECTION 2: TRANSFORMS
#
# TASK:
# Build a transform pipeline that:
#   1. converts images to grayscale
#   2. resizes them
#   3. converts them to tensors
#
# You may add more transforms later if you want.
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# SECTION 3: LOAD DATASETS
#
# TASK:
# Use torchvision.datasets.ImageFolder to load:
#   - train dataset
#   - validation dataset
#   - test dataset
#
# Then create DataLoaders for each one.
# ============================================================

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_dataset.classes)


# ============================================================
# SECTION 4: BUILD THE MODEL
#
# TASK:
# Complete the CNN below.
#
# Requirements:
#   - at least two convolution layers
#   - ReLU activations
#   - max pooling
#   - a classifier at the end
#
# Hint:
#   Since the images are grayscale, the input channel count is 1.
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        flattened_size = 16 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN().to(DEVICE)
print(model)


# ============================================================
# SECTION 5: LOSS AND OPTIMIZER
#
# TASK:
# Choose a loss function and optimizer.
#
# Hint:
#   For a 2-class classification problem with raw output scores,
#   CrossEntropyLoss is a good choice.
# ============================================================

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ============================================================
# SECTION 6: TRAINING FUNCTION
#
# TASK:
# Complete the training loop.
#
# For each batch:
#   1. move images and labels to DEVICE
#   2. make predictions
#   3. compute loss
#   4. zero the gradients
#   5. backpropagate
#   6. update the weights
#   7. track accuracy
# ============================================================

def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================
# SECTION 7: EVALUATION FUNCTION
#
# TASK:
# Complete the evaluation loop.
#
# This is like training, but:
#   - no gradient updates
#   - no optimizer step
#   - use model.eval()
#   - use torch.no_grad()
# ============================================================

def evaluate(model, loader, loss_fn):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================
# SECTION 8: TRAIN THE MODEL
#
# TASK:
# Train for EPOCHS epochs.
# Print:
#   - training loss
#   - training accuracy
#   - validation loss
#   - validation accuracy
# ============================================================

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train loss: {train_loss:.4f} | "
        f"train acc: {train_acc:.4f} | "
        f"val loss: {val_loss:.4f} | "
        f"val acc: {val_acc:.4f}"
    )


# ============================================================
# SECTION 9: TEST THE MODEL
#
# TASK:
# Evaluate the final model on the test set.
# ============================================================

test_loss, test_acc = evaluate(model, test_loader, loss_fn)

print("\nFinal test results")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")


# ============================================================
# SECTION 10: EXTENSIONS
#
# Ideas to try:
#   - increase the number of channels
#   - train for more epochs
#   - add a third convolution block
#   - add data augmentation
#   - inspect misclassified images
# ============================================================
