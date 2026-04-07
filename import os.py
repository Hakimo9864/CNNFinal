import os
import random
import math
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# SETTINGS
# ============================================================
IMAGE_SIZE = 64
TRAIN_PER_CLASS = 200
VAL_PER_CLASS = 50
DATA_DIR = "data"

CLASS_A = "filled_circle"
CLASS_B = "hollow_circle"

BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# MAKE FOLDERS
# ============================================================
def make_folders():
    for split in ["train", "val"]:
        for cls in [CLASS_A, CLASS_B]:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)

# ============================================================
# IMAGE GENERATION
# ============================================================
def add_random_noise(img, amount=10):
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-amount, amount + 1, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def draw_shape_image(label, hard=False):
    """
    label = CLASS_A or CLASS_B
    hard = whether to make dataset more difficult
    """
    bg_color = random.randint(220, 255) if hard else 255
    img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)

    # shape size and position
    radius = random.randint(12, 20) if hard else random.randint(14, 18)
    cx = random.randint(radius + 5, IMAGE_SIZE - radius - 5)
    cy = random.randint(radius + 5, IMAGE_SIZE - radius - 5)

    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]

    shape_color = random.randint(0, 80) if hard else 0

    if label == CLASS_A:
        draw.ellipse(bbox, fill=shape_color, outline=shape_color)
    else:
        line_width = random.randint(2, 5) if hard else 3
        draw.ellipse(bbox, outline=shape_color, width=line_width)

    if hard:
        # random clutter lines
        for _ in range(random.randint(1, 5)):
            x1 = random.randint(0, IMAGE_SIZE - 1)
            y1 = random.randint(0, IMAGE_SIZE - 1)
            x2 = random.randint(0, IMAGE_SIZE - 1)
            y2 = random.randint(0, IMAGE_SIZE - 1)
            clutter_color = random.randint(100, 200)
            draw.line((x1, y1, x2, y2), fill=clutter_color, width=1)

        # blur a little sometimes
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=1))

        # add noise
        img = add_random_noise(img, amount=15)

    return img

def save_dataset(hard=False):
    make_folders()

    counts = {
        "train": TRAIN_PER_CLASS,
        "val": VAL_PER_CLASS
    }

    for split in ["train", "val"]:
        for i in range(counts[split]):
            img_a = draw_shape_image(CLASS_A, hard=hard)
            img_b = draw_shape_image(CLASS_B, hard=hard)

            img_a.save(os.path.join(DATA_DIR, split, CLASS_A, f"{CLASS_A}_{i}.png"))
            img_b.save(os.path.join(DATA_DIR, split, CLASS_B, f"{CLASS_B}_{i}.png"))

    print("Dataset created.")
    print("Hard mode:", hard)

# ============================================================
# MODEL
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================
# TRAIN / EVAL FUNCTIONS
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

# ============================================================
# MAIN TRAINING
# ============================================================
def run_training():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=transform_train
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"),
        transform=transform_val
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Classes:", train_dataset.classes)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        print()

    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    # First: easier dataset
    save_dataset(hard=False)
    run_training()

    # Then: harder dataset
    # Uncomment these lines after testing the easy version
    # save_dataset(hard=True)
    # run_training()