import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import ChessDataset, NUM_CLASSES, NUM_SQUARES

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

# --- Config ---
TRAIN_DIR = "kaggle/train"
MAX_SAMPLES = 1000
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
VAL_SPLIT = 0.2

# --- Dataset & DataLoaders ---
full_dataset = ChessDataset(TRAIN_DIR, max_samples=MAX_SAMPLES)
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# --- Model ---
model = timm.create_model(
    "vit_base_patch16_224.augreg_in21k",
    pretrained=True,
    num_classes=NUM_SQUARES * NUM_CLASSES,
)
model = model.to(DEVICE)

# --- Training ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)       # (B, 3, 224, 224)
        labels = labels.to(DEVICE)       # (B, 64)

        logits = model(images)           # (B, 64*13)
        logits = logits.view(-1, NUM_SQUARES, NUM_CLASSES)  # (B, 64, 13)

        loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=-1)    # (B, 64)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    avg_loss = total_loss / len(train_dataset)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate():
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        logits = logits.view(-1, NUM_SQUARES, NUM_CLASSES)

        loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    avg_loss = total_loss / len(val_dataset)
    accuracy = correct / total
    return avg_loss, accuracy


for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()
    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
    )

torch.save(model.state_dict(), "chess_vit.pth")
print("Model saved to chess_vit.pth")
