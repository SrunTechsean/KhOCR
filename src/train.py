import os
import pandas as pd
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from khmernormalizer import normalize


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class KhmerDataset(Dataset):
    """Fixed dataset with proper target handling"""

    def __init__(self, parquet_path, transform=None):
        print(f"Loading {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform

        # Build vocabulary
        all_text = " ".join(self.df["text"].values)
        self.chars = sorted(list(set(all_text)))
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.char_to_idx["<BLANK>"] = 0
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        print(f"  Dataset: {len(self.df)} samples")
        print(f"  Vocab: {self.vocab_size} characters")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Encode text
        text = row["text"]
        text = normalize(text)  # Fixes unicode issues (splitting vowels)
        encoded = [self.char_to_idx.get(char, 0) for char in text if char in self.char_to_idx]

        # Remove any zeros (unknown chars)
        encoded = [x for x in encoded if x != 0]

        if len(encoded) == 0:
            encoded = [1]  # Space as fallback

        return image, torch.tensor(encoded, dtype=torch.long), len(encoded)


def collate_fn(batch):
    """Custom collate function for variable length targets"""
    images, targets, target_lengths = zip(*batch)

    images = torch.stack(images, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    # Concatenate all targets
    targets = torch.cat(targets)

    return images, targets, target_lengths


# Transforms
train_transform = transforms.Compose(
    [
        transforms.Resize((64, 512)),
        transforms.RandomAffine(degrees=3, shear=5),
        transforms.RandomRotation(3),  # Reduced from 5
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((64, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class CRNN(nn.Module):
    """FIXED CRNN with correct dimensions"""

    def __init__(self, vocab_size, hidden_size=256):
        super(CRNN, self).__init__()

        # CNN - produces [batch, 512, 4, 64]
        self.cnn = nn.Sequential(
            # Block 1: 64x256 → 32x128
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 32x128 → 16x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 16x64 → 16x64 (no pool)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Block 4: 16x64 → 8x64
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            # Block 5: 8x64 → 8x64 (no pool)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Block 6: 8x64 → 4x64
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )

        # RNN - FIXED: 512 * 4 = 2048
        self.rnn = nn.LSTM(
            512 * 4,
            hidden_size,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        # CNN: [batch, 3, 64, 256] → [batch, 512, 4, 64]
        conv_out = self.cnn(x)

        # Reshape for RNN: [batch, width, channels*height]
        batch, channel, height, width = conv_out.size()
        conv_out = conv_out.permute(0, 3, 1, 2)  # [batch, width, channel, height]
        conv_out = conv_out.contiguous().view(batch, width, channel * height)
        # Now: [batch, 64, 2048]

        # RNN
        rnn_out, _ = self.rnn(conv_out)

        # Output: [batch, 64, vocab_size]
        output = self.fc(rnn_out)

        return output


def decode_prediction(output, idx_to_char):
    """Decode CTC output to text"""
    _, preds = output.max(2)
    preds = preds.squeeze(0) if preds.dim() > 1 else preds

    decoded = []
    prev_char = None

    for idx in preds:
        idx = idx.item()
        if idx != 0 and idx != prev_char:
            char = idx_to_char.get(idx, "")
            if char and char != "<BLANK>":
                decoded.append(char)
        prev_char = idx

    return "".join(decoded)


def calculate_accuracy(predictions, ground_truths):
    """Calculate character-level accuracy"""
    correct = 0
    total = 0

    for pred, gt in zip(predictions, ground_truths):
        for i in range(max(len(pred), len(gt))):
            total += 1
            if i < len(pred) and i < len(gt) and pred[i] == gt[i]:
                correct += 1

    return (correct / total * 100) if total > 0 else 0


def test_predictions(model, dataloader, device, idx_to_char, num_samples=5):
    """Test and display predictions"""
    model.eval()

    images, targets, target_lengths = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    predictions = []
    ground_truths = []

    # Reconstruct ground truth texts
    start = 0
    for length in target_lengths[:num_samples]:
        target_seq = targets[start : start + length]
        gt_text = "".join([idx_to_char[idx.item()] for idx in target_seq])
        ground_truths.append(gt_text)
        start += length

    # Get predictions
    for i in range(min(num_samples, images.size(0))):
        pred_text = decode_prediction(outputs[i : i + 1], idx_to_char)
        predictions.append(pred_text)

    # Display
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        match_chars = sum(1 for a, b in zip(pred, gt) if a == b)
        accuracy = (match_chars / max(len(gt), 1)) * 100

        print(f"\nSample {i + 1}:")
        print(f"  Truth: {gt}")
        print(f"  Pred:  {pred}")
        print(f"  Match: {match_chars}/{len(gt)} chars ({accuracy:.1f}%)")

    overall_acc = calculate_accuracy(predictions, ground_truths)
    print(f"\nOverall Character Accuracy: {overall_acc:.1f}%")
    print("=" * 70 + "\n")

    return overall_acc


def train_model(model, train_loader, val_loader, device, epochs=30, lr=0.0005):
    """Training with proper CTC loss"""

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Better optimizer settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0001,  # Reduced weight decay
        betas=(0.9, 0.999),
    )

    # Cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    patience = 10
    patience_counter = 0

    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{epochs} [Train]")
        for images, targets, target_lengths in pbar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(images)  # [batch, width, vocab_size]

            # Prepare for CTC loss: [width, batch, vocab_size]
            log_probs = outputs.permute(1, 0, 2).log_softmax(2)

            # Input lengths (same for all in batch since fixed width)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)

            # CTC Loss
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nSkipping batch with NaN/Inf loss")
                continue

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, targets, target_lengths in tqdm(val_loader, desc=f"Epoch {epoch + 1:02d}/{epochs} [Val]  "):
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                outputs = model(images)
                log_probs = outputs.permute(1, 0, 2).log_softmax(2)
                input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)

                loss = criterion(log_probs, targets, input_lengths, target_lengths)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        # Test predictions
        val_acc = test_predictions(model, val_loader, device, train_loader.dataset.idx_to_char)
        history["val_acc"].append(val_acc)

        # Summary
        print(f"\nEpoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.2f}%")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                },
                "models/best_khmer_ocr_model.pth",
            )
            print(f"  ✓ Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping after {epoch + 1} epochs")
                break

        print("-" * 70 + "\n")

    return history


def main():
    print("=" * 70)
    print("KHMER OCR - FIXED TRAINING")
    print("=" * 70)

    BATCH_SIZE = 32
    EPOCHS = 40
    LEARNING_RATE = 0.0005

    device = get_device()
    print(f"Device: {device}\n")

    # Load data
    train_dataset = KhmerDataset("data/trainset.parquet", transform=train_transform)
    val_dataset = KhmerDataset("data/valset.parquet", transform=val_transform)

    # Share vocab
    val_dataset.char_to_idx = train_dataset.char_to_idx
    val_dataset.idx_to_char = train_dataset.idx_to_char
    val_dataset.vocab_size = train_dataset.vocab_size

    # Save vocab
    vocab_data = {
        "char_to_idx": train_dataset.char_to_idx,
        "idx_to_char": train_dataset.idx_to_char,
        "vocab_size": train_dataset.vocab_size,
        "chars": train_dataset.chars,
    }
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    # DataLoaders with custom collate
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Model
    model = CRNN(vocab_size=train_dataset.vocab_size).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Train
    history = train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LEARNING_RATE)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Training Loss")

    ax2.plot(history["val_acc"], marker="o", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Validation Accuracy")

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best Val Loss: {min(history['val_loss']):.4f}")
    print(f"Best Val Acc:  {max(history['val_acc']):.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
