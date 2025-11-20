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

try:
    from khmernormalizer import normalize

    NORMALIZE_AVAILABLE = True
    print("✓ Using khmernormalizer")
except ImportError:
    NORMALIZE_AVAILABLE = False
    print("⚠ khmernormalizer not available, install with: pip install khmernormalizer")
    normalize = lambda x: x


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class KhmerDataset(Dataset):
    """Dataset with Khmer normalization"""

    def __init__(self, parquet_path, transform=None):
        print(f"Loading {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform

        # Normalize all text first
        if NORMALIZE_AVAILABLE:
            print("  Normalizing Khmer text...")
            self.df["text"] = self.df["text"].apply(normalize)

        # Build vocabulary
        all_text = " ".join(self.df["text"].values)
        self.chars = sorted(list(set(all_text)))
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.char_to_idx["<BLANK>"] = 0
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        print(f"  Samples: {len(self.df)}")
        print(f"  Vocab: {self.vocab_size} characters")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get normalized text and encode
        text = row["text"]
        encoded = [self.char_to_idx.get(char, 0) for char in text if char in self.char_to_idx]
        encoded = [x for x in encoded if x != 0]

        if len(encoded) == 0:
            encoded = [1]

        return image, torch.tensor(encoded, dtype=torch.long), len(encoded)


def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    targets = torch.cat(targets)
    return images, targets, target_lengths


train_transform = transforms.Compose(
    [
        transforms.Resize((64, 400)),
        transforms.RandomRotation(3, fill=255),  # White background
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.02), fill=255),
        # Random erasing to make model robust
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Simulate occlusions
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((64, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class ImprovedCRNN(nn.Module):
    """Enhanced CRNN with dropout and better architecture"""

    def __init__(self, vocab_size, hidden_size=256):
        super(ImprovedCRNN, self).__init__()

        # CNN with residual-like connections
        self.cnn = nn.Sequential(
            # Block 1: 64x400 → 32x200
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # Block 2: 32x200 → 16x100
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # Block 3: 16x100 → 16x100
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Block 4: 16x100 → 8x100
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.1),
            # Block 5: 8x100 → 8x100
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Block 6: 8x100 → 4x100
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.1),
        )

        # RNN with more layers for Khmer complexity
        self.rnn = nn.LSTM(
            512 * 4,
            hidden_size,
            bidirectional=True,
            num_layers=3,  # Increased from 2
            batch_first=True,
            dropout=0.2,
        )

        # Output with dropout
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        # CNN
        conv_out = self.cnn(x)

        # Reshape
        batch, channel, height, width = conv_out.size()
        conv_out = conv_out.permute(0, 3, 1, 2)
        conv_out = conv_out.contiguous().view(batch, width, channel * height)

        # RNN
        rnn_out, _ = self.rnn(conv_out)

        # Output
        rnn_out = self.dropout(rnn_out)
        output = self.fc(rnn_out)

        return output


def decode_prediction(output, idx_to_char):
    """Improved decoding with confidence threshold"""
    probs = output.softmax(2)
    _, preds = probs.max(2)

    if preds.dim() > 1:
        preds = preds.squeeze(0)
        probs = probs.squeeze(0)

    decoded = []
    prev_char = None

    for i, idx in enumerate(preds):
        idx = idx.item()
        confidence = probs[i, idx].item()

        # Skip low confidence predictions (< 30%)
        if confidence < 0.3:
            prev_char = None
            continue

        if idx != 0 and idx != prev_char:
            char = idx_to_char.get(idx, "")
            if char and char != "<BLANK>":
                decoded.append(char)
        prev_char = idx

    return "".join(decoded)


def calculate_cer(predictions, ground_truths):
    """Character Error Rate"""
    import editdistance

    total_chars = 0
    total_errors = 0

    for pred, gt in zip(predictions, ground_truths):
        errors = editdistance.eval(pred, gt)
        total_errors += errors
        total_chars += len(gt)

    return (total_errors / total_chars * 100) if total_chars > 0 else 100


def test_predictions(model, dataloader, device, idx_to_char, num_samples=5):
    """Test with CER metric"""
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

    # Get ground truths
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
        print(f"  Chars: {match_chars}/{len(gt)} ({accuracy:.1f}%)")

    try:
        cer = calculate_cer(predictions, ground_truths)
        print(f"\nCharacter Error Rate: {cer:.2f}%")
        print(f"Accuracy: {100 - cer:.2f}%")
    except:
        cer = 50.0
        print("\n(Install editdistance: pip install editdistance)")

    print("=" * 70 + "\n")

    return 100 - cer


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.0003):
    """Training with better hyperparameters"""

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001, betas=(0.9, 0.999))

    # Warmup + Cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,  # 15% warmup
        anneal_strategy="cos",
        div_factor=25.0,  # Start with lr/25
        final_div_factor=10000.0,  # End with lr/10000
    )

    best_val_acc = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    patience = 15
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

            outputs = model(images)
            log_probs = outputs.permute(1, 0, 2).log_softmax(2)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

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

        print(f"\nEpoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.2f}%")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        # Save best based on accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
            print(f"  ✓ Best model saved! (Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping at epoch {epoch + 1}")
                break

        print("-" * 70)

    return history


def main():
    print("=" * 70)
    print("KHMER OCR - OPTIMIZED TRAINING")
    print("=" * 70)

    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0003

    device = get_device()
    print(f"Device: {device}\n")

    # Load data
    train_dataset = KhmerDataset("data/combined_train.parquet", transform=train_transform)
    val_dataset = KhmerDataset("data/valset.parquet", transform=val_transform)

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

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True
    )

    # Model
    model = ImprovedCRNN(vocab_size=train_dataset.vocab_size, hidden_size=256).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Train
    history = train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LEARNING_RATE)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Loss")

    axes[1].plot(history["val_acc"], marker="o", linewidth=2, markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Validation Accuracy")

    axes[2].plot(np.array(history["train_loss"]) - np.array(history["val_loss"]), linewidth=2)
    axes[2].axhline(y=0, color="r", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Train Loss - Val Loss")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Overfitting Check")

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best Val Acc: {max(history['val_acc']):.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
