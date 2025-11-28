import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Train Khmer OCR Model (CRNN) - Mac/MPS Optimized")
    parser.add_argument("--train", type=str, required=True, help="Path to training parquet file")
    parser.add_argument("--val", type=str, required=True, help="Path to validation parquet file")
    parser.add_argument("--output", type=str, default="best_khmer_ocr_model.pth", help="Path to save trained model")
    parser.add_argument("--resume", type=str, default=None, help="Path to a pre-trained .pth file to fine-tune")
    parser.add_argument("--width", type=int, default=512, help="Image width (recommended: 512 for full sentences)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (lower if Out Of Memory)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning Rate (use 0.0001 for fine-tuning)")

    args = parser.parse_args()

    print("=" * 70)
    print(f"KHMER OCR TRAINING CONFIG")
    print(f"Train Data:  {args.train}")
    print(f"Val Data:    {args.val}")
    print(f"Output:      {args.output}")
    print(f"Resume From: {args.resume if args.resume else 'None (Scratch)'}")
    print(f"Image Size:  64x{args.width}")
    print("=" * 70)
    print("Loading libraries... (this may take a few seconds)")

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import json
    from PIL import Image
    import io

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: Please run 'pip install datasets'")
        sys.exit(1)

    try:
        from khmernormalizer import normalize

        NORMALIZE_AVAILABLE = True
        print("✓ Using khmernormalizer")
    except ImportError:
        NORMALIZE_AVAILABLE = False
        print("⚠ khmernormalizer not available. Recommendation: pip install khmernormalizer")
        normalize = lambda x: x

    def get_device():
        # MPS (Metal Performance Shaders) is the specific accelerator for Mac M1/M2/M3 chips.
        if torch.backends.mps.is_available():
            print("✓ Using Apple MPS (Metal Performance Shaders)")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("✓ Using NVIDIA CUDA")
            return torch.device("cuda")
        print("⚠ Using CPU (Training will be slow)")
        return torch.device("cpu")

    def get_transforms(width=512, height=64, is_train=True):
        # We resize to a fixed height (64) but a wide width (512).
        # This preserves the aspect ratio of long sentences, preventing characters
        # from getting squashed together which confuses the RNN.
        if is_train:
            return transforms.Compose(
                [
                    transforms.Resize((height, width)),
                    transforms.RandomRotation(3, fill=255),  # Simulates slightly crooked scanning
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.1
                    ),  # Simulates different lighting/ink
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.02), fill=255),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    transforms.RandomErasing(
                        p=0.1, scale=(0.02, 0.1)
                    ),  # Forces model to learn context if a letter is blocked
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((height, width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    class KhmerDataset(Dataset):
        def __init__(self, parquet_path, transform=None):
            print(f"Streaming {parquet_path} (Zero RAM Load)...")

            # Use Hugging Face 'load_dataset' with 'parquet' engine
            # This maps the file to memory without loading it.
            self.dataset = load_dataset("parquet", data_files={"train": parquet_path}, split="train")
            self.transform = transform

            # Build vocabulary from the first 5000 samples (approximate)
            # or load a pre-computed one if possible to save time.
            # For accuracy, we scan all, but using the dataset iterator is RAM safe.
            print("Building vocabulary (scanning text only)...")
            chars = set()
            for item in tqdm(self.dataset.select(range(min(len(self.dataset), 10000))), desc="Scanning Vocab"):
                txt = item["text"]
                if NORMALIZE_AVAILABLE:
                    txt = normalize(txt)
                chars.update(list(txt))

            self.chars = sorted(list(chars))
            self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
            self.char_to_idx["<BLANK>"] = 0
            self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
            self.vocab_size = len(self.char_to_idx)

            print(f"  Samples: {len(self.dataset)}")
            print(f"  Vocab: {self.vocab_size} characters")

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # Load ONE row from disk
            item = self.dataset[idx]

            # Decode image bytes
            try:
                img_bytes = item["image"]["bytes"]
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                # Fallback for bad images
                image = Image.new("RGB", (512, 64), (255, 255, 255))

            if self.transform:
                image = self.transform(image)

            text = item["text"]
            if NORMALIZE_AVAILABLE:
                text = normalize(text)

            encoded = [self.char_to_idx.get(char, 0) for char in text if char in self.char_to_idx]
            encoded = [x for x in encoded if x != 0]

            # CTC loss will crash if target length is 0
            if len(encoded) == 0:
                encoded = [1]

            return image, torch.tensor(encoded, dtype=torch.long), len(encoded)

    def collate_fn(batch):
        images, targets, target_lengths = zip(*batch)
        images = torch.stack(images, 0)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        # Targets must be concatenated into one long 1D tensor for CTC Loss
        targets = torch.cat(targets)
        return images, targets, target_lengths

    class ImprovedCRNN(nn.Module):
        def __init__(self, vocab_size, hidden_size=256):
            super(ImprovedCRNN, self).__init__()

            # CNN: Feature Extractor
            # We use MaxPool((2,1)) in later layers to squash Height but preserve Width.
            # This is crucial for OCR because text is wide, we need high horizontal resolution.
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.1),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.1),
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.MaxPool2d((2, 1)),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.MaxPool2d((2, 1)),
                nn.Dropout2d(0.1),
            )

            # RNN: Sequence Modeler
            # Bidirectional LSTM allows the model to use context from both left and right
            # to resolve ambiguous characters.
            self.rnn = nn.LSTM(512 * 4, hidden_size, bidirectional=True, num_layers=3, batch_first=True, dropout=0.2)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden_size * 2, vocab_size)

        def forward(self, x):
            conv_out = self.cnn(x)
            batch, channel, height, width = conv_out.size()
            # Reshape CNN output to be (Batch, Width, Features) for the RNN
            conv_out = conv_out.permute(0, 3, 1, 2).contiguous().view(batch, width, channel * height)
            rnn_out, _ = self.rnn(conv_out)
            return self.fc(self.dropout(rnn_out))

    def decode_prediction(output, idx_to_char):
        # Take the highest probability character at each step
        probs = output.softmax(2)
        _, preds = probs.max(2)
        if preds.dim() > 1:
            preds = preds.squeeze(0)
            probs = probs.squeeze(0)

        decoded = []
        prev_char = None
        for i, idx in enumerate(preds):
            idx = idx.item()
            # CTC requires collapsing repeats (AA -> A) and removing blanks
            if idx != 0 and idx != prev_char:
                char = idx_to_char.get(idx, "")
                if char and char != "<BLANK>":
                    decoded.append(char)
            prev_char = idx
        return "".join(decoded)

    def test_predictions(model, dataloader, device, idx_to_char, num_samples=5):
        model.eval()
        images, targets, target_lengths = next(iter(dataloader))
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            # Must move outputs to CPU before decoding because basic python operations
            # interact better with CPU tensors, and it prevents MPS synchronization bugs.
            outputs_cpu = outputs.cpu()

        print("\n" + "=" * 70)
        print("SAMPLE PREDICTIONS")
        print("=" * 70)

        predictions = []
        ground_truths = []
        start = 0
        for length in target_lengths[:num_samples]:
            target_seq = targets[start : start + length]
            gt_text = "".join([idx_to_char[idx.item()] for idx in target_seq])
            ground_truths.append(gt_text)
            start += length

        for i in range(min(num_samples, images.size(0))):
            pred_text = decode_prediction(outputs_cpu[i : i + 1], idx_to_char)
            predictions.append(pred_text)

        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            match = sum(1 for a, b in zip(pred, gt) if a == b)
            acc = (match / max(len(gt), 1)) * 100
            print(f"Truth: {gt}\nPred:  {pred}\nAcc:   {acc:.1f}%\n")

        return 0

    def train_model(model, train_loader, val_loader, device, epochs, lr, save_path):
        # CTCLoss is calculated on CPU to avoid Not a Number errors that often
        # happen on Mac MPS accelerators with this specific loss function.
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001, betas=(0.9, 0.999))
        scaler = torch.amp.GradScaler("cuda")  # Updated for newer PyTorch versions
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.15,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}
        patience = 15
        patience_counter = 0

        print(f"\nStarting training... Output: {save_path}\n")

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

            for images, targets, target_lengths in pbar:
                images = images.to(device)
                optimizer.zero_grad()

                # 2. AutoCast Context (Runs forward pass in FP16)
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    # Log Softmax must run in FP32 for numerical stability
                    log_probs = outputs.permute(1, 0, 2).log_softmax(2)
                    input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    # 3. Scaled Backward Pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    train_loss += loss.item()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                else:
                    print("Warning: NaN loss detected (batch skipped)")

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, targets, target_lengths in tqdm(val_loader, desc="[Val]"):
                    images = images.to(device)
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)

                        # Use same CPU logic for validation to keep metrics consistent
                        log_probs = outputs.permute(1, 0, 2).log_softmax(2)
                        input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
                        loss = criterion(log_probs, targets, input_lengths, target_lengths)

                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            test_predictions(model, val_loader, device, train_loader.dataset.idx_to_char)

            print(
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # We save the optimizer state too, in case you want to resume exact training state later
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val_loss,
                    },
                    save_path,
                )
                print(f"✓ Saved {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("⚠ Early stopping triggered.")
                    break
            print("-" * 70)

        return history

    device = get_device()
    t_train = get_transforms(width=args.width, is_train=True)
    t_val = get_transforms(width=args.width, is_train=False)

    train_ds = KhmerDataset(args.train, transform=t_train)
    val_ds = KhmerDataset(args.val, transform=t_val)

    # Vocab syncing logic simplified for stream mode
    # We use the Train DS vocab for both.
    val_ds.char_to_idx = train_ds.char_to_idx
    val_ds.idx_to_char = train_ds.idx_to_char
    val_ds.vocab_size = train_ds.vocab_size

    vocab_path = args.output.replace(".pth", "_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "char_to_idx": train_ds.char_to_idx,
                "idx_to_char": train_ds.idx_to_char,
                "vocab_size": train_ds.vocab_size,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True
    )

    model = ImprovedCRNN(vocab_size=train_ds.vocab_size).to(device)

    if args.resume:
        print(f"Loading pre-trained weights from {args.resume}...")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("✓ Weights loaded.")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    history = train_model(model, train_loader, val_loader, device, args.epochs, args.lr, args.output)


if __name__ == "__main__":
    main()
