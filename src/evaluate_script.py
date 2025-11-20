import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
import os
import argparse
import io
from tqdm import tqdm

# Try to import editdistance
try:
    import editdistance

    CER_AVAILABLE = True
except ImportError:
    print("⚠ Warning: 'editdistance' not found. Install with: pip install editdistance")
    CER_AVAILABLE = False

# Try to import normalizer
try:
    from khmernormalizer import normalize
except ImportError:
    normalize = lambda x: x


class ImprovedCRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(ImprovedCRNN, self).__init__()

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

        self.rnn = nn.LSTM(512 * 4, hidden_size, bidirectional=True, num_layers=3, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        conv_out = self.cnn(x)
        batch, channel, height, width = conv_out.size()
        conv_out = conv_out.permute(0, 3, 1, 2).contiguous().view(batch, width, channel * height)
        rnn_out, _ = self.rnn(conv_out)
        return self.fc(self.dropout(rnn_out))


class OCREvaluator:
    def __init__(self, model_path, vocab_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load Vocab
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
            self.vocab_size = data["vocab_size"]

        # Load Model
        self.model = ImprovedCRNN(vocab_size=self.vocab_size).to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

        self.model.eval()

        # Transform (Matches 64x512)
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def decode(self, output):
        probs = output.softmax(2)
        _, preds = probs.max(2)
        preds = preds.squeeze(0)

        decoded = []
        prev_char = None
        for idx in preds:
            idx = idx.item()
            # Removed confidence check to see raw output
            if idx != 0 and idx != prev_char:
                char = self.idx_to_char.get(idx, "")
                decoded.append(char)
            prev_char = idx
        return "".join(decoded)

    def calculate_cer(self, pred, truth):
        if not CER_AVAILABLE:
            return 0.0
        dist = editdistance.eval(pred, truth)
        length = max(len(truth), 1)
        return min(1.0, dist / length)  # Cap at 1.0 (100%)

    def evaluate_parquet(self, parquet_path, limit=None):
        print(f"\nLoading data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        if limit:
            df = df.head(limit)

        total_cer = 0
        perfect_matches = 0
        results = []

        print("Running evaluation...")
        with torch.no_grad():
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                truth = normalize(row["text"])
                img_bytes = row["image"]["bytes"]
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_tensor = self.transform(image).unsqueeze(0).to(self.device)

                output = self.model(img_tensor)
                pred = self.decode(output)

                cer = self.calculate_cer(pred, truth)
                is_perfect = pred == truth

                total_cer += cer
                if is_perfect:
                    perfect_matches += 1

                # Store non-perfect results
                if cer > 0:
                    results.append({"truth": truth, "pred": pred, "cer": cer})

        avg_cer = (total_cer / len(df)) * 100
        accuracy = (perfect_matches / len(df)) * 100

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Images:     {len(df)}")
        print(f"Perfect Matches:  {perfect_matches}")
        print(f"Sequence Acc:     {accuracy:.2f}%")
        print(f"Avg CER:          {avg_cer:.2f}%")
        print("=" * 60)

        # Sort by worst errors first
        results.sort(key=lambda x: x["cer"], reverse=True)

        print("\nTOP 10 WORST ERRORS (Detailed View):")
        print("-" * 60)

        for i, r in enumerate(results[:10]):
            print(f"#{i + 1} [CER: {r['cer']:.2f}]")
            print(f"  Truth:  {r['truth']}")
            print(f"  Pred:   {r['pred']}")
            print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to .parquet file")
    parser.add_argument("--model", default="models/best_khmer_ocr_model.pth")
    parser.add_argument("--vocab", default="vocab.json")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    evaluator = OCREvaluator(args.model, args.vocab)
    evaluator.evaluate_parquet(args.dataset, args.limit)
