import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import argparse


class ImprovedCRNN(nn.Module):
    """
    Must match the class used in train.py exactly.
    """

    def __init__(self, vocab_size, hidden_size=256):
        super(ImprovedCRNN, self).__init__()

        # CNN Structure (Matches your optimized script)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.1),
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Block 6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.1),
        )

        # RNN - Note: num_layers=3 to match training
        self.rnn = nn.LSTM(
            512 * 4,  # Input size
            hidden_size,
            bidirectional=True,
            num_layers=3,  # MUST BE 3
            batch_first=True,
            dropout=0.2,
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        conv_out = self.cnn(x)
        batch, channel, height, width = conv_out.size()
        conv_out = conv_out.permute(0, 3, 1, 2)
        conv_out = conv_out.contiguous().view(batch, width, channel * height)
        rnn_out, _ = self.rnn(conv_out)
        rnn_out = self.dropout(rnn_out)
        output = self.fc(rnn_out)
        return output


# ============================================================================
# 2. THE INFERENCE CLASS
# ============================================================================


class KhmerOCR:
    def __init__(self, model_path, vocab_path="vocab.json", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load vocabulary
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                self.char_to_idx = vocab_data["char_to_idx"]
                # Ensure keys are integers
                self.idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
                self.vocab_size = vocab_data["vocab_size"]
        except FileNotFoundError:
            print(f"Error: Could not find {vocab_path}")
            exit(1)

        # Load model
        # IMPORTANT: Using ImprovedCRNN now
        self.model = ImprovedCRNN(vocab_size=self.vocab_size).to(self.device)

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle both full checkpoint dict and just state_dict
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Did you train with ImprovedCRNN but are trying to load a different architecture?")
            exit(1)

        self.model.eval()

        # Image preprocessing (Must match training resize)
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 512)),  # Fixed width 512
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        print(f"Model loaded successfully! Vocab size: {self.vocab_size}")

    def decode_prediction(self, output):
        """Decode without confidence threshold for raw visibility"""
        probs = output.softmax(2)
        _, preds = probs.max(2)
        preds = preds.squeeze(0)

        decoded = []
        prev_char = None

        for idx in preds:
            idx = idx.item()
            # removed confidence check to see everything
            if idx != 0 and idx != prev_char:
                char = self.idx_to_char.get(idx, "")
                if char and char != "<BLANK>":
                    decoded.append(char)
            prev_char = idx

        return "".join(decoded)

    def predict(self, image_path):
        if isinstance(image_path, str):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                return f"Error opening image: {e}"
        else:
            image = image_path.convert("RGB")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = self.decode_prediction(output)

        return prediction


# ============================================================================
# 3. MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--model", type=str, default="models/best_khmer_ocr_model.pth")
    parser.add_argument("--vocab", type=str, default="vocab.json")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    ocr = KhmerOCR(args.model, args.vocab, args.device)

    print(f"\nProcessing: {args.image}")
    result = ocr.predict(args.image)
    print(f"Prediction: {result}")


if __name__ == "__main__":
    main()
