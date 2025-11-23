import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", required=True, help="Path to trained .pth model")
    parser.add_argument("--vocab", default=None, help="Path to vocab.json (optional)")
    parser.add_argument("--width", type=int, default=512, help="Image width used in training (default: 512)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    print("Loading libraries... (this may take a moment)")

    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    import json

    class CRNN(nn.Module):
        def __init__(self, vocab_size, hidden_size=256):
            super(CRNN, self).__init__()
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

    class KhmerOCR:
        def __init__(self, model_path, vocab_path=None, width=512, device=None):
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            if vocab_path is None:
                vocab_path = model_path.replace(".pth", "_vocab.json")
                if not os.path.exists(vocab_path):
                    vocab_path = "vocab.json"

            print(f"Loading Model: {model_path}")

            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
                    self.vocab_size = data["vocab_size"]
            except Exception as e:
                print(f"Error loading vocab: {e}")
                exit(1)

            self.model = CRNN(vocab_size=self.vocab_size).to(self.device)
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Error loading model weights: {e}")
                exit(1)

            self.model.eval()

            self.transform = transforms.Compose(
                [
                    transforms.Resize((64, width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        def predict(self, image_path):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                return f"Error: {e}"

            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(image_tensor)

                probs = output.softmax(2)
                _, preds = probs.max(2)
                preds = preds.squeeze(0)

                decoded = []
                prev_char = None
                for idx in preds:
                    idx = idx.item()
                    if idx != 0 and idx != prev_char:
                        char = self.idx_to_char.get(idx, "")
                        decoded.append(char)
                    prev_char = idx

            return "".join(decoded)

    ocr = KhmerOCR(args.model, args.vocab, args.width)
    result = ocr.predict(args.image)

    print("-" * 50)
    print(f"Result: {result}")
    print("-" * 50)


if __name__ == "__main__":
    main()
