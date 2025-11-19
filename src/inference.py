import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import argparse


class CRNN(nn.Module):
    """Same architecture as training"""

    def __init__(self, vocab_size, hidden_size=256):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )

        self.rnn = nn.LSTM(512 * 4, hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        conv_out = self.cnn(x)
        batch, channel, height, width = conv_out.size()
        conv_out = conv_out.permute(0, 3, 1, 2)
        conv_out = conv_out.contiguous().view(batch, width, channel * height)
        rnn_out, _ = self.rnn(conv_out)
        output = self.fc(rnn_out)
        return output


class KhmerOCR:
    """Easy-to-use OCR inference class"""

    def __init__(self, model_path, vocab_path="vocab.json", device=None):
        """
        Args:
            model_path: Path to trained model (.pth file)
            vocab_path: Path to vocabulary JSON file
            device: 'cuda' or 'cpu' (auto-detects if None)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            self.char_to_idx = vocab_data["char_to_idx"]
            self.idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
            self.vocab_size = vocab_data["vocab_size"]

        # Load model
        self.model = CRNN(vocab_size=self.vocab_size).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        print(f"Model loaded successfully! Vocab size: {self.vocab_size}")

    def decode_prediction(self, output):
        """Decode CTC output to text using greedy decoding"""
        _, preds = output.max(2)
        preds = preds.squeeze(0)  # Remove batch dimension

        decoded = []
        prev_char = None

        for idx in preds:
            idx = idx.item()
            if idx != 0 and idx != prev_char:  # Not blank and not duplicate
                char = self.idx_to_char.get(idx, "")
                if char:
                    decoded.append(char)
            prev_char = idx

        return "".join(decoded)

    def predict(self, image_path):
        """
        Predict text from an image

        Args:
            image_path: Path to image file or PIL Image object

        Returns:
            Predicted Khmer text
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = self.decode_prediction(output)

        return prediction

    def predict_batch(self, image_paths):
        """
        Predict text from multiple images

        Args:
            image_paths: List of image paths

        Returns:
            List of predicted texts
        """
        predictions = []
        for img_path in image_paths:
            pred = self.predict(img_path)
            predictions.append(pred)
        return predictions


def save_vocabulary(train_dataset, output_path="vocab.json"):
    """Save vocabulary for inference"""
    vocab_data = {
        "char_to_idx": train_dataset.char_to_idx,
        "idx_to_char": train_dataset.idx_to_char,
        "vocab_size": train_dataset.vocab_size,
        "chars": train_dataset.chars,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary saved to {output_path}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Khmer OCR Inference")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--model", type=str, default="best_khmer_ocr_model.pth", help="Path to model file")
    parser.add_argument("--vocab", type=str, default="vocab.json", help="Path to vocabulary file")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Initialize OCR
    ocr = KhmerOCR(args.model, args.vocab, args.device)

    # Predict
    print(f"\nProcessing: {args.image}")
    prediction = ocr.predict(args.image)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    # Example usage without command line
    # ocr = KhmerOCR('best_khmer_ocr_model.pth', 'vocab.json')
    # result = ocr.predict('test_image.jpg')
    # print(result)

    main()
