import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import torch.nn.functional as F


# ==========================================
# PASTE YOUR CORRECTED CRNN CLASS HERE
# (Make sure it has the 512 * 4 fix)
# ==========================================
class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),
        )
        # ENSURE THIS IS 512 * 4
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


# ==========================================
# DEBUGGING SCRIPT
# ==========================================
def debug(image_path, model_path, vocab_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Vocab
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}

    # 2. Load Model
    model = CRNN(vocab_size=vocab["vocab_size"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.eval()

    # 3. Load Image
    transform = transforms.Compose(
        [
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 4. Predict
    with torch.no_grad():
        output = model(img_tensor)  # [1, Width, Vocab]
        probs = F.softmax(output, dim=2)  # Convert to percentages

    # 5. PRINT THE RAW "BRAIN" OUTPUT
    print(f"\nAnalyzing: {image_path}")
    print("-" * 40)
    print(f"{'Step':<5} | {'Pred Idx':<10} | {'Char':<10} | {'Confidence'}")
    print("-" * 40)

    output_seq = output.max(2)[1].squeeze().cpu().numpy()
    probs_seq = probs.max(2)[0].squeeze().cpu().numpy()

    for i, (idx, conf) in enumerate(zip(output_seq, probs_seq)):
        char = idx_to_char.get(idx, "<BLANK>")
        # Only print if it's NOT blank or if it changes
        if idx != 0:
            print(f"{i:<5} | {idx:<10} | {char:<10} | {conf:.4f}")
        elif i % 5 == 0:  # Print a blank every few steps just to see
            print(f"{i:<5} | {idx:<10} | {'<BLANK>':<10} | {conf:.4f}")


# Run it
debug("test_image.jpg", "models/best_khmer_ocr_model.pth", "vocab.json")
