import argparse
import sys
import os
import math
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", required=True, help="Path to trained .pth model")
    parser.add_argument("--vocab", default=None, help="Path to vocab.json")
    parser.add_argument("--width", type=int, default=512, help="Image width used in training (default: 512)")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width")
    parser.add_argument("--debug", action="store_true", help="Show top characters per step")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        return
    if not os.path.exists(args.model):
        print(f"Error: {args.model} not found")
        return

    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    import json
    import numpy as np

    def logsumexp(*args):
        if all(a == -float("inf") for a in args):
            return -float("inf")
        a_max = max(args)
        return a_max + math.log(sum(math.exp(a - a_max) for a in args))

    def ctc_beam_search(log_probs, beam_width=10, blank_token=0):
        log_probs = log_probs.cpu().numpy()
        T, V = log_probs.shape
        beam = {(): (0.0, -float("inf"))}

        for t in range(T):
            next_beam = defaultdict(lambda: (-float("inf"), -float("inf")))
            valid_paths = sorted(beam.items(), key=lambda x: logsumexp(*x[1]), reverse=True)[:beam_width]

            # Optimization: Pick top candidates from current step
            top_indices = log_probs[t].argsort()[-beam_width:]

            for seq, (p_b, p_nb) in valid_paths:
                p_t_blank = log_probs[t, blank_token]
                n_p_b, n_p_nb = next_beam[seq]
                next_beam[seq] = (logsumexp(n_p_b, p_b + p_t_blank, p_nb + p_t_blank), n_p_nb)

                for c in top_indices:
                    if c == blank_token:
                        continue
                    p_t_c = log_probs[t, c]
                    new_seq = seq + (c,)
                    n_p_b, n_p_nb = next_beam[new_seq]

                    if len(seq) > 0 and seq[-1] == c:
                        cur_p_b, cur_p_nb = next_beam[seq]
                        next_beam[seq] = (cur_p_b, logsumexp(cur_p_nb, p_nb + p_t_c))
                        next_beam[new_seq] = (n_p_b, logsumexp(n_p_nb, p_b + p_t_c))
                    else:
                        next_beam[new_seq] = (n_p_b, logsumexp(n_p_nb, p_b + p_t_c, p_nb + p_t_c))
            beam = next_beam

        best_seq = max(beam.items(), key=lambda x: logsumexp(*x[1]))[0]
        return list(best_seq)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Vocab
    vocab_path = args.vocab if args.vocab else args.model.replace(".pth", "_vocab.json")
    if not os.path.exists(vocab_path):
        vocab_path = "vocab.json"

    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
        vocab_size = data["vocab_size"]

    # Load Model
    model = ImprovedCRNN(vocab_size=vocab_size).to(device)
    try:
        cp = torch.load(args.model, map_location=device)
        state = cp["model_state_dict"] if "model_state_dict" in cp else cp
        model.load_state_dict(state)
    except Exception as e:
        print(f"Load error: {e}")
        return
    model.eval()

    # Transform
    transform = transforms.Compose(
        [
            transforms.Resize((64, args.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Run
    img = Image.open(args.image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        log_probs = output.log_softmax(2).squeeze(0)

    # Diagnostic output
    if args.debug:
        print("\n--- DEBUG: Top 3 Predictions per Timestep ---")
        probs = torch.exp(log_probs).cpu().numpy()
        for t in range(probs.shape[0]):
            # Get indices of top 3
            top3_idx = probs[t].argsort()[-3:][::-1]
            top3_probs = probs[t][top3_idx]

            # Only print if the top choice isn't Blank with high confidence
            # (Reduces noise)
            if top3_idx[0] != 0 or top3_probs[0] < 0.9:
                line = f"Step {t:03d}: "
                for idx, p in zip(top3_idx, top3_probs):
                    char = idx_to_char.get(idx, "<BLK>")
                    line += f"'{char}' ({p:.2f})  "
                print(line)

    if args.beam_width > 1:
        path = ctc_beam_search(log_probs, beam_width=args.beam_width)
        result = "".join([idx_to_char.get(idx, "") for idx in path])
    else:
        preds = log_probs.argmax(1)
        result = ""
        prev = None
        for idx in preds:
            idx = idx.item()
            if idx != 0 and idx != prev:
                result += idx_to_char.get(idx, "")
            prev = idx

    print("-" * 50)
    print(f"Result: {result}")
    print("-" * 50)


if __name__ == "__main__":
    main()
