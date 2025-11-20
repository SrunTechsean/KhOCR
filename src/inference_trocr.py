import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Inference with TrOCR")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="models/trocr_khmer", help="Path to model FOLDER (not .pth file)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    # Check paths before loading heavy libraries
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    if not os.path.exists(args.model):
        print(f"Error: Model directory not found at {args.model}")
        return

    print(f"Loading AI libraries...")

    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model from: {args.model} ...")

    try:
        # TrOCR loads from a DIRECTORY, finding config.json and pytorch_model.bin automatically
        processor = TrOCRProcessor.from_pretrained(args.model)
        model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
        model.eval()  # Set to evaluation mode
    except Exception as e:
        print(f"\nFailed to load model. Did you pass the folder path?\nError: {e}")
        return

    try:
        image = Image.open(args.image).convert("RGB")

        # Preprocess
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("-" * 50)
        print(f"RESULT: {generated_text}")
        print("-" * 50)

    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    main()
