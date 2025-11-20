import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Train TrOCR for Khmer")
    parser.add_argument("--train", required=True, help="Path to training parquet")
    parser.add_argument("--val", required=True, help="Path to validation parquet")
    parser.add_argument("--output", default="models/trocr_khmer", help="Output directory")
    parser.add_argument("--base_model", default="microsoft/trocr-base-stage1", help="Base TrOCR model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")

    args = parser.parse_args()

    print("=" * 60)
    print(f"TrOCR TRAINING CONFIGURATION")
    print(f"Train Data:   {args.train}")
    print(f"Val Data:     {args.val}")
    print(f"Output Dir:   {args.output}")
    print(f"Batch Size:   {args.batch}")
    print("=" * 60)
    print("Loading AI libraries (this takes a moment)...")

    import pandas as pd
    import torch
    from PIL import Image
    import io
    from datasets import Dataset
    import evaluate
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        default_data_collator,
    )

    # Check if the program is using mps for my mac
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✓ Using device: {device}")

    try:
        from khmernormalizer import normalize

        print("✓ Using khmernormalizer")
    except ImportError:
        print("⚠ khmernormalizer not found. Install via: pip install khmernormalizer")
        normalize = lambda x: x

    def load_parquet_dataset(parquet_path):
        """Loads parquet and converts to HF Dataset format"""
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        df["text"] = df["text"].apply(normalize)

        def gen():
            for idx, row in df.iterrows():
                try:
                    img_bytes = row["image"]["bytes"]
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    yield {"image": image, "text": row["text"]}
                except Exception as e:
                    print(f"Skipping bad image at index {idx}: {e}")
                    continue

        return Dataset.from_generator(gen)

    train_dataset = load_parquet_dataset(args.train)
    eval_dataset = load_parquet_dataset(args.val)

    processor = TrOCRProcessor.from_pretrained(args.base_model)

    def process_data(examples):
        pixel_values = processor(images=examples["image"], return_tensors="pt").pixel_values

        labels = processor(text=examples["text"], padding="max_length", max_length=args.max_len).input_ids

        labels = [[(l if l != processor.tokenizer.pad_token_id else -100) for l in label] for label in labels]

        return {"pixel_values": pixel_values, "labels": labels}

    print("Processing datasets...")
    train_dataset = train_dataset.map(process_data, batched=True, remove_columns=["image", "text"])
    eval_dataset = eval_dataset.map(process_data, batched=True, remove_columns=["image", "text"])

    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = args.max_len
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"✓ Model saved to {args.output}")


if __name__ == "__main__":
    main()
