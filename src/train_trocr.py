import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

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
        AutoTokenizer,
        ViTImageProcessor,
        XLMRobertaForCausalLM
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
    else:
        device = "cpu"
        
    print(f"✓ Device: {device}")

    # 1. Normalization
    try:
        from khmernormalizer import normalize as kh_normalize
    except ImportError:
        kh_normalize = lambda x: x

    def clean_text(text):
        if text is None: return ""
        text = kh_normalize(text)
        return text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '').strip()

    # 2. Model Setup
    feature_extractor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-stage1")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    if args.checkpoint:
        print(f"🔄 Loading Checkpoint: {args.checkpoint}")
        model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)
    else:
        print("🆕 Building Fresh Hybrid Model")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
        decoder = XLMRobertaForCausalLM.from_pretrained("xlm-roberta-base", is_decoder=True, add_cross_attention=True)
        model.decoder = decoder
        model.config.decoder = model.decoder.config
        model.config.vocab_size = model.decoder.config.vocab_size
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.no_repeat_ngram_size = 0
        model.config.num_beams = 4

    model.config.max_length = args.max_len
    model.config.early_stopping = True
    model.to(device)

    # 3. Data Loading
    def load_parquet_dataset(parquet_path):
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        df["text"] = df["text"].apply(clean_text)
        df = df[df["text"].str.len() > 0] 

        def gen():
            for idx, row in df.iterrows():
                try:
                    img_data = row["image"]
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                    else:
                        image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    yield {"image": image, "text": row["text"]}
                except Exception as e: continue
        return Dataset.from_generator(gen)

    train_dataset = load_parquet_dataset(args.train)
    eval_dataset = load_parquet_dataset(args.val)

    # --- DISK SPACE FIX: Use set_transform instead of map ---
    def transform(examples):
        pixel_values = processor(images=examples["image"], return_tensors="pt").pixel_values
        labels = tokenizer(examples["text"], padding="max_length", max_length=args.max_len, truncation=True).input_ids
        labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
        return {"pixel_values": pixel_values, "labels": labels}

    print("Setting up On-the-Fly Transforms (Saves Disk Space)...")
    train_dataset.set_transform(transform)
    eval_dataset.set_transform(transform)
    # --------------------------------------------------------

    # 4. Metrics & Training
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}

    args_train = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        fp16=args.fp16,
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        warmup_ratio=0.05,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model, tokenizer=tokenizer, feature_extractor=feature_extractor,
        args=args_train, compute_metrics=compute_metrics,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"✓ Model saved to {args.output}")

if __name__ == "__main__":
    main()