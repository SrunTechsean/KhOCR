import argparse
import os
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
    ViTImageProcessor
)

def main():
    parser = argparse.ArgumentParser(description="Universal TrOCR Training Script (Khmer)")
    
    # Data Arguments
    parser.add_argument("--train", required=True, help="Path to training parquet")
    parser.add_argument("--val", required=True, help="Path to validation parquet")
    parser.add_argument("--output", required=True, help="Directory to save the model")
    
    # Model Arguments
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to local model folder (for Stage 2). If None, builds fresh Hybrid model (Stage 1).")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (faster on GPU)")

    args = parser.parse_args()
    
    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Device: {device}")

    # ---------------------------------------------------------
    # 1. TEXT NORMALIZATION (FIXING UNICODE & COENG)
    # ---------------------------------------------------------
    try:
        from khmernormalizer import normalize as kh_normalize
    except ImportError:
        print("⚠ Warning: khmernormalizer not found. `pip install khmernormalizer`")
        kh_normalize = lambda x: x

    def clean_text(text):
        if text is None: return ""
        # 1. Apply standard Khmer normalization (fixes Coeng order)
        text = kh_normalize(text)
        # 2. Remove Zero Width Spaces (\u200b) and other invisible formatters
        # These kill CER scores but are invisible
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
        # 3. Strip whitespace
        return text.strip()

    # ---------------------------------------------------------
    # 2. MODEL LOADING LOGIC
    # ---------------------------------------------------------
    # Always need these for data processing
    # Use stage1 for the visual backbone as it is more generic than 'handwritten'
    feature_extractor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-stage1")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    if args.checkpoint:
        print(f"🔄 STAGE 2: Loading Pre-trained model from: {args.checkpoint}")
        # Load the weights you trained in Stage 1
        model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)
    else:
        print("🆕 STAGE 1: Building Fresh Hybrid Model (ViT + XLM-R)")
        # Create the Hybrid from scratch
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "microsoft/trocr-base-stage1", 
            "xlm-roberta-base"
        )
        
        # Set Model Configs for Fresh Model
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.no_repeat_ngram_size = 0  # Important for Khmer
        model.config.num_beams = 4

    # Ensure config matches dataset constraints
    model.config.max_length = args.max_len
    model.config.early_stopping = True
    model.to(device)

    # ---------------------------------------------------------
    # 3. DATASET PREPARATION
    # ---------------------------------------------------------
    def load_parquet_dataset(parquet_path):
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        # Apply the cleaning
        df["text"] = df["text"].apply(clean_text)
        df = df[df["text"].str.len() > 0] # Remove empty labels

        def gen():
            for idx, row in df.iterrows():
                try:
                    # Handle different parquet structures (sometimes image is dict, sometimes bytes)
                    img_data = row["image"]
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                    else:
                        # If it's already raw bytes or a path (adjust based on your parquet)
                        image = Image.open(io.BytesIO(img_data)).convert("RGB")
                        
                    yield {"image": image, "text": row["text"]}
                except Exception as e:
                    continue

        return Dataset.from_generator(gen)

    train_dataset = load_parquet_dataset(args.train)
    eval_dataset = load_parquet_dataset(args.val)

    def process_data(examples):
        pixel_values = processor(images=examples["image"], return_tensors="pt").pixel_values
        labels = tokenizer(
            examples["text"], 
            padding="max_length", 
            max_length=args.max_len,
            truncation=True
        ).input_ids
        labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
        return {"pixel_values": pixel_values, "labels": labels}

    print(f"Processing {len(train_dataset)} training examples...")
    train_dataset = train_dataset.map(process_data, batched=True, remove_columns=["image", "text"])
    eval_dataset = eval_dataset.map(process_data, batched=True, remove_columns=["image", "text"])

    # ---------------------------------------------------------
    # 4. METRICS & TRAINING
    # ---------------------------------------------------------
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        fp16=args.fp16,
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        warmup_ratio=0.05, 
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output}...")
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)

if __name__ == "__main__":
    main()