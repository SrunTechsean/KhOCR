import argparse
import sys
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
    AutoFeatureExtractor,
    AutoModelForCausalLM
)

def main():
    parser = argparse.ArgumentParser(description="Train TrOCR for Khmer")
    # ... (Your arguments remain the same) ...
    parser.add_argument("--train", required=True, help="Path to training parquet")
    parser.add_argument("--val", required=True, help="Path to validation parquet")
    parser.add_argument("--output", default="models/trocr_khmer_xlmr", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--max_len", type=int, default=128)
    
    args = parser.parse_args()
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")

    try:
        from khmernormalizer import normalize
    except ImportError:
        normalize = lambda x: x

    # ---------------------------------------------------------
    # 1. FIX: CREATE HYBRID ARCHITECTURE
    # ---------------------------------------------------------
    print("Constructing Hybrid Model (ViT Encoder + XLM-RoBERTa Decoder)...")
    
    # A. Load the Vision part (Encoder) from TrOCR
    # We use the feature extractor to handle image resizing/normalization
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/trocr-base-stage1")
    
    # B. Load the Text part (Decoder) from XLM-RoBERTa (Knows Khmer natively!)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # C. Combine them into a Processor
    processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # D. Create the Model using Pre-trained weights from both sources
    # This downloads the Vision weights from Microsoft and Language weights from Facebook/Meta
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "microsoft/trocr-base-stage1", 
        "xlm-roberta-base"
    )

    # Important: Set special tokens for the model configuration
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # ---------------------------------------------------------
    # 2. CONFIG FIXES FOR KHMER
    # ---------------------------------------------------------
    model.config.max_length = args.max_len
    model.config.early_stopping = True
    model.config.num_beams = 4
    
    # CRITICAL FIX: Remove no_repeat_ngram_size
    # Khmer has many repeating vowels/subscripts. 
    # Preventing repeats will cause the model to predict wrong chars.
    model.config.no_repeat_ngram_size = 0 
    
    model.to(device)
    print("✓ Hybrid Model Loaded successfully")

    # ---------------------------------------------------------
    # DATA LOADING (Same as your code)
    # ---------------------------------------------------------
    def load_parquet_dataset(parquet_path):
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        df["text"] = df["text"].apply(normalize)
        
        # Filter empty text to prevent crashes
        df = df[df["text"].str.len() > 0]

        def gen():
            for idx, row in df.iterrows():
                try:
                    img_bytes = row["image"]["bytes"]
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    yield {"image": image, "text": row["text"]}
                except Exception as e:
                    continue
        return Dataset.from_generator(gen)

    train_dataset = load_parquet_dataset(args.train)
    eval_dataset = load_parquet_dataset(args.val)

    def process_data(examples):
        # Setup for images
        pixel_values = processor(images=examples["image"], return_tensors="pt").pixel_values
        
        # Setup for text (using XLM-R tokenizer)
        labels = tokenizer(
            examples["text"], 
            padding="max_length", 
            max_length=args.max_len,
            truncation=True
        ).input_ids
        
        # Replace padding with -100 to ignore in loss
        labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
        
        return {"pixel_values": pixel_values, "labels": labels}

    print("Processing datasets...")
    train_dataset = train_dataset.map(process_data, batched=True, remove_columns=["image", "text"])
    eval_dataset = eval_dataset.map(process_data, batched=True, remove_columns=["image", "text"])

    # ---------------------------------------------------------
    # TRAINING SETUP
    # ---------------------------------------------------------
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        # Debug print to see if it's learning
        if len(pred_str) > 0:
            print(f"Pred: {pred_str[0]}")
            print(f"True: {label_str[0]}")

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,       # Check more frequently for small datasets
        save_steps=500,
        logging_steps=100,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
        # Crucial parameters for fine-tuning pre-trained models
        warmup_ratio=0.1,
        weight_decay=0.01,
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
    # Try to resume if checkpoint exists, otherwise start fresh
    try:
        trainer.train(resume_from_checkpoint=True)
    except:
        print("No valid checkpoint found, starting fresh...")
        trainer.train()

    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"✓ Model saved to {args.output}")

if __name__ == "__main__":
    main()