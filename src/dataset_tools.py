"""
Dataset Tools for Khmer OCR
1. Extract images from parquet (for inspection)
2. Add your own images to the dataset
3. Create augmented data
4. Split data with custom naming
"""

import pandas as pd
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path
import json
import argparse

# ============================================================================
# PART 1: EXTRACT IMAGES FROM PARQUET (for inspection/backup)
# ============================================================================


def extract_parquet_to_images(parquet_path, output_dir="extracted_images"):
    """
    Extract images from parquet file to individual image files
    """
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save images and labels
    labels = []
    for idx, row in df.iterrows():
        # Extract image
        img = Image.open(io.BytesIO(row["image"]["bytes"]))

        # Save image
        img_filename = f"{idx:05d}.png"
        img_path = os.path.join(output_dir, img_filename)
        img.save(img_path)

        # Store label
        labels.append({"filename": img_filename, "text": row["text"]})

        if (idx + 1) % 100 == 0:
            print(f"  Extracted {idx + 1}/{len(df)} images...")

    # Save labels to JSON
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"✓ Extracted {len(df)} images to {output_dir}/")
    print(f"✓ Labels saved to {labels_path}")


# ============================================================================
# PART 2: ADD YOUR OWN IMAGES TO DATASET
# ============================================================================


def images_to_parquet(image_dir, labels_file, output_parquet="custom_data.parquet"):
    """
    Convert your own images + labels to parquet format
    """
    print(f"Loading labels from {labels_file}...")
    with open(labels_file, "r", encoding="utf-8") as f:
        labels = json.load(f)

    data = []
    missing = []

    for item in labels:
        filename = item["filename"]
        text = item["text"]
        img_path = os.path.join(image_dir, filename)

        if not os.path.exists(img_path):
            missing.append(filename)
            continue

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")

            # Convert to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            # Add to dataset
            data.append({"image": {"bytes": img_bytes, "path": None}, "text": text})
        except Exception as e:
            print(f"Error processing {filename}: {e}")

        if len(data) % 100 == 0:
            print(f"  Processed {len(data)} images...")

    if missing:
        print(f"⚠ Warning: {len(missing)} images not found")

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_parquet(output_parquet)

    print(f"✓ Created {output_parquet} with {len(df)} samples")
    return output_parquet


# ============================================================================
# PART 3: MERGE DATASETS
# ============================================================================


def merge_parquet_files(parquet_files, output_file="merged_dataset.parquet"):
    """
    Merge multiple parquet files into one
    """
    print(f"Merging {len(parquet_files)} parquet files...")

    dfs = []
    for pq_file in parquet_files:
        df = pd.read_parquet(pq_file)
        print(f"  {pq_file}: {len(df)} samples")
        dfs.append(df)

    # Concatenate
    merged_df = pd.concat(dfs, ignore_index=True)

    # Shuffle
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    merged_df.to_parquet(output_file)

    print(f"✓ Merged dataset saved to {output_file}")
    print(f"  Total samples: {len(merged_df)}")
    return output_file


# ============================================================================
# PART 4: DATA AUGMENTATION
# ============================================================================


def augment_dataset(parquet_path, output_parquet="augmented_data.parquet", num_augments=2):
    """
    Create augmented versions of existing data
    """
    from PIL import ImageEnhance, ImageFilter
    import random

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    augmented_data = []

    for idx, row in df.iterrows():
        # Load original image
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        text = row["text"]

        for aug_idx in range(num_augments):
            aug_img = img.copy()

            # Random rotation (-5 to +5 degrees)
            angle = random.uniform(-5, 5)
            aug_img = aug_img.rotate(angle, fillcolor="white")

            # Random brightness
            brightness = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Brightness(aug_img)
            aug_img = enhancer.enhance(brightness)

            # Random contrast
            contrast = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(aug_img)
            aug_img = enhancer.enhance(contrast)

            # Slight blur (50% chance)
            if random.random() > 0.5:
                aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=0.5))

            # Convert to bytes
            img_buffer = io.BytesIO()
            aug_img.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            augmented_data.append({"image": {"bytes": img_bytes, "path": None}, "text": text})

        if (idx + 1) % 100 == 0:
            print(f"  Augmented {idx + 1}/{len(df)} images...")

    # Create DataFrame
    aug_df = pd.DataFrame(augmented_data)
    aug_df.to_parquet(output_parquet)

    print(f"✓ Created {output_parquet} with {len(aug_df)} augmented samples")
    return output_parquet


# ============================================================================
# PART 5: SPLIT DATASET (UPDATED)
# ============================================================================


def split_dataset(parquet_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, output_prefix="data"):
    """
    Split parquet dataset into train/val/test sets with custom naming.

    Args:
        parquet_path: Input parquet file
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        output_prefix: Prefix for output files (e.g., "data/clean" -> "data/clean_train.parquet")
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate split points
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # Ensure directory exists
    dirname = os.path.dirname(output_prefix)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Define output filenames based on prefix
    train_out = f"{output_prefix}_train.parquet"
    val_out = f"{output_prefix}_val.parquet"
    test_out = f"{output_prefix}_test.parquet"

    # Save
    train_df.to_parquet(train_out)
    val_df.to_parquet(val_out)
    test_df.to_parquet(test_out)

    print(f"✓ Dataset split complete:")
    print(f"  Train: {len(train_df)} samples → {train_out}")
    print(f"  Val:   {len(val_df)} samples → {val_out}")
    print(f"  Test:  {len(test_df)} samples → {test_out}")

    return train_out, val_out, test_out


# ============================================================================
# EXAMPLE USAGE & CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Khmer OCR Dataset Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract images from parquet")
    extract_parser.add_argument("parquet", help="Input parquet file")
    extract_parser.add_argument("-o", "--output", default="extracted_images", help="Output directory")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add your own images to dataset")
    add_parser.add_argument("image_dir", help="Directory with images")
    add_parser.add_argument("labels", help="Labels JSON file")
    add_parser.add_argument("-o", "--output", default="custom_data.parquet", help="Output parquet")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple parquet files")
    merge_parser.add_argument("files", nargs="+", help="Parquet files to merge")
    merge_parser.add_argument("-o", "--output", default="merged_dataset.parquet", help="Output file")

    # Augment command
    aug_parser = subparsers.add_parser("augment", help="Augment dataset")
    aug_parser.add_argument("parquet", help="Input parquet file")
    aug_parser.add_argument("-n", "--num", type=int, default=2, help="Augmentations per image")
    aug_parser.add_argument("-o", "--output", default="augmented_data.parquet", help="Output file")

    # Split command (UPDATED)
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("parquet", help="Input parquet file")
    split_parser.add_argument("-o", "--output", default="dataset", help="Output prefix (e.g., 'data/clean')")
    split_parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    split_parser.add_argument("--val", type=float, default=0.1, help="Val ratio")
    split_parser.add_argument("--test", type=float, default=0.1, help="Test ratio")

    args = parser.parse_args()

    if args.command == "extract":
        extract_parquet_to_images(args.parquet, args.output)
    elif args.command == "add":
        images_to_parquet(args.image_dir, args.labels, args.output)
    elif args.command == "merge":
        merge_parquet_files(args.files, args.output)
    elif args.command == "augment":
        augment_dataset(args.parquet, args.output, args.num)
    elif args.command == "split":
        split_dataset(args.parquet, args.train, args.val, args.test, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
