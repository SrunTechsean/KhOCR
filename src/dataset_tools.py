import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Khmer OCR Dataset Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 1. CONVERT (Excel -> JSON)
    convert_parser = subparsers.add_parser("convert", help="Convert Excel labels to JSON")
    convert_parser.add_argument("excel", help="Input .xlsx file")
    convert_parser.add_argument("-o", "--output", default="labels.json", help="Output .json file")

    # 2. EXTRACT
    extract_parser = subparsers.add_parser("extract", help="Extract images from parquet")
    extract_parser.add_argument("parquet", help="Input parquet file")
    extract_parser.add_argument("-o", "--output", default="extracted_images", help="Output directory")

    # 3. ADD
    add_parser = subparsers.add_parser("add", help="Add images to dataset")
    add_parser.add_argument("image_dir", help="Directory with images")
    add_parser.add_argument("labels", help="Labels file (.json)")
    add_parser.add_argument("-o", "--output", default="custom_data.parquet", help="Output parquet")

    # 4. MERGE
    merge_parser = subparsers.add_parser("merge", help="Merge parquet files")
    merge_parser.add_argument("files", nargs="+", help="Parquet files to merge")
    merge_parser.add_argument("-o", "--output", default="merged_dataset.parquet", help="Output file")

    # 5. AUGMENT
    aug_parser = subparsers.add_parser("augment", help="Augment dataset")
    aug_parser.add_argument("parquet", help="Input parquet file")
    aug_parser.add_argument("-n", "--num", type=int, default=2, help="Augmentations per image")
    aug_parser.add_argument("-o", "--output", default="augmented_data.parquet", help="Output file")

    # 6. SPLIT
    split_parser = subparsers.add_parser("split", help="Split dataset")
    split_parser.add_argument("parquet", help="Input parquet file")
    split_parser.add_argument("-o", "--output", default="dataset", help="Output prefix")
    split_parser.add_argument("--train", type=float, default=0.8)
    split_parser.add_argument("--val", type=float, default=0.1)
    split_parser.add_argument("--test", type=float, default=0.1)

    args = parser.parse_args()

    if args.command == "convert":
        convert_excel_to_json(args.excel, args.output)
    elif args.command == "extract":
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


def convert_excel_to_json(excel_path, output_path="labels.json"):
    """Converts an Excel file to the required JSON format"""
    print(f"Loading {excel_path}...")
    try:
        import pandas as pd
        import json
    except ImportError:
        print("Error: Pandas is required. Run: pip install pandas openpyxl")
        return

    try:
        # Load Excel
        df = pd.read_excel(excel_path)

        # Normalize column names (lowercase, strip spaces)
        df.columns = [c.lower().strip() for c in df.columns]

        # Check for required columns
        if "filename" not in df.columns or "text" not in df.columns:
            print("Error: Excel file must have 'filename' and 'text' columns.")
            print(f"Found columns: {list(df.columns)}")
            return

        # Clean data
        # Ensure filename is string and text is string
        df["filename"] = df["filename"].astype(str)
        df["text"] = df["text"].astype(str)

        # Convert to list of dicts
        data = df[["filename", "text"]].to_dict("records")

        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✓ Converted {len(data)} rows to {output_path}")
    except Exception as e:
        print(f"Error converting excel: {e}")


def extract_parquet_to_images(parquet_path, output_dir="extracted_images"):
    import pandas as pd
    from PIL import Image
    import io
    import json

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    os.makedirs(output_dir, exist_ok=True)
    labels = []

    print(f"Extracting {len(df)} images...")
    for idx, row in df.iterrows():
        img = Image.open(io.BytesIO(row["image"]["bytes"]))
        img_filename = f"{idx:05d}.png"
        img_path = os.path.join(output_dir, img_filename)
        img.save(img_path)
        labels.append({"filename": img_filename, "text": row["text"]})

    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"✓ Extracted to {output_dir}/")


def images_to_parquet(image_dir, labels_file, output_parquet="custom_data.parquet"):
    import pandas as pd
    from PIL import Image
    import io
    import json

    print(f"Loading labels from {labels_file}...")
    with open(labels_file, "r", encoding="utf-8") as f:
        labels = json.load(f)

    data = []
    missing = []

    print(f"Processing {len(labels)} items...")
    for item in labels:
        filename = item["filename"]
        text = item["text"]
        img_path = os.path.join(image_dir, filename)

        if not os.path.exists(img_path):
            missing.append(filename)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            data.append({"image": {"bytes": img_bytes, "path": None}, "text": text})
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if missing:
        print(f"⚠️ Warning: {len(missing)} images not found")

    df = pd.DataFrame(data)
    df.to_parquet(output_parquet)
    print(f"✓ Created {output_parquet} with {len(df)} samples")


def merge_parquet_files(parquet_files, output_file="merged_dataset.parquet"):
    import pandas as pd

    print(f"Merging {len(parquet_files)} files...")
    dfs = []
    for pq_file in parquet_files:
        df = pd.read_parquet(pq_file)
        print(f"  {pq_file}: {len(df)} samples")
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    merged_df.to_parquet(output_file)

    print(f"✓ Saved to {output_file} ({len(merged_df)} total samples)")


def augment_dataset(parquet_path, output_parquet, num_augments):
    import pandas as pd
    from PIL import Image, ImageEnhance, ImageFilter
    import io
    import random

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    augmented_data = []

    print(f"Augmenting {len(df)} images x {num_augments}...")
    for idx, row in df.iterrows():
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        text = row["text"]

        for _ in range(num_augments):
            aug_img = img.copy()

            # Rotation
            aug_img = aug_img.rotate(random.uniform(-5, 5), fillcolor="white")

            # Brightness
            aug_img = ImageEnhance.Brightness(aug_img).enhance(random.uniform(0.8, 1.2))

            # Contrast
            aug_img = ImageEnhance.Contrast(aug_img).enhance(random.uniform(0.8, 1.2))

            # Blur (Low chance)
            if random.random() > 0.7:
                aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=0.5))

            img_buffer = io.BytesIO()
            aug_img.save(img_buffer, format="PNG")
            augmented_data.append({"image": {"bytes": img_buffer.getvalue(), "path": None}, "text": text})

    aug_df = pd.DataFrame(augmented_data)
    aug_df.to_parquet(output_parquet)
    print(f"✓ Created {output_parquet} with {len(aug_df)} augmented samples")

def split_dataset(parquet_path, train_ratio, val_ratio, test_ratio, output_prefix):
    import pandas as pd

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    dirname = os.path.dirname(output_prefix)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    train_out = f"{output_prefix}_train.parquet"
    val_out = f"{output_prefix}_val.parquet"
    test_out = f"{output_prefix}_test.parquet"

    train_df.to_parquet(train_out)
    val_df.to_parquet(val_out)
    test_df.to_parquet(test_out)

    print(f"✓ Split Complete:")
    print(f"  Train: {len(train_df)} -> {train_out}")
    print(f"  Val:   {len(val_df)} -> {val_out}")
    print(f"  Test:  {len(test_df)} -> {test_out}")


if __name__ == "__main__":
    main()