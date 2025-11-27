import pandas as pd
import re
import argparse
import os


def is_pure_khmer(text):
    """
    Returns True if the text contains ONLY:
    - Khmer characters (Unicode \u1780-\u17ff)
    - Khmer Symbols (\u19e0-\u19ff)
    - Arabic Numbers (0-9)
    - Basic Punctuation (spaces, ?, !, ., etc.)

    Returns False if it contains:
    - English Letters (a-z, A-Z)
    - Math symbols (=, >, <, +, |)
    - Weird artifacts
    """
    if re.search(r"[a-zA-Z]", text):
        return False

    # We disallow: = < > | [ ] { } _ @ # $ % ^ *
    if re.search(r"[=<>|\[\]\{\}_@#\$%\^\*]", text):
        return False

    return True


def clean_parquet(input_path, output_path):
    print(f" Scanning {input_path}...")

    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    original_count = len(df)
    valid_rows = []
    trash_rows = []

    print("Analyzing text labels...")
    for idx, row in df.iterrows():
        text = str(row["text"])
        is_valid, reason = is_pure_khmer(text)

        if is_valid:
            valid_rows.append(row)
        else:
            trash_rows.append({"text": text, "reason": reason})

    if valid_rows:
        clean_df = pd.DataFrame(valid_rows)
    else:
        print("⚠ WARNING: No valid data left! Check your filter settings.")
        return

    clean_df.to_parquet(output_path)

    removed_count = original_count - len(clean_df)

    print("\n" + "=" * 50)
    print("CLEANING REPORT")
    print("=" * 50)
    print(f"Original size: {original_count}")
    print(f"Clean size:    {len(clean_df)}")
    print(f"Removed:       {removed_count} images ({removed_count / original_count * 100:.1f}%)")
    print("=" * 50)

    if trash_rows:
        print("\nEXAMPLES OF REMOVED DATA (The Garbage):")
        print(f"{'TEXT':<30} | {'REASON'}")
        print("-" * 50)
        # Show first 10 and last 5 deleted items
        examples = trash_rows[:10]
        for item in examples:
            print(f"{item['text']:<30} | {item['reason']}")

    print(f"\nSaved cleaned dataset to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to dirty .parquet file (e.g., data/trainset.parquet)")
    parser.add_argument("-o", "--output", default="clean_trainset.parquet", help="Output filename")
    args = parser.parse_args()

    clean_parquet(args.input, args.output)
