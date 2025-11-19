import sys
import subprocess


def check_python_version():
    """Verify Python version"""
    print("1. Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False


def check_packages():
    """Check if required packages are installed"""
    print("\n2. Checking required packages...")
    required = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "pandas": "Pandas",
        "PIL": "Pillow",
        "numpy": "NumPy",
        "tqdm": "tqdm",
    }

    all_ok = True
    for package, name in required.items():
        try:
            __import__(package)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} (run: pip install {name.lower()})")
            all_ok = False

    return all_ok


def check_gpu():
    """Check GPU availability"""
    print("\n3. Checking GPU...")
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✓ GPU: {gpu_name}")
            print(f"   ✓ VRAM: {gpu_memory:.1f} GB")

            # Test GPU computation
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            print(f"   ✓ GPU computation test passed")
            return True
        else:
            print(f"   ⚠ No GPU detected (will use CPU - training will be SLOW)")
            return False
    except Exception as e:
        print(f"   ✗ GPU check failed: {e}")
        return False


def check_dataset():
    """Check if dataset files exist"""
    print("\n4. Checking dataset files...")
    import os

    files = ["trainset.parquet", "valset.parquet", "testset.parquet"]
    locations = ["data/", "./"]

    found = {}
    for file in files:
        found[file] = False
        for loc in locations:
            path = os.path.join(loc, file)
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1e6
                print(f"   ✓ {file} ({size_mb:.1f} MB)")
                found[file] = True
                break

        if not found[file]:
            print(f"   ✗ {file} not found")

    return all(found.values())


def test_data_loading():
    """Test loading a small batch of data"""
    print("\n5. Testing data loading...")
    try:
        import pandas as pd
        from PIL import Image
        import io
        import os

        # Find trainset
        if os.path.exists("data/trainset.parquet"):
            path = "data/trainset.parquet"
        elif os.path.exists("trainset.parquet"):
            path = "trainset.parquet"
        else:
            print(f"   ✗ Cannot find trainset.parquet")
            return False

        # Load small sample
        df = pd.read_parquet(path)
        print(f"   ✓ Loaded {len(df)} samples")

        # Test loading first image
        row = df.iloc[0]
        img = Image.open(io.BytesIO(row["image"]["bytes"]))
        text = row["text"]

        print(f"   ✓ Sample image size: {img.size}")
        print(f"   ✓ Sample text: {text}")

        # Check vocabulary
        all_text = " ".join(df["text"].values)
        unique_chars = set(all_text)
        print(f"   ✓ Vocabulary size: {len(unique_chars)} characters")

        return True

    except Exception as e:
        print(f"   ✗ Data loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_creation():
    """Test if model can be created"""
    print("\n6. Testing model creation...")
    try:
        import torch
        import torch.nn as nn

        # Simple CRNN for testing
        class TestCRNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.lstm = nn.LSTM(64, 128, bidirectional=True)
                self.fc = nn.Linear(256, 100)

            def forward(self, x):
                x = self.conv(x)
                return x

        model = TestCRNN()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Test forward pass
        x = torch.randn(1, 3, 64, 256).to(device)
        out = model(x)

        print(f"   ✓ Model created successfully")
        print(f"   ✓ Test forward pass successful")
        print(f"   ✓ Output shape: {out.shape}")

        return True

    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def estimate_training_time():
    """Estimate training time"""
    print("\n7. Estimating training time...")
    import torch

    if torch.cuda.is_available():
        # Rough estimates for RTX 3050
        print(f"   • Training 4,200 images for 30 epochs")
        print(f"   • Estimated time: 6-10 hours")
        print(f"   • ~12-20 minutes per epoch")
        print(f"   • Recommendation: Start training before bed!")
    else:
        print(f"   • CPU training will take 3-5x longer")
        print(f"   • Estimated time: 20-40 hours")
        print(f"   • Recommendation: Use GPU if possible!")


def print_next_steps(all_tests_passed):
    """Print what to do next"""
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ ALL CHECKS PASSED! You're ready to start training!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run: python train.py")
        print("2. Monitor GPU: watch -n 1 nvidia-smi")
        print("3. Check progress: tail -f training.log")
        print("\nTraining will take 6-10 hours. Go do something else!")
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before training.")
        print("Common fixes:")
        print("• Install missing packages: pip install -r requirements.txt")
        print("• Download dataset: huggingface-cli download SoyVitou/khmer-handwritten-dataset-4.2k")
        print("• Check GPU drivers: nvidia-smi")


def main():
    print("=" * 60)
    print("KHMER OCR - PRE-TRAINING CHECKS")
    print("=" * 60)
    print("This will verify everything is set up correctly.\n")

    results = []

    results.append(check_python_version())
    results.append(check_packages())
    results.append(check_gpu())
    results.append(check_dataset())
    results.append(test_data_loading())
    results.append(test_model_creation())
    estimate_training_time()

    all_passed = all(results)
    print_next_steps(all_passed)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
