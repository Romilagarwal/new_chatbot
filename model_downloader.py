import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import time
from huggingface_hub import login

# Replace with your actual token
def check_requirements():
    """Check if system meets requirements"""
    print("\n" + "="*60)
    print("SYSTEM REQUIREMENTS CHECK")
    print("="*60)

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"  └─ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  └─ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  ⚠ WARNING: CUDA not available, will use CPU only (SLOW)")

    # Check disk space
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        import shutil
        total, used, free = shutil.disk_usage(cache_dir.drive if hasattr(cache_dir, 'drive') else "C:\\")
        free_gb = free / (1024**3)
    else:
        import shutil
        total, used, free = shutil.disk_usage(cache_dir)
        free_gb = free / (1024**3)

    print(f"✓ Disk Space Available: {free_gb:.1f} GB")
    if free_gb < 60:
        print(f"  ⚠ WARNING: Less than 60GB free. Recommended: 60GB+")
        response = input("  Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(0)

    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"✓ System RAM: {ram_gb:.1f} GB")
        if ram_gb < 64:
            print(f"  ⚠ WARNING: Less than 64GB RAM. Model may not load.")
    except ImportError:
        print("  ℹ Install psutil for RAM check: pip install psutil")

    print("\nCache directory:", cache_dir)
    print("="*60 + "\n")

    return cuda_available

def download_model():
    """Download Mixtral 8x22B with 4-bit quantization"""

    model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"

    print("\n" + "="*60)
    print("DOWNLOADING MIXTRAL 8x22B MODEL")
    print("="*60)
    print(f"Model: {model_name}")
    print("Size: ~45-50 GB")
    print("Time: 30-45 minutes (depending on internet speed)")
    print("="*60 + "\n")

    input("Press Enter to start download...")

    start_time = time.time()

    try:
        # Step 1: Download tokenizer (small, fast)
        print("\n[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded successfully")

        # Step 2: Download model with 4-bit quantization
        print("\n[2/2] Downloading model (this will take a while)...")
        print("You'll see progress bars for each file shard...")

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Download and load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        elapsed_time = time.time() - start_time

        print("\n" + "="*60)
        print("✓ MODEL DOWNLOADED SUCCESSFULLY!")
        print("="*60)
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"Model cached at: {Path.home() / '.cache' / 'huggingface'}")

        # Verify model works
        print("\n[VERIFICATION] Testing model generation...")
        test_prompt = "What is machine troubleshooting?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nTest prompt: {test_prompt}")
        print(f"Model response: {response[:200]}...")
        print("\n✓ Model verification successful!")

        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()

        print("\n" + "="*60)
        print("SETUP COMPLETE")
        print("="*60)
        print("Next steps:")
        print("1. Model is cached and ready to use")
        print("2. Future loads will be much faster (~30 seconds)")
        print("3. You can now integrate this into your Flask app")
        print("="*60 + "\n")

        return True

    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR DURING DOWNLOAD")
        print("="*60)
        print(f"Error: {str(e)}")
        print("\nCommon solutions:")
        print("1. Check internet connection")
        print("2. Ensure you have HuggingFace account (free)")
        print("3. Try again - download resumes from where it stopped")
        print("4. Check disk space (need 60GB+ free)")
        print("="*60 + "\n")
        return False

def check_existing_model():
    """Check if model is already downloaded"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    # Look for Mixtral model folders
    if cache_dir.exists():
        model_folders = list(cache_dir.glob("models--mistralai--Mixtral-8x22B*"))
        if model_folders:
            print("\n" + "="*60)
            print("✓ MODEL ALREADY DOWNLOADED")
            print("="*60)
            print(f"Found at: {model_folders[0]}")
            print("\nThe model is ready to use!")
            print("You can proceed to integrate it into your Flask app.")
            print("="*60 + "\n")
            return True

    return False

def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "MIXTRAL 8x22B DOWNLOADER" + " "*19 + "║")
    print("╚" + "="*58 + "╝")

    # Check if model already exists
    if check_existing_model():
        response = input("Download again? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting...")
            return

    # Check system requirements
    cuda_available = check_requirements()

    if not cuda_available:
        print("⚠ WARNING: No CUDA detected. Model will be VERY slow without GPU.")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting...")
            return

    # Download model
    success = download_model()

    if success:
        print("\n✓ All done! Model is ready for your Flask app.")
    else:
        print("\n❌ Download failed. Please check errors above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("Note: Next run will resume from where it stopped.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
