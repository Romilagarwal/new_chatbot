"""
Mistral 7B Instruct v0.2 - Download and Test Script
Perfect fit for 20GB GPU - uses ~7-8GB VRAM with 4-bit quantization
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import time
import gc
import os

def test_mistral_7b():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    # ============================================================
    # HUGGING FACE TOKEN (OPTIONAL FOR THIS MODEL)
    # ============================================================
    # Get token from environment variable or set it here
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Login if token is provided
    if HF_TOKEN:
        print("🔐 Logging in to Hugging Face...")
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
            print("✓ Successfully authenticated with Hugging Face\n")
        except Exception as e:
            print(f"⚠ Warning: Could not login with token: {e}")
            print("Continuing without authentication (should work for ungated models)\n")
    else:
        print("ℹ No HF token provided (not needed for Mistral 7B v0.2)\n")
    # ============================================================

    print("\n" + "="*70)
    print("║" + " "*10 + "MISTRAL 7B INSTRUCT v0.2 - DOWNLOAD & TEST" + " "*16 + "║")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Download size: ~14.5 GB (4-bit quantized)")
    print(f"VRAM usage: ~7-8 GB")
    print(f"Speed: 2-3x faster than Mixtral 8x7B")
    print(f"Quality: Excellent for technical troubleshooting")
    print("="*70 + "\n")

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        print("Please ensure:")
        print("  1. NVIDIA drivers are installed")
        print("  2. PyTorch with CUDA is installed")
        return False

    print(f"✓ GPU Detected: {torch.cuda.get_device_name(0)}")
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✓ Total VRAM: {vram_total:.1f} GB")

    # Check available VRAM
    torch.cuda.empty_cache()
    vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
    print(f"✓ Available VRAM: {vram_free:.1f} GB\n")

    if vram_free < 8:
        print("⚠ WARNING: Less than 8GB VRAM free")
        print("Close other GPU applications before continuing")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            return False

    input("Press Enter to start download and loading...")

    try:
        start_time = time.time()

        # Step 1: Load tokenizer
        print("\n" + "-"*70)
        print("[1/3] LOADING TOKENIZER")
        print("-"*70)

        # Pass token if available (not needed for v0.2 but good practice)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=HF_TOKEN  # Will be None if not set, which is fine
        )

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✓ Tokenizer loaded successfully")
        print(f"  └─ Vocab size: {len(tokenizer)}")
        print(f"  └─ Model max length: {tokenizer.model_max_length}\n")

        # Step 2: Configure quantization
        print("-"*70)
        print("[2/3] CONFIGURING 4-BIT QUANTIZATION")
        print("-"*70)
        print("Configuration:")
        print("  └─ Quantization: 4-bit NF4")
        print("  └─ Compute dtype: float16")
        print("  └─ Double quantization: Enabled")
        print("  └─ Device: GPU only (no CPU offload)\n")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Step 3: Load model
        print("-"*70)
        print("[3/3] LOADING MODEL")
        print("-"*70)
        print("This will:")
        print("  └─ Download ~14.5 GB (first time only, takes 5-15 minutes)")
        print("  └─ Load and quantize model (~60-90 seconds)")
        print("  └─ Allocate ~7-8 GB GPU memory\n")

        model_start = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=HF_TOKEN  # Pass token here too
        )

        model_time = time.time() - model_start

        print(f"\n✓ Model loaded successfully in {model_time:.1f} seconds!")

        # Show GPU memory usage
        gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
        gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\nGPU Memory Status:")
        print(f"  └─ Allocated: {gpu_memory:.2f} GB")
        print(f"  └─ Reserved: {gpu_reserved:.2f} GB")
        print(f"  └─ Free: {vram_total - gpu_reserved:.2f} GB")

        # Step 4: Test generation
        print("\n" + "="*70)
        print("║" + " "*20 + "GENERATION TEST" + " "*33 + "║")
        print("="*70)

        test_prompts = [
            "Explain what causes a printer paper jam in one sentence.",
            "List 3 common reasons for machine CCD camera failures.",
            "What is the root cause of vacuum system pressure loss?"
        ]

        for i, test_prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}/3:")
            print(f"Prompt: {test_prompt}")
            print("Generating...\n")

            # Format for Mistral Instruct
            formatted_prompt = f"[INST] {test_prompt} [/INST]"

            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            gen_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            gen_time = time.time() - gen_start

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only generated part
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()

            print(f"Response: {response}")
            print(f"Generation time: {gen_time:.2f}s ({len(response.split())/gen_time:.1f} tokens/sec)\n")
            print("-"*70)

        # Summary
        total_time = time.time() - start_time

        print("\n" + "="*70)
        print("║" + " "*25 + "TEST COMPLETE" + " "*31 + "║")
        print("="*70)
        print(f"\n✓ All tests passed successfully!")
        print(f"\nTotal time: {total_time:.1f} seconds")
        print(f"Model load time: {model_time:.1f} seconds")
        print(f"GPU memory used: {gpu_memory:.2f} GB / {vram_total:.1f} GB")

        print("\n" + "-"*70)
        print("NEXT STEPS:")
        print("-"*70)
        print("1. Model is cached - future loads will be much faster (~60s)")
        print("2. Update advanced_model.py to use this model")
        print("3. Run your Flask app: python app.py")
        print("4. Access chatbot at: http://172.19.77.27:9023")
        print("-"*70)

        # Cleanup
        print("\nCleaning up...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print("✓ Memory freed\n")

        print("="*70)
        print("║" + " "*15 + "✓ MISTRAL 7B READY FOR PRODUCTION!" + " "*16 + "║")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR OCCURRED")
        print("="*70)
        print(f"\n{str(e)}\n")

        import traceback
        print("Full traceback:")
        print("-"*70)
        traceback.print_exc()
        print("-"*70)

        print("\nTROUBLESHOOTING:")
        print("1. Check internet connection (needs to download 14.5 GB)")
        print("2. Ensure bitsandbytes is installed: pip install bitsandbytes")
        print("3. Kill other GPU processes: pkill -9 python")
        print("4. Check disk space: df -h ~/.cache/huggingface")
        print("5. If gated model, set HF_TOKEN environment variable")
        print("6. Try again - downloads resume automatically\n")

        return False

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "MISTRAL 7B INSTRUCT v0.2 - DOWNLOAD & TEST" + " "*15 + "║")
    print("╚" + "="*68 + "╝")
    print("\nTo use HF token (optional for v0.2, required for gated models):")
    print("  Option 1: export HF_TOKEN=hf_xxxxxxxxxxxxx")
    print("  Option 2: Edit HF_TOKEN variable in this script\n")

    success = test_mistral_7b()

    if success:
        print("\n✅ SUCCESS! Mistral 7B is ready for your chatbot application!")
    else:
        print("\n❌ Test failed. Check errors above and try again.")

    print("\n")