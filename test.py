"""
Mixtral 8x7B Downloader and Test
Smaller, more reliable model that fits entirely in your GPU
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

def test_mixtral_8x7b():
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    print("\n" + "="*60)
    print("MIXTRAL 8x7B - QUICK TEST")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Expected size: ~26 GB")
    print(f"Expected time: 10-15 minutes download, 60s load")
    print("="*60 + "\n")

    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠ WARNING: CUDA not available!")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            return False
    else:
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    input("Press Enter to start download/loading...")

    try:
        start_time = time.time()

        # Load tokenizer
        print("\n[1/2] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")

        # Load model with 4-bit quantization
        print("\n[2/2] Loading model...")
        print("  (First time: downloading ~26GB)")
        print("  (Subsequent: loading from cache ~60s)\n")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            max_memory={
                0: "16GB",
                "cpu": "60GB"
            }
        )

        load_time = time.time() - start_time
        print(f"\n✓ Model loaded in {load_time:.1f} seconds!")

        # Show device placement
        if hasattr(model, 'hf_device_map'):
            devices = set(str(v) for v in model.hf_device_map.values())
            print(f"  Model location: {', '.join(devices)}")

        # Test generation
        print("\n" + "="*60)
        print("GENERATION TEST")
        print("="*60)

        test_prompt = "Explain the most common cause of printer paper jams in one sentence."
        print(f"\nPrompt: {test_prompt}")
        print("Generating...\n")

        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        gen_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - gen_start

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}\n")
        print(f"Generation time: {gen_time:.2f} seconds")

        # Test troubleshooting format
        print("\n" + "="*60)
        print("TROUBLESHOOTING FORMAT TEST")
        print("="*60)

        troubleshooting_prompt = """You are an expert machine troubleshooting assistant.

MACHINE: Conveyor Belt System
PROBLEM: Belt stops intermittently during operation

Provide diagnosis in this format:

ROOT CAUSE:
[Your analysis]

IMMEDIATE ACTIONS:
1. [Step 1]
2. [Step 2]

Respond:"""

        print("\nGenerating structured response...\n")

        inputs = tokenizer(troubleshooting_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response[len(troubleshooting_prompt):])

        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nMixtral 8x7B is ready for your Flask app!")
        print("Next: Update your app files and run the server.")
        print("="*60 + "\n")

        return True

    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR")
        print("="*60)
        print(f"\n{str(e)}\n")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*12 + "MIXTRAL 8x7B QUICK TEST" + " "*23 + "║")
    print("╚" + "="*58 + "╝")

    success = test_mixtral_8x7b()

    if not success:
        print("\n❌ Test failed. Check errors above.")
