"""Mixtral 8x7B with 8-bit + CPU offload (WORKS on 20GB GPU)"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

def test_mixtral_8x7b():
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    print("\n" + "="*60)
    print("MIXTRAL 8x7B - 8-BIT + CPU OFFLOAD")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    # Clear GPU memory first
    torch.cuda.empty_cache()
    
    input("Press Enter to start...")
    
    try:
        start_time = time.time()
        
        print("\n[1/2] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")
        
        print("\n[2/2] Loading model with 8-bit quantization + CPU offload...")
        
        # 8-bit quantization WITH CPU offload support
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True  # This works with 8-bit!
        )
        
        # Set memory limits for GPU + CPU
        max_memory = {
            0: "17GB",  # Leave 2GB headroom
            "cpu": "40GB"  # Allow CPU RAM offload
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True
        )
        
        load_time = time.time() - start_time
        print(f"\n✓ Model loaded in {load_time:.1f} seconds!")
        
        gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  └─ GPU Memory: {gpu_mem:.2f} GB (8-bit + CPU offload)")
        
        # Test generation
        print("\n" + "="*60)
        print("GENERATION TEST")
        print("="*60)
        
        test_prompt = "Explain printer paper jams in one sentence."
        print(f"\nPrompt: {test_prompt}")
        print("Generating...\n")
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}\n")
        
        print("="*60)
        print("✓ SUCCESS! Model works with 8-bit + CPU offload")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mixtral_8x7b()


