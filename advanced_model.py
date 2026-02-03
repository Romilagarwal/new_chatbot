import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from typing import Optional, Dict, List
import time
import os

class AdvancedTroubleshootingModel:
    """Mistral 7B Instruct v0.3 - FP16 Mode (No Quantization)"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load Mistral 7B in FP16 (uses ~13.5GB VRAM - perfect for 20GB GPU)"""
        try:
            print(f"Loading {self.model_name}...")
            start_time = time.time()
            
            # Get HF token if available
            HF_TOKEN = os.getenv("HF_TOKEN", None)
            
            # Load tokenizer
            print("  [1/2] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=HF_TOKEN
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("  [2/2] Loading model in FP16...")
            print("  └─ Expected VRAM: ~13.5 GB")
            print("  └─ Mode: Full precision (no quantization)\n")
            
            # Load model in FP16 - NO quantization needed!
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                token=HF_TOKEN
            )
            
            elapsed = time.time() - start_time
            self.is_loaded = True
            print(f"✓ Model loaded successfully in {elapsed:.1f} seconds")
            
            # Show memory info
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                print(f"  └─ GPU Memory: {gpu_mem:.2f} GB")
                print(f"  └─ Device: {torch.cuda.get_device_name(0)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.is_loaded = False
            raise
    
    def generate_troubleshooting_response(
        self,
        machine_type: str,
        machine_name: str,
        problem: str,
        context_examples: Optional[List[Dict]] = None,
        max_tokens: int = 512
    ) -> str:
        """Generate structured troubleshooting response using RAG"""
        if not self.is_loaded:
            return "Error: Model not loaded"
        
        prompt = self._build_troubleshooting_prompt(
            machine_type, machine_name, problem, context_examples
        )
        
        response = self._generate(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )
        
        return self._format_troubleshooting_response(response)
    
    def generate_chat_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None,
        max_tokens: int = 200
    ) -> str:
        """Generate conversational response"""
        if not self.is_loaded:
            return "Error: Model not loaded"
        
        prompt = self._build_chat_prompt(message, conversation_history)
        response = self._generate(
            prompt,
            max_tokens=max_tokens,
            temperature=0.8,
            top_p=0.92
        )
        
        return response.strip()
    
    def _build_troubleshooting_prompt(
        self,
        machine_type: str,
        machine_name: str,
        problem: str,
        context_examples: Optional[List[Dict]] = None
    ) -> str:
        """Build structured prompt with RAG from your CSV dataset"""
        
        prompt = f"""[INST] You are an expert industrial machine troubleshooting assistant. Analyze the following problem and provide a detailed, structured diagnosis.

MACHINE INFORMATION:
- Machine Type: {machine_type}
- Machine Name: {machine_name}
- Problem: {problem}
"""
        
        # Add similar cases from your CSV (RAG component)
        if context_examples and len(context_examples) > 0:
            prompt += "\n\nSIMILAR CASES FROM DATABASE:\n"
            for i, example in enumerate(context_examples[:3], 1):
                prompt += f"\nCase {i}:\n"
                prompt += f"  Machine: {example.get('MACHINE', 'Unknown')}\n"
                prompt += f"  Problem: {example.get('Problem Description', 'Unknown')}\n"
                prompt += f"  Root Cause: {example.get('Root Cause', 'Unknown')}\n"
                prompt += f"  Solution: {example.get('Action Taken', 'Unknown')}\n"
        
        prompt += """

Based on the information above, provide a comprehensive diagnosis in this EXACT format:

ROOT CAUSE:
[Detailed technical analysis of why this problem is occurring]

IMMEDIATE ACTIONS:
1. [First diagnostic/troubleshooting step]
2. [Second diagnostic/troubleshooting step]
3. [Third diagnostic/troubleshooting step]

ADDITIONAL SOLUTIONS:
- [Alternative approach 1]
- [Alternative approach 2]

PREVENTIVE MEASURES:
- [How to prevent this issue in the future]
- [Maintenance recommendations]

Provide your response: [/INST]"""
        
        return prompt
    
    def _build_chat_prompt(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Build prompt for conversational chat"""
        
        prompt = "[INST] You are a helpful Machine Troubleshooting Assistant. Provide clear, professional responses about machine maintenance and troubleshooting.\n\n"
        
        if conversation_history:
            for turn in conversation_history[-6:]:
                prompt += f"Human: {turn.get('user', '')}\n"
                prompt += f"Assistant: {turn.get('assistant', '')}\n\n"
        
        prompt += f"Human: {message}\nAssistant: [/INST]"
        
        return prompt
    
    def _generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Core generation function"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=32768  # Mistral 7B v0.3 supports 32k context
            )
            
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if "[/INST]" in full_response:
                response = full_response.split("[/INST]")[-1].strip()
            else:
                response = full_response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _format_troubleshooting_response(self, response: str) -> str:
        """Format response to match expected output"""
        if "ROOT CAUSE:" not in response:
            parts = response.split('\n\n')
            if len(parts) >= 2:
                response = f"ROOT CAUSE:\n{parts[0]}\n\nIMMEDIATE ACTIONS:\n{parts[1]}"
        
        response = response.replace("Action Taken:", "Possible Solutions:")
        response = response.replace("ACTIONS:", "IMMEDIATE ACTIONS:")
        
        return response.strip()
    
    def cleanup(self):
        """Free memory"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.is_loaded = False
        print("Model unloaded and memory freed")


# Global model instance
_model_instance = None

def get_model() -> AdvancedTroubleshootingModel:
    """Get or create global model instance"""
    global _model_instance
    if _model_instance is None or not _model_instance.is_loaded:
        _model_instance = AdvancedTroubleshootingModel()
    return _model_instance

def free_model():
    """Free global model instance"""
    global _model_instance
    if _model_instance is not None:
        _model_instance.cleanup()
        _model_instance = None

