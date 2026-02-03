"""
Advanced Troubleshooting Model - Mixtral 8x22B Integration
Unified model for both chat and technical troubleshooting
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc
from typing import Optional, Dict, List
import time

class AdvancedTroubleshootingModel:
    """Unified model manager for Mixtral 8x22B"""

    def __init__(self, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        """Load model with optimized configuration for RTX 4000 Ada + 128GB RAM"""
        try:
            print(f"Loading {self.model_name}...")
            start_time = time.time()

            # Load tokenizer
            print("  [1/2] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Ensure proper padding
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("  [2/2] Loading model (this takes 2-3 minutes)...")

            # Configure 4-bit quantization with CPU offloading
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )

            # Load model with automatic device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={
                    0: "18GB",      # GPU VRAM (leave 2GB headroom)
                    "cpu": "100GB"  # System RAM
                }
            )

            elapsed = time.time() - start_time
            self.is_loaded = True

            print(f"✓ Model loaded successfully in {elapsed:.1f} seconds")

            # Show memory distribution
            if hasattr(self.model, 'hf_device_map'):
                gpu_layers = sum(1 for v in self.model.hf_device_map.values() if str(v).startswith('cuda'))
                cpu_layers = sum(1 for v in self.model.hf_device_map.values() if v == 'cpu')
                print(f"  └─ GPU layers: {gpu_layers}, CPU layers: {cpu_layers}")

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
        """
        Generate structured troubleshooting response

        Args:
            machine_type: Type of machine
            machine_name: Specific machine name
            problem: Problem description
            context_examples: Similar cases from semantic search (optional)
            max_tokens: Maximum response length

        Returns:
            Structured troubleshooting response
        """
        if not self.is_loaded:
            return "Error: Model not loaded"

        # Build prompt with context
        prompt = self._build_troubleshooting_prompt(
            machine_type, machine_name, problem, context_examples
        )

        # Generate response
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
        """
        Generate conversational response

        Args:
            message: User message
            conversation_history: Previous conversation (optional)
            max_tokens: Maximum response length

        Returns:
            Chat response
        """
        if not self.is_loaded:
            return "Error: Model not loaded"

        # Build chat prompt
        prompt = self._build_chat_prompt(message, conversation_history)

        # Generate response
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
        """Build structured prompt for troubleshooting"""

        prompt = f"""You are an expert industrial machine troubleshooting assistant. Analyze the following problem and provide a detailed, structured diagnosis.

MACHINE INFORMATION:
- Machine Type: {machine_type}
- Machine Name: {machine_name}
- Problem: {problem}
"""

        # Add context examples if available
        if context_examples and len(context_examples) > 0:
            prompt += "\n\nSIMILAR CASES FROM DATABASE:\n"
            for i, example in enumerate(context_examples[:3], 1):
                prompt += f"\nCase {i}:\n"
                prompt += f"  Machine: {example.get('machine_name', 'Unknown')}\n"
                prompt += f"  Problem: {example.get('problem', 'Unknown')}\n"
                prompt += f"  Root Cause: {example.get('root_cause', 'Unknown')}\n"
                prompt += f"  Solution: {example.get('action', 'Unknown')}\n"

        prompt += """

Based on the information above, provide a comprehensive diagnosis in this EXACT format:

ROOT CAUSE:
[Provide detailed technical analysis of why this problem is occurring. Consider mechanical, electrical, software, and environmental factors.]

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

Provide your response:"""

        return prompt

    def _build_chat_prompt(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Build prompt for conversational chat"""

        prompt = "You are a helpful Machine Troubleshooting Assistant. You provide clear, concise, and professional responses about machine maintenance, troubleshooting, and technical support.\n\n"

        # Add conversation history
        if conversation_history:
            for turn in conversation_history[-6:]:  # Last 3 exchanges
                prompt += f"Human: {turn.get('user', '')}\n"
                prompt += f"Assistant: {turn.get('assistant', '')}\n\n"

        prompt += f"Human: {message}\n"
        prompt += "Assistant:"

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
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096  # Use reasonable context window
            )

            # Move to appropriate device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
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

            # Decode
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part (remove prompt)
            if "Assistant:" in full_response:
                response = full_response.split("Assistant:")[-1].strip()
            elif "Provide your response:" in full_response:
                response = full_response.split("Provide your response:")[-1].strip()
            else:
                response = full_response[len(prompt):].strip()

            return response

        except Exception as e:
            print(f"Generation error: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _format_troubleshooting_response(self, response: str) -> str:
        """Format response to match expected output format"""

        # Ensure proper formatting
        if "ROOT CAUSE:" not in response:
            # Try to structure unstructured response
            parts = response.split('\n\n')
            if len(parts) >= 2:
                response = f"ROOT CAUSE:\n{parts[0]}\n\nIMMEDIATE ACTIONS:\n{parts[1]}"

        # Replace "Action Taken" with "Possible Solutions" for consistency
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

# Global model instance (singleton pattern)
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
