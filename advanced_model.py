"""
Mistral Nemo Model Integration using Ollama
Replaces transformers-based loading with Ollama API
Much faster and more efficient!
"""

import requests
import json
import time
from typing import Optional, Dict, List

class OllamaTroubleshootingModel:
    """
    Mistral Nemo model manager using Ollama
    Replaces the transformers-based AdvancedTroubleshootingModel
    """
    
    def __init__(self, model_name: str = "mistral-nemo", ollama_host: str = "http://localhost:11434"):
        """
        Initialize Ollama model
        
        Args:
            model_name: Name of the Ollama model (e.g., 'mistral-nemo', 'mistral-nemo:latest')
            ollama_host: Ollama API endpoint (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.api_url = f"{ollama_host}/api/generate"
        self.chat_api_url = f"{ollama_host}/api/chat"
        self.is_loaded = False
        
        # Verify Ollama is running and model is available
        self._verify_setup()
    
    def _verify_setup(self):
        """Verify Ollama is running and model is available"""
        try:
            print(f"Verifying Ollama setup...")
            
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            
            if response.status_code == 200:
                available_models = response.json().get('models', [])
                model_names = [m['name'] for m in available_models]
                
                print(f"  ✓ Ollama is running at {self.ollama_host}")
                print(f"  ✓ Available models: {', '.join(model_names)}")
                
                # Check if our model is available
                model_found = any(self.model_name in name for name in model_names)
                
                if model_found:
                    print(f"  ✓ Model '{self.model_name}' is ready")
                    self.is_loaded = True
                else:
                    print(f"  ⚠ Model '{self.model_name}' not found!")
                    print(f"  Available models: {model_names}")
                    print(f"\n  To download Mistral Nemo, run:")
                    print(f"  ollama pull {self.model_name}")
                    self.is_loaded = False
            else:
                print(f"  ✗ Failed to connect to Ollama (status {response.status_code})")
                self.is_loaded = False
                
        except requests.exceptions.ConnectionError:
            print(f"  ✗ Cannot connect to Ollama at {self.ollama_host}")
            print(f"\n  Is Ollama running? Start it with:")
            print(f"  ollama serve")
            self.is_loaded = False
        except Exception as e:
            print(f"  ✗ Error verifying Ollama: {str(e)}")
            self.is_loaded = False
    
    def generate_troubleshooting_response(
        self,
        machine_type: str,
        machine_name: str,
        problem: str,
        context_examples: Optional[List[Dict]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate structured troubleshooting response using Ollama
        
        Args:
            machine_type: Type of machine
            machine_name: Specific machine name
            problem: Problem description
            context_examples: Similar cases from semantic search (optional)
            max_tokens: Maximum response length
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        
        Returns:
            Structured troubleshooting response
        """
        if not self.is_loaded:
            return "Error: Ollama model not available. Please ensure Ollama is running and model is downloaded."
        
        # Build prompt
        prompt = self._build_troubleshooting_prompt(
            machine_type, machine_name, problem, context_examples
        )
        
        # Generate response using Ollama
        response = self._generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return self._format_troubleshooting_response(response)
    
    def generate_chat_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None,
        max_tokens: int = 200,
        temperature: float = 0.8
    ) -> str:
        """
        Generate conversational response using Ollama
        
        Args:
            message: User message
            conversation_history: Previous conversation (optional)
            max_tokens: Maximum response length
            temperature: Sampling temperature
        
        Returns:
            Chat response
        """
        if not self.is_loaded:
            return "Error: Ollama model not available."
        
        # Build chat prompt
        prompt = self._build_chat_prompt(message, conversation_history)
        
        # Generate response
        response = self._generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
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
                source_badge = "👤 User Contribution" if example.get('source') == 'user_contribution' else "📚 Database"
                verified_badge = "✓ Verified" if example.get('verified') else ""
                
                prompt += f"\nCase {i} {source_badge} {verified_badge}:\n"
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
        temperature: float = 0.7
    ) -> str:
        """
        Core generation function using Ollama API
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        try:
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,  # Get complete response at once
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60  # 60 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                
                # Log generation stats
                total_duration = result.get('total_duration', 0) / 1e9  # Convert to seconds
                eval_count = result.get('eval_count', 0)
                
                if total_duration > 0 and eval_count > 0:
                    tokens_per_sec = eval_count / total_duration
                    print(f"  ✓ Generated {eval_count} tokens in {total_duration:.2f}s ({tokens_per_sec:.1f} tok/s)")
                
                return generated_text
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                print(f"  ✗ {error_msg}")
                return f"Error: {error_msg}"
                
        except requests.exceptions.Timeout:
            error_msg = "Request timeout - generation took too long"
            print(f"  ✗ {error_msg}")
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            print(f"  ✗ {error_msg}")
            return f"Error: {error_msg}"
    
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
        """Cleanup - Ollama handles model management, nothing to free"""
        print("Ollama model connection closed")
        self.is_loaded = False


# Global model instance (singleton pattern)
_model_instance = None

def get_model() -> OllamaTroubleshootingModel:
    """Get or create global model instance"""
    global _model_instance
    if _model_instance is None or not _model_instance.is_loaded:
        _model_instance = OllamaTroubleshootingModel()
    return _model_instance

def free_model():
    """Free global model instance"""
    global _model_instance
    if _model_instance is not None:
        _model_instance.cleanup()
        _model_instance = None


# Test function
def test_ollama_connection():
    """Test Ollama connection and model availability"""
    print("\n" + "="*60)
    print("TESTING OLLAMA CONNECTION")
    print("="*60)
    
    model = OllamaTroubleshootingModel()
    
    if model.is_loaded:
        print("\n✓ Ollama is ready!")
        
        # Test generation
        print("\nTesting generation...")
        test_prompt = "What causes printer paper jams?"
        
        response = model._generate(test_prompt, max_tokens=100, temperature=0.7)
        
        print(f"\nTest prompt: {test_prompt}")
        print(f"Response: {response[:200]}...")
        
        return True
    else:
        print("\n✗ Ollama setup failed. Please check the messages above.")
        return False


if __name__ == "__main__":
    test_ollama_connection()
