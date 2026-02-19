#!/usr/bin/env python3
"""
Test Script for Ollama Integration
Verifies that Mistral Nemo is working correctly with your chatbot
"""

import sys
import requests
import json
from datetime import datetime

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def test_ollama_service():
    """Test 1: Check if Ollama service is running"""
    print("\n" + "="*60)
    print("TEST 1: Ollama Service Status")
    print("="*60)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            print_success("Ollama service is running")
            
            models = response.json().get('models', [])
            print(f"\n  Available models ({len(models)}):")
            for model in models:
                size_gb = model.get('size', 0) / (1024**3)
                print(f"    - {model['name']} ({size_gb:.1f} GB)")
            
            return True, models
        else:
            print_error(f"Ollama returned status {response.status_code}")
            return False, []
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to Ollama at localhost:11434")
        print_info("Start Ollama with: ollama serve")
        return False, []
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False, []

def test_mistral_nemo_available(models):
    """Test 2: Check if Mistral Nemo is downloaded"""
    print("\n" + "="*60)
    print("TEST 2: Mistral Nemo Availability")
    print("="*60)
    
    model_names = [m['name'] for m in models]
    mistral_found = any('mistral-nemo' in name.lower() for name in model_names)
    
    if mistral_found:
        print_success("Mistral Nemo model is available")
        
        for model in models:
            if 'mistral-nemo' in model['name'].lower():
                size_gb = model.get('size', 0) / (1024**3)
                print(f"  Model: {model['name']}")
                print(f"  Size: {size_gb:.1f} GB")
        
        return True
    else:
        print_error("Mistral Nemo model not found")
        print_info("Download with: ollama pull mistral-nemo")
        return False

def test_model_generation():
    """Test 3: Test model generation"""
    print("\n" + "="*60)
    print("TEST 3: Model Generation Test")
    print("="*60)
    
    try:
        payload = {
            "model": "mistral-nemo",
            "prompt": "What causes printer paper jams? Answer in one sentence.",
            "stream": False,
            "options": {
                "num_predict": 50,
                "temperature": 0.7
            }
        }
        
        print("  Sending test prompt...")
        start_time = datetime.now()
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            eval_count = result.get('eval_count', 0)
            
            print_success(f"Generation successful in {duration:.2f}s")
            print(f"\n  Prompt: {payload['prompt']}")
            print(f"  Response: {generated_text[:200]}...")
            print(f"  Tokens generated: {eval_count}")
            
            if duration > 0:
                tokens_per_sec = eval_count / duration
                print(f"  Speed: {tokens_per_sec:.1f} tokens/second")
            
            return True
        else:
            print_error(f"Generation failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Generation error: {str(e)}")
        return False

def test_ollama_model_class():
    """Test 4: Test OllamaTroubleshootingModel class"""
    print("\n" + "="*60)
    print("TEST 4: OllamaTroubleshootingModel Class")
    print("="*60)
    
    try:
        from ollama_model import OllamaTroubleshootingModel
        
        print("  Initializing model...")
        model = OllamaTroubleshootingModel()
        
        if model.is_loaded:
            print_success("Model initialized successfully")
            
            # Test troubleshooting response
            print("\n  Testing troubleshooting response...")
            response = model.generate_troubleshooting_response(
                machine_type="Printer",
                machine_name="HP-3000",
                problem="Paper jam in tray 2",
                context_examples=None,
                max_tokens=200
            )
            
            if response and "ROOT CAUSE:" in response:
                print_success("Troubleshooting response generated")
                print(f"\n  Response preview:")
                print(f"  {response[:300]}...")
            else:
                print_warning("Response generated but format unexpected")
                print(f"  {response[:200]}...")
            
            # Test chat response
            print("\n  Testing chat response...")
            chat_response = model.generate_chat_response(
                message="How do I maintain a printer?",
                max_tokens=100
            )
            
            if chat_response:
                print_success("Chat response generated")
                print(f"  {chat_response[:200]}...")
            
            return True
        else:
            print_error("Model failed to initialize")
            return False
            
    except ImportError:
        print_error("Cannot import ollama_model.py")
        print_info("Make sure ollama_model.py is in the current directory")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_flask_integration():
    """Test 5: Test Flask app integration (if running)"""
    print("\n" + "="*60)
    print("TEST 5: Flask App Integration (Optional)")
    print("="*60)
    
    try:
        # Check if Flask app is running
        response = requests.get("http://172.19.66.141:9352/health", timeout=5)
        
        if response.status_code == 200:
            print_success("Flask app is running")
            health_data = response.json()
            print(f"  Model: {health_data.get('model', 'Unknown')}")
            
            # Test diagnosis endpoint
            print("\n  Testing /diagnose endpoint...")
            diagnose_payload = {
                "machine_type": "Printer",
                "machine_name": "TEST-PRINTER",
                "problem_description": "Paper jam test"
            }
            
            diagnose_response = requests.post(
                "http://172.19.66.141:9352/diagnose",
                json=diagnose_payload,
                timeout=30
            )
            
            if diagnose_response.status_code == 200:
                print_success("Diagnosis endpoint working")
                data = diagnose_response.json()
                if 'ai_response' in data:
                    print(f"  AI response preview: {data['ai_response'][:150]}...")
            else:
                print_error(f"Diagnosis endpoint failed: {diagnose_response.status_code}")
            
            return True
        else:
            print_warning("Flask app not running (this is OK if testing separately)")
            print_info("Start with: python app.py")
            return None
            
    except requests.exceptions.ConnectionError:
        print_warning("Flask app not running (this is OK if testing separately)")
        print_info("Start with: python app.py")
        return None
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def main():
    print("\n" + "="*60)
    print("OLLAMA INTEGRATION TEST SUITE")
    print("="*60)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Ollama Service
    success, models = test_ollama_service()
    results['ollama_service'] = success
    
    if not success:
        print_error("\n❌ Ollama service not running. Cannot continue tests.")
        print_info("Start Ollama with: ollama serve")
        return
    
    # Test 2: Mistral Nemo
    success = test_mistral_nemo_available(models)
    results['mistral_nemo'] = success
    
    if not success:
        print_error("\n❌ Mistral Nemo not available. Download required.")
        print_info("Download with: ollama pull mistral-nemo")
        return
    
    # Test 3: Generation
    success = test_model_generation()
    results['generation'] = success
    
    # Test 4: Python Class
    success = test_ollama_model_class()
    results['python_class'] = success
    
    # Test 5: Flask Integration (optional)
    success = test_flask_integration()
    results['flask_integration'] = success
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, result in results.items():
        if result is not None:
            total_tests += 1
            if result:
                passed_tests += 1
                print_success(f"{test_name}: PASSED")
            else:
                print_error(f"{test_name}: FAILED")
        else:
            print_info(f"{test_name}: SKIPPED (optional)")
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests and total_tests >= 3:
        print("\n" + "="*60)
        print_success("🎉 ALL TESTS PASSED! Your Ollama integration is ready!")
        print("="*60)
        print("\nNext steps:")
        print("1. Update your app.py to use ollama_model")
        print("2. Start your Flask app: python app.py")
        print("3. Test in browser: http://172.19.66.141:9352")
    else:
        print("\n" + "="*60)
        print_warning("⚠ Some tests failed. Please check the errors above.")
        print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
