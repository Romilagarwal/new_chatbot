"""
Industrial Machine Troubleshooting Chatbot
Production Version - Ready for Customer Demo
"""

from flask import Flask, render_template, request, jsonify, Response, send_file
import webbrowser
import os
set HF_HOME="D:\.cache\huggingface\hub"
import threading
import time
import torch
from flask_compress import Compress
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
import json
from datetime import datetime
import csv
import io
import pytz
from pathlib import Path

load_dotenv()

# Configuration
DEVICE = os.getenv('DEVICE', 'auto')
if DEVICE == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True

TITLE = "Machine Troubleshooting AI Assistant"

# Data directories
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FEEDBACK_FILE = DATA_DIR / "feedback.csv"
USER_CORRECTIONS_FILE = DATA_DIR / "user_corrections.csv"
CHAT_LOGS_FILE = DATA_DIR / "chat_logs.json"

# Import model manager
try:
    from advanced_model import get_model, free_model
    MIXTRAL_AVAILABLE = True
    print("✓ Mistral model manager imported successfully")
except ImportError as e:
    MIXTRAL_AVAILABLE = False
    print(f"⚠ Warning: Could not import Mistral model: {e}")

app = Flask(__name__)
Compress(app)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Cache headers for static files
@app.after_request
def add_cache_headers(response):
    if request.path.startswith('/static'):
        response.cache_control.max_age = 604800  # 1 week
        response.cache_control.public = True
    return response

# Global storage
CHAT_HISTORY = {}  # session_id -> list of messages
FEEDBACK_DATA = []
STATISTICS = {
    'total_queries': 0,
    'successful_responses': 0,
    'corrections_submitted': 0,
    'average_rating': 0.0
}

# Import dataset utilities
try:
    from model_utils import reference_df, find_in_dataset
    print("✓ Dataset utilities imported successfully")
    
    if reference_df is not None and len(reference_df) > 0:
        print(f"✓ Dataset loaded: {len(reference_df)} records")
    else:
        print("⚠ Warning: Dataset is empty")
        
except ImportError as e:
    print(f"⚠ Warning: Could not import dataset utilities: {e}")
    reference_df = None
    
    def find_in_dataset(*args, **kwargs):
        """Fallback function when dataset utilities not available"""
        return {'primary': [], 'additional': []}


# Initialize feedback file
def init_feedback_file():
    if not FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'session_id', 'query', 'response', 
                'rating', 'feedback_text', 'was_helpful'
            ])

# Initialize user corrections file
def init_corrections_file():
    if not USER_CORRECTIONS_FILE.exists():
        with open(USER_CORRECTIONS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'machine_type', 'machine_name', 'problem_description',
                'root_cause', 'action_taken', 'submitted_by', 'status'
            ])

init_feedback_file()
init_corrections_file()

# Load existing feedback for statistics
def load_statistics():
    global STATISTICS
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                ratings = []
                for row in reader:
                    STATISTICS['total_queries'] += 1
                    if row.get('was_helpful') == 'true':
                        STATISTICS['successful_responses'] += 1
                    if row.get('rating'):
                        try:
                            ratings.append(int(row['rating']))
                        except:
                            pass
                
                if ratings:
                    STATISTICS['average_rating'] = sum(ratings) / len(ratings)
        except Exception as e:
            print(f"Error loading statistics: {e}")
    
    if USER_CORRECTIONS_FILE.exists():
        try:
            with open(USER_CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
                STATISTICS['corrections_submitted'] = sum(1 for _ in f) - 1  # minus header
        except:
            pass

load_statistics()

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html', title=TITLE)

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': MIXTRAL_AVAILABLE,
        'dataset_loaded': reference_df is not None,
        'timestamp': datetime.now(pytz.UTC).isoformat()
    }
    return jsonify(status)

@app.route('/stats')
def stats():
    """Get system statistics"""
    return jsonify(STATISTICS)

# ============================================================
# CHAT ENDPOINT - Enhanced with RAG
# ============================================================

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').strip()
        session_id = data.get('sessionId', 'default')
        
        if not message:
            return jsonify(error="No message provided"), 400
        
        STATISTICS['total_queries'] += 1
        lower_message = message.lower()
        
        # Quick greetings - hardcoded responses
        quick_greetings = {
            'hi': "Hi! I'm your AI troubleshooting assistant trained on your facility's equipment data. What can I help with?",
            'hello': "Hello! I have access to your machine maintenance history. What equipment issue would you like to discuss?",
            'hey': "Hey! Ask me about any machine in your facility - I know the common issues and solutions.",
            'hii': "Hi there! Ready to help diagnose machine issues. What's the problem?",
            'hiiii': "Hello! What machine problem can I help you solve today?"
        }
        
        if lower_message in quick_greetings:
            response = quick_greetings[lower_message]
            
            if session_id not in CHAT_HISTORY:
                CHAT_HISTORY[session_id] = []
            CHAT_HISTORY[session_id].append({'user': message, 'assistant': response, 'timestamp': datetime.now().isoformat()})
            
            return jsonify(response=response)
        
        # Special handling for "what are you"
        if 'what are you' in lower_message or 'who are you' in lower_message:
            response = "I'm an AI troubleshooting assistant trained on your facility's 2-year maintenance history. I can diagnose machine problems, provide solutions from past cases, suggest preventive measures, and answer technical questions about your equipment."
            
            if session_id not in CHAT_HISTORY:
                CHAT_HISTORY[session_id] = []
            CHAT_HISTORY[session_id].append({'user': message, 'assistant': response, 'timestamp': datetime.now().isoformat()})
            
            return jsonify(response=response)
        
        # Check if machine-related
        machine_keywords = [
            'machine', 'equipment', 'printer', 'assembly', 'welding', 'plasma',
            'vacuum', 'ccd', 'camera', 'cylinder', 'motor', 'sensor', 'alarm',
            'error', 'failure', 'broken', 'issue', 'problem', 'not working',
            'malfunction', 'fault', 'maintenance', 'repair', 'fix', 'troubleshoot',
            'apmt', 'bmu', 'vhb', 'tape', 'wrapping', 'sticking', 'leakage',
            'feeder', 'nozzle', 'drive', 'servo', 'encoder', 'relay'
        ]
        
        is_machine_related = any(keyword in lower_message for keyword in machine_keywords)
        
        # Get relevant context from dataset
        def get_relevant_context(query, top_n=2):
            try:
                if reference_df is None:
                    return []
                
                results = find_in_dataset(
                    machine_type="",
                    machine_name="", 
                    problem=query,
                    threshold=0.45,
                    top_n=top_n
                )
                
                context = []
                for result_tuple in results.get('primary', []):
                    if isinstance(result_tuple, tuple) and len(result_tuple) >= 3:
                        result_text, score, machine_info = result_tuple
                        
                        parts = result_text.split('Action Taken:')
                        root_cause = parts[0].replace('Root Cause:', '').strip() if len(parts) > 0 else ''
                        action = parts[1].strip() if len(parts) > 1 else ''
                        
                        context.append({
                            'MACHINE': machine_info.get('machine_name', 'Unknown'),
                            'Machine Type': machine_info.get('machine_type', ''),
                            'Problem Description': machine_info.get('problem', ''),
                            'Root Cause': root_cause,
                            'Action Taken': action
                        })
                
                return context
            except Exception as e:
                print(f"Error getting context: {e}")
                return []
        
        # Generate response with Mistral
        if not MIXTRAL_AVAILABLE:
            response = "I'm here to help with machine troubleshooting. Could you provide more details about the issue?"
        else:
            try:
                # Get dataset context
                context_examples = []
                if is_machine_related:
                    context_examples = get_relevant_context(message, top_n=2)
                
                # Simple system prompt
                system_msg = "You are a machine troubleshooting expert. Provide direct, concise technical answers based on facility maintenance data. This facility opened in 2023 - never reference cases before that."
                
                if context_examples:
                    system_msg += "\n\nRecent facility cases:\n"
                    for ex in context_examples[:2]:
                        system_msg += f"- {ex['MACHINE']}: {ex['Action Taken'][:100]}\n"
                
                # Simplified prompt
                prompt = f"[INST] {system_msg}\n\nUser: {message}\nAnswer: [/INST]"
                
                # Generate
                model = get_model()
                inputs = model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.15,
                        pad_token_id=model.tokenizer.pad_token_id,
                        eos_token_id=model.tokenizer.eos_token_id
                    )
                
                full_response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean extraction
                if "[/INST]" in full_response:
                    response = full_response.split("[/INST]")[-1].strip()
                else:
                    response = full_response.split("Answer:")[-1].strip() if "Answer:" in full_response else full_response
                
                # Remove artifacts
                response = response.replace("User:", "").replace("Answer:", "").strip()
                
                # Remove first line if it contains system prompt fragments
                lines = response.split('\n')
                if lines and len(lines[0]) < 100 and any(x in lines[0].lower() for x in ['expert', 'assistant', 'facility', 'opened']):
                    response = '\n'.join(lines[1:]).strip()
                
                # Ensure we have content
                if not response or len(response) < 20:
                    response = "Based on our facility's maintenance history, I can help diagnose that issue. Could you provide more specific details about the machine and symptoms?"
                
            except Exception as e:
                print(f"Error in Mistral chat: {e}")
                import traceback
                traceback.print_exc()
                response = "I'm experiencing technical difficulties. Please try the diagnosis form or rephrase your question."
        
        # Store in history
        if session_id not in CHAT_HISTORY:
            CHAT_HISTORY[session_id] = []
        
        CHAT_HISTORY[session_id].append({
            'user': message,
            'assistant': response,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(CHAT_HISTORY[session_id]) > 10:
            CHAT_HISTORY[session_id] = CHAT_HISTORY[session_id][-10:]
        
        return jsonify(response=response)
        
    except Exception as e:
        print(f"Error in chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500


# ============================================================
# DIAGNOSIS ENDPOINT
# ============================================================

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        data = request.json
        machine_type = data.get('machineType', '').strip()
        machine_name = data.get('machineName', '').strip()
        problem = data.get('problem', '').strip()
        
        if not all([machine_type, machine_name, problem]):
            return jsonify(error="Please fill in all fields"), 400
        
        STATISTICS['total_queries'] += 1
        
        # Find similar cases
        try:
            print(f"Running search for: {machine_type} - {machine_name} - {problem}")
            
            results = find_in_dataset(
                machine_type=machine_type,
                machine_name=machine_name,
                problem=problem,
                threshold=0.5,
                top_n=3
            )
            
            context_examples = []
            
            # Parse results - they come as (result_text, score, machine_info)
            for result_tuple in results.get('primary', []):
                if isinstance(result_tuple, tuple) and len(result_tuple) >= 3:
                    result_text, score, machine_info = result_tuple
                    
                    # Parse the result_text
                    parts = result_text.split('Action Taken:')
                    root_cause_text = parts[0].replace('Root Cause:', '').strip() if len(parts) > 0 else ''
                    action_text = parts[1].strip() if len(parts) > 1 else ''
                    
                    context_examples.append({
                        'MACHINE': machine_info.get('machine_name', 'Unknown'),
                        'Machine Type': machine_info.get('machine_type', ''),
                        'Problem Description': machine_info.get('problem', ''),
                        'Root Cause': root_cause_text,
                        'Action Taken': action_text
                    })
            
            print(f"Found {len(context_examples)} context examples")
            
        except Exception as e:
            print(f"Error finding similar cases: {e}")
            import traceback
            traceback.print_exc()
            context_examples = []
        
        # Generate diagnosis
        if not MIXTRAL_AVAILABLE:
            diagnosis = {
                'rootCause': "Model not available. Please check similar cases.",
                'immediateActions': ["Check machine connections", "Restart system", "Contact maintenance"],
                'additionalSolutions': ["Refer to manual"],
                'preventiveMeasures': ["Regular maintenance"]
            }
        else:
            try:
                model = get_model()
                response = model.generate_troubleshooting_response(
                    machine_type=machine_type,
                    machine_name=machine_name,
                    problem=problem,
                    context_examples=context_examples,
                    max_tokens=600
                )
                
                diagnosis = parse_diagnosis_response(response)
                
            except Exception as e:
                print(f"Error generating diagnosis: {e}")
                import traceback
                traceback.print_exc()
                diagnosis = {
                    'rootCause': f"Error: {str(e)}",
                    'immediateActions': ["Contact technical support"],
                    'additionalSolutions': [],
                    'preventiveMeasures': []
                }
        
        # Return response with properly formatted similar cases
        similar_cases = []
        for case in context_examples[:3]:
            if isinstance(case, dict):
                similar_cases.append({
                    'machine': case.get('MACHINE', 'Unknown'),
                    'problem': case.get('Problem Description', ''),
                    'rootCause': case.get('Root Cause', ''),
                    'solution': case.get('Action Taken', '')
                })
        
        return jsonify({
            'diagnosis': diagnosis,
            'similarCases': similar_cases
        })
        
    except Exception as e:
        print(f"Error in diagnose: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500

def parse_diagnosis_response(response):
    """Parse structured diagnosis response"""
    diagnosis = {
        'similarCasesAnalysis': '',
        'rootCause': '',
        'immediateActions': [],
        'additionalSolutions': [],
        'preventiveMeasures': []
    }
    
    try:
        sections = response.split('\n\n')
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            if 'SIMILAR CASES ANALYSIS:' in section:
                diagnosis['similarCasesAnalysis'] = section.replace('SIMILAR CASES ANALYSIS:', '').strip()
            elif 'ROOT CAUSE:' in section:
                diagnosis['rootCause'] = section.replace('ROOT CAUSE:', '').strip()
            elif 'IMMEDIATE ACTIONS:' in section:
                actions = section.replace('IMMEDIATE ACTIONS:', '').strip().split('\n')
                diagnosis['immediateActions'] = [a.strip('0123456789. -') for a in actions if a.strip() and not a.strip() == 'IMMEDIATE']
            elif 'ADDITIONAL SOLUTIONS:' in section:
                solutions = section.replace('ADDITIONAL SOLUTIONS:', '').strip().split('\n')
                diagnosis['additionalSolutions'] = [s.strip('• -*') for s in solutions if s.strip()]
            elif 'PREVENTIVE MEASURES:' in section:
                measures = section.replace('PREVENTIVE MEASURES:', '').strip().split('\n')
                diagnosis['preventiveMeasures'] = [m.strip('• -*') for m in measures if m.strip()]
        
        # Ensure content
        if not diagnosis['rootCause']:
            diagnosis['rootCause'] = "Analysis in progress. Please refer to similar cases for immediate guidance."
        if not diagnosis['immediateActions']:
            diagnosis['immediateActions'] = ["Check HMI for error codes", "Inspect communication cables", "Test individual drive operation"]
            
    except Exception as e:
        print(f"Error parsing diagnosis: {e}")
        diagnosis['rootCause'] = response[:300]
    
    return diagnosis

# ============================================================
# FEEDBACK ENDPOINT - Fixed
# ============================================================

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        session_id = data.get('sessionId', 'default')
        rating = data.get('rating')
        feedback_text = data.get('feedback', '')
        was_helpful = data.get('wasHelpful', False)
        
        # Get last query/response from history
        query = ""
        response = ""
        if session_id in CHAT_HISTORY and CHAT_HISTORY[session_id]:
            last_msg = CHAT_HISTORY[session_id][-1]
            query = last_msg.get('user', '')
            response = last_msg.get('assistant', '')
        
        # Save to CSV
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                session_id,
                query,
                response,
                rating if rating else '',
                feedback_text,
                'true' if was_helpful else 'false'
            ])
        
        # Update statistics
        if was_helpful:
            STATISTICS['successful_responses'] += 1
        if rating:
            # Recalculate average
            load_statistics()
        
        return jsonify(success=True, message="Thank you for your feedback!")
        
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return jsonify(error=str(e)), 500

# ============================================================
# USER CORRECTION SUBMISSION
# ============================================================

@app.route('/submit-correction', methods=['POST'])
def submit_correction():
    try:
        data = request.json
        
        required_fields = ['machineType', 'machineName', 'problemDescription', 'rootCause', 'actionTaken']
        if not all(data.get(field) for field in required_fields):
            return jsonify(error="All fields are required"), 400
        
        # Save to corrections file
        with open(USER_CORRECTIONS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                data['machineType'],
                data['machineName'],
                data['problemDescription'],
                data['rootCause'],
                data['actionTaken'],
                data.get('submittedBy', 'Anonymous'),
                'pending_review'
            ])
        
        STATISTICS['corrections_submitted'] += 1
        
        return jsonify(success=True, message="Correction submitted successfully! This will help improve our AI.")
        
    except Exception as e:
        print(f"Error submitting correction: {e}")
        return jsonify(error=str(e)), 500

# ============================================================
# EXPORT ENDPOINTS
# ============================================================

@app.route('/export/chat-history')
def export_chat_history():
    try:
        # Create CSV from chat history
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Session ID', 'Timestamp', 'User Message', 'Assistant Response'])
        
        for session_id, messages in CHAT_HISTORY.items():
            for msg in messages:
                writer.writerow([
                    session_id,
                    msg.get('timestamp', ''),
                    msg.get('user', ''),
                    msg.get('assistant', '')
                ])
        
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=chat_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
        )
        
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/export/corrections')
def export_corrections():
    try:
        if not USER_CORRECTIONS_FILE.exists():
            return jsonify(error="No corrections to export"), 404
        
        return send_file(
            USER_CORRECTIONS_FILE,
            as_attachment=True,
            download_name=f'user_corrections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify(error=str(e)), 500

# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@app.route('/admin/clear-history', methods=['POST'])
def clear_history():
    global CHAT_HISTORY
    CHAT_HISTORY = {}
    return jsonify(success=True, message="Chat history cleared")

# ============================================================
# STARTUP
# ============================================================

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://172.19.66.141:9352')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 Machine Troubleshooting AI Assistant")
    print("="*60)
    print(f"✓ Model: {'Mistral 7B (Loaded)' if MIXTRAL_AVAILABLE else 'Not Available'}")
    print(f"✓ Dataset: {len(reference_df) if reference_df is not None else 0} records loaded")
    print(f"✓ Statistics: {STATISTICS['total_queries']} total queries")
    print("="*60)
    print("🌐 Server starting at http://172.19.66.141:9352")
    print("="*60 + "\n")
    
    # Open browser
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run app
    app.run(host='172.19.66.141', port=9352, debug=False, threaded=True)

