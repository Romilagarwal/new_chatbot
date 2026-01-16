"""
Industrial Machine Troubleshooting Chatbot
Production Version - Ready for Customer Demo
"""

import re
from flask import Flask, render_template, request, jsonify, Response, send_file
import webbrowser
import os
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
    from model_utils import reference_df, find_in_dataset, load_data
    print("✓ Dataset utilities imported successfully")

    # Load dataset
    dataset_path = "mix_dataset_final.csv"
    if os.path.exists(dataset_path):
        load_data()
        print(f"✓ Dataset loaded: {len(reference_df)} records")
    else:
        print(f"⚠ Warning: Dataset not found at {dataset_path}")
        reference_df = None

except ImportError as e:
    print(f"⚠ Warning: Could not import dataset utilities: {e}")
    reference_df = None
    def find_in_dataset(*args, **kwargs):
        return {'primary': [], 'fallback': []}

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

        # Get relevant context from dataset
        def get_relevant_context(query, top_n=2):
            """
            Get relevant context from the dataset for a given query.

            Returns:
                List of dicts with machine info, problem, root cause, and action taken
            """
            try:
                if reference_df is None:
                    return []

                results = find_in_dataset(
                    machine_type="",
                    machine_name="",
                    problem_desc=query,
                    threshold=0.45,
                    top_n=top_n
                )

                context = []

                for result_tuple in results.get('primary', []):
                    text = result_tuple[0]
                    score = result_tuple[1]
                    machine_info = result_tuple[2]

                    root_cause = ""
                    action_taken = ""

                    if "Action Taken:" in text:
                        parts = text.split("Action Taken:")
                        root_cause_part = parts[0].replace("Root Cause:", "").strip()
                        action_taken_part = parts[1].strip() if len(parts) > 1 else ""
                        root_cause = root_cause_part.replace(". ", "", 1) if root_cause_part.startswith(". ") else root_cause_part
                        action_taken = action_taken_part
                    else:
                        root_cause = text.replace("Root Cause:", "").strip()

                    context.append({
                        'MACHINE': machine_info.get('machine_name', 'Unknown'),
                        'Machine Type': machine_info.get('machine_type', 'N/A'),
                        'Problem Description': machine_info.get('problem', ''),
                        'Root Cause': root_cause,
                        'Action Taken': action_taken
                    })

                return context

            except Exception as e:
                print(f"Error getting context: {e}")
                import traceback
                traceback.print_exc()
                return []

        # Quick greetings
        quick_greetings = {
            'hi': "Hi! I'm your AI troubleshooting assistant trained on your facility's equipment data. What can I help with?",
            'hello': "Hello! I have access to your machine maintenance history. What equipment issue would you like to discuss?",
            'hey': "Hey! Ask me about any machine in your facility - I know the common issues and solutions.",
            'hii': "Hi there! Ready to help diagnose machine issues. What's the problem?",
            'hiiii': "Hello! What machine problem can I help you solve today?"
        }

        if lower_message in quick_greetings:
            response = quick_greetings[lower_message]

            # Store in history
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

        # Generate response with Mistral
        if not MIXTRAL_AVAILABLE:
            response = "I'm here to help with machine troubleshooting. Could you provide more details about the issue?"
        else:
            try:
                # Get conversation history
                conversation_history = []
                if session_id in CHAT_HISTORY:
                    recent = [msg for msg in CHAT_HISTORY[session_id][-2:]]  # Last 3 exchanges
                    conversation_history = recent

                # Get dataset context
                context_examples = []
                if is_machine_related:
                    context_examples = get_relevant_context(message, top_n=2)

                # Build system prompt
                system_context = """You are an expert Machine Troubleshooting Assistant for an industrial manufacturing facility.

CORE CAPABILITIES:
- You have access to the facility's complete machine maintenance history
- You know common problems, root causes, and proven solutions for all equipment
- You provide technical, specific answers based on actual past cases
- You speak the language of maintenance technicians

RESPONSE STYLE:
- Be concise but thorough (2-3 paragraphs max for chat)
- Reference specific machines and past cases when relevant
- Provide actionable technical guidance
- If you don't have exact information, acknowledge it and provide general guidance"""

                if context_examples:
                    system_context += "\n\nRELEVANT CASES FROM YOUR DATABASE:\n"
                    for i, ex in enumerate(context_examples, 1):
                        system_context += f"\n{i}. Machine: {ex['MACHINE']} ({ex.get('Machine Type', 'N/A')})"
                        system_context += f"\n   Problem: {ex['Problem Description']}"
                        system_context += f"\n   Root Cause: {ex['Root Cause']}"
                        system_context += f"\n   Solution: {ex['Action Taken']}\n"

                # Build full prompt
                full_prompt = f"[INST] {system_context}\n\n"

                if conversation_history:
                    for turn in conversation_history[-4:]:  # Last 2 exchanges
                        full_prompt += f"Human: {turn.get('user', '')}\n"
                        full_prompt += f"Assistant: {turn.get('assistant', '')}\n\n"

                full_prompt += f"Human: {message}\nAssistant: [/INST]"

                # Generate
                model = get_model()
                inputs = model.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=8192)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.model.generate(
                        **inputs,
                        max_new_tokens=300,
                        do_sample=True,
                        temperature=0.75,
                        top_p=0.9,
                        repetition_penalty=1.15,
                        pad_token_id=model.tokenizer.pad_token_id,
                        eos_token_id=model.tokenizer.eos_token_id
                    )

                # Decode and clean response
                full_response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract only the assistant's response after [/INST]
                if "[/INST]" in full_response:
                    response = full_response.split("[/INST]")[-1].strip()
                else:
                    response = full_response

                # Clean up conversation markers
                response = response.replace("Human:", "").replace("Assistant:", "").strip()

                # CRITICAL FIX: Remove echoed conversation history
                if conversation_history:
                    for turn in conversation_history[-4:]:
                        prev_user = turn.get('user', '').strip()
                        prev_assistant = turn.get('assistant', '').strip()

                        if prev_user:
                            response = response.replace(prev_user, '')
                        if prev_assistant:
                            response = response.replace(prev_assistant, '')

                # Remove system prompt text that might leak into response
                unwanted_phrases = [
                    "CORE CAPABILITIES:",
                    "RESPONSE STYLE:",
                    "You are an expert Machine Troubleshooting Assistant",
                    "You have access to the facility's complete machine maintenance history",
                    "You know common problems, root causes, and proven solutions",
                    "You provide technical, specific answers based on actual past cases",
                    "You speak the language of maintenance technicians",
                    "Be concise but thorough",
                    "Reference specific machines",
                    "Provide actionable technical guidance",
                    "If you don't have exact information",
                    "[INST]",
                    "RELEVANT CASES FROM YOUR DATABASE:"
                ]

                # Remove any lines that contain unwanted phrases
                lines = response.split('\n')
                cleaned_lines = []
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    if not any(phrase.lower() in line.lower() for phrase in unwanted_phrases):
                        cleaned_lines.append(line_stripped)

                response = '\n'.join(cleaned_lines).strip()

                # Remove leading dashes or bullet points if they're artifacts
                response = re.sub(r'^[\-\*\•]\s+', '', response, flags=re.MULTILINE)

                # Remove multiple newlines
                response = re.sub(r'\n{3,}', '\n\n', response)

                # Final cleanup
                response = response.strip()

                # If response is empty or too short after cleaning, provide a fallback
                if len(response) < 10:
                    response = "I can help you troubleshoot that issue. Could you provide more details about the problem?"

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

        # Keep last 10 messages
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
            results = find_in_dataset(
                machine_type=machine_type,
                machine_name=machine_name,
                problem=problem,
                threshold=0.5,
                top_n=3
            )

            context_examples = []
            for result_tuple in results.get('primary', []):
                text = result_tuple[0]          # "Root Cause: ... Action Taken: ..."
                score = result_tuple[1]         # similarity score
                machine_info = result_tuple[2]  # dict with machine details

                # Parse the text to extract Root Cause and Action Taken
                root_cause = ""
                action_taken = ""

                if "Action Taken:" in text:
                    parts = text.split("Action Taken:")
                    root_cause = parts[0].replace("Root Cause:", "").strip()
                    action_taken = parts[1].strip() if len(parts) > 1 else ""
                else:
                    root_cause = text.replace("Root Cause:", "").strip()

                # Build proper dict format expected by advanced_model.py and similarCases
                context_examples.append({
                    'MACHINE': machine_info.get('machine_name', 'Unknown'),
                    'Machine Type': machine_info.get('machine_type', 'N/A'),
                    'Problem Description': machine_info.get('problem', ''),
                    'Root Cause': root_cause,
                    'Action Taken': action_taken
                })

        except Exception as e:
            print(f"Error finding similar cases: {e}")
            context_examples = []

        # Generate diagnosis with Mistral
        if not MIXTRAL_AVAILABLE:
            diagnosis = {
                'rootCause': "Model not available. Please check similar cases from the database.",
                'immediateActions': ["Check machine connections", "Restart the system", "Contact maintenance"],
                'additionalSolutions': ["Refer to manual", "Check error logs"],
                'preventiveMeasures': ["Regular maintenance", "Operator training"]
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

                # Parse response
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

        return jsonify({
            'diagnosis': diagnosis,
            'similarCases': [
                {
                    'machine': case.get('MACHINE', 'Unknown'),
                    'problem': case.get('Problem Description', ''),
                    'rootCause': case.get('Root Cause', ''),
                    'solution': case.get('Action Taken', '')
                }
                for case in context_examples[:3]
            ]
        })

    except Exception as e:
        print(f"Error in diagnose: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500

def parse_diagnosis_response(response):
    """Parse structured diagnosis response"""
    diagnosis = {
        'rootCause': '',
        'immediateActions': [],
        'additionalSolutions': [],
        'preventiveMeasures': []
    }

    try:
        sections = response.split('\n\n')
        current_section = None

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if 'ROOT CAUSE:' in section:
                current_section = 'rootCause'
                diagnosis['rootCause'] = section.replace('ROOT CAUSE:', '').strip()
            elif 'IMMEDIATE ACTIONS:' in section:
                current_section = 'immediateActions'
                actions = section.replace('IMMEDIATE ACTIONS:', '').strip().split('\n')
                diagnosis['immediateActions'] = [a.strip('0123456789. -') for a in actions if a.strip()]
            elif 'ADDITIONAL SOLUTIONS:' in section:
                current_section = 'additionalSolutions'
                solutions = section.replace('ADDITIONAL SOLUTIONS:', '').strip().split('\n')
                diagnosis['additionalSolutions'] = [s.strip('- •*') for s in solutions if s.strip()]
            elif 'PREVENTIVE MEASURES:' in section:
                current_section = 'preventiveMeasures'
                measures = section.replace('PREVENTIVE MEASURES:', '').strip().split('\n')
                diagnosis['preventiveMeasures'] = [m.strip('- •*') for m in measures if m.strip()]

        # Ensure at least some content
        if not diagnosis['rootCause']:
            diagnosis['rootCause'] = response[:200] + "..."
        if not diagnosis['immediateActions']:
            diagnosis['immediateActions'] = ["Check machine status", "Review error logs", "Contact maintenance"]

    except Exception as e:
        print(f"Error parsing diagnosis: {e}")
        diagnosis['rootCause'] = response

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

@app.route('/admin/update-stats', methods=['POST'])
def update_statistics():
    """
    Admin endpoint to manually update statistics
    POST body example:
    {
        "total_queries": 100,
        "successful_responses": 85,
        "average_rating": 4.5,
        "corrections_submitted": 20
    }
    """
    try:
        data = request.json

        # Update STATISTICS dictionary
        if 'total_queries' in data:
            STATISTICS['total_queries'] = int(data['total_queries'])

        if 'successful_responses' in data:
            STATISTICS['successful_responses'] = int(data['successful_responses'])

        if 'average_rating' in data:
            STATISTICS['average_rating'] = float(data['average_rating'])

        if 'corrections_submitted' in data:
            STATISTICS['corrections_submitted'] = int(data['corrections_submitted'])

        return jsonify({
            'success': True,
            'message': 'Statistics updated successfully',
            'current_stats': STATISTICS
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
# ============================================================
# STARTUP
# ============================================================

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:9025')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 Machine Troubleshooting AI Assistant")
    print("="*60)
    print(f"✓ Model: {'Mistral 7B (Loaded)' if MIXTRAL_AVAILABLE else 'Not Available'}")
    print(f"✓ Dataset: {len(reference_df) if reference_df is not None else 0} records loaded")
    print(f"✓ Statistics: {STATISTICS['total_queries']} total queries")
    print("="*60)
    print("🌐 Server starting at http://127.0.0.1:9025")
    print("="*60 + "\n")

    # Open browser
    threading.Thread(target=open_browser, daemon=True).start()

    # Run app
    app.run(host='0.0.0.0', port=9025, debug=False, threaded=True)

