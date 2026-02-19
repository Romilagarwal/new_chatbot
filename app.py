"""
Enhanced Machine Troubleshooting Chatbot
Now with Ollama (Mistral Nemo) + Dynamic Learning System
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_compress import Compress
import json
import os
from datetime import datetime
import pytz
from pathlib import Path

# Import enhanced model utilities with Ollama and dynamic learning
from model_utils import (
    find_in_dataset,
    generate_ai_response,
    add_user_solution,
    update_solution_effectiveness,
    get_kb_statistics
)

app = Flask(__name__)
Compress(app)

# Ensure data directory exists
Path('data').mkdir(exist_ok=True)

# ============================================================================
# MAIN ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main application interface"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        stats = get_kb_statistics()
        return jsonify({
            'status': 'healthy',
            'model': 'Mistral Nemo (Ollama)',
            'knowledge_base': _sanitize_for_json(stats),
            'timestamp': datetime.now(pytz.UTC).isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get usage statistics"""
    try:
        # Read feedback data
        feedback_file = 'data/feedback.csv'
        chat_log_file = 'data/chat_logs.json'

        stats_data = {
            'total_queries': 0,
            'total_feedback': 0,
            'average_rating': 0,
            'knowledge_base': get_kb_statistics()
        }

        # Count feedback entries
        if os.path.exists(feedback_file):
            import pandas as pd
            df = pd.read_csv(feedback_file)
            stats_data['total_feedback'] = len(df)
            if 'rating' in df.columns:
                stats_data['average_rating'] = float(df['rating'].mean())

        # Count chat logs
        if os.path.exists(chat_log_file):
            with open(chat_log_file, 'r') as f:
                logs = json.load(f)
                stats_data['total_queries'] = len(logs)

        return jsonify(_sanitize_for_json(stats_data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# DIAGNOSIS ENDPOINTS
# ============================================================================

@app.route('/diagnose', methods=['POST'])
def diagnose():
    """
    Enhanced diagnosis endpoint with dynamic knowledge base
    """
    try:
        data = request.json
        machine_type = data.get('machine_type', '').strip()
        machine_name = data.get('machine_name', '').strip()
        problem = data.get('problem_description', '').strip()

        if not problem or not machine_name:
            return jsonify({
                'error': 'Machine name and problem description are required'
            }), 400

        print(f"\n{'='*60}")
        print(f"DIAGNOSIS REQUEST")
        print(f"{'='*60}")
        print(f"Machine Type: {machine_type}")
        print(f"Machine Name: {machine_name}")
        print(f"Problem: {problem}")

        # Search dynamic knowledge base (includes user contributions)
        search_results = find_in_dataset(
            machine_type=machine_type,
            machine_name=machine_name,
            problem=problem,
            threshold=0.60,
            top_n=3
        )

        primary_results = search_results.get('primary', [])
        additional_results = search_results.get('additional', [])

        # Generate AI response with context using Ollama
        ai_response = generate_ai_response(
            machine_type=machine_type,
            machine_name=machine_name,
            problem_description=problem,
            context_results=primary_results
        )

        # Format results with source information
        formatted_results = []
        for text, score, machine_info in primary_results:
            # Sanitize machine_info to convert numpy types to Python types
            sanitized_machine_info = _sanitize_for_json(machine_info)

            formatted_results.append({
                'text': text,
                'confidence': int(score * 100),
                'machine_info': sanitized_machine_info,
                'source': sanitized_machine_info.get('source', 'database'),
                'verified': bool(sanitized_machine_info.get('verified', True)),
                'entry_id': int(sanitized_machine_info.get('entry_id', 0)) if sanitized_machine_info.get('entry_id') is not None else None
            })

        print(f"\n✓ Found {len(formatted_results)} solutions")
        print(f"  - Original DB: {sum(1 for r in formatted_results if r['source'] == 'original')}")
        print(f"  - User contributed: {sum(1 for r in formatted_results if r['source'] == 'user_contribution')}")
        print(f"{'='*60}\n")

        return jsonify({
            'ai_response': ai_response,
            'database_results': formatted_results,
            'additional_results': _sanitize_for_json(additional_results),
            'metadata': {
                'total_results': len(formatted_results),
                'user_contributed': sum(1 for r in formatted_results if r['source'] == 'user_contribution'),
                'original': sum(1 for r in formatted_results if r['source'] == 'original')
            }
        })

    except Exception as e:
        print(f"Error in diagnose: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Diagnosis failed: {str(e)}'
        }), 500


# ============================================================================
# CHAT ENDPOINT
# ============================================================================

@app.route('/chat', methods=['POST'])
def chat():
    """
    Enhanced chat endpoint with Ollama
    """
    try:
        data = request.json
        message = data.get('message', '').strip()
        conversation_history = data.get('history', [])

        if not message:
            return jsonify({
                'error': 'Message cannot be empty'
            }), 400

        # Check if this is a troubleshooting query
        troubleshooting_keywords = ['alarm', 'error', 'not working', 'problem',
                                   'issue', 'fault', 'broken', 'malfunction', 'fix']

        is_troubleshooting = any(keyword in message.lower() for keyword in troubleshooting_keywords)

        if is_troubleshooting:
            # Try to extract machine info
            words = message.split()
            machine_name = words[0] if words else ''

            # Quick search for relevant context
            search_results = find_in_dataset(
                machine_type='',
                machine_name=machine_name,
                problem=message,
                threshold=0.50,
                top_n=2
            )

            context = search_results.get('primary', [])

            # Generate contextualized response using Ollama
            from ollama_model import get_model
            model = get_model()

            response = model.generate_chat_response(
                message=message,
                conversation_history=conversation_history,
                max_tokens=200
            )

            # Add context hint if relevant solutions found
            if context and len(context) > 0:
                source_info = context[0][2].get('source', 'database')
                if source_info == 'user_contribution':
                    response += "\n\n💡 (This answer includes insights from user-contributed solutions)"
        else:
            # Regular conversational response
            from ollama_model import get_model
            model = get_model()

            response = model.generate_chat_response(
                message=message,
                conversation_history=conversation_history,
                max_tokens=200
            )

        # Log chat
        _log_chat(message, response)

        return jsonify({
            'response': response,
            'timestamp': datetime.now(pytz.UTC).isoformat()
        })

    except Exception as e:
        print(f"Error in chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Chat failed: {str(e)}'
        }), 500


# ============================================================================
# DYNAMIC LEARNING ENDPOINTS (NEW!)
# ============================================================================

@app.route('/submit-solution', methods=['POST'])
def submit_solution():
    """
    NEW: Submit user solution - integrates into knowledge base immediately
    """
    try:
        data = request.json

        # Extract data
        machine_type = data.get('machine_type', '').strip()
        machine_name = data.get('machine_name', '').strip()
        problem = data.get('problem', '').strip()
        root_cause = data.get('root_cause', '').strip()
        solution = data.get('solution', '').strip()
        user_id = data.get('user_id', 'anonymous')

        # Validation
        if not all([machine_type, machine_name, problem, root_cause, solution]):
            return jsonify({
                'success': False,
                'message': 'All fields are required'
            }), 400

        # Add to knowledge base (immediate integration)
        result = add_user_solution(
            machine_type=machine_type,
            machine_name=machine_name,
            problem=problem,
            root_cause=root_cause,
            solution=solution,
            verified=False,  # Requires admin verification
            effectiveness_score=0.8,  # Default score for new solutions
            user_id=user_id
        )

        if result['success']:
            # Log the submission
            _log_user_contribution(data, result)

            return jsonify({
                'success': True,
                'message': 'Solution submitted and integrated successfully! It will be used immediately in future searches.',
                'entry_id': result.get('entry_id')
            })
        else:
            return jsonify({
                'success': False,
                'message': result.get('message', 'Failed to add solution')
            }), 400

    except Exception as e:
        print(f"Error in submit_solution: {e}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@app.route('/rate-solution', methods=['POST'])
def rate_solution():
    """
    NEW: Rate solution effectiveness - updates knowledge base dynamically
    """
    try:
        data = request.json

        entry_id = data.get('entry_id')
        rating = data.get('rating')  # 1-5 star rating
        feedback = data.get('feedback', '')

        if entry_id is None or rating is None:
            return jsonify({
                'success': False,
                'message': 'Entry ID and rating are required'
            }), 400

        # Convert 1-5 rating to 0-1 effectiveness score
        effectiveness_score = rating / 5.0

        # Update knowledge base
        result = update_solution_effectiveness(
            entry_id=entry_id,
            effectiveness_score=effectiveness_score,
            user_feedback=feedback
        )

        if result['success']:
            return jsonify({
                'success': True,
                'message': 'Thank you for your feedback! This helps improve future recommendations.',
                'new_score': result.get('new_score')
            })
        else:
            return jsonify({
                'success': False,
                'message': result.get('message', 'Failed to update rating')
            }), 400

    except Exception as e:
        print(f"Error in rate_solution: {e}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@app.route('/kb-stats', methods=['GET'])
def kb_stats():
    """
    NEW: Get knowledge base statistics
    """
    try:
        stats = get_kb_statistics()
        return jsonify({
            'success': True,
            'statistics': _sanitize_for_json(stats)
        })
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# ============================================================================
# FEEDBACK ENDPOINTS
# ============================================================================

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
    try:
        data = request.json

        feedback_entry = {
            'rating': data.get('rating'),
            'comment': data.get('comment', ''),
            'query': data.get('query', ''),
            'response': data.get('response', ''),
            'timestamp': datetime.now(pytz.UTC).isoformat()
        }

        # Save feedback
        feedback_file = 'data/feedback.csv'

        import pandas as pd
        df = pd.DataFrame([feedback_entry])

        if os.path.exists(feedback_file):
            df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            df.to_csv(feedback_file, index=False)

        return jsonify({'success': True})

    except Exception as e:
        print(f"Error saving feedback: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# LEGACY ENDPOINT (kept for backward compatibility)
# ============================================================================

@app.route('/submit-correction', methods=['POST'])
def submit_correction():
    """
    Legacy endpoint - redirects to submit-solution
    Kept for backward compatibility with old frontend
    """
    try:
        data = request.json

        # Map old field names to new ones
        mapped_data = {
            'machine_type': data.get('machine_type', ''),
            'machine_name': data.get('machine_name', ''),
            'problem': data.get('problem', ''),
            'root_cause': data.get('root_cause', ''),
            'solution': data.get('solution', ''),
            'user_id': 'anonymous'
        }

        # Call new endpoint
        return submit_solution()

    except Exception as e:
        print(f"Error in submit_correction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# EXPORT ENDPOINTS
# ============================================================================

@app.route('/export/chat-history', methods=['GET'])
def export_chat_history():
    """Export chat history as JSON"""
    try:
        chat_log_file = 'data/chat_logs.json'
        if os.path.exists(chat_log_file):
            return send_file(
                chat_log_file,
                mimetype='application/json',
                as_attachment=True,
                download_name='chat_history.json'
            )
        else:
            return jsonify({'error': 'No chat history found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export/corrections', methods=['GET'])
def export_corrections():
    """Export user corrections as CSV"""
    try:
        corrections_file = 'data/user_corrections.csv'
        if os.path.exists(corrections_file):
            return send_file(
                corrections_file,
                mimetype='text/csv',
                as_attachment=True,
                download_name='user_corrections.csv'
            )
        else:
            return jsonify({'error': 'No corrections found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _sanitize_for_json(obj):
    """
    Convert numpy types to Python native types for JSON serialization
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_sanitize_for_json(item) for item in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _log_chat(message, response):
    """Log chat conversations"""
    try:
        log_entry = {
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'message': message,
            'response': response
        }

        log_file = 'data/chat_logs.json'

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        # Keep only last 1000 entries
        logs = logs[-1000:]

        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    except Exception as e:
        print(f"Error logging chat: {e}")


def _log_user_contribution(data, result):
    """Log successful user contributions"""
    try:
        log_entry = {
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'machine_type': data.get('machine_type'),
            'machine_name': data.get('machine_name'),
            'problem': data.get('problem'),
            'entry_id': result.get('entry_id'),
            'user_id': data.get('user_id', 'anonymous')
        }

        log_file = 'data/contribution_log.json'

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    except Exception as e:
        print(f"Error logging contribution: {e}")


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("ENHANCED MACHINE TROUBLESHOOTING CHATBOT")
    print("="*60)
    print("Features:")
    print("  ✓ Mistral Nemo via Ollama (fast & efficient)")
    print("  ✓ Dynamic learning from user solutions")
    print("  ✓ Real-time knowledge base updates")
    print("  ✓ Solution effectiveness tracking")
    print("  ✓ Hybrid search (semantic + keyword)")
    print("="*60 + "\n")

    # Start Flask app
    app.run(
        host='172.19.78.50',
        port=9025,
        debug=False,
        threaded=True
    )
