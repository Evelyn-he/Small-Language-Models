from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import create_user_session, process_message

multiprocessing.set_start_method('spawn', force=True)

app = Flask(__name__)
CORS(app)

sessions = {}

class Args:
    def __init__(self):
        self.verbose = True
        self.generate_data = False

@app.route('/api/session/start', methods=['POST'])
def start_session():
    try:
        data = request.json
        user_id = data.get('user_id')
        
        print(f"\n{'='*60}")
        print(f"SESSION START REQUEST")
        print(f"{'='*60}")
        print(f"User ID received: {user_id}")
        
        if not user_id:
            print("Error: No user ID provided")
            return jsonify({'error': 'User ID is required'}), 400
        
        try:
            user_id = int(user_id)
            print(f"User ID converted to integer: {user_id}")
        except ValueError:
            print(f"Error: User ID '{user_id}' is not a valid number")
            return jsonify({'error': 'User ID must be a number'}), 400
        
        print(f"Creating session for user {user_id}...")
        args = Args()
        
        try:
            retrievers, router = create_user_session(args, user_id)
            print(f"Retrievers and router created successfully")
        except Exception as e:
            print(f"Failed to create session: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        sessions[user_id] = {
            'retrievers': retrievers,
            'router': router,
            'conversation': [],
            'filtered_convo': [],
            'args': args
        }
        
        print(f"Session stored for user {user_id}")
        print(f"Active sessions: {list(sessions.keys())}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'message': 'Session started successfully',
            'user_id': user_id
        })
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR IN SESSION START")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_id = data.get('user_id')
        message = data.get('message')
        
        print(f"\n{'='*60}")
        print(f"CHAT MESSAGE REQUEST")
        print(f"{'='*60}")
        print(f"User ID: {user_id}")
        print(f"Message: {message}")
        print(f"Active sessions: {list(sessions.keys())}")
        
        if not user_id or not message:
            print("Error: Missing user ID or message")
            return jsonify({'error': 'User ID and message are required'}), 400
        
        try:
            user_id = int(user_id)
        except ValueError:
            print(f"Error: Invalid user ID '{user_id}'")
            return jsonify({'error': 'User ID must be a number'}), 400
        
        if user_id not in sessions:
            print(f"Session not found for user {user_id}")
            print(f"   Available sessions: {list(sessions.keys())}")
            print(f"{'='*60}\n")
            return jsonify({'error': 'Session not found. Please start a new session.'}), 404
        
        session = sessions[user_id]
        print("Session found")
        print("Processing message...")
        
        reply = process_message(
            user_id=user_id,
            user_input=message,
            args=session['args'],
            conversation=session['conversation'],
            filtered_convo=session['filtered_convo'],
            retrievers=session['retrievers'],
            router=session['router']
        )
        
        print(f"Reply generated: {reply[:100]}...")
        print(f"{'='*60}\n")
        
        return jsonify({
            'reply': reply,
            'confidence': True,
            'used_llm': False,
            'user_id': user_id
        })
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR IN CHAT")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/end', methods=['POST'])
def end_session():
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        try:
            user_id = int(user_id)
        except ValueError:
            return jsonify({'error': 'User ID must be a number'}), 400
        
        if user_id in sessions:
            session = sessions[user_id]
            try:
                if hasattr(session['router'], 'close'):
                    session['router'].close()
                for r in session['retrievers']:
                    if hasattr(r, 'close'):
                        r.close()
            except Exception as e:
                print(f"Error closing session resources: {e}")
            del sessions[user_id]
            print(f"Session ended for user {user_id}")
        
        return jsonify({
            'success': True,
            'message': 'Session ended successfully'
        })
        
    except Exception as e:
        print(f"Error in end_session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'sessions': len(sessions),
        'active_users': list(sessions.keys())
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Customer Support API Server Starting...")
    print("="*60)
    print("API will be available at: http://localhost:5001")
    print("Health check: http://localhost:5001/health")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)