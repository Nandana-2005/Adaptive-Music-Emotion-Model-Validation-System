from flask import Flask, request, jsonify
from flask_cors import CORS
from database import *
import uuid

app = Flask(__name__)
CORS(app)  


create_tables()

@app.route('/')
def home():
    return "AMECS API is running!"

@app.route('/api/log-selection', methods=['POST'])
def log_selection():
    data = request.json
    
    interaction_id = log_interaction(
        user_id=data['user_id'],
        emotion=data['emotion'],
        music_file=data['music_file'],
        session_id=data.get('session_id', str(uuid.uuid4()))
    )
    
    return jsonify({
        'status': 'success',
        'interaction_id': interaction_id
    })

@app.route('/api/get-history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    """Get all selections for a child"""
    history = get_user_history(user_id)
    
    
    formatted = [
        {
            'emotion': row[0],
            'music_file': row[1],
            'timestamp': row[2]
        }
        for row in history
    ]
    
    return jsonify({
        'status': 'success',
        'history': formatted
    })

@app.route('/api/create-user', methods=['POST'])
def create_user():
    """Create a new child profile"""
    data = request.json
    user_id = insert_user(data['name'], data['age'])
    
    return jsonify({
        'status': 'success',
        'user_id': user_id
    })

@app.route('/api/save-feedback', methods=['POST'])
def api_save_feedback():
    data = request.json
    save_feedback(
        interaction_id=data['interaction_id'],
        accurate=data['accurate'],
        notes=data.get('notes', '')
    )
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
