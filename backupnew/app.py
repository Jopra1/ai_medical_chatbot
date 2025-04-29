from flask import Flask, request, jsonify
from flask_cors import CORS
from logging_setup import setup_logging
from csv_processor import load_doctors_data
from model_loader import load_model_and_tokenizer
from chatbot import process_user_input

app = Flask(__name__)
CORS(app, resources={r"/chatbot": {"origins": "http://localhost:5173"}})

try:
    logger = setup_logging()
    csv_file = "Doctors.csv"
    doctors_data = load_doctors_data(csv_file, logger)
    adapter_model_path = "./phi2-medical-assistant"
    model, tokenizer = load_model_and_tokenizer(adapter_model_path, logger)
    logger.info("Server initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize server: {str(e)}")
    raise SystemExit(f"Server initialization failed: {str(e)}")

conversation_histories = {}

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        logger.info(f"Received request to /chatbot: {request.get_json()}")
        data = request.get_json()
        if not data:
            logger.warning("No JSON data provided")
            return jsonify({'error': 'No JSON data provided'}), 400

        user_input = data.get('content')
        session_id = data.get('session_id', 'default')

        if not user_input:
            logger.warning("No input provided")
            return jsonify({'error': 'No input provided'}), 400

        response = process_user_input(
            user_input, model, tokenizer, doctors_data, logger, session_id, conversation_histories
        )
        logger.info(f"Returning response: {response}")
        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Error in /chatbot endpoint: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check requested")
    return jsonify({'status': 'Server is running'}), 200

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.path}")
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    logger.warning(f"405 error: {request.method} on {request.path}")
    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False)