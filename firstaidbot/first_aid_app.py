from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import os

app = Flask(__name__)

# Enable CORS for requests from http://localhost:5173
CORS(app, resources={r"/first_aid": {"origins": "http://localhost:5173"}})

# Silence "not accessed" warning
torch.set_grad_enabled(False)

# Define CSV path
csv_path = os.path.join(os.path.dirname(__file__), "first_aid_dataset.csv")

# Load dataset
try:
    data = pd.read_csv(csv_path)
    symptoms = data["Symptom"].tolist()
    recommendations = data["Recommendation"].tolist()
    print(f"Loaded CSV with {len(data)} entries.")
except FileNotFoundError:
    print(f"Error: '{csv_path}' not found. Please ensure 'first_aid_dataset.csv' is in: {os.path.dirname(csv_path)}")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: '{csv_path}' is empty or corrupted.")
    exit(1)

# Load MiniLM model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Compute embeddings
symptom_embeddings = model.encode(symptoms, convert_to_tensor=True)

def get_recommendation(user_input, threshold=0.6):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, symptom_embeddings)
    best_match_idx = similarities.argmax()
    best_similarity = similarities[0][best_match_idx].item()
    
    if best_similarity < threshold:
        return "Sorry, I don’t understand. Please describe your symptom clearly or call emergency services."
    
    recommendation = recommendations[best_match_idx]
    return f"{recommendation}\n\nDisclaimer: This is general first aid advice. Call emergency services for severe injuries."

@app.route("/first_aid", methods=["POST"])
def first_aid():
    try:
        user_input = request.json.get("symptom", "")
        if not user_input:
            return jsonify({"error": "Please provide a symptom."}), 400
        recommendation = get_recommendation(user_input)
        return jsonify({"recommendation": recommendation})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server error. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5004)  