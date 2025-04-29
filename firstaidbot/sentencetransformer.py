from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch  # Required by sentence-transformers
import os

# Silence "not accessed" warning by using torch
torch.set_grad_enabled(False)  # Disable gradient computation for inference

# Define the CSV path
csv_path = os.path.join(os.path.dirname(__file__), "first_aid_dataset.csv")

# Load the dataset
try:
    data = pd.read_csv(csv_path)
    symptoms = data["Symptom"].tolist()
    recommendations = data["Recommendation"].tolist()
    print(f"Loaded CSV with {len(data)} entries.")  # Debug output
except FileNotFoundError:
    print(f"Error: '{csv_path}' not found. Please ensure 'first_aid_dataset.csv' is in: {os.path.dirname(csv_path)}")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: '{csv_path}' is empty or corrupted. Please check the CSV file.")
    exit(1)

# Load MiniLM model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded successfully.")  # Debug output
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Compute embeddings for all symptoms
symptom_embeddings = model.encode(symptoms, convert_to_tensor=True)

# Function to get recommendation for user input
def get_recommendation(user_input, threshold=0.6):
    # Encode the user's input
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.cos_sim(user_embedding, symptom_embeddings)
    
    # Find the index of the most similar symptom
    best_match_idx = similarities.argmax()
    best_similarity = similarities[0][best_match_idx].item()
    
    # Debug: Print matched symptom and similarity
    print(f"Matched symptom: {symptoms[best_match_idx]} (Similarity: {best_similarity:.2f})")
    
    # Check if similarity is above threshold
    if best_similarity < threshold:
        return f"Sorry, I don’t understand. Please describe your symptom clearly or call emergency services. (Similarity: {best_similarity:.2f})"
    
    # Return the recommendation with a disclaimer
    recommendation = recommendations[best_match_idx]
    return f"{recommendation}\n\nDisclaimer: This is general first aid advice. Call emergency services for severe injuries."

# Command-line chatbot interface
def chatbot():
    print("First Aid Chatbot: Enter a symptom (e.g., 'I burned my hand') or type 'quit' to exit.")
    print("Note: This is general first aid advice. Call emergency services for severe injuries.")
    
    while True:
        user_input = input("Your symptom: ").strip()
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter a symptom.")
            continue
        
        # Get and print the recommendation
        recommendation = get_recommendation(user_input)
        print(f"Recommendation: {recommendation}\n")

# Run the chatbot
if __name__ == "__main__":
    chatbot()