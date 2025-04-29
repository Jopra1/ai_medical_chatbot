import logging
import os
from fuzzywuzzy import process
from pymongo import MongoClient
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList

# Set up logging (reduced verbosity)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom stopping criteria to stop generation at "User:"
class StopOnUserPrompt(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.user_prompt = "User:"

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0][-5:], skip_special_tokens=True)
        return self.user_prompt in generated_text

# MongoDB connection
mongo_uri = "mongodb+srv://aswin22:aswin2003@cluster0.i0du4.mongodb.net/Generalmedicine?retryWrites=true&w=majority"
try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.server_info()  # Test connection
    db = client['Generalmedicine']
    doctors_collection = db['Doctors']
    doc_count = doctors_collection.count_documents({})
    if doc_count == 0:
        logger.warning("Doctors collection is empty. Please populate the database.")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise SystemExit("Exiting due to MongoDB connection failure")

# Load dynamic disease-to-specialty mapping
json_file = "disease_to_specialty_mapping_dict.json"
try:
    if not os.path.exists(json_file):
        logger.error(f"JSON file not found: {json_file}")
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    with open(json_file, "r") as f:
        specialty_mapping = json.load(f)
except Exception as e:
    logger.error(f"Failed to load JSON file: {str(e)}")
    raise SystemExit("Exiting due to JSON file loading failure")

# Fallback normalization dictionary for common specialties
specialty_normalization = {
    "dermatology": "Dermatology",
    "a dermatology specialist": "Dermatology",
    "neurology": "Neurology",
    "a neurology specialist": "Neurology",
    "neurology or neurosurgery": "Neurology & Neurosurgery",
    "a neurology or neurosurgery specialist": "Neurology & Neurosurgery",
    "pulmonology": "Pulmonology",
    "a pulmonology specialist": "Pulmonology",
    "otolaryngology": "Otolaryngology",
    "otolaryngology (ent)": "Otolaryngology",
    "ent": "Otolaryngology",
    "an otolaryngology (ent) specialist": "Otolaryngology",
    "vascular surgery": "Vascular Surgery",
    "a vascular surgery specialist": "Vascular Surgery",
    "cardiology": "Cardiology",
    "a cardiology specialist": "Cardiology",
    "urology": "Urology",
    "a urology specialist": "Urology",
    "orthopedics": "Orthopedics",
    "a orthopedics specialist": "Orthopedics"
}

# Helper function to clean specialty names
def clean_specialty(specialty):
    try:
        cleaned = re.sub(r'\b(a|an)\b', '', specialty, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'\bspecialist\b', '', cleaned, flags=re.IGNORECASE).strip()
        if "(ent)" in cleaned.lower():
            cleaned_base = cleaned.replace("(ENT)", "").strip()
            return [cleaned_base, "ENT"]
        return [cleaned]
    except Exception as e:
        logger.error(f"Error cleaning specialty '{specialty}': {str(e)}")
        return [specialty]

# Helper function to get all unique specialties from the database
def get_all_specialties():
    try:
        specialties = doctors_collection.distinct("specialty")
        specialties += doctors_collection.distinct("speciality")
        specialties = list(set([s for s in specialties if s]))
        return specialties
    except Exception as e:
        logger.error(f"Error fetching specialties: {str(e)}")
        return []

# Function to search doctors by specialty with fuzzy matching
def search_doctors_by_specialty(specialty):
    try:
        # Clean the specialty name
        cleaned_specialties = clean_specialty(specialty)

        # Normalize each cleaned specialty
        normalized_specialties = []
        for cleaned in cleaned_specialties:
            cleaned_lower = cleaned.lower()
            normalized = specialty_normalization.get(cleaned_lower, cleaned)
            normalized_specialties.append(normalized)

        # Get all specialties from the database
        all_specialties = get_all_specialties()
        if not all_specialties:
            return ["No specialties found in the database. Please check the database connection or data."]

        # Try fuzzy matching (case-insensitive) with a stricter score cutoff
        matched_specialty = None
        best_score = 0
        all_specialties_lower = [s.lower() for s in all_specialties]
        for normalized in normalized_specialties:
            best_match = process.extractOne(normalized.lower(), all_specialties_lower, score_cutoff=80)  # Increased to 80 for stricter matching
            if best_match and best_match[1] > best_score:
                matched_specialty = all_specialties[all_specialties_lower.index(best_match[0])]
                best_score = best_match[1]

        if not matched_specialty:
            return [f"No matching specialty found for '{specialty}' in the database. Try contacting a general practitioner."]

        # Query the database with the matched specialty
        doctors = doctors_collection.find({
            "$or": [
                {"specialty": {"$regex": f"^{matched_specialty}$", "$options": "i"}},
                {"speciality": {"$regex": f"^{matched_specialty}$", "$options": "i"}}
            ]
        })

        doctor_list = []
        for doc in doctors:
            # Verify the doctor's specialty matches the matched_specialty exactly (case-insensitive)
            doc_specialty = doc.get('specialty', doc.get('speciality', ''))
            if doc_specialty.lower() != matched_specialty.lower():
                continue  # Skip doctors whose specialty doesn't exactly match
            doctor_info = (
                f"Dr. {doc.get('name', 'N/A')} - {doc_specialty}, "
                f"Contact: {doc.get('contact', 'N/A')}, "
                f"Hospital: {doc.get('hospital', 'N/A')}, "
                f"Consultation Time: {doc.get('consultation_time', 'N/A')}, "
                f"Rating: {doc.get('rating', 'N/A')}"
            )
            doctor_list.append(doctor_info)

        if not doctor_list:
            return [f"No doctors found for specialty '{matched_specialty}'. Try contacting a general practitioner."]
        return doctor_list
    except Exception as e:
        logger.error(f"Error in search_doctors_by_specialty: {str(e)}")
        return [f"Error searching doctors: {str(e)}"]

# Load model and adapter
adapter_model_path = "./phi2-medical-assistant"
try:
    peft_config = PeftConfig.from_pretrained(adapter_model_path)
    base_model_name = peft_config.base_model_name_or_path
except Exception as e:
    logger.error(f"Failed to load PeftConfig: {str(e)}")
    raise SystemExit("Exiting due to PeftConfig loading failure")

try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
except Exception as e:
    logger.error(f"Failed to load tokenizer: {str(e)}")
    raise SystemExit("Exiting due to tokenizer loading failure")

try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = PeftModel.from_pretrained(model, adapter_model_path)
    model.eval()
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise SystemExit("Exiting due to model loading failure")

if torch.cuda.is_available():
    try:
        model.cuda()
    except Exception as e:
        logger.error(f"Failed to move model to GPU: {str(e)}")
        raise SystemExit("Exiting due to GPU setup failure")

print("Model loaded! You can now chat.\n")

conversation_history = []

try:
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot")
            break

        conversation_history.append({"role": "user", "content": user_input})

        prompt = ""
        for turn in conversation_history:
            if turn["role"] == "user":
                prompt += f"User: {turn['content']}\n"
            elif turn["role"] == "assistant":
                prompt += f"Assistant: {turn['content']}\n"
        prompt += "Assistant: "

        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        stopping_criteria = StoppingCriteriaList([StopOnUserPrompt(tokenizer)])

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.5,
                top_k=40,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
                stopping_criteria=stopping_criteria,
            )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        assistant_response = response[len(prompt):].strip()

        if "User:" in assistant_response:
            assistant_response = assistant_response.split("User:")[0].strip()

        # Identify specialty only if assistant is giving a final recommendation
        specialty = None
        trigger_words = ["recommend", "consult", "specialist", "evaluation"]
        if any(word in assistant_response.lower() for word in trigger_words):
            response_lower = assistant_response.lower()
            for disease, mapped_specialty in specialty_mapping.items():
                if disease.lower() in response_lower:
                    specialty = mapped_specialty
                    break

            # Fallback check in user input
            if not specialty:
                user_input_lower = user_input.lower()
                for disease, mapped_specialty in specialty_mapping.items():
                    if disease.lower() in user_input_lower:
                        specialty = mapped_specialty
                        break

            if specialty:
                doctor_results = search_doctors_by_specialty(specialty)
                assistant_response += "\n\nRecommended Doctors:\n" + "\n".join(doctor_results)

        print("Assistant:", assistant_response)
        conversation_history.append({"role": "assistant", "content": assistant_response})

except Exception as e:
    logger.error(f"Error in chatbot loop: {str(e)}")
    raise SystemExit("Exiting due to error in chatbot loop")

finally:
    client.close()