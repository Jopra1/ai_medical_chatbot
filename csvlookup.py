import logging
import os
import csv
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from fuzzywuzzy import process

# Set up logging with error handling
try:
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',     
        handlers=[
            logging.FileHandler('chatbot.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed successfully")
except Exception as e:
    print(f"Failed to configure logging: {str(e)}")
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

# Custom stopping criteria to stop generation at "User:"
class StopOnUserPrompt(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.user_prompt = "User:"

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0][-5:], skip_special_tokens=True)
        return self.user_prompt in generated_text

# Load doctor data from CSV file
csv_file = "Doctors.csv"
doctors_data = []
try:
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader, None)  # Get the header row
        if not headers:
            logger.error("No header row found in CSV file")
            raise SystemExit("Exiting due to missing header row in CSV file")

        logger.info(f"CSV headers: {headers}")
        # Map header names to indices
        header_indices = {header: idx for idx, header in enumerate(headers)}

        required_fields = ['name', 'speciality', 'contact', 'hospital', 'consultation_time', 'rating', 'qualification']
        for field in required_fields:
            if field not in header_indices:
                logger.warning(f"Header '{field}' not found in CSV headers. Using default 'N/A' for this field.")

        for i, row in enumerate(reader, start=2):  # Start from line 2 (after header)
            if not row or not any(field.strip() for field in row):
                logger.warning(f"Skipping empty row at line {i}")
                continue

            logger.info(f"Processing row at line {i}: {row}")
            # Ensure the row has enough fields
            while len(row) < len(headers):
                row.append('')

            # Map CSV fields to doctor dictionary using header indices
            doctor = {
                "name": row[header_indices['name']] if 'name' in header_indices and row[header_indices['name']].strip() else 'N/A',
                "specialty": row[header_indices['speciality']].lower() if 'speciality' in header_indices and row[header_indices['speciality']].strip() else 'N/A',
                "contact": row[header_indices['contact']] if 'contact' in header_indices and row[header_indices['contact']].strip() else 'N/A',
                "hospital": row[header_indices['hospital']] if 'hospital' in header_indices and row[header_indices['hospital']].strip() else 'N/A',
                "consultation_time": row[header_indices['consultation_time']] if 'consultation_time' in header_indices and row[header_indices['consultation_time']].strip() else 'N/A',
                "rating": float(row[header_indices['rating']]) if 'rating' in header_indices and row[header_indices['rating']].replace('.', '').isdigit() else 'N/A',
                "qualifications": row[header_indices['qualification']] if 'qualification' in header_indices and row[header_indices['qualification']].strip() else 'N/A'
            }

            if doctor["name"] != "N/A":
                doctors_data.append(doctor)
            else:
                logger.warning(f"Skipping row at line {i} with no name field: {row}")

    if not doctors_data:
        logger.error("No valid doctors found in CSV file after parsing")
        raise SystemExit("Exiting due to empty CSV file after parsing")

    logger.info(f"Loaded {len(doctors_data)} doctors from CSV")
    specialties_found = list(set(doctor["specialty"] for doctor in doctors_data if doctor["specialty"] != "N/A"))
    logger.info(f"Specialties in CSV data: {specialties_found}")
except Exception as e:
    logger.error(f"Failed to load CSV file: {str(e)}")
    raise SystemExit("Exiting due to CSV file loading failure")

# Fallback normalization dictionary for common specialties
specialty_normalization = {
    "dermatology": "dermatology",
    "a dermatology specialist": "dermatology",
    "neurology": "neurology",
    "a neurology specialist": "neurology",
    "neurology or neurosurgery": "neurology & neurosurgery",
    "a neurology or neurosurgery specialist": "neurology & neurosurgery",
    "pulmonology": "pulmonology",
    "a pulmonology specialist": "pulmonology",
    "otolaryngology": "otolaryngology",
    "otolaryngology (ent)": "otolaryngology",
    "ent": "otolaryngology",
    "an otolaryngology (ent) specialist": "otolaryngology",
    "vascular surgery": "vascular surgery",
    "a vascular surgery specialist": "vascular surgery",
    "cardiology": "cardiology",
    "a cardiology specialist": "cardiology",
    "urology": "urology",
    "a urology specialist": "urology",
    "orthopedics": "orthopedics",
    "a orthopedics specialist": "orthopedics",
    "gastroenterology": "gastroenterology",
    "a gastroenterology specialist": "gastroenterology",
    "anesthesiology": "anesthesiology",
    "a anesthesiology specialist": "anesthesiology",
    "infectious disease": "emergency medicine",
    "a infectious disease specialist": "emergency medicine",
    "emergency medicine": "emergency medicine",
    "a emergency medicine specialist": "emergency medicine"
}

# Helper function to clean specialty names
def clean_specialty(specialty):
    try:
        cleaned = re.sub(r'\b(a|an)\b', '', specialty, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'\bspecialist\b', '', cleaned, flags=re.IGNORECASE).strip()
        if "(ent)" in cleaned.lower():
            cleaned_base = cleaned.replace("(ENT)", "").strip()
            return [cleaned_base.lower(), "ent"]
        return [cleaned.lower()]
    except Exception as e:
        logger.error(f"Error cleaning specialty '{specialty}': {str(e)}")
        return [specialty.lower()]

# Helper function to extract specialty from assistant's response
def extract_specialty_from_response(response):
    match = re.search(r'(?:consult|recommend)\s+(?:a|an)?\s*([\w\s&]+(?:\s\(ent\))?)\s*specialist', response, re.IGNORECASE)
    if match:
        specialty = match.group(1).strip()
        return specialty
    for specialty in specialty_normalization.keys():
        if specialty in response.lower():
            return specialty
    return None

# Helper function to get all unique specialties from the CSV data
def get_all_specialties():
    try:
        specialties = [doctor["specialty"] for doctor in doctors_data if doctor["specialty"] != "N/A"]
        return list(set(specialties))
    except Exception as e:
        logger.error(f"Error fetching specialties: {str(e)}")
        return []

# Function to search doctors by specialty using CSV data
def search_doctors_by_specialty(specialty):
    try:
        logger.info(f"Searching for specialty: {specialty}")
        cleaned_specialties = clean_specialty(specialty)
        normalized_specialties = []
        for cleaned in cleaned_specialties:
            normalized = specialty_normalization.get(cleaned, cleaned)
            normalized_specialties.append(normalized)
        logger.info(f"Normalized specialties: {normalized_specialties}")

        all_specialties = get_all_specialties()
        if not all_specialties:
            logger.warning("No specialties found in the data")
            return ["No specialties found in the data. Please check the CSV file."]
        logger.info(f"All available specialties: {all_specialties}")

        matched_specialty = None
        for norm_specialty in normalized_specialties:
            if norm_specialty in all_specialties:
                matched_specialty = norm_specialty
                break

        if not matched_specialty:
            best_score = 0
            for norm_specialty in normalized_specialties:
                best_match = process.extractOne(norm_specialty, all_specialties, score_cutoff=80)
                if best_match and best_match[1] > best_score:
                    matched_specialty = best_match[0]
                    best_score = best_match[1]
            logger.info(f"Fuzzy match result: matched_specialty={matched_specialty}, score={best_score}")

        if not matched_specialty:
            logger.warning(f"No matching specialty found for '{specialty}' in the data")
            if "emergency medicine" in all_specialties:
                logger.info("Falling back to Emergency Medicine")
                fallback_results = search_doctors_by_specialty("emergency medicine")
                if fallback_results and not fallback_results[0].startswith("No doctors found"):
                    return [
                        f"No {specialty} specialists found in the data. However, you can consult an Emergency Medicine specialist for initial evaluation:"
                    ] + fallback_results
            return [f"No {specialty} specialists found in the data. Try contacting a general practitioner for a referral."]

        logger.info(f"Matched specialty: {matched_specialty}")
        doctor_list = []
        for doc in doctors_data:
            doc_specialty = doc.get('specialty', 'N/A')
            doc_name = doc.get('name', 'N/A')
            if doc_name == "N/A":
                continue
            if doc_specialty == matched_specialty:
                doctor_info = (
                    f"Dr. {doc_name} - {doc_specialty}, "
                    f"Contact: {doc.get('contact', 'N/A')}, "
                    f"Hospital: {doc.get('hospital', 'N/A')}, "
                    f"Consultation Time: {doc.get('consultation_time', 'N/A')}, "
                    f"Rating: {doc.get('rating', 'N/A')}"
                )
                doctor_list.append(doctor_info)

        if not doctor_list:
            logger.warning(f"No doctors found for matched specialty '{matched_specialty}'")
            if "emergency medicine" in all_specialties and matched_specialty != "emergency medicine":
                logger.info("No doctors found for specialty, falling back to Emergency Medicine")
                fallback_results = search_doctors_by_specialty("emergency medicine")
                if fallback_results and not fallback_results[0].startswith("No doctors found"):
                    return [
                        f"No {specialty} specialists found in the data. However, you can consult an Emergency Medicine specialist for initial evaluation:"
                    ] + fallback_results
            return [f"No doctors found for specialty '{matched_specialty}'. Try contacting a general practitioner for a referral."]
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

        # Identify specialty from the assistant's response
        specialty = None
        trigger_words = ["recommend", "consult", "specialist", "evaluation"]
        if any(word in assistant_response.lower() for word in trigger_words):
            specialty = extract_specialty_from_response(assistant_response)
            logger.info(f"Extracted specialty from response: {specialty}")

            if specialty:
                doctor_results = search_doctors_by_specialty(specialty)
                if doctor_results:
                    assistant_response += "\n\nRecommended Doctors:\n" + "\n".join(doctor_results)

        print("Assistant:", assistant_response)
        conversation_history.append({"role": "assistant", "content": assistant_response})

except Exception as e:
    logger.error(f"Error in chatbot loop: {str(e)}")
    raise SystemExit("Exiting due to error in chatbot loop")