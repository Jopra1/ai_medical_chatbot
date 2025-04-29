from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from pymongo import MongoClient
import json

# Custom stopping criteria to stop generation at "User:"
class StopOnUserPrompt(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.user_prompt = "User:"

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0][-5:], skip_special_tokens=True)
        return self.user_prompt in generated_text

# MongoDB connection
mongo_uri = "mongodb+srv://aswin22:aswin2003@cluster0.i0du4.mongodb.net/"
client = MongoClient(mongo_uri)
db = client['medical']
doctors_collection = db['doctors']

# Load dynamic disease-to-specialty mapping
with open("disease_to_specialty_mapping_dict.json", "r") as f:
    specialty_mapping = json.load(f)

# Function to search doctors by specialty
def search_doctors_by_specialty(specialty):
    try:
        doctors = doctors_collection.find({
            "$or": [
                {"specialty": {"$regex": f"^{specialty}$", "$options": "i"}},
                {"speciality": {"$regex": f"^{specialty}$", "$options": "i"}}
            ]
        })

        doctor_list = []
        for doc in doctors:
            doctor_info = (
                f"Dr. {doc.get('name', 'N/A')} - {doc.get('speciality', doc.get('specialty', 'N/A'))}, "
                f"Contact: {doc.get('contact', 'N/A')}, "
                f"Hospital: {doc.get('hospital', 'N/A')}, "
                f"Consultation Time: {doc.get('consultation_time', 'N/A')}, "
                f"Rating: {doc.get('rating', 'N/A')}"
            )
            doctor_list.append(doctor_info)

        return doctor_list if doctor_list else ["No doctors found for this specialty."]
    except Exception as e:
        return [f"Error searching doctors: {str(e)}"]

# Load model and adapter
adapter_model_path = "./phi2-medical-assistant"
peft_config = PeftConfig.from_pretrained(adapter_model_path)
base_model_name = peft_config.base_model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model = PeftModel.from_pretrained(model, adapter_model_path)
model.eval()

if torch.cuda.is_available():
    model.cuda()

print("Model loaded! You can now chat.\n")

conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
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
