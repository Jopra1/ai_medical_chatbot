from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList

# Custom stopping criteria to stop generation at "User:"
class StopOnUserPrompt(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.user_prompt = "User:"

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0][-5:], skip_special_tokens=True)
        return self.user_prompt in generated_text

# Path to the adapter folder
adapter_model_path = "./phi2-medical-assistant"

# Load PEFT adapter config to get base model
peft_config = PeftConfig.from_pretrained(adapter_model_path)
base_model_name = peft_config.base_model_name_or_path

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Fix the attention mask warning by setting a pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Load the adapter weights
model = PeftModel.from_pretrained(model, adapter_model_path)
model.eval()

if torch.cuda.is_available():
    model.cuda()

print("Model loaded! You can now chat.\n")

# Maintain conversation history and state
conversation_history = []
follow_up_count = 0  # Track number of follow-up questions
max_follow_ups = 3   # Maximum number of follow-up questions before diagnosis

def is_follow_up_question(response):
    """Check if the response is a follow-up question."""
    return response.strip().startswith("Do you") or "?" in response

def is_diagnosis(response):
    """Check if the response is a diagnosis."""
    return "It sounds like" in response or "I recommend consulting" in response

# Interactive loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Append user input to conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Check if the last user input is "no" and the previous assistant response was a follow-up question
    force_follow_up = False
    if (len(conversation_history) >= 2 and
        conversation_history[-1]["content"].lower() == "no" and
        is_follow_up_question(conversation_history[-2]["content"]) and
        follow_up_count < max_follow_ups):
        force_follow_up = True
        follow_up_count += 1

    # Format the input to match training data structure
    prompt = ""
    for turn in conversation_history:
        if turn["role"] == "user":
            prompt += f"User: {turn['content']}\n"
        elif turn["role"] == "assistant":
            prompt += f"Assistant: {turn['content']}\n"
    
    # If forcing a follow-up question, add a hint to the prompt
    if force_follow_up:
        prompt += "Assistant: Ask another follow-up question about a different symptom. "
    else:
        prompt += "Assistant: "

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    # Define stopping criteria
    stopping_criteria = StoppingCriteriaList([StopOnUserPrompt(tokenizer)])

    # Generate response
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

    # Decode the response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    assistant_response = response[len(prompt):].strip()

    # Clean up any residual "User:" or extra content
    if "User:" in assistant_response:
        assistant_response = assistant_response.split("User:")[0].strip()

    # If the response is a diagnosis and we're forcing a follow-up, retry
    if force_follow_up and is_diagnosis(assistant_response):
        prompt += "Do not provide a diagnosis. Ask another follow-up question about a different symptom. "
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
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

    print("Assistant:", assistant_response)

    # Append assistant response to conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Reset follow-up count if a diagnosis is provided
    if is_diagnosis(assistant_response):
        follow_up_count = 0