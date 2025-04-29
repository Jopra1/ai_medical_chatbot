from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Path to the adapter folder
adapter_model_path = "./phi2-medical-assistant"

# Load PEFT adapter config to get base model
peft_config = PeftConfig.from_pretrained(adapter_model_path)
base_model_name = peft_config.base_model_name_or_path

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Fix the attention mask warning by setting a pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token (common fix)
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Load the adapter weights (fine-tuned part)
model = PeftModel.from_pretrained(model, adapter_model_path)
model.eval()

if torch.cuda.is_available():
    model.cuda()

print("Model loaded! You can now chat.\n")

# Maintain conversation history
conversation_history = []

# Interactive loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Append user input to conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Format the input to match training data structure
    # Create a prompt that includes the full conversation history
    prompt = ""
    for turn in conversation_history:
        if turn["role"] == "user":
            prompt += f"User: {turn['content']}\n"
        elif turn["role"] == "assistant":
            prompt += f"Assistant: {turn['content']}\n"
    prompt += "Assistant: "  # Indicate the assistant's turn

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.5,  # Lower temperature for less randomness
            top_k=40,        # Slightly tighter top-k sampling
            top_p=0.9,       # Slightly tighter top-p sampling
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),  # Explicit attention mask
        )

    # Decode the response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (remove the prompt)
    assistant_response = response[len(prompt):].strip()

    # Clean up the response to ensure it doesn't include unwanted text
    if assistant_response.startswith("Assistant:"):
        assistant_response = assistant_response[len("Assistant:"):].strip()

    print("Assistant:", assistant_response)

    # Append assistant response to conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})