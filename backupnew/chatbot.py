from transformers.generation import StoppingCriteria, StoppingCriteriaList
from specialty_normalizer import extract_specialty_from_response
from doctor_search import search_doctors_by_specialty
import torch

class StopOnUserPrompt(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.user_prompt = "User:"

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0][-5:], skip_special_tokens=True)
        return self.user_prompt in generated_text

def process_user_input(user_input, model, tokenizer, doctors_data, logger, session_id, conversation_histories):
    try:
        # Initialize conversation history for the session if not exists
        if session_id not in conversation_histories:
            conversation_histories[session_id] = []

        # Add user input to conversation history
        conversation_histories[session_id].append({"role": "user", "content": user_input})

        # Build prompt from conversation history
        prompt = ""
        for turn in conversation_histories[session_id]:
            if turn["role"] == "user":
                prompt += f"User: {turn['content']}\n"
            elif turn["role"] == "assistant":
                prompt += f"Assistant: {turn['content']}\n"
        prompt += "Assistant: "

        # Encode input
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # Set stopping criteria
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

        # Decode response
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        assistant_response = response[len(prompt):].strip()

        if "User:" in assistant_response:
            assistant_response = assistant_response.split("User:")[0].strip()

        # Check for specialty in response
        specialty = None
        trigger_words = ["recommend", "consult", "specialist", "evaluation"]
        if any(word in assistant_response.lower() for word in trigger_words):
            specialty = extract_specialty_from_response(assistant_response)
            logger.info(f"Extracted specialty from response: {specialty}")

            if specialty:
                doctor_results = search_doctors_by_specialty(specialty, doctors_data, logger)
                if doctor_results:
                    assistant_response += "\n\nRecommended Doctors:\n" + "\n".join(doctor_results)

        # Add assistant response to conversation history
        conversation_histories[session_id].append({"role": "assistant", "content": assistant_response})

        return assistant_response

    except Exception as e:
        logger.error(f"Error processing user input: {str(e)}")
        return f"Error: {str(e)}"

def run_chatbot(model, tokenizer, doctors_data, logger):
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

            specialty = None
            trigger_words = ["recommend", "consult", "specialist", "evaluation"]
            if any(word in assistant_response.lower() for word in trigger_words):
                specialty = extract_specialty_from_response(assistant_response)
                logger.info(f"Extracted specialty from response: {specialty}")

                if specialty:
                    doctor_results = search_doctors_by_specialty(specialty, doctors_data, logger)
                    if doctor_results:
                        assistant_response += "\n\nRecommended Doctors:\n" + "\n".join(doctor_results)

            print("Assistant:", assistant_response)
            conversation_history.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        logger.error(f"Error in chatbot loop: {str(e)}")
        raise SystemExit("Exiting due to error in chatbot loop")