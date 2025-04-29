from logging_setup import setup_logging
from csv_processor import load_doctors_data
from model_loader import load_model_and_tokenizer
from chatbot import run_chatbot

def main():
    # Setup logging
    logger = setup_logging()

    # Load CSV data
    csv_file = "Doctors.csv"
    doctors_data = load_doctors_data(csv_file, logger)

    # Load model and tokenizer
    adapter_model_path = "./phi2-medical-assistant"
    model, tokenizer = load_model_and_tokenizer(adapter_model_path, logger)

    print("Model loaded! You can now chat.\n")

    # Run chatbot
    run_chatbot(model, tokenizer, doctors_data, logger)

if __name__ == "__main__":
    main()