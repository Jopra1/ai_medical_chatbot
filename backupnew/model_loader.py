from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

def load_model_and_tokenizer(adapter_model_path, logger):
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

    return model, tokenizer