import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

def get_model_and_tokenizer(model_name="distilgpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    
    model = get_peft_model(model, peft_config)
    return model, tokenizer

if __name__ == "__main__":
    model, _ = get_model_and_tokenizer()
    model.print_trainable_parameters()