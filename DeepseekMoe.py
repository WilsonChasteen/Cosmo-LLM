import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from coconut import Coconut

model_name = "deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# Initialize Coconut with correct token IDs
coconut_model = Coconut(
    base_causallm=model,
    latent_token_id=tokenizer.convert_tokens_to_ids("[LATENT]"),  # Replace with actual token
    start_latent_id=tokenizer.bos_token_id,  # 100000
    end_latent_id=tokenizer.eos_token_id,    # 100001
    eos_token_id=tokenizer.eos_token_id,
)