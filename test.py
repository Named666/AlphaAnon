from transformers import pipeline
from safetensors.torch import save_file, load_file
from safetensors.torch import load_model, save_model
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    device = torch.device("cuda")

model_name = "GRPO_4chan"
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
model_name = "GRPO_4chan"
last_chpt = "GRPO_4chan/V01_checkpoint"
safetensors_file = f"{model_name}/{model_name}.safetensors"
if os.path.exists(model_name):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).to(device)
        # Load from safetensors file
        model_state_dict = load_model(model=model, filename=safetensors_file)
        if isinstance(model_state_dict, tuple):
            model_state_dict = model_state_dict[0]
        if isinstance(model_state_dict, set):
            model_state_dict = {k: v for k, v in model_state_dict}
        model.load_state_dict(model_state_dict, strict=False)
        model.to(device)
        print("Loaded model safetensors...")
    except FileNotFoundError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).to(device)
        print("Loaded model from local files...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(model_name)
    for param in model.parameters():
        param.requires_grad = True
    tokenizer.save_pretrained(model_name)




messages = [
    {"role": "user", "content": prompt},
]

generate_kwargs = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.5,
    "min_p": 0.1,
}
generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer, **generate_kwargs)