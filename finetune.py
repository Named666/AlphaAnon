import torch
import wandb
import re
from datasets import load_dataset
from safetensors.torch import save_file, load_file
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
from datetime import datetime, timezone, timedelta
from utils import get_swatch_time

os.environ["WANDB_DISABLED"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

swatch_time = get_swatch_time()
current_date = datetime.now(timezone.utc).strftime("%d/%m/%Y")
example = "<thread>Who is the Antichrist?</thread> <think>I'm going to tell a lie and live out a good hearted LARP for teh lulz. Who knows, maybe I'll even deceive the elect?</think><reply>It's me, I'm the Antichrist.</reply><|im_end|><|im_start|>"
system_prompt = f"<|im_start|><System> Current Date & Time: {current_date}@{swatch_time} \n(You) are a 4chan bot. Have private thoughts about the thread inside of <think> </think> tags, then respond within <reply> </reply></System><|im_end|><|im_start|>"

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
model_name = "GRPO_4chan"
safetensors_file = f"{model_name}.safetensors"
if os.path.exists(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",)
    # Check if we have a checkpoint to load
    if os.path.exists(safetensors_file):
        state_dict = load_file(safetensors_file)  # Load state dictionary from safetensors file
        model.load_state_dict(state_dict)  # Load state dictionary into the model
        print("Loaded model from safetensors file.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    for param in model.parameters():
        param.requires_grad = True
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load LoRA
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)

dataset = load_dataset("theantichrist/4chan_Select_100")
dataset.shuffle()
eos_token = tokenizer.eos_token
for split in dataset.keys():
    dataset[split] = dataset[split].map(lambda example: {
        "prompt": system_prompt + "<thread>" + example["prompt"] + "</thread>",
        "completion": example["completion"] + f"{eos_token}",
    })
prompt_to_completion = {example["prompt"]: example["completion"] for example in dataset["train"]}


# Tokenize the dataset
def concat__and_tokenize(examples):
    # Concatenate prompt and completion
    combined_text = [p + c for p, c in zip(examples["prompt"], examples["completion"])]
    # Tokenize the combined text
    tokenized_output = tokenizer(combined_text)
    return tokenized_output

tokenized_datasets = dataset.map(concat__and_tokenize, batched=True, remove_columns=["prompt", "completion"])
# Print out some examples from the tokenized dataset
# for i in range(5):  # Adjust the range to print more or fewer examples
#     print(f"Example {i}:")
# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# Training arguments
training_args = TrainingArguments(
    output_dir=model_name,
    learning_rate=216e-6,
    per_device_train_batch_size=16,  # Reduced to fit 8GB VRAM
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    num_train_epochs=1,
    optim="adamw_8bit",
    bf16=True,
    logging_steps=1,
    report_to=["wandb"],
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

# Train model
wandb.init(project=model_name)
trainer.train()

# Save model
trainer.save_model(model_name)
tokenizer.save_pretrained(model_name)
print("Model saved.")
prompt = system_prompt + "Who is the Antichrist?</thread>"
#state_dict = model.state_dict()
#save_file(state_dict, f"{model_name}.safetensors")
# Maybe we should save the tokenizer as well

from transformers import pipeline

generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)

generate_kwargs = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.5,
    "min_p": 0.1,
}

generated_text = generator(prompt, **generate_kwargs)

print(generated_text)