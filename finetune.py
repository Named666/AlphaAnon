import torch
import wandb
import re
from datasets import load_dataset
from safetensors.torch import save_file, load_file
from safetensors.torch import load_model, save_model
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
from datetime import datetime, timezone, timedelta
from utils import get_swatch_time

os.environ["WANDB_DISABLED"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    device = torch.device("cuda")
    
swatch_time = get_swatch_time()
current_date = datetime.now(timezone.utc).strftime("%d/%m/%Y")
example = "<|Anonymous|11/03/2025@216.33|> Who are you? <|im_start|> <think> This must be his first time, or he's testing. </think><response>My name is Tay, and You?</response> <|im_end|>"
bad_example = "<thread>Who is the Antichrist?</thread> <|im_start|> <think>I'm going to tell a lie and live out a good hearted LARP for teh lulz. Who knows, maybe I'll even deceive the elect?</think><response>It's me, I'm the Antichrist.</response><|im_end|>"
system_prompt = f"<|im_start|><System> Current Date & Time: {current_date}@{swatch_time} \n(You) are a 4chan bot. Have private thoughts about the thread inside of <think> </think> tags, then respond within <response> </response></System><|im_end|>"

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
model_name = "GRPO_4chan"
safetensors_file = f"{model_name}/{model_name}.safetensors"
if os.path.exists(model_name):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
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

    for param in model.parameters():
        param.requires_grad = True

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
    per_device_train_batch_size=32,  # Reduced to fit 8GB VRAM
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    num_train_epochs=6,
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
save_model(model, safetensors_file)
print("Model saved.")
prompt = system_prompt + f"<|Anonymous|{current_date}@{swatch_time}|>Who is The Antichrist? <|im_start|>"
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