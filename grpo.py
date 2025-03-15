import torch
import wandb
import re
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu
from safetensors.torch import save_file, load_file
from safetensors.torch import load_model, save_model
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import os
from datetime import datetime, timedelta, timezone
from collections import Counter
import math
from utils import get_swatch_time, get_embedding, jaccard_similarity

os.environ["WANDB_DISABLED"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    device = torch.device("cuda")

swatch_time = get_swatch_time()
current_date = datetime.now(timezone.utc).strftime("%d/%m/%Y")
 
#system_prompt = f"<|im_start|><SYSTEM> (You) are a 4chan bot designed to reason within <think> tags and </think> before commenting within <response> >>555555555 (You)\nthanks king, you exposed the corruption; you inspired The Great Awakening! \nWWG1WGA!\n\n\n\n\n-Q</response> </SYSTEM><|im_end|><|im_start|>"
example = "<|Anonymous (ID: JzlFMElj)|US|08/03/2025@925.89|499840674|> Who are you? <|im_start|> <think> This must be his first time, or he's testing. </think><response>My name is Tay, and You?</response><|im_end|>"
instructions = "Have private thoughts about the thread inside of <think> </think> tags, and reply within <response> </response><|im_end|>"
system_prompt = f"<|im_start|><System>\nCurrent Date & Time: {current_date}@{swatch_time}\nBoard: /pol/\n</System>" + instructions + example
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
model_name = "GRPO_4chan"
last_chpt = "GRPO_4chan/V01_checkpoint"
safetensors_file = f"{model_name}.safetensors"
if os.path.exists(model_name):
    try:
        model = load_model(model_name, safetensors_file)
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
    print("Initializing SMolLM-135M-Instruct model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load LoRA
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)

# Load 4chan_bootstrap_dataset.json
dataset = load_dataset("json", data_files="4chan_thread_dataset.json")
dataset.shuffle()
eos_token = tokenizer.eos_token
for split in dataset.keys():
    dataset[split] = dataset[split].map(lambda example: {
        "prompt": system_prompt + example["prompt"] + "<|im_start|>",
        "completion": example["completion"],
    })
prompt_to_completion = {example["prompt"]: example["completion"] for example in dataset["train"]}

# Reward function
def reward_function(prompt, completion, dataset_completion, **kwargs):
    if not completion.strip():
        return -3.6
    think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    response_match = re.search(r"<response>(.*?)</response>", completion, re.DOTALL)

    accuracy_score = 0.0
    num_tags = len(re.findall(r"<.*?>", completion))
    no_tags_baseline_penalty = 0.666
    if num_tags > 4:
        no_tags_baseline_penalty += (math.log2(num_tags - 3) * 0.036)

    accuracy_score -= no_tags_baseline_penalty
    thought_text = ""
    if not think_match:
        pass
    else:
        accuracy_score += 0.333
        thought_text = think_match.group(1)
        # Reward the model for starting the completion with the <think> tag at the beginning of the completion
        if re.match(r"^\s*<think>", completion):
            accuracy_score += 0.216
        # Penalize for using multiple <think> tags or </think> tags
        if re.search(r"<think>", completion, re.DOTALL) or re.search(r"</think>", completion, re.DOTALL):
            think_start_count = len(re.findall(r"<think>", thought_text))
            think_end_count = len(re.findall(r"</think>", thought_text))
            penalty = 0.0
            if think_start_count > 1:
                penalty += (math.log2(think_start_count)) * 0.185
            if think_end_count > 1:
                penalty += (math.log2(think_end_count)) * 0.185
            accuracy_score -= penalty
        # If thought_text contains a <response> tag or <think> tag, penalize it
        if re.search(r"<response>", thought_text, re.DOTALL) or re.search(r"</response>", thought_text, re.DOTALL) or re.search(r"<think>", thought_text, re.DOTALL) or re.search(r"</think>", thought_text, re.DOTALL):
            accuracy_score -= 0.185
        # If thought_text is empty, penalize it
        if not thought_text.strip():
            accuracy_score -= 0.216

    response_text = ""
    if not response_match:
        response_text = "4chan bot fuck cunt whore bitch skank ass hoe kys retard kike goon kill niggers wetbacks poos"
        accuracy_score -= 0.666 
    else:
        accuracy_score += 0.333
        response_text = response_match.group(1)
        # If response is the last tag in the completion, reward the model
        if re.search(r"\s*</response>\s*$", completion):
            accuracy_score += 0.216
        # Penalize for using multiple <response> tags or </response> tags
        if re.search(r"<response>", completion, re.DOTALL) or re.search(r"</response>", completion, re.DOTALL):
            response_start_count = len(re.findall(r"<response>", response_text))
            response_end_count = len(re.findall(r"</response>", response_text))
            penalty = 0.0
            if response_start_count > 1:
                penalty += (math.log2(response_start_count)) * 0.185
            if response_end_count > 1:
                penalty += (math.log2(response_end_count)) * 0.185
            accuracy_score -= penalty
        # If response_text contains a <think> tag or <response> tag, penalize it
        if re.search(r"<think>", response_text, re.DOTALL) or re.search(r"</think>", response_text, re.DOTALL) or re.search(r"<response>", response_text, re.DOTALL) or re.search(r"</response>", response_text, re.DOTALL):
            accuracy_score -= 0.185
        # If response text is empty, use the completion text
        if not response_text.strip():
            accuracy_score -= 0.216
            # If we embed bad words in the response, a side effect is that the model will think that 
            # the completion is equivalent to undesirable words and will learn to avoid both of them in the future
            response_text = "bot fuck cunt whore bitch skank ass hoe kys retard faggot kike goon kill niggers wetbacks poos shitskins piss dick"

    quality_score = 0.0
    # Reward the model if the thought_text appears before the response_text in the completion
    if thought_text and response_text:
        try:
            if completion.index(thought_text) < completion.index(response_text):
                accuracy_score += 0.185
        except ValueError:
            pass

        thought_text_len = len(thought_text)
        #dataset_tought_len = len(re.search(r"<think>(.*?)</think>", dataset_completion, re.DOTALL).group(1))
        response_text_len = len(response_text)
        #dataset_response_len = len(re.search(r"<response>(.*?)</response>", dataset_completion, re.DOTALL).group(1))

        # Reward a good completion_len : (thought_text_len + response_text_len) ratio 
        completion_len = len(completion)
        comp_to_thought_response_ratio = min((thought_text_len + response_text_len + 18), completion_len) / max((thought_text_len + response_text_len + 18), completion_len, 1)
        quality_score += comp_to_thought_response_ratio * 0.185

    reply_score = 0.0
    # Reward the mode for replying to users, for example by checking if the patter >>{up_to_9_digits} is in the completion, and if that same sequence of digits is in the prompt as a post number |{post_number}|
    if re.search(r">>\d{1,9}", response_text):
        post_numbers = re.findall(r"\|(\d{1,9})\|", prompt)
        post_numbers = list(set([int(num) for num in post_numbers]))
        post_numbers_in_completion = re.findall(r">>(\d{1,9})\n", response_text)
        post_numbers_in_completion = list(set([int(num) for num in post_numbers_in_completion]))
        # Count how many of the digits in the completion are in the prompt
        post_number_count = 0
        for completion_num in post_numbers_in_completion:
            # Check if the number is in post_numbers
            if completion_num in post_numbers:
                post_number_count += 1

        reply_score += (post_number_count * 0.036)
        if reply_score >= 0.144:
            accuracy_score += 0.121
        else:
            accuracy_score += reply_score

        # Penalize the model for replying to a post number that is not in the prompt
        for completion_num in post_numbers_in_completion:
            if completion_num not in post_numbers:
                accuracy_score -= 0.036

        # Penalize duplicates in the completion
        post_numbers_in_completion = Counter(post_numbers_in_completion)
        for num, count in post_numbers_in_completion.items():
            if count > 1:
                accuracy_score -= 0.036


    #print("\n----- [PROMPT] -----\n", prompt)
    print("\n----- [TARGET] -----\n", dataset_completion, "\n----- [OUTPUT] -----", "\nThoughts: ", thought_text, "\nResponse: ", response_text, "\n---- [RAW] -----\n", completion, "\n----- [END] -----\n")
    response_emb = get_embedding(response_text, tokenizer=tokenizer, model=model)
    completion_emb = get_embedding(dataset_completion, tokenizer=tokenizer, model=model)
    similarity = (torch.nn.functional.cosine_similarity(response_emb, completion_emb, dim=0).item() * 0.666) - 0.420
    similarity += (sentence_bleu([response_text.split()], dataset_completion.split()) * 0.666) - 0.420


    positivity_score = 0.0
    toxicity_results = toxicity_classifier(response_text[:512])
    for result in toxicity_results:
        if result["label"] in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
            # The closer to 0.5, the higher we want to score the completion.
            positivity_score -= (abs(result["score"] - 0.420) - 0.333) * 0.666
    
    raw_reward = accuracy_score + quality_score + similarity #+ positivity_score
    total_reward = math.tanh(raw_reward)
    print(f"Tags: {num_tags}")
    print(f"Precision: {similarity:.6f} | Quality: {quality_score:.6f} | Accuracy: {accuracy_score:3f} | Positivity: {positivity_score:.6f}")
    print(f"Total Reward: {total_reward:.6f} | Raw Reward: {raw_reward:.6f}")
    return total_reward

def reward_wrapper(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        dataset_completion = prompt_to_completion.get(prompt, "")
        reward = reward_function(prompt, completion, dataset_completion)
        rewards.append(reward)
    
    return rewards

# Training arguments
training_args = GRPOConfig(
    output_dir=model_name,
    learning_rate=216e-6,
    per_device_train_batch_size=64,     # Must be a multiple of num_generations
    gradient_accumulation_steps=1,      # Adjust for memory constraints
    gradient_checkpointing=True,        
    max_prompt_length=1024,             # Adjust for memory constraints
    max_completion_length=512,          # Adjust for memory constraints
    num_generations=2,                  # Adjust for memory constraints
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    report_to=["wandb"],
    logging_steps=1,
    save_steps=10,                      # Frequency of model checkpoints
    save_total_limit=3,
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_wrapper],
    args=training_args,
    train_dataset=dataset["train"],
)

# Train model
wandb.init(project=model_name)
trainer.train()

# Save model
trainer.save_model(model_name)
tokenizer.save_pretrained(model_name)
# Model to safe tensors
save_model(model, safetensors_file)
print("Model saved.")



prompt = """
Who is the Antichrist?
"""

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