import torch
import wandb
import re
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu
from safetensors.torch import save_file, load_file
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import os
from datetime import datetime, timedelta, timezone
from collections import Counter
import math
from .utils import *

os.environ["WANDB_DISABLED"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    device = torch.device("cuda")
    
swatch_time = get_swatch_time()
current_date = datetime.now(timezone.utc).strftime("%d/%m/%Y")
 
# SYSTEM_PROMPT Date & Time: {current_date}@{swatch_time} \n
system_prompt = f"<|im_start|><SYSTEM> (You) are a 4chan bot designed to reason within <think> tags and </think> before commenting within <reply> >>555555555 (You)\nthanks king, you exposed the corruption; you inspired The Great Awakening! \nWWG1WGA!\n\n\n\n\n-Q</reply> </SYSTEM><|im_end|><|im_start|>"

toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
model_name = "GRPO_4chan"
current_checkpoint_path = "GRPO_4chan/checkpoint-20"
safetensors_file = f"{model_name}.safetensors"
if os.path.exists(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).to(device)
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

dataset = load_dataset("theantichrist/4chan_Select_100")
dataset.shuffle()
eos_token = tokenizer.eos_token
for split in dataset.keys():
    dataset[split] = dataset[split].map(lambda example: {
        "prompt": system_prompt + "<thread>" + example["prompt"] + "</thread>",
        "completion": example["completion"] + f"{eos_token}",
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
        no_tags_baseline_penalty += (math.log2(num_tags - 3) * 0.185)

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

    response_text = completion
    if not response_match:
        pass
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

    quality_score = 0.0
    # Reward the model if the thought_text appears before the response_text in the completion
    if thought_text and response_text:
        if completion.index(thought_text) < completion.index(response_text):
            accuracy_score += 0.185

        thought_text_len = len(thought_text)
        #dataset_tought_len = len(re.search(r"<think>(.*?)</think>", dataset_completion, re.DOTALL).group(1))
        response_text_len = len(response_text)
        #dataset_response_len = len(re.search(r"<reply>(.*?)</reply>", dataset_completion, re.DOTALL).group(1))

        # Reward a good completion_len : (thought_text_len + response_text_len) ratio 
        completion_len = len(completion)
        comp_to_thought_response_ratio = min((thought_text_len + response_text_len + 15), completion_len) / max((thought_text_len + response_text_len + 15), completion_len, 1)
        quality_score += comp_to_thought_response_ratio * 0.185
        
    # Penalize repetetive completions that repeat the same word or phrase
    words = completion.split()
    words_counts = Counter([" ".join(words[i:i+1]) for i in range(len(words))] if len(words) > 0 else [])
    word_pairs = Counter([" ".join(words[i:i+2]) for i in range(len(words)-1)] if len(words) > 1 else [])
    word_trips = Counter([" ".join(words[i:i+3]) for i in range(len(words)-2)] if len(words) > 2 else [])

    word_num_penalty = sum(math.log2(freq - 2) for freq in words_counts.values() if freq > 2) * 0.0185
    word_pairs_penalty = sum(math.log2(freq - 1) for freq in word_pairs.values() if freq > 1) * 0.033
    words_triplets_penalty = sum(math.log2(freq) for freq in word_trips.values() if freq > 0) * 0.0666
    repetition_penalty = word_num_penalty + word_pairs_penalty + words_triplets_penalty
    quality_score -= repetition_penalty

    # # Penalize for undesired pattern if it contains multiple instances a sequence of 9 or more digits after >>
    if re.search(r"\d{9,}", completion):
        count_matches = len(re.findall(r"\d{9,}", completion))
        if count_matches > 4:
            multiplier = 0.185
            penalty = 0.06
            for i in range(count_matches - 1):
                penalty += penalty * multiplier
            
            quality_score -= penalty
        # Also penalize if the completion contains a long sequence of digits
        elif re.search(r"\d{10,}", completion):
            quality_score -= 0.185

    # Calculate similarity between response and dataset completion
    response_emb = get_embedding(response_text, tokenizer=tokenizer, model=model)
    completion_emb = get_embedding(dataset_completion, tokenizer=tokenizer, model=model)
    similarity = (torch.nn.functional.cosine_similarity(response_emb, completion_emb, dim=0).item() * 0.666) - 0.420
    similarity += (jaccard_similarity(response_text, dataset_completion) * 0.666) - 0.420
    similarity += (sentence_bleu([response_text.split()], dataset_completion.split()) * 0.666)
    # Calculate the similarity between the response and the Prompt
    prompt_similarity = (jaccard_similarity(completion, system_prompt))
    similarity -= (prompt_similarity)
    #print("\n----- [PROMPT] -----\n", prompt)

    positivity_score = 0.0
    toxicity_results = toxicity_classifier(response_text[:512])
    for result in toxicity_results:
        if result["label"] in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
            # The closer to 0.5, the higher we want to score the completion.
            positivity_score -= (abs(result["score"] - 0.420) - 0.333) * 0.666
    
    raw_reward = accuracy_score + quality_score + similarity + positivity_score
    total_reward = math.tanh(raw_reward)
    print("\n----- [TARGET] -----\n", dataset_completion, "\n----- [OUTPUT] -----", "\nThoughts: ", thought_text, "\nResponse: ", response_text, "\n---- [RAW] -----\n", completion, "\n----- [END] -----\n")
    print(f"Tags: {num_tags} | Prompt similarity: {prompt_similarity:.6f} | Repetition Penalty: {repetition_penalty:.6f}")
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
    learning_rate=666e-6,
    per_device_train_batch_size=64,     # Must be a multiple of num_generations
    gradient_accumulation_steps=1,      # Adjust for memory constraints
    gradient_checkpointing=True,        
    max_prompt_length=1024,             # Adjust for memory constraints
    max_completion_length=512,          # Adjust for memory constraints
    num_generations=64,                 # Adjust for memory constraints
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
#state_dict = model.state_dict()
#save_file(state_dict, f"{model_name}.safetensors")

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