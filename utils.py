from datetime import datetime, timedelta, timezone
import torch

def get_swatch_time():
    bmt_now = datetime.now(timezone.utc) + timedelta(hours=1)  # BMT is UTC+1
    total_seconds = (bmt_now.hour * 3600) + (bmt_now.minute * 60) + bmt_now.second
    swatch_time = total_seconds / 86.4  # There are 86,400 seconds in a day
    return f"{int(swatch_time):03d}.{int((swatch_time % 1) * 100):02d}"

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    inputs = {key: value.long() for key, value in inputs.items()}  # Ensure input_ids are of type LongTensor
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1].mean(dim=1).squeeze()

def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)
