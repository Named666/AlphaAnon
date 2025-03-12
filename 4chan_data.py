import requests
import json
import re
from datasets import Dataset
from datetime import datetime, timedelta, timezone

def get_swatch_time(dt):
    bmt_now = dt + timedelta(hours=1)  # BMT is UTC+1
    total_seconds = (bmt_now.hour * 3600) + (bmt_now.minute * 60) + bmt_now.second
    swatch_time = total_seconds / 86.4  # There are 86,400 seconds in a day
    return f"{int(swatch_time):03d}.{int((swatch_time % 1) * 100):02d}"

# Function to format the date Day/Month/Year format
def get_date(unix_timestamp):
    dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
    return dt.strftime("%d/%m/%Y")

def clean_text(text):
    if not text:
        return ""
    # Remove HTML tags and condense all repeating \n to a single \n using regex
    text = text.replace("&gt;", ">").replace("<br>", "\n").replace("<wbr>", "").replace("&quot;", "\"").replace("&amp;", "&").replace("&lt;", "<").replace("&apos;", "'").replace("&nbsp;", " ").replace("&#039;", "'").replace("’", "'").replace("“", "\"").replace("”", "\"").replace("—", "-").replace("–", "-").replace("…", "...")
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\n+", "\n", text)
    return text

# --- 1. Fetching and Preprocessing 4chan Thread Data ---
def fetch_4chan_data(board="pol", num_threads=100, max_prior_posts=100, max_length=2048):
    url = f"https://a.4cdn.org/{board}/threads.json"
    response = requests.get(url)
    threads_data = response.json()
    latest_threads = []
    for page in threads_data:
        for thread in page["threads"]:
            latest_threads.append(thread)

    # sort threads by number of replies
    latest_threads = sorted(latest_threads, key=lambda x: x["replies"], reverse=True)

    dataset = []
    for thread in latest_threads[:num_threads]:
        thread_id = thread["no"]
        thread_url = f"https://a.4cdn.org/{board}/thread/{thread_id}.json"
        thread_response = requests.get(thread_url)
        thread_data = thread_response.json()

        posts = thread_data["posts"]
        for i in range(1, len(posts)):
            prior_posts = []
            for p in posts[max(0, i-max_prior_posts):i]:
                if "com" in p:
                    post_number = p["no"]
                    try:
                        post_flag = p["country"]
                    except KeyError:
                        post_flag = p["board_flag"] if "board_flag" in p else "Unknown"
                    post_text = clean_text(p["com"])
                    post_name = p["name"] if "name" in p else "Anonymous"
                    post_time = p["time"]
                    if "id" in p:
                        post_name = f"{post_name} (ID: {p['id']})"
                    date_time = f"{get_date(post_time)}@{get_swatch_time(datetime.fromtimestamp(post_time, tz=timezone.utc))}"
                    if p == posts[0]:
                        prior_posts.append(f"|(OP) {post_name}|{post_flag}|{date_time}|{post_number}|\n{post_text}")
                    else:
                        prior_posts.append(f"|{post_name}|{post_flag}|{date_time}|{post_number}|\n{post_text}")
            # Check if all prior posts fit into max_length
            if sum(len(p) for p in prior_posts) > max_length:
                pass
            else:
                prompt = "".join(prior_posts)
                completion = posts[i]["com"] if "com" in posts[i] else ""
                # Check if completion contains a link (http, https, www) and skip if so
                if re.search(r"(http|https|www)", completion):
                    pass
                elif completion:
                    dataset.append({"thread": thread_id, "prompt": prompt, "completion": completion})
                

    dataset = [{"thread": d["thread"], "prompt": d["prompt"], "completion": clean_text(d["completion"])} 
               for d in dataset]
    
    with open("4chan_thread_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    return Dataset.from_list(dataset)


train_dataset = fetch_4chan_data()
train_dataset.save_to_disk("4chan_thread_dataset")
print(f"Collected {len(train_dataset)} thread-contextual pairs from /pol/")

# Useful for feeding the data to a regular fine-tuning pipeline (rather than GRPO)
def preprocess_function(examples, idx):
    return {"prompt": examples["prompt"], "dataset_idx": idx}
train_dataset_with_idx = train_dataset.map(preprocess_function, with_indices=True)