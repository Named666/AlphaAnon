import requests
import json
import re
from datasets import Dataset
from datetime import datetime, timedelta, timezone
from utils import get_swatch_time, clean_text, get_date

topic_filter_file = "topic_filter.txt"
with open(topic_filter_file, "r", encoding="utf-8") as f:
    topic_filter_list = f.read().split("\n")


def thread_filter(latest_threads):
    filtered_threads = []
    for thread in latest_threads:
        thread_text = thread["com"] if "com" in thread else ""
        thread_text = clean_text(thread_text)
        if any(topic in thread_text for topic in topic_filter_list):
            filtered_threads.append(thread)

    # Remove the filtered threads from the latest_threads list
    latest_threads = [thread for thread in latest_threads if thread not in filtered_threads]
    return latest_threads

# --- 1. Fetching and Preprocessing 4chan Thread Data ---
def fetch_4chan_data(board="pol", num_threads=5, max_prior_posts=6, max_length=2048, min_completion_length=154):
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
                    date_time = f"{get_date(post_time)}@{get_swatch_time(post_time, tz=timezone.utc)}"
                    if p == posts[0]:
                        prior_posts.append(f"\n<|(OP) {post_name}|{post_flag}|{date_time}|{post_number}|>\n{post_text}")
                    else:
                        prior_posts.append(f"\n<|{post_name}|{post_flag}|{date_time}|{post_number}|>\n{post_text}")
            # Check if all prior posts fit into max_length
            if sum(len(p) for p in prior_posts) > max_length:
                pass
            else:
                prompt = "".join(prior_posts)
                completion = posts[i]["com"] if "com" in posts[i] else ""
                # Check if completion contains a link (http, https, www) and skip if so
                if re.search(r"(http|https|www)", completion):
                    pass
                elif len(completion) == 61: # This is a common length for 4chan posts that are just a quote of another post (ex: >>123456789)
                    pass
                # Check if completion is a quote ">>\d{1,9}" and doesn't contain any other text by using [2:] 
                elif re.match(r">>\d{1,9}", completion) and len(completion[2:]) < 10:
                    pass
                elif len(completion) < min_completion_length:
                    pass
                elif prompt == "":
                    pass
                elif completion == "":
                    pass
                else:
                    dataset.append({"prompt": prompt, "completion": completion})
            

    dataset = [{ "prompt": d["prompt"], "completion": clean_text(d["completion"])} 
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