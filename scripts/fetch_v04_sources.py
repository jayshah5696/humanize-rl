import json
import os
import argparse
from datasets import load_dataset

def fetch_stream_b(limit_per_dataset=None):
    print("Starting Stream B sources fetch...")
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/v04_real_world_sources.jsonl"
    
    collected = []
    
    # 1. wardacoder/business-email-dataset (email)
    try:
        limit = limit_per_dataset or 2000
        print(f"Fetching wardacoder/business-email-dataset (limit={limit})...")
        ds = load_dataset("wardacoder/business-email-dataset", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= limit:
                break
            body = row.get("output", "")
            if body and len(body.strip()) > 50:
                collected.append({
                    "text": body.strip(),
                    "domain": "email",
                    "source_dataset": "wardacoder/business-email-dataset"
                })
                count += 1
        print(f"Loaded {count} email samples.")
    except Exception as e:
        print(f"Error fetching email dataset: {e}")
        
    # 2. liamdugan/raid (mixed domain, filter model=='human')
    try:
        limit = limit_per_dataset or 2000
        print(f"Fetching liamdugan/raid (limit={limit})...")
        ds = load_dataset("liamdugan/raid", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= limit:
                break
            if row.get("model") == "human":
                text = row.get("generation", "")
                if text and len(text.strip()) > 50:
                    collected.append({
                        "text": text.strip(),
                        "domain": row.get("domain", "general"),
                        "source_dataset": "liamdugan/raid"
                    })
                    count += 1
        print(f"Loaded {count} raid human samples.")
    except Exception as e:
        print(f"Error fetching raid dataset: {e}")
        
    # 3. nikcane/slack (chat)
    try:
        limit = limit_per_dataset or 500
        print(f"Fetching nikcane/slack (limit={limit})...")
        ds = load_dataset("nikcane/slack", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= limit:
                break
            text = row.get("text", "")
            # Ignore automated bot notifications or empty messages
            if text and len(text.strip()) > 15 and not text.startswith("<http"):
                collected.append({
                    "text": text.strip(),
                    "domain": "chat",
                    "source_dataset": "nikcane/slack"
                })
                count += 1
        print(f"Loaded {count} slack samples.")
    except Exception as e:
        print(f"Error fetching slack dataset: {e}")

    # 4. euclaise/writingprompts (creative)
    try:
        limit = limit_per_dataset or 1000
        print(f"Fetching euclaise/writingprompts (limit={limit})...")
        ds = load_dataset("euclaise/writingprompts", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= limit:
                break
            story = row.get("story", "")
            if story and len(story.strip()) > 100:
                collected.append({
                    "text": story.strip(),
                    "domain": "creative",
                    "source_dataset": "euclaise/writingprompts"
                })
                count += 1
        print(f"Loaded {count} writingprompts samples.")
    except Exception as e:
        print(f"Error fetching writingprompts: {e}")

    # 5. HuggingFaceFW/fineweb-edu (informative)
    try:
        limit = limit_per_dataset or 1500
        print(f"Fetching HuggingFaceFW/fineweb-edu (limit={limit})...")
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= limit:
                break
            score = row.get("score", 0)
            if score >= 3.5:
                text = row.get("text", "")
                if text and len(text.strip()) > 100:
                    collected.append({
                        "text": text.strip(),
                        "domain": "informative",
                        "source_dataset": "HuggingFaceFW/fineweb-edu"
                    })
                    count += 1
        print(f"Loaded {count} fineweb-edu samples.")
    except Exception as e:
        print(f"Error fetching fineweb-edu: {e}")

    # 6. common-pile/peS2o (academic/technical)
    try:
        limit = limit_per_dataset or 1000
        print(f"Fetching common-pile/peS2o (limit={limit})...")
        ds = load_dataset("common-pile/peS2o", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= limit:
                break
            text = row.get("text", "")
            if text and len(text.strip()) > 100:
                collected.append({
                    "text": text.strip(),
                    "domain": "technical",
                    "source_dataset": "common-pile/peS2o"
                })
                count += 1
        print(f"Loaded {count} peS2o samples.")
    except Exception as e:
        print(f"Error fetching peS2o: {e}")

    # Save to file
    with open(output_path, "w") as f:
        for item in collected:
            f.write(json.dumps(item) + "\n")
            
    print(f"Wrote {len(collected)} total normalized source rows to {output_path}")

def fetch_stream_d(limit_per_dataset=None):
    print("Starting Stream D sources fetch...")
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/v04_ocr_sources.jsonl"
    
    collected = []
    
    # pixparse/pdfa-eng-wds (document OCR text)
    try:
        limit = limit_per_dataset or 500
        print(f"Fetching pixparse/pdfa-eng-wds (limit={limit})...")
        ds = load_dataset("pixparse/pdfa-eng-wds", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= limit:
                break
            pages = row.get("json", {}).get("pages", [])
            if pages:
                lines_dict = pages[0].get("lines", {})
                if lines_dict and "text" in lines_dict:
                    lines_text = lines_dict["text"]
                    if lines_text:
                        ocr_text = "\n".join(lines_text).strip()
                        if len(ocr_text) > 80:
                            # Construct instruction schema
                            instruction = (
                                "The screenshot below is a formal document. "
                                "Rewrite the key content in plain, natural language:\n\n"
                                f"[Screenshot content:]\n{ocr_text}\n\nHumanize this."
                            )
                            image_url = row.get("__url__", "")
                            collected.append({
                                "instruction": instruction,
                                "image_url": image_url,
                                "domain": "document",
                                "task_type": "ocr_document_text",
                                "stream": "D"
                            })
                            count += 1
        print(f"Loaded {count} pdfa-eng-wds OCR samples.")
    except Exception as e:
        print(f"Error fetching pdfa-eng-wds: {e}")
        
    with open(output_path, "w") as f:
        for item in collected:
            f.write(json.dumps(item) + "\n")
            
    print(f"Wrote {len(collected)} total normalized OCR rows to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Download only a small pilot subset (50 samples per dataset)")
    args = parser.parse_args()
    
    limit = 50 if args.pilot else None
    
    fetch_stream_b(limit_per_dataset=limit)
    fetch_stream_d(limit_per_dataset=limit)
