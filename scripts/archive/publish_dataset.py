import argparse
import json
import os

import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import HfApi, get_token


def combine_and_format_datasets():
    parser = argparse.ArgumentParser(description="Merge SFT streams, plot domain distribution, and publish to HF.")
    parser.add_argument("--dry-run", action="store_true", help="Perform merge and plotting without pushing to Hugging Face.")
    parser.add_argument("--repo-id", type=str, default="jayshah5696/humanize-rl-sft-dataset", help="Target Hugging Face Hub dataset repo ID.")
    args = parser.parse_args()

    sft_rows = []
    domain_counts = {
        "email": 0,
        "instruction_technical": 0,
        "blog_opinion": 0,
        "academic": 0,
        "creative": 0
    }

    # Domain instructions template fallback mapping for text-rewriting tasks
    domain_instructions = {
        "email": "Rewrite the following email to sound more natural, professional, and human-written. Keep the core meaning and facts exactly the same:\n\n{text}",
        "instruction_technical": "Rewrite the following technical explanation or note to sound natural, concise, and human-written. Keep the core meaning and facts exactly the same:\n\n{text}",
        "blog_opinion": "Rewrite the following blog post or opinion piece to sound natural, engaging, and human-written. Keep the core meaning and facts exactly the same:\n\n{text}",
        "academic": "Rewrite the following academic passage to sound natural, clear, and human-written. Keep the core meaning and facts exactly the same:\n\n{text}",
        "creative": "Rewrite the following creative writing piece to sound natural, vivid, and human-written. Keep the core meaning and facts exactly the same:\n\n{text}",
    }
    default_instruction = "Rewrite the following text to sound natural, fluent, and human-written. Keep the core meaning and facts exactly the same:\n\n{text}"

    # 1. Load and reformat Legacy Gold Pairs (sft_pairs_v01.jsonl)
    legacy_file = "data/processed/sft_pairs_v01.jsonl"
    if os.path.exists(legacy_file):
        print(f"Loading legacy gold pairs from {legacy_file}...")
        count = 0
        with open(legacy_file) as f:
            for line in f:
                row = json.loads(line)
                metadata = row.get("metadata", {})
                orig_inst = metadata.get("original_instruction", "")
                if not orig_inst:
                    orig_inst = row.get("instruction", "")
                if not orig_inst:
                    orig_inst = "Rewrite the following text to sound natural and human:\n\n" + row.get("input", "")
                humanized_resp = row.get("output", "")
                
                # Deduce domain or map it from original instruction keywords
                domain = "instruction_technical"
                inst_lower = orig_inst.lower()
                if "email" in inst_lower or "ood" in inst_lower or "out-of-office" in inst_lower:
                    domain = "email"
                elif "blog" in inst_lower or "opinion" in inst_lower or "meeting" in inst_lower or "macro" in inst_lower:
                    domain = "blog_opinion"
                elif "story" in inst_lower or "narrative" in inst_lower or "creative" in inst_lower:
                    domain = "creative"
                
                sft_rows.append({
                    "instruction": orig_inst,
                    "response": humanized_resp,
                    "messages": [
                        {"role": "user", "content": orig_inst},
                        {"role": "assistant", "content": humanized_resp}
                    ],
                    "domain": domain,
                    "source": "legacy_gold_v01"
                })
                domain_counts[domain] += 1
                count += 1
        print(f"Processed {count} legacy gold SFT pairs.")
    else:
        print(f"Warning: Legacy file {legacy_file} not found.")

    # 2. Load and process scaleup SFT dataset (v03_corpus_sft.jsonl)
    corpus_file = "data/processed/v03_corpus_sft.jsonl"
    if os.path.exists(corpus_file):
        print(f"Loading corpus SFT dataset from {corpus_file}...")
        count = 0
        with open(corpus_file) as f:
            for line in f:
                row = json.loads(line)
                domain = row.get("domain", "blog_opinion")
                # Normalize domain key
                if domain == "technical":
                    domain = "instruction_technical"
                
                instruction = row.get("instruction", row.get("input", ""))
                sft_rows.append({
                    "instruction": instruction,
                    "response": row["output"],
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": row["output"]}
                    ],
                    "domain": domain,
                    "source": "corpus_scaleup_v03"
                })
                if domain in domain_counts:
                    domain_counts[domain] += 1
                else:
                    domain_counts[domain] = 1
                count += 1
        print(f"Processed {count} corpus SFT rows.")
    else:
        print(f"Warning: Corpus file {corpus_file} not found.")

    # 3. Load and process expansion SFT dataset (expansion_sft.jsonl)
    expansion_file = "data/processed/expansion_sft.jsonl"
    if os.path.exists(expansion_file):
        print(f"Loading expansion SFT dataset from {expansion_file}...")
        count = 0
        with open(expansion_file) as f:
            for line in f:
                row = json.loads(line)
                domain = row.get("domain", "instruction_technical")
                if domain == "technical":
                    domain = "instruction_technical"
                
                instruction = row.get("instruction")
                if not instruction:
                    # Construct instruction from input using the fallback mapping
                    input_text = row.get("input", "")
                    domain_inst_tpl = domain_instructions.get(domain, default_instruction)
                    instruction = domain_inst_tpl.format(text=input_text)

                sft_rows.append({
                    "instruction": instruction,
                    "response": row["output"],
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": row["output"]}
                    ],
                    "domain": domain,
                    "source": "expansion_v03"
                })
                if domain in domain_counts:
                    domain_counts[domain] += 1
                else:
                    domain_counts[domain] = 1
                count += 1
        print(f"Processed {count} expansion SFT rows.")

    # 4. Load and process walking skeleton SFT (v03_walking_skeleton_sft.jsonl)
    ws_file = "data/processed/v03_walking_skeleton_sft.jsonl"
    if os.path.exists(ws_file):
        print(f"Loading walking skeleton SFT dataset from {ws_file}...")
        count = 0
        with open(ws_file) as f:
            for line in f:
                row = json.loads(line)
                domain = row.get("domain", "instruction_technical")
                if domain == "technical":
                    domain = "instruction_technical"
                
                instruction = row.get("instruction", row.get("input", ""))
                sft_rows.append({
                    "instruction": instruction,
                    "response": row["output"],
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": row["output"]}
                    ],
                    "domain": domain,
                    "source": "walking_skeleton_v03"
                })
                if domain in domain_counts:
                    domain_counts[domain] += 1
                count += 1
        print(f"Processed {count} walking skeleton SFT rows.")

    if not sft_rows:
        print("No SFT rows found to merge.")
        return

    # Write merged SFT dataset
    combined_output_path = "data/processed/v03_combined_sharegpt.jsonl"
    with open(combined_output_path, "w") as f:
        for row in sft_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(sft_rows)} unified SFT rows to {combined_output_path}")

    # Generate Domain Distribution Plot
    os.makedirs("runs/v03", exist_ok=True)
    labels = list(domain_counts.keys())
    sizes = list(domain_counts.values())
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, 
            textprops={'fontsize': 10, 'weight': 'bold', 'color': '#1e293b'})
    plt.title("Domain Distribution of Combined SFT Dataset", fontsize=14, weight='bold', color='#1e293b')
    plt.tight_layout()
    plot_path = "runs/v03/domain_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved domain distribution plot to {plot_path}")

    # HF Hub upload
    if args.dry_run:
        print("Dry-run specified. Skipping Hugging Face upload.")
        return

    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN") or get_token()
    if not hf_token:
        print("Warning: HUGGING_FACE_HUB_TOKEN env var not set and no cached HF token found. Skipping Hugging Face upload.")
        print("To upload, run 'huggingface-cli login' or set the HUGGING_FACE_HUB_TOKEN env var.")
        return

    print(f"Uploading SFT dataset to HF Hub: {args.repo_id}...")
    try:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True, token=hf_token)
        
        # Load using datasets library and push
        dataset = load_dataset("json", data_files=combined_output_path, split="train")
        dataset.push_to_hub(args.repo_id, token=hf_token)
        
        # Upload the distribution plot too!
        api.upload_file(
            path_or_fileobj=plot_path,
            path_in_repo="domain_distribution.png",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=hf_token
        )
        print(f"Successfully published SFT dataset and metadata plot to https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")

if __name__ == "__main__":
    combine_and_format_datasets()
