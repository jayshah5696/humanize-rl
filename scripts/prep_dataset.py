import os
import json
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from humanize_rl.data.filters import LangDetectFilter, TokenPriorPerplexityFilter, SpaCyActionFilter

def prep_dataset():
    # Ensure processed data directory exists
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("runs/v03", exist_ok=True)

    mlflow.set_experiment("SFT_Data_Preparation")
    
    with mlflow.start_run(run_name="dataset_preprocessing_funnel"):
        print("Initializing NLP Filters...")
        lang_filter = LangDetectFilter()
        perplexity_filter = TokenPriorPerplexityFilter()
        action_filter = SpaCyActionFilter()

        # Config parameters
        target_sample_size = 2000  # Target number of raw prompts to process from each stream for the vertical slice
        mlflow.log_param("target_sample_size", target_sample_size)
        mlflow.log_param("perp_max_perplexity", perplexity_filter.max_perplexity)
        mlflow.log_param("perp_min_ttr", perplexity_filter.min_ttr)

        streams = {
            "lmsys": {"path": "lmsys/lmsys-chat-1m", "split": "train"},
            "alpaca": {"path": "tatsu-lab/alpaca", "split": "train"}
        }

        funnel_stats = {}

        for stream_name, config in streams.items():
            print(f"\nProcessing stream: {stream_name}...")
            
            # Load streaming dataset to avoid downloading gigabytes
            dataset = load_dataset(config["path"], split=config["split"], streaming=True)
            
            raw_count = 0
            lang_count = 0
            perp_count = 0
            action_count = 0
            
            cleaned_prompts = []
            
            # Stream a subset of data
            for row in tqdm(dataset, total=target_sample_size, desc=f"Filtering {stream_name}"):
                if raw_count >= target_sample_size:
                    break
                    
                raw_count += 1
                
                # Extract prompt text
                if stream_name == "lmsys":
                    # lmsys is conversational turns. Extract the first user message.
                    # Structure: row['conversation'] is list of dicts with role, text/content
                    conv = row.get("conversation", [])
                    if not conv or len(conv) == 0:
                        continue
                    prompt = conv[0].get("text", "") or conv[0].get("content", "")
                else:
                    # Alpaca is standard instruction/input
                    instruction = row.get("instruction", "")
                    input_text = row.get("input", "")
                    prompt = f"{instruction}\n{input_text}".strip() if input_text else instruction
                
                if not prompt:
                    continue
                
                # Stage 1: Lang Detect
                if not lang_filter.filter(prompt):
                    continue
                lang_count += 1
                
                # Stage 2: Token Prior Perplexity
                if not perplexity_filter.filter(prompt):
                    continue
                perp_count += 1
                
                # Stage 3: spaCy POS Dependency Parser
                if not action_filter.filter(prompt):
                    continue
                action_count += 1
                
                cleaned_prompts.append({
                    "instruction": prompt,
                    "source": stream_name
                })
            
            funnel_stats[stream_name] = {
                "1_raw": raw_count,
                "2_lang_detect": lang_count,
                "3_perplexity": perp_count,
                "4_action_filter": action_count
            }
            
            # Log metrics to MLflow
            mlflow.log_metric(f"{stream_name}_raw", raw_count)
            mlflow.log_metric(f"{stream_name}_post_lang", lang_count)
            mlflow.log_metric(f"{stream_name}_post_perp", perp_count)
            mlflow.log_metric(f"{stream_name}_post_action", action_count)
            
            survival_rate = action_count / max(1, raw_count)
            mlflow.log_metric(f"{stream_name}_survival_rate", survival_rate)
            
            # Save local stream file
            output_file = f"data/processed/cleaned_{stream_name}_tasks.jsonl"
            with open(output_file, "w") as f:
                for p in cleaned_prompts:
                    f.write(json.dumps(p) + "\n")
            print(f"Saved {len(cleaned_prompts)} prompts to {output_file} (Survival rate: {survival_rate:.2%})")

        # Create Funnel Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for idx, (stream_name, stats) in enumerate(funnel_stats.items()):
            stages = list(stats.keys())
            counts = list(stats.values())
            
            axes[idx].barh(stages[::-1], counts[::-1], color=['#4caf50', '#ffeb3b', '#2196f3', '#f44336'])
            axes[idx].set_title(f"Preprocessing Funnel: {stream_name.upper()}")
            axes[idx].set_xlabel("Number of Prompts")
            for i, v in enumerate(counts[::-1]):
                axes[idx].text(v + 10, i, str(v), va='center')
        
        plt.tight_layout()
        funnel_plot_path = "runs/v03/funnel_dropoff.png"
        plt.savefig(funnel_plot_path)
        plt.close()
        
        # Log artifacts to MLflow
        mlflow.log_artifact(funnel_plot_path)
        print("Logged funnel dropoff chart to MLflow.")

if __name__ == "__main__":
    prep_dataset()
