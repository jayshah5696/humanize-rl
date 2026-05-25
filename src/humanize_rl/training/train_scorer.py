import argparse
import os
import pickle

import mlflow

from humanize_rl.benchmark.evaluate_scorer import evaluate_scorer
from humanize_rl.data.scorer_dataset import load_scorer_data, scorer_data_summary
from humanize_rl.scoring.distilled.baselines import FastTextScorer, RidgeScorer
from humanize_rl.scoring.distilled.dense import DenseScorer


def main():
    parser = argparse.ArgumentParser(
        description="Train Distilled Layer 2 Scorer Candidate"
    )
    parser.add_argument(
        "--model-type",
        choices=[
            "baseline_ridge",
            "baseline_fasttext",
            "dense_nomic",
            "dense_gemma",
            "dense_luxical",
            "dense_tiny",
        ],
        required=True,
        help="Type of scorer model to train",
    )
    parser.add_argument(
        "--local-data-path",
        default="data/benchmark/scored_combined_v01.jsonl",
        help="Path to local scored JSONL data",
    )
    parser.add_argument(
        "--hf-dataset-name",
        default="gsingh1-py/train",
        help="Name of Hugging Face dataset for binary labels",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Epochs for dense model training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for dense model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--model-dir",
        default="models/distilled",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--experiment-name",
        default="Scorer-Calibration-TrackA",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--split-strategy",
        choices=[
            "random",
            "source_holdout",
            "domain_holdout",
            "format_holdout",
            "group_holdout",
        ],
        default="random",
        help="Validation split strategy",
    )
    parser.add_argument(
        "--load-all-local",
        action="store_true",
        help="Legacy mode: load broad local files except test/diagnostic/OOD files",
    )
    parser.add_argument(
        "--balance-format",
        action="store_true",
        help="Downsample so AI/human labels are balanced within format buckets",
    )

    args = parser.parse_args()

    # Set up MLflow
    mlflow.set_experiment(args.experiment_name)

    run_name = f"{args.model_type}"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Starting MLflow Run: {run_name} (ID: {run.info.run_id})")

        # 1. Log training hyperparameters
        mlflow.log_params(
            {
                "model_type": args.model_type,
                "local_data_path": args.local_data_path,
                "hf_dataset_name": args.hf_dataset_name,
                "epochs": args.epochs if "dense" in args.model_type else None,
                "learning_rate": args.lr if "dense" in args.model_type else None,
                "batch_size": args.batch_size if "dense" in args.model_type else None,
                "split_strategy": args.split_strategy,
                "load_all_local": args.load_all_local,
                "balance_format": args.balance_format,
            }
        )

        # 2. Load dataset
        print("Loading training and validation data...")
        train_data, val_data = load_scorer_data(
            local_path=args.local_data_path,
            hf_dataset_name=args.hf_dataset_name,
            val_split=0.2,
            load_all_local=args.load_all_local,
            split_strategy=args.split_strategy,
            balance_format=args.balance_format,
        )
        train_summary = scorer_data_summary(train_data)
        val_summary = scorer_data_summary(val_data)
        print(
            f"Loaded {train_summary['samples']} train and {val_summary['samples']} validation samples."
        )
        print(
            f"Rubric labels: {train_summary['rubric_samples']} train / {val_summary['rubric_samples']} validation."
        )
        for key, value in train_summary.items():
            mlflow.log_metric(f"train_{key}", value)
        for key, value in val_summary.items():
            mlflow.log_metric(f"val_{key}", value)

        # 3. Initialize model
        if args.model_type == "baseline_ridge":
            model = RidgeScorer()
        elif args.model_type == "baseline_fasttext":
            model = FastTextScorer(model_dir=args.model_dir)
        elif args.model_type == "dense_nomic":
            model = DenseScorer(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
            )
        elif args.model_type == "dense_gemma":
            model = DenseScorer(
                model_name="google/embeddinggemma-300m",  # 300M Gemma embedding model
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
            )
        elif args.model_type == "dense_luxical":
            model = DenseScorer(
                model_name="DatologyAI/luxical-one",
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                device="cpu",
            )
        elif args.model_type == "dense_tiny":
            model = DenseScorer(
                model_name="all-MiniLM-L6-v2",  # Tiny model for local tests/quick verification
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

        # 4. Train/Fit model
        print(f"Fitting model {args.model_type}...")
        model.fit(train_data["texts"], train_data["labels"], train_data["rubrics"])
        print("Model fitting complete.")

        # 5. Save model checkpoint
        os.makedirs(args.model_dir, exist_ok=True)
        model_save_path = os.path.join(args.model_dir, f"{args.model_type}.pkl")
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved local model to {model_save_path}")

        # Log model file directly to MLflow
        mlflow.log_artifact(model_save_path)

        # 6. Evaluate and log metrics/plots
        print("Running validation evaluation...")
        metrics = evaluate_scorer(model, val_data, run_mlflow=True)
        print("Evaluation results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print(
            "MLflow Run Complete. Visit the MLflow UI via 'just ui' to inspect details."
        )


if __name__ == "__main__":
    main()
