import os
import tempfile
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve

from humanize_rl.data.scorer_dataset import RUBRIC_DIMS


def evaluate_scorer(model, val_data: dict, run_mlflow: bool = True) -> dict:
    """Evaluate a distilled scorer candidate against validation data.

    Computes AUROC, MSE, R^2, and latency. Generates scatter and ROC plots.
    """
    texts = val_data["texts"]
    true_binary = val_data["labels"]
    true_rubric = val_data["rubrics"]

    # 1. Measure Latency (single-item inference over up to 100 samples)
    latency_samples = min(len(texts), 100)
    latencies = []
    for i in range(latency_samples):
        start = time.perf_counter()
        _ = model.predict_binary([texts[i]])
        _ = model.predict_rubric([texts[i]])
        latencies.append(time.perf_counter() - start)
    mean_latency_ms = float(np.mean(latencies) * 1000)

    # 2. Get predictions for entire validation set
    pred_binary = model.predict_binary(texts)
    pred_rubric = model.predict_rubric(texts)

    # 3. Compute Binary metrics
    try:
        binary_auroc = float(roc_auc_score(true_binary, pred_binary))
    except ValueError:
        # Fails if only 1 class is present in split
        binary_auroc = 0.5

    # 4. Compute Rubric metrics (ignoring -1.0 sentinel values)
    valid_mask = np.all(true_rubric != -1.0, axis=1)

    rubric_mses = {}
    rubric_r2s = {}

    if np.any(valid_mask):
        val_true_rub = true_rubric[valid_mask]
        val_pred_rub = pred_rubric[valid_mask]

        for idx, dim in enumerate(RUBRIC_DIMS):
            mse = float(mean_squared_error(val_true_rub[:, idx], val_pred_rub[:, idx]))
            try:
                r2 = float(r2_score(val_true_rub[:, idx], val_pred_rub[:, idx]))
            except Exception:
                r2 = 0.0
            rubric_mses[dim] = mse
            rubric_r2s[dim] = r2

        mean_rubric_mse = float(np.mean(list(rubric_mses.values())))
        mean_rubric_r2 = float(np.mean(list(rubric_r2s.values())))
    else:
        mean_rubric_mse = 0.0
        mean_rubric_r2 = 0.0

    metrics = {
        "binary_auroc": binary_auroc,
        "rubric_mean_mse": mean_rubric_mse,
        "rubric_mean_r2": mean_rubric_r2,
        "latency_ms": mean_latency_ms,
    }

    # 5. Generate and Log Charts
    with tempfile.TemporaryDirectory() as tmpdir:
        # ROC Curve Plot
        if len(np.unique(true_binary)) > 1:
            fpr, tpr, _ = roc_curve(true_binary, pred_binary)
            plt.figure(figsize=(6, 5))
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (area = {binary_auroc:.3f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC)")
            plt.legend(loc="lower right")
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            plt.savefig(roc_path)
            plt.close()

            if run_mlflow:
                mlflow.log_artifact(roc_path)

        # Rubric Scatter Plots (Predicted vs Actual)
        if np.any(valid_mask):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            val_true_rub = true_rubric[valid_mask]
            val_pred_rub = pred_rubric[valid_mask]

            for idx, dim in enumerate(RUBRIC_DIMS):
                ax = axes[idx]
                sns.scatterplot(
                    x=val_true_rub[:, idx],
                    y=val_pred_rub[:, idx],
                    ax=ax,
                    alpha=0.6,
                    color="teal",
                )
                ax.plot([0, 1], [0, 1], color="red", linestyle="--")
                ax.set_xlim([-0.05, 1.05])
                ax.set_ylim([-0.05, 1.05])
                ax.set_title(f"{dim}\n(MSE: {rubric_mses[dim]:.3f})")
                ax.set_xlabel("Gemini 3.1 Pro (Actual)")
                ax.set_ylabel("Distilled Scorer (Pred)")

            plt.tight_layout()
            scatter_path = os.path.join(tmpdir, "rubric_predictions_scatter.png")
            plt.savefig(scatter_path)
            plt.close()

            if run_mlflow:
                mlflow.log_artifact(scatter_path)

    # 6. Log parameters/metrics to MLflow
    if run_mlflow:
        mlflow.log_metrics(metrics)
        for dim in RUBRIC_DIMS:
            if dim in rubric_mses:
                mlflow.log_metric(f"rubric_mse_{dim}", rubric_mses[dim])
                mlflow.log_metric(f"rubric_r2_{dim}", rubric_r2s[dim])

    return metrics
