from unittest.mock import MagicMock

import numpy as np

from humanize_rl.benchmark.evaluate_scorer import evaluate_scorer


def test_evaluate_scorer():
    # Mock a scorer
    mock_scorer = MagicMock()
    mock_scorer.predict_binary.return_value = np.array([0.1, 0.9, 0.2, 0.8])
    mock_scorer.predict_rubric.return_value = np.random.rand(4, 8)

    # Dummy validation data
    val_data = {
        "texts": ["Text 1", "Text 2", "Text 3", "Text 4"],
        "labels": np.array([0, 1, 0, 1]),
        "rubrics": np.random.rand(4, 8),
    }

    # Run evaluation
    metrics = evaluate_scorer(mock_scorer, val_data, run_mlflow=False)

    # Assert metrics structure
    assert "binary_auroc" in metrics
    assert "rubric_mean_mse" in metrics
    assert "latency_ms" in metrics

    assert 0.0 <= metrics["binary_auroc"] <= 1.0
    assert metrics["rubric_mean_mse"] >= 0.0
    assert metrics["latency_ms"] >= 0.0
