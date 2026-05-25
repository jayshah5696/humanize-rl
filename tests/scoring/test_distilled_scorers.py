import numpy as np

from humanize_rl.scoring.distilled.baselines import FastTextScorer, RidgeScorer


def test_ridge_scorer_interface():
    scorer = RidgeScorer()

    # Dummy data
    texts = [
        "This is a human written text example.",
        "Certainly! I would be happy to assist you with this task.",
        "Another casual email update.",
        "Moreover, it is worth noting that we must consider this.",
    ]
    binary_labels = np.array([0, 1, 0, 1])
    rubric_scores = np.random.rand(4, 8)

    # Fit
    scorer.fit(texts, binary_labels, rubric_scores)

    # Predict binary
    probs = scorer.predict_binary(texts)
    assert probs.shape == (4,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Predict rubric
    preds = scorer.predict_rubric(texts)
    assert preds.shape == (4, 8)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_fasttext_scorer_interface(tmp_path):
    scorer = FastTextScorer(model_dir=str(tmp_path))

    # Dummy data
    texts = [
        "This is a human written text example.",
        "Certainly! I would be happy to assist you with this task.",
        "Another casual email update.",
        "Moreover, it is worth noting that we must consider this.",
    ]
    binary_labels = np.array([0, 1, 0, 1])
    rubric_scores = np.random.rand(4, 8)

    # Fit
    scorer.fit(texts, binary_labels, rubric_scores)

    # Predict binary
    probs = scorer.predict_binary(texts)
    assert probs.shape == (4,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Predict rubric
    preds = scorer.predict_rubric(texts)
    assert preds.shape == (4, 8)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_dense_scorer_interface():
    # Use a tiny fast embedding model for testing
    from humanize_rl.scoring.distilled.dense import DenseScorer

    scorer = DenseScorer(model_name="all-MiniLM-L6-v2", epochs=2)

    # Dummy data
    texts = [
        "This is a human written text example.",
        "Certainly! I would be happy to assist you with this task.",
        "Another casual email update.",
        "Moreover, it is worth noting that we must consider this.",
    ]
    binary_labels = np.array([0, 1, 0, 1])
    rubric_scores = np.random.rand(4, 8)

    # Fit
    scorer.fit(texts, binary_labels, rubric_scores)

    # Predict binary
    probs = scorer.predict_binary(texts)
    assert probs.shape == (4,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Predict rubric
    preds = scorer.predict_rubric(texts)
    assert preds.shape == (4, 8)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
