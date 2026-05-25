from unittest.mock import MagicMock, patch

import numpy as np

from humanize_rl.data.scorer_dataset import (
    format_bucket,
    load_scorer_data,
    scorer_data_summary,
)


@patch("humanize_rl.data.scorer_dataset.load_dataset")
def test_load_scorer_data(mock_load_dataset, tmp_path):
    # Mock the HuggingFace dataset
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 2
    mock_dataset.__iter__.return_value = iter(
        [{"Human_story": "Human text", "gemma-2-9b": "AI text"}]
    )
    mock_load_dataset.return_value = {"train": mock_dataset}

    # Create dummy local scored combined jsonl file
    dummy_local_file = tmp_path / "scored_combined_v01.jsonl"
    with open(dummy_local_file, "w") as f:
        f.write(
            '{"label": "human", "text_preview": "some text", "combined_overall": 0.8}\n'
        )
        f.write(
            '{"label": "ai", "text_preview": "other text", "combined_overall": 0.2, "l2_per_dim": {"structural_symmetry": 0.0, "specificity": 0.5, "formality_gradient": 0.25, "voice_consistency": 0.0, "rhetorical_sophistication": 0.0, "padding_density": 0.0, "personality_presence": 0.0, "copula_avoidance": 0.5}}\n'
        )

    # Run the loader
    train_data, val_data = load_scorer_data(
        local_path=str(dummy_local_file),
        hf_dataset_name="dummy/dataset",
        val_split=0.5,
        load_all_local=False,
    )

    # Check structure
    assert "texts" in train_data
    assert "labels" in train_data
    assert "rubrics" in train_data
    assert "label_types" in train_data
    assert "format_buckets" in train_data

    assert len(train_data["texts"]) > 0
    assert len(val_data["texts"]) > 0

    # Assert shapes
    assert isinstance(train_data["labels"][0], (int, np.integer))
    assert train_data["rubrics"].shape[1] == 8


@patch("humanize_rl.data.scorer_dataset.load_dataset")
def test_load_scorer_data_all_local_integration(mock_load_dataset):
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 0
    mock_dataset.__iter__.return_value = iter([])
    mock_load_dataset.return_value = mock_dataset

    # Run the loader with all local data enabled
    train_data, val_data = load_scorer_data(
        local_path="data/benchmark/scored_combined_v01.jsonl",
        hf_dataset_name="dummy/dataset",
        val_split=0.2,
        load_all_local=True,
    )

    assert "texts" in train_data
    assert "labels" in train_data
    assert "rubrics" in train_data

    # Check that we loaded a substantial number of local samples (e.g. >1000)
    total_samples = len(train_data["texts"]) + len(val_data["texts"])
    assert total_samples > 1000
    assert not any("test_set" in source for source in train_data["sources"])
    assert not any("diagnostic" in source for source in train_data["sources"])

    # Assert that all rubrics have the correct dimensionality
    assert train_data["rubrics"].shape[1] == 8


def test_format_bucket_detects_markdown_without_dropping_it():
    assert format_bucket("# Title\nBody") == "markdown_heading"
    assert format_bucket("- one\n- two") == "list_or_bullets"
    assert format_bucket("By [Your Name]\nPublished: [Date]") == "byline_template"
    assert format_bucket("plain paragraph") == "plain"


def test_scorer_data_summary_counts_rubric_labels():
    data = {
        "texts": ["a", "b"],
        "labels": np.array([0, 1]),
        "rubrics": np.array([[0.5] * 8, [-1.0] * 8], dtype=np.float32),
    }
    summary = scorer_data_summary(data)
    assert summary["samples"] == 2
    assert summary["rubric_samples"] == 1
