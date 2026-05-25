import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split

RUBRIC_DIMS = [
    "structural_symmetry",
    "specificity",
    "formality_gradient",
    "voice_consistency",
    "rhetorical_sophistication",
    "padding_density",
    "personality_presence",
    "copula_avoidance",
]

AI_MODEL_COLUMNS = [
    "gemma-2-9b",
    "mistral-7B",
    "qwen-2-72B",
    "llama-8B",
    "accounts/yi-01-ai/models/yi-large",
    "GPT_4-o",
]

DEFAULT_TRAIN_LOCAL_PATHS = [
    "data/benchmark/v03_combined_matched.jsonl",
    "data/benchmark/expansion_matched.jsonl",
    "data/benchmark/scored_combined_v01.jsonl",
]

EXCLUDED_LOCAL_NAME_PARTS = (
    "test_set",
    "diagnostic",
    "ood",
    "scored_output",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _binary_label(raw_label: Any) -> int | None:
    if raw_label in (1, "ai", "ai_generated"):
        return 1
    if raw_label in (0, "human", "humanized", "human_authored", "humanized_synthetic"):
        return 0
    return None


def _label_type(raw_label: Any, *, is_seed: bool = False) -> str:
    if raw_label in (1, "ai", "ai_generated"):
        return "ai_generated"
    if raw_label == "humanized":
        return "humanized_synthetic"
    if raw_label in (0, "human", "human_authored") or is_seed:
        return "human_authored"
    return "unknown"


def format_bucket(text: str) -> str:
    lines = text.splitlines()
    has_heading = any(line.lstrip().startswith("#") for line in lines)
    has_bullet = any(line.lstrip().startswith(("- ", "* ", "1. ")) for line in lines)
    has_byline = "By [" in text or "Published:" in text
    if has_byline:
        return "byline_template"
    if has_heading:
        return "markdown_heading"
    if has_bullet:
        return "list_or_bullets"
    return "plain"


def _rubric_vector(record: dict[str, Any]) -> np.ndarray:
    rubric = np.full(8, -1.0, dtype=np.float32)
    l2_per_dim = record.get("l2_per_dim")
    if isinstance(l2_per_dim, dict):
        for idx, dim in enumerate(RUBRIC_DIMS):
            rubric[idx] = float(l2_per_dim.get(dim, 0.5))
        return rubric

    per_dim = record.get("per_dim")
    if isinstance(per_dim, dict):
        layer1_dims = {
            "structural_symmetry": "list_overuse",
            "specificity": "opener_pattern",
            "formality_gradient": "hedging_density",
            "voice_consistency": "closing_pattern",
            "rhetorical_sophistication": "transition_overuse",
            "padding_density": "transition_overuse",
            "personality_presence": "contractions",
            "copula_avoidance": "copula_avoidance",
        }
        if "copula_avoidance" in per_dim:
            for idx, dim in enumerate(RUBRIC_DIMS):
                rubric[idx] = float(
                    per_dim.get(dim, per_dim.get(layer1_dims[dim], 0.5))
                )
    return rubric


def _text_from_record(record: dict[str, Any], path: Path) -> str | None:
    if "seeds" in path.parts:
        value = record.get("response")
    else:
        value = (
            record.get("text") or record.get("text_preview") or record.get("response")
        )
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _iter_local_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    is_seed = "seeds" in path.parts
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = _text_from_record(record, path)
            if text is None:
                continue
            raw_label = record.get("label", "human" if is_seed else None)
            label = _binary_label(raw_label)
            if label is None:
                continue
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "label_type": _label_type(raw_label, is_seed=is_seed),
                    "rubric": _rubric_vector(record),
                    "source": str(path),
                    "domain": record.get("domain", "unknown"),
                    "group": record.get("id") or f"{path}:{line_no}",
                    "format_bucket": format_bucket(text),
                }
            )
    return rows


def _load_local_paths(paths: list[str] | None) -> list[dict[str, Any]]:
    root = _repo_root()
    rows = []
    for raw_path in paths or DEFAULT_TRAIN_LOCAL_PATHS:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            rows.extend(_iter_local_rows(path))
    return rows


def _load_legacy_all_local(local_path: str) -> list[dict[str, Any]]:
    root = _repo_root()
    paths = sorted((root / "data/benchmark").rglob("*.jsonl")) + sorted(
        (root / "seeds").rglob("*.jsonl")
    )
    safe_paths = [
        path
        for path in paths
        if not any(part in path.name for part in EXCLUDED_LOCAL_NAME_PARTS)
    ]
    primary = Path(local_path)
    if not primary.is_absolute():
        primary = root / primary
    if primary.exists() and primary not in safe_paths:
        safe_paths.append(primary)
    rows = []
    for path in safe_paths:
        rows.extend(_iter_local_rows(path))
    return rows


def _load_hf_rows(hf_dataset_name: str | None) -> list[dict[str, Any]]:
    if not hf_dataset_name:
        return []
    rows = []
    dataset = load_dataset(hf_dataset_name, split="train")
    for idx, item in enumerate(dataset):
        human_text = item.get("Human_story")
        if isinstance(human_text, str) and human_text.strip():
            text = human_text.strip()
            rows.append(
                {
                    "text": text,
                    "label": 0,
                    "label_type": "human_authored",
                    "rubric": np.full(8, -1.0, dtype=np.float32),
                    "source": f"{hf_dataset_name}:Human_story",
                    "domain": "nyt_story",
                    "group": f"{hf_dataset_name}:{idx}",
                    "format_bucket": format_bucket(text),
                }
            )
        for column in AI_MODEL_COLUMNS:
            ai_text = item.get(column)
            if isinstance(ai_text, str) and ai_text.strip():
                text = ai_text.strip()
                rows.append(
                    {
                        "text": text,
                        "label": 1,
                        "label_type": "ai_generated",
                        "rubric": np.full(8, -1.0, dtype=np.float32),
                        "source": f"{hf_dataset_name}:{column}",
                        "domain": "nyt_story",
                        "group": f"{hf_dataset_name}:{idx}",
                        "format_bucket": format_bucket(text),
                    }
                )
    return rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        text = row["text"]
        if text in seen:
            continue
        seen.add(text)
        deduped.append(row)
    return deduped


def _balance_format_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["format_bucket"], row["label"])].append(row)

    balanced = []
    formats = {row["format_bucket"] for row in rows}
    for bucket in sorted(formats):
        human_rows = buckets[(bucket, 0)]
        ai_rows = buckets[(bucket, 1)]
        if not human_rows or not ai_rows:
            continue
        keep = min(len(human_rows), len(ai_rows))
        for group_rows in (human_rows, ai_rows):
            indices = rng.choice(len(group_rows), size=keep, replace=False)
            balanced.extend(group_rows[int(idx)] for idx in indices)
    return balanced or rows


def _split_indices(
    rows: list[dict[str, Any]], split_strategy: str, val_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([row["label"] for row in rows])
    indices = np.arange(len(rows))
    if split_strategy == "random":
        return train_test_split(
            indices,
            test_size=val_split,
            random_state=seed,
            stratify=labels if len(set(labels)) > 1 else None,
        )

    group_field = {
        "source_holdout": "source",
        "domain_holdout": "domain",
        "format_holdout": "format_bucket",
        "group_holdout": "group",
    }.get(split_strategy)
    if group_field is None:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    groups = np.array([row[group_field] for row in rows])
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    return next(splitter.split(indices, labels, groups))


def _pack(rows: list[dict[str, Any]], indices: np.ndarray) -> dict[str, Any]:
    selected = [rows[int(idx)] for idx in indices]
    return {
        "texts": [row["text"] for row in selected],
        "labels": np.array([row["label"] for row in selected], dtype=np.int64),
        "rubrics": np.vstack([row["rubric"] for row in selected]),
        "sources": [row["source"] for row in selected],
        "domains": [row["domain"] for row in selected],
        "label_types": [row["label_type"] for row in selected],
        "format_buckets": [row["format_bucket"] for row in selected],
        "groups": [row["group"] for row in selected],
    }


def scorer_data_summary(data: dict[str, Any]) -> dict[str, int]:
    rubrics = data["rubrics"]
    rubric_valid = np.all(rubrics != -1.0, axis=1)
    return {
        "samples": len(data["texts"]),
        "ai_samples": int(np.sum(data["labels"] == 1)),
        "humanish_samples": int(np.sum(data["labels"] == 0)),
        "rubric_samples": int(np.sum(rubric_valid)),
    }


def load_scorer_data(
    local_path: str = "data/benchmark/scored_combined_v01.jsonl",
    hf_dataset_name: str | None = "gsingh1-py/train",
    val_split: float = 0.2,
    seed: int = 42,
    load_all_local: bool = False,
    local_paths: list[str] | None = None,
    split_strategy: str = "random",
    balance_format: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load scorer calibration rows with explicit metadata and safer defaults."""
    if load_all_local:
        rows = _load_legacy_all_local(local_path)
    else:
        rows = _load_local_paths(
            local_paths or [local_path, *DEFAULT_TRAIN_LOCAL_PATHS]
        )

    try:
        rows.extend(_load_hf_rows(hf_dataset_name))
    except Exception as exc:
        print(f"HF dataset {hf_dataset_name} load skipped: {exc}")

    rows = _dedupe_rows(rows)
    if balance_format:
        rows = _balance_format_rows(rows, seed)

    if not rows:
        rows = [
            {
                "text": "Dummy human text",
                "label": 0,
                "label_type": "human_authored",
                "rubric": np.full(8, -1.0, dtype=np.float32),
                "source": "dummy",
                "domain": "dummy",
                "group": "dummy-human",
                "format_bucket": "plain",
            },
            {
                "text": "Dummy AI text",
                "label": 1,
                "label_type": "ai_generated",
                "rubric": np.full(8, -1.0, dtype=np.float32),
                "source": "dummy",
                "domain": "dummy",
                "group": "dummy-ai",
                "format_bucket": "plain",
            },
        ]

    train_idx, val_idx = _split_indices(rows, split_strategy, val_split, seed)
    return _pack(rows, train_idx), _pack(rows, val_idx)
