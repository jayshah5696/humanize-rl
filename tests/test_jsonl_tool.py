from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from scripts.jsonl_tool import cli


@pytest.fixture
def temp_jsonl_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "data.jsonl"
    data = [
        {
            "id": "row_1",
            "instruction": "email formatting",
            "response": "Hi",
            "split": "train",
        },
        {
            "id": "row_2",
            "instruction": "technical explanation",
            "response": "Hello",
            "split": "train",
        },
        {
            "id": "row_3",
            "instruction": "creative writing",
            "response": "Once upon",
            "split": "train",
        },
        {
            "id": "row_4",
            "instruction": "email formatting",
            "response": "Hi",
            "split": "validation",
        },
        {
            "id": "row_5",
            "instruction": "compression task",
            "response": "Short",
            "split": "test",
        },
    ]
    file_path.write_text("\n".join(json.dumps(r) for r in data) + "\n")
    return file_path


def test_count(temp_jsonl_file: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["count", str(temp_jsonl_file)])
    assert result.exit_code == 0
    assert result.output.strip() == "5"


def test_sample_random(temp_jsonl_file: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "sampled.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sample",
            "--input",
            str(temp_jsonl_file),
            "--output",
            str(out_file),
            "--n",
            "3",
            "--seed",
            "123",
        ],
    )
    assert result.exit_code == 0
    rows = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert len(rows) == 3


def test_sample_slice(temp_jsonl_file: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "sliced.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sample",
            "--input",
            str(temp_jsonl_file),
            "--output",
            str(out_file),
            "--n",
            "2",
            "--slice",
        ],
    )
    assert result.exit_code == 0
    rows = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["id"] == "row_1"
    assert rows[1]["id"] == "row_2"


def test_dedupe_single_key(temp_jsonl_file: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "deduped.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dedupe",
            "--input",
            str(temp_jsonl_file),
            "--output",
            str(out_file),
            "--key",
            "instruction",
        ],
    )
    assert result.exit_code == 0
    rows = [json.loads(line) for line in out_file.read_text().splitlines()]
    # row_1 and row_4 have the same instruction 'email formatting'
    assert len(rows) == 4
    instructions = [r["instruction"] for r in rows]
    assert instructions.count("email formatting") == 1


def test_dedupe_composite_key(temp_jsonl_file: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "deduped_composite.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dedupe",
            "--input",
            str(temp_jsonl_file),
            "--output",
            str(out_file),
            "--key",
            "instruction,response",
        ],
    )
    assert result.exit_code == 0
    rows = [json.loads(line) for line in out_file.read_text().splitlines()]
    # row_1 and row_4 both have ("email formatting", "Hi")
    assert len(rows) == 4


def test_merge(temp_jsonl_file: Path, tmp_path: Path) -> None:
    file2 = tmp_path / "data2.jsonl"
    file2.write_text(json.dumps({"id": "row_6", "instruction": "extra"}) + "\n")

    out_file = tmp_path / "merged.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "merge",
            "--output",
            str(out_file),
            str(temp_jsonl_file),
            str(file2),
        ],
    )
    assert result.exit_code == 0
    rows = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert len(rows) == 6
    assert rows[-1]["id"] == "row_6"


def test_split(temp_jsonl_file: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "splits"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "split",
            "--input",
            str(temp_jsonl_file),
            "--out-dir",
            str(out_dir),
            "--train",
            "0.6",
            "--valid",
            "0.2",
            "--test",
            "0.2",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0
    train_file = out_dir / "train.jsonl"
    valid_file = out_dir / "validation.jsonl"
    test_file = out_dir / "test.jsonl"

    assert train_file.exists()
    assert valid_file.exists()
    assert test_file.exists()

    train_len = len(train_file.read_text().splitlines())
    valid_len = len(valid_file.read_text().splitlines())
    test_len = len(test_file.read_text().splitlines())
    assert train_len + valid_len + test_len == 5


def test_duplicate(temp_jsonl_file: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "duplicated.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "duplicate",
            "--input",
            str(temp_jsonl_file),
            "--output",
            str(out_file),
            "--copies",
            "3",
        ],
    )
    assert result.exit_code == 0
    rows = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert len(rows) == 15
    assert rows[0]["id"] == "row_1__c0"
    assert rows[1]["id"] == "row_1__c1"
    assert rows[2]["id"] == "row_1__c2"
    assert rows[3]["id"] == "row_2__c0"
