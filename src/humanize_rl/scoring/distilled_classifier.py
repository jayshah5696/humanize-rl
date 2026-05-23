"""Distilled Layer 2 stylistic classifier based on DeBERTa-v3.

Provides local, low-latency, reference-free surrogate reward scores
for humanness during reinforcement learning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Graceful imports to prevent crash if torch/transformers are not in runtime dependencies yet
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    HAS_ML_DEPS = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    HAS_ML_DEPS = False


@dataclass(frozen=True)
class DistilledScoreResult:
    """Surrogate score result from the local distilled model."""

    overall: float
    ai_probability: float
    per_dim: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


if HAS_ML_DEPS:

    class StylisticDeBERTa(nn.Module):
        """DeBERTa-v3 with multi-task heads for stylistic alignment evaluation.

        Projects the context stylistic representation (CLS token) to:
        1. Binary classification: AI vs. Human probability.
        2. Rubric regression: 8 continuous rubric scores in [0, 1].
        """

        def __init__(
            self,
            model_name: str = "microsoft/deberta-v3-small",
            num_rubric_dims: int = 8,
            hidden_dropout_prob: float = 0.1,
        ) -> None:
            super().__init__()
            self.deberta = AutoModel.from_pretrained(model_name)
            hidden_size = self.deberta.config.hidden_size

            self.dropout = nn.Dropout(hidden_dropout_prob)

            # Head 1: Binary humanness classifier (AI vs Human)
            self.binary_classifier = nn.Linear(hidden_size, 2)

            # Head 2: Rubric regression (8 dimensions, predicted as continuous values in [0, 1])
            self.rubric_regressor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(hidden_size // 2, num_rubric_dims),
                nn.Sigmoid(),  # Restricts predictions strictly to [0, 1]
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.

            Returns:
                Tuple of (binary_logits, rubric_predictions).
            """
            outputs = self.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # Take CLS token representation (first token)
            cls_repr = outputs.last_hidden_state[:, 0, :]
            cls_repr = self.dropout(cls_repr)

            binary_logits = self.binary_classifier(cls_repr)
            rubric_preds = self.rubric_regressor(cls_repr)

            return binary_logits, rubric_preds
else:
    # Fallback placeholder to maintain type safety when importing
    class StylisticDeBERTa:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch and Hugging Face Transformers are required for StylisticDeBERTa. "
                "Run `uv add torch transformers` to install them."
            )


class DistilledScorer:
    """Wrapper to run local inference using the distilled DeBERTa-v3 model."""

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

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
    ) -> None:
        if not HAS_ML_DEPS:
            raise ImportError(
                "PyTorch and Hugging Face Transformers are required for DistilledScorer. "
                "Run `uv add torch transformers` to install."
            )

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = StylisticDeBERTa()

        # Load weights
        state_dict = torch.load(
            Path(model_path) / "pytorch_model.bin",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def score(self, text: str) -> DistilledScoreResult:
        """Score a single text sample locally.

        Args:
            text: The text to score.

        Returns:
            DistilledScoreResult containing overall humanness score and individual rubric dimensions.
        """
        import time

        start_time = time.perf_counter()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            logits, rubric_preds = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # Binary probability (class 1 is human, class 0 is AI)
            probs = torch.softmax(logits, dim=-1)
            ai_prob = probs[0, 0].item()
            human_prob = probs[0, 1].item()

            # Rubric predictions [0, 1]
            rubric_scores = rubric_preds[0].cpu().tolist()

        per_dim = {
            dim: float(score)
            for dim, score in zip(self.RUBRIC_DIMS, rubric_scores, strict=True)
        }

        # Combine: 40% binary humanness score + 60% average rubric score
        mean_rubric = sum(rubric_scores) / len(rubric_scores)
        overall = 0.4 * human_prob + 0.6 * mean_rubric

        latency = (time.perf_counter() - start_time) * 1000.0

        return DistilledScoreResult(
            overall=overall, ai_probability=ai_prob, per_dim=per_dim, latency_ms=latency
        )
