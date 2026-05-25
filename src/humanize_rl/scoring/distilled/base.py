from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDistilledScorer(ABC):
    """Abstract Base Class for all Distilled Layer 2 Scorers."""

    @abstractmethod
    def fit(
        self, texts: list[str], binary_labels: np.ndarray, rubric_scores: np.ndarray
    ) -> BaseDistilledScorer:
        """Fit the scorer on text inputs, binary labels, and 8-dim continuous rubric scores.

        Args:
            texts: List of input strings.
            binary_labels: Binary array where 1 = AI, 0 = Human.
            rubric_scores: Matrix of shape (N, 8) representing continuous Layer 2 scores in [0, 1].

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict_binary(self, texts: list[str]) -> np.ndarray:
        """Predict the probability that each text is AI-generated.

        Args:
            texts: List of input strings.

        Returns:
            1D array of shape (N,) containing predicted probability in [0, 1].
        """
        pass

    @abstractmethod
    def predict_rubric(self, texts: list[str]) -> np.ndarray:
        """Predict the 8 continuous Layer 2 rubric scores for each text.

        Args:
            texts: List of input strings.

        Returns:
            2D array of shape (N, 8) containing predicted rubric scores in [0, 1].
        """
        pass
