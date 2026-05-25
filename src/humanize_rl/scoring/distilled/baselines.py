from __future__ import annotations

import os
import tempfile

import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

from humanize_rl.scoring.distilled.base import BaseDistilledScorer

# NumPy 2.0+ compatibility monkeypatch for fasttext library which calls np.array(..., copy=False)
_original_array = np.array


def _patched_array(object, *args, **kwargs):
    if kwargs.get("copy") is False:
        try:
            return _original_array(object, *args, **kwargs)
        except ValueError:
            kwargs.pop("copy")
            return np.asarray(object, *args, **kwargs)
    return _original_array(object, *args, **kwargs)


np.array = _patched_array


class RidgeScorer(BaseDistilledScorer):
    """TF-IDF + Ridge Regression baseline scorer."""

    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_features=max_features)
        self.classifier = LogisticRegression(C=1.0, max_iter=1000)
        self.regressors = [Ridge(alpha=1.0) for _ in range(8)]
        self.is_fitted = False

    def fit(
        self, texts: list[str], binary_labels: np.ndarray, rubric_scores: np.ndarray
    ) -> RidgeScorer:
        # Fit vectorizer
        features = self.vectorizer.fit_transform(texts)

        # Fit binary classifier
        self.classifier.fit(features, binary_labels)

        # Fit 8 independent regressors, ignoring sentinel -1.0 rows
        valid_indices = np.all(rubric_scores != -1.0, axis=1)
        if np.any(valid_indices):
            valid_features = features[valid_indices]
            valid_rubrics = rubric_scores[valid_indices]
            for i in range(8):
                self.regressors[i].fit(valid_features, valid_rubrics[:, i])
        else:
            # Fallback if no valid rubric scores exist (e.g. testing with only HF data)
            dummy_target = np.zeros(features.shape[0])
            for i in range(8):
                self.regressors[i].fit(features, dummy_target)

        self.is_fitted = True
        return self

    def predict_binary(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        features = self.vectorizer.transform(texts)
        # return probability of class 1 (AI)
        return self.classifier.predict_proba(features)[:, 1]

    def predict_rubric(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        features = self.vectorizer.transform(texts)
        preds = np.zeros((len(texts), 8), dtype=np.float32)
        for i in range(8):
            preds[:, i] = self.regressors[i].predict(features)
        # Clip to [0, 1] bounds
        return np.clip(preds, 0.0, 1.0)


class FastTextScorer(BaseDistilledScorer):
    """FastText supervised classifier + Ridge regression on sentence vectors."""

    def __init__(self, model_dir: str = "models", dim: int = 100):
        self.model_dir = model_dir
        self.dim = dim
        self.clf_model = None
        self.regressors = [Ridge(alpha=1.0) for _ in range(8)]
        self.is_fitted = False
        os.makedirs(self.model_dir, exist_ok=True)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self.clf_model is not None:
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                temp_name = f.name
            try:
                self.clf_model.save_model(temp_name)
                with open(temp_name, "rb") as f:
                    state["clf_model_bytes"] = f.read()
            finally:
                if os.path.exists(temp_name):
                    os.remove(temp_name)
            state["clf_model"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if "clf_model_bytes" in state and state["clf_model_bytes"] is not None:
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                temp_name = f.name
                f.write(state["clf_model_bytes"])
            try:
                self.clf_model = fasttext.load_model(temp_name)
            finally:
                if os.path.exists(temp_name):
                    os.remove(temp_name)

    def fit(
        self, texts: list[str], binary_labels: np.ndarray, rubric_scores: np.ndarray
    ) -> FastTextScorer:
        # Create temp file to train fastText
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for text, lbl in zip(texts, binary_labels, strict=False):
                cleaned_text = text.replace("\n", " ")
                label_str = "__label__ai" if lbl == 1 else "__label__human"
                f.write(f"{label_str} {cleaned_text}\n")
            temp_path = f.name

        try:
            # Train supervised fastText model
            # Silence fastText training logs with thread=1 and low epoch for baselines speed
            self.clf_model = fasttext.train_supervised(
                input=temp_path, epoch=10, lr=0.5, dim=self.dim, wordNgrams=2, verbose=0
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Get sentence vectors for rubric training
        valid_indices = np.all(rubric_scores != -1.0, axis=1)
        if np.any(valid_indices):
            valid_texts = [
                texts[idx] for idx, valid in enumerate(valid_indices) if valid
            ]
            valid_rubrics = rubric_scores[valid_indices]

            # Extract sentence vectors
            vectors = np.array(
                [
                    self.clf_model.get_sentence_vector(t.replace("\n", " "))
                    for t in valid_texts
                ]
            )
            for i in range(8):
                self.regressors[i].fit(vectors, valid_rubrics[:, i])
        else:
            # Fallback
            vectors = np.array(
                [
                    self.clf_model.get_sentence_vector(t.replace("\n", " "))
                    for t in texts
                ]
            )
            dummy_target = np.zeros(len(texts))
            for i in range(8):
                self.regressors[i].fit(vectors, dummy_target)

        self.is_fitted = True
        return self

    def predict_binary(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted or self.clf_model is None:
            raise ValueError("Model is not fitted yet.")

        probs = []
        for text in texts:
            cleaned_text = text.replace("\n", " ")
            labels, predictions = self.clf_model.predict(cleaned_text, k=2)
            # Find probability of AI label
            ai_idx = labels.index("__label__ai") if "__label__ai" in labels else -1
            if ai_idx != -1:
                probs.append(predictions[ai_idx])
            else:
                probs.append(
                    1.0 - predictions[0]
                )  # fall back if only human label predicted
        return np.array(probs, dtype=np.float32)

    def predict_rubric(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted or self.clf_model is None:
            raise ValueError("Model is not fitted yet.")

        vectors = np.array(
            [self.clf_model.get_sentence_vector(t.replace("\n", " ")) for t in texts]
        )
        preds = np.zeros((len(texts), 8), dtype=np.float32)
        for i in range(8):
            preds[:, i] = self.regressors[i].predict(vectors)
        return np.clip(preds, 0.0, 1.0)
