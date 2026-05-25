from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset

from humanize_rl.scoring.distilled.base import BaseDistilledScorer


class DenseRegressionClassificationModel(nn.Module):
    """Two-head PyTorch model for binary classification + 8-dim regression."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.binary_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.rubric_head = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.Sigmoid(),  # Rubrics are strictly within [0, 1] bounds
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_rep = self.shared(x)
        binary_out = self.binary_head(shared_rep).squeeze(-1)
        rubric_out = self.rubric_head(shared_rep)
        return binary_out, rubric_out


class DenseScorer(BaseDistilledScorer):
    """Dense Transformer (Nomic / Gemma) embedding + PyTorch two-head classifier/regressor."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        batch_size: int = 32,
        lr: float = 1e-3,
        epochs: int = 20,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.transformer = None
        self.model = None
        self.is_fitted = False

    def _get_transformer(self) -> SentenceTransformer:
        if self.transformer is None:
            # Load with trust_remote_code=True for Nomic
            try:
                self.transformer = SentenceTransformer(
                    self.model_name, trust_remote_code=True, device=self.device
                )
            except Exception as e:
                if self.device == "mps":
                    print(
                        f"Warning: Failed to load SentenceTransformer on mps: {e}. Falling back to cpu..."
                    )
                    self.device = "cpu"
                    self.transformer = SentenceTransformer(
                        self.model_name, trust_remote_code=True, device="cpu"
                    )
                else:
                    raise e
        return self.transformer

    def _encode(self, texts: list[str]) -> np.ndarray:
        transformer = self._get_transformer()
        prefixed_texts = texts
        if "nomic" in self.model_name:
            prefixed_texts = [f"search_query: {t}" for t in texts]

        try:
            return transformer.encode(
                prefixed_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as e:
            if self.device == "mps":
                print(f"Warning: Encoding failed on mps: {e}. Falling back to cpu...")
                self.device = "cpu"
                # Re-initialize transformer on CPU
                self.transformer = SentenceTransformer(
                    self.model_name, trust_remote_code=True, device="cpu"
                )
                # Re-map/move the PyTorch head model to CPU if initialized
                if self.model is not None:
                    self.model = self.model.to("cpu")

                # Retry encoding
                prefixed_texts = texts
                if "nomic" in self.model_name:
                    prefixed_texts = [f"search_query: {t}" for t in texts]
                return self.transformer.encode(
                    prefixed_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            else:
                raise e

    def fit(
        self, texts: list[str], binary_labels: np.ndarray, rubric_scores: np.ndarray
    ) -> DenseScorer:
        # Encode text inputs to dense embeddings using fallback-safe _encode
        embeddings = self._encode(texts)

        # 2. Prepare PyTorch dataloader
        X = torch.tensor(embeddings, dtype=torch.float32)
        y_binary = torch.tensor(binary_labels, dtype=torch.float32)
        y_rubric = torch.tensor(rubric_scores, dtype=torch.float32)

        dataset = TensorDataset(X, y_binary, y_rubric)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 3. Initialize PyTorch model & optimizer
        input_dim = embeddings.shape[1]
        self.model = DenseRegressionClassificationModel(input_dim=input_dim).to(
            self.device
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )

        bce_loss_fn = nn.BCELoss()
        mse_loss_fn = nn.MSELoss(
            reduction="none"
        )  # manual masking for -1.0 sentinel values

        # 4. Training loop
        self.model.train()
        for _epoch in range(self.epochs):
            for batch_x, batch_y_bin, batch_y_rub in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y_bin = batch_y_bin.to(self.device)
                batch_y_rub = batch_y_rub.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                bin_preds, rub_preds = self.model(batch_x)

                # Binary classification loss
                bin_loss = bce_loss_fn(bin_preds, batch_y_bin)

                # Rubric regression loss with mask (ignore -1.0 sentinel values)
                rubric_mask = batch_y_rub != -1.0
                if rubric_mask.any():
                    # Compute raw MSE, then average over elements that are valid
                    raw_mse = mse_loss_fn(rub_preds, batch_y_rub)
                    rubric_loss = raw_mse[rubric_mask].mean()
                else:
                    rubric_loss = torch.tensor(0.0, device=self.device)

                # Combined Multi-Task Loss
                total_loss = bin_loss + 2.0 * rubric_loss

                total_loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict_binary(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted yet.")

        embeddings = self._encode(texts)

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            bin_preds, _ = self.model(X)
            return bin_preds.cpu().numpy()

    def predict_rubric(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted yet.")

        embeddings = self._encode(texts)

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            _, rub_preds = self.model(X)
            return rub_preds.cpu().numpy()
