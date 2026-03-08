# -*- coding: utf-8 -*-
"""Transformer encoder producing (z, sigma) for NWF charges."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class _TransformerHead(nn.Module):
    """Head: hidden -> (z, log_sigma)."""

    def __init__(self, in_features: int, latent_dim: int) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fc_mu(x), self.fc_logvar(x)


class TransformerEncoder:
    """HuggingFace transformer with (z, sigma) head for NWF."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        latent_dim: int = 64,
        pooling: str = "cls",
        freeze_backbone: bool = True,
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.latent_dim = latent_dim
        self.pooling = pooling
        self.freeze_backbone = freeze_backbone
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self._backbone.config.hidden_size
        self._head = _TransformerHead(hidden_size, latent_dim)

        self._backbone = self._backbone.to(self.device)
        self._head = self._head.to(self.device)
        if freeze_backbone:
            for p in self._backbone.parameters():
                p.requires_grad = False

    def _tokenize(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
    ) -> dict:
        if isinstance(texts, str):
            texts = [texts]
        return self._tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _get_pooled(self, outputs: "torch.Tensor", attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            return last_hidden[:, 0, :]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (last_hidden * mask).sum(1) / (mask.sum(1) + 1e-10)
        raise ValueError("pooling must be 'cls' or 'mean'")

    def fit(
        self,
        train_texts: List[str],
        train_labels: Optional[np.ndarray] = None,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> "TransformerEncoder":
        """Train head on texts. Unsupervised: maximize Gaussian prior likelihood."""
        trainable = list(self._head.parameters())
        if not self.freeze_backbone:
            trainable += list(self._backbone.parameters())
        opt = torch.optim.Adam(trainable, lr=lr)
        self._backbone.train()
        self._head.train()

        n = len(train_texts)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size].tolist()
                batch_texts = [train_texts[j] for j in idx]
                enc = self._tokenize(batch_texts)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    out = self._backbone(**enc)
                pooled = self._get_pooled(out, enc["attention_mask"])
                mu, logvar = self._head(pooled)
                loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._backbone.eval()
        self._head.eval()
        return self

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode texts to (z, sigma). Returns numpy arrays."""
        self._backbone.eval()
        self._head.eval()
        if isinstance(texts, str):
            texts = [texts]
        z_list, s_list = [], []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self._tokenize(batch)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self._backbone(**enc)
                pooled = self._get_pooled(out, enc["attention_mask"])
                mu, logvar = self._head(pooled)
                sigma = torch.exp(0.5 * logvar)
                z_list.append(mu.cpu().numpy())
                s_list.append(sigma.cpu().numpy())
        z = np.vstack(z_list)
        s = np.vstack(s_list)
        if len(texts) == 1:
            return z[0], s[0]
        return z, s
