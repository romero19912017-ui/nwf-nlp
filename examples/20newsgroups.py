# -*- coding: utf-8 -*-
"""Example: 20 Newsgroups classification with NWF and TransformerEncoder.

Train on subset of categories, add new categories without full retraining.
Run: python 20newsgroups.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add src to path for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nwf import Charge, Field
from nwf.nlp import TransformerEncoder


def main() -> None:
    print("Loading 20 Newsgroups (subset)...")
    cats = ["sci.space", "comp.graphics", "rec.sport.hockey"]
    data = fetch_20newsgroups(
        subset="train",
        categories=cats,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Classes: {len(cats)}")

    print("Training TransformerEncoder (head only, 2 epochs)...")
    enc = TransformerEncoder(
        model_name="distilbert-base-uncased",
        latent_dim=32,
        pooling="cls",
        freeze_backbone=True,
        max_length=128,
    )
    enc.fit(X_train[:500], epochs=2, batch_size=16, lr=1e-3)

    print("Encoding and building Field...")
    field = Field()
    z_all, s_all = enc.encode(X_train[:500], batch_size=32)
    for i in range(len(z_all)):
        sigma = np.maximum(s_all[i], 1e-6)
        field.add(Charge(z=z_all[i], sigma=sigma), labels=[y_train[i]], ids=[i])

    print("Classification: k-NN in charge space...")
    z_val, s_val = enc.encode(X_val[:100], batch_size=32)
    preds = []
    for i in range(len(z_val)):
        sigma = np.maximum(s_val[i], 1e-6)
        q = Charge(z=z_val[i], sigma=sigma)
        _, idx, labs = field.search(q, k=5)
        votes = np.bincount(np.array(labs[0]), minlength=len(cats))
        preds.append(np.argmax(votes))
    acc = accuracy_score(y_val[:100], preds)
    print(f"Accuracy (k=5): {acc:.3f}")

    print("Adding new category without retraining...")
    data2 = fetch_20newsgroups(
        subset="train",
        categories=["sci.med"],
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=43,
    )
    z_new, s_new = enc.encode(data2.data[:100], batch_size=32)
    new_label = len(cats)
    for i in range(len(z_new)):
        sigma = np.maximum(s_new[i], 1e-6)
        field.add(
            Charge(z=z_new[i], sigma=sigma),
            labels=[new_label],
            ids=[len(field) + i],
        )
    print(f"Field size after adding sci.med: {len(field)}")
    print("Done.")


if __name__ == "__main__":
    main()
