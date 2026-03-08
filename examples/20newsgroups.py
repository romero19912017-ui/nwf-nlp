# -*- coding: utf-8 -*-
"""20 Newsgroups: incremental text classification with NWF TransformerEncoder.

Train on 3 categories, add sci.med without retraining. k-NN in charge space.
Run: python 20newsgroups.py [--epochs 2] [--k 5] [--save results/nlp.png]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from nwf import Charge, Field
from nwf.nlp import TransformerEncoder

if "--save" in sys.argv or os.environ.get("MPLBACKEND"):
    import matplotlib
    matplotlib.use("Agg")


def main() -> None:
    parser = argparse.ArgumentParser(description="20 Newsgroups incremental with NWF")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--max-val", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading 20 Newsgroups (subset)...")
    cats = ["sci.space", "comp.graphics", "rec.sport.hockey"]
    data = fetch_20newsgroups(
        subset="train",
        categories=cats,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=args.seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=args.seed, stratify=data.target
    )
    n_train = min(args.max_train, len(X_train))
    n_val = min(args.max_val, len(X_val))
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_val = X_val[:n_val]
    y_val = y_val[:n_val]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Classes: {len(cats)}")

    print("Training TransformerEncoder...")
    enc = TransformerEncoder(
        model_name="distilbert-base-uncased",
        latent_dim=args.latent_dim,
        pooling="cls",
        freeze_backbone=True,
        max_length=args.max_length,
    )
    enc.fit(X_train, epochs=args.epochs, batch_size=16, lr=1e-3)

    print("Encoding and building Field...")
    field = Field()
    z_all, s_all = enc.encode(X_train, batch_size=32)
    for i in range(len(z_all)):
        sigma = np.maximum(s_all[i], 1e-6)
        field.add(Charge(z=z_all[i], sigma=sigma), labels=[int(y_train[i])], ids=[i])

    print("Classification: k-NN in charge space...")
    z_val_enc, s_val_enc = enc.encode(X_val, batch_size=32)
    preds = []
    for i in range(len(z_val_enc)):
        sigma = np.maximum(s_val_enc[i], 1e-6)
        q = Charge(z=z_val_enc[i], sigma=sigma)
        _, _, labs = field.search(q, k=args.k)
        votes = np.bincount(np.array(labs[0]).astype(int), minlength=len(cats))
        preds.append(int(np.argmax(votes)))
    acc_before = accuracy_score(y_val, preds)
    print(f"Accuracy before adding sci.med (k={args.k}): {acc_before:.3f}")

    print("Adding new category sci.med without retraining...")
    data2 = fetch_20newsgroups(
        subset="train",
        categories=["sci.med"],
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=args.seed + 1,
    )
    z_new, s_new = enc.encode(data2.data[:100], batch_size=32)
    new_label = len(cats)
    base_id = len(field)
    for i in range(len(z_new)):
        sigma = np.maximum(s_new[i], 1e-6)
        field.add(
            Charge(z=z_new[i], sigma=sigma),
            labels=[new_label],
            ids=[base_id + i],
        )
    print(f"Field size after adding sci.med: {len(field)}")

    data_val4 = fetch_20newsgroups(
        subset="train",
        categories=cats + ["sci.med"],
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=args.seed + 2,
    )
    X_val4, y_val4 = data_val4.data[:50], data_val4.target[:50]
    z_v4, s_v4 = enc.encode(X_val4, batch_size=32)
    preds4 = []
    for i in range(len(z_v4)):
        sigma = np.maximum(s_v4[i], 1e-6)
        q = Charge(z=z_v4[i], sigma=sigma)
        _, _, labs = field.search(q, k=args.k)
        votes = np.bincount(np.array(labs[0]).astype(int), minlength=4)
        preds4.append(int(np.argmax(votes)))
    acc_after = accuracy_score(y_val4, preds4)
    print(f"Accuracy after adding sci.med (4 classes): {acc_after:.3f}")

    if args.save:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["3 classes", "4 classes (+sci.med)"], [acc_before, acc_after], color=["C0", "C1"])
        ax.set_ylabel("Accuracy")
        ax.set_title("20 Newsgroups: incremental category addition")
        ax.set_ylim(0, 1)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {args.save}")
    print("Done.")


if __name__ == "__main__":
    main()
