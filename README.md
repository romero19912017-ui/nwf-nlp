# nwf-nlp

[![PyPI version](https://badge.fury.io/py/nwf-nlp.svg)](https://pypi.org/project/nwf-nlp/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## NWF for Natural Language Processing

`nwf-nlp` provides Transformer-based text encoders that produce **semantic charges** `(z, sigma)` for incremental learning, semantic search, and classification. Built on HuggingFace models (BERT, RoBERTa, DistilBERT) with a custom head for uncertainty estimation. Uses `nwf-core` Charge (supports `alpha` for weighted superposition since 0.3+).

### Features

- **TransformerEncoder** — wrapper over HuggingFace transformers (DistilBERT, BERT, RoBERTa)
- **Output (z, sigma)** — compatible with `nwf-core` Field and Mahalanobis search
- **Incremental learning** — add new categories without retraining the full model
- **Semantic search** — find documents by query using charge similarity
- **Batch encoding** — GPU support, configurable batch size
- **Pooling** — [CLS] or mean pooling

---

## Installation

```bash
pip install nwf-core nwf-nlp
```

Requires: `nwf-core>=0.2.3`, `torch`, `transformers`, `scikit-learn`.

---

## Quick Start

```python
from nwf import Charge, Field
from nwf.nlp import TransformerEncoder

enc = TransformerEncoder("distilbert-base-uncased", latent_dim=64)
enc.fit(train_texts, epochs=3)
z, sigma = enc.encode("Some text")
charge = Charge(z=z, sigma=sigma)

field = Field()
for i, text in enumerate(texts):
    z, s = enc.encode(text)
    field.add(Charge(z=z, sigma=s), labels=[labels[i]], ids=[i])
```

---

## API

### TransformerEncoder

| Parameter | Description |
|-----------|-------------|
| `model_name` | HuggingFace model: "distilbert-base-uncased", "bert-base-uncased", "roberta-base" |
| `latent_dim` | Output dimension of z |
| `pooling` | "cls" or "mean" |
| `freeze_backbone` | If True, only train the head (faster) |
| `max_length` | Max tokens (default 512) |

| Method | Description |
|--------|-------------|
| `fit(texts, epochs, batch_size, lr)` | Train head on texts (unsupervised: Gaussian prior) |
| `encode(texts, batch_size)` | Returns `(z, sigma)` as numpy arrays |

---

## Examples

Install with examples: `pip install nwf-nlp[examples]`

| Script | Description |
|--------|-------------|
| [20newsgroups.py](examples/20newsgroups.py) | Incremental text classification: 3 categories, add sci.med without retraining |

Run:
```bash
python examples/20newsgroups.py --epochs 2 --k 5
python examples/20newsgroups.py --save results/nlp.png
```

Notebook: `notebooks/20newsgroups.ipynb`

---

## Application areas (сферы применения)

| Area | Use case | Components |
|------|----------|------------|
| **Incremental text classification** | Add new categories without retraining | TransformerEncoder, Field, k-NN |
| **Semantic search** | Find documents by query in charge space | encode(query), Field.search |
| **Topic modeling** | Cluster documents by latent charges | z, sigma from TransformerEncoder |
| **Weighted charges** | Charge.alpha for category importance (nwf-core 0.3+) | Charge(z, sigma, alpha=...) |

---

## License

MIT

---

# nwf-nlp (Русский)

## NWF для обработки естественного языка

`nwf-nlp` предоставляет Transformer-энкодеры для текста с выходом **семантических зарядов** `(z, sigma)` для инкрементального обучения, семантического поиска и классификации.

### Компоненты

- **TransformerEncoder** — обёртка над HuggingFace (DistilBERT, BERT, RoBERTa)
- **Выход (z, sigma)** — совместим с Field и поиском по Махаланобису
- **Инкрементальность** — добавление новых тем без переобучения
- **Семантический поиск** — поиск документов по запросу

### Установка

```bash
pip install nwf-core nwf-nlp
```

### Пример

```python
from nwf.nlp import TransformerEncoder
from nwf import Charge, Field

enc = TransformerEncoder("distilbert-base-uncased", latent_dim=64)
enc.fit(тексты, epochs=3)
z, sigma = enc.encode("Текст для кодирования")
field.add(Charge(z=z, sigma=sigma), labels=[метка])
```

### Лицензия

MIT
