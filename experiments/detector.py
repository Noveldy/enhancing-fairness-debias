"""Sensitive-attribute detection (revision Phase 6).

The app (bias_identification/datasets.py:dataset_inspection) has two methods:

(a) ``bert``: when column descriptions are given, a BERT classifier labels each
    description "Sensitive" / "Not Sensitive"
    (Models/sensitive_features_text_classifier.py). Training data:
    bias_identification/sensitive_features_dataset.csv (825 descriptions).
    Original training: bert-base-uncased, encoder frozen except the pooler, AdamW
    lr 2e-4 (weight decay 0, gradient-norm clip 1.0), batch 8, 10 epochs, linear decay, the epoch with the lowest loss on the
    evaluation split kept (Trainer load_best_model_at_end) -- and that evaluation
    split was also the reported test split (AUDIT section 6).
(b) ``keyword``: without descriptions, a column is protected when its lower-cased
    name equals, or contains as a substring, one of 16 legal protected-class words.

Here (a) is retrained with the same model and hyperparameters, but the epoch is
chosen on a validation split that is separate from every test split. The loop is
plain PyTorch instead of the HF Trainer (same optimizer, schedule, batch size,
dropout in train mode, dynamic padding), so `datasets`/`evaluate`/`accelerate`
are not needed.
"""
import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

# Copied from bias_identification/datasets.py:29-31 (importing that module loads the DeBERTa
# zero-shot pipeline at import time). Kept verbatim, including the typo 'ancesty'.
PROTECTED_CLASSES = ['race', 'color', 'religion', 'sex', 'gender', 'gender identity', 'gender expression',
                     'sexual orientation', 'marital status', 'medical condition', 'military status', 'nationality',
                     'ancesty', 'disability', 'age', 'leave request']
LABELS = {"Not Sensitive": 0, "Sensitive": 1}
BASE_MODEL = "google-bert/bert-base-uncased"
# The Hugging Face snapshot used for every reported result (Phase 9: pinned so a fresh download is identical).
BASE_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def keyword_detect(name):
    """The original rule (datasets.py:76): exact match or substring of a protected-class word."""
    s = str(name).lower()
    return int(s in PROTECTED_CLASSES or any(item in s for item in PROTECTED_CLASSES))


def load_training_data(path):
    """Training descriptions with exact duplicates (after lower-casing, trimming and a final '.')
    removed; no duplicate has conflicting labels (checked)."""
    d = pd.read_csv(path)
    key = d["text"].str.lower().str.strip().str.rstrip(".")
    if (d.assign(k=key).groupby("k")["label"].nunique() > 1).any():
        raise ValueError("duplicate descriptions with conflicting labels")
    out = d[~key.duplicated()].reset_index(drop=True)
    out["y"] = out["label"].map(LABELS)
    return out, int(key.duplicated().sum())


def paraphrase_groups(texts, distance_threshold=0.7):
    """Groups of near-paraphrases (templated descriptions such as 'Percentage of Black residents' /
    'Ratio of Black individuals'): average-linkage clustering of TF-IDF vectors, cosine distance.
    Average linkage, because single linkage chains 42% of the texts into one group."""
    X = TfidfVectorizer(stop_words="english").fit_transform(pd.Series(texts).str.lower().str.strip()).toarray()
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=distance_threshold).fit_predict(X)


class BertDetector:
    def __init__(self, seed, lr=2e-4, batch_size=8, epochs=10, threads=8, model_path=BASE_MODEL):
        self.seed, self.lr, self.batch_size, self.epochs = seed, lr, batch_size, epochs
        self.threads, self.model_path = threads, model_path

    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(self.threads)
        rev = {"revision": BASE_REVISION} if self.model_path == BASE_MODEL else {}
        self.tok_ = AutoTokenizer.from_pretrained(self.model_path, **rev)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=2, id2label={0: "Not Sensitive", 1: "Sensitive"}, label2id=LABELS, **rev)
        return model

    def _batches(self, texts, labels, shuffle, gen):
        idx = torch.randperm(len(texts), generator=gen).tolist() if shuffle else list(range(len(texts)))
        for i in range(0, len(idx), self.batch_size):
            b = idx[i:i + self.batch_size]
            enc = self.tok_([texts[j] for j in b], truncation=True, padding=True, return_tensors="pt")
            yield enc, (torch.tensor([labels[j] for j in b]) if labels is not None else None)

    def fit(self, texts, y, val_texts, val_y):
        torch.manual_seed(self.seed)
        model = self._load()
        for name, p in model.base_model.named_parameters():   # as the original: only the pooler trains
            p.requires_grad = "pooler" in name
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.0)   # Trainer defaults
        steps = self.epochs * int(np.ceil(len(texts) / self.batch_size))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
        gen = torch.Generator().manual_seed(self.seed)
        texts, val_texts = list(texts), list(val_texts)
        self.history_, best = [], (np.inf, None, -1)
        t0 = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            model.train()
            tr_loss = []
            for enc, lab in self._batches(texts, list(y), True, gen):
                out = model(**enc, labels=lab)
                opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)              # Trainer default max_grad_norm
                opt.step()
                sched.step()
                tr_loss.append(out.loss.item())
            val_loss = self._loss(model, val_texts, list(val_y))
            self.history_.append({"epoch": epoch, "train_loss": float(np.mean(tr_loss)), "val_loss": val_loss})
            if val_loss < best[0]:        # lowest validation loss, earliest on ties (Trainer default metric)
                best = (val_loss, copy.deepcopy({k: v for k, v in model.state_dict().items()
                                                 if "pooler" in k or "classifier" in k}), epoch)
        model.load_state_dict(best[1], strict=False)
        model.eval()
        self.model_, self.best_epoch_, self.fit_seconds_ = model, best[2], time.perf_counter() - t0
        return self

    @torch.no_grad()
    def _loss(self, model, texts, y):
        model.eval()
        losses = []
        for enc, lab in self._batches(texts, y, False, None):
            losses.append(model(**enc, labels=lab).loss.item() * len(lab))
        return float(np.sum(losses) / len(texts))

    @torch.no_grad()
    def predict_proba(self, texts):
        out = []
        for enc, _ in self._batches(list(texts), None, False, None):
            out.append(torch.softmax(self.model_(**enc).logits, dim=-1)[:, 1].numpy())
        return np.concatenate(out)

    def head_state(self):
        return {k: v for k, v in self.model_.state_dict().items() if "pooler" in k or "classifier" in k}


class LegacyCheckpoint:
    """The local checkpoint-830 (not the paper's checkpoint-610; AUDIT section 8), for reference."""

    def __init__(self, path, threads=8):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(threads)
        self.tok_ = AutoTokenizer.from_pretrained(path)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(path).eval()

    @torch.no_grad()
    def predict_proba(self, texts):
        enc = self.tok_(list(texts), truncation=True, padding=True, return_tensors="pt")
        return torch.softmax(self.model_(**enc).logits, dim=-1)[:, 1].numpy()
