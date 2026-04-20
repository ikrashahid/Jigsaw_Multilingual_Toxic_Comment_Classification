import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert-base-uncased"
RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_split_data(
    csv_path: str,
    train_size: int = 100_000,
    eval_size: int = 20_000,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, usecols=["comment_text", "toxic", "black", "white", "muslim", "jewish", "lgbtq"])
    df["label"] = (df["toxic"] >= 0.5).astype(int)
    sampled_total = train_size + eval_size
    sampled, _ = train_test_split(
        df,
        train_size=sampled_total,
        stratify=df["label"],
        random_state=random_state,
    )
    train_df, eval_df = train_test_split(
        sampled,
        test_size=eval_size,
        stratify=sampled["label"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


class CommentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def tokenize_df(df: pd.DataFrame, tokenizer, max_length: int = 128) -> CommentDataset:
    encodings = tokenizer(
        df["comment_text"].fillna("").tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return CommentDataset(encodings, df["label"].tolist())


def train_distilbert(train_df: pd.DataFrame, eval_df: pd.DataFrame, output_dir: str = "saved_model/baseline"):
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_dataset = tokenize_df(train_df, tokenizer)
    eval_dataset = tokenize_df(eval_df, tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
        seed=RANDOM_STATE,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, tokenizer


def predict_proba(trainer: Trainer, dataset: CommentDataset) -> np.ndarray:
    predictions = trainer.predict(dataset).predictions
    probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()[:, 1]
    return probs


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({"threshold": t, "f1_macro": f1_score(y_true, y_pred, average="macro")})
    return pd.DataFrame(rows)


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    axes[1].plot(recall, precision)
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    plt.tight_layout()
    plt.show()


def cohort_masks(eval_df: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "high_black": eval_df["black"] >= 0.5,
        "reference": (eval_df["black"] < 0.1) & (eval_df["white"] >= 0.5),
    }


def safe_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def cohort_metrics(eval_df: pd.DataFrame, y_prob: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    y_pred = (y_prob >= threshold).astype(int)
    results = []
    masks = cohort_masks(eval_df)

    for name, mask in masks.items():
        y_true_c = eval_df.loc[mask, "label"].values
        y_pred_c = y_pred[mask.values]
        if len(y_true_c) == 0:
            continue
        tn, fp, fn, tp = safe_confusion(y_true_c, y_pred_c).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        results.append(
            {
                "cohort": name,
                "size": len(y_true_c),
                "TPR": tpr,
                "FPR": fpr,
                "FNR": fnr,
                "precision": precision,
            }
        )
    out = pd.DataFrame(results)
    if {"high_black", "reference"}.issubset(set(out["cohort"].tolist())):
        hb = out.loc[out["cohort"] == "high_black", "FPR"].iloc[0]
        ref = out.loc[out["cohort"] == "reference", "FPR"].iloc[0]
        out["disparate_impact_ratio_fpr"] = hb / ref if ref > 0 else np.nan
    return out


def perturb(text: str) -> str:
    zws = "\u200b"
    homoglyph_map = str.maketrans({"a": "а", "e": "е", "o": "о"})  # Cyrillic lookalikes
    words = text.split()
    out_words = []
    for w in words:
        w2 = w.translate(homoglyph_map)
        # Insert zero-width spaces every 2-3 characters.
        chunks = []
        i = 0
        while i < len(w2):
            step = 2 if (i // 2) % 2 == 0 else 3
            chunks.append(w2[i : i + step])
            i += step
        w3 = zws.join(chunks)
        # Duplicate ~20% characters.
        chars = []
        for ch in w3:
            chars.append(ch)
            if random.random() < 0.2 and ch.isalpha():
                chars.append(ch)
        out_words.append("".join(chars))
    return " ".join(out_words)


def fit_probability_calibrator(y_prob: np.ndarray, y_true: np.ndarray) -> CalibratedClassifierCV:
    base = LogisticRegression(max_iter=2000)
    x = y_prob.reshape(-1, 1)
    base.fit(x, y_true)
    calibrator = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calibrator.fit(x, y_true)
    return calibrator


@dataclass
class ThresholdBandResult:
    low: float
    high: float
    auto_fraction: float
    review_fraction: float


def evaluate_bands(calibrated_probs: np.ndarray, low: float, high: float) -> ThresholdBandResult:
    review_mask = (calibrated_probs > low) & (calibrated_probs < high)
    review_fraction = review_mask.mean()
    return ThresholdBandResult(low=low, high=high, auto_fraction=1 - review_fraction, review_fraction=review_fraction)


def plot_grouped_rates(metrics_df: pd.DataFrame) -> None:
    plot_df = metrics_df.melt(id_vars=["cohort"], value_vars=["TPR", "FPR", "FNR"], var_name="metric", value_name="value")
    sns.barplot(data=plot_df, x="metric", y="value", hue="cohort")
    plt.title("Cohort Rate Comparison")
    plt.ylim(0, 1)
    plt.show()
