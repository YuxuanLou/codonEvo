import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import torch
from scipy.stats import spearmanr
import math

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    probs = 1 / (1 + np.exp(
        -predictions))
    pred_labels = (probs >= 0.5).astype(int)

    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels,
                                pred_labels,
                                zero_division=0)
    recall = recall_score(labels, pred_labels,
                          zero_division=0)
    f1 = f1_score(labels, pred_labels,
                  zero_division=0)
    mcc = matthews_corrcoef(labels, pred_labels)
    auc = roc_auc_score(labels, probs)


    cm = confusion_matrix(labels, pred_labels)
    print(f"Confusion Matrix:\n{cm}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "auc": auc
    }

def compute_classification_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)


    return {
        "accuracy": accuracy
    }

def default_data_collator(features):
    batch = {}
    for key in features[0].keys():
        if isinstance(features[0][key],
                      torch.Tensor):
            batch[key] = torch.stack(
                [f[key] for f in features])
        else:
            batch[key] = torch.tensor(
                [f[key] for f in features])
    return batch


def compute_regression_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = predictions.flatten()


    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    spearman_corr, _ = spearmanr(labels,predictions)
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "spearmanr": spearman_corr
    }


class MLMDataCollator:
    def __init__(self, tokenizer,
                 mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, examples):
        batch = default_data_collator(examples)

        device = batch["cds_input_ids"].device


        inputs = batch["cds_input_ids"].clone()
        labels = inputs.clone()


        probability_matrix = torch.full(
            inputs.shape, self.mlm_probability,
            device=device)


        special_tokens_mask = torch.zeros_like(
            inputs, dtype=torch.bool)
        for special_id in [self.pad_token_id,
                           self.bos_token_id,
                           self.eos_token_id]:
            if special_id is not None:
                special_tokens_mask = special_tokens_mask | (inputs == special_id)

        probability_matrix.masked_fill_(
            special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(
            ~batch["cds_attention_mask"].bool(),
            value=0.0)


        masked_indices = torch.bernoulli(
            probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_mask = torch.bernoulli(
            torch.full(labels.shape, 0.8,
                       device=device)).bool() & masked_indices
        inputs[indices_mask] = self.mask_token_id

        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5,
                       device=device)).bool() & masked_indices & ~indices_mask
        random_words = torch.randint(
            self.vocab_size, labels.shape,
            device=device)
        inputs[indices_random] = random_words[
            indices_random]
        batch["cds_input_ids"] = inputs
        batch["labels"] = labels
        batch[
            "masked_indices"] = masked_indices
        batch[
            "mask_token_positions"] = indices_mask
        return batch


def compute_mlm_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = logits.argmax(-1)
    mask = labels != -100
    valid_predictions = predictions[mask]
    valid_labels = labels[mask]
    correct_predictions = (valid_predictions == valid_labels).sum()
    total_predictions = mask.sum()
    accuracy = float(correct_predictions) / float(
        total_predictions) if total_predictions > 0 else 0
    return {
        "mask_accuracy": accuracy,
    }







