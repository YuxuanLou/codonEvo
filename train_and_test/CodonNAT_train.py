import os
import torch
import multimolecule
from multimolecule import RnaTokenizer, RnaFmModel
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig
from transformers import EarlyStoppingCallback
import numpy as np
import argparse
from collections import defaultdict

from models.CodonNAT import CustomPlantRNAModelmlm
from utils import CustomDataset, MLMDataCollator, compute_mlm_metrics
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from torch.utils.data import DataLoader, Dataset

def translate_codon_to_aa(codon):
    codon_table = {
        'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
        'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
        'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
        'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
        'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
        'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
        'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
        'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }

    if 'N' in codon or codon not in codon_table:
        return 'X'

    return codon_table.get(codon, 'X')


def create_optimizer(model, training_args):
    pretrained_params = []
    custom_params = []
    trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    for name, param in trainable_params.items():
        if name.startswith('plantrna.'):
            pretrained_params.append(param)
        elif name.startswith('esm2.'):
            pretrained_params.append(param)
        else:
            custom_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": pretrained_params,
            "lr": 1e-4,  
            "weight_decay": 0.01, 
        },
        {
            "params": custom_params,
            "lr": 1e-4,  
            "weight_decay": 0.01,  
        },
    ]

    # 创建带参数组的优化器
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    return optimizer


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logs_dir = os.path.join(args.output_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    df = pd.read_csv(args.dataset_path, sep=",")

    train_val_data, test_data = train_test_split(df, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=42)

    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    cds_tokenizer = RnaTokenizer.from_pretrained("multimolecule/mrnafm")
    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    config = AutoConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")

    def process_data(data):
        processed_samples = []
        for _, row in data.iterrows():
            cds_encoding = cds_tokenizer(
                row['cds_sequence'],
                padding='max_length',
                truncation=True,
                max_length=1024,
                return_tensors='pt'
            )
            protein_encoding = protein_tokenizer(
                row['protein_sequence'],
                padding='max_length',
                truncation=True,
                max_length=1024,
                return_tensors='pt'
            )
            processed_sample = {
                'cds_input_ids': cds_encoding['input_ids'].squeeze(),
                'cds_attention_mask': cds_encoding['attention_mask'].squeeze(),
                'protein_input_ids': protein_encoding['input_ids'].squeeze(),
                'protein_attention_mask': protein_encoding['attention_mask'].squeeze()
            }
            processed_samples.append(processed_sample)
        return processed_samples

    processed_train = process_data(train_data)
    processed_val = process_data(val_data)
    processed_test = process_data(test_data)

    train_dataset = CustomDataset(processed_train)
    val_dataset = CustomDataset(processed_val)
    test_dataset = CustomDataset(processed_test)

    model = CustomPlantRNAModelmlm(config).to(device)

    def make_model_contiguous(model):
        for name, param in model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        for name, buffer in model.named_buffers():
            if not buffer.is_contiguous():
                buffer.data = buffer.data.contiguous()
        return model

    model = make_model_contiguous(model)


    data_collator = MLMDataCollator(cds_tokenizer)

    model_output_dir = os.path.join(args.output_dir, 'model-results')

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        save_total_limit=1,
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=50,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mask_accuracy",
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        compute_metrics=compute_mlm_metrics,
        optimizers=(create_optimizer(model, training_args), None)
    )


    trainer.train()

    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]

    model_save_path = os.path.join(args.output_dir, f"{dataset_name}-{args.model_name}")
    trainer.save_model(model_save_path)



if __name__ == "__main__":
    main()

