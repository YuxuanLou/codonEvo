import math
import argparse
import re
import os
import os.path as osp
import torch
import multimolecule
from multimolecule import RnaTokenizer, \
    RnaFmForMaskedLM
import numpy as np
import random
import pandas as pd
from Bio.Seq import Seq
from transformers import AutoTokenizer, \
    AutoConfig, default_data_collator
from copy import deepcopy
from safetensors.torch import load_file
import hashlib
import torch.nn.functional as F
import time


from model2_inference import CustomPlantRNAModel
from mrnafm_pro_mlm_inference import CustomPlantRNAModelmlm
from utils import CustomDataset


def parse_fasta(fasta_file):
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[
                        current_id] = ''.join(
                        current_seq)
                current_id = line[1:]
                current_seq = []
            elif line:
                current_seq.append(line)


    if current_id is not None:
        sequences[current_id] = ''.join(
            current_seq)

    return sequences


def stable_hash(s):
    return int(hashlib.sha256(
        s.encode('utf-8')).hexdigest(), 16)


class MLMDataCollator:
    def __init__(self, tokenizer,
                 mlm_probability=0.15, seed=42):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.seed = seed
        self.mask_cache = {}
        self.random_cache = {}
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __call__(self, examples):
        batch = default_data_collator(examples)
        device = batch["cds_input_ids"].device

        # Clone input IDs for labels
        inputs = batch["cds_input_ids"].clone()
        labels = inputs.clone()

        batch_size, seq_length = inputs.shape


        if seq_length not in self.mask_cache:

            self._create_mask_pattern(seq_length,
                                      device)


        special_tokens_mask = torch.zeros(
            (batch_size, seq_length),
            dtype=torch.bool, device=device)
        for special_id in [self.pad_token_id,
                           self.bos_token_id,
                           self.eos_token_id]:
            if special_id is not None:
                special_tokens_mask = special_tokens_mask | (
                            inputs == special_id)


        attention_mask = batch[
            "cds_attention_mask"].bool()


        masked_indices = self.mask_cache[
            seq_length].clone().to(
            device).unsqueeze(0).expand(
            batch_size, -1)
        masked_indices = masked_indices & (
            ~special_tokens_mask) & attention_mask


        labels[~masked_indices] = -100


        indices_mask = \
        self.random_cache[seq_length][
            "mask"].clone().to(device).unsqueeze(
            0).expand(batch_size,
                      -1) & masked_indices
        inputs[indices_mask] = self.mask_token_id


        indices_random = \
        self.random_cache[seq_length][
            "random"].clone().to(
            device).unsqueeze(0).expand(
            batch_size,
            -1) & masked_indices & ~indices_mask
        random_words = \
        self.random_cache[seq_length][
            "words"].clone().to(device).unsqueeze(
            0).expand(batch_size, -1)
        inputs[indices_random] = random_words[
            indices_random]

        # 更新批次
        batch["cds_input_ids"] = inputs
        batch["labels"] = labels

        return batch

    def _create_mask_pattern(self, seq_length,
                             device):

        torch.manual_seed(self.seed)


        probability_matrix = torch.full(
            (seq_length,), self.mlm_probability)
        masked_indices = torch.bernoulli(
            probability_matrix,
            generator=self.generator).bool()


        mask_prob_matrix = torch.full(
            (seq_length,), 0.8)
        indices_mask = torch.bernoulli(
            mask_prob_matrix,
            generator=self.generator).bool() & masked_indices


        random_prob_matrix = torch.full(
            (seq_length,), 0.5)
        indices_random = torch.bernoulli(
            random_prob_matrix,
            generator=self.generator).bool() & masked_indices & ~indices_mask


        random_words = torch.randint(
            self.vocab_size, (seq_length,),
            generator=self.generator)


        self.mask_cache[
            seq_length] = masked_indices.cpu()
        self.random_cache[seq_length] = {
            "mask": indices_mask.cpu(),
            "random": indices_random.cpu(),
            "words": random_words.cpu()
        }



codon_table = {
    'AUA': 'I', 'AUC': 'I', 'AUU': 'I',
    'AUG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T',
    'ACU': 'T',
    'AAC': 'N', 'AAU': 'N', 'AAA': 'K',
    'AAG': 'K',
    'AGC': 'S', 'AGU': 'S', 'AGA': 'R',
    'AGG': 'R',
    'CUA': 'L', 'CUC': 'L', 'CUG': 'L',
    'CUU': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P',
    'CCU': 'P',
    'CAC': 'H', 'CAU': 'H', 'CAA': 'Q',
    'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R',
    'CGU': 'R',
    'GUA': 'V', 'GUC': 'V', 'GUG': 'V',
    'GUU': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A',
    'GCU': 'A',
    'GAC': 'D', 'GAU': 'D', 'GAA': 'E',
    'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G',
    'GGU': 'G',
    'UCA': 'S', 'UCC': 'S', 'UCG': 'S',
    'UCU': 'S',
    'UUC': 'F', 'UUU': 'F', 'UUA': 'L',
    'UUG': 'L',
    'UAC': 'Y', 'UAU': 'Y', 'UAA': '*',
    'UAG': '*',
    'UGC': 'C', 'UGU': 'C', 'UGA': '*',
    'UGG': 'W',
}


synonymous_codons = {}
for codon, aa in codon_table.items():
    if aa not in synonymous_codons:
        synonymous_codons[aa] = []
    synonymous_codons[aa].append(codon)


class GeneticOptimizer:
    def __init__(self,
                 model_dir='./full_dataset_models',
                 perplexity_model_dir='./perplexity_model',
                 population_size=50,
                 mutation_rate=5,
                 crossover_rate=0.7,
                 max_generations=100,
                 batch_size=50,
                 weights_save_path='./attention_weights',
                 selection_top_percent=0.2,
                 top_n_return=5,
                 perplexity_weight=1):

        self.model_dir = model_dir
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.batch_size = batch_size
        self.weights_save_path = weights_save_path
        self.selection_top_percent = selection_top_percent
        self.top_n_return = top_n_return
        self.perplexity_model_dir = perplexity_model_dir
        self.perplexity_weight = perplexity_weight  # 新增：混淆度权重

        # Create directory for saving weights if it doesn't exist
        os.makedirs(self.weights_save_path,
                    exist_ok=True)

        # Initialize tokenizers and models
        self.cds_tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/mrnafm")
        self.protein_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t33_650M_UR50D")
        # Load configuration and models
        self.config = AutoConfig.from_pretrained(
            "facebook/esm2_t33_650M_UR50D")
        self.config.num_labels = 1

        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.perplexity_model = CustomPlantRNAModelmlm(self.config).to(self.device)
        full_state_dict = load_file(f"{self.perplexity_model_dir}/model.safetensors",device="cpu")
        self.perplexity_model.set_state_dict(full_state_dict)
        self.perplexity_model.eval()

        # Load all models for ensemble prediction
        self.models = []
        for fold in range(1, 6):
            model_path = os.path.join(model_dir,
                                      f"classification-model-fold-{fold}")
            model = CustomPlantRNAModel(
                self.config)
            full_state_dict = load_file(f"{model_path}/model.safetensors",device="cpu")
            model.set_state_dict(full_state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)


        self.all_sequences_data = {}

    def calculate_perplexity(self, cds_sequences, nat_protein_emb):

        n_seeds = 10
        seeds = range(42,
                      42 + n_seeds)
        mlm_collators = [
            MLMDataCollator(self.cds_tokenizer,
                            seed=seed) for seed in
            seeds]


        all_perplexities_by_seed = [[] for _ in
                                    range(
                                        n_seeds)]


        for i in range(0, len(cds_sequences),
                       self.batch_size):
            batch_sequences = cds_sequences[
                              i:i + self.batch_size]

            cds_encoding = self.cds_tokenizer(
                batch_sequences,
                padding='max_length',
                truncation=True,
                max_length=1024,
                return_tensors='pt'
            )
            batch_samples = []
            for j in range(len(batch_sequences)):
                sample = {
                    'cds_input_ids':
                        cds_encoding['input_ids'][
                            j].squeeze(),
                    'cds_attention_mask':
                        cds_encoding[
                            'attention_mask'][
                            j].squeeze()

                }
                batch_samples.append(sample)


            for seed_idx, mlm_collator in enumerate(
                    mlm_collators):
                batch = mlm_collator(
                    batch_samples)


                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                with torch.no_grad():

                    outputs = self.perplexity_model(
                        cds_input_ids=batch['cds_input_ids'],
                        cds_attention_mask=batch['cds_attention_mask'],
                        pre_computed_protein_embeddings=nat_protein_emb,
                        labels=None

                    )

                    logits = outputs["logits"]  # [batch_size, seq_len, vocab_size]
                    labels = batch['labels']  # [batch_size, seq_len]


                    batch_perplexities = []
                    for b in range(
                            logits.size(0)):
                        sample_logits = logits[
                            b]  # [seq_len, vocab_size]
                        sample_labels = labels[
                            b]  # [seq_len]


                        mask_positions = (
                                    sample_labels != -100)
                        if mask_positions.sum() == 0:

                            batch_perplexities.append(
                                1.0)
                            continue

                        masked_logits = \
                        sample_logits[
                            mask_positions]  # [num_masks, vocab_size]
                        masked_labels = \
                        sample_labels[
                            mask_positions]  # [num_masks]


                        masked_probs = F.softmax(
                            masked_logits,
                            dim=-1)  # [num_masks, vocab_size]


                        label_probs = torch.gather(
                            masked_probs, 1,
                            masked_labels.unsqueeze(
                                1)).squeeze(
                            1)  # [num_masks]


                        log_probs = torch.log(
                            label_probs)  # [num_masks]


                        avg_log_prob = log_probs.mean().item()


                        perplexity = math.exp(
                            avg_log_prob)
                        batch_perplexities.append(
                            perplexity)

                    all_perplexities_by_seed[
                        seed_idx].extend(
                        batch_perplexities)


        avg_perplexities = []
        for seq_idx in range(len(cds_sequences)):

            if all(seq_idx < len(perplexities) for
                   perplexities in
                   all_perplexities_by_seed):
                avg_perplexity = sum(
                    all_perplexities_by_seed[
                        seed_idx][seq_idx] for
                    seed_idx in
                    range(n_seeds)) / n_seeds
                avg_perplexities.append(
                    avg_perplexity)

        return avg_perplexities

    def translate_cds(self, cds_sequence):

        if isinstance(cds_sequence, list):

            protein_sequences = []
            for seq in cds_sequence:
                coding_dna = Seq(seq)
                protein_seq = str(
                    coding_dna.translate())
                if protein_seq.endswith("*"):
                    protein_seq = protein_seq[:-1]
                protein_sequences.append(
                    protein_seq)
            return protein_sequences
        else:

            coding_dna = Seq(cds_sequence)
            protein_sequence = str(
                coding_dna.translate())
            if protein_sequence.endswith("*"):
                protein_sequence = protein_sequence[
                                   :-1]
            return protein_sequence

    def get_attention_weights(self, cds_sequence):

        protein_sequence = self.translate_cds(
            cds_sequence)

        cds_encoding = self.cds_tokenizer(
            cds_sequence,
            padding='max_length',
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        ).to(self.device)

        protein_encoding = self.protein_tokenizer(
            protein_sequence,
            padding='max_length',
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        ).to(self.device)


        all_attention_weights = []

        for model in self.models:

            original_forward = model.pooler.forward

            attention_weights = []

            def hook_forward(self, hidden_state,
                             attention_mask):
                batch_size, seq_len, _ = hidden_state.size()
                Q = self.query_linear(
                    hidden_state[:, 0:1, :])
                K = self.key_linear(hidden_state)
                attention_scores = torch.matmul(Q,
                                                K.transpose(
                                                    -2,
                                                    -1))
                attention_scores = attention_scores / torch.sqrt(
                    torch.tensor(K.size(-1),
                                 dtype=torch.float32,
                                 device=K.device))
                attention_mask = attention_mask.unsqueeze(
                    1).float()
                attention_scores = torch.where(
                    attention_mask > 0,
                    attention_scores,
                    torch.tensor(-1e9,
                                 device=attention_scores.device))
                weights = torch.nn.functional.softmax(
                    attention_scores, dim=-1)


                nonlocal attention_weights
                attention_weights = weights.cpu().detach().numpy()

                pooled_output = torch.matmul(
                    weights, hidden_state)

                return pooled_output.squeeze(1)


            model.pooler.forward = hook_forward.__get__(
                model.pooler, type(model.pooler))

            with torch.no_grad():
                model(cds_input_ids=cds_encoding[
                    'input_ids'],
                      cds_attention_mask=
                      cds_encoding[
                          'attention_mask'],
                      protein_input_ids=
                      protein_encoding[
                          'input_ids'],
                      protein_attention_mask=
                      protein_encoding[
                          'attention_mask'], )


            model.pooler.forward = original_forward


            weights = attention_weights[0, 0, :]
            all_attention_weights.append(weights)


        mean_attention_weights = np.mean(
            all_attention_weights, axis=0)

        return mean_attention_weights

    def predict_expression_batch(self,
                                 cds_sequences, expr_protein_embs):

        expression_probs = []


        for i in range(0, len(cds_sequences),
                       self.batch_size):
            batch_sequences = cds_sequences[
                              i:i + self.batch_size]
            batch_proteins = self.translate_cds(
                batch_sequences)


            cds_encoding = self.cds_tokenizer(
                batch_sequences,
                padding='max_length',
                truncation=True,
                max_length=1024,
                return_tensors='pt'
            ).to(self.device)


            batch_probs = []
            with torch.no_grad():
                for model in self.models:
                    outputs = model(
                        cds_input_ids=
                        cds_encoding['input_ids'],
                        cds_attention_mask=
                        cds_encoding[
                            'attention_mask'],
                        pre_computed_protein_embeddings=expr_protein_embs[model] )
                    logits = outputs[
                        'logits'].cpu().numpy()
                    probs = 1 / (1 + np.exp(-logits))  # sigmoid
                    batch_probs.append(probs)


            ensemble_probs = np.mean(batch_probs,
                                     axis=0).flatten()
            expression_probs.extend(
                ensemble_probs)

        return expression_probs

    def predict_expression(self, cds_sequence,expr_protein_embs):

        if isinstance(cds_sequence, list):
            return self.predict_expression_batch(
                cds_sequence, expr_protein_embs)
        else:
            result = self.predict_expression_batch(
                [cds_sequence],expr_protein_embs)
            return result[0]

    def calculate_fitness(self, cds_sequences, nat_protein_emb,expr_protein_embs):

        expression_probs = self.predict_expression_batch(
            cds_sequences, expr_protein_embs)


        perplexities = self.calculate_perplexity(
            cds_sequences, nat_protein_emb)

        fitness_scores = [expr * (
                    perp ** self.perplexity_weight)
                          for expr, perp in
                          zip(expression_probs,
                              perplexities)]
        return fitness_scores, expression_probs, perplexities

    def create_initial_population(self,
                                  cds_sequence):

        attention_weights = self.get_attention_weights(
            cds_sequence)


        max_length = min(1022,
                         len(cds_sequence) // 3)
        codon_attn_weights = attention_weights[
                             1:(max_length + 1)]


        codon_attn_weights = codon_attn_weights / np.sum(
            codon_attn_weights)

        hash_value = stable_hash(cds_sequence)
        weights_filename = os.path.join(
            self.weights_save_path,
            f"attention_weights{hash_value}.npy")
        np.save(weights_filename,
                codon_attn_weights)
        print(
            f"Saved attention weights to {weights_filename}")

        mutation_probs = codon_attn_weights * self.mutation_rate * 100 + 1 / len(codon_attn_weights) * self.mutation_rate * 100

        original_protein = self.translate_cds(
            cds_sequence)


        synonymous_options = []
        for i in range(0, max_length):
            codon = cds_sequence[i * 3 : i * 3 + 3]
            amino_acid = codon_table.get(codon,
                                         '*')
            valid_replacements = [c for c in
                                  synonymous_codons.get(
                                      amino_acid,
                                      [])]
            synonymous_options.append(
                valid_replacements)


        population = [cds_sequence]


        while len(
                population) < self.population_size:
            mutated_seq = self.mutate(
                cds_sequence, mutation_probs,
                synonymous_options)
            population.append(mutated_seq)

        return population, mutation_probs, original_protein, synonymous_options

    def mutate(self, cds_sequence, mutation_probs,
               synonymous_options=None):

        sequence_list = list(cds_sequence)
        codon_count = len(sequence_list) // 3


        mutation_mask = np.random.random(
            len(mutation_probs)) < mutation_probs

        for i in np.where(mutation_mask)[0]:
            if synonymous_options and i < len(
                    synonymous_options) and \
                    synonymous_options[i]:
                codon_pos = i * 3
                new_codon = random.choice(
                    synonymous_options[i])
                sequence_list[
                codon_pos:codon_pos + 3] = list(
                    new_codon)

        return ''.join(sequence_list)

    def crossover(self, parent1, parent2):

        if random.random() > self.crossover_rate:
            return parent1, parent2


        crossover_point = random.randint(0,
                                         len(parent1) // 3) * 3


        child1 = parent1[
                 :crossover_point] + parent2[
                                     crossover_point:]
        child2 = parent2[
                 :crossover_point] + parent1[
                                     crossover_point:]

        return child1, child2

    def optimize(self, cds_sequence):

        print(
            "Starting CDS sequence optimization...")


        self.all_sequences_data = {}


        start_time = time.time()
        population, mutation_probs, original_protein, synonymous_options = self.create_initial_population(
            cds_sequence)
        creation_time = time.time() - start_time
        print(
            f"Initial population creation time: {creation_time:.2f} seconds")

        start_time = time.time()
        protein_encoding = self.protein_tokenizer(
            original_protein,
            padding='max_length',
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            nat_protein_emb = self.perplexity_model.compute_protein_embeddings(
                protein_input_ids=protein_encoding.input_ids,
                protein_attention_mask=protein_encoding.attention_mask
            )

            expr_protein_embs = {}
            for model in self.models:
                expr_protein_embs[
                    model] = model.compute_protein_embeddings(
                    protein_input_ids=protein_encoding.input_ids,
                    protein_attention_mask=protein_encoding.attention_mask
                )
        fitness_scores, expression_probs, perplexities = self.calculate_fitness(
            population, nat_protein_emb,expr_protein_embs)
        evaluation_time = time.time() - start_time
        print(
            f"Initial population evaluation time: {evaluation_time:.2f} seconds")

        for seq, expr, perp in zip(population,
                                   expression_probs,
                                   perplexities):
            self.all_sequences_data[seq] = {
                'expression': expr,
                'perplexity': perp}

        best_idx = np.argmax(fitness_scores)
        best_sequence = population[best_idx]
        best_fitness = fitness_scores[best_idx]
        best_expression = expression_probs[
            best_idx]
        best_perplexity = perplexities[best_idx]
        population_with_scores = list(
            zip(population, fitness_scores,
                expression_probs, perplexities))
        print(
            f"Initial best fitness: {best_fitness:.4f}")
        print(
            f"Initial best expression probability: {best_expression:.4f}")
        print(
            f"Initial best perplexity: {best_perplexity:.4f}")


        history = {
            'generation': [0],
            'best_fitness': [best_fitness],
            'best_expression': [best_expression],
            'best_perplexity': [best_perplexity],
            'avg_fitness': [
                np.mean(fitness_scores)],
            'avg_expression': [
                np.mean(expression_probs)],
            'avg_perplexity': [
                np.mean(perplexities)],
            'all_scores': [population_with_scores]
        }


        no_improvement_gen = 0
        patience = 10


        for generation in range(1,
                                self.max_generations + 1):
            print(f"\nGeneration {generation}:")


            num_top = int(
                self.population_size * self.selection_top_percent)
            top_indices = np.argpartition(
                fitness_scores, -num_top)[
                          -num_top:]
            selected = [population[i] for i in
                        top_indices]


            new_population = []

            elite_idx = np.argmax(fitness_scores)
            new_population.append(
                population[elite_idx])


            start_crossover_time = time.time()
            while len(
                    new_population) < self.population_size:

                parent1 = random.choice(selected)
                parent2 = random.choice(selected)


                child1, child2 = self.crossover(
                    parent1, parent2)


                child1 = self.mutate(child1,
                                     mutation_probs,
                                     synonymous_options)
                child2 = self.mutate(child2,
                                     mutation_probs,
                                     synonymous_options)

                # 添加到新种群
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            crossover_mutation_time = time.time() - start_crossover_time
            print(
                f"Crossover and mutation time: {crossover_mutation_time:.2f} seconds")
            # 更新种群
            population = new_population[
                         :self.population_size]

            # 批量评估新种群 - 添加计时
            start_eval_time = time.time()
            fitness_scores, expression_probs, perplexities = self.calculate_fitness(
                population, nat_protein_emb, expr_protein_embs)
            evaluation_time = time.time() - start_eval_time
            print(
                f"Population evaluation time: {evaluation_time:.2f} seconds")
            # 更新全局序列跟踪器
            for seq, expr, perp in zip(population,
                                       expression_probs,
                                       perplexities):
                self.all_sequences_data[seq] = {
                    'expression': expr,
                    'perplexity': perp}

            current_best_idx = np.argmax(
                fitness_scores)
            current_best_fitness = fitness_scores[
                current_best_idx]
            current_best_expression = \
            expression_probs[current_best_idx]
            current_best_perplexity = \
            perplexities[current_best_idx]
            population_with_scores = list(
                zip(population, fitness_scores,
                    expression_probs,
                    perplexities))
            # 更新全局最优
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_sequence = population[
                    current_best_idx]
                best_expression = current_best_expression
                best_perplexity = current_best_perplexity
                no_improvement_gen = 0
            else:
                no_improvement_gen += 1

            # 记录历史
            history['generation'].append(
                generation)
            history['best_fitness'].append(
                best_fitness)
            history['best_expression'].append(
                best_expression)
            history['best_perplexity'].append(
                best_perplexity)
            history['avg_fitness'].append(
                np.mean(fitness_scores))
            history['avg_expression'].append(
                np.mean(expression_probs))
            history['avg_perplexity'].append(
                np.mean(perplexities))
            history['all_scores'].append(
                population_with_scores)

            print(
                f"Current best fitness: {current_best_fitness:.4f}")
            print(
                f"Current best expression probability: {current_best_expression:.4f}")
            print(
                f"Current best perplexity: {current_best_perplexity:.4f}")
            print(
                f"Historical best fitness: {best_fitness:.4f}")


            if no_improvement_gen >= patience:
                print(
                    f"\nNo improvement for {patience} generations, stopping optimization")
                break

        # 验证最终序列编码相同蛋白质
        final_protein = self.translate_cds(
            best_sequence)
        assert final_protein == original_protein, "Error: Optimized sequence encodes a different protein!"

        print("\nOptimization complete!")
        original_expr = self.predict_expression(cds_sequence, expr_protein_embs)
        original_perp = \
        self.calculate_perplexity([cds_sequence],nat_protein_emb)[0]
        print(
            f"Original expression probability: {original_expr:.4f}")
        print(
            f"Original perplexity: {original_perp:.4f}")
        print(
            f"Optimized expression probability: {best_expression:.4f}")
        print(
            f"Optimized perplexity: {best_perplexity:.4f}")

        # 从全局序列跟踪器中获取前n个最优序列
        sorted_sequences = sorted([(seq, data[
            'expression'] * (data[
                                 'perplexity'] ** self.perplexity_weight))
                                   for seq, data
                                   in
                                   self.all_sequences_data.items()],
                                  key=lambda x: x[
                                      1],
                                  reverse=True)

        top_n = min(self.top_n_return,
                    len(sorted_sequences))
        top_sequences = [seq for seq, _ in
                         sorted_sequences[:top_n]]
        top_expressions = [
            self.all_sequences_data[seq][
                'expression'] for seq in
            top_sequences]
        top_perplexities = [
            self.all_sequences_data[seq][
                'perplexity'] for seq in
            top_sequences]
        # top_fitness_scores = [expr + self.perplexity_weight * perp for expr, perp in zip(top_expressions, top_perplexities)]
        top_fitness_scores = [expr * (
                    perp ** self.perplexity_weight)
                              for expr, perp in
                              zip(top_expressions,
                                  top_perplexities)]

        return {
            'original_sequence': cds_sequence,
            'optimized_sequence': best_sequence,
            'original_expression': original_expr,
            'optimized_expression': best_expression,
            'original_perplexity': original_perp,
            'optimized_perplexity': best_perplexity,
            'protein_sequence': original_protein,
            'history': pd.DataFrame(history),
            'top_sequences': top_sequences,
            'top_expressions': top_expressions,
            'top_perplexities': top_perplexities,
            'top_fitness_scores': top_fitness_scores
        }


def t_to_u(sequence):
    """将字符串中的 T 替换为 U"""
    return sequence.replace('T', 'U')


def u_to_t(sequence):
    """将字符串中的 U 替换为 T"""
    return sequence.replace('U', 'T')


def is_valid_cds(cds_sequence):
    # 检查长度是否为3的倍数
    if len(cds_sequence) % 3 != 0:
        print("序列长度不为3的倍数")
        return False
    # 检查是否只包含AUGC字符
    if not re.fullmatch('^[AUGC]*$',
                        cds_sequence):
        print("序列含有未知核苷酸")
        return False
    # 检查每个密码子是否都不是终止密码子
    stop_codons = {'UAA', 'UAG', 'UGA'}
    for i in range(0, len(cds_sequence) - 3, 3):
        codon = cds_sequence[i:i + 3]
        if codon in stop_codons:
            print("序列含有终止密码子", i)
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='优化CDS序列')
    parser.add_argument('--input', type=str,
                        required=True,
                        help='输入fasta文件路径')
    parser.add_argument('--output', type=str,
                        required=True,
                        help='输出fasta文件路径')
    parser.add_argument('--model_dir', type=str,
                        default='./model',
                        help='模型路径')
    parser.add_argument('--perplexity_model_dir',
                        type=str, default=None,
                        help='计算困惑度模型路径')
    parser.add_argument('--population_size',
                        type=int, default=100,
                        help='种群大小')
    parser.add_argument('--mutation_rate',
                        type=float, default=0.05,
                        help='突变率')
    parser.add_argument('--crossover_rate',
                        type=float, default=0.7,
                        help='交叉率')
    parser.add_argument('--max_generations',
                        type=int, default=100,
                        help='最大迭代次数')
    parser.add_argument('--selection_top_percent',
                        type=float, default=0.2,
                        help='选择百分比')
    parser.add_argument('--batch_size', type=int,
                        default=50,
                        help='批处理大小')
    parser.add_argument('--seed', type=int,
                        default=42,
                        help='随机种子')
    parser.add_argument('--top_n', type=int,
                        default=5,
                        help='返回top N结果')
    parser.add_argument('--perplexity_weight',
                        type=float, default=0.3,
                        help='困惑度权重')
    parser.add_argument(
        '--attention_weights_save_path', type=str,
        default=None, help='注意力权重保存路径')
    parser.add_argument('--results', type=str,
                        default=None,
                        help='结果保存路径')
    parser.add_argument('--history', type=str,
                        default=None,
                        help='历史记录保存路径')
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 读取fasta文件
    sequences = parse_fasta(args.input)

    # 优化每个序列
    optimized_sequences = {}
    for seq_id, cds_sequence in sequences.items():
        print(f"正在优化序列: {seq_id}")
        cds_sequence = t_to_u(
            cds_sequence.upper())
        # 创建基于序列名的子目录
        seq_name = seq_id.split()[
            0]  # 获取序列ID的第一部分作为名称

        # 为每个路径参数添加序列名子文件夹
        if args.attention_weights_save_path:
            seq_dir = osp.join(
                args.attention_weights_save_path,
                seq_name)
            os.makedirs(seq_dir, exist_ok=True)
        else:
            seq_dir = None

        if args.results:
            seq_results = osp.join(args.results,
                                   seq_name,
                                   "results.txt")
            os.makedirs(osp.dirname(seq_results),
                        exist_ok=True)
        else:
            seq_results = None

        if args.history:
            seq_history = osp.join(args.history,
                                   seq_name,
                                   "optimization_history.csv")
            os.makedirs(osp.dirname(seq_history),
                        exist_ok=True)
        else:
            seq_history = None

        # 创建优化器
        optimizer = GeneticOptimizer(
            model_dir=args.model_dir,
            population_size=args.population_size,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            max_generations=args.max_generations,
            batch_size=args.batch_size,
            weights_save_path=seq_dir,
            selection_top_percent=args.selection_top_percent,
            top_n_return=args.top_n,
            perplexity_weight=args.perplexity_weight,
            perplexity_model_dir=args.perplexity_model_dir
        )

        if is_valid_cds(cds_sequence):
            result = optimizer.optimize(
                cds_sequence)

            # 保存结果
            if seq_history:
                result['history'].to_csv(
                    seq_history, index=False)

            if seq_results:
                with open(seq_results, 'w') as f:
                    f.write(
                        f"原始CDS序列: {result['original_sequence']}\n")
                    f.write(
                        f"蛋白质序列: {result['protein_sequence']}\n")
                    f.write(
                        f"原始高表达概率: {result['original_expression']:.4f}\n")
                    f.write(
                        f"原始自然度: {result['original_perplexity']:.4f}\n")
                    original_fitness = result[
                                           'original_expression'] * (
                                                   result[
                                                       'original_perplexity'] ** optimizer.perplexity_weight)
                    f.write(
                        f"原始适合度: {original_fitness:.4f}\n\n")

                    f.write(
                        f"优化后的Top {len(result['top_sequences'])} 序列:\n\n")
                    for i in range(len(result[
                                           'top_sequences'])):
                        seq = \
                        result['top_sequences'][i]
                        expr = \
                        result['top_expressions'][
                            i]
                        perp = result[
                            'top_perplexities'][i]
                        fitness = result[
                            'top_fitness_scores'][
                            i]

                        # 计算变化率
                        original_seq = result[
                            'original_sequence']
                        nucleotide_changes = sum(
                            1 for a, b in
                            zip(original_seq, seq)
                            if a != b)
                        nucleotide_change_rate = nucleotide_changes / len(
                            original_seq)

                        codon_changes = 0
                        for j in range(0,
                                       len(original_seq),
                                       3):
                            if original_seq[
                               j:j + 3] != seq[
                                           j:j + 3]:
                                codon_changes += 1
                        codon_change_rate = codon_changes / (
                                    len(original_seq) // 3)

                        f.write(f"Top {i + 1}:\n")
                        f.write(f"序列: {seq}\n")
                        f.write(
                            f"高表达概率: {expr:.4f}\n")
                        f.write(
                            f"自然度: {perp:.4f}\n")
                        f.write(
                            f"适合度: {fitness:.4f}\n")
                        f.write(
                            f"核苷酸变化率: {nucleotide_change_rate:.4f}\n")
                        f.write(
                            f"密码子变化率: {codon_change_rate:.4f}\n\n")

            # 使用适合度最高的序列
            optimized_sequences[seq_id] = \
            result['top_sequences'][0]
        else:
            print(
                f"序列 {seq_id} 不是有效的CDS序列，将保持原样")
            optimized_sequences[
                seq_id] = cds_sequence

    # 保存优化后的序列到fasta文件
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output, 'w') as f:
        for seq_id, sequence in optimized_sequences.items():
            f.write(f">{seq_id}\n")
            # 每60个字符一行
            for i in range(0, len(sequence), 60):
                f.write(f"{sequence[i:i + 60]}\n")


if __name__ == "__main__":
    main()