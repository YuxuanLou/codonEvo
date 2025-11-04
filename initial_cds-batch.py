import os
import torch
import argparse
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable
from multimolecule import RnaTokenizer
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file
from mrnafm_pro_mlm import CustomPlantRNAModelmlm


def translate_cds_to_protein(cds_sequence):
    seq = Seq(cds_sequence)
    protein = seq.translate()
    return str(protein)


def get_masked_cds_for_protein(protein_sequence):
    masked_cds = "NNN" * len(protein_sequence)
    return masked_cds


def get_codon_to_amino_acid_map():
    standard_table = CodonTable.standard_rna_table
    codon_to_aa = {}
    for codon, aa in standard_table.forward_table.items():
        codon_to_aa[codon] = aa
    for stop_codon in standard_table.stop_codons:
        codon_to_aa[stop_codon] = '*'
    return codon_to_aa


def mask_all_codons(input_ids, attention_mask,tokenizer):
    masked_inputs = input_ids.clone()
    special_tokens = [tokenizer.pad_token_id,tokenizer.eos_token_id,tokenizer.bos_token_id]
    valid_positions = attention_mask.bool()
    for token_id in special_tokens:
        if token_id is not None:
            valid_positions &= (input_ids != token_id)
    masked_inputs[valid_positions] = tokenizer.mask_token_id
    mask_indices = valid_positions.nonzero(as_tuple=True)
    return masked_inputs, mask_indices


def predict_cds_from_protein(model, cds_tokenizer,protein_tokenizer,protein_sequence,device):
    codon_to_aa = get_codon_to_amino_acid_map()
    protein_encoding = protein_tokenizer(
        protein_sequence,
        padding='max_length',
        truncation=True,
        max_length=1024,
        return_tensors='pt'
    )

    masked_cds = get_masked_cds_for_protein(protein_sequence)

    cds_encoding = cds_tokenizer(
        masked_cds,
        padding='max_length',
        truncation=True,
        max_length=1024,
        return_tensors='pt'
    )

    masked_inputs, mask_positions = mask_all_codons(
        input_ids=cds_encoding['input_ids'],
        attention_mask=cds_encoding['attention_mask'],
        tokenizer=cds_tokenizer
    )

    protein_input_ids = protein_encoding[
        'input_ids'].to(device)
    protein_attention_mask = protein_encoding[
        'attention_mask'].to(device)
    cds_input_ids = masked_inputs.to(device)
    cds_attention_mask = cds_encoding[
        'attention_mask'].to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(
            cds_input_ids=cds_input_ids,
            cds_attention_mask=cds_attention_mask,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask
        )

        logits = outputs["logits"]
        mask_logits = logits.squeeze()[mask_positions[1]]

        best_codons = []

        for pos, aa in enumerate(protein_sequence):
            pos_logits = mask_logits[pos]
            probabilities = torch.softmax(pos_logits, dim=0)
            sorted_indices = torch.argsort(probabilities, descending=True)

            found_valid_codon = False
            for idx in sorted_indices:
                token_id = idx.item()
                codon = cds_tokenizer.decode([token_id]).strip()

                if len(codon) == 3 and codon in codon_to_aa:
                    if codon_to_aa[codon] == aa:
                        best_codons.append(codon)
                        found_valid_codon = True
                        break

        predicted_cds = "".join(best_codons)
        translated_protein = translate_cds_to_protein(predicted_cds)
    return predicted_cds, translated_protein


def process_protein_sequences(model,
                              cds_tokenizer,
                              protein_tokenizer,
                              input_file,
                              output_file,
                              device):
    with open(output_file, 'w') as out_f:
        for record in SeqIO.parse(input_file,"fasta"):
            protein_id = record.id
            protein_sequence = str(record.seq).upper().replace(' ','').replace('\n', '').replace('\r', '')
            if len(protein_sequence) > 1022:
                print(f"警告: 蛋白质序列 {protein_id} 长度({len(protein_sequence)})超过最大限制(1022)，跳过")
                continue
            print(f"\n处理蛋白质: {protein_id}")
            print(f"蛋白质序列: {protein_sequence}")

            try:
                predicted_cds, translated_protein = predict_cds_from_protein(
                    model=model,
                    cds_tokenizer=cds_tokenizer,
                    protein_tokenizer=protein_tokenizer,
                    protein_sequence=protein_sequence,
                    device=device
                )
                out_f.write(f">{protein_id}\n")
                out_f.write(f"{predicted_cds}\n")
                print(f"预测的CDS序列: {predicted_cds}")
                print(f"翻译后的蛋白质: {translated_protein}")
                print(f"与原始蛋白质匹配: {translated_protein.rstrip('*') == protein_sequence}")

            except Exception as e:
                print(f"处理蛋白质 {protein_id} 时出错: {str(e)}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description='预测蛋白质序列对应的CDS序列')
    parser.add_argument('--model_path', type=str,
                        required=True,
                        help='训练好的模型路径')
    parser.add_argument('--input_file', type=str,
                        required=True,
                        help='输入的FASTA文件路径')
    parser.add_argument('--output_file', type=str,
                        required=True,
                        help='输出的FASTA文件路径')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("加载分词器...")
    cds_tokenizer = RnaTokenizer.from_pretrained("multimolecule/mrnafm")
    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    print(f"从 {args.model_path} 加载模型...")
    config = AutoConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = CustomPlantRNAModelmlm(config).to(device)
    model.load_state_dict(load_file(f"{args.model_path}/model.safetensors"))
    model.eval()
    fusion_params = model.get_learned_parameters()
    print(f"RNA权重: {fusion_params['alpha']:.6f}")
    print(f"蛋白质权重: {fusion_params['beta']:.6f}")
    print(f"\n开始处理FASTA文件: {args.input_file}")
    process_protein_sequences(
        model=model,
        cds_tokenizer=cds_tokenizer,
        protein_tokenizer=protein_tokenizer,
        input_file=args.input_file,
        output_file=args.output_file,
        device=device
    )

    print(f"\n所有预测完成! 结果已保存到 {args.output_file}")


if __name__ == "__main__":
    main()
