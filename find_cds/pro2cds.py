import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import argparse
import concurrent.futures
from tqdm import tqdm
import os
import subprocess
import tempfile
import re


def setup_blast_db(cds_proteins, temp_dir):
    db_file = os.path.join(temp_dir,
                           "translated_cds.fasta")
    with open(db_file, "w") as f:
        for i, (cds, protein) in enumerate(
                cds_proteins):
            protein_length = len(protein)
            f.write(
                f">cds_{i}_{protein_length}\n{protein}\n")

    cmd = ["makeblastdb", "-in", db_file,
           "-dbtype", "prot"]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)
    return db_file


def run_blast(query_seq, db_file, temp_dir):
    query_file = os.path.join(temp_dir,
                              "query.fasta")
    os.makedirs(temp_dir, exist_ok=True) 
    with open(query_file, "w") as f:
        f.write(f">query\n{query_seq}\n")

    output_file = os.path.join(temp_dir,
                               "blast_result.txt")
    cmd = [
        "blastp",
        "-query", query_file,
        "-db", db_file,
        "-out", output_file,
        "-outfmt",
        "6 qseqid sseqid pident length", 
        "-max_target_seqs", "1"
    ]

    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)

        if os.path.exists(
                output_file) and os.path.getsize(
                output_file) > 0:
            with open(output_file, "r") as f:
                line = f.readline().strip()
                if line:
                    fields = line.split()
                    if len(fields) >= 4:
                        target_id = fields[1]
                        identity = float(
                            fields[2])
                        align_length = int(
                            fields[3])


                        match = re.search(
                            r"cds_(\d+)_(\d+)",
                            target_id)
                        if match:
                            idx = int(
                                match.group(1))
                            target_len = int(
                                match.group(2))
                            query_len = len(
                                query_seq)


                            coverage = (align_length / max(query_len,target_len)) * 100
                            return idx, identity, coverage
        return None, 0.0, 0.0
    except Exception as e:
        return None, 0.0, 0.0


def translate_cds(cds_seq):

    stop_codons = ['TAA', 'TAG', 'TGA']
    stop_pos = len(cds_seq)

    for i in range(0, len(cds_seq) - 2, 3):
        codon = cds_seq[i:i + 3]
        if codon in stop_codons:
            stop_pos = i + 3 
            break

    truncated_cds = cds_seq[:stop_pos]

    if len(truncated_cds) % 3 != 0:
        return None, None

    try:
        protein = str(
            Seq(truncated_cds).translate(
                to_stop=True))
        return truncated_cds, protein
    except Exception as e:
        return None, None


def find_best_cds_match_blast(protein_seq,
                              cds_proteins,
                              db_file, temp_dir,
                              threshold,
                              coverage_threshold):

    for i, (cds, translated) in enumerate(cds_proteins):
        if protein_seq == translated:
            return cds, translated, 100.0, 100.0

    idx, similarity, coverage = run_blast(protein_seq, db_file, temp_dir)

    if idx is not None and similarity >= threshold and coverage >= coverage_threshold:
        cds_seq, translated_seq = cds_proteins[idx]
        return cds_seq, translated_seq, similarity, coverage

    return None, None, 0.0, 0.0


def process_row(row_data, cds_proteins, db_file,
                temp_dir_base, threshold,
                coverage_threshold, worker_id):

    row_idx, row = row_data
    protein_seq = row['protein_sequence_ori']

    temp_dir = os.path.join(temp_dir_base, f"worker_{worker_id}")
    os.makedirs(temp_dir, exist_ok=True)

    best_cds, best_protein, similarity, coverage = find_best_cds_match_blast(
        protein_seq, cds_proteins, db_file,
        temp_dir, threshold, coverage_threshold
    )

    if similarity >= threshold and coverage >= coverage_threshold:
        row_dict = row.to_dict()
        row_dict.update({
            'cds_sequence': best_cds,
            'protein_sequence': best_protein,
            'similarity': similarity,
            'coverage': coverage
        })
        return row_dict
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mpb_csv')
    parser.add_argument('fasta_file')
    parser.add_argument('output_csv')
    parser.add_argument('--threshold', type=float,
                        default=90.0)
    parser.add_argument('--coverage', type=float,
                        default=50.0)
    parser.add_argument('--parallel', type=int,
                        default=4)
    parser.add_argument('--temp_dir')
    args = parser.parse_args()

    temp_dir = args.temp_dir or tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)

    cds_proteins = []
    for record in SeqIO.parse(args.fasta_file,
                              'fasta'):
        truncated_cds, translated = translate_cds(str(record.seq))
        if truncated_cds and translated:
            cds_proteins.append((truncated_cds, translated))

    db_file = setup_blast_db(cds_proteins,
                             temp_dir)

    mpb_df = pd.read_csv(args.mpb_csv, sep=',')
    if 'seq' in mpb_df.columns and 'protein_sequence_ori' not in mpb_df.columns:
        mpb_df.rename(columns={
            'seq': 'protein_sequence_ori'},
                      inplace=True)

    results = []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.parallel) as executor:
        futures = []
        for idx, row in mpb_df.iterrows():
            worker_id = idx % args.parallel
            future = executor.submit(
                process_row,
                (idx, row),
                cds_proteins,
                db_file,
                temp_dir, 
                args.threshold,
                args.coverage,
                worker_id 
            )
            futures.append(future)

        for future in tqdm(
                concurrent.futures.as_completed(
                        futures),
                total=len(futures)):
            if (
            result := future.result()) is not None:
                results.append(result)

    if results:
        pd.DataFrame(results).to_csv(
            args.output_csv, index=False)
    else:
        pd.DataFrame().to_csv(args.output_csv,index=False)

    if not args.temp_dir:
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
