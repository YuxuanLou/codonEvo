# codonEvo: A Codon Optimization Tool Based on Pretrained Language Models



<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#Installation">Installation</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

CodonEvo, through its two major modules CodonNAT and CodonEXP, learns the codon usage preferences of endogenous proteins within the host species and the latent features of highly expressed CDS sequences. It guides the codon optimization of heterologous proteins in the host species for “high naturalness” and “high expression levels” using a genetic algorithm.



## Installation

Our CUDA version is 12.2.

1. Create a conda environment

   ```sh
   conda create -n codonEvo python=3.10
   ```

3. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```







<!-- USAGE EXAMPLES -->

## Usage

1. Initialize CDS

   ```sh
   python initial_cds-batch.py \
   --model_path ./Ntabacum4097/Ntabacum4097-finetune-mrnafm-with-pro-top10csi2/Ntabacum4097-finetune-mrnafm-with-pro-csitop10 \
   --input_file input.fasta \
   --output_file initial_cds.fasta
   ```

   

2. Optimize CDS with codonEvo

   ```sh
   python naturality-aux-evo-batch.py \ 
   --model_dir ./Ntabacum4097/Ntabacum4097-aux1-2-classify \
   --population_size 100 \
   --mutation_rate 5 \
   --crossover_rate 0.7 \
   --max_generations 100 \
   --batch_size 50 \
   --selection_top_percent 0.2 \
   --attention_weights_save_path ./Ntabacum4097/20-pro-100/attention_weights \
   --results ./Ntabacum4097/20-pro-100/results \
   --history ./Ntabacum4097/20-pro-100/history \
   --perplexity_weight 1 \
   --perplexity_model_dir ./Ntabacum4097/Ntabacum4097-finetune-mrnafm-with-pro-top10csi2/Ntabacum4097-finetune-mrnafm-with-pro-csitop10 \
   --input ./Ntabacum4097/cds_list.fasta \
   --output ./Ntabacum4097/cds_codonEvo_list-100.fasta
   ```

The model weights and detailed explanations of the parameters will be made public after the manuscript is submitted or published.

3.  Optimize CDS with codonHallucination

   ```sh
   python CodonHallucination.py --model_dir ./Ntabacum4097/Ntabacum4097-aux1-2-classify \
   --mutation_rate 0.15 \
   --iterations 16 \
   --max_iterations 96 \
   --min_expression_threshold 0.9 \
   --min_naturality_threshold 0.6 \
   --batch_size 16 \
   --top_n 1 \
   --results_dir ./Ntabacum4097/20-pro-100/results \
   --perplexity_weight 1 \
   --hallucination_perplexity_weight 1 \
   --patience 20 \
   --perplexity_model_dir ./Ntabacum4097/Ntabacum4097-finetune-mrnafm-with-pro-top10csi2/Ntabacum4097-finetune-mrnafm-with-pro-csitop10 \
   --input ./Ntabacum4097/cds_list.fasta \
   --output ./Ntabacum4097/cds_codonHallucination_test.txt \
   --use_reversibility_check
   ```

<!-- LICENSE -->

## License

Distributed under the project_license. See `LICENSE.txt` for more information.



