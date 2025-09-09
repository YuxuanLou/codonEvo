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

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

1. Initialize CDS

   ```sh
   python initial_cds-batch.py \
   --model_path ./Ntabacum4097/Ntabacum4097-finetune-mrnafm-with-pro-top10csi2/Ntabacum4097-finetune-mrnafm-with-pro-csitop10 \
   --input_file input.fasta \
   --output_file initial_cds.fasta
   ```

   

2. Optimize CDS

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

<!-- LICENSE -->

## License

Distributed under the project_license. See `LICENSE.txt` for more information.



