import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import multimolecule
from multimolecule import RnaTokenizer, RnaFmModel
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig

import random
class CustomPlantRNAModelmlm(nn.Module):
    def __init__(self, config):
        super(CustomPlantRNAModelmlm,
              self).__init__()

        self.plantrna = RnaFmModel.from_pretrained(
            "multimolecule/mrnafm")

        self.vocab_size = self.plantrna.config.vocab_size
        self.mask_token_id = self.plantrna.config.mask_token_id

        self.esm2 = None
        self.esm2_model_name = "facebook/esm2_t33_650M_UR50D"
        self.model_specific_state_dict = None

        self.hidden_size = 1280

        # Normalization and activation
        self.cds_layernorm = nn.LayerNorm(
            self.hidden_size)
        self.protein_layernorm = nn.LayerNorm(
            self.hidden_size)
        self.resnet_layernorm = nn.LayerNorm(
            self.hidden_size)

        self.fusion_raw_weights = nn.Parameter(torch.tensor([0.0, 0.0]))


        self.mlm_head = nn.Linear(self.hidden_size, self.vocab_size)
        self.mlm_loss_fn = nn.CrossEntropyLoss()

    def set_state_dict(self, state_dict: dict):

        if not isinstance(state_dict, dict):
            raise TypeError(
                "state_dict must be a dictionary.")

        print(
            "Setting model-specific state dictionary...")
        self.model_specific_state_dict = state_dict

        non_esm2_state_dict = {k: v for k, v in
                               state_dict.items()
                               if
                               not k.startswith(
                                   'esm2.')}

        missing_keys, unexpected_keys = self.load_state_dict(
            non_esm2_state_dict, strict=False)

        if missing_keys:
            print(
                f"Warning: Missing keys when loading non-ESM2 state_dict: {missing_keys}")
        if unexpected_keys:
            print(
                f"Warning: Unexpected keys when loading non-ESM2 state_dict: {unexpected_keys}")
        print(
            "Non-ESM2 weights have been loaded.")

    def _load_esm2(self, device="cuda"):

        if self.esm2 is not None:
            return

        print(
            "Dynamically loading ESM-2 for computation...")

        self.esm2 = AutoModel.from_pretrained(
            self.esm2_model_name)

        if self.model_specific_state_dict:

            esm2_state_dict = {
                k.replace('esm2.', '', 1): v
                for k, v in
                self.model_specific_state_dict.items()
                if k.startswith('esm2.')
            }


            if esm2_state_dict:
                print(
                    "Applying fine-tuned weights to ESM-2...")
                self.esm2.load_state_dict(
                    esm2_state_dict)
                print(
                    "Fine-tuned ESM-2 weights applied successfully.")
            else:
                print(
                    "No fine-tuned ESM-2 weights found in the provided state_dict. Using original pre-trained weights.")
        else:
            print(
                "No model-specific state_dict found. Using original pre-trained ESM-2 weights.")


        self.esm2.to(device)
        self.esm2.eval()

    def _unload_esm2(self):

        if self.esm2 is not None:
            print(
                "Unloading ESM-2 model from VRAM...")
            del self.esm2
            self.esm2 = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    def get_fusion_weights(self):
        return F.softmax(self.fusion_raw_weights,
                         dim=0)

    def forward(self, cds_input_ids=None,
                cds_attention_mask=None,
                protein_input_ids=None,
                protein_attention_mask=None,
                inputs_embeds=None,
                pre_computed_protein_embeddings=None,
                labels=None, **kwargs):
        outputs = {}
        if pre_computed_protein_embeddings is not None:
            pro_last_hidden_state = pre_computed_protein_embeddings
        else:
            protein_outputs = self.esm2(
                input_ids=protein_input_ids,
                attention_mask=protein_attention_mask)
            pro_last_hidden_state = protein_outputs.last_hidden_state
            pro_last_hidden_state = self.protein_layernorm(pro_last_hidden_state)
        cds_outputs = self.plantrna(input_ids=cds_input_ids,attention_mask=cds_attention_mask)
        cds_last_hidden_state = cds_outputs.last_hidden_state
        cds_last_hidden_state = self.cds_layernorm(
            cds_last_hidden_state)

        weights = self.get_fusion_weights()
        alpha, beta = weights[0], weights[1]
        resnet_out = (
                    alpha * cds_last_hidden_state + beta * pro_last_hidden_state)
        resnet_out = self.resnet_layernorm(resnet_out)


        # MLM prediction
        mlm_prediction_scores = self.mlm_head(resnet_out)

        outputs = {"logits": mlm_prediction_scores}

        # Calculate loss if labels are provided
        if labels is not None:
            active_loss = labels.view(-1) != -100
            active_logits = mlm_prediction_scores.view(-1,self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            if len(active_labels) > 0:
                mlm_loss = self.mlm_loss_fn(
                    active_logits, active_labels)
            else:
                mlm_loss = torch.tensor(0.0,device=resnet_out.device, requires_grad=True)

            outputs["loss"] = mlm_loss

        return outputs

    def get_learned_parameters(self):
        weights = self.get_fusion_weights()
        return {
            "alpha": weights[0].item(),
            "beta": weights[1].item()
        }

    def compute_protein_embeddings(self,
                                   protein_input_ids,
                                   protein_attention_mask):
        device = next(self.parameters()).device
        try:
            self._load_esm2(device=device)
            with torch.no_grad():
                protein_outputs = self.esm2(
                    input_ids=protein_input_ids.to(
                        device),
                    attention_mask=protein_attention_mask.to(
                        device)
                )
                pro_last_hidden_state = protein_outputs.last_hidden_state
                pro_last_hidden_state = self.protein_layernorm(
                    pro_last_hidden_state)
                return pro_last_hidden_state
        finally:
            self._unload_esm2()