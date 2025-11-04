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

        self.plantrna = RnaFmModel.from_pretrained("multimolecule/mrnafm")
        self.plantrna.to('cpu')
        self.vocab_size = self.plantrna.config.vocab_size
        self.mask_token_id = self.plantrna.config.mask_token_id

        self.esm2_config = AutoConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self._esm2_initialized = False

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

    @property
    def esm2(self):
        """属性访问器，延迟初始化ESM2"""
        if not hasattr(self, '_esm2_instance'):
            self._esm2_instance = AutoModel.from_config(
                self.esm2_config)
            self._esm2_initialized = True
        return self._esm2_instance

    def load_state_dict(self, state_dict,strict=True):
        own_state = self.state_dict()
        esm2_state_dict = {}
        other_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith('esm2.'):
                esm2_state_dict[name] = param
            else:
                other_state_dict[name] = param
        super().load_state_dict(other_state_dict,strict=False)
        if esm2_state_dict:
            _ = self.esm2
            converted_esm2_state_dict = {}
            for key, value in esm2_state_dict.items():
                converted_esm2_state_dict[
                    key[5:]] = value
            self.esm2.load_state_dict(
                converted_esm2_state_dict,
                strict=False)
        return self

    def _unload_esm2(self):

        if self.esm2 is not None:
            print(
                "Unloading ESM-2 model from VRAM...")
            del self._esm2_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_fusion_weights(self):
        return F.softmax(self.fusion_raw_weights,dim=0)

    def forward(self, cds_input_ids=None,
                cds_attention_mask=None,
                protein_input_ids=None,
                protein_attention_mask=None,
                inputs_embeds=None,
                pre_computed_protein_embeddings=None,
                labels=None, **kwargs):
        outputs = {}
        pro_last_hidden_state = pre_computed_protein_embeddings
        cds_outputs = self.plantrna(input_ids=cds_input_ids,attention_mask=cds_attention_mask)
        cds_last_hidden_state = cds_outputs.last_hidden_state
        cds_last_hidden_state = self.cds_layernorm(cds_last_hidden_state)
        weights = self.get_fusion_weights()
        alpha, beta = weights[0], weights[1]
        resnet_out = (alpha * cds_last_hidden_state + beta * pro_last_hidden_state)
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
            #if device.type == 'cuda':
                #self.plantrna.to('cpu')
        return outputs

    def get_learned_parameters(self):
        weights = self.get_fusion_weights()
        return {
            "alpha": weights[0].item(),
            "beta": weights[1].item()
        }

    def compute_protein_embeddings(self,protein_input_ids,protein_attention_mask):
        device = next(self.parameters()).device
        with torch.no_grad():
            protein_outputs = self.esm2(input_ids=protein_input_ids.to(device),attention_mask=protein_attention_mask.to(device))
            pro_last_hidden_state = protein_outputs.last_hidden_state
            pro_last_hidden_state = self.protein_layernorm(pro_last_hidden_state)
            self._unload_esm2()
            return pro_last_hidden_state
