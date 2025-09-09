import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import multimolecule
from multimolecule import RnaTokenizer, RnaFmModel
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.query_linear = nn.Linear(hidden_size,
                                      hidden_size)
        self.key_linear = nn.Linear(hidden_size,
                                    hidden_size)

    def forward(self, hidden_state,
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
            attention_mask != 0, attention_scores,
            torch.tensor(-1e9,
                         device=attention_scores.device))
        attention_weights = F.softmax(
            attention_scores, dim=-1)
        pooled_output = torch.matmul(
            attention_weights, hidden_state)
        return pooled_output.squeeze(1)


class CustomPlantRNAModel(nn.Module):
    def __init__(self, config):
        super(CustomPlantRNAModel,
              self).__init__()


        self.plantrna = RnaFmModel.from_pretrained(
            "multimolecule/mrnafm")
        self.esm2 = None
        self.esm2_model_name = "facebook/esm2_t33_650M_UR50D"
        self.model_specific_state_dict = None
        self.hidden_size = config.hidden_size if hasattr(
            config, 'hidden_size') else 1280
        self.pooler = AttentionPooling(
            self.hidden_size)
        self.cds_layernorm = nn.LayerNorm(
            self.hidden_size)
        self.protein_layernorm = nn.LayerNorm(
            self.hidden_size)
        self.resnet_layernorm = nn.LayerNorm(
            self.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.fusion_raw_weights = nn.Parameter(
            torch.tensor([0.0, 0.0]))
        self.fc1 = nn.Linear(self.hidden_size,
                             self.hidden_size // 2)
        self.fc2 = nn.Linear(
            self.hidden_size // 2,
            self.hidden_size // 4)
        self.fc3 = nn.Linear(
            self.hidden_size // 4, 1)
        self.aux_fc1 = nn.Linear(self.hidden_size,
                                 self.hidden_size // 2)
        self.aux_fc2 = nn.Linear(
            self.hidden_size // 2, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

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
                pre_computed_protein_embeddings=None,
                labels=None, **kwargs):

        device = cds_input_ids.device if cds_input_ids is not None else next(self.parameters()).device


        if pre_computed_protein_embeddings is not None:
            pro_last_hidden_state = pre_computed_protein_embeddings
        else:
            try:
                self._load_esm2(device=device)
                with torch.no_grad():
                    protein_outputs = self.esm2(
                        input_ids=protein_input_ids,
                        attention_mask=protein_attention_mask)
                    pro_last_hidden_state = protein_outputs.last_hidden_state
                    pro_last_hidden_state = self.protein_layernorm(
                        pro_last_hidden_state)
            finally:
                self._unload_esm2()


        cds_outputs = self.plantrna(
            input_ids=cds_input_ids,
            attention_mask=cds_attention_mask)
        cds_last_hidden_state = cds_outputs.last_hidden_state
        cds_last_hidden_state = self.cds_layernorm(
            cds_last_hidden_state)

        original_cds_features = cds_last_hidden_state.clone()

        weights = self.get_fusion_weights()
        alpha, beta = weights[0], weights[1]

        fused_hidden_state = (
                    alpha * cds_last_hidden_state + beta * pro_last_hidden_state)

        fused_hidden_state = self.resnet_layernorm(
            fused_hidden_state)
        fused_hidden_state = self.dropout(
            fused_hidden_state)
        fused_hidden_state = self.activation(
            fused_hidden_state)

        pooled_output = self.pooler(
            fused_hidden_state,
            cds_attention_mask)
        x = self.fc1(pooled_output)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc3(x)

        outputs = {"logits": logits}

        pooled_cds = self.pooler(
            original_cds_features,
            cds_attention_mask)
        aux_x = self.aux_fc1(pooled_cds)
        aux_x = self.activation(aux_x)
        aux_x = self.dropout(aux_x)
        aux_logits = self.aux_fc2(aux_x)

        if labels is not None:
            labels_float = labels.float()
            main_loss = self.loss_fn(
                logits.squeeze(-1), labels_float)
            aux_loss = self.loss_fn(
                aux_logits.squeeze(-1),
                labels_float)
            total_loss = main_loss + 1.0 * aux_loss
            outputs["loss"] = total_loss

        return outputs

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
