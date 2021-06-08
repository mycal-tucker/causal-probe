import copy
from abc import ABC

import torch.nn as nn

# Create a model that copies the last layers of a transformer model. This enables us to intervene or insert new
# embeddings within the model and see what the output would be.
# Core functionality is in TailTransformer (the tail end of the model), with specific classes for different types
# of models (e.g., cloze vs. QA)


class TailTransformer(nn.Module, ABC):
    def __init__(self, qa_model, layer_idx):
        super().__init__()
        # Taken from: https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertModel
        self.transformer = copy.deepcopy(qa_model.base_model.encoder)
        self.layers = nn.ModuleList(self.transformer.layer[layer_idx:])
        self.transformer.layer = self.layers

    def forward(self, x):
        hidden_inputs = {'hidden_states': x}
        outputs = self.transformer(**hidden_inputs)
        return outputs


# Model the last few layers of a pre-trained Transformer-based Question-Answering model.
# Just leverages the tail end of the transformer plus the linear layer at the end, which is used for logits over
# the start and end tokens.
class QATail(nn.Module):
    def __init__(self, qa_model, layer_idx):
        super().__init__()
        self.transformer = TailTransformer(qa_model, layer_idx)
        self.last_layer = qa_model.qa_outputs

    def forward(self, x):
        transformer_output = self.transformer(x)[0]
        logits = self.last_layer(transformer_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)
        return start_logits, end_logits

# Model for cloze task, which predicts a missing word.
class ClozeTail(nn.Module):
    def __init__(self, cloze_model, layer_idx):
        super(ClozeTail, self).__init__()
        self.transformer = TailTransformer(cloze_model, layer_idx)
        self.last_layer = cloze_model.cls

    def forward(self, x):
        transformer_output = self.transformer(x)[0]
        logits = self.last_layer(transformer_output)
        return logits