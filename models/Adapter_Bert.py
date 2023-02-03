import torch.nn as nn
import torch
import numpy as np
from models.BertClassification import init_weights
from transformers import BertModelWithHeads, BertTokenizerFast, BertConfig
from transformers.adapters import PfeifferConfig


# This Version works with
class AdapterBert(nn.Module):

    def __init__(self, model_name, num_classes, embed_size, fusion = False, num_adapters=1, target_attribute='gender',
                 max_doc_length=75, num_gate_layers=1,
                 reduction_factor=2):
        super(AdapterBert, self).__init__()

        config = BertConfig.from_pretrained(
            model_name,
        )
        self.bert  = BertModelWithHeads.from_pretrained(
            model_name,
            config=config,
        )

        adap_config = PfeifferConfig(reduction_factor=reduction_factor)
        self.bert.add_adapter(target_attribute, adap_config)
        self.bert.train_adapter(target_attribute, adap_config)

        self.first_run = True
        self.max_doc_length = max_doc_length
        self.target_attribute = target_attribute
        self.tgt_gate = self.target_attribute.split(' ')
        self.num_gate_layers = num_gate_layers

        self.FC = nn.Linear(embed_size[0], num_classes)
        self.FC.apply(init_weights)

    def param_spec(self):
        num_param = []
        trainable = []
        frozen = []
        for param in self.parameters():
            if param.requires_grad == True:
                trainable.append(len(param.reshape(-1)))
            else:
                frozen.append(len(param.reshape(-1)))
            num_param.append(len(param.reshape(-1)))
            percentage = np.round(sum(trainable) / sum(num_param) * 100, 1)
        return sum(num_param), sum(trainable), sum(frozen), percentage

    def forward(self, input_ids, token_type_ids, attention_mask, return_embeds=False,
                active_gates='gender age', temp=[1]):
        da_embeds = []
        x = self.bert.forward(input_ids, attention_mask).pooler_output
        da_embeds.append(x.view(x.shape[0], -1))
        out = self.FC(x)
        self.first_run = False
        if return_embeds:
            return out, da_embeds
        else:
            return out
