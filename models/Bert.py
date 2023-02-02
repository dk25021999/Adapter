import torch.nn as nn
from transformers import BertModel
import torch
import numpy as np
from models.BertClassification import init_weights


# Custom Sigmoid Function designed to transform from sigmoid to uniform 1
class Bert(nn.Module):

    def __init__(self, model_name, num_classes, embed_size, target_attribute='gender', trainable_param='all',
                 new_embed=False, train_layer_norm=True, last_hidden=False, max_doc_length=75, num_gate_layers=1):
        super(Bert, self).__init__()

        self.first_run = True
        self.last_hidden = last_hidden
        self.max_doc_length = max_doc_length
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)
        self.target_attribute = target_attribute
        self.num_encoder_layers = len(self.bert.encoder.layer)
        self.tgt_gate = self.target_attribute.split(' ')
        self.num_gate_layers = num_gate_layers

        ln_gate = False
        if trainable_param == 'none':
            for param in self.bert.parameters():
                param.requires_grad = False
        elif trainable_param == 'only_output':
            for name, param in self.bert.named_parameters():
                if ('output' in name) and ('attention' not in name):
                    ln_gate = True
                    param.requires_grad = True
                elif (train_layer_norm and ln_gate) and ('LayerNorm' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif trainable_param == 'all':
            pass
        else:
            for name, param in self.bert.named_parameters():
                if self.condition_(trainable_param, name):
                    ln_gate = True
                    param.requires_grad = True
                elif (train_layer_norm and ln_gate) and ('LayerNorm' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if not last_hidden:
            self.FC = nn.Linear(embed_size[0], num_classes)
        else:
            self.FC = nn.Linear(self.max_doc_length * embed_size[0], num_classes)
            self.dropout = nn.Dropout(0.5)
        self.FC.apply(init_weights)

    def condition_(self, param, text):
        if '+' in param:
            a = param.split('+')
            b = []
            for t in a:
                b.append(t.split(' '))
        else:
            b = param.split(' ')
        if type(b[0]) == str:
            if all(word in text for word in b):
                return True
            else:
                return False
        if len(b) == 2:
            if all(word in text for word in b[0]) or all(word in text for word in b[1]):
                return True
            else:
                return False
        if len(b) == 3:
            if all(word in text for word in b[0]) or all(word in text for word in b[1]) or all(
                    word in text for word in b[2]):
                return True
            else:
                return False

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
