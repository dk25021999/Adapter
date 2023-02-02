import torch.nn as nn
from transformers import BertModel
import torch
import numpy as np


class OrgBert(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', freeze_bert=False, last_hidden=False, max_doc_length=75):
        super(OrgBert, self).__init__()

        self.last_hidden = last_hidden
        self.max_doc_length = max_doc_length
        self.bert = BertModel.from_pretrained(bert_model, return_dict=True)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output

        return hook

    def forward(self, input_ids, token_type_ids, attention_mask, return_layers=False):
        if not self.last_hidden:
            x = self.bert(input_ids, attention_mask=attention_mask).pooler_output

        else:
            x = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        x = x.view(x.size(0), -1)
        if return_layers:
            return self.selected_output, x
        else:
            return x


class BertClassification(nn.Module):
    def __init__(self, num_classes, bert_model='bert-base-uncased',
                 freeze_bert=False, last_hidden=False, max_doc_length=75):
        super(BertClassification, self).__init__()
        self.bert = OrgBert(bert_model, freeze_bert, last_hidden, max_doc_length)

        if not self.bert.last_hidden:
            self.FC = nn.Linear(768, num_classes)
        else:
            self.FC = nn.Linear(self.max_doc_length * 768, num_classes)
            # self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        out = self.FC(x)
        return out


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class CustomBert(nn.Module):
    def __init__(self, model_name, num_classes, trainable_param='all', new_embed=False,
                 train_layer_norm=True, last_hidden=False, max_doc_length=75):
        super(CustomBert, self).__init__()

        self.first_run = True
        self.last_hidden = last_hidden
        self.max_doc_length = max_doc_length
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)

        # Ex. Map the Embeddings to a new embedding space and check how it does.
        #         self.embed_bias = nn.Parameter(torch.ones(256),requires_grad=True)
        if new_embed:
            self.embed_bias = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        else:
            self.embed_bias = None
        # Custom Trainable Parameter selection for the Bert,
        # Select the name of layers up to three layers can be handled
        if trainable_param == 'None':
            for param in self.bert.parameters():
                param.requires_grad = False
        elif trainable_param == 'only output':
            for name, param in self.bert.named_parameters():
                if ('output' in name) and not ('attention' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif trainable_param == 'all':
            pass
        else:
            for name, param in self.bert.named_parameters():
                if self.condition_(trainable_param, name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if train_layer_norm:
            for name, param in self.bert.named_parameters():
                if 'LayerNorm' in name:
                    param.requires_grad = True

        if model_name == 'bert-base-uncased':
            num_embeddings = 768
        elif model_name == "google/bert_uncased_L-4_H-256_A-4":
            num_embeddings = 256
        else:
            print('model not initialization might be wrong')
            num_embeddings = 256

        if not last_hidden:
            self.FC = nn.Linear(num_embeddings, num_classes)
        else:
            self.FC = nn.Linear(self.max_doc_length * num_embeddings, num_classes)
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
        return sum(num_param), sum(trainable), sum(frozen), np.round(sum(trainable) / sum(num_param) * 100, 1)

    def forward(self, input_ids, token_type_ids, attention_mask, return_embeds=False):
        da_embeds = []
        embed = self.bert.embeddings(input_ids, attention_mask)
        #         embed_biased = embed + self.embed_bias
        if self.embed_bias:
            embed = self.embed_bias(embed)

        x = self.bert.encoder.layer[0](embed)
        da_embeds.append(x[0].reshape(x[0].shape[0], -1))
        if self.first_run: print('encoder0 embed shape:', x[0].reshape(x[0].shape[0], -1).shape)

        x = self.bert.encoder.layer[1](x[0])
        da_embeds.append(x[0].reshape(x[0].shape[0], -1))
        if self.first_run: print('encoder1 embed shape:', x[0].reshape(x[0].shape[0], -1).shape)

        x = self.bert.encoder.layer[2](x[0])
        da_embeds.append(x[0].view(x[0].shape[0], -1))
        if self.first_run: print('encoder2 embed shape:', x[0].reshape(x[0].shape[0], -1).shape)

        x = self.bert.encoder.layer[3](x[0])
        if not self.last_hidden:
            x = self.bert.pooler(x[0])
        else:
            x = x[0].view(x[0].size(0), -1)
            x = self.dropout(x)
        da_embeds.append(x.view(x.shape[0], -1))
        if self.first_run: print('encoder3 embed shape:', x.view(x.shape[0], -1).shape)
        out = self.FC(x)
        self.first_run = False
        if return_embeds:
            return out, da_embeds
        else:
            return out





