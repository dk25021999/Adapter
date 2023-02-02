import torch.nn as nn
from transformers import BertModel
import torch
import numpy as np
from models.BertClassification import init_weights


# Custom Sigmoid Function designed to transform from sigmoid to uniform 1
class CustomSigmoid(nn.Module):
    def __init__(self):
        super(CustomSigmoid, self).__init__()
    def forward(self, x, temp):
        scale = temp + 1 if temp <= 1 else 2
        out = scale / (1 + torch.exp(-x / (10 ** temp)))
        out[out > 1] = 1
        return out

# Gating Mechanism Can be Linear or none linear transform
class GateLayer(nn.Module):
    def __init__(self, embed_size, num_layers=1):
        super(GateLayer, self).__init__()
        self.nonlinear = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            if i < (num_layers-1):
                self.nonlinear.append(nn.Sequential(nn.Linear(embed_size, embed_size), nn.Tanh()))
            else:
                self.nonlinear.append(nn.Linear(embed_size, embed_size))

        self.activation = CustomSigmoid()

    def forward(self, embed, temp):
        for i in range(self.num_layers):
            embed = self.nonlinear[i](embed)
        
        return self.activation(embed, temp)


#  Version Zero of the Bert gate (Not Functional)
class BertGate(nn.Module):

    def __init__(self, model_name, num_classes,  embed_size, target_attribute='gender',  trainable_param='all',
                 new_embed=False, train_layer_norm=True, last_hidden=False, max_doc_length=75):
        super(BertGate, self).__init__()

        self.first_run = True
        self.last_hidden = last_hidden
        self.max_doc_length = max_doc_length
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)
        self.target_attribute = target_attribute
        if model_name == 'bert-base-uncased':
            num_embeddings = 768
        elif model_name == "google/bert_uncased_L-4_H-256_A-4":
            num_embeddings = 256
        else:
            print('model initialization might be wrong')
            num_embeddings = 256
        ln_gate = False
        if 'gender' in target_attribute:
            self.gender_gate = nn.ModuleList()
            for i in range(len(embed_size)):
                self.gender_gate.append(GateLayer(num_embeddings))
        if 'age' in target_attribute:
            self.age_gate = nn.ModuleList()
            for i in range(len(embed_size)):
                self.age_gate.append(GateLayer(num_embeddings))

        if new_embed:
            self.embed_bias = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        else:
            self.embed_bias = None

        if trainable_param == 'None':
            for param in self.bert.parameters():
                param.requires_grad = False
        elif trainable_param == 'only output':
            for name, param in self.bert.named_parameters():
                if ('output' in name) and not ('attention' in name):
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
            percentage = np.round(sum(trainable) / sum(num_param) * 100, 1)
        return sum(num_param), sum(trainable), sum(frozen), percentage

    def forward_gate(self, x, layer, active_gates='gender age'):
        # x = x.reshape(x.size(0), -1)
        x_gender = torch.ones_like(x)
        x_age = torch.ones_like(x)
        if 'gender' in active_gates:
            if 'gender' in self.target_attribute:
                x_gender = self.gender_gate[layer](x)
        if 'age' in active_gates:
            if 'age' in self.target_attribute:
                x_age = self.age_gate[layer](x)
        return x_gender, x_age

    def forward(self, input_ids, token_type_ids, attention_mask, return_embeds=False, active_gates='gender age'):
        da_embeds = []
        embed = self.bert.embeddings(input_ids, attention_mask)
        #         embed_biased = embed + self.embed_bias
        if self.embed_bias:
            embed = self.embed_bias(embed)

        x = self.bert.encoder.layer[0](embed)[0]
        da_embeds.append(x.reshape(x.shape[0], -1))
        if self.first_run: print('encoder0 embed shape:', x.reshape(x.shape[0], -1).shape)
        x_gender, x_age = self.forward_gate(x, 0, active_gates=active_gates)
        x = (x*x_gender + x*x_age)/2

        x = self.bert.encoder.layer[1](x)[0]
        da_embeds.append(x.reshape(x.shape[0], -1))

        x_gender, x_age = self.forward_gate(x, 1, active_gates=active_gates)
        x = (x*x_gender + x*x_age)/2
        if self.first_run: print('encoder1 embed shape:', x.reshape(x.shape[0], -1).shape)

        x = self.bert.encoder.layer[2](x)[0]
        da_embeds.append(x.view(x.shape[0], -1))
        x_gender, x_age = self.forward_gate(x, 2, active_gates=active_gates)
        x = (x*x_gender + x*x_age)/2
        if self.first_run: print('encoder2 embed shape:', x.reshape(x.shape[0], -1).shape)

        x = self.bert.encoder.layer[3](x)[0]
        da_embeds.append(x.view(x.shape[0], -1))
        x_gender, x_age = self.forward_gate(x, 3, active_gates=active_gates)
        x = (x*x_gender + x*x_age)/2
        if not self.last_hidden:
            x = self.bert.pooler(x)
        else:
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
        da_embeds.append(x.view(x.shape[0], -1))
        x_gender, x_age = self.forward_gate(x, 4, active_gates=active_gates)
        x = (x*x_gender + x*x_age)/2
        if self.first_run: print('pooler embed shape:', x.view(x.shape[0], -1).shape)
        out = self.FC(x)
        self.first_run = False
        if return_embeds:
            return out, da_embeds
        else:
            return out


# This Version works with
class BertGateV1(nn.Module):

    def __init__(self, model_name, num_classes,  embed_size, target_attribute='gender',  trainable_param='all',
                 new_embed=False, train_layer_norm=True, last_hidden=False, max_doc_length=75, num_gate_layers=1):
        super(BertGateV1, self).__init__()

        self.first_run = True
        self.last_hidden = last_hidden
        self.max_doc_length = max_doc_length
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)
        self.target_attribute = target_attribute
        self.num_encoder_layers = len(self.bert.encoder.layer)
        self.tgt_gate = self.target_attribute.split(' ')
        self.num_gate_layers = num_gate_layers

        ln_gate = False
        if 'none' not in self.target_attribute:
            self.gate = nn.ModuleDict()
            for name in self.tgt_gate:
                self.gate[name] = nn.ModuleList()
                for i in range(self.num_encoder_layers+1):
                    self.gate[name].append(GateLayer(embed_size[0], num_layers=self.num_gate_layers))

        if new_embed:
            self.embed_bias = nn.Sequential(nn.Linear(embed_size[0], embed_size[0]), nn.ReLU())
        else:
            self.embed_bias = None

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

    def forward_gate(self, x, temp, layer, active_gates='gender age'):
        x_gate = []
        counter = 0
        for debias_gate in self.tgt_gate:
            if 'none' not in self.tgt_gate:
                if debias_gate in active_gates:
                    # print(f'{debias_gate} gate active')
                    x_gate.append(self.gate[debias_gate][layer](x, temp[counter]))
                    counter += 1
                else:
                    # print(f'{debias_gate} gate inactive')
                    x_gate.append(self.gate[debias_gate][layer](x, 4))
            else:
                # print('gate doesnt exist')
                x_gate.append(torch.ones_like(x))

        if len(x_gate) > 1:
            x_gate = [x_gate[i]*x_gate[i-1] for i in range(1, len(x_gate))][0]
        else:
            x_gate = x_gate[0]
        return x_gate

    def forward(self, input_ids, token_type_ids, attention_mask, return_embeds=False,
                active_gates='gender age', temp=[0]):
        da_embeds = []
        x = self.bert.embeddings(input_ids, attention_mask)
        if self.embed_bias:
            x = self.embed_bias(x)
        for i in range(self.num_encoder_layers):
            x = self.bert.encoder.layer[i](x)[0]
            if self.first_run: print(f'encoder{i} embed shape:', x.shape)
            # x_gender, x_age = self.forward_gate(x, temp, i, active_gates=active_gates)
            # x = x * x_gender * x_age
            x_gate = self.forward_gate(x, temp, i, active_gates=active_gates)
            x = x * x_gate
        if not self.last_hidden:
            x = self.bert.pooler(x)
        else:
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
        if self.first_run: print('pooler embed shape:', x.view(x.shape[0], -1).shape)
        x_gate = self.forward_gate(x, temp, -1, active_gates=active_gates)
        x = x * x_gate
        da_embeds.append(x.view(x.shape[0], -1))
        out = self.FC(x)
        self.first_run = False
        if return_embeds:
            return out, da_embeds
        else:
            return out






