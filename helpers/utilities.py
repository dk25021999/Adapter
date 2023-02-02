import torch
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score


def score_metric(prediction: list, target: list):
    prediction = np.array(prediction)
    target = np.array(target)
    unique_target = np.unique(target)
    unique_pred = np.unique(prediction)
    if max(unique_target) > 1 or max(unique_pred) > 1:
        method = 'micro'
    else:
        method = 'binary'
    accuracy = accuracy_score(target, prediction)
    b_accuracy = balanced_accuracy_score(target, prediction)
    prec = precision_score(target, prediction, average=method)
    recall = recall_score(target, prediction, average=method)
    f1 = f1_score(target, prediction, average=method)
    dict_ = {'acc': np.round(accuracy, 3), 'bacc': np.round(b_accuracy, 3), 'prec': np.round(prec, 3),
             'rec': np.round(recall, 3), 'f1': np.round(f1, 3)}
    return dict_


def torch_accuracy(predictions: list, target: list):
    pred = torch.tensor(predictions)
    tgt = torch.tensor(target)
    correct = torch.sum(pred == tgt)
    total = len(target)
    accuracy = torch.round(correct/total, decimals=3)
    dict_ = {'acc': accuracy.item()}
    return dict_


def dict_mean(dictionary):
    mean_dict = dict(zip(dictionary.keys(), [None] * len(dictionary)))
    for name, item in dictionary.items():
        mean_dict[name] = mean_round(item)
    return mean_dict


def mean_round(some_list):
    if type(sum(some_list)) == torch.tensor:
        print(some_list)
        return np.round(sum(some_list).cpu() / len(some_list), 3)
    else:
        return np.round(sum(some_list) / len(some_list), 3)


def attribute_gap(task_pred: list, task_target: list, attribute_pred: dict, attribute_target: dict):
    keys = attribute_pred.keys()
    task_pred = np.array(task_pred)
    task_target = np.array(task_target)
    attr_gap = dict(zip(keys, [0 for _ in keys]))
    for key in keys:
        TP = np.sum((np.array(attribute_pred[key]) == np.array(attribute_target[key])) & (task_pred == task_target))+1
        N = np.sum((task_pred == task_target))+1
        attr_gap[key] = np.round(TP / N, 3)
        for attr_unique in np.unique(attribute_target[key]):
            for task_unique in np.unique(task_target):
                TP = np.sum(
                    (np.array(attribute_pred[key]) == attr_unique) & (np.array(attribute_target[key]) == attr_unique)
                    & (task_pred == task_unique) & (task_target == task_unique))
                N = np.sum((np.array(attribute_target[key]) == attr_unique) &
                        (task_pred == task_unique) & (task_target == task_unique))
                attr_gap[f"{key}={attr_unique}|label={task_unique}"] = np.round(TP / N, 3)
    return attr_gap


def save_checkpoint(path, model):
    torch.save(model.state_dict(), path)


def get_accuracy(prediction, target):
    correct = [True for i, j in zip(prediction, target) if i == j]
    return len(correct) / len(target)


def dict_to_device(data, device='cuda'):
    for key, value in data.items():
        data[key] = value.to(device)
    return data


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def dict_to_class(dictionary):
    modified_dict = dictionary.copy()
    for key, value in dictionary.items():
        if type(value) == dict and key != 'debias':
            modified_dict[key] = Dict2Class(value)
    modified_dict = Dict2Class(modified_dict)
    return modified_dict


def load_checkpoint(path, map_location='cuda:0', attacker_weight=False):
    import yaml
    from models.BertClassification import CustomBert
    from models.BertGate import BertGateV1
    from models.Bert import Bert
    from models.BertGateV2 import BertGateV2
    from models.Adapter_Bert import AdapterBert
    from models.attribute_classifier import AttackerModel
    with open(os.path.join(path, 'config.yml'), 'r') as f:
        config_ = yaml.unsafe_load(f)
    config = dict_to_class(config_)


    if 'v0' in path:

        if 'adapter' in path:

            model = AdapterBert(config.model.model_name, config.model.num_labels, config.model.embed_size,
                               target_attribute=config.run.target_attribute,
                               max_doc_length=config.model.max_doc_length, reduction_factor=2)
        else:
            model = Bert(config.model.model_name, config.model.num_labels, config.model.embed_size,
                                trainable_param=config.model.trainable_parameters,
                                new_embed=config.model.new_embed_layer, train_layer_norm=config.model.train_layer_norm,
                                last_hidden=config.model.capture_last_hidden,
                                max_doc_length=config.model.max_doc_length)
    elif 'v1' in path:
        model = BertGateV1(config.model.model_name, config.model.num_labels, config.model.embed_size,
                           target_attribute=config.run.target_attribute,
                           trainable_param=config.model.trainable_parameters, new_embed=config.model.new_embed_layer,
                           train_layer_norm=config.model.train_layer_norm, last_hidden=config.model.capture_last_hidden,
                           max_doc_length=config.model.max_doc_length)
    elif 'v2' in path:
        model = BertGateV2(config.model.model_name, config.model.num_labels, config.model.embed_size,
                           target_attribute=config.run.target_attribute,
                           trainable_param=config.model.trainable_parameters, new_embed=config.model.new_embed_layer,
                           train_layer_norm=config.model.train_layer_norm, last_hidden=config.model.capture_last_hidden,
                           max_doc_length=config.model.max_doc_length)

    model_weights = torch.load(os.path.join(path, f'model.pt'), map_location=map_location)
    model.load_state_dict(model_weights)
    attackers = dict(zip(config.dataset.dataset_attributes, [None] * len(config.dataset.dataset_attributes)))
    att_optimizer = dict(zip(config.dataset.dataset_attributes, [None] * len(config.dataset.dataset_attributes)))
    for name, unique in zip(config.dataset.dataset_attributes, config.dataset.attributes_unique):
        attackers[name] = AttackerModel(config.model.embed_size[-1], config.run.attacker_hidden_layer,
                                        num_attributes=unique,
                                        activation_function='ReLU')
        att_optimizer[name] = torch.optim.AdamW(attackers[name].parameters(), lr=config.run.attr_lr / 2)
        if attacker_weight:
            attack_weights = torch.load(os.path.join(path, f'attack_{name}.pt'), map_location=map_location)
            attackers[name].load_state_dict(attack_weights)
    return model, attackers, config










