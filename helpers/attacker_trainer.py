from helpers.utilities import score_metric
from tqdm import tqdm
import torch
from helpers.utilities import dict_mean, attribute_gap, torch_accuracy
import numpy as np


def train_attacker(attacker_model, data_loader, loss_func, optim, device, target_attribute='gender age'):
    pred, target = [], []
    for attacker in attacker_model.values():
        attacker.train()
    attr_list = target_attribute.split(' ')
    attr_loss_ = dict(zip(attr_list, [[] for i in attr_list]))
    attr_preds = dict(zip(attr_list, [[] for i in attr_list]))
    attr_targets = dict(zip(attr_list, [[] for i in attr_list]))
    attr_accuracy = dict(zip(attr_list, [[] for i in attr_list]))
    tqdm_bar = tqdm(data_loader)
    for embeds, preds, labels, attributes in tqdm_bar:
        embeds = embeds.to(device)
        counter = 0
        for name, attacker in attacker_model.items():
            attributes[counter] = attributes[counter].to(device)
            pred_ = attacker(embeds)
            loss_ = loss_func(pred_, attributes[counter])
            attr_loss_[name].append(loss_.item())
            attr_loss_[name] = attr_loss_[name][-2000:]
            optim[name].zero_grad()
            loss_.backward()
            optim[name].step()

            attr_preds[name] += torch.argmax(pred_, dim=1).tolist()
            attr_preds[name] = attr_preds[name][-2000:]
            attr_targets[name] += attributes[counter].tolist()
            attr_targets[name] = attr_targets[name][-2000:]
            attr_accuracy[name] = score_metric(attr_preds[name], attr_targets[name])
            counter += 1
        pred += preds.tolist()
        pred = pred[-2000:]
        target += labels.tolist()
        target = target[-2000:]
        tqdm_bar.set_description(
            f"att t_loss: {dict_mean(attr_loss_)},"
            f"att t_score: {attr_accuracy},")

    attr_gap = attribute_gap(pred, target, attr_preds, attr_targets)
    return dict_mean(attr_loss_), attr_accuracy, attr_gap


def validate_attacker(attacker_model, data_loader, loss_func, device,
                      target_attribute='gender age', no_gap=False):

    pred, target = [], []
    attr_list = target_attribute.split(' ')
    attr_loss_ = dict(zip(attr_list, [[] for i in attr_list]))
    attr_preds = dict(zip(attr_list, [[] for i in attr_list]))
    attr_targets = dict(zip(attr_list, [[] for i in attr_list]))
    attr_accuracy = dict(zip(attr_list, [[] for i in attr_list]))
    attr_score= dict(zip(attr_list, [0 for i in attr_list]))
    tqdm_bar = tqdm(data_loader)

    b_count = 0
    with torch.no_grad():
        for embeds, preds, labels, attributes_labels in tqdm_bar:
            embeds = embeds.to(device)
            pred += preds.tolist()
            target += labels.tolist()
            # model_score = score_metric(pred, target)
            if b_count == 0:
                model_score = score_metric(preds, labels.tolist())
            else:
                new_score = score_metric(preds.tolist(), labels.tolist())
                model_score = {k: np.round((b_count * model_score.get(k, 0) + new_score.get(k, 0)) / (b_count + 1), 3)
                          for k in set(model_score) & set(new_score)}

            counter = 0
            for name, attacker in attacker_model.items():
                attacker.eval()
                pred_ = attacker(embeds)
                classes = torch.argmax(pred_, dim=1).tolist()
                attr_preds[name] += classes
                attr_targets[name] += attributes_labels[counter]
                # attr_accuracy[name] = score_metric(attr_preds[name], attr_targets[name])
                if b_count == 0:
                    attr_score[name] = score_metric(classes, attributes_labels[counter])
                else:
                    new_score = score_metric(classes, attributes_labels[counter])
                    attr_score[name] = {k: np.round((b_count * attr_score[name].get(k, 0) + new_score.get(k, 0)) / (b_count+1), 3)
                              for k in set(attr_score[name]) & set(new_score)}
                # attr_accuracy[name] = torch_accuracy(attr_preds[name], attr_targets[name])

                loss_ = loss_func(pred_, attributes_labels[counter].to(device))
                attr_loss_[name].append(loss_.item())
                counter += 1
            b_count+=1
            tqdm_bar.set_description(f"model_score: {model_score['acc']},"
                                     f"att v_score: {attr_score},")
    model_score = score_metric(pred, target)
    for name, attacker in attacker_model.items():
        attr_score[name] = score_metric(attr_preds[name], attr_targets[name])
    if no_gap:
        attr_gap = 1
    else:
        attr_gap = attribute_gap(pred, target, attr_preds, attr_targets)
    tqdm_bar.set_description(f"model_score: {model_score['acc']},"
                             f"att v_score: {attr_score},")
    return model_score, dict_mean(attr_loss_), attr_score, attr_gap
