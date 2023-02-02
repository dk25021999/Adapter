from abc import ABC

from tqdm import tqdm
import torch
from helpers.utilities import score_metric
import numpy as np
import wandb as wand_b
from dataset.dataset import embed_dataloader
from helpers.utilities import dict_mean, mean_round, attribute_gap
from helpers.attacker_trainer import train_attacker, validate_attacker


def gate_train(model, attr_models, debias_method, data_loader, loss_func, optimizers,
               device, target_attribute='gender'):
    model.train()
    loss_list, attr_loss_list = [], []
    pred, target = [], []
    tqdm_bar = tqdm(data_loader)
    tgt_attribute = target_attribute.split(' ')
    attr_loss_ = dict(zip(tgt_attribute, [[] for i in tgt_attribute]))
    for ids, token_types, att_masks, labels, attributes in tqdm_bar:
        ids, token_types, att_masks, labels, = ids.to(device), token_types.to(device), \
                                               att_masks.to(device), labels.to(device)
        for i, attribute in enumerate(attributes):
            attributes[i] = attribute.to(device)
        if debias_method == 'dann':
            attributes = [attributes[0], attributes[1]]
        else:
            attributes = [attributes[0], attributes[2]]
        if (len(tgt_attribute) == 2) or ('multi' in tgt_attribute):
            pass
        elif 'gender' in tgt_attribute:
            attributes = [attributes[0]]
        elif 'age' in tgt_attribute:
            attributes = [attributes[1]]
        logits, embeds = model(ids, token_type_ids=token_types, attention_mask=att_masks, return_embeds=True)
        task_loss = loss_func(logits, labels)
        task_loss.backward(retain_graph=True)
        param_grad = {}
        for n, param in model.named_parameters():
            if (('gate' not in n) or ('task' in n)) and (param.requires_grad == True):
                try:
                    param_grad[n] = param.grad.clone()
                except:
                    pass

        if attr_models is not None:
            for count, packet in enumerate(zip(attributes, tgt_attribute)):
                attribute = packet[0]
                name = packet[1]
                if debias_method != 'dann':
                    source_embeds_, source_preds_ = embeds[-1][attribute == 0], logits[attribute == 0]
                    target_embeds_ = embeds[-1][attribute == 1]
                    source_embeds_ = source_embeds_[:min(len(source_embeds_), len(target_embeds_))]
                    target_embeds_ = target_embeds_[:min(len(source_embeds_), len(target_embeds_))]
                    attr_loss, attr_info = attr_models[name].get_da_loss([
                        torch.cat((source_embeds_, target_embeds_))],
                        torch.cat((torch.tensor([0] * len(source_embeds_), device=device),
                                   torch.tensor([1] * len(source_embeds_), device=device))))
                else:
                    # if name == 'age':
                    #     weight = torch.FloatTensor([2.75, 0.59,  0.49, 1.11, 20.88]).to(device)
                    #     attr_loss, attr_info = attr_models[name].get_da_loss([embeds[-1]], attribute, weight)
                    # else:
                    attr_loss, attr_info = attr_models[name].get_da_loss([embeds[-1]], attribute)

                attr_loss.backward(retain_graph=True)
                for n, param in model.named_parameters():
                    if name in n:
                        try:
                            param_grad[n] = param.grad.clone()
                        except:
                            pass
                attr_loss_[name].append(attr_loss.item())
            for n, param in model.named_parameters():
                if n in param_grad.keys():
                    param.grad = param_grad[n]

            for optim in optimizers.values():
                optim.step()
                optim.zero_grad()

                # print(attr_models[name].da_net.nets[-1][-1].linear.weight)
        else:
            optimizers['task'].step()
            optimizers['task'].zero_grad()
            attr_loss_[tgt_attribute[0]].append(0.)
        loss_list.append(task_loss.item())
        pred += torch.argmax(logits, dim=1).tolist()
        target += labels.tolist()
        pred = pred[-5000:]
        target = target[-5000:]
        score_ = score_metric(pred, target)
        tqdm_bar.set_description(
            f"model t_loss: {mean_round(loss_list[-len(loss_list) // 10:])},"
            # f"model t_acc: {np.round(pred_accuracy, 3)}, "
            f"model scores: {score_}, "
            f"attribute t_loss: {dict_mean(attr_loss_)},"
        )
        tqdm_bar.update()
    #         break
    return sum(loss_list) / len(loss_list), dict_mean(attr_loss_), score_


def validation(model, attr_models, debias_method, data_loader,
               loss_func, device, target_attribute='gender', active_gate='gender age'):
    model.eval()
    loss_list, attr_loss_list = [], []
    pred, target = [], []
    tqdm_bar = tqdm(data_loader)
    tgt_attribute = target_attribute.split(' ')
    attr_loss_ = dict(zip(tgt_attribute, [[] for i in tgt_attribute]))
    with torch.no_grad():
        for ids, token_types, att_masks, labels, attributes in tqdm_bar:
            ids, token_types, att_masks, labels, = ids.to(device), token_types.to(device), \
                                                   att_masks.to(device), labels.to(device)

            for i, attribute in enumerate(attributes):
                attributes[i] = attribute.to(device)
            if debias_method == 'dann':
                attributes = [attributes[0], attributes[1]]
            else:
                attributes = [attributes[0], attributes[2]]
            if (len(tgt_attribute) == 2) or ('multi' in tgt_attribute):
                pass
            elif 'gender' in tgt_attribute:
                attributes = [attributes[0]]
            elif 'age' in tgt_attribute:
                attributes = [attributes[1]]

            logits, embeds = model(ids, token_type_ids=token_types, attention_mask=att_masks, return_embeds=True)
            # logits = model(ids, token_type_ids=token_types, attention_mask=att_masks) # No embeds

            total_attr_loss = 0
            if attr_models is not None:
                for count, packet in enumerate(zip(attributes, tgt_attribute)):
                    attribute = packet[0]
                    name = packet[1]
                    if not debias_method == 'dann':
                        source_embeds_, source_preds_ = embeds[-1][attribute == 0], logits[attribute == 0]
                        target_embeds_ = embeds[-1][attribute == 1]
                        source_embeds_ = source_embeds_[:min(len(source_embeds_), len(target_embeds_))]
                        target_embeds_ = target_embeds_[:min(len(source_embeds_), len(target_embeds_))]
                        attr_loss, attr_info = attr_models[name].get_da_loss([
                            torch.cat((source_embeds_, target_embeds_))],
                            torch.cat((torch.tensor([0] * len(source_embeds_), device=device),
                                       torch.tensor([1] * len(source_embeds_), device=device))))
                        attr_loss_[name].append(attr_loss.item())
                    else:
                        attr_loss, attr_info = attr_models[name].get_da_loss([embeds[-1]],
                                                                             attribute)
                        attr_loss_[name].append(attr_loss.item())
                        # print(attr_models[name].da_net.nets[-1][-1].linear.weight)
            else:
                attr_loss_[tgt_attribute[0]].append(0.)
            task_loss = loss_func(logits, labels)
            loss_list.append(task_loss.item())
            pred += torch.argmax(logits, dim=1).tolist()
            target += labels.tolist()
            score_ = score_metric(pred, target)
            tqdm_bar.set_description(
                f"model v_loss: {mean_round(loss_list)}, "
                # f"model v_acc: {np.round(pred_accuracy, 3)}, "
                f"model v_score: {score_}, "
                f"attribute v_loss: {dict_mean(attr_loss_)},")
            tqdm_bar.update()
    return sum(loss_list) / len(loss_list), dict_mean(attr_loss_), score_


def gate_train_model(model, attr_models, attacker_model, debias_method, train_loader, validate_loader,
                     test_loader, train_epochs, attack_epochs, loss_func, optimizers, attacker_optim, lr_scheduler,
                     device, attacker_attribute='gender age', target_attribute='gender', wandb=None,
                     train_type='model attacker'):
    t_loss_list, t_attr_loss_list, t_acc_list = [], [], []
    v_loss_list, v_attr_loss_list, v_acc_list = [], [], []
    tgt_attribute = target_attribute.split(' ')
    train_type = train_type.split(' ')

    if 'model' in train_type:
        for i in range(train_epochs):
            print(f'Epoch:  {i + 1}')
            train_loss, train_attr_loss, train_acc = gate_train(model, attr_models, debias_method, train_loader,
                                                                loss_func, optimizers, device, target_attribute)
            t_loss_list.append(train_loss)
            t_attr_loss_list.append(train_attr_loss)
            t_acc_list.append(train_acc)

            val_loss, val_attr_loss, val_acc = validation(model, attr_models, debias_method, validate_loader,
                                                          loss_func, device, target_attribute)
            v_loss_list.append(val_loss)
            v_attr_loss_list.append(val_attr_loss)
            v_acc_list.append(val_acc)

            if 'none' not in target_attribute:
                if 'gender' in target_attribute:
                    current_attr_lr = optimizers['gender'].param_groups[0]['lr']
                else:
                    current_attr_lr = optimizers['age'].param_groups[0]['lr']
                current_task_lr = optimizers['task'].param_groups[0]['lr']
            else:
                current_task_lr = optimizers['task'].param_groups[0]['lr']
                current_attr_lr = 0

            if lr_scheduler is not None:
                for sched in lr_scheduler.values():
                    sched.step()
            log_dict = {
                "epochs": i + 1,
                "task lr": current_task_lr,
                "attr lr": current_attr_lr,
                "debias lambda": attr_models[tgt_attribute[0]].get_current_lambda() if
                attr_models else 0.0,
                "train loss": train_loss,
                "validation loss": val_loss,
            }
            train_score = dict(zip([f"train task {key}" for key in train_acc.keys()], list(train_acc.values())))
            val_score = dict(zip([f"validation task {key}" for key in val_acc.keys()], list(val_acc.values())))
            log_dict.update(train_score)
            log_dict.update(val_score)
            if wandb:
                wandb.log(log_dict)
                for key in attacker_attribute.split(' '):
                    try:
                        attribute_train_log = train_attr_loss[key]
                        attribute_validation_log = val_attr_loss[key]
                    except:
                        attribute_train_log = 0
                        attribute_validation_log = 0
                    wandb.log({"epochs": i + 1,
                               f"train {key} loss": attribute_train_log,
                               f"validation {key} loss": attribute_validation_log})

    if 'attacker' in train_type:
        embed_trainloader = embed_dataloader(model, train_loader, device, debias_method,
                                             attacker_attribute, active_gates='gender age')
        embed_valloader = embed_dataloader(model, validate_loader, device, debias_method,
                                           attacker_attribute, active_gates='gender age')
        for i in range(attack_epochs):
            print(f'Attacker Epoch:  {i + 1}')
            train_attribute_loss, train_attribute_accuracy, train_attr_gap = train_attacker(attacker_model,
                                                                                            embed_trainloader,
                                                                                            loss_func,
                                                                                            attacker_optim,
                                                                                            device,
                                                                                            target_attribute=attacker_attribute, )
            val_model_score, val_attribute_loss, val_attribute_accuracy, val_attr_gap = validate_attacker(attacker_model,
                                                                                         embed_valloader,
                                                                                         loss_func,
                                                                                         device,
                                                                                         target_attribute=attacker_attribute)

            if wandb:
                for key in attacker_attribute.split(' '):
                    wandb.log({f"attacker train {key} loss": train_attribute_loss[key],
                               f"attacker validation {key} loss ": val_attribute_loss[key],
                               f"attacker train {key} acc": train_attribute_accuracy[key]['acc'],
                               f"attacker validation {key} accuracy": val_attribute_accuracy[key]['acc'],
                               f"attacker validation {key} prec": val_attribute_accuracy[key]['prec'],
                               f"attacker validation {key} rec": val_attribute_accuracy[key]['rec'],
                               f"attacker validation {key} f1": val_attribute_accuracy[key]['f1'],
                               "epochs": i + 1})

        test_loss, test_attr_loss, test_acc = validation(model, attr_models, debias_method, test_loader, loss_func,
                                                         device, target_attribute)
        test_model_score, test_attribute_loss, test_attribute_accuracy, test_attr_gap = validate_attacker(attacker_model,
                                                                                        embed_valloader,
                                                                                        loss_func, device,
                                                                                        target_attribute=attacker_attribute)

        if wandb:
            table_ = wand_b.Table(columns=list(train_attr_gap.keys()))
            table_.add_data(*list(train_attr_gap.values()))
            table_.add_data(*list(val_attr_gap.values()))
            table_.add_data(*list(test_attr_gap.values()))
            test_log = {"test loss": test_loss,
                        "conditional_accuracy": table_,
                        "epochs": i + 1}
            test_score = dict(zip([f"test task {key}" for key in test_acc.keys()], list(test_acc.values())))
            test_log.update(test_score)
            wandb.log(test_log)

            for key in attacker_attribute.split(' '):
                try:
                    attribute_test_log = test_attr_loss[key]
                except:
                    attribute_test_log = 0
                wandb.log({f"test {key} loss": attribute_test_log,
                           f"attacker test {key} loss": test_attribute_loss[key],
                           f"attacker test {key} accuracy": test_attribute_accuracy[key]['acc'],
                           f"attacker test {key} prec": test_attribute_accuracy[key]['prec'],
                           f"attacker test {key} rec": test_attribute_accuracy[key]['rec'],
                           f"attacker test {key} f1": test_attribute_accuracy[key]['f1'],
                           "epochs": i + 1, })

    if (not 'attacker' in train_type) and (not 'model' in train_type):
        print('Train type should be either "model" "attacker" or "model attacker"')
        exit()

    return t_loss_list, t_attr_loss_list, t_acc_list, v_loss_list, v_attr_loss_list, v_acc_list, test_loss, \
           test_attr_loss, test_acc
#
#
# def train_attacker(attacker_model, data_loader, loss_func, optim, device, target_attribute='gender age'):
#     pred, target = [], []
#     for attacker in attacker_model.values():
#         attacker.train()
#     attr_list = target_attribute.split(' ')
#     attr_loss_ = dict(zip(attr_list, [[] for i in attr_list]))
#     attr_preds = dict(zip(attr_list, [[] for i in attr_list]))
#     attr_targets = dict(zip(attr_list, [[] for i in attr_list]))
#     attr_accuracy = dict(zip(attr_list, [[] for i in attr_list]))
#     tqdm_bar = tqdm(data_loader)
#
#     for embeds, preds, labels, attributes in tqdm_bar:
#         embeds = embeds.to(device)
#         # for i, attribute in enumerate(attributes):
#         #     attributes[i] = attribute.to(device)
#         counter = 0
#         for name, attacker in attacker_model.items():
#             attributes[counter] = attributes[counter].to(device)
#             pred_ = attacker(embeds)
#             loss_ = loss_func(pred_, attributes[counter])
#             attr_loss_[name].append(loss_.item())
#             attr_loss_[name] = attr_loss_[name][-5000:]
#             optim[name].zero_grad()
#             loss_.backward()
#             optim[name].step()
#
#             attr_preds[name] += torch.argmax(pred_, dim=1).tolist()
#             attr_preds[name] = attr_preds[name][-5000:]
#             attr_targets[name] += attributes[counter].tolist()
#             attr_targets[name] = attr_targets[name][-5000:]
#             attr_accuracy[name] = score_metric(attr_preds[name], attr_targets[name])
#             counter += 1
#         pred += preds.tolist()
#         target += labels.tolist()
#         tqdm_bar.set_description(
#             f"att t_loss: {dict_mean(attr_loss_)},"
#             f"att t_score: {attr_accuracy},")
#         # tqdm_bar.update()
#
#     attr_gap = attribute_gap(pred[-5000:], target[-5000:], attr_preds, attr_targets)
#     #         break
#     return dict_mean(attr_loss_), attr_accuracy, attr_gap
#
#
# def validate_attacker(attacker_model, data_loader, loss_func, device,
#                       target_attribute='gender age'):
#
#     pred, target = [], []
#     attr_list = target_attribute.split(' ')
#     attr_loss_ = dict(zip(attr_list, [[] for i in attr_list]))
#     attr_preds = dict(zip(attr_list, [[] for i in attr_list]))
#     attr_targets = dict(zip(attr_list, [[] for i in attr_list]))
#     attr_accuracy = dict(zip(attr_list, [[] for i in attr_list]))
#     tqdm_bar = tqdm(data_loader)
#     with torch.no_grad():
#         for embeds, preds, labels, attributes_labels in tqdm_bar:
#             embeds = embeds.to(device)
#             # for i, attribute in enumerate(attributes):
#             #     attributes[i] = attribute.to(device)
#             pred += preds.tolist()
#             target += labels.tolist()
#             model_score = score_metric(pred, target)
#             counter = 0
#             for name, attacker in attacker_model.items():
#                 attacker.eval()
#                 pred_ = attacker(embeds)
#                 attr_preds[name] += torch.argmax(pred_, dim=1).tolist()
#                 attr_targets[name] += attributes_labels[counter].tolist()
#                 attr_accuracy[name] = score_metric(attr_preds[name], attr_targets[name])
#                 loss_ = loss_func(pred_, attributes_labels[counter].to(device))
#                 attr_loss_[name].append(loss_.item())
#                 counter += 1
#
#             tqdm_bar.set_description(f"model_score: {model_score['acc']},"
#                                      # f"att v_loss: {dict_mean(attr_loss_)},"
#                                      f"att v_score: {attr_accuracy},")
#             # tqdm_bar.update()
#
#     attr_gap = attribute_gap(pred, target, attr_preds, attr_targets)
#     #         break
#     return model_score, dict_mean(attr_loss_), attr_accuracy, attr_gap


# def model_prediction(model, data_loader, device):
#     model.eval()
#     pred, target = [], []
#     tqdm_bar = tqdm(data_loader)
#     with torch.no_grad():
#         for ids, token_types, att_masks, labels, _ in tqdm_bar:
#             ids, token_types, att_masks, labels = ids.to(device), token_types.to(device), att_masks.to(
#                 device), labels.to(device)
#             logits = model(ids, attention_mask=att_masks, token_type_ids=token_types, return_embeds=False)
#             pred += (torch.argmax(logits, dim=1).tolist())
#             target += (labels.tolist())
#         return pred, target




