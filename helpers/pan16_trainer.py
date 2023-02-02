from tqdm import tqdm
import torch
from helpers.utilities import get_accuracy
import numpy as np
import wandb as wand_b


def train(model, attr_models, debias_method, data_loader, loss_func, optim,
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
        # logits = model(ids, token_type_ids=token_types, attention_mask=att_masks) # No embeds
        total_attr_loss = torch.tensor(0.).to(device)
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
                        torch.cat((torch.tensor([0]*len(source_embeds_), device=device),
                                   torch.tensor([1]*len(source_embeds_), device=device))))
                    attr_loss_[name].append(attr_loss.item())
                    total_attr_loss += attr_loss
                else:
                    attr_loss, attr_info = attr_models[name].get_da_loss([embeds[-1]], attribute)
                    attr_loss_[name].append(attr_loss.item())
                    total_attr_loss += attr_loss

                # print(attr_models[name].da_net.nets[-1][-1].linear.weight)
        else:
            attr_loss_[tgt_attribute[0]].append(0.)
            total_attr_loss = torch.tensor(0.).to(device)
        optim.zero_grad()
        task_loss = loss_func(logits, labels)
        total_loss = task_loss + total_attr_loss
        total_loss.backward()
        optim.step()
        loss_list.append(task_loss.item())
        pred += torch.argmax(logits, dim=1).tolist()
        target += labels.tolist()
        pred_accuracy = get_accuracy(pred, target)
        tqdm_bar.set_description(
            f"model t_loss: {mean_round(loss_list)}, "
            f"model t_acc: {np.round(pred_accuracy, 3)}, "
            f"attribute t_loss: {dict_mean(attr_loss_)},")
        tqdm_bar.update()
    #         break
    return sum(loss_list) / len(loss_list), dict_mean(attr_loss_), pred_accuracy


def validation(model, attr_models, debias_method, data_loader,
               loss_func, device, target_attribute='gender'):
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
            pred_accuracy = get_accuracy(pred, target)
            tqdm_bar.set_description(
                f"model v_loss: {mean_round(loss_list)}, "
                f"model v_acc: {np.round(pred_accuracy, 3)}, "
                f"attribute v_loss: {dict_mean(attr_loss_)},")
            tqdm_bar.update()
    return sum(loss_list) / len(loss_list), dict_mean(attr_loss_), pred_accuracy


def train_model(model, attr_models, attacker_model, debias_method, train_loader, validate_loader,
                test_loader, train_epochs, attack_epochs, loss_func, optim, attacker_optim, lr_scheduler, device,
                attacker_attribute='gender age', target_attribute='gender', wandb=None, train_type='model attacker'):
    t_loss_list, t_attr_loss_list, t_acc_list = [], [], []
    v_loss_list, v_attr_loss_list, v_acc_list = [], [], []
    tgt_attribute = target_attribute.split(' ')
    train_type = train_type.split(' ')

    if 'model' in train_type:
        for i in range(train_epochs):
            print(f'Epoch:  {i + 1}')
            train_loss, train_attr_loss, train_acc = train(model, attr_models, debias_method, train_loader,
                                                           loss_func, optim, device, target_attribute)
            t_loss_list.append(train_loss)
            t_attr_loss_list.append(train_attr_loss)
            t_acc_list.append(train_acc)
            val_loss, val_attr_loss, val_acc = validation(model, attr_models, debias_method, validate_loader,
                                                          loss_func, device, target_attribute)
            v_loss_list.append(val_loss)
            v_attr_loss_list.append(val_attr_loss)
            v_acc_list.append(val_acc)

            if 'none' not in target_attribute:
                current_attr_lr = optim.param_groups[0]['lr']
                current_task_lr = optim.param_groups[1]['lr']
            else:
                current_task_lr = optim.param_groups[0]['lr']
                current_attr_lr = 0

            if lr_scheduler is not None:
                lr_scheduler.step()
            if wandb:
                wandb.log({
                        "epochs": i+1,
                        "task lr": current_task_lr,
                        "attr lr": current_attr_lr,
                        "debias lambda": attr_models[tgt_attribute[0]].get_current_lambda() if
                        attr_models else 0.0,
                        "train loss": train_loss,
                        "train accuracy": train_acc,
                        "validation loss": val_loss,
                        "validation accuracy": val_acc,
                           })
                for key in attacker_attribute.split(' '):
                    try:
                        attribute_train_log = train_attr_loss[key]
                        attribute_validation_log = val_attr_loss[key]
                    except:
                        attribute_train_log = 0
                        attribute_validation_log = 0
                    wandb.log({"epochs": i+1,
                               f"train {key} loss": attribute_train_log,
                               f"validation {key} loss": attribute_validation_log})

    if 'attacker' in train_type:
        for i in range(attack_epochs):
            print(f'Attacker Epoch:  {i + 1}')
            _, train_attribute_loss, _, train_attribute_accuracy, train_attr_gap = train_attacker(model, attacker_model,
                                                                                                  train_loader,
                                                                                                  loss_func,
                                                                                                  attacker_optim,
                                                                                                  device,
                                                                                  target_attribute=attacker_attribute,)
            _, val_attribute_loss, _, val_attribute_accuracy, val_attr_gap = validate_attacker(model, attacker_model,
                                                                                               validate_loader,
                                                                                               loss_func, device,
                                                                                    target_attribute=attacker_attribute)

            if wandb:
                for key in attacker_attribute.split(' '):
                    wandb.log({f"attacker train {key} loss": train_attribute_loss[key],
                               f"attacker validation {key} loss ": val_attribute_loss[key],
                               f"attacker train {key} accuracy": train_attribute_accuracy[key],
                               f"attacker validation {key} accuracy": val_attribute_accuracy[key],
                               "epochs": i+1})

        test_loss, test_attr_loss, test_acc = validation(model, attr_models, debias_method, test_loader, loss_func,
                                                         device, target_attribute)
        _, test_attribute_loss, _, test_attribute_accuracy, test_attr_gap = validate_attacker(model, attacker_model,
                                                                                              validate_loader,
                                                                                              loss_func, device,
                                                                               target_attribute=attacker_attribute)

        if wandb:
            table_ = wand_b.Table(columns=list(train_attr_gap.keys()))
            table_.add_data(*list(train_attr_gap.values()))
            table_.add_data(*list(val_attr_gap.values()))
            table_.add_data(*list(test_attr_gap.values()))

            wandb.log({"test loss": test_loss,
                       "test accuracy": test_acc,
                       "conditional_accuracy": table_,
                       "epochs": i+1})
            for key in attacker_attribute.split(' '):
                try:
                    attribute_test_log = test_attr_loss[key]
                except:
                    attribute_test_log = 0
                wandb.log({f"test {key} loss": attribute_test_log,
                           f"attacker test {key} loss": test_attribute_loss[key],
                           f"attacker test {key} accuracy": test_attribute_accuracy[key],
                           "epochs": i+1,})

    if (not 'attacker' in train_type) and (not 'model' in train_type):
        print('Train type should be either "model" "attacker" or "model attacker"')
        exit()

    return t_loss_list, t_attr_loss_list, t_acc_list, v_loss_list, v_attr_loss_list, v_acc_list, test_loss, \
           test_attr_loss, test_acc


def train_attacker(model, attacker_model, data_loader, loss_func, optim, device, target_attribute='gender age'):
    model.eval()
    loss_list = []
    pred, target = [], []
    attr_list = target_attribute.split(' ')
    attr_loss_ = dict(zip(attr_list, [[] for i in attr_list]))
    attr_preds = dict(zip(attr_list, [[] for i in attr_list]))
    attr_targets = dict(zip(attr_list, [[] for i in attr_list]))
    attr_accuracy = dict(zip(attr_list, [[] for i in attr_list]))
    tqdm_bar = tqdm(data_loader)
    for ids, token_types, att_masks, labels, attributes in tqdm_bar:
        ids, token_types, att_masks, labels, = ids.to(device), token_types.to(device), \
                                               att_masks.to(device), labels.to(device)

        attributes = attributes[:-1]  # make sure original attributes are used for attacker AGE and GENDER
        for i, attribute in enumerate(attributes):
            attributes[i] = attribute.to(device)


        with torch.no_grad():
            logits, embeds = model(ids, token_type_ids=token_types, attention_mask=att_masks, return_embeds=True)
            task_loss = loss_func(logits, labels)
            loss_list.append(task_loss.item())
            pred += torch.argmax(logits, dim=1).tolist()
            target += labels.tolist()
            task_accuracy = get_accuracy(pred, target)

        counter = 0
        for name, attacker in attacker_model.items():
            attacker.train()
            pred_ = attacker(embeds[-1])
            attr_preds[name] += torch.argmax(pred_, dim=1).tolist()
            attr_targets[name] += attributes[counter].tolist()
            attr_accuracy[name].append(get_accuracy(attr_preds[name], attr_targets[name]))
            loss_ = loss_func(pred_, attributes[counter])
            attr_loss_[name].append(loss_.item())
            optim[name].zero_grad()
            loss_.backward()
            optim[name].step()
            counter += 1
        tqdm_bar.set_description(
            f"model t_loss: {mean_round(loss_list)}, "
            f"model t_acc: {np.round(task_accuracy, 3)}, "
            f"attacker t_loss: {dict_mean(attr_loss_)},"
            f"attacker t_acc: {dict_mean(attr_accuracy)},")
        tqdm_bar.update()
    attr_gap = attribute_gap(pred, target, attr_preds, attr_targets)
    #         break
    return sum(loss_list) / len(loss_list), dict_mean(attr_loss_), task_accuracy, dict_mean(attr_accuracy), attr_gap


def validate_attacker(model, attacker_model, data_loader, loss_func, device, target_attribute='gender age'):
    model.eval()
    loss_list = []
    pred, target = [], []
    attr_list = target_attribute.split(' ')
    attr_loss_ = dict(zip(attr_list, [[] for i in attr_list]))
    attr_preds = dict(zip(attr_list, [[] for i in attr_list]))
    attr_targets = dict(zip(attr_list, [[] for i in attr_list]))
    attr_accuracy = dict(zip(attr_list, [[] for i in attr_list]))
    tqdm_bar = tqdm(data_loader)
    with torch.no_grad():
        for ids, token_types, att_masks, labels, attributes in tqdm_bar:
            ids, token_types, att_masks, labels, = ids.to(device), token_types.to(device), \
                                                   att_masks.to(device), labels.to(device)
            attributes = attributes[:-1]
            for i, attribute in enumerate(attributes):
                attributes[i] = attribute.to(device)

            logits, embeds = model(ids, token_type_ids=token_types, attention_mask=att_masks, return_embeds=True)
            task_loss = loss_func(logits, labels)
            loss_list.append(task_loss.item())
            pred += torch.argmax(logits, dim=1).tolist()
            target += labels.tolist()
            task_accuracy = get_accuracy(pred, target)

            counter = 0
            for name, attacker in attacker_model.items():
                attacker.eval()
                pred_ = attacker(embeds[-1])
                attr_preds[name] += torch.argmax(pred_, dim=1).tolist()
                attr_targets[name] += attributes[counter].tolist()
                attr_accuracy[name].append(get_accuracy(attr_preds[name], attr_targets[name]))
                loss_ = loss_func(pred_, attributes[counter])
                attr_loss_[name].append(loss_.item())
                counter += 1

            tqdm_bar.set_description(
                f"model v_loss: {mean_round(loss_list)}, "
                f"model v_acc: {np.round(task_accuracy, 3)}, "
                f"attacker v_loss: {dict_mean(attr_loss_)},"
                f"attacker v_acc: {dict_mean(attr_accuracy)},")
            tqdm_bar.update()
    attr_gap = attribute_gap(pred, target, attr_preds, attr_targets)
    #         break
    return sum(loss_list) / len(loss_list), dict_mean(attr_loss_), task_accuracy, dict_mean(attr_accuracy), attr_gap


def model_prediction(model, data_loader, device):
    model.eval()
    pred, target = [], []
    tqdm_bar = tqdm(data_loader)
    with torch.no_grad():
        for ids, token_types, att_masks, labels, _ in tqdm_bar:
            ids, token_types, att_masks, labels = ids.to(device), token_types.to(device), att_masks.to(
                device), labels.to(device)
            logits = model(ids, attention_mask=att_masks, token_type_ids=token_types, return_embeds=False)
            pred += (torch.argmax(logits, dim=1).tolist())
            target += (labels.tolist())
        return pred, target


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
        TP = sum((np.array(attribute_pred[key]) == np.array(attribute_target[key])) & (task_pred == task_target))
        N = sum((task_pred == task_target))
        attr_gap[key] = np.round(TP/N, 3)
        for attr_unique in np.unique(attribute_target[key]):
            for task_unique in np.unique(task_target):
                TP = sum((np.array(attribute_pred[key]) == attr_unique) & (np.array(attribute_target[key]) == attr_unique)
                         & (task_pred == task_unique) & (task_target == task_unique))
                N = sum((np.array(attribute_target[key]) == attr_unique) &
                        (task_pred == task_unique) & (task_target == task_unique))
                attr_gap[f"{key}={attr_unique}|label={task_unique}"] = np.round(TP/N, 3)
    return attr_gap

