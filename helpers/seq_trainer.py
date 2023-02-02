from tqdm import tqdm
import torch
from helpers.utilities import score_metric
import numpy as np
import wandb as wand_b
from dataset.dataset import embed_dataloader
from helpers.utilities import dict_mean, mean_round, attribute_gap
from helpers.attacker_trainer import train_attacker, validate_attacker


def task_train(model, data_loader, loss_func, optimizers, device, target_attribute='gender', gate_version=1):

    model.train()
    loss_list, attr_loss_lis = 0, 0
    tqdm_bar = tqdm(data_loader)
    tgt_attribute = target_attribute.split(' ')
    b_count = 0
    for ids, token_types, att_masks, labels, attributes in tqdm_bar:
        ids, token_types, att_masks, labels, = ids.to(device), token_types.to(device), \
                                               att_masks.to(device), labels.to(device)
        # logits, embeds = model(ids, token_type_ids=token_types, attention_mask=att_masks,
        #                        return_embeds=True, active_gates='none', temp=[4])
        temp_values = 4 if gate_version == 1 else 0
        logits, embeds = model(ids, token_type_ids=token_types, attention_mask=att_masks,
                               return_embeds=True, active_gates='none', temp=[temp_values])
        task_loss = loss_func(logits, labels)
        task_loss.backward()
        optimizers['task'].step()
        optimizers['task'].zero_grad()

        loss_list = np.round((loss_list + task_loss.item()) / 2, 3)
        pred = torch.argmax(logits, dim=1).tolist()
        if b_count == 0:
            score_ = score_metric(pred, labels.tolist())
        else:
            new_score = score_metric(pred, labels.tolist())
            score_ = {k: np.round((19*score_.get(k, 0) + new_score.get(k, 0)) / 20, 3)
                      for k in set(score_) & set(new_score)}
        tqdm_bar.set_description(
            f"model t_loss: {loss_list},"
            f"model t_score: {score_},")
        b_count += 1
    return loss_list, score_


def adv_train(model, attr_models, debias_method, data_loader, loss_func, optimizers,
              device, target_attribute='gender', active_gates='gender age', gate_version=1):
    model.train()
    loss_list = 0
    pred, target = [], []
    tqdm_bar = tqdm(data_loader)
    tgt_attribute = target_attribute.split(' ')
    attr_loss_ = dict(zip(tgt_attribute, [0 for _ in tgt_attribute]))

    b_count = 0
    for ids, token_types, att_masks, labels, attributes in tqdm_bar:
        ids, token_types, att_masks, labels, = ids.to(device), token_types.to(device), \
                                               att_masks.to(device), labels.to(device)
        if len(tgt_attribute) == 2:
            if debias_method == 'dann':
                attributes = [attributes[0], attributes[1]]
            else:
                attributes = [attributes[0], attributes[2]]
        elif 'gender' in tgt_attribute or 'dialect' in tgt_attribute:
            attributes = [attributes[0]]
        elif 'age' in tgt_attribute:
            attributes = [attributes[1]]

        for count, packet in enumerate(zip(attributes, tgt_attribute)):
            attribute = packet[0].to(device)
            name = packet[1]
            temp_values = [4, 0] if gate_version == 1 else [0, 1]
            temperature = [temp_values[0]]*len(tgt_attribute)
            temperature[count] = temp_values[1]
            # Active gates are from the attribute and the other play no role in the activation of certain gate
            # temperature is 0 to make sure that normal sigmoid is used.
            for param_name, param in model.named_parameters():
                if name in param_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            logits, embeds = model(ids, token_type_ids=token_types, attention_mask=att_masks,
                                   return_embeds=True, active_gates=name, temp=temperature)
            # for n_, p_ in model.named_parameters():
            #     if n_ == 'gate.dialect.4.nonlinear':
            #         print(p_)

            task_loss = loss_func(logits, labels)
            task_loss.backward(retain_graph=True)
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
                    attr_loss, attr_info = attr_models[name].get_da_loss([embeds[-1]], attribute)

            attr_loss.backward()
            optimizers[name].step()
            optimizers[name].zero_grad()
            attr_loss_[name] = np.round((19*attr_loss_[name] + attr_loss.item()) / 20, 3)

        loss_list = np.round((19*loss_list + task_loss.item()) / 20, 3)
        pred = torch.argmax(logits, dim=1).tolist()
        if b_count == 0:
            score_ = score_metric(pred, labels.tolist())
        else:
            new_score = score_metric(pred, labels.tolist())
            score_ = {k: np.round((19*score_.get(k, 0) + new_score.get(k, 0)) / 20, 3)
                      for k in set(score_) & set(new_score)}
        tqdm_bar.set_description(
            f"debiased model t_loss: {loss_list},"
            f"debiased model t_score: {score_},"
            f"debiased attribute t_loss: {attr_loss_},")
        b_count += 1
    return loss_list, attr_loss_, score_


def validation(model, attr_models, debias_method, data_loader,
               loss_func, device, target_attribute='gender', active_gates='gender age', gate_version=1):
    model.eval()
    loss_list, attr_loss_list = [], []
    pred, target = [], []

    tqdm_bar = tqdm(data_loader)
    tgt_attribute = target_attribute.split(' ')

    ac_g = active_gates.split(' ')
    attr_loss_ = dict(zip(tgt_attribute, [[] for i in tgt_attribute]))

    with torch.no_grad():
        for ids, token_types, att_masks, labels, attributes in tqdm_bar:
            ids, token_types, att_masks, labels, = ids.to(device), token_types.to(device), \
                                                   att_masks.to(device), labels.to(device)

            if len(tgt_attribute) > 1:
                if debias_method == 'dann':
                    attributes = [attributes[0], attributes[1]]
                else:
                    attributes = [attributes[0], attributes[2]]
            else:
                attributes = [attributes[0]]
            temp_values = 0 if gate_version == 1 else 1
            temperature = [temp_values]*len(ac_g)
            logits, embeds = model(ids, token_type_ids=token_types, attention_mask=att_masks,
                                   return_embeds=True, active_gates=active_gates, temp=temperature)

            if attr_models is not None:
                for count, packet in enumerate(zip(attributes, tgt_attribute)):
                    attribute = packet[0].to(device)
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
                    else:
                        attr_loss, attr_info = attr_models[name].get_da_loss(embeds, attribute)

                    attr_loss_[name].append(attr_loss.item())

            else:
                attr_loss_[tgt_attribute[0]].append(0.)

            task_loss = loss_func(logits, labels)
            loss_list.append(task_loss.item())
            pred += torch.argmax(logits, dim=1).tolist()
            target += labels.tolist()
            score_ = score_metric(pred, target)
            tqdm_bar.set_description(
                f"model v_loss: {mean_round(loss_list)}, "
                f"model v_score: {score_},"
                f"attribute v_loss: {dict_mean(attr_loss_)},")
    return sum(loss_list) / len(loss_list), dict_mean(attr_loss_), score_


def seq_train_model(model, attr_models, attacker_model, debias_method, train_loader, validate_loader,
                    test_loader, train_epochs, attack_epochs, loss_func, optimizers, attacker_optim, lr_scheduler,
                    device, attacker_attribute='gender age', target_attribute='gender', wandb=None,
                    train_type='model attacker', gate_version=1):
    t_loss_list, t_attr_loss_list, t_acc_list = [], [], []
    v_loss_list, v_attr_loss_list, v_acc_list = [], [], []
    train_type = train_type.split(' ')
    tgt_attribute = target_attribute.split(' ')
    # print('\n', 'Trainable Parameters During Training the Task:')
    if 'model' in train_type:
        for name, param in model.named_parameters():
            if any(attribute in name for attribute in tgt_attribute) and param.requires_grad:
                param.requires_grad = False
            if 'task' in name:
                param.requires_grad = True
            # if param.requires_grad:
                # print(name)

        for i in range(train_epochs):
            print(f'Epoch:  {i + 1}')
            train_loss, train_acc = task_train(model, train_loader, loss_func, optimizers,
                                               device, target_attribute, gate_version=gate_version)
            t_loss_list.append(train_loss)
            t_acc_list.append(train_acc)

            val_loss, val_attr_loss, val_acc = validation(model, attr_models, debias_method, validate_loader,
                                                          loss_func, device, target_attribute, active_gates='none',
                                                          gate_version=gate_version)
            v_loss_list.append(val_loss)
            v_attr_loss_list.append(val_attr_loss)
            v_acc_list.append(val_acc)
            if v_loss_list[-1] == min(v_loss_list):
                best_epoch = i
                best_weights = model.state_dict()

            current_task_lr = optimizers['task'].param_groups[0]['lr']

            if lr_scheduler is not None:
                for sched in lr_scheduler.values():
                    sched.step()
            log_dict = {
                "epochs": i + 1,
                "task lr": current_task_lr,
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
                        attribute_validation_log = val_attr_loss[key]
                    except:
                        attribute_validation_log = 0
                    wandb.log({"epochs": i + 1,
                               f"validation {key} loss": attribute_validation_log})

        print(f'Best Weights Saved at Epoch {best_epoch} ')
        model.load_state_dict(best_weights)

        if 'none' not in target_attribute:

            # debiasing stage
            adv_train_loss = []
            adv_val_loss = []
            print('\n', "Trainbale Parameters During Debiasing:")
            for name, param in model.named_parameters():
                if any(attribute in name for attribute in tgt_attribute) and not param.requires_grad:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                if param.requires_grad:
                    print(name)

            for i in range(train_epochs):
                print(f'Debiasing Epoch:  {i + 1}')
                train_loss, train_attr_loss, train_acc = adv_train(model, attr_models, debias_method, train_loader,
                                                                   loss_func, optimizers, device, target_attribute,
                                                                   gate_version=gate_version)
                adv_train_loss.append(train_loss)
                # t_acc_list.append(train_acc)

                val_loss, val_attr_loss, val_acc = validation(model, attr_models, debias_method,
                                                              validate_loader, loss_func, device,
                                                              target_attribute, active_gates=target_attribute,
                                                              gate_version=gate_version)
                v_loss_list.append(val_loss)
                adv_val_loss.append(val_attr_loss)

                for opt_name in tgt_attribute:
                    current_attr_lr = optimizers[opt_name].param_groups[0]['lr']

                if lr_scheduler is not None:
                    for sched in lr_scheduler.values():
                        sched.step()
                log_dict = {
                    "epochs": i + 1,
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
        else:
            adv_train_loss = [0.]
            adv_val_loss = [0.]

    if 'attacker' in train_type:
        temp_values = 0 if gate_version == 1 else 1
        temperature = [temp_values for _ in tgt_attribute]
        train_loader = embed_dataloader(model, train_loader, device, debias_method,
                                             attacker_attribute, active_gates=target_attribute, temp=temperature)
        validate_loader = embed_dataloader(model, validate_loader, device, debias_method,
                                           attacker_attribute, active_gates=target_attribute, temp=temperature)
        temp_test_loader = embed_dataloader(model, test_loader, device, debias_method,
                                           attacker_attribute, active_gates=target_attribute, temp=temperature)

        for i in range(attack_epochs):
            print(f'Attacker Epoch:  {i + 1}')
            train_attribute_loss, train_attribute_accuracy, train_attr_gap = train_attacker(attacker_model,
                                                                                            train_loader,
                                                                                            loss_func,
                                                                                            attacker_optim,
                                                                                            device,
                                                                                            target_attribute=attacker_attribute, )
            val_model_score, val_attribute_loss, val_attribute_accuracy, val_attr_gap = validate_attacker(attacker_model,
                                                                                                          validate_loader,
                                                                                                          loss_func,
                                                                                                          device,
                                                                                                          target_attribute=attacker_attribute)

            if wandb:
                for key in attacker_attribute.split(' '):
                    wandb.log({f"attacker train {key} loss": train_attribute_loss[key],
                               f"attacker validation {key} loss ": val_attribute_loss[key],
                               f"attacker train {key} acc": train_attribute_accuracy[key]['acc'],
                               f"attacker validation {key} accuracy": val_attribute_accuracy[key]['acc'],
                               f"attacker train {key} bacc": train_attribute_accuracy[key]['bacc'],
                               f"attacker validation {key} bacc": val_attribute_accuracy[key]['bacc'],
                               # f"attacker validation {key} f1": val_attribute_accuracy[key]['f1'],
                               "epochs": i + 1})

        test_loss, test_attr_loss, test_acc = validation(model, attr_models, debias_method, test_loader, loss_func,
                                                         device, target_attribute, gate_version=gate_version)
        test_model_score, test_attribute_loss, test_attribute_accuracy, test_attr_gap = validate_attacker(attacker_model,
                                                                                        temp_test_loader,
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
                           f"attacker test {key} bacc": test_attribute_accuracy[key]['bacc'],
                           # f"attacker test {key} rec": test_attribute_accuracy[key]['rec'],
                           # f"attacker test {key} f1": test_attribute_accuracy[key]['f1'],
                           "epochs": i + 1, })

    if (not 'attacker' in train_type) and (not 'model' in train_type):
        print('Train type should be either "model" ,  "attacker" or "model attacker"')
        exit()

    return t_loss_list, t_attr_loss_list, t_acc_list, v_loss_list, v_attr_loss_list, v_acc_list, test_loss, \
           test_attr_loss, test_acc
