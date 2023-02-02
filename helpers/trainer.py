import torch
from tqdm import tqdm
from helpers.utilities import get_accuracy


def train(data_loader, model, loss_func, optim, device):
    model.train()
    Loss = []
    pred, target = [], []
    tqdm_bar = tqdm(data_loader)
    for ids, token_types, att_masks, labels in tqdm_bar:
        ids, token_types, att_masks, labels = ids.to(device), token_types.to(device), att_masks.to(device), labels.to(
            device)
        optim.zero_grad()
        logits = model(ids, token_type_ids=token_types, attention_mask=att_masks)

        loss = loss_func(logits, labels)
        loss.backward()
        optim.step()

        Loss.append(loss.item())
        pred += (torch.argmax(logits, dim=1).tolist())
        target += (labels.tolist())
        acc = get_accuracy(pred, target)
        tqdm_bar.set_description("train Loss, train Acc: %0.3f, %0.3f" % (loss, acc))
        tqdm_bar.update()
    return sum(Loss) / len(Loss), acc


def validate(data_loader, model, loss_func, device):
    model.eval()
    Loss = []
    pred, target = [], []
    tqdm_bar = tqdm(data_loader, desc='validation')
    with torch.no_grad():
        for ids, token_types, att_masks, labels in tqdm_bar:
            ids, token_types, att_masks, labels = ids.to(device), token_types.to(device), att_masks.to(
                device), labels.to(device)
            logits = model(ids, attention_mask=att_masks, token_type_ids=token_types)
            loss = loss_func(logits, labels)
            Loss.append(loss.item())
            pred += (torch.argmax(logits, dim=1).tolist())
            target += (labels.tolist())
            acc = get_accuracy(pred, target)
            tqdm_bar.set_description("validation Loss, validation Acc: %0.3f, %0.3f" % (loss, acc))
            tqdm_bar.update()

        return sum(Loss) / len(Loss), acc


def model_prediction(data_loader, model, device):
    model.eval()
    pred, target = [], []
    tqdm_bar = tqdm(data_loader, desc='prediction')
    with torch.no_grad():
        for ids, token_types, att_masks, labels in tqdm_bar:
            ids, token_types, att_masks, labels = ids.to(device), token_types.to(device), att_masks.to(
                device), labels.to(device)
            logits = model(ids, attention_mask=att_masks, token_type_ids=token_types)
            pred += (torch.argmax(logits, dim=1).tolist())
            target += (labels.tolist())
        return pred, target


def train_model(train_loader, validate_loader, test_loader, model, loss_func, optim, epochs, device):
    val_loss, val_acc = validate(validate_loader, model, loss_func, device)
    print("Epoch %2d,  Loss: %0.4f, Accuracy: %0.4f" % (0, val_loss, val_acc))
    t = tqdm(range(epochs), desc='Training:')
    for i in t:
        train_loss, train_acc = train(train_loader, model, loss_func, optim, device)
        val_loss, val_acc = validate(validate_loader, model, loss_func, device)
        t.set_description("Epoch %2d, Train Loss,Acc: %0.3f, %0.3f Val Loss,Acc: %0.3f, %0.3f " % (
            i + 1, train_loss, train_acc, val_loss, val_acc))
        t.update()
    test_loss, test_acc = validate(test_loader, model, loss_func, device)
    print("Test Acc,  Loss: %0.4f, Accuracy: %0.4f" % (test_loss, test_acc))
