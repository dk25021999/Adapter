# %%
import wandb

learning_rate = 0.00002
batch_size = 64
epochs = 20
model_name = "bert-base-uncased" 
#model_name = "google/bert_uncased_L-4_H-256_A-4" 
home_path = '/home/deepak/DiffPaper/Adv'
wandb.init(project="Diff GenderAge",config={"learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size}, entity="dk2502")


# %%
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2")

# %% [markdown]
# Training

# %%
import pandas as pd

df_train = pd.read_csv("/share/cp/datasets/nlp/text_classification_bias/Mention_PAN16/train.tsv", delimiter="\t", lineterminator='\n')
df_test = pd.read_csv("/share/cp/datasets/nlp/text_classification_bias/Mention_PAN16/test.tsv", delimiter="\t", lineterminator='\n')
df_validation = pd.read_csv("/share/cp/datasets/nlp/text_classification_bias/Mention_PAN16/validation.tsv", delimiter="\t", lineterminator='\n')

# %%
#Changing Label to integers of Jobs

from sklearn.preprocessing import LabelEncoder

le_age = LabelEncoder()
le_age.fit(df_train['age'])
df_train['age'] = le_age.transform(df_train['age'])
df_validation['age'] = le_age.transform(df_validation['age'])
df_test['age'] = le_age.transform(df_test['age'])

# %%
#Changing Label to integers of gender

from sklearn.preprocessing import LabelEncoder
le_gender = LabelEncoder()
le_gender.fit(df_train['gender'])
df_train['gender'] = le_gender.transform(df_train['gender'])
df_validation['gender'] = le_gender.transform(df_validation['gender'])
df_test['gender'] = le_gender.transform(df_test['gender'])

# %%
#del model

# %%
from transformers import BertModelWithHeads, BertTokenizerFast, BertConfig

config = BertConfig.from_pretrained(
    model_name,
)
old_model = BertModelWithHeads.from_pretrained(
    model_name,
    config=config,
)

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name) 


# %%
from transformers.adapters import PfeifferConfig

adap_config = PfeifferConfig(reduction_factor=2)
old_model.add_adapter("optima",adap_config)
old_model.train_adapter("optima",adap_config)

# %%
from transformers import AutoConfig, BertConfig, BertModel
import torch.nn as nn
import torch
from torch.autograd import Function
import pdb

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        grad_input = -grad_output*ctx.alpha
        return grad_input, None
    



class AdvNN(nn.Module):
    def __init__(self, hid_size, out_size, adv_count=1, adv_midlayers_size=[-1], adv_dropout=0.3):
        super(AdvNN, self).__init__()
        
        self.dropout_nn = nn.Dropout(adv_dropout)
        
        self.adv_mlps = nn.ModuleDict({})
        for i in range(adv_count):
            self.adv_mlps[str(i)] = nn.ModuleList()
            _last_size = hid_size
            for _midlayer_size in adv_midlayers_size:
                if _midlayer_size == -1:
                    _midlayer_size = _last_size
                self.adv_mlps[str(i)].append(nn.Linear(_last_size, _midlayer_size))
                self.adv_mlps[str(i)].append(nn.Tanh())
                _last_size = _midlayer_size
            self.adv_mlps[str(i)].append(nn.Linear(_last_size, out_size))

        self._adv_count = adv_count
        
    def adv_mlp_i(self, _encoded, adv_ind):
        _out = self.dropout_nn(_encoded)
        for module_id, _ in enumerate(self.adv_mlps[str(adv_ind)]):
            _out = self.adv_mlps[str(adv_ind)][module_id](_out)
        return _out
    
    def forward(self, _encoded):
        adversarial_res = []
        for i in range(self._adv_count):
            adversarial_res.append(self.adv_mlp_i(_encoded, i))
            
        return adversarial_res
    

class BertAdvNetClassifier(nn.Module):
    def __init__(self, old_model, task_out_size, gender_out_size, age_out_size, adv_midlayers_size=[-1], adv_count=5, adv_rev_ratio=1.0, dropout=0.3):
        super(BertAdvNetClassifier, self).__init__()
        
        self.encoder = old_model
        
        _hid_size = self.encoder.bert.embeddings.word_embeddings.embedding_dim
        
        self.adv_rev_ratio = adv_rev_ratio
        
        self.task_decoder_nn = nn.Sequential(nn.Dropout(dropout), 
                                             nn.Linear(_hid_size, _hid_size),
                                             nn.Tanh(),
                                             nn.Linear(_hid_size, task_out_size))
        self.gender_decoder_nn = AdvNN(hid_size = _hid_size, out_size=gender_out_size, 
                             adv_count=adv_count, adv_midlayers_size=adv_midlayers_size, adv_dropout=dropout)
        self.age_decoder_nn = AdvNN(hid_size = _hid_size, out_size=age_out_size, 
                             adv_count=adv_count, adv_midlayers_size=adv_midlayers_size, adv_dropout=dropout)


    
    def forward(self, sent_id, mask):
        
        hidden = self.encoder.forward(sent_id, mask).pooler_output
        

        adv_gender = self.gender_decoder_nn.forward(ReverseLayerF.apply(hidden, self.adv_rev_ratio))
        adv_age = self.age_decoder_nn.forward(ReverseLayerF.apply(hidden, self.adv_rev_ratio))
        
        task_res = self.task_decoder_nn(hidden)
        
        return task_res, adv_gender,adv_age

# %%
model = BertAdvNetClassifier(old_model, task_out_size=2, gender_out_size=2, age_out_size=5, adv_midlayers_size=[-1], adv_count=5, adv_rev_ratio=1.0, dropout=0.3 )

# %%
wandb.log({"trainable parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

model.to(device)

# %%
train_text = df_train["text"]
train_labels = df_train["task_label"]
train_gender = df_train["gender"]
train_age = df_train["age"]


val_text = df_validation["text"] 
val_labels= df_validation["task_label"]
val_gender = df_validation["gender"]
val_age = df_validation["age"]

# %%
del df_train
del df_validation

# %%
# truncate, tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 30,
    pad_to_max_length=True,
    truncation=True
)

# truncate, tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 30,
    pad_to_max_length=True,
    truncation=True
)

# %%
import torch

# convert lists to tensors

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())
train_g = torch.tensor(train_gender.tolist())
train_a = torch.tensor(train_age.tolist())


val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())
val_g = torch.tensor(val_gender.tolist())
val_a = torch.tensor(val_age.tolist())

# %%
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size


# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y, train_g, train_a)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y, val_g, val_a)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# %%
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

#compute the class weights
#class_weights_task = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y= train_labels)
class_weights_gender = compute_class_weight(class_weight='balanced', classes=np.unique(train_gender), y= train_gender)
class_weights_age = compute_class_weight(class_weight='balanced', classes=np.unique(train_age), y= train_age)

# %%
import torch.nn as nn

# adding class weights to loss function to counter class imbalence problem

# converting list of class weights to a tensor
#weights_task= torch.tensor(class_weights_task,dtype=torch.float)
weights_gender= torch.tensor(class_weights_gender,dtype=torch.float)
weights_age= torch.tensor(class_weights_age,dtype=torch.float)

# push to GPU
#weights_task = weights_task.to(device)
weights_gender = weights_gender.to(device)
weights_age = weights_age.to(device)

# define the loss function
cross_entropy_task  = nn.CrossEntropyLoss() 
cross_entropy_gender  = nn.CrossEntropyLoss(weight=weights_gender) 
cross_entropy_age  = nn.CrossEntropyLoss(weight=weights_age) 

# %%
### optimizer
params_group_model = []
params_group_adv = []

_added_params_name_model = []
_added_params_name_adv = []
for p_name, par in model.named_parameters():
    if par.requires_grad:
        if ("gender_decoder_nn" not in p_name) or ("age_decoder_nn" not in p_name):
            params_group_model.append(par)
            _added_params_name_model.append(p_name)
        else:
            params_group_adv.append(par)
            _added_params_name_adv.append(p_name)

optimizer_model = torch.optim.Adam([{"params":params_group_model, "lr":learning_rate, "weight_decay":0}], betas=(0.9, 0.999), eps=0.00001)
optimizer_adv = torch.optim.Adam([{"params":params_group_adv, "lr":learning_rate, "weight_decay":0}], betas=(0.9, 0.999), eps=0.00001)

# %%
# function to train the model
def train():
  
    model.train()

    total_task_loss = 0
    total_gender_loss = 0
    total_age_loss = 0
  
  # iterate over batches
    for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
        if step % 100 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
        batch = [r.to(device) for r in batch]
 
        sent_id, mask, task_labels, gender_labels, age_labels = batch

    # clear previously calculated gradients 
        model.zero_grad()        

    # get model predictions for the current batch
        task_preds, gender_preds, age_preds = model(sent_id, mask)

    # compute the loss between actual and predicted values
        task_loss = cross_entropy_task(task_preds, task_labels)

    # add on to the total loss
        total_task_loss = total_task_loss + task_loss.item()


        gender_loss = [cross_entropy_gender(preds_item, gender_labels) for preds_item in gender_preds]
        gender_loss = torch.stack(gender_loss).mean()
        age_loss = [cross_entropy_age(preds_item, age_labels) for preds_item in age_preds]
        age_loss = torch.stack(age_loss).mean()

    # add on to the total loss
        total_gender_loss = total_gender_loss + gender_loss.item()
        total_age_loss = total_age_loss + age_loss.item()

    # backward pass to calculate the gradients
        loss = task_loss+gender_loss+age_loss
        loss.backward()

        wandb.log({"train_task_step_loss": task_loss})
        wandb.log({"train_gender_step_loss": gender_loss})
        wandb.log({"train_age_step_loss": age_loss})

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
        optimizer_adv.step()
    # update parameters
        optimizer_model.step()


  # compute the training loss of the epoch
    avg_task_loss = total_task_loss / len(train_dataloader)
  



  # compute the training loss of the epoch
    avg_gender_loss = total_gender_loss / len(train_dataloader)
    avg_age_loss = total_age_loss / len(train_dataloader)
  


  #returns the loss and predictions
    return avg_task_loss,  avg_gender_loss, avg_age_loss

# %%
# function for evaluating the model
def evaluate():
  
    print("\nEvaluating...")
  
  # deactivate dropout layers
    model.eval()

    total_task_loss = 0
    total_gender_loss = 0
    total_age_loss = 0
  
  # empty list to save the model predictions
    total_task_preds=[]
    total_gender_preds=[]
    total_age_preds=[]
    total_task_labels=[]
    total_gender_labels=[]
    total_age_labels=[]  

  # iterate over batches
    for step,batch in enumerate(val_dataloader):
    
    # Progress update every 50 batches.
        if step % 100 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)
            
      # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, task_labels, gender_labels, age_labels = batch

    # deactivate autograd
        with torch.no_grad():
      
      # model predictions
            task_preds, gender_preds_list, age_preds_list = model(sent_id, mask)

      # compute the validation loss between actual and predicted values
            task_loss = cross_entropy_task(task_preds,task_labels)
            total_task_loss = total_task_loss + task_loss.item()

            task_preds = task_preds.detach().cpu().numpy()
            task_labels = task_labels.detach().cpu().numpy()

            total_task_labels.append(task_labels)

            total_task_preds.append(task_preds)


      # compute the validation loss between actual and predicted values
            genderList_argmax = []

            for i in range(len(gender_preds_list)):
              genderList_argmax.append(torch.argmax(gender_preds_list[i], dim=1))
            gender_preds = torch.mode(torch.stack(genderList_argmax), dim=0).values.tolist()

            gender_loss = [cross_entropy_gender(preds_item, gender_labels) for preds_item in gender_preds_list]
            gender_loss = torch.stack(gender_loss).mean()

            total_gender_loss = total_gender_loss + gender_loss.item()

            gender_labels = gender_labels.detach().cpu().numpy()

            total_gender_labels.append(gender_labels)
            total_gender_preds.append(gender_preds)


            ageList_argmax = []

            for i in range(len(age_preds_list)):
              ageList_argmax.append(torch.argmax(age_preds_list[i], dim=1))
            age_preds = torch.mode(torch.stack(ageList_argmax), dim=0).values.tolist()

            age_loss = [cross_entropy_age(preds_item, age_labels) for preds_item in age_preds_list]
            age_loss = torch.stack(age_loss).mean()

            total_age_loss = total_age_loss + age_loss.item()

            age_labels = age_labels.detach().cpu().numpy()

            total_age_labels.append(age_labels)
            total_age_preds.append(age_preds)
            


  # compute the validation loss of the epoch
    avg_task_loss = total_task_loss / len(val_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
    total_task_preds  = np.concatenate(total_task_preds, axis=0)
    total_task_preds = [np.argmax(i) for i in total_task_preds]
    total_task_labels  = np.concatenate(total_task_labels, axis=0)

  # compute the validation loss of the epoch
    avg_gender_loss = total_gender_loss / len(val_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
    total_gender_preds  = np.concatenate(total_gender_preds, axis=0)
    total_gender_labels = np.concatenate(total_gender_labels, axis=0)

    avg_age_loss = total_age_loss / len(val_dataloader) 
    total_age_preds  = np.concatenate(total_age_preds, axis=0)
    total_age_labels = np.concatenate(total_age_labels, axis=0)

    return avg_task_loss, total_task_preds, total_task_labels, avg_gender_loss, total_gender_preds, total_gender_labels, avg_age_loss, total_age_preds, total_age_labels

# %%
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score


# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_task_losses=[]
valid_task_losses=[]
train_gender_losses=[]
valid_gender_losses=[]

#for each epoch
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_task_loss, train_gender_loss, train_age_loss = train()

    wandb.log({"train_task_loss": train_task_loss})
    wandb.log({"train_gender_loss": train_gender_loss})
    wandb.log({"train_age_loss": train_age_loss})
    
    #evaluate model
    valid_task_loss, task_preds, task_labels, valid_gender_loss, gender_prdes, gender_labels, valid_age_loss, age_prdes, age_labels = evaluate()
    
    wandb.log({"valid_task_loss": valid_task_loss})
    wandb.log({"valid_gender_loss": valid_gender_loss})
    wandb.log({"valid_age_loss": valid_age_loss})
    b_acc_task = balanced_accuracy_score(task_preds, task_labels)
    acc_task = accuracy_score(task_preds, task_labels)
    wandb.log({"valid_b_acc_task": b_acc_task})
    wandb.log({"valid_acc_task": acc_task})
    b_acc_gender = balanced_accuracy_score(gender_prdes, gender_labels)
    acc_gender = accuracy_score(gender_prdes, gender_labels)
    wandb.log({"valid_b_acc_gender": b_acc_gender})
    wandb.log({"valid_acc_gender": acc_gender})
    b_acc_age = balanced_accuracy_score(age_prdes, age_labels)
    acc_age = accuracy_score(age_prdes, age_labels)
    wandb.log({"valid_b_acc_age": b_acc_age})
    wandb.log({"valid_acc_age": acc_age})

    
    #save the best model
    torch.save(model.state_dict(), 'Adapter_TGA.pt')
    
    # append training and validation loss
    train_task_losses.append(train_task_loss)
    valid_task_losses.append(valid_task_loss)
    train_gender_losses.append(train_gender_loss)
    valid_gender_losses.append(valid_gender_loss)
    
    print(f'\nTraining Task Loss: {train_task_loss:.3f}')
    print(f'Validation Task Loss: {valid_task_loss:.3f}')
    print(f'\nTraining gender Loss: {train_gender_loss:.3f}')
    print(f'Validation gender Loss: {valid_gender_loss:.3f}')

# %%


# %% [markdown]
# Training Gender Attacker
learning_rate = 0.0001
batch_size = 64
epochs = 40
# %%
#Loading saved model for testing

model.load_state_dict(torch.load(home_path+'/Adapter_TGA.pt'))

# %%
import pandas as pd

df_train = pd.read_csv("/share/cp/datasets/nlp/text_classification_bias/Mention_PAN16/train.tsv", delimiter="\t", lineterminator='\n')
df_test = pd.read_csv("/share/cp/datasets/nlp/text_classification_bias/Mention_PAN16/test.tsv", delimiter="\t", lineterminator='\n')
df_validation = pd.read_csv("/share/cp/datasets/nlp/text_classification_bias/Mention_PAN16/validation.tsv", delimiter="\t", lineterminator='\n')


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_train['gender'])
df_train['gender'] = le.transform(df_train['gender'])
df_validation['gender'] = le.transform(df_validation['gender'])
df_test['gender'] = le.transform(df_test['gender'])


from transformers import BertModelWithHeads, BertTokenizerFast, BertConfig

#id2label = {id: label for (id, label) in enumerate(le.classes_)}

train_text = df_train["text"]
train_labels = df_train["gender"]


val_text = df_validation["text"] 
val_labels= df_validation["gender"] 

# truncate, tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 30,
    pad_to_max_length=True,
    truncation=True
)

# truncate, tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 30,
    pad_to_max_length=True,
    truncation=True
)

import torch

# convert lists to tensors

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)



# %%
class Attacker(nn.Module):
    def __init__(self, old_model, prot_out_size,attacker_heads, dropout=0.3):
        super(Attacker, self).__init__()
        
        self.encoder = old_model.encoder
        
        _hid_size = self.encoder.bert.embeddings.word_embeddings.embedding_dim

        self.adv_net = AdvNN(hid_size = _hid_size, out_size=prot_out_size, 
                             adv_count=attacker_heads, adv_midlayers_size=[-1], adv_dropout=dropout)


    def forward(self, sent_id, mask):
        
        hidden = self.encoder.forward(sent_id, mask).pooler_output

        output = self.adv_net(hidden)


        return output

# %%
Attacker_model = Attacker(old_model= model,prot_out_size=2, attacker_heads=5, dropout=0.3)

# %%
Attacker_model.to(device)

# %%

for para in Attacker_model.encoder.parameters():
    para.requires_grad = False 


for name, param in Attacker_model.named_parameters():
    if param.requires_grad:
        print(name)

# %%
### optimizer
params_group_attacker = []

for p_name, par in Attacker_model.named_parameters():
    if par.requires_grad:
        if "adv_net" in p_name:
            params_group_attacker.append(par)

optimizer = torch.optim.Adam([{"params":params_group_attacker, "lr":learning_rate, "weight_decay":0}], betas=(0.9, 0.999), eps=0.00001)

# %%
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

#compute the class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y= train_labels)

print("Class Weights:",class_weights)

import torch.nn as nn

# adding class weights to loss function to counter class imbalence problem

# converting list of class weights to a tensor
weights= torch.tensor(class_weights,dtype=torch.float)

# push to GPU
weights = weights.to(device)

# define the loss function
cross_entropy  = nn.CrossEntropyLoss(weight=weights) 

# %%
# function to train the model
def train():
  
    Attacker_model.train()

    total_loss = 0

  
  # iterate over batches
    for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
            

    # push the batch to gpu
        batch = [r.to(device) for r in batch]
 
        sent_id, mask, labels = batch

    # clear previously calculated gradients 
        Attacker_model.zero_grad()        

    # get model predictions for the current batch
        preds = Attacker_model(sent_id, mask)

        attack_losses = [cross_entropy(preds_item, labels) for preds_item in preds]
        loss = torch.stack(attack_losses).mean()

    # add on to the total loss
        total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
        loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(Attacker_model.parameters(), 1.0)

    # update parameters
        optimizer.step()


  # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
  

    return avg_loss


# function for evaluating the model
def evaluate():
  
    print("\nEvaluating...")
  
  # deactivate dropout layers
    Attacker_model.eval()

    total_loss = 0
  
  # empty list to save the model predictions
    total_preds = []
    total_labels = []
    
  # iterate over batches
    for step,batch in enumerate(val_dataloader):
    
    # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)
            
      # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

    # deactivate autograd
        with torch.no_grad():
      
      # model predictions
            preds = Attacker_model(sent_id, mask)

            predsList_argmax = []

            for i in range(len(preds)):
              predsList_argmax.append(torch.argmax(preds[i], dim=1))
            prot_preds = torch.mode(torch.stack(predsList_argmax), dim=0).values.tolist()

            attack_losses = [cross_entropy(preds_item, labels) for preds_item in preds]
            loss = torch.stack(attack_losses).mean()
            

            
            total_loss = total_loss + loss.item()

            labels = labels.detach().cpu().numpy()

            total_labels.append(labels)
            total_preds.append(prot_preds)

  # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    total_labels  = np.concatenate(total_labels, axis=0)

    return avg_loss, total_preds, total_labels

# %%
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score


# set initial loss to infinite
best_valid_acc = float(0)

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(20-6):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss = train()
    
    wandb.log({"Gender_attacker_train_loss": train_loss})

    #evaluate model
    valid_loss, attacker_preds, attacker_labels = evaluate()
    
    wandb.log({"Gender_attacker_valid_loss": valid_loss})

    b_acc_prot = balanced_accuracy_score(attacker_preds, attacker_labels)
    acc_prot = accuracy_score(attacker_preds, attacker_labels)
    wandb.log({"Gender_Attacker_b_acc_valid": b_acc_prot})
    wandb.log({"Gender_Attacker_acc_valid": acc_prot})


    #save the best model
    if acc_prot > best_valid_acc:
        best_valid_acc = acc_prot
        torch.save(Attacker_model.state_dict(), 'Adapter_TGA_Gender_attacker.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

# %%


# %% [markdown]
# Attacker Performance

# %%
#Loading saved model for testing

Attacker_model.load_state_dict(torch.load(home_path+'/Adapter_TGA_Gender_attacker.pt'))

# %%
Attacker_model.to(device)

# %%


test_text = df_test["text"] 
test_labels= df_test["gender"] 


# truncate, tokenize and encode sequences in the validation set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 30,
    pad_to_max_length=True,
    truncation=True
)


import torch

# convert lists to tensors

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())



from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size


# wrap tensors
test_data = TensorDataset(test_seq, test_mask, test_y)

# sampler for sampling the data during training
test_sampler = SequentialSampler(test_data)

# dataLoader for validation set
test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)


import torch.nn as nn
import numpy as np

cross_entropy = nn.CrossEntropyLoss()


# function for evaluating the model
def testing():
  
    print("\nEvaluating...")
  
  # deactivate dropout layers
    Attacker_model.eval()

    total_loss = 0
  
  # empty list to save the model predictions
    total_preds = []
    total_labels = []

  # iterate over batches
    for step,batch in enumerate(test_dataloader):
    
    # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)
            
      # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

    # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

    # deactivate autograd
        with torch.no_grad():
      
      # model predictions
            preds = Attacker_model(sent_id, mask)

            predsList_argmax = []

            for i in range(len(preds)):
              predsList_argmax.append(torch.argmax(preds[i], dim=1))
            prot_preds = torch.mode(torch.stack(predsList_argmax), dim=0).values.tolist()

            attack_losses = [cross_entropy(preds_item, labels) for preds_item in preds]
            loss = torch.stack(attack_losses).mean()
            

            total_loss = total_loss + loss.item()

            labels = labels.detach().cpu().numpy()

            total_labels.append(labels)
            total_preds.append(prot_preds)

  # compute the validation loss of the epoch
    avg_loss = total_loss / len(test_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    total_labels  = np.concatenate(total_labels, axis=0)

    return avg_loss, total_preds, total_labels

# %%
#evaluate model
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

#evaluate model
valid_loss, model_pred, g_truth = testing()

print(f'Validation Loss: {valid_loss:.3f}')

b_acc_prot = balanced_accuracy_score(model_pred, g_truth)
acc_prot = accuracy_score(model_pred, g_truth)
wandb.log({"Gender_Attacker_b_acc_Test": b_acc_prot})
wandb.log({"Gender_Attacker_acc_Test": acc_prot})

# %%


# %% [markdown]
# Training Age Attacker

# %%
model.load_state_dict(torch.load(home_path+'/Adapter_TGA.pt'))

# %%
import pandas as pd

df_train = pd.read_csv("/share/cp/datasets/nlp/text_classification_bias/Mention_PAN16/train.tsv", delimiter="\t", lineterminator='\n')
df_test = pd.read_csv("/share/cp/datasets/nlp/text_classification_bias/Mention_PAN16/test.tsv", delimiter="\t", lineterminator='\n')
df_validation = pd.read_csv("/share/cp/datasets/nlp/text_classification_bias/Mention_PAN16/validation.tsv", delimiter="\t", lineterminator='\n')


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_train['age'])
df_train['age'] = le.transform(df_train['age'])
df_validation['age'] = le.transform(df_validation['age'])
df_test['age'] = le.transform(df_test['age'])


from transformers import BertModelWithHeads, BertTokenizerFast, BertConfig

#id2label = {id: label for (id, label) in enumerate(le.classes_)}

train_text = df_train["text"]
train_labels = df_train["age"]


val_text = df_validation["text"] 
val_labels= df_validation["age"] 

# truncate, tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 30,
    pad_to_max_length=True,
    truncation=True
)

# truncate, tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 30,
    pad_to_max_length=True,
    truncation=True
)

import torch

# convert lists to tensors

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)



# %%
class Attacker(nn.Module):
    def __init__(self, old_model, prot_out_size,attacker_heads, dropout=0.3):
        super(Attacker, self).__init__()
        
        self.encoder = old_model.encoder
        
        _hid_size = self.encoder.bert.embeddings.word_embeddings.embedding_dim

        self.adv_net = AdvNN(hid_size = _hid_size, out_size=prot_out_size, 
                             adv_count=attacker_heads, adv_midlayers_size=[-1], adv_dropout=dropout)


    def forward(self, sent_id, mask):
        
        hidden = self.encoder.forward(sent_id, mask).pooler_output

        output = self.adv_net(hidden)


        return output

# %%
Attacker_model = Attacker(old_model= model,prot_out_size=5, attacker_heads=5, dropout=0.3)

# %%
Attacker_model.to(device)

# %%
for para in Attacker_model.encoder.parameters():
    para.requires_grad = False 


for name, param in Attacker_model.named_parameters():
    if param.requires_grad:
        print(name)

# %%
### optimizer
params_group_attacker = []

for p_name, par in Attacker_model.named_parameters():
    if par.requires_grad:
        if "adv_net" in p_name:
            params_group_attacker.append(par)

optimizer = torch.optim.Adam([{"params":params_group_attacker, "lr":learning_rate, "weight_decay":0}], betas=(0.9, 0.999), eps=0.00001)

# %%
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

#compute the class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y= train_labels)

print("Class Weights:",class_weights)

import torch.nn as nn

# adding class weights to loss function to counter class imbalence problem

# converting list of class weights to a tensor
weights= torch.tensor(class_weights,dtype=torch.float)

# push to GPU
weights = weights.to(device)

# define the loss function
cross_entropy  = nn.CrossEntropyLoss(weight=weights) 

# %%
# function to train the model
def train():
  
    Attacker_model.train()

    total_loss = 0

  
  # iterate over batches
    for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
            

    # push the batch to gpu
        batch = [r.to(device) for r in batch]
 
        sent_id, mask, labels = batch

    # clear previously calculated gradients 
        Attacker_model.zero_grad()        

    # get model predictions for the current batch
        preds = Attacker_model(sent_id, mask)

        attack_losses = [cross_entropy(preds_item, labels) for preds_item in preds]
        loss = torch.stack(attack_losses).mean()

    # add on to the total loss
        total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
        loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(Attacker_model.parameters(), 1.0)

    # update parameters
        optimizer.step()


  # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
  

    return avg_loss


# function for evaluating the model
def evaluate():
  
    print("\nEvaluating...")
  
  # deactivate dropout layers
    Attacker_model.eval()

    total_loss = 0
  
  # empty list to save the model predictions
    total_preds = []
    total_labels = []
    
  # iterate over batches
    for step,batch in enumerate(val_dataloader):
    
    # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)
            
      # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

    # deactivate autograd
        with torch.no_grad():
      
      # model predictions
            preds = Attacker_model(sent_id, mask)

            predsList_argmax = []

            for i in range(len(preds)):
              predsList_argmax.append(torch.argmax(preds[i], dim=1))
            prot_preds = torch.mode(torch.stack(predsList_argmax), dim=0).values.tolist()

            attack_losses = [cross_entropy(preds_item, labels) for preds_item in preds]
            loss = torch.stack(attack_losses).mean()
            

            
            total_loss = total_loss + loss.item()

            labels = labels.detach().cpu().numpy()

            total_labels.append(labels)
            total_preds.append(prot_preds)

  # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    total_labels  = np.concatenate(total_labels, axis=0)

    return avg_loss, total_preds, total_labels

# %%
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score


# set initial loss to infinite
best_valid_acc = float(0)

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss = train()
    
    wandb.log({"Age_attacker_train_loss": train_loss})

    #evaluate model
    valid_loss, attacker_preds, attacker_labels = evaluate()
    
    wandb.log({"Age_attacker_valid_loss": valid_loss})

    b_acc_prot = balanced_accuracy_score(attacker_preds, attacker_labels)
    acc_prot = accuracy_score(attacker_preds, attacker_labels)
    wandb.log({"Age_Attacker_b_acc_valid": b_acc_prot})
    wandb.log({"Age_Attacker_acc_valid": acc_prot})



    #save the best model
    if acc_prot > best_valid_acc:
        best_valid_acc = acc_prot
        torch.save(Attacker_model.state_dict(), 'Adapter_TGA_Age_attacker.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

# %% [markdown]
# Attacker Performance

# %%
#Loading saved model for testing

Attacker_model.load_state_dict(torch.load(home_path+'/Adapter_TGA_Age_attacker.pt'))

# %%
Attacker_model.to(device)

# %%


test_text = df_test["text"] 
test_labels= df_test["age"] 


# truncate, tokenize and encode sequences in the validation set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 30,
    pad_to_max_length=True,
    truncation=True
)


import torch

# convert lists to tensors

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())



from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size


# wrap tensors
test_data = TensorDataset(test_seq, test_mask, test_y)

# sampler for sampling the data during training
test_sampler = SequentialSampler(test_data)

# dataLoader for validation set
test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)


import torch.nn as nn
import numpy as np

cross_entropy = nn.CrossEntropyLoss()


# function for evaluating the model
def testing():
  
    print("\nEvaluating...")
  
  # deactivate dropout layers
    Attacker_model.eval()

    total_loss = 0
  
  # empty list to save the model predictions
    total_preds = []
    total_labels = []

  # iterate over batches
    for step,batch in enumerate(test_dataloader):
    
    # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)
            
      # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

    # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

    # deactivate autograd
        with torch.no_grad():
      
      # model predictions
            preds = Attacker_model(sent_id, mask)

            predsList_argmax = []

            for i in range(len(preds)):
              predsList_argmax.append(torch.argmax(preds[i], dim=1))
            prot_preds = torch.mode(torch.stack(predsList_argmax), dim=0).values.tolist()

            attack_losses = [cross_entropy(preds_item, labels) for preds_item in preds]
            loss = torch.stack(attack_losses).mean()
            

            total_loss = total_loss + loss.item()

            labels = labels.detach().cpu().numpy()

            total_labels.append(labels)
            total_preds.append(prot_preds)

  # compute the validation loss of the epoch
    avg_loss = total_loss / len(test_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    total_labels  = np.concatenate(total_labels, axis=0)

    return avg_loss, total_preds, total_labels

# %%
#evaluate model
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

#evaluate model
valid_loss, model_pred, g_truth = testing()

print(f'Validation Loss: {valid_loss:.3f}')

b_acc_prot = balanced_accuracy_score(model_pred, g_truth)
acc_prot = accuracy_score(model_pred, g_truth)
wandb.log({"Age_Attacker_b_acc_Test": b_acc_prot})
wandb.log({"Age_Attacker_acc_Test": acc_prot})

# %% [markdown]
# Testing

# %%
#Loading saved model for testing

model.load_state_dict(torch.load(home_path+'/Adapter_TGA.pt'))



# %%
model.to(device)

# %%
test_text = df_test["text"] 
test_labels= df_test["task_label"] 


# truncate, tokenize and encode sequences in the validation set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 30,
    pad_to_max_length=True,
    truncation=True
)


import torch

# convert lists to tensors

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())



from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size


# wrap tensors
test_data = TensorDataset(test_seq, test_mask, test_y)

# sampler for sampling the data during training
test_sampler = SequentialSampler(test_data)

# dataLoader for validation set
test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)


# %%
import torch.nn as nn
import numpy as np

cross_entropy = nn.CrossEntropyLoss()


# function for evaluating the model
def testing():
  
    print("\nEvaluating...")
  
  # deactivate dropout layers
    model.eval()

    total_loss = 0
  
  # empty list to save the model predictions
    total_preds = []
    total_labels = []

  # iterate over batches
    for step,batch in enumerate(test_dataloader):
    
    # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)
            
      # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

    # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

    # deactivate autograd
        with torch.no_grad():
      
      # model predictions
            preds, _,_ = model(sent_id, mask)

      # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            total_labels.append(labels)

            total_preds.append(preds)

  # compute the validation loss of the epoch
    avg_loss = total_loss / len(test_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    total_preds = [np.argmax(i) for i in total_preds]
    total_labels  = np.concatenate(total_labels, axis=0)

    return avg_loss, total_preds, total_labels

# %%
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

#evaluate model
valid_loss, model_pred, g_truth = testing()

print(f'Validation Loss: {valid_loss:.3f}')

b_acc_prot = balanced_accuracy_score(model_pred, g_truth)
acc_prot = accuracy_score(model_pred, g_truth)
wandb.log({"Test Bal Acc": b_acc_prot})
wandb.log({"Test Acc": acc_prot})

