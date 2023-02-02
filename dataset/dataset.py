from abc import ABC

import pandas as pd
import numpy as np
import os
# import nltk
# from nltk.corpus import stopwords
import seaborn as sns
from tqdm import tqdm

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn import preprocessing
import torch
from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader
from transformers import BertTokenizer


def load_dataset(dataset_path):
    print(dataset_path)
    if 'PAN16' in dataset_path:
        data = pd.read_csv(dataset_path, sep='\t', lineterminator='\n')
        data.columns = ['age', 'gender', 'text', 'label']
    elif 'bios' in dataset_path:
        data = pd.read_pickle(dataset_path)
        data = pd.DataFrame(data)
        data = data[['bio', 'title', 'gender']]
        data.rename(columns={'bio': 'text', 'title': 'label'}, inplace=True)
    elif 'hate_speech' in dataset_path:
        data = pd.read_pickle(dataset_path)
        data = pd.DataFrame(data)
        data = data[['tweet', 'label', 'dialect']]
        data.rename(columns={'tweet': 'text'}, inplace=True)

    name_maps = {}
    for name in (['label', 'gender', 'age', 'dialect']):
        if name in data.columns and type(data[name].iloc[-1]) == str:
            encoder = preprocessing.LabelEncoder()
            encoder.fit(data[name])
            data[name] = encoder.transform(data[name])
            name_maps[name] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return data, name_maps


def balanced_dataloader(dataset, batch_size=10, num_workers=8, equal_sampling=None):
    if equal_sampling:
        unique_labels, counts = np.unique(np.array(dataset[equal_sampling]), return_counts=True)
        weight_list = []
        for i, count in enumerate(counts):
            weight_list.append(max(counts) / count)
        sampling_weight = [weight_list[i] for i in dataset[equal_sampling]]
        sampler = WeightedRandomSampler(torch.tensor(sampling_weight).type('torch.DoubleTensor'), len(sampling_weight))
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, )


class CustomDataset(Dataset, ABC):
    def __init__(self, data, dataset_type, dataset_attributes, max_len, tokenizer="bert-base-uncased",
                 add_special_tokens=True, padding='max_length', truncation=True,
                 return_attention_mask=True, balanced=False):
        super(CustomDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_len = max_len
        self.special_token = add_special_tokens
        self.pad = padding
        self.trunc = truncation
        self.attention = return_attention_mask
        self.dataset_type = dataset_type
        self.dataframe = data.copy()
        self.dataset_attributes = dataset_attributes
        if balanced:
            if self.dataset_type == 'pan16':
                self.balance_oversample('age')
            elif self.dataset_type == 'bios':
                self.balance_oversample('label')
            elif self.dataset_type == 'hatespeech':
                self.balance_oversample('dialect')


        self.attribute = {}
        for name in dataset_attributes:
            self.attribute[name] = self.dataframe[name]
            if 'age' in name:
                self.attribute[f'binary_{name}'] = self.dataframe['age'].apply(lambda x: 0 if x == 2 else 1)
        self.text = self.dataframe['text']
        self.label = self.dataframe['label']

    def balance_oversample(self, column):
        if column!='dialect':
            repeats = self.dataframe.groupby(column, as_index=False).gender.value_counts()
            num_repeats = (self.dataframe.groupby(column, as_index=False).gender.value_counts().max()['count'] /
                           self.dataframe.groupby(column, as_index=False).gender.value_counts()['count']).round()
            repeats['count'] = num_repeats
            print(repeats)
            for i in range(len(repeats)):
                if repeats['count'].iloc[i] == 1:
                    pass
                else:
                    target_attribute = self.dataframe[
                        (self.dataframe[column] == repeats[column].iloc[i]) & (
                                    self.dataframe['gender'] == repeats['gender'].iloc[i])]
                    for _ in tqdm(range(int(repeats['count'].iloc[i])), desc='Generating Balanced Labels'):
                        self.dataframe = pd.concat((self.dataframe, target_attribute)).reset_index(drop=True)
        else:
            repeats = self.dataframe.groupby('label', as_index=False).dialect.value_counts()
            print(repeats)
            num_repeats = (self.dataframe.groupby('label', as_index=False).dialect.value_counts().max()['count'] // (
                self.dataframe.groupby('label', as_index=False).dialect.value_counts()['count']).round())
            repeats['count'] = num_repeats
            for i in tqdm(range(len(repeats)), desc='Generating Balanced Dataset'):
                if repeats['count'].iloc[i] == 1:
                    pass
                else:
                    target_attribute = self.dataframe[(self.dataframe['label'] == repeats['label'].iloc[i]) & (
                            self.dataframe['dialect'] == repeats['dialect'].iloc[i])]
                    for _ in range(int(repeats['count'].iloc[i])):
                        self.dataframe = pd.concat((self.dataframe, target_attribute)).reset_index(drop=True)


    def show_distribution(self, column):
        unique_labels, counts = np.unique(np.array(self.dataframe[column]), return_counts=True)
        print('unique columns:', f'{column}:{unique_labels}')
        print('unique counts:', f'{column}:{counts}')
        num_labels = len(unique_labels)
        print(f"labels in f'{column}:'{num_labels}")
        return sns.barplot(x=unique_labels, y=counts, color="blue", alpha=0.6)

    def visualize_doc_length(self, column):
        self.dataframe['doc_length'] = self.dataframe[column].apply(lambda x: len(x.split(' ')))
        return sns.histplot(self.dataframe['doc_length'])

    def __getitem__(self, index):
        x = self.tokenizer(self.text[index],
                           add_special_tokens=self.special_token,
                           max_length=self.max_len,
                           padding=self.pad,
                           truncation=self.trunc,
                           return_attention_mask=self.attention,
                           return_tensors='pt')

        data_attribute = []
        for name, value in self.attribute.items():
            data_attribute.append(torch.tensor(value[index], dtype=torch.long))

        return x['input_ids'].flatten(), x['token_type_ids'].flatten(), x['attention_mask'].flatten(), \
               torch.tensor(self.label[index], dtype=torch.long), data_attribute

    def __len__(self):
        return len(self.dataframe)


def embed_dataloader(model, data_loader, device, debias_method,
                     attack_attribute='gender age', active_gates='gender age', temp=[0], pred_evaluation=False):

    tgt_attribute = attack_attribute.split(' ')
    tqdm_bar = tqdm(data_loader, desc='Generate Embeddings')
    all_logits = []
    attr_targets = dict(zip(tgt_attribute, [[] for i in tgt_attribute]))
    pred, target = [], []
    temp = torch.tensor(temp).to(device)
    first_run = True
    for ids, token_types, att_masks, labels, attributes in tqdm_bar:
        ids, token_types, att_masks = ids.to(device), token_types.to(device), att_masks.to(device)
        if len(attributes) > 1:
            for i, attribute in enumerate(attributes):
                attributes[i] = attribute.to(device)
            if debias_method == 'dann':
                # print('Multi label attribute for Age selected')
                attributes = [attributes[0], attributes[1]]
            else:
                # print('single label attribute for Age selected')
                attributes = [attributes[0], attributes[2]]
        with torch.no_grad():
            logits, embeds = model(ids, token_type_ids=token_types, attention_mask=att_masks,
                                   return_embeds=True, active_gates=active_gates, temp=temp)
            if pred_evaluation:
                all_logits.append(logits)
        embeds = embeds[-1].to('cpu')

        if first_run:
            all_embeds = embeds
            first_run = False
        else:
            all_embeds = torch.concat((all_embeds, embeds), dim=0)
        pred += torch.argmax(logits, dim=1).tolist()
        target += labels.tolist()
        for counter, name in enumerate(tgt_attribute):
            attr_targets[name] += attributes[counter].tolist()
    if pred_evaluation:
        return torch.concat(all_logits), pred, target, attr_targets
    else:
        new_dataset = EmbedDataset(all_embeds, pred, target, attr_targets)
        new_dataloader = DataLoader(new_dataset, batch_size=64,
                                    num_workers=8, shuffle=False)
        return new_dataloader


class EmbedDataset(Dataset, ABC):
    def __init__(self, all_emeds, pred, target, attr_target):
        super(EmbedDataset, self).__init__()

        self.all_embeds = all_emeds
        self.target = target
        self.attr_target = attr_target
        self.pred = pred

    def __getitem__(self, item):
        attr = []
        for attr_name in self.attr_target.keys():
            # attr = [torch.tensor(self.attr_target['gender'][item]), torch.tensor(self.attr_target['age'][item])]
            attr.append(self.attr_target[attr_name][item])
        task_prediction = torch.tensor(self.pred[item], dtype=torch.long)
        task_label = torch.tensor(self.target[item], dtype=torch.long)
        return self.all_embeds[item], task_prediction, task_label, attr

    def __len__(self):
        return len(self.target)
