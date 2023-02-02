#  General Libraries
import numpy as np
import yaml
import os
import random
import warnings
import argparse

# Debiasing and Torch based libraies
# from torchinfo import summary
from da import DABase
# import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

# custom written functions and classes from the project
from helpers.adv_seq_training import adv_seq_train
from helpers.utilities import dict_to_class
from models.Bert import Bert
from models.BertGate import BertGateV1
from models.BertGateV2 import BertGateV2
from models.Adapter_Bert import AdapterBert
from models.attribute_classifier import AttackerModel
from dataset.dataset import load_dataset, balanced_dataloader, CustomDataset
from helpers.seq_trainer import seq_train_model
from datetime import datetime
#  Transformers Initial weight warning remover
from transformers import logging
# Remove Open file problem
import torch.multiprocessing

# Version deprecated warning removal
warnings.filterwarnings('ignore')
# Transformer Warning Remover
logging.set_verbosity_error()
# Change Multiprocess setting to avoid too many file open error
torch.multiprocessing.set_sharing_strategy('file_system')

# Generate args.parser
arg_parser = argparse.ArgumentParser(description='Experiment Arguments')
arg_parser.add_argument('--dataset', type=str, default='pan16')
arg_parser.add_argument('--device_id', type=int, default=0, help='GPU id when you run on server')
arg_parser.add_argument('--lr', type=float, default=5e-5, help='learning rate for task solver')
arg_parser.add_argument('--attr_lr', type=float, default=1e-3, help='learning rate of debiasing unit in case of DANN')
arg_parser.add_argument('--atk_lr', type=float, default=1e-4, help='Attacker learning rate')
arg_parser.add_argument('--lr_scheduler', type=int, default=1, help='Boolean Value indicating decaying Scheduler')
arg_parser.add_argument('--train_epochs', type=int, default=15, help='training epoch of the network')  # pan16 15
arg_parser.add_argument('--attack_epochs', type=int, default=30,
                        help='number of attacker epochs after training')  # pan16 30
arg_parser.add_argument('--batch_size', type=int, default=64, help='batch size of the dataloaders')  # pan16 64
arg_parser.add_argument('--num_workers', type=int, default=8, help='number of workers in dataloaders')
arg_parser.add_argument('--balance_dataset', type=int, default=0,
                        help='Boolean value to balance dataset based on age')  # pan16 0
arg_parser.add_argument('--last_hidden', type=int, default=0,
                        help='boolean value for using bert pooler or hidden')  # pan16 0
arg_parser.add_argument('--model_name', type=str, default="mini")  # mini or base
arg_parser.add_argument('--max_doc_length', type=int, default=30,
                        help='max document length for the dataset')  # pan16 30
arg_parser.add_argument('--new_embed_layer', type=int, default=0,
                        help='generates a new embedding linear layer')  # pan16 0
arg_parser.add_argument('--padding', type=str, default="max_length",
                        help='padding of the tokenizer')  # pan16 max_length
arg_parser.add_argument('--train_layer_norm', type=int, default=1, help='if True all layer norm are trained')  # pan16 1
arg_parser.add_argument('--trainable_parameters', type=str, default="encoder+pooler",
                        help='trainbale parameters selection')
arg_parser.add_argument('--target_attribute', type=str, default="gender age",
                        help='attribute to debias')  # pan16 'gender age'
arg_parser.add_argument('--debias_lambda', type=float, default=1.)  # pan16 1
arg_parser.add_argument('--lambda_scheduler', type=int, default=0)  # pan16 0
arg_parser.add_argument('--debias_loss', type=str, default="dann")  # pan16 'dann'
arg_parser.add_argument('--loss_function', type=str, default="CrossEntropy")  # pan16 CrossEntropy
arg_parser.add_argument('--train_type', type=str, default="model attacker")
arg_parser.add_argument('--wandb', type=int, default=1)
arg_parser.add_argument('--server', type=int, default=1)
arg_parser.add_argument('--gate_version', type=int, default=1)  # pan16 1
arg_parser.add_argument('--adapter', type=int, default=0)
arg_parser.add_argument('--num_gate_layers', type=int, default=1)
arg_parser.add_argument('--sequence_training', type=int, default=1)  # pan16 1
arg_parser.add_argument('--comment', type=str, default='')
arg_parser.add_argument('--num_runs', type=int, default=1)


args = arg_parser.parse_args()

if args.adapter:
    assert args.gate_version == 0

if args.target_attribute == 'none':
    assert args.sequence_training == 1

for i in range(args.num_runs):
    # Set seed for the experiment
    seed = np.random.randint(0, 1e9, 1)[0]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    now = datetime.now()
    dt_string = now.strftime("%y-%m%d%H%M")

    project_name = f"acl_final_{args.dataset}"

    model_name = "google/bert_uncased_L-4_H-256_A-4" if 'mini' in args.model_name else 'bert-base-uncased'

    # Unique name of the experiment during saving and reproducing results
    experiment_name = f"{args.dataset}v{args.gate_version}{'_adapter_' if args.adapter else ''}" \
                      f"{f'-{args.num_gate_layers}' if args.num_gate_layers != 1 else ''}" \
                      f"{'seq' if args.sequence_training else ''}" \
                      f"-attr-{args.target_attribute.replace(' ', '-')}" \
                      f"-{args.trainable_parameters.replace(' ', '-')}" \
                      f"-{args.debias_loss}{str(args.debias_lambda)}" \
                      f"{'sc' if args.lambda_scheduler else ''}-bl{args.balance_dataset}-{seed}" \
                      f"-{'mini' if 'google' in model_name else 'base'}-{dt_string}"

    device = f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # initial assumptions related to specific dataset. unique to dataset
    dataset_type = args.dataset
    if dataset_type == 'pan16':
        train_path = f'{"" if args.server else "~"}/share/cp/datasets/nlp/text_classification_bias/Mention_PAN16/train.tsv'
        validate_path = f"{'' if args.server else '~'}/share/cp/datasets/nlp/text_classification_bias" \
                        f"/Mention_PAN16/validation.tsv"
        test_path = f"{'' if args.server else '~'}/share/cp/datasets/nlp/text_classification_bias/Mention_PAN16/test.tsv"
        dataset_attributes = ['gender', 'age']  # attributes which exist in dataset
        attacker_attribute = 'gender age'  # attacker attributes as string with space to seperate them
        num_labels = 2  # number of labels which exist in dataset
        attributes_unique = [2, 5]  # unique labels in each attribute (adversarial based method require this number)
    elif 'bios' in dataset_type:
        train_path = f'{"" if args.server else "~"}/share/cp/datasets/nlp/text_classification_bias' \
                     f'/bios/only_task_balanced/train_balanced_every_job.pkl'
        validate_path = f"{'' if args.server else '~'}/share/cp/datasets/nlp/text_classification_bias" \
                        f"/bios/only_task_balanced/val.pkl"
        test_path = f"{'' if args.server else '~'}/share/cp/datasets/nlp/text_classification_bias" \
                    f"/bios/only_task_balanced/test.pkl"
        dataset_attributes = ['gender']  # attributes which exist in dataset
        attacker_attribute = 'gender'  # attacker attributes as string with space to seperate them
        num_labels = 28  # number of labels which exist in dataset
        attributes_unique = [2]  # unique labels in each attribute (adversarial based method require this number)
    elif 'hatespeech' in dataset_type:
        train_path = f'{"" if args.server else "~"}/share/cp/datasets/nlp/hate_speech/' \
                     f'Twitter/AAEpredicted/pickle_format/train.pkl'
        validate_path = f"{'' if args.server else '~'}/share/cp/datasets/nlp/hate_speech/" \
                        f"Twitter/AAEpredicted/pickle_format/val.pkl"
        test_path = f"{'' if args.server else '~'}/share/cp/datasets/nlp/hate_speech/" \
                    f"Twitter/AAEpredicted/pickle_format/test.pkl"
        dataset_attributes = ['dialect']  # attributes which exist in dataset
        attacker_attribute = 'dialect'  # attacker attributes as string with space to seperate them
        num_labels = 4  # number of labels which exist in dataset
        attributes_unique = [2]  # unique labels in each attribute (adversarial based method require this number)

    checkpoint_path = f"{dataset_type}_models"  # Checkpoint path folder
    checkpoint_file = os.path.join(checkpoint_path, experiment_name)  # checkpoint file name
    capture_last_hidden = False  # option to get last hidden layer output instead of pooler layer output
    equal_sampling = None  # equal sampling of the dataset ( NOT functional YET)
    loss_function = nn.CrossEntropyLoss()  # loss for debiasing and task in case dann is used otherwise only task
    num_workers = args.num_workers  # number of cpu workers during data loading
    embed_ = 256 if 'google' in model_name else 256 * 3
    attacker_hidden_layer = [embed_]  # width and depth of the attackers
    embed = args.max_doc_length * embed_  # embed value inside network based on the max document length
    embed_size = [embed_]  # debiasing embed size, currently only last layer works

    try:
        os.mkdir(checkpoint_path)
    except:
        print(f'checkpoint path "{checkpoint_path}" already exists')

    # default arguments for the debiasing library Domain Adaptation Library
    with open('debias_config.yml', 'r') as f:
        debias_config = yaml.safe_load(f)

    # change important arguments from the command line
    debias_config['da_type'] = args.debias_loss
    debias_config['da_lambda'] = args.debias_lambda
    debias_config['lambda_auto_schedule'] = args.lambda_scheduler
    debias_config['lambda_final'] = debias_config['da_lambda']
    debias_config['adv_config']['da_net_config']['layers_width'] = embed_size
    debias_method = debias_config['da_type']

    # log dictionary of all the configs to be saved later
    config_dict = {"run": {'experiment_name': experiment_name,
                           'seed': seed,
                           'task_lr': args.lr,  # 2e-5
                           'attr_lr': args.attr_lr,  # 5e-3
                           'lr_scheduler': args.lr_scheduler,
                           'train_epochs': args.train_epochs,
                           'attack_epochs': args.attack_epochs,
                           "target_attribute": args.target_attribute,
                           'loss_function': args.loss_function,
                           'attacker_hidden_layer': attacker_hidden_layer,
                           'device': device,
                           'sequence_training': args.sequence_training},

                   "dataset": {
                       'dataset_type': dataset_type,
                       'max_doc_length': args.max_doc_length,
                       'tokenizer_name': model_name,  # "bert-base-uncased"
                       'add_special_tokens': True,
                       'padding': args.padding,
                       'truncation': True,
                       'return_attention_mask': True,
                       'balance_dataset': args.balance_dataset,
                       'train_path': train_path,
                       'validate_path': validate_path,
                       'test_path': test_path,
                       'dataset_attributes': dataset_attributes,
                       'attributes_unique': attributes_unique,

                   },
                   "data_loader": {
                       'batch_size': args.batch_size,
                       'num_workers': num_workers,
                   },

                   "model": {
                       'checkpoint name': os.path.join(checkpoint_path, experiment_name),
                       'model_name': model_name,  # 'bert-base-uncased'
                       'num_labels': num_labels,
                       'new_embed_layer': args.new_embed_layer,
                       'capture_last_hidden': capture_last_hidden,
                       'trainable_parameters': args.trainable_parameters,
                       'train_layer_norm': args.train_layer_norm,
                       'max_doc_length': args.max_doc_length,
                       'embed': embed,
                       'embed_size': embed_size,
                       'gate_version': args.gate_version,
                       'num_gate_layers': args.num_gate_layers,
                       'adapter': args.adapter,
                   },
                   'debias': debias_config,
                   }

    # change dictionary to class with attributes
    config = dict_to_class(config_dict)

    # loading dataset
    train_df, train_encoder = load_dataset(config.dataset.train_path)
    val_df, _ = load_dataset(config.dataset.validate_path)
    test_df, _ = load_dataset(config.dataset.test_path)

    # notes to log inside weights and biases  (currently empty)
    notes = args.comment

    # Data Loader
    train_loader = DataLoader(CustomDataset(train_df, dataset_attributes=dataset_attributes,
                                            dataset_type=config.dataset.dataset_type,
                                            max_len=config.dataset.max_doc_length,
                                            tokenizer=config.dataset.tokenizer_name,
                                            add_special_tokens=config.dataset.add_special_tokens,
                                            padding=config.dataset.padding, truncation=config.dataset.truncation,
                                            return_attention_mask=config.dataset.return_attention_mask,
                                            balanced=config.dataset.balance_dataset),
                              batch_size=config.data_loader.batch_size, num_workers=config.data_loader.num_workers,
                              shuffle=True)

    val_loader = DataLoader(CustomDataset(val_df, dataset_attributes=dataset_attributes,
                                          dataset_type=config.dataset.dataset_type,
                                          max_len=config.dataset.max_doc_length,
                                          tokenizer=config.dataset.tokenizer_name,
                                          add_special_tokens=config.dataset.add_special_tokens,
                                          padding=config.dataset.padding, truncation=config.dataset.truncation,
                                          return_attention_mask=config.dataset.return_attention_mask, ),
                            batch_size=config.data_loader.batch_size, num_workers=config.data_loader.num_workers,
                            shuffle=False)

    test_loader = DataLoader(CustomDataset(test_df, dataset_attributes=dataset_attributes,
                                           dataset_type=config.dataset.dataset_type,
                                           max_len=config.dataset.max_doc_length,
                                           tokenizer=config.dataset.tokenizer_name,
                                           add_special_tokens=config.dataset.add_special_tokens,
                                           padding=config.dataset.padding, truncation=config.dataset.truncation,
                                           return_attention_mask=config.dataset.return_attention_mask, ),
                             batch_size=config.data_loader.batch_size, num_workers=config.data_loader.num_workers,
                             shuffle=False)

    # train_loader = val_loader

    # Creating Model BertGateV1: Custom Sigmoid is Implemented to give Flexibility of Information Leak During Inference
    #                BertGate: First Idea of the Bert Gate With Temporal Sigmoid Activation Function (DO NOT USE)
    if config.model.gate_version == 1:
        model = BertGateV1(config.model.model_name, config.model.num_labels, config.model.embed_size,
                           target_attribute=config.run.target_attribute,
                           trainable_param=config.model.trainable_parameters, new_embed=config.model.new_embed_layer,
                           train_layer_norm=config.model.train_layer_norm, last_hidden=config.model.capture_last_hidden,
                           max_doc_length=config.model.max_doc_length, num_gate_layers=config.model.num_gate_layers)
    elif config.model.gate_version == 0:
        if config.model.adapter:
            model = AdapterBert(config.model.model_name, config.model.num_labels, config.model.embed_size,
                         target_attribute=config.run.target_attribute,
                         max_doc_length=config.model.max_doc_length, reduction_factor=2)
        else:
            model = Bert(config.model.model_name, config.model.num_labels, config.model.embed_size,
                             target_attribute=config.run.target_attribute,
                             trainable_param=config.model.trainable_parameters, new_embed=config.model.new_embed_layer,
                             train_layer_norm=config.model.train_layer_norm, last_hidden=config.model.capture_last_hidden,
                             max_doc_length=config.model.max_doc_length)
    elif config.model.gate_version == 2:
        model = BertGateV2(config.model.model_name, config.model.num_labels, config.model.embed_size,
                           target_attribute=config.run.target_attribute,
                           trainable_param=config.model.trainable_parameters, new_embed=config.model.new_embed_layer,
                           train_layer_norm=config.model.train_layer_norm, last_hidden=config.model.capture_last_hidden,
                           max_doc_length=config.model.max_doc_length, num_gate_layers=config.model.num_gate_layers)


    model = model.to(device)

    # Create dictionary to save debias models ( in case multiple attribute needs to be debiased)
    debias_models = dict(
        zip(dataset_attributes, [None] * len(dataset_attributes))) if config.run.target_attribute != 'none' else None
    if args.gate_version==0:
        parameter_groups = []
        if 'none' not in config.run.target_attribute:
            da_keys = config.run.target_attribute.split(' ')
        else:
            da_keys = []
        #  Creating Adversarial Models based on the Target Attribute With Optimizer to train them
        if len(da_keys) > 0 and 'none' not in config.run.target_attribute:
            da_ = dict(zip(da_keys, [0 for i in da_keys]))
            for name in da_:
                if args.debias_loss == 'dann':
                    if len(attributes_unique) > 1:
                        debias_config['num_domains'] = attributes_unique[1] if 'age' in name else attributes_unique[0]
                    else:
                        debias_config['num_domains'] = attributes_unique[0]
                debias_models[name] = DABase(embed_size, **debias_config)
                parameter_groups += debias_models[name].get_da_params()

        parameter_groups += model.parameters()
        optimizer = torch.optim.AdamW(list(model.parameters()) + parameter_groups, lr=config.run.task_lr,
                                                weight_decay=0.001)


        # Create Attackers for each attribute in the dataset
        attackers = dict(zip(dataset_attributes, [None] * len(dataset_attributes)))
        att_optimizer = dict(zip(dataset_attributes, [None] * len(dataset_attributes)))
        for name, unique in zip(dataset_attributes, attributes_unique):
            attackers[name] = AttackerModel(embed_size[-1], attacker_hidden_layer,
                                            num_attributes=unique,
                                            activation_function='ReLU').to(device)
            # Attribute Optimizer Should Only be trained by the attacker and the rest of the network should not be trained
            att_optimizer[name] = torch.optim.AdamW(attackers[name].parameters(), lr=args.atk_lr)

        # Specification of the final network and trainable parameters
        total_param, trainable_param, frozen_param, trainable_percentage = model.param_spec()

        # Scheduler for the learning rate of the task model and debiasing models
        if config.run.lr_scheduler:
                scheduler = ExponentialLR(optimizer, gamma=0.98)
        else:
            scheduler = None

    else:
        if 'none' not in config.run.target_attribute:
            optimizer_keys = config.run.target_attribute.split(' ')
        else:
            optimizer_keys = []
        #  Creating Adversarial Models based on the Target Attribute With Optimizer to train them
        if len(optimizer_keys) > 0 and 'none' not in config.run.target_attribute:
            optimizer = dict(zip(optimizer_keys, [0 for i in optimizer_keys]))
            parameter_groups = dict(zip(optimizer_keys, [[] for i in optimizer_keys]))
            for name in optimizer_keys:
                if args.debias_loss == 'dann':
                    if len(attributes_unique) > 1:
                        debias_config['num_domains'] = attributes_unique[1] if 'age' in name else attributes_unique[0]
                    else:
                        debias_config['num_domains'] = attributes_unique[0]
                debias_models[name] = DABase(embed_size, **debias_config)
                debias_parameters = debias_models[name].get_da_params()
                for param_name, parameter in model.named_parameters():
                    if name in param_name:
                        parameter_groups[name].append(parameter)
                optimizer[name] = torch.optim.AdamW(parameter_groups[name] + debias_parameters, lr=config.run.attr_lr,
                                                    weight_decay=0.001)
        else:
            optimizer = {}

        task_parameters = []

        # Task Optimizer Contains All the task Parameters Except the Information Gates During Initial Task Training
        for name, parameter in model.named_parameters():
            if ('gate' not in name) or ('task' in name):
                task_parameters.append(parameter)
        optimizer['task'] = torch.optim.AdamW(task_parameters, lr=config.run.task_lr, weight_decay=0.001)

        # Create Attackers for each attribute in the dataset
        attackers = dict(zip(dataset_attributes, [None] * len(dataset_attributes)))
        att_optimizer = dict(zip(dataset_attributes, [None] * len(dataset_attributes)))
        for name, unique in zip(dataset_attributes, attributes_unique):
            attackers[name] = AttackerModel(embed_size[-1], attacker_hidden_layer,
                                            num_attributes=unique,
                                            activation_function='ReLU').to(device) #lr=config.run.attr_lr, weight_decay=0.001
            # Attribute Optimizer Should Only be trained by the attacker and the rest of the network should not be trained
            att_optimizer[name] = torch.optim.AdamW(attackers[name].parameters(), lr=args.atk_lr)

        # Specification of the final network and trainable parameters
        total_param, trainable_param, frozen_param, trainable_percentage = model.param_spec()

        # Scheduler for the learning rate of the task model and debiasing models
        if config.run.lr_scheduler:
            scheduler = dict(zip(list(optimizer.keys()), [0 for i in optimizer.keys()]))
            for name in scheduler.keys():
                scheduler[name] = ExponentialLR(optimizer[name], gamma=0.98)
        else:
            scheduler = None

    # add specifications to logs
    config_dict['parameter_spec'] = {
        'total_parameters': total_param,
        'frozen_parameters': frozen_param,
        'trainable_percentage': trainable_percentage
    }

    # create folder for the log files


    # initialize weights and biases if --wandb=1.  only imports wandb if user wants to use weights and biases
    if args.wandb:
        try:
            os.mkdir(checkpoint_file)
        except:
            print(f'Warning: experiment with name "{checkpoint_file}" already exists')
        import wandb

        wandb = wandb.init(
            dir=os.path.join(checkpoint_file),
            project=project_name,
            name=f"{experiment_name}",
            notes=f"{notes}",
            config=config_dict, )
    else:
        wandb = None

    # printing necessary information
    # create folder for the experiment if it does not exist

    try:
        try:
            for d_name, d_model in debias_models.items():
                print(f'debias {d_name}:', d_model.nets)
        except:
            print(f'debias {d_name}:', None)
    except:
        print(f'debias model:', None)
    print(model)
    print(optimizer)
    print("Start Date Time =", now)
    print("unique experiment name:", experiment_name)
    print(device)
    print('debiasing models:', debias_models)
    for name, value in train_encoder.items():
        print(f"{name} Mapping:", value)
        config_dict[f'{name}_mapping'] = value
    print('traget attributes:', config.run.target_attribute.split(' '))
    print('mitigation method:', config.debias['da_type'])
    print('mitigation loss weight:', config.debias['da_lambda'])
    print('total number of parameters:', f'{total_param:,}', '\n'
                                                             'trainbale parameters:', f'{trainable_param:,}', '\n'
                                                                                                              'frozen parameters:',
          f'{frozen_param:,}', '\n'
                               'trainbale pecentage:', f'{trainable_percentage:,}%', '\n')

    # call train model function to train the network

    if args.gate_version != 0:
        seq_train_model(model, debias_models, attackers, debias_method, train_loader, val_loader, test_loader,
                        config.run.train_epochs, config.run.attack_epochs, loss_function, optimizer, att_optimizer,
                        scheduler, device, attacker_attribute, config.run.target_attribute, wandb, args.train_type,
                        gate_version=args.gate_version)
    else:
            adv_seq_train(model, debias_models, attackers, debias_method, train_loader, val_loader, test_loader,
                          config.run.train_epochs, config.run.attack_epochs, loss_function, optimizer, att_optimizer,
                          scheduler, device, attacker_attribute, config.run.target_attribute, wandb, args.train_type,
                          gate_version=args.gate_version, sequence_training=args.sequence_training)


    try:
        os.mkdir(checkpoint_file)
    except:
        print(f'Folder Already Exist "{checkpoint_file}"')
    # save model state dict
    model_path = os.path.join(checkpoint_file, 'model.pt')
    torch.save(model.state_dict(), model_path)

    # save attackers state dict
    for key, attacker in attackers.items():
        model_path = os.path.join(checkpoint_file, f'attack_{key}.pt')
        torch.save(attacker.state_dict(), model_path)

    # save config as yml file to be called later.
    with open(os.path.join(checkpoint_file, 'config.yml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)

    if args.wandb:
        wandb.finish()
