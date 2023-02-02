# ModulaizedDebiasing-private
This repository is dedicated to debiasing networks



In order to work with the repository you can use 

1. crete environemtn with `conda env create -f debias_env.yml` to create the 'adapter' environment.
2. use `conda activate adapter` to enter the environment 
3. you need to clone da package seperately from `git clone https://github.com/CPJKU/da`
4. go to the da folder that you cloned and remove python requirement from setup.py 
5. use `python setup.py develop` to setup the da package
6. you can use the code to train your network now:
7. important settings:
`--dataset:` hatespeech, pan16, bios

`--model_name`: base, mini

`--max_doc_lengt=30` , (any integer number)

`--gate_version=0` no gate, 1 gate version 

`--adapter=0` no adapter 1 adapter ( right now if you select adapter you need to have gate_version=0) (#TODO make this config generic)

`--device_id`:  select gpu 

`--sequence_training=1` ( train task then debiasing ) 0 (train task and debiasing simultanously)

`--target_attribute=none` no debiasing ( you should have sequence learning=1) '<<name of the attribute>> (to do debiasing)

`--wandb=0` if you want to log the files on wandb

`--trainable_parameters`: only for gate and baseline selects which part of the network you want to train
  
`--server=1` always 1 if you are using server to run your models 

  
## Example:
  ### Baseline:
  `python experiment_gate.py --dataset=hatespeech --model_name=base --gate_version=0 --target_attribute='none' --debias_lambda=1 --sequence_training=1 --train_epochs=15 --trainable_parameters='encoder+pooler' --num_runs=3  --server=1 --wandb=0 --device_id=0`
  
  ## Adapter finetune (for debias target_attribute should be 'dialect' in case of hatespeech
  `python experiment_gate.py --dataset=hatespeech --model_name=base --gate_version=0 --adapter=1 --target_attribute='none' --debias_lambda=1 --train_epochs=15  --num_runs=3 --server=1 --wandb=0 --sequence_training=1  --device_id=0`

