# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno
# ----------------------------------------------------------------------------

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Python libraries import
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from lightning.pytorch.utilities.model_summary import ModelSummary
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning.loggers import MLFlowLogger
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from lightning import Trainer
import argparse
import logging
import shutil
import mlflow
import torch
import munch
import toml
import glob
import sys
import os
import warnings
warnings.simplefilter("ignore")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program Info + Parse CLI params
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

PROGRAM_NAME = r'''
    _   __           __                 
   / | / /__  ____  / /___  ______  ___ 
  /  |/ / _ \/ __ \/ __/ / / / __ \/ _ \
 / /|  /  __/ /_/ / /_/ /_/ / / / /  __/
/_/ |_/\___/ .___/\__/\__,_/_/ /_/\___/ 
          /_/                           
                                        
'''
PROGRAM_DESCRIPTION = 'Multi-Node Training of Atlas case study'
PROGRAM_ARGUMENTS = [
    [('-c', '--config'), {'type':str, 'help':'Configuration file for this program', 'required':True}],
    [('-n', '--num_nodes'), {'type':int, 'help':'Number of cluster GPU nodes', 'required':True}],
    [('-d', '--devices'), {'type':int, 'help':'Number of GPU devices per node', 'required':True}],
    [('-o', '--ocean'), {'type':str, 'default':'default', 'choices':['default','oras5','cglors','glorys'], 'help':'Select the oceanic dataset to use', 'required':True}],
    [('-a', '--atmos'), {'type':str, 'default':'era5', 'choices':['era5'], 'help':'Select the atmospheric dataset to use', 'required':True}],
]

parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=PROGRAM_DESCRIPTION)
for arg in PROGRAM_ARGUMENTS:
    names, kwargs = arg
    sname, lname = names
    parser.add_argument(sname, lname, **kwargs)
args = parser.parse_args()

# read configuration file for this execution
config = munch.munchify(toml.load(args.config))

# get number of nodes and devices for this run
num_nodes = args.num_nodes
num_devices_per_node = args.devices
ocean = args.ocean
atmos = args.atmos

# set enviroment variables 
os.environ['ocean'] = ocean.upper()
os.environ['atmos'] = atmos.upper()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Custom libraries import
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import sys
LIB_DIR = 'library'
sys.path.append(LIB_DIR)

from atmocean.losses import *
from atmocean.models import *
from atmocean.macros import ALL_DRIVERS
from atmocean.dataset import NeptuneDataset
from atmocean.service import DiscordLogging
from atmocean.callbacks import BenchmarkCSV, DiscordLog
from atmocean.utils import log_message, get_num_parameters

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Parse Config params for this experiment
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# main params
experiment_class = config.main.experiment_class
run_name = config.main.run_name

# data path
data_dir = config.paths.data
experiments_dir = config.paths.experiments

# random seed for reproducibility
seed = config.random.seed

# pytorch
dtype = eval(config.torch.dtype)
matmul_precision = config.torch.matmul_precision

# lightning
accelerator = config.lightning.accelerator
precision = config.lightning.precision

# model
model_cls = eval(config.model.cls)
model_args = dict(config.model.args)

# data
drivers = dict(config.data.drivers)
fill_value = config.data.fill_value
epsilon = config.data.epsilon
perlin_noise = config.data.perlin_noise
avg_atmos = config.data.avg_atmos if hasattr(config.data, "avg_atmos") else False

# metrics
metrics_list = [eval(mts) for mts in config.model.metrics]

# loss
loss_cls = eval(config.loss.cls)
loss_cls_name = config.loss.cls
loss_args = dict(config.loss.args)

# optimizer
optimizer_cls = eval(config.optimizer.cls)
optimizer_args = dict(config.optimizer.args)

# scheduler
scheduler_warmup_args = config.scheduler.warmup
scheduler_annealing_args = config.scheduler.annealing

# external loggers (Discord/Slack)
discord_webhook_url = config.logger.discord_webhook if hasattr(config.logger, 'discord_webhook') else None

# get train configuration
epochs = config.train.epochs
batch_size = config.train.batch_size
accumulation_steps = config.train.accumulation_steps
insteps = config.train.insteps
outsteps = config.train.outsteps
lead_time = config.train.lead_time
rollout = config.train.rollout
is_residual = config.train.is_residual
use_noise = config.train.use_noise
drop_remainder = config.train.drop_remainder
num_workers = config.train.num_workers if hasattr(config.train, "num_workers") else 0
shuffle = config.train.shuffle
train_years = config.train.train_years
valid_years = config.train.valid_years

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  MLFlow initialization
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# init mlflow logger
mlflow_logger = MLFlowLogger(experiment_name=experiment_class, run_name=run_name)
mlflow_logger.log_hyperparams({'RUN TYPE' : 'PRE-TRAINING'})
mlflow_logger.log_hyperparams(config.train)
mlflow_logger.log_hyperparams({'model_class': config.model.cls, **config.model.args})
mlflow_logger.log_hyperparams({'optim_class': config.optimizer.cls, **config.optimizer.args})
mlflow.set_experiment(experiment_class)
mlflow.start_run(mlflow_logger.run_id)
mlflow.log_artifact(local_path=args.config)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Environment setup
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# get the source directory of the run
run_dir = os.path.join(experiments_dir, run_name)
config_src = args.config
config_dst = os.path.join(run_dir, 'configuration-pretrain.toml')
library_src = LIB_DIR
library_dst = os.path.join(run_dir, 'library')

# get the total number of GPUs and number of dataset samples
num_gpus = num_nodes * num_devices_per_node

# scale the training parameters with the number of GPUs
local_batch_size = batch_size
global_batch_size = accumulation_steps * local_batch_size * num_gpus

# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision(matmul_precision)

# define important directories
log_dir = os.path.join(run_dir, 'logs')
logging_dir = os.path.join(log_dir, 'logging')
checkpoint_dir = os.path.join(run_dir, 'checkpoints')

# define important filenames
benchmark_csv = os.path.join(run_dir, 'benchmark.csv')

# create experiment directory
os.makedirs(run_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# save training hyperparameters
shutil.copy(
    src=config_src, 
    dst=config_dst, 
)

# Save a snapshot of the library in the experiment directory. 
# This helps us to recover the code in case we need again to 
# load the experiment after the code has been updated.
shutil.copytree(
    src=library_src, 
    dst=library_dst, 
    dirs_exist_ok=True, 
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program variables
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

# eventually resume from checkpoint
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*.ckpt')))

# checkpoint is the last of the checkpoints
checkpoint = checkpoints[-1] if len(checkpoints) > 0 else None

# add arguments to core configuration
model_args = {**model_args, 'drivers': ALL_DRIVERS, 'insteps': insteps, 'outsteps': outsteps}

# determine if we have ensemble or not
is_ens = True if 'ensemble_size' in model_args.keys() else False

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Setup Parallel Trainer
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# define user callbacks
callbacks = [
    ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:04d}-val_loss-{val_loss:.2f}",
        monitor='val_loss',
        save_last=True,
        save_top_k=5,
        auto_insert_metric_name=False,
    ), 
    BenchmarkCSV(
        filename=benchmark_csv,
    ), 
    DiscordLog(
        model_name=run_name,
        webhook_url=discord_webhook_url,
        benchmark_csv=benchmark_csv,
        msg_every_n_epochs=1,
        plot_every_n_epochs=10,
    ), 
]

# init distribution strategy
# strategy = DeepSpeedStrategy(
#     zero_optimization=True,
#     stage=1,
#     load_full_weights=True,
# ) if accelerator=='cuda' else 'auto'
strategy = DDPStrategy(find_unused_parameters=True) if accelerator=='cuda' else 'auto'

# init trainer
trainer = Trainer(
    accelerator=accelerator,
    strategy=strategy,
    devices=num_devices_per_node,
    num_nodes=num_nodes,
    precision=precision,
    callbacks=callbacks,
    max_epochs=epochs,
    logger=mlflow_logger,
    enable_checkpointing=True,
    enable_progress_bar=True,
    accumulate_grad_batches=accumulation_steps,
    use_distributed_sampler=False,
    default_root_dir=run_dir,
    num_sanity_val_steps=0,
    enable_model_summary=False,
    profiler=None,
)

# init the seed
seed_everything(seed)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Init Logger
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# initialize logger
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", 
                    filename=f'{logging_dir}/dev-{trainer.global_rank}.log', 
                    filemode='w', 
                    level=logging_level, 
                    datefmt='%Y-%m-%d %H:%M:%S', 
                    force=True)

# initialize discord logger
discord_logger = DiscordLogging(webhook=discord_webhook_url, global_rank=trainer.global_rank)

# define the logging functions
log_fns = [logging.info, discord_logger.log]
# log_fns = [logging.info]

# log
log_message(log_fns, f"Logger initialized. Starting the execution")
log_message(log_fns, f"   World size  : {trainer.world_size}")
log_message(log_fns, f"   Node rank   : {trainer.node_rank}")
log_message(log_fns, f"   Global rank : {trainer.global_rank}")
log_message(log_fns, f"   Local rank  : {trainer.local_rank}\n")
log_message(log_fns, f'---------------- Hyperparameters ----------------')
log_message(log_fns, f'   num_gpus : {num_gpus}')
log_message(log_fns, f'   epochs : {epochs}')
log_message(log_fns, f'   batch_size : {batch_size}')
log_message(log_fns, f'   local_batch_size : {local_batch_size}')
log_message(log_fns, f'   accumulation_steps : {accumulation_steps}')
log_message(log_fns, f'   global_batch_size : {global_batch_size}')
log_message(log_fns, f'   insteps : {insteps}')
log_message(log_fns, f'   outsteps : {outsteps}')
log_message(log_fns, f'   lead_time : {lead_time}')
log_message(log_fns, f'   rollout : {rollout}')
log_message(log_fns, f'   drop_remainder : {drop_remainder}')
log_message(log_fns, f'   shuffle : {shuffle}')
log_message(log_fns, f'   train_years : {train_years}')
log_message(log_fns, f'   valid_years : {valid_years}')
log_message(log_fns, f'------------------------------------------------\n')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Initialize ML Model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# init model
model:Neptune = model_cls(**model_args)

# get the number of parameters of this model
n_params = get_num_parameters(model)

# log to mlflow
mlflow.log_param('n_params', n_params)

try:
    model.configure_activation_checkpointing()
    log_message(log_fns, f'Activation checkpointing correctly activated')
except Exception as e:
    log_message(log_fns, f'Error occurred during configure_activation_checkpointing(): {e}')

# init loss and metrics
if loss_cls_name == 'WeightedMAELoss':
    loss_fn = loss_cls(drivers=drivers, **loss_args)
else:
    loss_fn = loss_cls(**loss_args)
metrics_fns = [mts() for mts in metrics_list]

# init optimizer
optimizer = optimizer_cls(model.parameters(), **optimizer_args)

# init scheduler
verbose = True if trainer.global_rank == 0 else False
warmup_epochs = scheduler_warmup_args['total_iters']
warmup = lr_scheduler.LinearLR(
    optimizer = optimizer, 
    **scheduler_warmup_args, 
    # verbose = verbose, 
)
annealing = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer = optimizer, 
    T_0 = epochs+1, 
    **scheduler_annealing_args, 
    # verbose = verbose, 
)
scheduler = lr_scheduler.SequentialLR(
    optimizer = optimizer, 
    schedulers = [warmup, annealing], 
    milestones = [warmup_epochs], 
    # verbose = verbose, 
)

# add attributes to the model
model.set_attributes(
    optimizer = optimizer, 
    scheduler = scheduler, 
    loss = loss_fn, 
    metrics = metrics_fns, 
    insteps = insteps, 
    outsteps = outsteps, 
    lead_time = lead_time, 
    rollout = rollout, 
    is_rollout = False, # this is a pretraining, therefore we do not have rollout
    is_ens = is_ens, 
    is_residual = is_residual, 
    use_noise = use_noise, 
)

# log summary
if not trainer.global_rank:
    print(ModelSummary(model=model, max_depth=2))
    log_message(log_fns, ModelSummary(model=model, max_depth=2))

# log
log_message(log_fns, f'Model and Optimizer inizialized')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Load dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# initialize training and validation datasets
train_dataset = NeptuneDataset(
    src = data_dir, 
    years = train_years, 
    drivers = drivers, 
    insteps = insteps, 
    outsteps = outsteps, 
    lead_time = lead_time, 
    rollout = rollout, 
    fill_value = fill_value, 
    scale = True, 
    noise = perlin_noise, 
    epsilon = epsilon, 
    dtype = dtype, 
    avg_atmos = avg_atmos, 
)
valid_dataset = NeptuneDataset(
    src = data_dir, 
    years = valid_years, 
    drivers = drivers, 
    insteps = insteps, 
    outsteps = outsteps, 
    lead_time = lead_time, 
    rollout = rollout, 
    fill_value = fill_value, 
    scale = True, 
    noise = perlin_noise, 
    epsilon = epsilon, 
    dtype = dtype, 
    avg_atmos = avg_atmos, 
)

# log
log_message(log_fns, f'Train ({len(train_dataset)}) and valid ({len(valid_dataset)}) datasets inizialized')

# init train and val samplers
train_sampler = DistributedSampler(
    train_dataset, 
    num_replicas = trainer.world_size, 
    rank = trainer.global_rank, 
    drop_last = drop_remainder, 
    seed = seed, 
    shuffle = shuffle, 
)
valid_sampler = DistributedSampler(
    valid_dataset, 
    num_replicas = trainer.world_size, 
    rank = trainer.global_rank, 
    drop_last = drop_remainder, 
    seed = seed, 
    shuffle = False, 
)

# log
log_message(log_fns, f'Dataset samplers initialized')

# load dataloader
train_loader = DataLoader(
    train_dataset,
    sampler = train_sampler,
    batch_size = batch_size,
    drop_last = drop_remainder,
    num_workers = num_workers,
)
valid_loader = DataLoader(
    valid_dataset,
    sampler = valid_sampler,
    batch_size = batch_size,
    drop_last = drop_remainder,
    num_workers = num_workers,
)

# log
log_message(log_fns, f'Dataloaders created')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Train the model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# log
log_message(log_fns, f'Training the model')

# fit the model
trainer.fit(
    model = model, 
    train_dataloaders = train_loader, 
    val_dataloaders = valid_loader, 
    ckpt_path = checkpoint, 
)

# log
log_message(log_fns, f'Model trained')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program End
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# log
log_message(log_fns, f'Program completed')

# close program
exit(1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 