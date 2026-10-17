# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno
# ----------------------------------------------------------------------------

# Exposed Atlas Machine Learning Models
from .neptune import (
    Neptune,
    NeptuneSystemEmbed3d,
    NeptuneSystemRecover3d,
)
from .sfno import NeptuneOceaNO


from atmocean.utils import (
    clean_state_dict_checkpoint_wrapped_module_from_keys, 
    clean_state_dict_forward_module_from_keys, 
)
from atmocean.macros import ALL_DRIVERS

from lightning.pytorch.utilities.model_summary import ModelSummary
import torch.nn as nn
import torch
import munch
import toml
import glob
import os


def load_model_from_config(
    config_src, 
    main_dir, 
    to_eval: bool = False, 
    log_summary: bool = False, 
    verbose: bool = False, 
    device: str = 'cpu', 
):
    # load config file
    config = munch.munchify(toml.load(config_src))

    # get the run source directory
    run_src = os.path.join(main_dir, config.paths.experiments, config.main.run_name)

    # checkpoints
    checkpoint_rollout_dirs = sorted(glob.glob(os.path.join(main_dir, run_src, 'checkpoints_rollout*')))
    if len(checkpoint_rollout_dirs) != 0:
        checkpoint_dir = checkpoint_rollout_dirs[-1]
    else:
        checkpoint_dir = os.path.join(main_dir, run_src, 'checkpoints')

    # log
    if verbose: print(f'Selected checkpoints directory : {checkpoint_dir}')

    # resume from checkpoint
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*.ckpt')))

    # get the model.bin file
    load_kwargs = dict()
    is_bin = False
    if len(checkpoints) > 0:
        checkpoint = checkpoints[-1]
        if os.path.isdir(checkpoint):
            is_bin = True
            checkpoint = os.path.join(checkpoint, 'model.bin')
            if os.path.isdir(checkpoint):
                checkpoint = os.path.join(checkpoint, 'pytorch_model.bin')
        else:
            load_kwargs['weights_only'] = False
    else:
        checkpoint = None
    # load the PyTorch model
    checkpoint_state = torch.load(checkpoint, **load_kwargs)
    if not is_bin:
        checkpoint_state = checkpoint_state['state_dict']
    checkpoint_state = clean_state_dict_checkpoint_wrapped_module_from_keys(checkpoint_state)
    checkpoint_state = clean_state_dict_forward_module_from_keys(checkpoint_state)

    # log
    if verbose: print(f"State dict loaded from {checkpoint}")

    # add arguments to core configuration
    model_args = {'drivers': ALL_DRIVERS, 'insteps': config.train.insteps, 'outsteps': config.train.outsteps, **dict(config.model.args)}

    # init model
    model:Neptune = eval(config.model.cls)(**model_args)
    if hasattr(config.train, 'rollout'):
        rollout = config.train.rollout
    else:
        rollout = int(checkpoint_dir.split('_')[-1])
    if hasattr(config.train, 'is_residual'):
        is_residual = config.train.is_residual
    else:
        is_residual = False
    # add attributes to the model
    model.set_attributes(
        optimizer = None, 
        scheduler = None, 
        loss = None, 
        metrics = None, 
        insteps = config.train.insteps, 
        outsteps = config.train.outsteps, 
        lead_time = config.train.lead_time, 
        rollout = rollout, 
        is_rollout = False, 
        is_ens = True if 'ensemble_size' in model_args.keys() else False, 
        is_residual = is_residual, 
    )
    # if is_older_state_dict_version(checkpoint_state):
    #     checkpoint_state = convert_retro_compatible_model_state_dict(checkpoint_state)

    # load model weights
    mkeys, ukeys = model.load_state_dict(checkpoint_state, strict=False)
    if (not "embedding.weights.noise" in mkeys and len(mkeys) != 0):
        is_swin = False
        for key in mkeys:
            if 'swin_unet_transformer' in key:
                is_swin = True
                break
        if not is_swin:
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\nMissing keys: {mkeys}\nUnexpected keys: {ukeys}")

    # log
    if verbose: print(f"Model initialized")
    
    if to_eval:
        # set the model to evaluation
        model.eval()
        # disable gradient computation
        torch.set_grad_enabled(False)
    
    # log
    if verbose: print(f"Disabled gradient computation")
    
    # log
    if log_summary:
        # summary of the model structure
        print(ModelSummary(model, max_depth=2))
    
    # put the model to device
    model = model.to(device)
    return model, config, checkpoint_state
