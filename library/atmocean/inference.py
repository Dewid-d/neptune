import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import munch
import toml
import os

from atmocean.macros import ALL_DRIVERS, ALL_DATA_VARS, MEAN_DATA_VARS, STD_DATA_VARS
from atmocean.forecast import deaggregate_systems
from atmocean.scaling import unnormalize
from atmocean.models import *


class NeptuneInference():
    def __init__(self,
        cfg_src: str,
        weights_src: str,
        sealand_masks: tuple,
        device: str = 'cpu',
        dtype = torch.float32,
        grad_enabled: bool = False,
    ):
        self.dtype = dtype
        self.device = device
        # initialize the ML model
        model, config, state_dict = self.load_model_from_config(
            model_weights_src=weights_src,
            config_src=cfg_src,
            log_summary=True,
            verbose=True,
            device=device,
        )
        self.model = model
        self.model.ensemble_size = 1
        # get information from the loaded configuration
        self.insteps = config.train.insteps
        self.outsteps = config.train.outsteps
        self.lead_time = config.train.lead_time
        self.drivers = config.data.drivers
        self.fill_value = config.data.fill_value
        self.epsilon = config.data.epsilon
        self.sealand_mask_3d, self.sealand_mask_2d = sealand_masks
        # gradients should be disabled during inference
        torch.set_grad_enabled(grad_enabled)

    def model_summary(self, model: nn.Module, max_depth: int = 2) -> str:
        """
        Print a model summary similar to Lightning's ModelSummary.

        Args:
            model    : the nn.Module to summarise.
            max_depth: how many levels of submodules to show (default 2).

        Returns:
            The summary string (also printed to stdout).
        """

        def _fmt_num(n: int) -> str:
            """Format large numbers with K / M / B suffix."""
            if n >= 1_000_000_000:
                return f"{n / 1_000_000_000:.1f} B"
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f} M"
            if n >= 1_000:
                return f"{n / 1_000:.1f} K"
            return str(n)

        rows = []

        def _add_rows(module: nn.Module, prefix: str, depth: int):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if depth <= max_depth:
                    n_params  = sum(p.numel() for p in child.parameters())
                    trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
                    rows.append((full_name, child.__class__.__name__, n_params, trainable))
                    _add_rows(child, full_name, depth + 1)

        _add_rows(model, "", 1)

        # totals
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable    = total_params - trainable_params

        # column widths
        col_name  = max((len(r[0]) for r in rows), default=10)
        col_type  = max((len(r[1]) for r in rows), default=10)
        col_name  = max(col_name, len("Name"))
        col_type  = max(col_type, len("Type"))
        col_par   = max(len("Params"), 10)

        sep   = f"{'─' * (col_name + col_type + col_par * 2 + 13)}"
        fmt   = f"{{:<{col_name}}}  {{:<{col_type}}}  {{:>{col_par}}}  {{:>{col_par}}}"
        lines = []

        lines.append(sep)
        lines.append(f"  Model: {model.__class__.__name__}")
        lines.append(sep)
        lines.append(fmt.format("Name", "Type", "Params", "Trainable"))
        lines.append(sep)

        for name, cls, n, t in rows:
            lines.append(fmt.format(name, cls, _fmt_num(n), _fmt_num(t)))

        lines.append(sep)
        lines.append(f"{'Total params':>{col_name + col_type + 4}}  {_fmt_num(total_params):>{col_par}}")
        lines.append(f"{'Trainable params':>{col_name + col_type + 4}}  {_fmt_num(trainable_params):>{col_par}}")
        lines.append(f"{'Non-trainable params':>{col_name + col_type + 4}}  {_fmt_num(non_trainable):>{col_par}}")
        lines.append(sep)

        summary = "\n".join(lines)
        print(summary)
        return summary

    def load_model_from_config(self,
        model_weights_src,
        config_src,
        log_summary: bool = False,
        verbose: bool = False,
        device: str = 'cpu',
    ):
        # load config file
        cfg = munch.munchify(toml.load(config_src))

        # load the PyTorch model
        state_dict = torch.load(model_weights_src, weights_only=False, map_location=device)

        # clean the checkpoint state
        state_dict = clean_state_dict_checkpoint_wrapped_module_from_keys(state_dict)
        state_dict = clean_state_dict_forward_module_from_keys(state_dict)

        # log
        if verbose: print(f"State dict loaded from {model_weights_src}")

        # add arguments to core configuration
        model_args = {'drivers': ALL_DRIVERS, 'insteps': cfg.train.insteps, 'outsteps': cfg.train.outsteps, **dict(cfg.model.args)}

        # init model
        model:Neptune = eval(cfg.model.cls)(**model_args)

        if hasattr(cfg.train, 'is_residual'):
            is_residual = cfg.train.is_residual
        else:
            is_residual = False

        # add attributes to the model
        model.set_attributes(
            optimizer = None,
            scheduler = None,
            loss = None,
            metrics = None,
            insteps = cfg.train.insteps,
            outsteps = cfg.train.outsteps,
            lead_time = cfg.train.lead_time,
            rollout = 1,
            is_rollout = False,
            is_ens = True if 'ensemble_size' in model_args.keys() else False,
            is_residual = is_residual,
        )

        # load model weights
        mkeys, ukeys = model.load_state_dict(state_dict['state_dict'], strict=False)
        if (not "embedding.weights.noise" in mkeys and len(mkeys) != 0):
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\nMissing keys: {mkeys}\nUnexpected keys: {ukeys}")

        # log
        if verbose: print(f"Model initialized")

        # log
        if log_summary:
            # summary of the model structure
            self.model_summary(model, max_depth=2)

        # put the model to device
        model = model.to(device)

        # return
        return model, cfg, state_dict

    def get_day_of_year(self, fname):
        '''Day and Month of the Year (DMoY)'''
        date = os.path.basename(fname).split('.npy')[0]
        date = pd.to_datetime(date)
        day_of_year = pd.Period(date, freq='d').day_of_year
        return day_of_year

    def add_day_of_year(self, files):
        doy = torch.cat([
            torch.as_tensor([self.get_day_of_year(file)]) 
            for file in files
        ])
        return doy

    def load_unpack(self, files, fill_value=0.0, epsilon=1e-7):
        tdim = 1
        # read all the timesteps and stack them
        data = torch.stack([torch.as_tensor(np.load(file, mmap_mode='c'), dtype=self.dtype) for file in files], dim=tdim)
        # get only the data belonging to the data vars
        batch = torch.stack([self.prepare_data(data[i], var, fill_value, epsilon) for i,var in enumerate(ALL_DATA_VARS)])
        return batch

    def prepare_data(self, data, var, fill_value=0.0, epsilon=1e-7):
        # scale the data with its own scaler
        data = (data - MEAN_DATA_VARS[var]) / (STD_DATA_VARS[var] + epsilon)
        # remove nan values
        data = torch.nan_to_num(data, nan=fill_value)
        return data
    
    def load_data(self,
        data_dir,
        dates,
        fill_value: float = 0.0,
        epsilon: float = 1e-7,
    ):
        # two input files
        files = [os.path.join(data_dir, f'{date.replace("-", "")}.npy') for date in dates]
        # read all the timesteps and stack them
        data = torch.stack([torch.as_tensor(np.load(file, mmap_mode='c'), dtype=self.dtype) for file in files], dim=1)
        # get only the data belonging to the data vars
        x = torch.stack([self.prepare_data(data[i], var, fill_value, epsilon) for i,var in enumerate(ALL_DATA_VARS)])
        # day of year
        t = self.add_day_of_year(files)
        # add batch dimension and move to device
        x = x.unsqueeze(0).to(self.device)
        t = t.unsqueeze(0).to(self.device)
        # create noise conditioning
        D = self.model.dim # latent dimension
        _, _, _, H, W = x.shape
        c = torch.randn((self.insteps, D, H, W), dtype=self.dtype, device=self.device)
        # store the data in the class
        return x, c, t
    
    def predict(self,
        initial_conditions,
        noise_conditioning,
        initial_date,
        epsilon: float = 1e-7,
    ):
        drivers = self.drivers
        # deaggregate variables into atmosphere-ocean components
        xa3d, xa2d, xo3d, xo2d = deaggregate_systems(initial_conditions,
            drivers['input']['atmos']['fields_3d'], drivers['input']['atmos']['fields_2d'],
            drivers['input']['ocean']['fields_3d'], drivers['input']['ocean']['fields_2d'],
        )
        preds = self.model(
            x=(xa3d,xa2d,xo3d,xo2d), # input data
            t=initial_date,          # day of year
            c=noise_conditioning,    # noise (if None, the network will create it)
            drivers=drivers,         # I/O drivers
        )
        # get each component
        dya3d, dya2d, dyo3d, dyo2d = preds
        # since Neptune predicts only the ocean, we will only have dyo3d and dyo2d
        # sum the input ocean to get the actual field
        ya3d = xa3d + dya3d if dya3d is not None else None
        ya2d = xa2d + dya2d if dya2d is not None else None
        yo3d = xo3d + dyo3d if dyo3d is not None else None
        yo2d = xo2d + dyo2d if dyo2d is not None else None
        # rescale the data to their physical value
        ya3d, ya2d, yo3d, yo2d = self.rescale((ya3d, ya2d, yo3d, yo2d), drivers, epsilon)
        # apply the sealand mask to remove data on the land
        yo3d = self.apply_sealand_mask(yo3d.cpu(), self.sealand_mask_3d).cpu().numpy()
        yo2d = self.apply_sealand_mask(yo2d.cpu(), self.sealand_mask_2d).cpu().numpy()
        # create a predictions dictionary
        predictions = {}
        for i,field in enumerate(drivers['output']['ocean']['fields_3d']):
            predictions[field] = yo3d[0,i]
        for i,field in enumerate(drivers['output']['ocean']['fields_2d']):
            predictions[field] = yo2d[0,i,:,0]
        return predictions
    
    def apply_sealand_mask(self, data, mask):
        # data: (b, d, t, z, h, w), mask: (z, h, w), True = valido (oceano)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=data.device)
        return data.masked_fill(~mask, float('nan'))
    
    def rescale(self, x, drivers: dict, epsilon: float):
        a3d, a2d, o3d, o2d = x # B x V x T x Z x H x W
        af3d, af2d, of3d, of2d = (drivers['output']['atmos']['fields_3d'], 
                                  drivers['output']['atmos']['fields_2d'], 
                                  drivers['output']['ocean']['fields_3d'], 
                                  drivers['output']['ocean']['fields_2d'])
        for i,field in enumerate(af3d):
            if a3d is not None:
                for l,level in enumerate(drivers['levels']):
                    a3d[:,i,:,l] = unnormalize(a3d[:,i,:,l], f'{field}_{level}', epsilon)
        for i,field in enumerate(af2d):
            if a2d is not None:
                a2d[:,i,:] = unnormalize(a2d[:,i,:], field, epsilon)
        for i,field in enumerate(of3d):
                if o3d is not None:
                    for d,depth in enumerate(drivers['depths']):
                        o3d[:,i,:,d] = unnormalize(o3d[:,i,:,d], f'{field}_{int(depth)}', epsilon)
        for i,field in enumerate(of2d):
            if o2d is not None:
                o2d[:,i,:] = unnormalize(o2d[:,i,:], field, epsilon)
        return (a3d, a2d, o3d, o2d)
