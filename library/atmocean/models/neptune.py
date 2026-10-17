# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno
# ----------------------------------------------------------------------------

from atmocean.forecast import (
    shift_tensors, 
    select_input, 
    select_output, 
    mean_prediction, 
    update_input_with_forcing,
    cum_sum_data,
)
from atmocean.utils import (
    deaggregate_systems_by_drivers, 
    deaggregate_systems, 
    aggregate_systems, 
    realign_drivers_dict, 
    physical_padding
)
from atmocean.macros import ALL_DRIVERS

import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
import lightning as L
import torchcfm
import torch
import math
import gc


class Neptune(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.callback_metrics = {}
        self.is_rollout = False # whether this run is a rollout
        self.insteps = None     # number of input time-steps
        self.outsteps = None    # number of output time-steps
        self.rollout = None     # number of rollout steps before computing gradient
        self.lead_time = None   # delta t between each time-step
        self.is_ens = None      # whether it is an ensemble model
        self.ensemble_size = None # the number of ensembles that the model produces
        self.is_residual = None # whether it is a residual model
        self._loss = {'train': {'sum': 0, 'steps': 0}, 
                      'val':   {'sum': 0, 'steps': 0}}
    
    def preprocess(self, x, ps):
        if not (type(x) == list or type(x) == tuple):
            raise ValueError('x must be passed as a tensor containing [a3d, a2d, o3d, o2d]')
        for xi in x:
            if xi is not None: 
                rs = xi.shape[-4:]
        x = [physical_padding(xi, ps[-2:], 'b d t z h w', False, rs[-2:]) if xi is not None else None for xi in x]
        return x, rs
    
    def postprocess(self, x, rs, ps):
        x = [physical_padding(xi, ps[-2:], 'b d t z h w', True, rs[-2:]) if xi is not None else None for xi in x]
        return x
    
    def deaggregate_systems(self, x, drivers):
        # get the variables for each system
        atmos_3d_vars = drivers['input']['atmos']['fields_3d']
        atmos_2d_vars = drivers['input']['atmos']['fields_2d']
        ocean_3d_vars = drivers['input']['ocean']['fields_3d']
        ocean_2d_vars = drivers['input']['ocean']['fields_2d']
        
        if type(x) == list or type(x) == tuple:
            a3d, a2d, o3d, o2d = x    
        else:
            a3d, a2d, o3d, o2d = deaggregate_systems(x, atmos_3d_vars, atmos_2d_vars, ocean_3d_vars, ocean_2d_vars)
        res_a3d, res_a2d, res_o3d, res_o2d = (
            a3d.shape[2:] if a3d is not None else None, 
            a2d.shape[2:] if a2d is not None else None, 
            o3d.shape[2:] if o3d is not None else None, 
            o2d.shape[2:] if o2d is not None else None, 
        )
        return (a3d, a2d, o3d, o2d), (res_a3d, res_a2d, res_o3d, res_o2d)

    def generic_step(self, batch, batch_idx, step_type):
        tdim = 2
        data, days, drivers, files = batch
        # prepare internal variables
        rollout, insteps, outsteps, is_ens, ensemble_size = (
            self.rollout,
            self.insteps,
            self.outsteps,
            self.is_ens,
            self.ensemble_size,
        )
        # realign drivers dictionary
        drivers = realign_drivers_dict(drivers)
        # get input fields
        atmos_input_fields_3d = drivers['input']['atmos']['fields_3d']
        atmos_input_fields_2d = drivers['input']['atmos']['fields_2d']
        ocean_input_fields_3d = drivers['input']['ocean']['fields_3d']
        ocean_input_fields_2d = drivers['input']['ocean']['fields_2d']
        # get output fields
        atmos_output_fields_3d = drivers['output']['atmos']['fields_3d']
        atmos_output_fields_2d = drivers['output']['atmos']['fields_2d']
        ocean_output_fields_3d = drivers['output']['ocean']['fields_3d']
        ocean_output_fields_2d = drivers['output']['ocean']['fields_2d']
        # deaggregate variables
        X_atmos_3d, X_atmos_2d, X_ocean_3d, X_ocean_2d = deaggregate_systems(
            data, 
            atmos_input_fields_3d, 
            atmos_input_fields_2d, 
            ocean_input_fields_3d, 
            ocean_input_fields_2d, 
        )
        Y_atmos_3d, Y_atmos_2d, Y_ocean_3d, Y_ocean_2d = deaggregate_systems(
            data, 
            atmos_output_fields_3d, 
            atmos_output_fields_2d, 
            ocean_output_fields_3d, 
            ocean_output_fields_2d, 
        )

        # get some info from data
        device, dtype = data.device, data.dtype
        batch_size = data.shape[0]
        loss = torch.tensor(0.0, device=device)

        # get only the depth levels that we actually use
        depths, levels = drivers['depths'], drivers['levels']
        all_depths, all_levels = ALL_DRIVERS['depths'], ALL_DRIVERS['levels']
        depth_ids = torch.as_tensor([i for i,d in enumerate(all_depths) if int(d) in [int(dpt) for dpt in depths]]).to(device)
        level_ids = torch.as_tensor([i for i,l in enumerate(all_levels) if int(l) in [int(lvl) for lvl in levels]]).to(device)
        X_atmos_3d = X_atmos_3d.index_select(dim=tdim+1, index=level_ids) if X_atmos_3d is not None else None
        Y_atmos_3d = Y_atmos_3d.index_select(dim=tdim+1, index=level_ids) if Y_atmos_3d is not None else None
        X_ocean_3d = X_ocean_3d.index_select(dim=tdim+1, index=depth_ids) if X_ocean_3d is not None else None
        Y_ocean_3d = Y_ocean_3d.index_select(dim=tdim+1, index=depth_ids) if Y_ocean_3d is not None else None

        # create noise tensors to inject into the networks
        if is_ens and self.__class__.__name__ in ["AtlasOceaNOv2", "AtlasOceaNOv3"]:
            noise_conds = []
            D = self.dim
            B, _, _, H, W = data.shape
            if self.conv_module == "spherical":
                nshape = (B * insteps, D, H, W)
            elif self.conv_module == "spectral":
                nshape = (B, D, insteps, H, W)
            for _ in range(self.ensemble_size):
                noise_conds.append(
                    torch.randn(nshape, device=device, dtype=dtype)
                )
        else:
            noise_conds = [None]
        # create the prediction ensemble list
        y_pred_ensemble_list = []
        # ensemble iteration
        for e in range(ensemble_size):
            # create the rollout lists
            y_pred_rollout_list = []
            y_true_rollout_list = []
            # rollout iteration
            for r in range(rollout):
                # determine the offset for the progressive 
                # rollout shifts in the data indices
                off = r if outsteps > 1 else 0
                # get index i
                i = insteps - 1 + r
                # only on the first iteration
                if r == 0:
                    # get input data from dataset
                    xa3d, xa2d, xo3d, xo2d = select_input(
                        (X_atmos_3d, X_atmos_2d, X_ocean_3d, X_ocean_2d), 
                        i+off, insteps
                    )
                # we are in autoregression
                else:
                    # shift the temporal window by outsteps
                    xa3d, xa2d, xo3d, xo2d = shift_tensors(
                        (xa3d, xa2d, xo3d, xo2d), outsteps, tdim, 
                    )
                    # substitute the last element with the prediction
                    xa3d, xa2d, xo3d, xo2d = update_input_with_forcing(
                        (xa3d, xa2d, xo3d, xo2d), 
                        (ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred), 
                        (X_atmos_3d, X_atmos_2d, X_ocean_3d, X_ocean_2d), 
                        i, insteps, outsteps, drivers
                    )
                # get static variables
                t = days[:, i+1-insteps+off : i+1+off]
                # get ground truth
                ya3d, ya2d, yo3d, yo2d = select_output(
                    (Y_atmos_3d, Y_atmos_2d, Y_ocean_3d, Y_ocean_2d), 
                    i+off, outsteps
                )
                # reaggregate to get the correct number of variables
                y_true = aggregate_systems(
                    ya3d, ya2d, yo3d, yo2d, 
                    is_ens = False
                )
                # forward pass
                ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred = self(
                    x=(xa3d, xa2d, xo3d, xo2d), 
                    t=t, r=r, c=noise_conds[e],
                    drivers=drivers,
                    out_attn=False,
                    agg_systems=False,
                    use_noise=self.use_noise,
                )
                # remove ensemble dimension if present
                if ya3d_pred is not None: ya3d_pred = ya3d_pred.squeeze(dim=-1) if ya3d_pred.shape[-1] == 1 else ya3d_pred
                if ya2d_pred is not None: ya2d_pred = ya2d_pred.squeeze(dim=-1) if ya2d_pred.shape[-1] == 1 else ya2d_pred
                if yo3d_pred is not None: yo3d_pred = yo3d_pred.squeeze(dim=-1) if yo3d_pred.shape[-1] == 1 else yo3d_pred
                if yo2d_pred is not None: yo2d_pred = yo2d_pred.squeeze(dim=-1) if yo2d_pred.shape[-1] == 1 else yo2d_pred
                # reaggregate to get the predictions in one tensor
                y_pred = aggregate_systems(
                    ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred,
                    is_ens=False,
                )
                # append the prediction to the list of model's predictions
                y_pred_rollout_list.append(y_pred)
                y_true_rollout_list.append(y_true)

            # create rollout tensors (B x R x V x T x H x W)
            y_pred_rollout = torch.stack(y_pred_rollout_list, dim=1)
            if e == 0: # do this only the first time
                y_true_rollout = torch.stack(y_true_rollout_list, dim=1)

            # append the rollout tensor to the ensemble's list
            y_pred_ensemble_list.append(y_pred_rollout)

        # create ensemble tensors (B x R x V x T x H x W x E)
        y_pred_ensemble = torch.stack(y_pred_ensemble_list, dim=-1)
        y_true_ensemble = y_true_rollout

        # compute the loss function
        if y_true_ensemble.shape != y_pred_ensemble.shape:
            # probably we have an ensemble of heads, test it
            if y_true_ensemble.shape != y_pred_ensemble.shape[:-1]:
                print(f'files {files}')
            assert y_true_ensemble.shape == y_pred_ensemble.shape[:-1], \
                f'Y true and Y pred have different shape: y_true {y_true.shape} != y_pred {y_pred.shape[:-1]} ({y_pred.shape})'

        # create rollout weighting function
        weights = torch.linspace(0, 1, rollout+1)[1:]
        for r in range(rollout):
            if is_ens:
                loss = loss + weights[r] * self.loss(y_true_ensemble.select(1, r), y_pred_ensemble.select(1, r))
            else:
                loss = loss + weights[r] * self.loss(y_true_ensemble.select(1, r), y_pred_ensemble.select(1, r)[...,0])
        loss_dict = None

        # divide by the number of rollout and ensemble steps
        loss = loss / rollout

        # define log dictionary
        if type(loss) == tuple:
            tot_loss, a3d_loss, a2d_loss, o3d_loss, o2d_loss = loss
            a3d_loss_log_dict = {f'{step_type}_{key}': value for key, value in a3d_loss.items()} if type(a3d_loss)==dict else {}
            a2d_loss_log_dict = {f'{step_type}_{key}': value for key, value in a2d_loss.items()} if type(a2d_loss)==dict else {}
            o3d_loss_log_dict = {f'{step_type}_{key}': value for key, value in o3d_loss.items()} if type(o3d_loss)==dict else {}
            o2d_loss_log_dict = {f'{step_type}_{key}': value for key, value in o2d_loss.items()} if type(o2d_loss)==dict else {}
            log_dict = {
                **{f'{step_type}_loss': tot_loss}, 
                **a3d_loss_log_dict, 
                **a2d_loss_log_dict, 
                **o3d_loss_log_dict, 
                **o2d_loss_log_dict, 
            }
        else:
            log_dict = {f'{step_type}_loss': loss}
            if loss_dict is not None:
                log_dict = {
                    **log_dict, 
                    **{key:value for key,value in loss_dict.items()}, 
                }

        # get the output loss value
        loss_value = tot_loss if type(loss) == tuple else loss

        # compute metrics
        metrics_dicts = []
        for metric in self.metrics:
            metrics_dict = {f'{step_type}_{metric.name}' : metric(y_true.detach(), y_pred.detach())}
            log_dict.update(metrics_dict)
            metrics_dicts.append(metrics_dict)

        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
        self.log(
            f'{step_type}_loss', 
            loss_value, 
            prog_bar=True, 
            batch_size=batch_size, 
            # synchronize among replicas only during validation
            sync_dist=False if step_type=='train' else True,
        )

        # mlflow logs
        if self.trainer.logger:
            step = self._loss[step_type]['steps']
            self.trainer.logger.log_metrics(log_dict, step=step)
            for metrics_dict in metrics_dicts:
                self.trainer.logger.log_metrics(log_dict, step=step)
            self._loss[step_type]['sum'] += loss_value
            self._loss[step_type]['steps'] += 1

        # return the loss
        return {'loss':loss_value}

    def generic_residual_step(self, batch, batch_idx, step_type):
        tdim = 2
        data, days, drivers, files = batch
        rollout, insteps, outsteps, is_ens, ensemble_size = (
            self.rollout, self.insteps, self.outsteps, self.is_ens, self.ensemble_size
        )
        drivers = realign_drivers_dict(drivers)

        atmos_input_fields_3d = drivers['input']['atmos']['fields_3d']
        atmos_input_fields_2d = drivers['input']['atmos']['fields_2d']
        ocean_input_fields_3d = drivers['input']['ocean']['fields_3d']
        ocean_input_fields_2d = drivers['input']['ocean']['fields_2d']

        atmos_output_fields_3d = drivers['output']['atmos']['fields_3d']
        atmos_output_fields_2d = drivers['output']['atmos']['fields_2d']
        ocean_output_fields_3d = drivers['output']['ocean']['fields_3d']
        ocean_output_fields_2d = drivers['output']['ocean']['fields_2d']

        X_atmos_3d, X_atmos_2d, X_ocean_3d, X_ocean_2d = deaggregate_systems(
            data, atmos_input_fields_3d, atmos_input_fields_2d,
            ocean_input_fields_3d, ocean_input_fields_2d,
        )
        Y_atmos_3d, Y_atmos_2d, Y_ocean_3d, Y_ocean_2d = deaggregate_systems(
            data, atmos_output_fields_3d, atmos_output_fields_2d,
            ocean_output_fields_3d, ocean_output_fields_2d,
        )

        device, dtype = data.device, data.dtype
        batch_size = data.shape[0]
        loss = torch.tensor(0.0, device=device)

        depths, levels = drivers['depths'], drivers['levels']
        all_depths, all_levels = ALL_DRIVERS['depths'], ALL_DRIVERS['levels']
        depth_ids = torch.as_tensor([i for i, d in enumerate(all_depths) if int(d) in [int(dpt) for dpt in depths]]).to(device)
        level_ids = torch.as_tensor([i for i, l in enumerate(all_levels) if int(l) in [int(lvl) for lvl in levels]]).to(device)
        X_atmos_3d = X_atmos_3d.index_select(dim=tdim + 1, index=level_ids) if X_atmos_3d is not None else None
        Y_atmos_3d = Y_atmos_3d.index_select(dim=tdim + 1, index=level_ids) if Y_atmos_3d is not None else None
        X_ocean_3d = X_ocean_3d.index_select(dim=tdim + 1, index=depth_ids) if X_ocean_3d is not None else None
        Y_ocean_3d = Y_ocean_3d.index_select(dim=tdim + 1, index=depth_ids) if Y_ocean_3d is not None else None

        if is_ens and self.__class__.__name__ in ["AtlasOceaNOv2", "AtlasOceaNOv3"]:
            noise_conds = []
            D = self.dim
            B, _, _, H, W = data.shape
            if self.conv_module == "spherical":
                nshape = (B * insteps, D, H, W)
            elif self.conv_module == "spectral":
                nshape = (B, D, insteps, H, W)
            for _ in range(self.ensemble_size):
                noise_conds.append(torch.randn(nshape, device=device, dtype=dtype))
        else:
            noise_conds = [None]

        dy_pred_ensemble_list = []
        for e in range(ensemble_size):
            dy_pred_rollout_list = []
            dy_true_rollout_list = []
            for r in range(rollout):
                off = r if outsteps > 1 else 0
                i = insteps - 1 + r

                if r == 0:
                    xa3d, xa2d, xo3d, xo2d = select_input(
                        (X_atmos_3d, X_atmos_2d, X_ocean_3d, X_ocean_2d), i + off, insteps
                    )
                else:
                    xa3d, xa2d, xo3d, xo2d = shift_tensors(
                        (xa3d, xa2d, xo3d, xo2d), outsteps, tdim,
                    )
                    # yi passata qui viene ora detach-ata dentro update_input_with_forcing
                    xa3d, xa2d, xo3d, xo2d = update_input_with_forcing(
                        (xa3d, xa2d, xo3d, xo2d),
                        (ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred),
                        (X_atmos_3d, X_atmos_2d, X_ocean_3d, X_ocean_2d),
                        i, insteps, outsteps, drivers
                    )

                t = days[:, i + 1 - insteps + off: i + 1 + off]
                ya3d, ya2d, yo3d, yo2d = select_output(
                    (Y_atmos_3d, Y_atmos_2d, Y_ocean_3d, Y_ocean_2d), i + off, outsteps
                )

                dya3d_pred, dya2d_pred, dyo3d_pred, dyo2d_pred = self(
                    x=(xa3d, xa2d, xo3d, xo2d),
                    t=t, r=r, c=noise_conds[e],
                    drivers=drivers, out_attn=False, agg_systems=False,
                    use_noise=self.use_noise,
                )
                if dya3d_pred is not None: dya3d_pred = dya3d_pred.squeeze(dim=-1) if dya3d_pred.shape[-1] == 1 else dya3d_pred
                if dya2d_pred is not None: dya2d_pred = dya2d_pred.squeeze(dim=-1) if dya2d_pred.shape[-1] == 1 else dya2d_pred
                if dyo3d_pred is not None: dyo3d_pred = dyo3d_pred.squeeze(dim=-1) if dyo3d_pred.shape[-1] == 1 else dyo3d_pred
                if dyo2d_pred is not None: dyo2d_pred = dyo2d_pred.squeeze(dim=-1) if dyo2d_pred.shape[-1] == 1 else dyo2d_pred

                dya3d = ya3d - xa3d.detach() if ya3d is not None else None
                dya2d = ya2d - xa2d.detach() if ya2d is not None else None
                dyo3d = yo3d - xo3d.detach() if yo3d is not None else None
                dyo2d = yo2d - xo2d.detach() if yo2d is not None else None
                dy = aggregate_systems(dya3d, dya2d, dyo3d, dyo2d, is_ens=False)

                ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred = (
                    xa3d.detach() + dya3d_pred if dya3d_pred is not None else None,
                    xa2d.detach() + dya2d_pred if dya2d_pred is not None else None,
                    xo3d.detach() + dyo3d_pred if dyo3d_pred is not None else None,
                    xo2d.detach() + dyo2d_pred if dyo2d_pred is not None else None,
                )
                dy_pred = aggregate_systems(dya3d_pred, dya2d_pred, dyo3d_pred, dyo2d_pred, is_ens=False)

                dy_pred_rollout_list.append(dy_pred)
                dy_true_rollout_list.append(dy)

            dy_pred_rollout = torch.stack(dy_pred_rollout_list, dim=1)
            if e == 0:
                dy_true_rollout = torch.stack(dy_true_rollout_list, dim=1)
            dy_pred_ensemble_list.append(dy_pred_rollout)

        dy_pred_ensemble = torch.stack(dy_pred_ensemble_list, dim=-1)
        dy_true_ensemble = dy_true_rollout

        if dy_true_ensemble.shape != dy_pred_ensemble.shape:
            if dy_true_ensemble.shape != dy_pred_ensemble.shape[:-1]:
                print(f'files {files}')
            assert dy_true_ensemble.shape == dy_pred_ensemble.shape[:-1], \
                f'Y true and Y pred have different shape: dy {dy.shape} != dy_pred {dy_pred.shape[:-1]} ({dy_pred.shape})'

        weights = torch.full((rollout,), 1.0 / rollout, device=device)

        for r in range(rollout):
            if is_ens:
                loss = loss + weights[r] * self.loss(dy_true_ensemble.select(1, r), dy_pred_ensemble.select(1, r))
            else:
                loss = loss + weights[r] * self.loss(dy_true_ensemble.select(1, r), dy_pred_ensemble.select(1, r)[..., 0])
        loss_dict = None

        if type(loss) == tuple:
            tot_loss, a3d_loss, a2d_loss, o3d_loss, o2d_loss = loss
            a3d_loss_log_dict = {f'{step_type}_{key}': value for key, value in a3d_loss.items()} if type(a3d_loss) == dict else {}
            a2d_loss_log_dict = {f'{step_type}_{key}': value for key, value in a2d_loss.items()} if type(a2d_loss) == dict else {}
            o3d_loss_log_dict = {f'{step_type}_{key}': value for key, value in o3d_loss.items()} if type(o3d_loss) == dict else {}
            o2d_loss_log_dict = {f'{step_type}_{key}': value for key, value in o2d_loss.items()} if type(o2d_loss) == dict else {}
            log_dict = {
                **{f'{step_type}_loss': tot_loss},
                **a3d_loss_log_dict, **a2d_loss_log_dict,
                **o3d_loss_log_dict, **o2d_loss_log_dict,
            }
        else:
            log_dict = {f'{step_type}_loss': loss}
            if loss_dict is not None:
                log_dict = {**log_dict, **{key: value for key, value in loss_dict.items()}}

        loss_value = tot_loss if type(loss) == tuple else loss

        metrics_dicts = []
        for metric in self.metrics:
            metrics_dict = {f'{step_type}_{metric.name}': metric(dy.detach(), dy_pred.detach())}
            log_dict.update(metrics_dict)
            metrics_dicts.append(metrics_dict)

        self.callback_metrics = {**self.callback_metrics, **log_dict}
        self.log(
            f'{step_type}_loss', loss_value, prog_bar=True, batch_size=batch_size,
            sync_dist=False if step_type == 'train' else True,
        )

        if self.trainer.logger:
            step = self._loss[step_type]['steps']
            self.trainer.logger.log_metrics(log_dict, step=step)
            for metrics_dict in metrics_dicts:
                self.trainer.logger.log_metrics(log_dict, step=step)
            self._loss[step_type]['sum'] += loss_value
            self._loss[step_type]['steps'] += 1

        return {'loss': loss_value}


    def generic_representation_learning_step(self, batch, batch_idx, step_type):
        """
        This step is used to let the DL model learn to reconstruct a representation
        of the data. A stochastic model built in this way will build a meaningful 
        latent representation of the input data.

        """
        tdim = 2
        data, days, drivers, files = batch
        # prepare internal variables
        rollout, insteps, outsteps, is_ens, ensemble_size = (
            self.rollout, self.insteps, self.outsteps, self.is_ens, self.ensemble_size
        )
        # realign drivers dictionary
        drivers = realign_drivers_dict(drivers)
        # get input fields
        atmos_input_fields_3d = drivers['input']['atmos']['fields_3d']
        atmos_input_fields_2d = drivers['input']['atmos']['fields_2d']
        ocean_input_fields_3d = drivers['input']['ocean']['fields_3d']
        ocean_input_fields_2d = drivers['input']['ocean']['fields_2d']
        # get output fields
        atmos_output_fields_3d = drivers['output']['atmos']['fields_3d']
        atmos_output_fields_2d = drivers['output']['atmos']['fields_2d']
        ocean_output_fields_3d = drivers['output']['ocean']['fields_3d']
        ocean_output_fields_2d = drivers['output']['ocean']['fields_2d']
        # deaggregate variables
        X_atmos_3d, X_atmos_2d, X_ocean_3d, X_ocean_2d = deaggregate_systems(
            data, 
            atmos_input_fields_3d, 
            atmos_input_fields_2d, 
            ocean_input_fields_3d, 
            ocean_input_fields_2d, 
        )
        Y_atmos_3d, Y_atmos_2d, Y_ocean_3d, Y_ocean_2d = deaggregate_systems(
            data, 
            atmos_output_fields_3d, 
            atmos_output_fields_2d, 
            ocean_output_fields_3d, 
            ocean_output_fields_2d, 
        )
        device, dtype = data.device, data.dtype
        batch_size = data.shape[0]
        loss = torch.tensor(0.0, device=device)

        # create noise tensors to inject into the networks
        if is_ens and self.__class__.__name__ in ["AtlasOceaNOv2"]:
            noise_conds = []
            D = self.dim
            B, _, _, H, W = data.shape
            if self.conv_module == "spherical":
                nshape = (B * insteps, D, H, W)
            elif self.conv_module == "spectral":
                nshape = (B, D, insteps, H, W)
            for _ in range(self.ensemble_size):
                noise_conds.append(
                    torch.randn(nshape, device=device, dtype=dtype)
                )
        else:
            noise_conds = [None]

        # create the prediction ensemble list
        y_pred_ensemble_list = []
        # ensemble iteration
        for e in range(ensemble_size):
            # create the rollout lists
            y_pred_rollout_list = []
            y_true_rollout_list = []
            # rollout iteration
            for r in range(rollout):
                # determine the offset for the progressive 
                # rollout shifts in the data indices
                off = r if outsteps > 1 else 0
                # get index i
                i = insteps - 1 + r
                # only on the first iteration
                if r == 0:
                    # get input data from dataset
                    xa3d, xa2d, xo3d, xo2d = select_input(
                        (X_atmos_3d, X_atmos_2d, X_ocean_3d, X_ocean_2d), 
                        i+off, insteps
                    )
                # we are in autoregression
                else:
                    # shift the temporal window by outsteps
                    xa3d, xa2d, xo3d, xo2d = shift_tensors(
                        (xa3d, xa2d, xo3d, xo2d), outsteps, tdim, 
                    )
                    # substitute the last element with the prediction
                    xa3d, xa2d, xo3d, xo2d = update_input_with_forcing(
                        (xa3d, xa2d, xo3d, xo2d), 
                        (ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred), 
                        (X_atmos_3d, X_atmos_2d, X_ocean_3d, X_ocean_2d), 
                        i, insteps, outsteps, drivers
                    )
                # get static variables
                t = days[:, i+1-insteps+off : i+1+off]
                # get ground truth (we take the same as input)
                ya3d, ya2d, yo3d, yo2d = select_input(
                    (Y_atmos_3d, Y_atmos_2d, Y_ocean_3d, Y_ocean_2d), 
                    i+off, insteps
                )
                # reaggregate to get the correct number of variables
                y_true = aggregate_systems(
                    ya3d, ya2d, yo3d, yo2d, 
                    is_ens = False
                )
                # forward pass (the network predicts the residual)
                if r < rollout - 1 and step_type == 'train':
                    # `pushforward` trick: compute gradients only on last rollout step
                    with torch.no_grad():
                        ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred = self(
                            x=(xa3d, xa2d, xo3d, xo2d), 
                            t=t, r=r, c=noise_conds[e], 
                            drivers=drivers, 
                            out_attn=False, 
                            agg_systems=False, 
                            use_noise=self.use_noise, 
                        )
                else:
                    ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred = self(
                        x=(xa3d, xa2d, xo3d, xo2d), 
                        t=t, r=r, c=noise_conds[e], 
                        drivers=drivers, 
                        out_attn=False, 
                        agg_systems=False, 
                        use_noise=self.use_noise, 
                    )
                # remove ensemble dimension if present
                if ya3d_pred is not None: ya3d_pred = ya3d_pred.squeeze(dim=-1) if ya3d_pred.shape[-1] == 1 else ya3d_pred
                if ya2d_pred is not None: ya2d_pred = ya2d_pred.squeeze(dim=-1) if ya2d_pred.shape[-1] == 1 else ya2d_pred
                if yo3d_pred is not None: yo3d_pred = yo3d_pred.squeeze(dim=-1) if yo3d_pred.shape[-1] == 1 else yo3d_pred
                if yo2d_pred is not None: yo2d_pred = yo2d_pred.squeeze(dim=-1) if yo2d_pred.shape[-1] == 1 else yo2d_pred
                # aggregate the predicted residuals 
                y_pred = aggregate_systems(
                    ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred, 
                    is_ens = False
                )
                # append the prediction to the list of model's predictions
                y_pred_rollout_list.append(y_pred)
                y_true_rollout_list.append(y_true)

            # create rollout tensors (B x R x V x T x H x W)
            y_pred_rollout = torch.stack(y_pred_rollout_list, dim=1)
            if e == 0: # do this only the first time
                y_true_rollout = torch.stack(y_true_rollout_list, dim=1)

            # append the rollout tensor to the ensemble's list
            y_pred_ensemble_list.append(y_pred_rollout)

        # create ensemble tensors (B x R x V x T x H x W x E)
        y_pred_ensemble = torch.stack(y_pred_ensemble_list, dim=-1)
        y_true_ensemble = y_true_rollout

        # compute the loss function
        if y_true_ensemble.shape != y_pred_ensemble.shape:
            # probably we have an ensemble of heads, test it
            if y_true_ensemble.shape != y_pred_ensemble.shape[:-1]:
                print(f'files {files}')
            assert y_true_ensemble.shape == y_pred_ensemble.shape[:-1], \
                f'Y true and Y pred have different shape: y_true {y_true.shape} != y_pred {y_pred.shape[:-1]} ({y_pred.shape})'

        # create rollout weighting function
        weights = torch.linspace(0, 1, rollout+1)[1:]
        for r in range(rollout):
            if is_ens:
                loss = loss + weights[r] * self.loss(y_true_ensemble.select(1, r), y_pred_ensemble.select(1, r))
            else:
                loss = loss + weights[r] * self.loss(y_true_ensemble.select(1, r), y_pred_ensemble.select(1, r)[...,0])
        loss_dict = None

        # divide by the number of rollout and ensemble steps
        loss = loss / rollout

        # define log dictionary
        if type(loss) == tuple:
            tot_loss, a3d_loss, a2d_loss, o3d_loss, o2d_loss = loss
            a3d_loss_log_dict = {f'{step_type}_{key}': value for key, value in a3d_loss.items()} if type(a3d_loss)==dict else {}
            a2d_loss_log_dict = {f'{step_type}_{key}': value for key, value in a2d_loss.items()} if type(a2d_loss)==dict else {}
            o3d_loss_log_dict = {f'{step_type}_{key}': value for key, value in o3d_loss.items()} if type(o3d_loss)==dict else {}
            o2d_loss_log_dict = {f'{step_type}_{key}': value for key, value in o2d_loss.items()} if type(o2d_loss)==dict else {}
            log_dict = {
                **{f'{step_type}_loss': tot_loss}, 
                **a3d_loss_log_dict, 
                **a2d_loss_log_dict, 
                **o3d_loss_log_dict, 
                **o2d_loss_log_dict, 
            }
        else:
            log_dict = {f'{step_type}_loss': loss}
            if loss_dict is not None:
                log_dict = {
                    **log_dict, 
                    **{key:value for key,value in loss_dict.items()}, 
                }

        # get the output loss value
        loss_value = tot_loss if type(loss) == tuple else loss

        # compute metrics
        metrics_dicts = []
        for metric in self.metrics:
            metrics_dict = {f'{step_type}_{metric.name}' : metric(y_true, y_pred)}
            log_dict.update(metrics_dict)
            metrics_dicts.append(metrics_dict)

        # log the outputs
        self.callback_metrics = {**self.callback_metrics, **log_dict}
        self.log(
            f'{step_type}_loss', 
            loss_value, 
            prog_bar=True, 
            batch_size=batch_size, 
            # synchronize among replicas only during validation
            sync_dist=False if step_type=='train' else True,
        )

        # mlflow logs
        if self.trainer.logger:
            step = self._loss[step_type]['steps']
            self.trainer.logger.log_metrics(log_dict, step=step)
            for metrics_dict in metrics_dicts:
                self.trainer.logger.log_metrics(log_dict, step=step)
            self._loss[step_type]['sum'] += loss_value
            self._loss[step_type]['steps'] += 1

        # return the loss
        return {'loss':loss_value}

    def training_step(self, batch, batch_idx):
        if self.is_residual:
            return self.generic_residual_step(batch=batch, batch_idx=batch_idx, step_type='train')
        else:
            return self.generic_step(batch=batch, batch_idx=batch_idx, step_type='train')

    def validation_step(self, batch, batch_idx):
        if self.is_residual:
            return self.generic_residual_step(batch=batch, batch_idx=batch_idx, step_type='val')
        else:
            return self.generic_step(batch=batch, batch_idx=batch_idx, step_type='val')

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}

    def on_validation_epoch_end(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()
        return super().on_validation_epoch_end()

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()
        return super().on_train_epoch_end()

    def set_attributes(self, 
        optimizer, 
        scheduler, 
        loss, 
        metrics, 
        insteps: int = 1, 
        outsteps: int = 1, 
        lead_time: int = 1, 
        rollout: int = 1, 
        is_rollout: bool = False, 
        is_ens: bool = False, 
        is_residual: bool = False, 
        use_noise: bool = False, 
    ):
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.loss = loss
        self.metrics = metrics
        self.insteps = insteps
        self.outsteps = outsteps
        self.lead_time = lead_time
        self.rollout = rollout
        self.is_rollout = is_rollout
        self.is_ens = is_ens
        self.is_residual = is_residual
        self.use_noise = use_noise


class NeptuneSystemEmbed3d(nn.Module):
    """
    Convolutions for each of the systems (atmos_3d, atmos_2d, ocean_3d, ocean_2d) separately.
    It projects the systems data into the same latent space of dimension `dim`. The resulting
    projections can be used as input to the backbone.
    """
    def __init__(self, 
            drivers: dict, 
            dim: int, 
            kernel_size: int, 
            use_swin_embed: bool = False, 
            inout: str = 'input', 
        ) -> None:
        super().__init__()
        self.inout = inout
        self.use_swin_embed = use_swin_embed
        self.kernel_size = kernel_size
        self.drivers = drivers
        self.dim = dim
        var_names = self.get_var_names(drivers)
        self.weights = nn.ParameterDict({
            var: nn.Parameter(torch.empty(dim, 1, 1, kernel_size, kernel_size))
            for var in var_names + ['noise']
        })
        self.bias = nn.Parameter(torch.empty(dim))
        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights."""
        for weight in self.weights.values():
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(next(iter(self.weights.values())))
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_var_names(self, drivers):
        z_levels = drivers['levels']
        z_depths = drivers['depths']
        atmos_fields_3d = drivers[self.inout]['atmos']['fields_3d']
        atmos_fields_2d = drivers[self.inout]['atmos']['fields_2d']
        ocean_fields_3d = drivers[self.inout]['ocean']['fields_3d']
        ocean_fields_2d = drivers[self.inout]['ocean']['fields_2d']
        # get the var names of each system
        var_names = []
        var_names += [f'{var}_{int(lev)}' for var in atmos_fields_3d for lev in z_levels]
        var_names += [f'{var}' for var in atmos_fields_2d]
        var_names += [f'{var}_{int(lev)}' for var in ocean_fields_3d for lev in z_depths]
        var_names += [f'{var}' for var in ocean_fields_2d]
        return var_names

    def forward(self, x: torch.Tensor, drivers: dict, use_noise: bool = False):
        # get the variables for each system
        a3d, a2d, o3d, o2d = x
        # aggregate systems
        x = torch.cat([rearrange(s, 'b v t z h w -> b (v z) t h w') for s in (a3d, a2d, o3d, o2d) if s is not None], dim = 1)
        # we inject noise in the data
        if use_noise:
            B, _, T, H, W = x.shape
            dtype, device = x.dtype, x.device
            # create a noise tensor
            noise = torch.randn(B, 1, T, H, W, dtype=dtype, device=device)
            # add the noise tensor with probability of 0.5
            p = torch.rand(1).item()
            if p >= 0.5:
                # inject the noise tensor in the data
                x = torch.cat([x, noise], dim = 1)
            else:
                # do not use noise
                use_noise = False
        # get the var names
        var_names = self.get_var_names(drivers)
        # prepare the weights needed for the convolution
        if not use_noise:
            weight = torch.cat([self.weights[var] for var in var_names], dim=1)
        else:
            # use the noise associated weights
            weight = torch.cat([self.weights[var] for var in var_names + ['noise']], dim=1)
        # convolve the kernels with the x tensor
        if self.use_swin_embed: 
            stride = (1, self.kernel_size, self.kernel_size)
            padding = 0
        else: 
            stride = 1
            padding = (0, 1, 1)
        x = F.conv3d(x, weight, self.bias, stride=stride, padding=padding)
        return x


class NeptuneSystemRecover3d(nn.Module):
    """
    Convolutions for each of the systems (atmos_3d, atmos_2d, ocean_3d, ocean_2d) separately.
    It projects the systems data into the same latent space of dimension `dim`. The resulting
    projections can be used as input to the backbone.
    """
    def __init__(self, 
            drivers: dict, 
            dim: int, 
            kernel_size: int, 
            insteps: int, 
            outsteps: int, 
            use_swin_embed: bool = False, 
        ) -> None:
        super().__init__()
        self.use_swin_embed = use_swin_embed
        self.drivers = drivers
        self.dim = dim
        self.outsteps = outsteps
        (atmos_3d_var_names, 
         atmos_2d_var_names, 
         ocean_3d_var_names, 
         ocean_2d_var_names), _ = self.get_var_names(drivers)
        self.time_conv = nn.Conv1d(dim, dim, 
            kernel_size = insteps - outsteps + 1 if insteps > outsteps else 1
        )
        if use_swin_embed:
            self.proj = nn.Linear(dim, dim * kernel_size * kernel_size)
            self.kernel_size = kernel_size
        self.atmos_3d_weights = nn.ParameterDict({
            var: nn.Parameter(torch.empty(1, dim, 1, 3, 3))
            for var in atmos_3d_var_names
        })
        self.atmos_2d_weights = nn.ParameterDict({
            var: nn.Parameter(torch.empty(1, dim, 1, 3, 3))
            for var in atmos_2d_var_names
        })
        self.ocean_3d_weights = nn.ParameterDict({
            var: nn.Parameter(torch.empty(1, dim, 1, 3, 3))
            for var in ocean_3d_var_names
        })
        self.ocean_2d_weights = nn.ParameterDict({
            var: nn.Parameter(torch.empty(1, dim, 1, 3, 3))
            for var in ocean_2d_var_names
        })
        self.init_weights()
    
    def init_weights(self) -> None:
        """Initialise weights."""
        for weights in (
            self.atmos_3d_weights, 
            self.atmos_2d_weights, 
            self.ocean_3d_weights, 
            self.ocean_2d_weights):
            for weight in weights.values():
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    
    def get_var_names(self, drivers):
        z_levels = drivers['levels']
        z_depths = drivers['depths']
        atmos_fields_3d = drivers['output']['atmos']['fields_3d']
        atmos_fields_2d = drivers['output']['atmos']['fields_2d']
        ocean_fields_3d = drivers['output']['ocean']['fields_3d']
        ocean_fields_2d = drivers['output']['ocean']['fields_2d']
        # get the var names of each system
        atmos_3d_var_names = [f'{var}_{int(lev)}' for var in atmos_fields_3d for lev in z_levels]
        atmos_2d_var_names = [f'{var}' for var in atmos_fields_2d]
        ocean_3d_var_names = [f'{var}_{int(lev)}' for var in ocean_fields_3d for lev in z_depths]
        ocean_2d_var_names = [f'{var}' for var in ocean_fields_2d]
        return (
            (atmos_3d_var_names, atmos_2d_var_names, ocean_3d_var_names, ocean_2d_var_names), 
            (atmos_fields_3d, atmos_fields_2d, ocean_fields_3d, ocean_fields_2d, z_levels, z_depths), 
            )
    
    def forward(self, x: torch.Tensor, drivers: dict):
        # time convolution
        B, D, _, H, W = x.shape
        x = rearrange(x, 'b d t h w -> (b h w) d t')
        x = self.time_conv(x)
        T = x.shape[-1]
        x = rearrange(x, '(b h w) d t -> b d t h w', b=B, h=H, w=W)
        # swin-like projection
        if self.use_swin_embed:
            P = self.kernel_size
            x = rearrange(x, 'b d t h w -> b (t h w) d')
            x = self.proj(x)
            x = rearrange(x, 'b (t h w) (d p1 p2) -> b d t (h p1) (w p2)', t=T, h=H, w=W, d=D, p1=P, p2=P)
        # get the var names
        (atmos_3d_var_names, 
         atmos_2d_var_names, 
         ocean_3d_var_names, 
         ocean_2d_var_names), (atmos_fields_3d, atmos_fields_2d, ocean_fields_3d, ocean_fields_2d, z_levels, z_depths) = self.get_var_names(drivers)
        # convolve the kernels with the x tensor
        a3d = torch.stack([F.conv3d(x, self.atmos_3d_weights[var], stride=1, padding=(0,1,1)) for var in atmos_3d_var_names], dim=1)[:,:,0] if len(atmos_3d_var_names) != 0 else None
        # convolve the kernels with the x tensor
        a2d = torch.stack([F.conv3d(x, self.atmos_2d_weights[var], stride=1, padding=(0,1,1)) for var in atmos_2d_var_names], dim=1)[:,:,0] if len(atmos_2d_var_names) != 0 else None
        # convolve the kernels with the x tensor
        o3d = torch.stack([F.conv3d(x, self.ocean_3d_weights[var], stride=1, padding=(0,1,1)) for var in ocean_3d_var_names], dim=1)[:,:,0] if len(ocean_3d_var_names) != 0 else None
        # convolve the kernels with the x tensor
        o2d = torch.stack([F.conv3d(x, self.ocean_2d_weights[var], stride=1, padding=(0,1,1)) for var in ocean_2d_var_names], dim=1)[:,:,0] if len(ocean_2d_var_names) != 0 else None
        # rearrange output
        a3d = rearrange(a3d, 'b (v z) t h w -> b v t z h w', v=len(atmos_fields_3d), z=len(z_levels)) if a3d is not None else None
        a2d = rearrange(a2d, 'b (v z) t h w -> b v t z h w', v=len(atmos_fields_2d), z=1) if a2d is not None else None
        o3d = rearrange(o3d, 'b (v z) t h w -> b v t z h w', v=len(ocean_fields_3d), z=len(z_depths)) if o3d is not None else None
        o2d = rearrange(o2d, 'b (v z) t h w -> b v t z h w', v=len(ocean_fields_2d), z=1) if o2d is not None else None
        return (a3d, a2d, o3d, o2d)
