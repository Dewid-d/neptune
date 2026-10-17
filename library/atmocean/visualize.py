from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
import numpy as np
import os

plot_kwargs = dict(dpi=100, bbox_inches='tight')


SST_BIAS_CMAP_RGB_COLORS = [
    'RGB(118,39,230)', 'RGB(25,6,241)', 'RGB(49,71,226)', 
    'RGB(69,116,228)', 'RGB(71,149,247)', 'RGB(86,188,249)', 
    'RGB(135,198,238)', 'RGB(188,222,242)','RGB(254,248,156)', 
    'RGB(248,216,82)', 'RGB(243,174,61)', 'RGB(238,129,49)', 
    'RGB(235,70,38)', 'RGB(214,46,31)', 'RGB(192,49,40)', 
    'RGB(244,180,179)', 
]

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def get_sst_bias_colormap(vmin=-1.0, vmax=1.0, num_levels=17):
    normalized_colors = []
    for color in SST_BIAS_CMAP_RGB_COLORS:
        R, G, B = color.strip('RGB(').strip(')').split(',')
        normalized_colors.append((int(R)/255,int(G)/255,int(B)/255))
    levels = np.linspace(vmin, vmax, num_levels)
    cmap = ListedColormap(normalized_colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N)
    return cmap, norm, levels

def get_colormap_from_cmap(cmap, vmin=-1.0, vmax=1.0, num_levels=17):
    base = plt.get_cmap(cmap)
    color_list = base(np.linspace(0, 1, num_levels))
    levels = np.linspace(vmin, vmax, num_levels)
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(levels, ncolors=cmap.N)
    return cmap, norm, levels

def plot_on_globe(data, instant, label, win_size, lat_range=(-90, 90), lon_range=(-180, 180), outfile: str=None):
    x = data[instant].numpy()
    lats = np.linspace(lat_range[1], lat_range[0], x.shape[0])
    lons = np.linspace(lon_range[0], lon_range[1], x.shape[1])
    nav_lon, nav_lat = np.meshgrid(lons, lats)

    central_longitude = 180
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=proj)

    plot = ax.pcolormesh(nav_lon, nav_lat, x, cmap='coolwarm', transform=proj)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    latitudes = np.linspace(lat_range[0], lat_range[1]+1, x.shape[0]//win_size[1]+1)
    longitudes = np.linspace(lon_range[0], lon_range[1]+1, x.shape[1]//win_size[2]+1)

    gl = ax.gridlines(draw_labels=False, dms=True, linewidth=0.5, color='gray', alpha=1.0, linestyle='--')
    gl.xlocator = mticker.FixedLocator(longitudes)
    gl.ylocator = mticker.FixedLocator(latitudes)
    gl.xlines = True
    gl.ylines = True

    cbar = plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.02)
    cbar.set_label(label)

    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def plot_rects(ax, boxes: list, colors: list):
    for box, color in zip(boxes, colors):
        rect = mpatches.Rectangle(
            (box[0], box[1]), box[2], box[3],
            linewidth=2, edgecolor=color, facecolor='none', transform=ccrs.PlateCarree(), 
        )
        ax.add_patch(rect)

def plot_dynamical_system_loss(dsys: pd.DataFrame, offset: int = 0, val_min = None, val_max = None, outfile = None):
    eps = 1e-2
    ymax = max((dsys['val_loss'][offset:].max(), dsys['train_loss'][offset:].max()))
    ymin = min((dsys['val_loss'][offset:].min(), dsys['train_loss'][offset:].min()))
    bbox = dict(boxstyle='round', edgecolor='green', facecolor='white', alpha=1.0)
    if val_max: ymax = min((ymax, val_max))
    if val_min: ymin = max((ymin, val_min))
    _, ax = plt.subplots(figsize=(13,4))
    plt.plot(range(len(dsys))[offset:], dsys['train_loss'][offset:], label = 'Train Loss')
    plt.plot(range(len(dsys))[offset:], dsys['val_loss'][offset:], label = 'Valid Loss')
    plt.vlines(x=150, ymin=ymin, ymax=ymax, color='green', linestyle='dashed')
    ax.text(150, ymax/2, 'rollout t+2', rotation=90, fontsize=8, color='green', horizontalalignment='center', verticalalignment='center', bbox=bbox)
    plt.vlines(x=160, ymin=ymin, ymax=ymax, color='green', linestyle='dashed')
    ax.text(160, ymax/2, 'rollout t+3', rotation=90, fontsize=8, color='green', horizontalalignment='center', verticalalignment='center', bbox=bbox)
    plt.vlines(x=170, ymin=ymin, ymax=ymax, color='green', linestyle='dashed')
    ax.text(170, ymax/2, 'rollout t+4', rotation=90, fontsize=8, color='green', horizontalalignment='center', verticalalignment='center', bbox=bbox)
    plt.vlines(x=180, ymin=ymin, ymax=ymax, color='green', linestyle='dashed')
    ax.text(180, ymax/2, 'rollout t+5', rotation=90, fontsize=8, color='green', horizontalalignment='center', verticalalignment='center', bbox=bbox)
    plt.vlines(x=190, ymin=ymin, ymax=ymax, color='green', linestyle='dashed')
    ax.text(190, ymax/2, 'rollout t+6', rotation=90, fontsize=8, color='green', horizontalalignment='center', verticalalignment='center', bbox=bbox)
    plt.ylim(ymin - eps, ymax + eps)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def surface_plots_model1_vs_model2(
    metric_model1: xr.Dataset, 
    metric_model2: xr.Dataset, 
    sdate: str, 
    edate: str, 
    fields_info: dict, 
    fields_3d: list, 
    fields_2d: list, 
    model1: str, 
    model2: str, 
    metric_name: str, 
    outfile: str = None, 
):
    print(f'DEPRECATED PLOT THAT WILL BE SUBSTITUTED BY surface_plots_multiple_models()')
    return 
    metric_model1_3d = metric_model1[fields_3d].to_array()
    metric_model2_3d = metric_model2[fields_3d].to_array()
    metric_model1_2d = metric_model1[fields_2d].to_array()
    metric_model2_2d = metric_model2[fields_2d].to_array()
    # plot with respect to the persistence
    fontdict = {'weight': 'bold', 'size': 12}
    width, height = 4, 4
    fig = plt.figure(figsize = (5.0 * width, 2.3 * height))
    plt.suptitle(f'{model1} vs {model2} {metric_name.upper()} ({sdate} - {edate})', fontsize=15, weight='bold')
    gs = fig.add_gridspec(2, max((len(fields_3d), len(fields_2d))))
    for i, field in enumerate(fields_3d):
        metric1 = metric_model1_3d[i, :, 0]
        metric2 = metric_model2_3d[i, :, 0]
        ax = fig.add_subplot(gs[0, i])
        x = np.arange(metric1.shape[0] - 1)
        y1 = metric1[1:]
        y2 = metric2[1:]
        ax.plot(x, y1, label = f'{model1}')
        ax.plot(x, y2, label = f'{model2}')
        ax.set_ylabel(f'{fields_info[field]["long_name"]}', fontdict=fontdict)
        ax.legend()
    for i, field in enumerate(fields_2d):
        metric1 = metric_model1_2d[i, :]
        metric2 = metric_model2_2d[i, :]
        ax = fig.add_subplot(gs[1, i])
        x = np.arange(metric1.shape[0] - 1)
        y1 = metric1[1:]
        y2 = metric2[1:]
        ax.plot(x, y1, label = f'{model1}')
        ax.plot(x, y2, label = f'{model2}')
        ax.set_ylabel(f'{fields_info[field]["long_name"]}', fontdict=fontdict)
        ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def surface_plots_multiple_models(
    metrics_list: list, 
    labels: list,       # list of labels associated with the RMSEs
    colors: list,       # list of colors associated with the RMSEs
    lead_times_ids: list,   # list of lead times ids
    lead_times_labels: list,   # list of lead times labels
    fields_info: dict, # fields information for prettier plots
    sdate: str, 
    edate: str, 
    fields_3d: list, 
    fields_2d: list, 
    metric_name: str = 'rmse', 
    outfile: str = None, 
):
    metrics_list_3d = [metric[fields_3d].to_array() for metric in metrics_list]
    metrics_list_2d = [metric[fields_2d].to_array() for metric in metrics_list]
    fontdict = {'weight': 'bold', 'size': 12}
    width, height = 4, 4
    fig = plt.figure(figsize = (5.0 * width, 2.3 * height))
    plt.suptitle(f'Surface {metric_name.upper()} ({sdate} - {edate})', fontsize=15, weight='bold')
    gs = fig.add_gridspec(2, max((len(fields_3d), len(fields_2d))))
    for i, field in enumerate(fields_3d):
        ax = fig.add_subplot(gs[0, i])
        for metric_ds, color, label in zip(metrics_list_3d, colors, labels):
            try:    metric = metric_ds[i, lead_times_ids, 0].values
            except: metric = metric_ds[i, lead_times_ids].values
            x = np.arange(metric.shape[0])
            y1 = metric
            if 'pers' in label.lower() or 'clim' in label.lower():
                ax.plot(x, y1, linestyle='dashed', label=f'{label}', color=color, linewidth=3.0)
            else:
                ax.plot(x, y1, label=f'{label}', color=color, linewidth=3.0)
        ax.set_ylabel(f'{fields_info[f"{field}"]["long_name"]}', fontdict=fontdict)
        ax.set_xlabel(f'Lead Time [days]', fontdict=fontdict)
        ax.set_xticks(np.arange(len(lead_times_ids)))
        ax.set_xticklabels(lead_times_labels)
        # ax.legend()
    for i, field in enumerate(fields_2d):
        ax = fig.add_subplot(gs[1, i])
        for metric_ds, color, label in zip(metrics_list_2d, colors, labels):
            metric = metric_ds[i, lead_times_ids].values
            x = np.arange(metric.shape[0])
            y1 = metric
            if 'pers' in label.lower() or 'clim' in label.lower():
                ax.plot(x, y1, linestyle='dashed', label=f'{label}', color=color, linewidth=3.0)
            else:
                ax.plot(x, y1, label=f'{label}', color=color, linewidth=3.0)
        ax.set_ylabel(f'{fields_info[f"{field}"]["long_name"]}', fontdict=fontdict)
        ax.set_xlabel(f'Lead Time [days]', fontdict=fontdict)
        ax.set_xticks(np.arange(len(lead_times_ids)))
        ax.set_xticklabels(lead_times_labels)
        # ax.legend()
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc='upper center', markerscale=10, edgecolor='gray', framealpha=1, ncol=len(metrics_list)//2, bbox_to_anchor=(2.3 , -0.14))
    
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def vertical_plots_multiple_models(
    metrics_list: list, # list of RMSEs 3D
    labels: list,       # list of labels associated with the RMSEs
    colors: list,       # list of colors associated with the RMSEs
    lead_times_ids: list, # list of lead times ids to plot
    lead_times_labels: list, # list of lead times to plot
    vert_levels: list, # list of vertical levels
    fields_info: dict, # fields information for prettier plots
    sdate: str, 
    edate: str, 
    fields_3d: list, 
    metric_name: str, 
    outfile: str = None, 
):
    metrics_list = [m[fields_3d].to_array() for m in metrics_list]
    fontdict = {'weight': 'bold', 'size': 12}
    width, height = 4, 4
    fig = plt.figure(figsize = (len(fields_3d) * width, len(lead_times_labels) * height))
    plt.suptitle(f'Vertical {metric_name.upper()} Profile ({sdate} - {edate})', fontsize=15, weight='bold', y=0.9)
    gs = fig.add_gridspec(len(lead_times_labels),len(fields_3d))
    vmin = np.nanmin(
        np.stack([
            metric.min(dim=['prediction_timedelta', 'depth'], skipna=True).values
            for metric in metrics_list
        ]), 
        axis=0
    )
    vmax = np.nanmax(
        np.stack([
            metric.max(dim=['prediction_timedelta', 'depth'], skipna=True).values
            for metric in metrics_list
        ]), 
        axis=0
    )
    # added to do more specific plots on ~/validation/emulators/comparison/seasonal/seasonal.ipynb
    # vmax[0] = 0.8
    # vmax[1] = 0.25
    col_axes = []
    for l,lead_time in enumerate(lead_times_labels):
        for i,field in enumerate(fields_3d):
            ax = fig.add_subplot(gs[l, i])
            if l == 0: col_axes.append(ax)
            for metric_ds, color, label in zip(metrics_list, colors, labels):
                metric = metric_ds[i, lead_times_ids[l]]
                x = np.arange(metric.shape[0])
                y1 = metric[::-1]
                kwargs = dict(linestyle='dashed') if 'pers' in label.lower() else dict()
                ax.plot(y1, x, label=f'{label}', color=color, linewidth=2.0, **kwargs)
            if l == len(lead_times_labels) - 1:
                ax.set_xlabel(f'{fields_info[field]["long_name"]}', fontdict=fontdict)
            if i == 0:
                ax.set_ylabel(f't + {lead_time}', fontdict=fontdict)
                ax.set_yticklabels(vert_levels[::-1])
            else:
                ax.set_yticklabels([])
            ax.set_xlim(vmin[i], vmax[i])
            ax.set_yticks(np.arange(len(vert_levels)))
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc='upper center', markerscale=10, edgecolor='gray', framealpha=1, ncol=5, bbox_to_anchor=(-1.3, -0.22))
    fig.tight_layout()
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def normalized_rmse_difference_scorecard(
    rmse_fc_3d, 
    rmse_ps_3d, 
    rmse_fc_2d, 
    rmse_ps_2d, 
    diff_type, 
    drivers, 
    fsystem, 
    sdate, 
    edate, 
    forecast_horizon, 
    normalize: bool = False, 
    outfile: str = None, 
): 
    print(f'DEPRECATED PLOT THAT WILL BE SUBSTITUTED BY metrics_difference_scorecard_by_lead_times()')
    return
    vert_levels = drivers['levels' if fsystem == 'atmos' else 'depths']
    label = 'Level [m]' if fsystem == 'atmos' else 'Depth [m]'
    fields_3d = drivers['output'][fsystem]['fields_3d']
    fields_2d = drivers['output'][fsystem]['fields_2d']
    cmap = 'bwr'

    fig = plt.figure(figsize=(5*len(fields_3d) + forecast_horizon, 30//len(fields_3d)))
    if normalize:
        plt.suptitle(f'Normalized {diff_type} Difference ({sdate} - {edate})', fontsize=20)
    else:
        plt.suptitle(f'{diff_type} Difference ({sdate} - {edate})', fontsize=20)
    gs = fig.add_gridspec(2,len(fields_3d))
    for i, field in enumerate(fields_3d):
        ax = fig.add_subplot(gs[0, i])

        diff = rmse_fc_3d[i].T - rmse_ps_3d[i].T
        if normalize:
            ext = np.max((np.abs(np.min((diff))), np.abs(np.max((diff)))))
            im = ax.imshow(diff/ext, cmap=cmap, vmin=-1, vmax=1)
        else:
            ext = np.max((np.abs(np.min((diff))), np.abs(np.max((diff)))))
            im = ax.imshow(diff, cmap=cmap, vmin=-ext, vmax=ext)

        ax.set_title(f'{field}')
        if i!=0: 
            ax.set_yticks([])
        else:
            ax.set_ylabel(label, fontdict={'size':11})
            ax.set_yticks(np.arange(len(vert_levels)))
            ax.set_yticklabels(vert_levels)
        ax.set_xticks(np.arange(forecast_horizon+1))
        ax.set_xticklabels(np.arange(forecast_horizon+1))

    for i, field in enumerate(fields_2d):
        ax = fig.add_subplot(gs[1, i])

        diff = rmse_fc_2d[i:i+1] - rmse_ps_2d[i:i+1]
        ext = np.max((np.abs(np.min((diff))), np.abs(np.max((diff)))))
        im = ax.imshow(diff/ext, cmap=cmap, vmin=-1, vmax=1)

        ax.set_title(f'{field}')
        ax.set_yticks([0])
        ax.set_xticks(np.arange(forecast_horizon+1))
        ax.set_xticklabels(np.arange(forecast_horizon+1))
        ax.set_xlabel('Lead Time [days]', fontdict={'size':11})

    fig.subplots_adjust(right=0.8, hspace=-0.45, wspace=0.1)
    cbar_ax = fig.add_axes([0.81, 0.33, 0.008, 0.44])
    if normalize:
        fig.colorbar(im, cax=cbar_ax, label=f'Normalized {diff_type} difference\n(Neptune - Pers)\nagainst Reanalysis')
    else:
        fig.colorbar(im, cax=cbar_ax, label=f'{diff_type} difference\n(Neptune - Pers)\nagainst Reanalysis')
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def metrics_difference_scorecard_by_lead_times(
    metrics_model1: xr.Dataset, 
    metrics_model2: xr.Dataset, 
    metric_name: str, 
    fields_info: dict, 
    fsystem: str, 
    sdate: str, 
    edate: str, 
    fields_3d: list[str], 
    fields_2d: list[str], 
    vert_levels: str, 
    lead_times: list = None, 
    lead_times1: list = None, 
    lead_times2: list = None, 
    lead_times_labels: list = None, 
    model1: str = 'Neptune', 
    model2: str = 'Persistence', 
    cmap: str = 'RdBu_r', # vanno bene ['RdBu_r','bwr']
    normalize: bool = False, 
    outfile: str = None, 
):
    metrics_model1_3d = metrics_model1[fields_3d].to_array()
    metrics_model2_3d = metrics_model2[fields_3d].to_array()
    metrics_model1_2d = metrics_model1[fields_2d].to_array()
    metrics_model2_2d = metrics_model2[fields_2d].to_array()
    lead_times = lead_times or lead_times1
    lead_times1 = lead_times1 or lead_times
    lead_times2 = lead_times2 or lead_times
    lead_times_labels = lead_times_labels or lead_times
    label = 'Level [m]' if fsystem == 'atmos' else 'Depth [m]'
    fig = plt.figure(figsize=(4*len(fields_3d), 30//len(fields_3d)))
    if normalize:
        # plt.suptitle(f'Normalized {metric_name.upper()} Difference between {model1} and {model2} (Test period: {sdate} - {edate})', fontsize=18)
        plt.suptitle(f'Normalized {metric_name.upper()} Difference ({model1} - {model2})', fontsize=18)
    else:
        # plt.suptitle(f'{metric_name.upper()} Difference between {model1} and {model2} (Test period: {sdate} - {edate})', fontsize=18)
        plt.suptitle(f'{metric_name.upper()} Difference ({model1} - {model2})', fontsize=18)
    gs = fig.add_gridspec(10,len(fields_3d))
    # print('asdfbasjd')
    for i, field in enumerate(fields_3d):
        # print('asdfhabs')
        ax = fig.add_subplot(gs[:7, i])
        # print(metrics_model1_3d.shape, metrics_model2_3d.shape, lead_times1, lead_times2, i)
        diff = metrics_model1_3d[i,lead_times1].T.values - metrics_model2_3d[i,lead_times2].T.values
        ext = np.nanmax((np.abs(np.nanmin((diff))), np.abs(np.nanmax((diff)))))
        if normalize:
            im = ax.imshow(diff/ext, cmap=cmap, vmin=-1, vmax=1)
        else:
            im = ax.imshow(diff, cmap=cmap, vmin=-ext, vmax=ext)
        ax.set_title(f'{fields_info[field]["long_name"]}')
        if i!=0: 
            ax.set_yticks([])
        else:
            ax.set_ylabel(label, fontdict={'size':11})
            ax.set_yticks(np.arange(len(vert_levels)))
            ax.set_yticklabels([np.around(lev,0) for lev in vert_levels])
        ax.set_xticks(np.arange(len(lead_times)))
        ax.set_xticklabels(lead_times_labels)
        ax.set_xlabel('Lead Time [days]', fontdict={'size':11})
        unit = fields_info[field]["units"]
        plt.colorbar(im, ax=ax, label=f'{unit if "deg" not in unit else "°C"}')

    for i, field in enumerate(fields_2d):
        ax = fig.add_subplot(gs[8:, i])
        diff = metrics_model1_2d[i:i+1,lead_times1].values - metrics_model2_2d[i:i+1,lead_times2].values
        ext = np.nanmax((np.abs(np.nanmin((diff))), np.abs(np.nanmax((diff)))))
        if normalize:
            im = ax.imshow(diff/ext, cmap=cmap, vmin=-1, vmax=1)
        else:
            im = ax.imshow(diff, cmap=cmap, vmin=-ext, vmax=ext)
        ax.set_title(f'{fields_info[field]["long_name"]}')
        ax.set_yticks([])
        ax.set_xticks([])
        plt.colorbar(im, ax=ax, label=f'{fields_info[field]["units"]}', orientation='horizontal')

    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def metrics_scorecard_by_lead_times(
    metrics_ds: xr.Dataset, 
    metric_name: str, 
    fields_info: dict, 
    fsystem: str, 
    sdate: str, 
    edate: str, 
    lead_times: list, 
    lead_times_labels: list, 
    fields_3d: list[str], 
    fields_2d: list[str], 
    vert_levels: str, 
    model: str = 'Neptune', 
    cmap: str = 'PuBu', # vanno bene ['PuBu', 'YlGnBu']
    dvmin: int = None, 
    dvmax: int = None, 
    outfile: str = None, 
):
    metrics_ds_3d = metrics_ds[fields_3d].to_array()
    metrics_ds_2d = metrics_ds[fields_2d].to_array()
    label = 'Level [m]' if fsystem == 'atmos' else 'Depth [m]'
    fig = plt.figure(figsize=(4*len(fields_3d), 30//len(fields_3d)))
    plt.suptitle(f'{metric_name.upper()} of {model} (Test period: {sdate} - {edate})', fontsize=18)
    gs = fig.add_gridspec(10,len(fields_3d))
    for i, field in enumerate(fields_3d):
        ax = fig.add_subplot(gs[:7, i])
        data = metrics_ds_3d[i,lead_times].T
        kwargs = dict()
        if dvmin is not None: kwargs['vmin'] = dvmin
        if dvmax is not None: kwargs['vmax'] = dvmax
        im = ax.imshow(data, cmap=cmap, **kwargs)
        ax.set_title(f'{fields_info[f"{field}"]["long_name"]}')
        if i!=0: 
            ax.set_yticks([])
        else:
            ax.set_ylabel(label, fontdict={'size':11})
            ax.set_yticks(np.arange(len(vert_levels)))
            ax.set_yticklabels([np.around(lev,0) for lev in vert_levels])
        ax.set_xticks(np.arange(len(lead_times)))
        ax.set_xticklabels(lead_times_labels)
        ax.set_xlabel('Lead Time [days]', fontdict={'size':11})
        unit = fields_info[f"{field}"]["units"]
        colorbar_label = f'{unit if "deg" not in unit else "°C"}'
        if 'acc' in metric_name.lower(): colorbar_label = ''
        plt.colorbar(im, ax=ax, label=colorbar_label)
    for i, field in enumerate(fields_2d):
        ax = fig.add_subplot(gs[8:, i])
        data = metrics_ds_2d[i:i+1,lead_times]
        kwargs = dict()
        if dvmin is not None: kwargs['vmin'] = dvmin
        if dvmax is not None: kwargs['vmax'] = dvmax
        im = ax.imshow(data, cmap=cmap, **kwargs)
        ax.set_title(f'{fields_info[f"{field}"]["long_name"]}')
        ax.set_yticks([])
        ax.set_xticks([])
        colorbar_label = f'{fields_info[f"{field}"]["units"]}'
        if 'acc' in metric_name.lower(): colorbar_label = ''
        plt.colorbar(im, ax=ax, label=colorbar_label, orientation='horizontal')
        
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def plot_spatial_average_forecasts(
    time_avg_fc_3d, 
    time_avg_fc_2d, 
    plot_type, 
    drivers, 
    fsystem, 
    lead_times, 
    sdate, 
    edate, 
    outfile: str = None, 
):
    print(f'DEPRECATED PLOT, AND WILL BE SUBSTITUTED BY plot_spatial_metric()')
    return 
    fields_3d = drivers['output'][fsystem]['fields_3d']
    fields_2d = drivers['output'][fsystem]['fields_2d']
    cmap = 'plasma'
    fontdict = {'weight': 'bold', 'size': 12}
    if plot_type == 'persistence':
        title = 'Persistence'
    elif plot_type == 'system':
        title = f'Neptune {fsystem.capitalize()}'
    fig = plt.figure(figsize = (5.0 * (len(lead_times)), 2.3 * (len(fields_3d)+len(fields_2d))))
    plt.suptitle(f'{title} Average ({sdate} - {edate})', fontsize=15, weight='bold')
    gs = fig.add_gridspec(len(fields_3d) + len(fields_2d), len(lead_times))
    k = 0
    for i, field in enumerate(fields_3d):
        vmin = np.nanmin((np.nanmin(time_avg_fc_3d[i, :, 0]), np.nanmin(time_avg_fc_3d[i, :, 0])))
        vmax = np.nanmax((np.nanmax(time_avg_fc_3d[i, :, 0]), np.nanmax(time_avg_fc_3d[i, :, 0])))
        for l, lead_time in enumerate(lead_times):
            spatial_avg = time_avg_fc_3d[i, lead_time, 0]
            ax = fig.add_subplot(gs[k, l])
            im = ax.imshow(spatial_avg, cmap=cmap, vmin=vmin, vmax=vmax)
            if l == 0:
                ax.set_ylabel(f'{field}', fontdict=fontdict)
            if i == 0:
                ax.set_title(f't + {lead_time}', fontdict=fontdict)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im)
        k += 1
    for i, field in enumerate(fields_2d):
        vmin = np.nanmin((np.nanmin(time_avg_fc_2d[i, :]), np.nanmin(time_avg_fc_2d[i, :])))
        vmax = np.nanmax((np.nanmax(time_avg_fc_2d[i, :]), np.nanmax(time_avg_fc_2d[i, :])))
        for l, lead_time in enumerate(lead_times):
            spatial_avg = time_avg_fc_2d[i, lead_time]
            ax = fig.add_subplot(gs[k, l])
            im = ax.imshow(spatial_avg, cmap=cmap, vmin=vmin, vmax=vmax)
            if l == 0:
                ax.set_ylabel(f'{field}', fontdict=fontdict)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im)
        k += 1
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def plot_spatial_average_bias_forecasts(
    time_avg_fc_3d, 
    time_avg_rea_3d, 
    time_avg_fc_2d, 
    time_avg_rea_2d, 
    fsystem, 
    lead_times: list, 
    lead_times_labels: list, 
    fields_3d: list, 
    fields_2d: list, 
    sdate, 
    edate, 
    model: str = 'Neptune', 
    ext_mul: int = 20, 
    outfile = None, 
):
    print(f'DEPRECATED PLOT AS IT CONTAINS A METRICS DIFFERENCE')
    return
    cmap = 'RdBu_r'
    fontdict = {'weight': 'bold', 'size': 12}

    fig = plt.figure(figsize = (5.0 * (len(lead_times)), 2.3 * (len(fields_3d)+len(fields_2d))))
    plt.suptitle(f'{model} {fsystem.capitalize()} Average Bias ({sdate} - {edate})', fontsize=15, weight='bold')
    gs = fig.add_gridspec(len(fields_3d) + len(fields_2d), len(lead_times))
    k = 0
    for i, field in enumerate(fields_3d):
        time_avg_bias = time_avg_fc_3d[i, :, 0] - time_avg_rea_3d[i, :, 0]
        vmin = np.nanmin((np.nanmin(time_avg_bias), np.nanmin(time_avg_bias)))
        vmax = np.nanmax((np.nanmax(time_avg_bias), np.nanmax(time_avg_bias)))
        ext = np.nanmax((np.abs(vmin), np.abs(vmax))) / ext_mul
        for l, _ in enumerate(lead_times):
            spatial_avg_bias = time_avg_bias[l]
            ax = fig.add_subplot(gs[k, l])
            im = ax.imshow(spatial_avg_bias, cmap=cmap, vmin=-ext, vmax=ext)
            if l == 0:
                ax.set_ylabel(f'{field}', fontdict=fontdict)
            if i == 0:
                ax.set_title(f't + {lead_times_labels[l]}', fontdict=fontdict)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im)
        k += 1
    for i, field in enumerate(fields_2d):
        time_avg_bias = time_avg_fc_2d[i, :] - time_avg_rea_2d[i, :]
        vmin = np.nanmin((np.nanmin(time_avg_bias), np.nanmin(time_avg_bias)))
        vmax = np.nanmax((np.nanmax(time_avg_bias), np.nanmax(time_avg_bias)))
        ext = np.nanmax((np.abs(vmin), np.abs(vmax))) / ext_mul
        for l, _ in enumerate(lead_times):
            spatial_avg_bias = time_avg_bias[l]
            ax = fig.add_subplot(gs[k, l])
            im = ax.imshow(spatial_avg_bias, cmap=cmap, vmin=-ext, vmax=ext)
            if l == 0:
                ax.set_ylabel(f'{field}', fontdict=fontdict)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im)
        k += 1
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def plot_spatial_metric(
    metric_spatial, 
    metric_name: str, 
    model: str, 
    fields_3d: list[str], 
    fields_2d: list[str], 
    fields_info: dict, 
    lead_times: list, 
    lead_times_labels: list, 
    sdate, 
    edate, 
    outfile, 
    dvmin: float = None, # data minimum
    dvmax: float = None, # data maximum
    divide_by: int = 1, # to suppress hotspots in vmin and vmax
    cmap: str = 'RdBu_r', 
):
    fontdict = {'weight': 'bold', 'size': 12}
    metric_spatial_3d = metric_spatial[fields_3d].to_array()
    metric_spatial_2d = metric_spatial[fields_2d].to_array()
    fig = plt.figure(figsize = (5.0 * (len(lead_times)), 2.3 * (len(fields_3d)+len(fields_2d))))
    plt.suptitle(f'{model} {metric_name.upper()} ({sdate} - {edate})', fontsize=15, weight='bold')
    gs = fig.add_gridspec(len(fields_3d) + len(fields_2d), len(lead_times))
    k = 0
    for i, field in enumerate(fields_3d):
        vmin = dvmin or np.nanmin(metric_spatial_3d[i, :, 0]) / divide_by
        vmax = dvmax or np.nanmax(metric_spatial_3d[i, :, 0]) / divide_by
        for l, _ in enumerate(lead_times):
            if len(metric_spatial_3d[i, lead_times[l], 0].shape)!=2:
                metric = metric_spatial_3d[i, lead_times[l]]
            else:
                metric = metric_spatial_3d[i, lead_times[l], 0]
            ax = fig.add_subplot(gs[k, l])
            im = ax.imshow(metric, cmap=cmap, vmin=vmin, vmax=vmax)
            if l == 0:
                ax.set_ylabel(f'{fields_info[f"{field}"]["long_name"][:20]}', fontdict=fontdict)
            if i == 0:
                ax.set_title(f't + {lead_times_labels[l]}', fontdict=fontdict)
            ax.set_xticks([])
            ax.set_yticks([])
            unit = fields_info[f"{field}"]["units"]
            plt.colorbar(im, label=f'{unit if "deg" not in unit else "°C"}')
        k += 1
    for i, field in enumerate(fields_2d):
        vmin = dvmin or np.nanmin(metric_spatial_2d[i, :]) / divide_by
        vmax = dvmax or np.nanmax(metric_spatial_2d[i, :]) / divide_by
        for l, _ in enumerate(lead_times):
            metric = metric_spatial_2d[i, lead_times[l]]
            ax = fig.add_subplot(gs[k, l])
            im = ax.imshow(metric, cmap=cmap, vmin=vmin, vmax=vmax)
            if l == 0:
                ax.set_ylabel(f'{fields_info[f"{field}"]["long_name"][:20]}', fontdict=fontdict)
            ax.set_xticks([])
            ax.set_yticks([])
            unit = fields_info[f"{field}"]["units"]
            plt.colorbar(im, label=f'{unit if "deg" not in unit else "°C"}')
        k += 1
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def plot_spatial_normalized_rmse_difference(
    rmse_fc_3d_spatial, 
    rmse_ps_3d_spatial, 
    rmse_fc_2d_spatial, 
    rmse_ps_2d_spatial, 
    fields_3d: list, 
    fields_2d: list, 
    lead_times: list, 
    lead_times_labels: list, 
    model1: str, 
    model2: str, 
    sdate, 
    edate, 
    outfile, 
):
    print(f'DEPRECATED PLOT THAT WILL BE SUBSTITUTED WITH plot_spatial_metrics_difference()')
    return
    cmap = 'RdBu_r'
    fontdict = {'weight': 'bold', 'size': 12}
    
    fig = plt.figure(figsize = (5.0 * (len(lead_times)), 2.3 * (len(fields_3d)+len(fields_2d))))
    vm = 0.1
    plt.suptitle(f'Normalized RMSE Difference ({model1} - {model2}) ({sdate} - {edate})', fontsize=15, weight='bold')
    gs = fig.add_gridspec(len(fields_3d) + len(fields_2d), len(lead_times))
    k = 0
    for i, field in enumerate(fields_3d):
        rmse_diff_spatial = rmse_fc_3d_spatial[i, :, 0].values - rmse_ps_3d_spatial[i, :, 0].values
        ext = np.nanmax((np.abs(np.nanmin(rmse_diff_spatial)), np.abs(np.nanmax(rmse_diff_spatial))))
        for l, _ in enumerate(lead_times):
            ax = fig.add_subplot(gs[k, l])
            diff = rmse_diff_spatial[lead_times[l]]
            im = ax.imshow(diff/ext, cmap=cmap, vmin=-vm, vmax=vm)
            if l == 0:
                ax.set_ylabel(f'{field}', fontdict=fontdict)
            if i == 0:
                ax.set_title(f't + {lead_times_labels[l]}', fontdict=fontdict)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im)
        k += 1
    for i, field in enumerate(fields_2d):
        rmse_diff_spatial = rmse_fc_2d_spatial[i, :].values - rmse_ps_2d_spatial[i, :].values
        ext = np.nanmax((np.abs(np.nanmin(rmse_diff_spatial)), np.abs(np.nanmax(rmse_diff_spatial))))
        for l, _ in enumerate(lead_times):
            diff = rmse_diff_spatial[lead_times[l]]
            ax = fig.add_subplot(gs[k, l])
            im = ax.imshow(diff/ext, cmap=cmap, vmin=-vm, vmax=vm)
            if l == 0:
                ax.set_ylabel(f'{field}', fontdict=fontdict)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im)
        k += 1
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def plot_spatial_metrics_difference(
    metric_model1: xr.Dataset, 
    metric_model2: xr.Dataset, 
    metric_name: str, 
    fields_3d: list[str], 
    fields_2d: list[str], 
    fields_info: dict, 
    lead_times_ids: list, 
    lead_times_labels: list, 
    model1: str, 
    model2: str, 
    sdate: str, 
    edate: str, 
    dext: float = None, 
    cmap: str = 'RdBu_r', 
    outfile = None, 
):
    metric_model1_3d = metric_model1[fields_3d].to_array() # .transpose('variable','prediction_timedelta','depth','latitude','longitude')
    metric_model1_2d = metric_model1[fields_2d].to_array() # .transpose('variable','prediction_timedelta','latitude','longitude')
    metric_model2_3d = metric_model2[fields_3d].to_array() # .transpose('variable','prediction_timedelta','depth','latitude','longitude')
    metric_model2_2d = metric_model2[fields_2d].to_array() # .transpose('variable','prediction_timedelta','latitude','longitude')
    fontdict = {'weight': 'bold', 'size': 12}
    
    fig = plt.figure(figsize = (5.0 * (len(lead_times_labels)), 2.3 * (len(fields_3d)+len(fields_2d))))
    plt.suptitle(f'{model1} - {model2} {metric_name.upper()} Difference ({sdate} - {edate})', fontsize=15, weight='bold')
    gs = fig.add_gridspec(len(fields_3d) + len(fields_2d), len(lead_times_labels))
    k = 0
    for i, field in enumerate(fields_3d):
        # print(metric_model1_3d.shape, metric_model2_3d.shape)
        metric_diff = metric_model1_3d[i, :, 0].values - metric_model2_3d[i, :, 0].values
        # print(metric_diff.shape)
        ext = np.nanmax((np.abs(np.nanmin(metric_diff)), np.abs(np.nanmax(metric_diff))))
        ext = dext or ext/10
        for l,(idx,label) in enumerate(zip(lead_times_ids, lead_times_labels)):
            ax = fig.add_subplot(gs[k, l])
            diff = metric_diff[idx]
            im = ax.imshow(diff, cmap=cmap, vmin=-ext, vmax=ext)
            if l == 0:
                ax.set_ylabel(f'{fields_info[f"{field}"]["long_name"]}', fontdict=fontdict)
            if i == 0:
                ax.set_title(f't + {label}', fontdict=fontdict)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im)
        k += 1
    for i, field in enumerate(fields_2d):
        metric_diff = metric_model1_2d[i, :].values - metric_model2_2d[i, :].values
        ext = np.nanmax((np.abs(np.nanmin(metric_diff)), np.abs(np.nanmax(metric_diff))))
        ext = dext or ext/10
        for l,(idx,label) in enumerate(zip(lead_times_ids, lead_times_labels)):
            diff = metric_diff[idx]
            ax = fig.add_subplot(gs[k, l])
            im = ax.imshow(diff, cmap=cmap, vmin=-ext, vmax=ext)
            if l == 0:
                ax.set_ylabel(f'{fields_info[f"{field}"]["long_name"]}', fontdict=fontdict)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im)
        k += 1
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def plot_per_variable_loss(
    df: pd.DataFrame, 
    offset: int = 0, 
    ttype: str = 'train', 
    outfile = None, 
):
    plt.subplots(figsize=(13,4))
    for col in df.columns:
        if col == 'train_loss' or col == 'val_loss': continue
        if not ttype in col: continue
        label = col.split('_')[1].capitalize()
        plt.plot(range(len(df))[offset:], df[f'{col}'][offset:], label = label)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()

def plot_dynamical_system_loss(dsys: pd.DataFrame, offset: int = 0, val_min = None, val_max = None, outfile = None):
    eps = 1e-2
    ymax = max((dsys['val_loss'][offset:].max(), dsys['train_loss'][offset:].max()))
    ymin = min((dsys['val_loss'][offset:].min(), dsys['train_loss'][offset:].min()))
    bbox = dict(boxstyle='round', edgecolor='green', facecolor='white', alpha=1.0)
    if val_max: ymax = min((ymax, val_max))
    if val_min: ymin = max((ymin, val_min))
    _, ax = plt.subplots(figsize=(13,4))
    plt.plot(range(len(dsys))[offset:], dsys['train_loss'][offset:], label = 'Train Loss')
    plt.plot(range(len(dsys))[offset:], dsys['val_loss'][offset:], label = 'Valid Loss')
    plt.vlines(x=150, ymin=ymin, ymax=ymax, color='green', linestyle='dashed')
    ax.text(150, ymax/2, 'rollout t+2', rotation=90, fontsize=8, color='green', horizontalalignment='center', verticalalignment='center', bbox=bbox)
    plt.vlines(x=160, ymin=ymin, ymax=ymax, color='green', linestyle='dashed')
    ax.text(160, ymax/2, 'rollout t+3', rotation=90, fontsize=8, color='green', horizontalalignment='center', verticalalignment='center', bbox=bbox)
    plt.vlines(x=170, ymin=ymin, ymax=ymax, color='green', linestyle='dashed')
    ax.text(170, ymax/2, 'rollout t+4', rotation=90, fontsize=8, color='green', horizontalalignment='center', verticalalignment='center', bbox=bbox)
    plt.vlines(x=180, ymin=ymin, ymax=ymax, color='green', linestyle='dashed')
    ax.text(180, ymax/2, 'rollout t+5', rotation=90, fontsize=8, color='green', horizontalalignment='center', verticalalignment='center', bbox=bbox)
    plt.vlines(x=190, ymin=ymin, ymax=ymax, color='green', linestyle='dashed')
    ax.text(190, ymax/2, 'rollout t+6', rotation=90, fontsize=8, color='green', horizontalalignment='center', verticalalignment='center', bbox=bbox)
    plt.ylim(ymin - eps, ymax + eps)
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=200)
    else:
        plt.show()


def rmse_scorecard_against_observations(df, model, fields, fields_info, lead_times, sdate, edate, cmap = 'RdBu_r', outfile=None):
    vert_levels = [np.around(lev, 2) for lev in df['depth'].unique()]
    n_depths = len(vert_levels)
    n_lead_times = len(lead_times)
    label = 'Depth [m]'
    fields_3d = [field for field in fields if field != 'zos_oras']
    field_2d = 'zos_oras'

    fig = plt.figure(figsize=(3*len(fields_3d), 30//len(fields_3d)))
    plt.suptitle(f'{" ".join(model.split("_"))} RMSE Scorecard against Observations ({sdate} - {edate})', fontsize=20)

    gs = fig.add_gridspec(8,len(fields_3d))
    for i, field in enumerate(fields_3d):
        ax = fig.add_subplot(gs[0:4, i])
        title = f'{fields_info[field].long_name}'
        colorbar = f'{fields_info[field].units}'
        if 'degrees' in colorbar: colorbar = '°C'
        rmse = np.sqrt(df[df['model']==model].sort_values(by=['depth', 'lead_time'])[field].values.reshape(n_depths, n_lead_times))
        im = ax.imshow(rmse, cmap=cmap)
        fig.colorbar(im, ax=ax, label=colorbar)
        ax.set_title(title)
        if i!=0: 
            ax.set_yticks([])
        else:
            ax.set_ylabel(label, fontdict={'size':11})
            ax.set_yticks(np.arange(len(vert_levels)))
            ax.set_yticklabels(vert_levels)
        ax.set_xticks(np.arange(len(lead_times)))
        ax.set_xticklabels(lead_times)
        ax.set_xlabel('Lead Time [days]', fontdict={'size':11})

    ax = fig.add_subplot(gs[6:8, 1:3])
    title = f'{fields_info[field_2d].long_name}'
    colorbar = f'{fields_info[field_2d].units}'
    rmse = np.sqrt(df[df['model']==model].sort_values(by=['depth', 'lead_time'])[field_2d].values[:n_lead_times])[np.newaxis,:]
    im = ax.imshow(rmse, cmap=cmap)
    fig.colorbar(im, ax=ax, label=colorbar, orientation='horizontal')
    ax.set_title(title)
    ax.set_yticks([0])
    ax.set_xticks(np.arange(len(lead_times)))
    ax.set_xticklabels(lead_times)
    ax.set_xlabel('Lead Time [days]', fontdict={'size':11})

    fig.subplots_adjust(right=0.8, hspace=-0.45, wspace=0.1)
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def rmse_scorecard_difference_against_observations(df, models, fields, fields_info, lead_times, sdate, edate, cmap = 'RdBu_r', outfile=None):
    model1, model2 = models
    vert_levels = [np.around(lev, 2) for lev in df['depth'].unique()]
    n_depths = len(vert_levels)
    n_lead_times = len(lead_times)
    label = 'Depth [m]'
    fields_3d = [field for field in fields if field != 'zos_oras']
    field_2d = 'zos_oras'

    fig = plt.figure(figsize=(3.5*len(fields_3d), 30//len(fields_3d)))
    plt.suptitle(f'RMSE Scorecard Difference ({" ".join(model1.split("_"))} - {" ".join(model2.split("_"))})\nagainst Observations ({sdate} - {edate})', fontsize=16)

    gs = fig.add_gridspec(8,len(fields_3d))
    for i, field in enumerate(fields_3d):
        ax = fig.add_subplot(gs[0:4, i])
        title = f'{fields_info[field].long_name}'
        colorbar = f'{fields_info[field].units}'
        if 'degrees' in colorbar: colorbar = '°C'
        rmse1 = np.sqrt(df[df['model']==model1].sort_values(by=['depth', 'lead_time'])[field].values.reshape(n_depths, n_lead_times))
        rmse2 = np.sqrt(df[df['model']==model2].sort_values(by=['depth', 'lead_time'])[field].values.reshape(n_depths, n_lead_times))
        rmse = rmse1 - rmse2
        ext = np.nanmax((np.abs(np.nanmin(rmse)), np.nanmax(rmse)))
        im = ax.imshow(rmse, cmap=cmap, vmin=-ext, vmax=ext)
        fig.colorbar(im, ax=ax, label=colorbar)
        ax.set_title(title)
        if i!=0: 
            ax.set_yticks([])
        else:
            ax.set_ylabel(label, fontdict={'size':11})
            ax.set_yticks(np.arange(len(vert_levels)))
            ax.set_yticklabels(vert_levels)
        ax.set_xticks(np.arange(len(lead_times)))
        ax.set_xticklabels(lead_times)
        ax.set_xlabel('Lead Time [days]', fontdict={'size':11})
    
    ax = fig.add_subplot(gs[6:8, 1:3])
    title = f'{fields_info[field_2d].long_name}'
    colorbar = f'{fields_info[field_2d].units}'
    rmse1 = np.sqrt(df[df['model']==model1].sort_values(by=['depth', 'lead_time'])[field_2d].values[:n_lead_times])[np.newaxis,:]
    rmse2 = np.sqrt(df[df['model']==model2].sort_values(by=['depth', 'lead_time'])[field_2d].values[:n_lead_times])[np.newaxis,:]
    rmse = rmse1 - rmse2
    ext = np.nanmax((np.abs(np.nanmin(rmse)), np.nanmax(rmse)))
    im = ax.imshow(rmse, cmap=cmap, vmin=-ext, vmax=ext)
    fig.colorbar(im, ax=ax, label=colorbar, orientation='horizontal')
    ax.set_title(title)
    ax.set_yticks([0])
    ax.set_xticks(np.arange(len(lead_times)))
    ax.set_xticklabels(lead_times)
    ax.set_xlabel('Lead Time [days]', fontdict={'size':11})
    
    fig.subplots_adjust(right=0.8, hspace=-0.45, wspace=0.1)
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def plot_regional_field_by_lead_time(regional_timeseries, regions, lead_times, field, fields_info, plot_type, sdate, edate, fsystem, out_img_dst, outfile=None):
    for region in regions.keys():
        _, ax = plt.subplots(figsize=(13,4))
        if outfile is not None:
            outfile = os.path.join(out_img_dst, 'regional', f'regional_{field}_plot_{region}-{plot_type}-{sdate}_{edate}_{fsystem}.png')
            os.makedirs(os.path.join(out_img_dst, 'regional'), exist_ok=True)
        alpha = 1.0
        x_axis = np.arange(regional_timeseries[region].time.shape[0])
        title = ' '.join([reg.capitalize() for reg in region.split('_')])
        for lead_time in [0] + lead_times:
            y_axis = regional_timeseries[region][str(lead_time)]
            label = f'Forecast t + {lead_time}' if lead_time!=0 else 'Reanalysis'
            if plot_type == 'bnw':
                plot_args = {'color': 'black', 'linestyle' : 'dashed' if lead_time != 0 else '-'}
                alpha -= 0.15
            elif plot_type == 'rgb':
                plot_args = {}
            ax.plot(x_axis, y_axis, label = label, alpha=alpha, **plot_args)
        ax.set_xlabel('Time [days]')
        date_range = pd.date_range(sdate, edate, freq='MS')
        time_labels = [f'{t.year}-{t.month:02d}' for t in date_range]
        time_ticks = [pd.Period(t, freq='1D').day_of_year for t in date_range]
        ax.set_xticklabels(time_labels)
        ax.set_xticks(time_ticks)
        ylabel = f'{fields_info[field].long_name}'
        if 'degrees' in fields_info[field].units: ylabel += ' [°C]'
        else: ylabel += f' [{fields_info[field].units}]'
        ax.set_ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def polar_plot_ice_fields(
        ds: xr.Dataset, 
        lead_times: list[int], 
        fields: list[str], 
        fields_info: dict, 
        remove_below_x: int, 
        sdate: str, 
        edate: str, 
        metric_name: str = 'rmse', 
        model_name: str = 'Neptune', 
        polar: str = 'north', 
        cmap: str = 'jet', 
        divide_by: int = 1, 
        dext: list[float] = None, 
        outfile: str = None, 
    ):
    nrows = len(fields)
    ncols = len(lead_times)
    if polar == 'south':
        projection = ccrs.SouthPolarStereo()
        extent = [0, 360, -90, -50]
    elif polar == 'north':
        projection = ccrs.NorthPolarStereo()
        extent = [0, 360, 60, 90]
    fig, axes = plt.subplots(nrows, ncols, figsize = (10 * nrows, 1 * ncols), subplot_kw = {'projection': projection})
    for j, field in enumerate(fields):
        all_fields_data = ds[field].isel(prediction_timedelta=[i+1 for i in range(len(lead_times))]).data
        vmin, vmax = np.nanmin(all_fields_data), np.nanmax(all_fields_data)
        if dext is not None:
            vmin, vmax = dext[0], dext[1]
        else:
            if vmin < 0.0:
                ext = np.nanmax((np.abs(vmin), vmax)) / divide_by
                vmin, vmax = -ext, ext
        axs = []
        for i, lead_time in enumerate(lead_times):
            ax = axes[j, i]
            if j == nrows - 1:
                ax.text(
                    0.5, -0.2, 
                    f'Lead Time [days]\nt + {lead_time}', 
                    va='bottom', 
                    ha='center', 
                    rotation='horizontal', 
                    rotation_mode='anchor', 
                    transform=ax.transAxes
                )
            # get field data
            field_data = all_fields_data[i]
            # put the 0.0 RMSE to nan
            if remove_below_x is not None:
                field_data = np.where(field_data <= remove_below_x, np.nan, field_data)
            # set the extent of the map
            ax.set_extent(extent, ccrs.PlateCarree())
            im = ax.imshow(
                field_data, 
                extent=[0, 360, -90, 90], 
                transform=ccrs.PlateCarree(), 
                cmap=cmap, vmin=vmin, vmax=vmax)
            ax.coastlines()
            axs.append(ax)
            capitalized_field_name = " ".join([v.capitalize() for v in fields_info[field].long_name.split(" ")])
            if i == ncols - 1:
                if metric_name=='acc':
                    colorbar_label = f'{capitalized_field_name}\n{metric_name.upper()}'
                else:
                    colorbar_label = f'{capitalized_field_name}\n{metric_name.upper()} [{fields_info[field].units}]'
                # cax = fig.add_axes([0.95, 0.1, 0.03, 0.4]) # [left, bottom, width, height]
                fig.colorbar(
                    im, ax=axs, # cax=cax, 
                    label=colorbar_label, 
                    # format=ScalarFormatter()
                )
    fig.suptitle(f'{metric_name.upper()} of {model_name} on {polar.capitalize()} Pole over Test Set ({sdate} - {edate})', fontsize=14)
    # plt.tight_layout()
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def power_spectra_plot_horizontal(
    psd_data: dict,  
    labels: dict, # dict of labels associated with the PSDs
    colors: dict, # dict of colors associated with the PSDs
    alphas: dict, # dict of alpha values
    fields_info: dict, # fields information for prettier plots
    fields_3d: list, 
    fields_2d: list, 
    outfile: str = None, 
):
    fontdict = {'weight': 'bold', 'size': 12}
    width, height = 4 * max(len(fields_3d), len(fields_2d)), 4 * 2
    fig = plt.figure(figsize = (width, height))
    plt.suptitle(f'Power Spectral Density', fontsize=15, weight='bold')
    gs = fig.add_gridspec(2, max(len(fields_3d), len(fields_2d)))
    for i,field in enumerate(fields_3d + fields_2d):
        if field in fields_3d: 
            row = 0
            k = i
        else: 
            row = 1
            k =  i - len(fields_3d)
        ax = fig.add_subplot(gs[row, k])
        if i == 0: main_ax = ax
        for p,model in enumerate(psd_data.keys()):
            if field not in psd_data[model].keys(): continue
            psd = psd_data[model][field]
            if model in ['oras5','om4','rea','clm']:
                kwargs = dict(linestyle='dashdot', linewidth=0.5)
            else:
                kwargs = dict(linewidth=1.5)
            ax.loglog(*psd, label=f'{labels[model]}', color=colors[model], alpha=alphas[model], **kwargs)
        ax.invert_xaxis()
        ax.set_title(f'{fields_info[field]["long_name"]}', fontdict=fontdict)
        if k == 0:
            ax.set_ylabel(f'Mean Power', fontdict=fontdict)
        if row == 1:
            ax.set_xlabel(f'Wavelength [km]', fontdict=fontdict)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    legend = main_ax.legend(loc='upper center', markerscale=20, edgecolor='gray', framealpha=1, ncol=4, bbox_to_anchor=(2.3, -1.4))
    for line in legend.get_lines():
        line.set_linewidth(2)
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def power_spectra_plot_vertical(
    psd_data: dict,  
    labels: dict, # dict of labels associated with the PSDs
    colors: dict, # dict of colors associated with the PSDs
    alphas: dict, # dict of alpha values
    fields_info: dict, # fields information for prettier plots
    fields_3d: list, 
    fields_2d: list, 
    outfile: str = None, 
):
    fontdict = {'weight': 'bold', 'size': 12}
    width, height = 12 * 2,  3 * max(len(fields_3d), len(fields_2d))
    fig = plt.figure(figsize = (width, height))
    plt.suptitle(f'Power Spectral Density', fontsize=15, weight='bold', y=0.95)
    gs = fig.add_gridspec(max(len(fields_3d), len(fields_2d)), 2)
    for i,field in enumerate(fields_3d + fields_2d):
        if field in fields_3d: 
            col = 0
            k = i
        else: 
            col = 1
            k =  i - len(fields_3d)
        ax = fig.add_subplot(gs[k, col])
        if i == 0: main_ax = ax
        for p,model in enumerate(psd_data.keys()):
            if field not in psd_data[model].keys(): continue
            psd = psd_data[model][field]
            if model in ['oras5','om4','rea','clm']:
                kwargs = dict(linestyle='dashdot', linewidth=1)
            else:
                kwargs = dict(linewidth=2)
            ax.loglog(*psd, label=f'{labels[model]}', color=colors[model], alpha=alphas[model], **kwargs)
        ax.invert_xaxis()
        ax.set_title(f'{fields_info[field]["long_name"]}', fontdict=fontdict)
        if col == 0:
            ax.set_ylabel(f'Mean Power', fontdict=fontdict)
        if k == max(len(fields_3d), len(fields_2d))-1:
            ax.set_xlabel(f'Wavelength [km]', fontdict=fontdict)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    legend = main_ax.legend(loc='upper center', markerscale=20, edgecolor='gray', framealpha=1, ncol=4, bbox_to_anchor=(1.0, -4.0))
    for line in legend.get_lines():
        line.set_linewidth(2)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def power_spectra_plot_grid(
    psd_data: dict, 
    labels: dict, # dict of labels associated with the PSDs
    colors: dict, # dict of colors associated with the PSDs
    alphas: dict, # dict of alpha values
    fields_info: dict, # fields information for prettier plots
    fields_3d: list, 
    fields_2d: list, 
    lead_times: list[int], 
    outfile: str = None, 
):
    fontdict = dict(weight='bold', size=16)
    width, height = 4 * len(lead_times), 4 * len(fields_2d + fields_3d)
    fig = plt.figure(figsize = (width, height))
    plt.suptitle(f'Power Spectral Density', fontsize=20, weight='bold', y=0.9)
    gs = fig.add_gridspec(len(fields_3d+fields_2d), len(lead_times))
    for l, lead_time in enumerate(lead_times):
        for f, field in enumerate(fields_3d + fields_2d):
            ax = fig.add_subplot(gs[f,l])
            if l == f == 0:
                main_ax = ax
            for i, model in enumerate(psd_data.keys()):
                if field not in psd_data[model][lead_time].keys(): continue
                psd = psd_data[model][lead_time][field]
                if model in ['oras5','om4','rea','clm']:
                    kwargs = dict(linestyle='dashdot', linewidth=0.5)
                else:
                    kwargs = dict(linewidth=1.5)
                ax.loglog(*psd, label=labels[model], color=colors[model], alpha=alphas[model], **kwargs)
            if f == 0:
                ax.set_title(f't + {lead_time}', fontdict=fontdict)
            if l == 0:
                ax.set_ylabel(f'{fields_info[field]["long_name"][:25]}', fontdict=fontdict)
            else: 
                ax.set_yticks([])
                ax.set_yticklabels([])
            if f == len(fields_3d + fields_2d) - 1:
                ax.set_xlabel(f'Wavelength [km]', fontdict=fontdict)
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])
            ax.invert_xaxis()
    box = main_ax.get_position()
    main_ax.set_position([box.x0, box.y0, box.width, box.height])
    legend = main_ax.legend(loc='upper center', markerscale=20, edgecolor='gray', framealpha=1, ncol=4, bbox_to_anchor=(2.5, -8.0))
    for line in legend.get_lines():
        line.set_linewidth(2)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def plot_spatial_metric_on_globe(
    metric, fig, grid, lons, lats, central_longitude, title, cmap, norm, levels, left_labels = True, bottom_labels = True
):
    ax = fig.add_subplot(grid, projection=ccrs.Robinson(central_longitude=central_longitude))
    im = ax.contourf(lons, lats, metric, levels=levels, cmap=cmap, norm=norm, extend='both', transform=ccrs.PlateCarree())
    ax.set_title(f'{title} ({np.nanmean(metric):.3f})', fontsize=14)
    # set white background for spatial plots
    ax.set_facecolor("white")
    # plot the coastlines along the image
    ax.coastlines(resolution='110m', linewidth=0.6, edgecolor="black", alpha=1.0)
    # color the land
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor='#B1B2B4', alpha=0.3, zorder=1)
    # create ticks every 90 deg
    longitude_ticks = mticker.MultipleLocator(base=90.0)
    gl = ax.gridlines(draw_labels=True, xlocs=longitude_ticks, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlabel_style = {"size": 12}
    gl.ylabel_style = {"size": 12}
    return im, ax


def plot_metrics_on_robinson_globe(
        metric_values: list, 
        model_names: list[str], 
        grid_spec_args: dict, 
        central_longitude: float, 
        metric_name: str, 
        field: str, 
        fields_info: dict, 
        lats, 
        lons, 
        cmap, 
        norm, 
        levels, 
        start_year: int, 
        end_year: int, 
        outfile = None, 
        use_units: bool = True, 
    ):
    def parse_ids(ids: str):
        ri, ci = ids.split(',')
        if ':' in ri:
            ri = ri.split(':')
            ri = slice(int(ri[0]), int(ri[1]))
        else: ri = int(ri)
        if ':' in ci:
            ci = ci.split(':')
            ci = slice(int(ci[0]), int(ci[1]))
        else: ci = int(ci)
        return ri, ci
    fig, axes = plt.subplots(1, 1, figsize=(15,5.5))
    fig.suptitle(f'Spatial {fields_info[field]["long_name"].capitalize()} {metric_name.upper()} ({start_year}-{end_year})', fontsize=16, y=1.05)
    axes.axis('off')

    # initialize grid spec
    gs = GridSpec(*grid_spec_args['num_figures'], figure=fig)
    axs = []

    for metric, name, grid_pos in zip(metric_values, model_names, grid_spec_args['positions']):
        im, ax = plot_spatial_metric_on_globe(
            metric=metric, grid=gs[parse_ids(grid_pos)], title=name, 
            fig=fig, lons=lons, lats=lats, central_longitude=central_longitude, cmap=cmap, norm=norm, levels=levels)
        axs.append(ax)

    # single colorbar
    cax = fig.add_axes([0.25, -0.05, 0.5, 0.03]) # [left, bottom, width, height]
    cbar_label = fields_info[field]["units"]
    if cbar_label == 'degrees_C': cbar_label = '°C'
    kwargs = dict(
        ax=axs, cax=cax, orientation='horizontal', 
        pad=0.05, aspect=60, shrink=0.9, format=ScalarFormatter()
    )
    if use_units: kwargs['label'] = cbar_label
    cbar = fig.colorbar(im, **kwargs)
    # adjust the plot
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=-0.4, hspace=0.5)
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def plot_decadal_timeseries(
    data: dict, 
    models: dict, 
    colors: dict, 
    fields_info: dict, 
    field: str, 
    start_year: int, 
    end_year: int, 
    lats, 
    outfile: str = None
):
    fig, ax = plt.subplots(1,1, figsize=(15,5), sharex=True)
    fig.suptitle(f'Surface {fields_info[field]["long_name"]} time-series starting from ({start_year}-{end_year})', fontweight='bold', fontsize=16)
    ax.set_title(f'LAT [{lats[0]}, {lats[-1]}]', fontweight='bold', fontsize=14)
    # plot the data
    for mname in data.keys():
        if field not in data[mname].data_vars: continue
        x = data[mname].time.data
        y = data[mname][field].values
        kwargs = dict()
        if mname in ['rea','clm','oras5','om4']: 
            kwargs['linestyle'] = 'dashdot'
            kwargs['linewidth'] = 0.5
        else:
            kwargs['linewidth'] = 1.5
        ax.plot(x, y, label=models[mname], color=colors[mname], zorder=999, **kwargs)
    # set the labels
    unit = fields_info[field]["units"]
    unit = '°C' if unit == 'degrees_C' else unit
    ax.set_ylabel(f'{fields_info[field]["long_name"]} [{unit}]', fontdict={'weight':'bold', 'size':12})
    ax.set_xlabel('Time [years]', fontdict={'weight':'bold', 'size':12})
    # enrich the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    legend = ax.legend(loc='upper center', markerscale=8, edgecolor='gray', framealpha=0.5, ncol=4, bbox_to_anchor=(0.5, -0.15))
    for line in legend.get_lines():
        line.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def plot_isotherm_20C(
    Z20s: list[xr.Dataset],
    model_names: list[str],
    model_colors: list[str],
    box: dict,
    sscale: float = 1.0,
    zorders: list[float] = None,
    use_anom: bool = False,
    outfile: str = None,
):
    _, ax = plt.subplots(1,1,figsize=(20,4))
    if len(Z20s) == 2:
        corr = xr.corr(Z20s[0], Z20s[1], dim="time").mean(skipna=True).values
        corr_label = f'[Pearson = {np.around(corr,2)}]'
    else: 
        corr = None
        corr_label = f''
    plt.suptitle(f'Isotherm at 20°C {box["type"]} Box ({box["coords"]}) {corr_label}')
    glob_min, glob_max = None, None
    if zorders is None: zorders = [i for i in range(len(Z20s))]
    for z20, name, color, zorder in zip(Z20s, model_names, model_colors, zorders):
        for i,subname in enumerate(model_names):
            if subname.lower() in ['rea', 'reanalysis'] and 'atl' in name.lower():
                corr = xr.corr(z20, Z20s[i], dim="time").mean(skipna=True).values
                corr_label = f'(r={np.around(corr,2)})'
            elif subname.lower() in ['rea', 'reanalysis'] and 'forecast' in name.lower():
                corr = xr.corr(z20, Z20s[i], dim="time").mean(skipna=True).values
                corr_label = f'(r={np.around(corr,2)})'
                break
            elif subname.lower() in ['rea', 'reanalysis', 'oras5'] and 'orc' in name.lower():
                corr = xr.corr(z20, Z20s[i], dim="time").mean(skipna=True).values
                corr_label = f'(r={np.around(corr,2)})'
            elif subname.lower() in ['om4'] and 'sam' in name.lower():
                corr = xr.corr(z20, Z20s[i], dim="time").mean(skipna=True).values
                corr_label = f'(r={np.around(corr,2)})'
            elif name.lower() in ['rea','reanalysis','oras5','om4']:
                corr_label = f''
        dates = z20.time.data
        mZ20 = z20.mean(dim=['latitude', 'longitude'], skipna=True)
        if use_anom:
            mZ20_clm = mZ20.groupby("time.month").mean("time")
            mZ20_anom = mZ20.groupby("time.month") - mZ20_clm
        else:
            mZ20_anom = mZ20
        sZ20 = z20.std(dim=['latitude', 'longitude'], skipna=True)
        if name.lower() in ['rea', 'reanalysis', 'om4', 'oras5']:
            kwargs = dict(linestyle='dashdot', linewidth=2.0, zorder=zorder)
        else:
            kwargs = dict(linewidth=2.0, zorder=zorder)
        plt.plot(dates, mZ20_anom, label=f'{name} {corr_label}', color=color, **kwargs)
        plt.fill_between(dates, mZ20_anom - sZ20*sscale, mZ20_anom + sZ20*sscale, alpha=0.1, color=color)
        if glob_min is None: glob_min = mZ20_anom.min()
        else: glob_min = np.stack([glob_min, mZ20_anom.min()]).min(axis=0)
        if glob_max is None: glob_max = mZ20_anom.max()
        else: glob_max = np.stack([glob_max, mZ20_anom.max()]).max(axis=0)
    plt.xlabel(f'Time [y]')
    plt.ylabel(f'Isotherm Depth [m]')
    plt.ylim([glob_min - 10, glob_max + 10])
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    legend = ax.legend(loc='upper center', markerscale=10, edgecolor='gray', framealpha=1, ncol=len(model_names)//2+1, bbox_to_anchor=(0.5, -0.15))
    for line in legend.get_lines(): line.set_linewidth(4.0)
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def plot_dmi_timeseries(
        data: dict,
        colors: dict,
        names: dict,
        dmi: str = '',  # dmi name
        threshold: float = 0.5,
        correlation: str = 'pearson',
        dmi_label: bool = True,
        outfile: str = None,
    ):
    try:
        dates = data[list(data.keys())[0]].time.data
        time_dim = "time"
    except:
        dates = data[list(data.keys())[0]].year.data
        time_dim = "year"
    fig, ax = plt.subplots(1,1,figsize=(20,4))
    if dmi is not None:
        if dmi_label:
            plt.title(f'{dmi} (DMI) Timeseries', fontdict={'weight':'bold', 'size':14})
        else:
            plt.title(f'{dmi} Timeseries', fontdict={'weight':'bold', 'size':14})

    # Plot the ONI for each model
    for m in data.keys():
        for i,subm in enumerate(data):
            if str(subm).lower() in ['0'] and str(m).lower() in ['10','20','30','40','50','60']:
                if correlation == 'spearman':
                    m1 = data[m]
                    m2 = data[subm]
                    m1dts = m1.time.values
                    m2dts = m2.time.values
                    intdts = sorted(list(set(m1dts).intersection(set(m2dts))))
                    m1 = m1.sel(time=intdts)
                    m2 = m2.sel(time=intdts)
                    corr = spearmanr(m1, m2).statistic
                elif correlation == 'pearson':
                    corr = xr.corr(data[m], data[subm], dim=time_dim).mean(skipna=True).values
                corr_label = f'(r={np.around(corr,2)})'
                break
            elif subm.lower() in ['rea', 'reanalysis', 'ground truth'] and ('atl' in m.lower() or 'for' in m.lower()):
                if correlation == 'pearson':
                    corr = xr.corr(data[m], data[subm], dim=time_dim).mean(skipna=True).values
                elif correlation == 'spearman':
                    raise NotImplementedError()
                corr_label = f'(r={np.around(corr,2)})'
            elif subm.lower() in ['rea', 'reanalysis', 'oras5'] and 'orc' in m.lower():
                if correlation == 'pearson':
                    corr = xr.corr(data[m], data[subm], dim=time_dim).mean(skipna=True).values
                elif correlation == 'spearman':
                    raise NotImplementedError()
                corr_label = f'(r={np.around(corr,2)})'
            elif subm.lower() in ['om4'] and 'sam' in m.lower():
                if correlation == 'pearson':
                    corr = xr.corr(data[m], data[subm], dim=time_dim).mean(skipna=True).values
                elif correlation == 'spearman':
                    raise NotImplementedError()
                corr_label = f'(r={np.around(corr,2)})'
            elif m.lower() in ['rea','reanalysis','oras5','om4']:
                corr_label = f''
            else:
                corr_label = f''
        kwargs = dict() if m not in ['oras5', 'om4', 'rea', 'reanalysis'] else dict(linestyle='dashdot')
        plt.plot(data[m].time, data[m], color=colors[m], label=f'{names[m]} {corr_label}', **kwargs)

    # Plot lines marking 0.5 and -0.5 threshold
    if threshold is not None:
        if threshold != "std":
            plt.hlines(0, dates[0], dates[-1], color='black', alpha=0.3, zorder=-999, linewidth=0.5)
            plt.hlines(threshold, dates[0], dates[-1], color='black', alpha=0.3, zorder=-999, linewidth=0.5)
            plt.hlines(-threshold, dates[0], dates[-1], color='black', alpha=0.3, zorder=-999, linewidth=0.5)

        fill_alpha=0.2
        for m in data.keys():
            y = data[m]
            if threshold == "std": thr = y.std().values
            else: thr = threshold
            plt.fill_between(data[m].time, y, thr, where=y>=thr, alpha=fill_alpha, color=colors[m])
            plt.fill_between(data[m].time, y, -thr, where=y<=-thr, alpha=fill_alpha, color=colors[m])

    plt.ylabel('Anomaly', fontdict={'weight':'bold', 'size':12})
    plt.xlabel('Time [year]', fontdict={'weight':'bold', 'size':12})
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    legend = ax.legend(loc='upper center', markerscale=20, edgecolor='gray', fontsize=15, framealpha=1, ncol=len(data)//2+1, bbox_to_anchor=(0.5, -0.15))
    for line in legend.get_lines():
        line.set_linewidth(4.0)
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def plot_dmi_timeseries_optim(
        data: dict,
        colors: dict,
        names: dict,
        dmi: str = '',  # dmi name
        threshold: float = 0.5,
        correlation: str = 'pearson',
        dmi_label: bool = True,
        outfile: str = None,
    ):
    
    # 1. Standardizzazione della dimensione temporale reale
    first_key = list(data.keys())[0]
    if 'valid_time' in data[first_key].coords or 'time' in data[first_key].dims:
        time_dim = "valid_time"
    else:
        time_dim = "year"

    processed_data = {}
    for k, v in data.items():
        # Se è un dataset da modello (ha valid_time coord e time dim), scambiamo le dimensioni
        if 'valid_time' in v.coords and 'time' in v.dims:
            v = v.swap_dims({'time': 'valid_time'})
        # Se è una reanalysis/ground truth (ha time dim ma non valid_time), rinominiamo in valid_time
        elif 'time' in v.dims and time_dim == "valid_time":
            v = v.rename({'time': 'valid_time'})
        processed_data[k] = v
        
    data = processed_data
    dates = data[first_key][time_dim].data
    
    fig, ax = plt.subplots(1,1,figsize=(20,4))
    if dmi is not None:
        if dmi_label:
            plt.title(f'{dmi} (DMI) Timeseries', fontdict={'weight':'bold', 'size':14})
        else:
            plt.title(f'{dmi} Timeseries', fontdict={'weight':'bold', 'size':14})

    # Plot the ONI for each model
    for m in data.keys():
        for i,subm in enumerate(data):
            if str(subm).lower() in ['0'] and str(m).lower() in ['10','20','30','40','50','60']:
                if correlation == 'spearman':
                    m1 = data[m]
                    m2 = data[subm]
                    m1dts = m1[time_dim].values  # Modificato .time -> [time_dim]
                    m2dts = m2[time_dim].values  # Modificato .time -> [time_dim]
                    intdts = sorted(list(set(m1dts).intersection(set(m2dts))))
                    m1 = m1.sel(**{time_dim: intdts})  # Modificato per selezione dinamica
                    m2 = m2.sel(**{time_dim: intdts})  # Modificato per selezione dinamica
                    corr = spearmanr(m1, m2).statistic
                elif correlation == 'pearson':
                    corr = xr.corr(data[m], data[subm], dim=time_dim).mean(skipna=True).values
                corr_label = f'(r={float(corr):.3f})'
                break
            elif subm.lower() in ['rea', 'reanalysis', 'ground truth'] and ('atl' in m.lower() or 'for' in m.lower()):
                if correlation == 'pearson':
                    corr = xr.corr(data[m], data[subm], dim=time_dim).mean(skipna=True).values
                elif correlation == 'spearman':
                    raise NotImplementedError()
                corr_label = f'(r={float(corr):.3f})'
            elif subm.lower() in ['rea', 'reanalysis', 'oras5'] and 'orc' in m.lower():
                if correlation == 'pearson':
                    corr = xr.corr(data[m], data[subm], dim=time_dim).mean(skipna=True).values
                elif correlation == 'spearman':
                    raise NotImplementedError()
                corr_label = f'(r={float(corr):.3f})'
            elif subm.lower() in ['om4'] and 'sam' in m.lower():
                if correlation == 'pearson':
                    corr = xr.corr(data[m], data[subm], dim=time_dim).mean(skipna=True).values
                elif correlation == 'spearman':
                    raise NotImplementedError()
                corr_label = f'(r={float(corr):.3f})'
            elif m.lower() in ['rea','reanalysis','oras5','om4']:
                corr_label = f''
            else:
                corr_label = f''
        kwargs = dict() if m not in ['oras5', 'om4', 'rea', 'reanalysis'] else dict(linestyle='dashdot')
        
        # Modificato .time -> [time_dim] per plottare sul tempo reale corretto
        plt.plot(data[m][time_dim], data[m], color=colors[m], label=f'{names[m]} {corr_label}', **kwargs)

    # Plot lines marking 0.5 and -0.5 threshold
    if threshold is not None:
        if threshold != "std":
            plt.hlines(0, dates[0], dates[-1], color='black', alpha=0.3, zorder=-999, linewidth=0.5)
            plt.hlines(threshold, dates[0], dates[-1], color='black', alpha=0.3, zorder=-999, linewidth=0.5)
            plt.hlines(-threshold, dates[0], dates[-1], color='black', alpha=0.3, zorder=-999, linewidth=0.5)

        fill_alpha=0.2
        for m in data.keys():
            y = data[m]
            if threshold == "std": thr = y.std().values
            else: thr = threshold
            # Modificato .time -> [time_dim] per il fill corretto
            plt.fill_between(data[m][time_dim], y, thr, where=y>=thr, alpha=fill_alpha, color=colors[m])
            plt.fill_between(data[m][time_dim], y, -thr, where=y<=-thr, alpha=fill_alpha, color=colors[m])

    plt.ylabel('Anomaly', fontdict={'weight':'bold', 'size':12})
    plt.xlabel('Time [year]', fontdict={'weight':'bold', 'size':12})
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    legend = ax.legend(loc='upper center', markerscale=20, edgecolor='gray', fontsize=15, framealpha=1, ncol=len(data)//2+1, bbox_to_anchor=(0.5, -0.15))
    for line in legend.get_lines():
        line.set_linewidth(4.0)
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


def plot_sam_timeseries(
        SAMs: list[xr.Dataset],
        model_names: list[str],
        model_colors: list[str],
        zorders: list[float] = None,
        outfile: str = None,
    ):
    _, ax = plt.subplots(1,1,figsize=(20,4))
    plt.suptitle(f'Southern Annular Mode')
    if zorders is None: zorders = [i for i in range(len(SAMs))]
    i = 0
    for sam, name, color, zorder in zip(SAMs, model_names, model_colors, zorders):
        dates = sam.time.data
        if name.lower() in ['rea', 'reanalysis', 'om4', 'oras5']:
            kwargs = dict(linestyle='dashdot', linewidth=2.0, zorder=zorder)
        else:
            kwargs = dict(linewidth=2.0, zorder=zorder)
        label = f'{name}'
        if i > 1:
            corr = xr.corr(sam, SAMs[1], dim='time')
            label += f' (r={np.around(float(corr.values), 2)})'
        mean_sam = sam.rolling(time=7, center=True).mean()
        std_sam = sam.rolling(time=7, center=True).std()
        plt.plot(dates, mean_sam, label=label, color=color, **kwargs)
        plt.fill_between(dates, mean_sam - std_sam, mean_sam + std_sam, alpha=0.1, color=color)
        i += 1
    plt.xlabel(f'Time')
    plt.ylabel(f'SAM Index')
    plt.legend()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc='upper center', markerscale=10, edgecolor='gray', framealpha=1, ncol=len(model_names)//2, bbox_to_anchor=(0.5, -0.15))
    plt.savefig(outfile, **plot_kwargs) if outfile else plt.show()


cmap, norm, levels = get_sst_bias_colormap()
def plot_map_on_grid(bias, fig, grid, lons, lats, central_longitude, title, projection=None, cmap=cmap, norm=norm, levels=levels, draw_labels=True, use_lat=False, use_lon=False, title_kwargs=dict(fontsize=14)):
    if projection is None:
        projection = ccrs.Robinson(central_longitude=central_longitude)
    ax = fig.add_subplot(grid, projection=projection)
    im = ax.contourf(lons, lats, bias, levels=levels, cmap=cmap, norm=norm, extend='both', transform=ccrs.PlateCarree())
    ax.set_title(title, **title_kwargs)
    # set white background for spatial plots
    ax.set_facecolor("white")
    # plot the coastlines along the image
    ax.coastlines(resolution='110m', linewidth=0.6, edgecolor="black", alpha=1.0)
    # color the land
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor='#B1B2B4', alpha=0.3, zorder=1)
    # create ticks every 90 deg
    kws = dict()
    if use_lon: kws['xloc'] = mticker.MultipleLocator(base=20.0)
    if use_lat: kws['yloc'] = mticker.MultipleLocator(base=20.0)
    gl = ax.gridlines(draw_labels=draw_labels, linestyle="--", linewidth=0.5, color="gray", alpha=0.5, **kws)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}
    return im, ax