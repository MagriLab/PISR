import argparse
import enum
import sys
from pathlib import Path
from typing import NamedTuple

import einops
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import torch
import torch.nn as nn
import yaml
from absl import app, flags
from ml_collections import config_dict, config_flags
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..pisr.data import generate_dataloader, load_data, train_validation_split
from ..pisr.experimental import define_path as pisr_flags
from ..pisr.layers import TimeDistributedWrapper
from ..pisr.loss import KolmogorovLoss
from ..pisr.model import SRCNN
from ..pisr.sampling import get_low_res_grid
from ..pisr.utils.config import ExperimentConfig


FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file('config')

_EXPERIMENT_PATH = pisr_flags.DEFINE_path(
    'experiment_path',
    None,
    'Path to experiment run.'
)

_DATA_PATH = pisr_flags.DEFINE_path(
    'data_path',
    None,
    'Path to data.'
)

_PLOT_PATH = pisr_flags.DEFINE_path(
    'plot_path',
    None,
    'Path to save plots to.'
)

_GPU = flags.DEFINE_integer(
    'run_gpu',
    0,
    'Which GPU to run on.'
)

_MEMORY_FRACTION = flags.DEFINE_float(
    'memory_fraction',
    None,
    'Memory fraction of GPU to use.'
)

flags.mark_flags_as_required(['config', 'experiment_path', 'data_path', 'plot_path'])


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True} if DEVICE == 'cuda' else {}


TimeDistributedUpsample = TimeDistributedWrapper(nn.Upsample)


class eUpsampling(enum.Enum):
    NEAREST = 'nearest'
    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'


class UpsampledResults(NamedTuple):

    hi_true: torch.Tensor | np.ndarray
    lo_true: torch.Tensor | np.ndarray

    hi_nearest: torch.Tensor | np.ndarray
    hi_bilinear: torch.Tensor | np.ndarray
    hi_bicubic: torch.Tensor | np.ndarray

    hi_predicted: torch.Tensor | np.ndarray


def get_upsampled(model: nn.Module, hi_res: torch.Tensor) -> UpsampledResults:

    """Upsample the generated low-resolution field.

    Parameters
    ==========
    model: nn.Module
        Model to use for predicted high-resolution field.
    hi_res: torch.Tensor
        Original data - this is then downsampled in the function.

    Returns
    =======
    results: UpsampledResults
        Super-resolved fields for each of the methods.
    """

    factor = FLAGS.config.experiment.sr_factor
    batch_size = FLAGS.config.training.batch_size

    def upsample_factory(shape: tuple[int, int], upsample: eUpsampling) -> nn.Module:
        return TimeDistributedUpsample(shape, mode=upsample.value)

    lo_res = get_low_res_grid(hi_res, factor=factor)

    shape = (hi_res.shape[-1], hi_res.shape[-1])
    hi_nearest = upsample_factory(shape, eUpsampling.NEAREST)(lo_res)
    hi_bilinear = upsample_factory(shape, eUpsampling.BILINEAR)(lo_res)
    hi_bicubic = upsample_factory(shape, eUpsampling.BICUBIC)(lo_res)

    _dataloader_kwargs = dict(shuffle=False, drop_last=False)
    dataloader = generate_dataloader(hi_res, batch_size, DEVICE_KWARGS, _dataloader_kwargs)

    hi_predicted = torch.zeros_like(hi_res)
    for idx, batched_hi_res in enumerate(dataloader):
        batched_lo_res = get_low_res_grid(batched_hi_res, factor=factor).to(DEVICE)
        hi_predicted[idx * batch_size : idx * batch_size + batched_hi_res.size(0), ...] = model(batched_lo_res).detach().cpu()

    results = UpsampledResults(
        hi_true=hi_res,
        lo_true=lo_res,
        hi_nearest=hi_nearest,
        hi_bilinear=hi_bilinear,
        hi_bicubic=hi_bicubic,
        hi_predicted=hi_predicted
    )

    return results


def get_energy_spectrum(upsampled_results: UpsampledResults, loss_fn: KolmogorovLoss) -> UpsampledResults:

    """Compute the energy spectrum.

    Parameters
    ==========
    upsampled_results: UpsampledResults
        Upsampled fields to compute energy spectrum for.
    loss_fn: KolmogorovLoss
        Loss function which contains solver used to compute spectrum.

    Returns
    =======
    es_results: UpsampledResults
        Energy spectrums for each of the upsampled results.
    """

    def to_fourier(arr: torch.Tensor) -> torch.Tensor:
        return loss_fn.solver.phys_to_fourier(einops.rearrange(arr[:, 0, ...], 'b u i j -> b i j u'))

    _results = {}
    for k, v in upsampled_results._asdict().items():

        if k == 'lo_true':
            _results[k] = None
            continue

        _results[k] = loss_fn.solver.energy_spectrum(to_fourier(v), agg=True).detach().cpu().numpy()

    es_results = UpsampledResults(**_results)

    return es_results


def get_data(data_path: Path) -> torch.Tensor:

    """Load data from .h5 file

    Parameters
    ==========
    data_path: Path
        Data to load from file.

    Returns
    =======
    u_data: torch.Tensor
        Loaded data.
    """

    u_all = load_data(h5_file=data_path, config=FLAGS.config)
    u_data, _ = train_validation_split(u_all, FLAGS.config.data.ntrain, FLAGS.config.data.nvalidation, step=FLAGS.config.data.tau)

    return u_data


def initialise_model(model_path: Path) -> nn.Module:

    """Load trained model

    Parameters
    ==========
    model_path: Path
        Model to load.

    Returns
    =======
    model: nn.Module
        Initialised model.
    """

    # initialise model
    model = SRCNN(FLAGS.config.experiment.nx_lr, FLAGS.config.experiment.sr_factor)

    # load model from file
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # make weights floats and move to device
    model.to(torch.float)
    model.to(DEVICE)

    return model


def generate_plot(upsampled_fields: UpsampledResults, energy_spectrums: UpsampledResults) -> None:

    """Generates the plot in the paper.

    Note: this is somewhat hardcoded, for best results please use experimental setting described in the paper.

    Parameters
    ==========
    upsampled_fields: UpsampledResults
        Super-resolved fields for each of the methods.
    energy_spectrums: UpsampledResults
        Energy spectrums for each of the super-resolved fields.
    """

    def _rc_preamble() -> None:

        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'Times New Roman',
            'font.size' : 10,
        })

        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{nicefrac}\usepackage{bm}')

    _rc_preamble()

    # convert all fields to numpy array
    _res = {}
    for k, v in upsampled_fields._asdict().items():
        _res[k] = v.detach().cpu().numpy()

    upsampled_fields = UpsampledResults(**_res)

    # create figure
    fig, axs = plt.subplots(2, 3, figsize=(5.5, 3.67), facecolor='white')
    axs = axs.flatten()

    NX_HR = int(upsampled_fields.hi_true.shape[-1])
    NX_LR = int(upsampled_fields.lo_true.shape[-1])

    cmap = 'RdBu_r'

    title_fontsize = 8
    fontsize = 6
    tick_fontsize = 4
    legend_fontsize = 4.5

    vmin, vmax = -1.2, 1.2

    random_idx = np.random.choice(np.arange(upsampled_fields.hi_true.shape[0]), 1, replace=False)[0]

    ## plotting the fields #########################################################################################
    im_lo = axs[0].imshow(upsampled_fields.lo_true[random_idx, 0, 0, ...], cmap=cmap, vmin=vmin, vmax=vmax)
    im_hi = axs[3].imshow(upsampled_fields.hi_true[random_idx, 0, 0, ...], cmap=cmap, vmin=vmin, vmax=vmax)

    im_bl = axs[1].imshow(upsampled_fields.hi_bilinear[random_idx, 0, 0, ...], cmap=cmap, vmin=vmin, vmax=vmax)
    im_bc = axs[2].imshow(upsampled_fields.hi_bicubic[random_idx, 0, 0, ...], cmap=cmap, vmin=vmin, vmax=vmax)

    im_pred = axs[4].imshow(upsampled_fields.hi_predicted[random_idx, 0, 0, ...], cmap=cmap, vmin=vmin, vmax=vmax)

    ## plotting the energy spectrum ################################################################################
    x_vals = np.arange(energy_spectrums.hi_true.shape[0])
    ccolors = plt.get_cmap('tab10')(np.arange(5, dtype=int))

    plot_kwargs = dict(lw=0.8)

    ls = (0, (1, 1))

    # nyquist cut-offs for experiment
    nq_lo = 7
    nq_k = 42

    nq_lo_dealiased = round(2/3 * nq_lo)
    nq_k_dealiased = round(2/3 * nq_k)

    axs[5].vlines(nq_lo, 10 ** -15, 10 ** 5, color='k', linestyle='--', linewidth=0.5, alpha=1.0, label=r'Nyquist $f_{\bm{\Omega}}^{n}$')
    axs[5].vlines(nq_k, 10 ** -15, 10 ** 5, color='k', linestyle='--', linewidth=0.5, alpha=1.0)

    axs[5].vlines(nq_lo_dealiased, 10 ** -15, 10 ** 5, color='k', linestyle=':', linewidth=0.5, alpha=1.0, label=r'$\nicefrac{2}{3}f_{\bm{\Omega}}^{n}$')
    axs[5].vlines(nq_k_dealiased, 10 ** -15, 10 ** 5, color='k', linestyle=':', linewidth=0.5, alpha=1.0)

    # high-resolution
    axs[5].plot(x_vals[:(nq_k_dealiased + 1)], energy_spectrums.hi_true[:(nq_k_dealiased + 1)], label=r'$u(\bm{\Omega_{H}}, t)$', color=ccolors[0], **plot_kwargs)
    axs[5].plot(x_vals[nq_k_dealiased:], energy_spectrums.hi_true[nq_k_dealiased:], color=ccolors[0], linestyle=ls, **plot_kwargs)

    # low-resolution
    axs[5].plot(x_vals[:(nq_lo_dealiased + 1)], energy_spectrums.hi_nearest[:(nq_lo_dealiased + 1)], label=r'$u(\bm{\Omega_{L}}, t)$', color=ccolors[1], **plot_kwargs)
    axs[5].plot(x_vals[nq_lo_dealiased:-1], energy_spectrums.hi_nearest[nq_lo_dealiased:-1], color=ccolors[1], linestyle=ls, **plot_kwargs)

    # bi-cubic upsampling
    axs[5].plot(x_vals[:(nq_k_dealiased + 1)], energy_spectrums.hi_bicubic[:(nq_k_dealiased + 1)], label=r'$BC(u(\bm{\Omega_{L}}, t))$', color=ccolors[2], **plot_kwargs)
    axs[5].plot(x_vals[nq_k:], energy_spectrums.hi_bicubic[nq_k:], color=ccolors[2], linestyle=ls, **plot_kwargs)

    # bi-linear upsampling
    axs[5].plot(x_vals[:(nq_k_dealiased + 1)], energy_spectrums.hi_bilinear[:(nq_k_dealiased + 1)], label=r'$BL(u(\bm{\Omega_{L}}, t))$', color=ccolors[3], **plot_kwargs)
    axs[5].plot(x_vals[nq_k_dealiased:], energy_spectrums.hi_bilinear[nq_k_dealiased:], color=ccolors[3], linestyle=ls, **plot_kwargs)

    # PREDICTED
    axs[5].plot(x_vals[:(nq_k_dealiased + 1)], energy_spectrums.hi_predicted[:(nq_k_dealiased + 1)], label=r'$f_{\theta}(u(\bm{\Omega_{L}}, t))$', color=ccolors[4], **plot_kwargs)
    axs[5].plot(x_vals[nq_k_dealiased:], energy_spectrums.hi_predicted[nq_k_dealiased:], color=ccolors[4], linestyle=ls, **plot_kwargs)

    ## setting the axis parameters #################################################################################
    for ax in axs:
        ax.set_box_aspect(1)

    tick_loc_lo = np.linspace(-0.5, NX_LR - 0.5, 4 + 1)
    tick_loc_hi = np.linspace(-0.5, NX_HR - 0.5, 4 + 1)

    # axes for low-resolution image
    for axis in [axs[0].xaxis, axs[0].yaxis]:
        axis.set_ticks(tick_loc_lo)
        axis.set_minor_locator(tck.AutoMinorLocator())
        axis.set_major_formatter(tck.NullFormatter())

    axs[0].set_xlim(-0.5, NX_LR - 0.5)
    axs[0].set_ylim(-0.5, NX_LR - 0.5)

    # axes for high-resolution images
    for i in range(1, 5):

        for axis in [axs[i].xaxis, axs[i].yaxis]:
            axis.set_ticks(tick_loc_hi)
            axis.set_minor_locator(tck.AutoMinorLocator())
            axis.set_major_formatter(tck.NullFormatter())

        axs[i].set_xlim(-0.5, NX_HR - 0.5)
        axs[i].set_ylim(-0.5, NX_HR - 0.5)


    axs[5].tick_params(which='both', labelsize=tick_fontsize)

    # setting titles
    title_pad = 5
    axs[0].set_title(r'$u(\bm{\Omega_{L}}, t)$', fontsize=title_fontsize, pad=title_pad)
    axs[3].set_title(r'$u(\bm{\Omega_{H}}, t)$', fontsize=title_fontsize, pad=title_pad)

    axs[1].set_title(r'$BL(u(\bm{\Omega_{L}}, t))$', fontsize=title_fontsize, pad=title_pad)
    axs[2].set_title(r'$BC(u(\bm{\Omega_{L}}, t))$', fontsize=title_fontsize, pad=title_pad)
    axs[4].set_title(r'$f_{\theta}(u(\bm{\Omega_{L}}, t))$', fontsize=title_fontsize, pad=title_pad)

    axs[5].set_title(r'Energy Spectrum', fontsize=title_fontsize, pad=title_pad)
    leg = axs[5].legend(fontsize=legend_fontsize, facecolor='white', framealpha=1, edgecolor='k')
    leg.get_frame().set_linewidth(0.1)
    leg.get_frame().set_boxstyle('square', pad=0.1)

    # adding text and scaling energy spectrum plot
    axs[5].set_xscale('log')
    axs[5].set_yscale('log')

    axs[5].text(7.3, 10 ** 3.7, r'$f_{\bm{\Omega_{L}}}^{n}$', fontsize=5)
    axs[5].text(43.1, 10 ** 3.7, r'$f_{\bm{\hat{\Omega}_{k}}}^{n}$', fontsize=5)

    axs[5].set_yticks([10 ** i for i in range(-15, 5)])
    axs[5].yaxis.set_major_formatter(tck.NullFormatter())

    ytick_labels = [r'$10^{-15}$'] + 18 * [''] + [r'$10^{4}$']
    axs[5].set_yticklabels(ytick_labels, fontsize=tick_fontsize)

    axs[5].set_xlabel(r'$|\bm{k}|$', fontsize=fontsize, labelpad=3)
    axs[5].set_ylabel(r'$E(|\bm{k}|)$', fontsize=fontsize, labelpad=-12)

    axs[5].yaxis.tick_right()
    axs[5].yaxis.set_label_position('right')

    axs[5].set_xlim(10 ** 0, 10 ** 2)
    axs[5].set_ylim(10 ** -15, 10 ** 5)

    axs[5].grid(which='both', alpha=0.05, color='k')

    ## adding the colourbar ########################################################################################
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%', pad=0.03)
    cb = fig.colorbar(im_bc, cax=cax)

    cax.tick_params(labelsize=tick_fontsize)


    ## trick to align last plot ####################################################################################
    divider = make_axes_locatable(axs[5])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im_bc, cax=cax)
    cax.tick_params(labelsize=tick_fontsize)

    cb.remove()


    ## aligning plots and saving ###################################################################################
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.3)
    fig.savefig(FLAGS.plot_path, dpi=1000, bbox_inches='tight')

    plt.close(fig)


def main(_) -> None:

    if FLAGS.run_gpu is not None and FLAGS.run_gpu >= 0 and FLAGS.run_gpu < torch.cuda.device_count():

        global DEVICE
        global DEVICE_KWARGS

        if not torch.cuda.is_available():
            raise ValueError('Specified CUDA device unavailable.')

        DEVICE = torch.device(f'cuda:{FLAGS.run_gpu}')
        DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True}

    if FLAGS.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(FLAGS.memory_fraction, DEVICE)

    model_path = FLAGS.experiment_path / 'model.pt'

    # get config from global flags
    config = FLAGS.config

    # load data
    u_data = get_data(FLAGS.data_path)

    # initialise loss function and set constraints to false
    loss_fn = KolmogorovLoss(nk=config.simulation.nk, re=config.simulation.re, dt=config.simulation.dt, fwt_lb=config.training.fwt_lb, device=DEVICE)
    loss_fn.constraints = False

    # load trained model
    model = initialise_model(model_path=model_path)

    # upsample fields and generate energy spectrums
    upsampled_fields = get_upsampled(model, u_data)
    energy_spectrums = get_energy_spectrum(upsampled_fields, loss_fn)

    # produce paper plots
    generate_plot(upsampled_fields, energy_spectrums)


if __name__ == '__main__':
    app.run(main)
