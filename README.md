# Neptune

Neptune is an end-to-end data-driven framework for global ocean and sea-ice components emulation tailored for S2S timescales, up to 60 days. Neptune combines Convolutional Neural Networks (CNNs) and Spherical Fourier Neural Operators (SFNOs) to effectively capture local features and global cross-scale interactions, thereby obtaining a coherent representation of the ocean state. 
Neptune is forced by prescribed daily atmospheric fields, and emulates ocean state variables, from temperature and salinity, to zonal and meridional currents, from sea surface height to sea ice thickness and concentration, with daily outputs at the ocean surface and through the water column.
The codebase contains a PyTorch training pipeline, custom model layers, data loaders, metrics, and experiment configuration files for pretraining and fine-tuning.

## Project structure

- `src/`: training entry points and configuration files
  - `src/training.py`: main training script
  - `src/training.sh`: example multi-node training command
  - `src/finetuning.sh`: example fine-tuning command
  - `src/config/`: model and training configurations (`neptune-1.toml`, `neptune-025.toml`)
- `library/atmocean/`: reusable project code
  - model definitions
  - datasets and normalization logic
  - losses and metrics
  - callbacks and utilities
- `data/`: expected dataset and mask directories
  - `data/datasets/all_1_deg/`
  - `data/masks/sealand_1_deg/`
  - `data/models/` contains pretrained models and benchmark outputs
- `validation/`: notebooks for reanalysis and physical evaluation
- `inference.ipynb`: notebook for inference-related work

## Main technologies used

From the repository code, the project depends on:

- Python
- PyTorch
- Lightning
- MLflow
- NumPy
- pandas
- xarray
- SciPy
- matplotlib
- cartopy
- dask
- einops
- toml
- munch
- tqdm
- requests
- pyperlin
- timm
- torch-harmonics
- torchcfm

The code also expects CUDA-capable hardware for the training workflow described in `src/training.py` and the config files under `src/config/`.

## Minimum environment requirements

### Hardware

- NVIDIA GPU with CUDA support is strongly expected.
- The training script is configured for multi-node / multi-GPU training, but a single GPU setup is also possible if you adapt the launcher arguments.
- Recommended: at least one recent NVIDIA GPU with sufficient VRAM for the configured model sizes.

### Operating system

- Linux is the most likely supported environment.
- macOS is not the intended runtime for this project because the training setup assumes CUDA and distributed training.
- Windows is not a good fit for this repository without substantial environment changes.

### Python

The repository is a Python project and the code imports modules that are typically used with modern Python 3.10+ environments.

Recommended baseline:

- Python 3.10 or 3.11
- pip or conda

### CUDA / PyTorch

The exact CUDA version depends on your GPU and driver. You should install a CUDA-enabled PyTorch build that matches your system.

In practice, the minimum requirement is:

- an NVIDIA driver compatible with your CUDA toolkit
- CUDA-enabled PyTorch build
- a working `torch.cuda.is_available()` check

If you are on Linux with a recent NVIDIA card, the usual route is:

1. Install NVIDIA drivers.
2. Install CUDA toolkit matching your PyTorch version.
3. Install the correct PyTorch build from the official PyTorch website.

This repository is not self-contained with a pinned `requirements.txt`, so the exact PyTorch package version should be selected based on your GPU and OS.

## Recommended setup using conda

If you use conda, the standard flow is:

```bash
conda create -n neptune python=3.11
conda activate neptune
```

Then install a CUDA-compatible PyTorch version. Example pattern (you should adapt to your installed CUDA and platform):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

You may need to replace `cu121` with a version supported by your hardware and driver.

## Python dependencies to install

Install the project dependencies with pip after creating the environment:

```bash
pip install numpy pandas xarray scipy matplotlib dask tqdm requests einops toml munch mlflow lightning cartopy pyperlin timm torch-harmonics
```

Additional package that appears to be imported by the model code:

```bash
pip install torchcfm
```

Important: some packages may need to be installed from GitHub or a specific package index depending on your platform. The code references `torch_harmonics`, `torchcfm`, `pyperlin`, and `timm`, and these are not always available in the same way across all environments.

## Project configuration notes

The repository uses TOML configuration files in `src/config/`.

You will likely need to review and adjust:

- `src/config/neptune-1.toml`
- `src/config/neptune-025.toml`

These files define:

- run name and experiment class
- dataset paths
- model architecture parameters
- training years
- optimizer and scheduler settings
- logger configuration
- CUDA precision settings

Make sure that the paths match your local filesystem structure.

## Data requirements

The repository expects a dataset directory structure similar to:

```text
data/
  datasets/
    all_1_deg/
    all_025_deg/
  masks/
    sealand_1_deg/
  models/
    neptune-025/
    neptune-1/
```

The code references specific atmospheric and oceanic variables such as:

- temperature
- wind components
- mean sea level pressure
- specific ocean variables like `thetao_oras`, `so_oras`, `uo_oras`, `vo_oras`, etc.

### TODO: external dataset acquisition

I could not verify the exact download source for the underlying ERA5 / ORAS / reconstruction data from the repository alone, because no public dataset URL or download script is included here.

You need to perform the following manually:

1. Download the atmospheric and oceanic dataset files required by the model.
2. Put them under the expected directories in `data/datasets/`.
3. Ensure the file names and coordinate layout match the assumptions in the training code.
4. Download or generate the mask files under `data/masks/` if they are not already present.
5. Ensure that the model weights under `data/models/` are available if you want to reproduce or continue training from a checkpoint.

## Model and checkpoint expectations

The repository contains model directories like:

- `data/models/neptune-1/`
- `data/models/neptune-025/`

These likely contain a trained `*.pt` model and benchmark artifacts. If you want to run inference or continue training from a checkpoint, you need to verify the naming and path expectations in the code and ensure the files are present.

## Training launch

The repository includes shell scripts with the expected command pattern.

### Single-node example

```bash
python src/training.py \
  --config src/config/neptune-1.toml \
  --num_nodes 1 \
  --devices 1 \
  --ocean default \
  --atmos era5
```

### Multi-node example

The script `src/training.sh` shows the intended multi-node launch pattern, but it still contains placeholders such as:

- `NUM_PROCS`
- `NUM_NODES`
- `NUM_DEVICES_PER_NODE`

You must replace those values with your actual cluster configuration.

## MLflow and logging setup

The training pipeline initializes MLflow and logs hyperparameters and artifacts.

Potential requirements:

- MLflow installed in the environment
- local tracking directory or MLflow server configured correctly
- experiment names matching your local setup

The code also includes optional Discord logging support via `DiscordLogging` and a webhook field in the TOML config files:

```toml
[logger]
    discord_webhook = ""
    slack_webhook = ""
```

You may need to fill these values if you want notifications enabled.

## Manual actions you should complete yourself

Because this repository does not include every external dependency or dataset source in a fully reproducible way, you should complete the following steps manually:

- Install CUDA and the matching PyTorch build for your machine.
- Confirm the exact Python environment version used in your lab or cluster.
- Download the required atmospheric and oceanic datasets.
- Place the training data under the expected folders in `data/datasets/`.
- Verify the mask files in `data/masks/` are correct.
- Ensure `experiments/` exists and is writable.
- Configure MLflow tracking if you want experiment logging.
- Set the Discord / Slack webhooks if desired.
- Validate the configuration files with your machine-specific paths and training resources.

## TODO / missing repository details

The following items should be added or verified by the project maintainer or the user before using the repository in a production environment:

- exact public dataset source and license
- exact preprocessing / normalization procedure
- reproducible installation script (`requirements.txt` or `environment.yml`)
- explicit instructions for model checkpoint download
- exact training hardware recommendations and memory budget
- cluster launch configuration examples for SLURM / MPI / DDP setups
- precise link to any benchmark or validation outputs

## Quick start checklist

1. Create a Python environment.
2. Install CUDA-compatible PyTorch.
3. Install the Python dependencies listed above.
4. Ensure `data/` contains the required datasets and masks.
5. Check the config files in `src/config/`.
6. Launch training with `python src/training.py ...` or the shell wrapper scripts.
7. Monitor MLflow logs and experiment outputs.

## Notes

This README was written from the repository itself and reflects the code and configuration that are visible in the project. Some external details (for example exact dataset download URLs, licensing details, and some environment-specific installation steps) cannot be fully determined without additional project documentation or external artifacts.

If you want, I can also create a more opinionated version of this README tailored to:

- a local single-GPU setup,
- a multi-GPU Linux cluster setup,
- or a more concise project landing page for GitHub.