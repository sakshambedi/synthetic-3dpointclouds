# Diffusion Point Cloud Autoencoder for Wheat3D

This project is based on [`luost26/diffusion-point-cloud`](https://github.com/luost26/diffusion-point-cloud), the official implementation by Shitong Luo and Wei Hu of the paper [**“Diffusion Probabilistic Models for 3D Point Cloud Generation”**](https://arxiv.org/abs/2103.01458) (CVPR 2021), released under the MIT licence (see [`LICENSE`](./LICENSE)). The model code is theirs; this repository is adapted to the [Wheat3D](https://huggingface.co/datasets/sakshambedi/wheat3d_pointcloud) dataset of wheat-plant point clouds. It keeps the original **PointNet** encoder and fine-tunes the diffusion decoder on Wheat3D. The project aims to reduce data collection time from 90 days to minutes by generating high-fidelity 3D shapes via learned latent representations.

**Input:** Normalized 3D point clouds (e.g., 2048 points per object)  
**Output:** Reconstructed or synthetic 3D point clouds from latent encodings

## Prerequisites

| Package       | Installed Version       | Purpose                                |
|---------------|--------------------------|----------------------------------------|
| `Python`       | >=3.10                    | Recommended python version           |
| `PyTorch`       | 2.6.0                    | Core deep learning framework           |
| `torchvision` | 0.21.0                   | Transformations and dataset utilities |
| `tqdm`        | 4.67.1                   | Loop progress visualization            |
| `tensorboard` | 2.19.0                   | Logging and training metrics display   |

Before running, ensure you've installed all dependencies listed in [`requirements.txt`](./requirements.txt), and you're in an environment with CUDA available for GPU acceleration.

## 📂 Codebase Structure

```txt
├── train_ae.py
├── test_ae.py
├── wheat_gen.py
├── models/
│   ├── autoencoder.py
│   ├── diffusion.py
│   ├── encoders/
│   │   ├── pointnet.py
│   │   └── pointcnn.py
│   └── flow.py
├── utils/
│   ├── dataset.py
│   ├── transform.py
│   ├── data.py
│   └── misc.py
├── Makefile
└── pretrained/           # pretrained model checkpoints (optional)
```

- **train_ae.py:** Train the autoencoder with diffusion decoder
- **test_ae.py:** Reconstruct a dataset split with a checkpoint and report Chamfer distance and EMD
- **wheat_gen.py:** Reconstruct a dataset split and save `.npy` outputs and `.png` renders
- **models/autoencoder.py:** Defines AE combining encoder and decoder
- **models/diffusion.py:** Forward and reverse diffusion processes
- **models/encoders/pointnet.py:** PointNet encoder
- **models/encoders/pointcnn.py:** Experimental PointCNN encoder (not used by the autoencoder)
- **models/flow.py:** Latent flow models (for VAE)
- **utils/dataset.py:** `Wheat3DPointCloud` dataset with normalization and resampling
- **utils/transform.py:** Point cloud augmentations (rotate, jitter, scale)
- **utils/data.py:** Data loading helpers
- **utils/misc.py:** Checkpointing, logging, seeding utilities
- **Makefile:** Automate training, generation, and cleaning targets

## 📦 Dataset

You can download the Wheat3D Point Cloud dataset from Hugging Face in two ways:

1. **Via the `datasets` library**  
   First install the library if you haven’t already:  

   ```bash
   pip install datasets
   ```

   Then in your Python script or notebook:

   ```python
   from datasets import load_dataset

   # this will download and cache the dataset for you
   dataset = load_dataset("sakshambedi/wheat3d_pointcloud")

   # access splits
   train_ds = dataset["train"]
   val_ds   = dataset["validation"]
   test_ds  = dataset["test"]
   ```

   Read more about using huggingface for downloading dataset :[Downloading datasets HuggingFace documentation](https://huggingface.co/docs/hub/en/datasets-downloading)

2. **By cloning the repository with Git-LFS**
   If you prefer to have the raw .npy files locally:

   ```bash
   git lfs install
   git clone https://huggingface.co/datasets/sakshambedi/wheat3d_pointcloud
   ```

   Then point your Wheat3DPointCloud root directory to the cloned folder.
   You can explore the dataset page here: <https://huggingface.co/datasets/sakshambedi/wheat3d_pointcloud>

## 🧠 Model Explanation

### Core Components

1. **Encoder**

   - Implemented in `models/encoders/pointnet.py`
   - Maps point clouds to latent representation `m` (and variance `v`)
   - Uses only deterministic encoding (`m`)

2. **Diffusion Decoder**

   - Implemented in `models/diffusion.py`
   - Reverse denoising process over time steps
   - Learns residual corrections via `PointwiseNet` and variance schedule

3. **Training Pipeline**
   - Defined in `train_ae.py`
   - Loads `.npy` data from `data/dataset` using `Wheat3DPointCloud`
   - Applies augmentations: jitter, rotation, scaling
   - Optimizes MSE between predicted and actual noise

### Diffusion Process

- **Forward Process:** Add noise to input based on variance schedule
- **Reverse Process:** Denoise sequentially to recover original shape
- **Loss:** MSE between predicted and actual noise

## Wheat3DPointCloud DataSet Class

This file defines a custom PyTorch `Dataset` subclass (`Wheat3dPointCloud`) that  

- locates and loads all `.npy` point-cloud files under a given directory  
- splits them into train/validation/test sets according to a configurable ratio  
- resamples each cloud to a fixed number of points  
- normalizes (centers and scales) the geometry (and optional RGB colors)  
- returns ready-to-use tensors (`pointcloud`, `shift`, `scale`) for downstream training  

### Contribution to the AI Model Training Process  

1. **Data preprocessing**  
   - Ensures every example has exactly `n_points` (e.g. 2048) via random down- or up-sampling  
   - Applies centering and scaling so that the network sees inputs in a consistent unit sphere  
   - Optionally normalizes RGB channels to [0, 1] if `colors=True`  
2. **Feature engineering**  
   - Produces tensors with shape `[n_points, 3]` (or `[n_points, 6]` when including color)  
   - Supplies `shift` and `scale` tensors so reconstructions can be un-normalized for evaluation or visualization  
3. **Integration with PyTorch data loaders**  
   - Implements `__len__` and `__getitem__` so you can wrap this in a `DataLoader` for batching, shuffling, multi-worker loading  

### Design Choices Optimized for AI Training  

- **Fixed-size sampling (`_resample`)**  
  Guarantees uniform input dimensionality, simplifying network architecture (no dynamic input handling)  
- **Statistical normalization (`_normalize`)**  
  Uses mean and standard deviation of geometry to center and scale, rather than a fixed scale or reference statistic, to adapt per-sample  
- **Split logic in `__init__`**  
  Allows on-the-fly dataset partitioning without maintaining separate folder structures  
- **Optional transform pipeline**  
  Supports common on-the-fly augmentations (jitter, rotation, etc.) passed via `transform`  
- **Legacy method (`_normalize_old`) preserved**  
  Documents alternative normalization (centroid+max-distance unit sphere) for reproducibility or ablation studies  

### Key Considerations and Dependencies  

- **Input format** – expects raw point clouds saved as NumPy `.npy` arrays of shape `[N, 3]` or `[N, 6]` if colors are included  
- **Directory structure** – all `.npy` files must live under `root_dir`, no subfolder parsing  
- **Hyperparameters to tune**  
  - `n_points` to match model capacity  
  - `split_ratio` to balance train/val/test sizes  
  - choice of normalization vs legacy method for best convergence  

## FineTuning the Model  

This script implements the full training pipeline for a 3D point-cloud autoencoder with diffusion, including  

- parsing hyperparameters and paths via `argparse`  
- setting up logging, checkpoints and TensorBoard summaries  
- constructing train and validation datasets/loaders with augmentations  
- building or resuming the `AutoEncoder` model on the chosen device  
- defining optimizer, scheduler and gradient-clipping rules  
- running the main loop with training, periodic validation, metric computation (Chamfer distance, EMD) and checkpointing  

### AI Model Training Process  

1. **Training orchestration**  
   - Manages iterations up to `max_iters`, invoking `train()` and conditional `validate_loss()` plus `validate_inspect()`  
   - Records losses, learning rate and gradient norms to TensorBoard for monitoring  
2. **Model update and optimization**  
   - Uses Adam optimizer on the diffusion module, with cosine‐annealing LR schedule  
   - Applies gradient clipping (`clip_grad_norm_`) to stabilize training  
   - Freezes encoder parameters initially, unfreezes diffusion decoder only, supporting fine-tuning scenarios  
3. **Validation and evaluation**  
   - Computes reconstruction metrics (MMD-CD, MMD-EMD) via `EMD_CD` on denormalized clouds  
   - Logs mesh snapshots of reconstructions for visual inspection  

### Design Choices  

- **Selective parameter freezing**  
  Focuses optimization exclusively on the decoder module during initial epochs
- **Cosine annealing scheduler**  
  Smooth decay of learning rate over `max_iters`, promoting convergence without manual LR adjustments  
- **Data augmentation pipeline**  
  Combines random rotation and jitter transforms for robust point-cloud modeling  
- **Checkpoint management**  
  Saves best models based on validation Chamfer loss, enabling resume and reproducibility  

### Key Considerations and Dependencies for Model Training

- **Input data** – expects point clouds under `data/dataset`, same format used by `Wheat3dPointCloud` (NumPy `.npy` arrays)  
- **Hardware** – designed for GPU training (`--device cuda` by default), adjust `--device` for CPU if needed  
- **Hyperparameter tuning**  
  - `num_steps`, `beta_1`, `beta_T` for diffusion noise schedule  
  - batch sizes and learning rates to match available memory  
  - validation frequency and number of inspected point clouds for efficient monitoring  
- **Resume logic** – if `--resume` is provided, the script reloads model state, optimizer and scheduler, ensuring seamless continuation  

🛠️ Running via Makefile

You can use the provided Makefile to train the model, generate samples, and clean your working directory. Below are the available commands:

Resume training the autoencoder from a saved checkpoint using specific training hyperparameters.

```bash
make run
```

Train the autoencoder from scratch without resuming from a checkpoint.

```bash
make run_nr
```

Generate and save reconstructed 3D point cloud images using a pretrained model. The generated files include .npy outputs and .png visualizations.

```bash
make sample
```

This command uses:

- The held-out `test` split (pass `--split train` to `wheat_gen.py` for the old behaviour)
- Checkpoint from pretrained/GEN_WHEAT_PLANT.pt
- Saves outputs to gen_samples/AE_wheat_plant_<timestamp>
- Generates 10 reconstructed images (by default)
- Each with 2048 points
- Uses CUDA if available

View all command-line options supported by train_ae.py.

```bash
make help
```

Clean Python bytecode and cached files from utils/, models/, and models/encoders/.

```bash
make clean
```

## ⚙️ Command Line Arguments

Run `python train_ae.py --help` for full options. Key arguments:

| Argument                  | Description                                                                                                |
|---------------------------|------------------------------------------------------------------------------------------------------------|
| `--latent_dim`            | Size of the autoencoder’s bottleneck (latent vector dimension).                                             |
| `--num_steps`             | Number of diffusion timesteps in the reverse (denoising) process.                                         |
| `--beta_1`                | Initial noise rate ($\beta_1$) at the first diffusion step.                                                |
| `--beta_T`                | Final noise rate ($\beta_T$) at the last diffusion step.                                                   |
| `--sched_mode`            | Variance-schedule interpolation mode (e.g., “linear”, “cosine”).                                          |
| `--flexibility`           | Extra scalar to stretch/compress the variance schedule.                                                    |
| `--residual`              | If `True`, decoder predicts residual noise; if `False`, predicts full denoised output.                     |
| `--resume`                | Path to checkpoint file for loading pretrained weights and optimizer state.                               |
| `--dataset_path`          | Root directory of the `.npy` point-cloud dataset.                                                        |
| `--scale_mode`            | Normalization mode for point clouds (e.g., “shape_unit”).                                                |
| `--train_batch_size`      | Number of samples per training batch.                                                                     |
| `--val_batch_size`        | Number of samples per validation batch.                                                                   |
| `--rotate`                | Apply random rotations as data augmentation (`True`/`False`).                                              |
| `--lr`                    | Initial learning rate for the optimizer (e.g., Adam).                                                      |
| `--weight_decay`          | L2 regularization coefficient on model weights.                                                          |
| `--max_grad_norm`         | Maximum gradient norm for gradient clipping.                                                              |
| `--end_lr`                | Final learning rate at end of the learning-rate schedule.                                                |
| `--sched_start_epoch`     | Iteration/epoch index where learning-rate decay begins.                                                   |
| `--sched_end_epoch`       | Iteration/epoch index where learning-rate decay ends (reaches `--end_lr`).                                 |
| `--seed`                  | Random seed for reproducibility (affects shuffling, initialization).                                      |
| `--logging`               | Enable/disable console logging of training metrics (`True`/`False`).                                        |
| `--log_root`              | Directory for TensorBoard logs and checkpoints.                                                           |
| `--device`                | Compute device to use (`"cuda"` or `"cpu"`).                                                               |
| `--max_iters`             | Total number of training iterations (batches).                                                            |
| `--val_freq`              | Frequency (in iterations) to run validation.                                                              |
| `--tag`                   | Optional string appended to run folders for experiment tracking.                                          |
| `--num_val_batches`       | Number of batches to sample during each validation pass.                                                 |
| `--num_inspect_batches`   | Number of batches to run through the inspection/visualization step.                                      |
| `--num_inspect_pointclouds` | Number of point clouds to visualize per batch during inspection.                                        |

## Evaluation and its limits

Read this before quoting any number from this repository.

**Splits.** `Wheat3dPointCloud` lists the `.npy` files, sorts them by name and cuts them 80/10/10: of the 405 clouds, 324 are `train`, 40 are `val` and 41 are `test`.

**What changed.** Every script used to build its dataset without naming a split, so it got the default, `train`. Validation during training (`train_ae.py`), evaluation (`test_ae.py`) and sampling (`wheat_gen.py`) therefore all ran on the 324 training clouds. They now default to held-out data:

| Script | Flag | Default | Old behaviour |
| --- | --- | --- | --- |
| `train_ae.py` | `--val_split` | `val` | `--val_split train` |
| `test_ae.py` | `--split` | `test` | `--split train` |
| `wheat_gen.py` | `--split` | `test` | `--split train` |

`test_ae.py` also logs the Chamfer distance and EMD of its reconstructions again; the upstream script did this, but the call had been dropped here.

**Figures recorded before this change were computed on the training split.** That includes the Chamfer distances in the checkpoint file names and the outputs under `results/` and `gen_samples/`. They measure how well the model reconstructs clouds it was trained on, not how well it generalises.

**The committed checkpoint is worse than the one it started from.** `pretrained/GEN_WHEAT_PLANT.pt` was saved as `ckpt_0.014577_10000.pt`: 10,000 further iterations resumed from `ckpt_0.004541_20000.pt`. The number in each name is the validation Chamfer distance at save time, on the training split in both cases. It rose from 0.004541 to 0.014577, so on the only metric recorded, the fine-tuning run that produced the committed checkpoint made reconstruction worse. The starting checkpoint is not in the repository.

**The committed checkpoint has no clean held-out set.** Until this change the file list was not sorted, so the 324 training clouds were whichever files the training machine's filesystem returned first. That order was not recorded. Sorting makes the split reproducible from now on, but some of today's `val` and `test` clouds were probably in the old training set. A held-out number for `GEN_WHEAT_PLANT.pt` is therefore not a clean generalisation measurement; a clean one needs a model retrained with the sorted split.

**The sorted split is a split by variety.** File names start with the wheat variety, so sorting puts Alsen, Brandon, Chara, Gladius and most of Kukri in `train` (324 clouds), the rest of Kukri plus Teal in `val` (13 + 27) and only Teal in `test` (41). No pot appears in more than one split, so segments of one scan (`..._cluster_N.npy`, `..._combined.npy`) stay together. The consequence is that `test` measures reconstruction of a variety that a model trained on this split has never seen, which is a harder and different question from reconstructing unseen plants of familiar varieties.

**Evaluation adds jitter.** The scripts apply `RandomJitter` (Gaussian noise, sigma 0.01) to the reference clouds at evaluation time as well as during training, so the references are noisy copies of the data.

### Measured on the sorted splits

One run each of the committed checkpoint on CPU (15 November 2026), seed 2020 from the checkpoint, jitter on, no rotation:

```bash
python test_ae.py --ckpt pretrained/GEN_WHEAT_PLANT.pt --device cpu --batch_size 64 --split test
python test_ae.py --ckpt pretrained/GEN_WHEAT_PLANT.pt --device cpu --batch_size 64 --split val
```

| Split | Clouds | Chamfer distance | EMD (Sinkhorn approximation) |
| --- | --- | --- | --- |
| `val` | 40 | 0.015404 | 0.010607 |
| `test` | 41 | 0.018096 | 0.013181 |

These are not clean held-out results for this checkpoint, for the reason given above: the split it was trained with is unknown. They are also not directly comparable with the 0.014577 in the checkpoint name, which was computed during training on the training split with random rotation enabled. The training split was not re-evaluated here (324 clouds take roughly half an hour on CPU); run `--split train` to get the like-for-like figure.

## References

- [Diffusion Probabilistic Models for 3D Point Cloud Generation](https://arxiv.org/abs/2103.01458), Shitong Luo and Wei Hu, CVPR 2021
- Upstream code: [luost26/diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud) (MIT licence, Copyright (c) 2021 Shitong Luo)

## Acknowledgements

- The model, diffusion and training code come from Shitong Luo and Wei Hu's [diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud) (MIT licence). The Wheat3D dataset class, the augmentations, the fine-tuning setup and the sampling script were added here.
- The evaluation metrics in `evaluation/` come from [PointFlow](https://github.com/stevenygd/PointFlow/tree/master/metrics), as the file header states.
- Thank you to Christopher Henry , Sajjad Heydari and Rob Guderian for supporting us for the project and providing us with the resources to complete this task.
