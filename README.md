# Diffusion-Point-Cloud (PointCNN version)

This project is based on the open source implementation of the paper [**“Diffusion Probabilistic Models for 3D Point Cloud Generation”**](https://arxiv.org/abs/2103.01458), extending its original version and replacing the **backbone** of point cloud feature extraction from **PointNet** to **PointCNN**. This version achieves better generation quality and diversity on several 3D point cloud datasets. The project aims to reduce data collection time from 90 days to minutes by generating high-fidelity 3D shapes via learned latent representations.

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
- **models/autoencoder.py:** Defines AE combining encoder and decoder
- **models/diffusion.py:** Forward and reverse diffusion processes
- **models/encoders/pointnet.py:** PointNet encoder
- **models/encoders/pointcnn.py:** Optional PointCNN encoder
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

## References

- [Diffusion Probabilistic Models for 3D Point Cloud Generation](https://arxiv.org/abs/2103.01458) Shitong Luo, Wei Hu

## Acknowledgements

- Thanks to the original open source project author for providing the basic framework and reference implementation.
- Thank you to Christopher Henry , Sajjad Heydari and Rob Guderian for supporting us for the project and providing us with the resources to complete this task.
