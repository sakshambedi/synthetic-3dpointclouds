import argparse
import os

import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import Compose
from tqdm.auto import tqdm

from evaluation import EMD_CD
from models.autoencoder import *
from utils.data import *
from utils.dataaugmentation import RandomJitter, RandomRotation
from utils.dataset import *
from utils.misc import *
from utils.transform import *


parser = argparse.ArgumentParser(description="Train a diffusion-based 3D point-cloud model")
# Model arguments
parser.add_argument(
    "--latent_dim",
    type=int,
    default=256,
    help="Dimension of the shape latent vector (bottleneck size) used to condition the reverse diffusion chain."
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=200,
    help="Total number of diffusion timesteps (T) for the reverse denoising process."
)
parser.add_argument(
    "--beta_1",
    type=float,
    default=1e-4,
    help="Initial variance β₁ in the forward diffusion schedule (controls noise added at step 1)."
)
parser.add_argument(
    "--beta_T",
    type=float,
    default=0.05,
    help="Final variance β_T in the forward diffusion schedule (noise level at last timestep)."
)
parser.add_argument(
    "--sched_mode",
    type=str,
    default="linear",
    help="Interpolation mode for β schedule (e.g. 'linear', 'cosine') from β₁ to β_T."
)
parser.add_argument(
    "--flexibility",
    type=float,
    default=0.0,
    help="Schedule flexibility scalar to stretch or compress the spacing of β values across timesteps."
)
parser.add_argument(
    "--residual",
    type=eval,
    default=True,
    choices=[True, False],
    help="If True, the model predicts residual noise at each step; otherwise predicts full denoised point positions."
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to a checkpoint to resume training (loads model + optimizer state)."
)

# Datasets and loaders
parser.add_argument(
    "--dataset_path",
    type=str,
    default="./data/dataset",
    help="Path to the dataset file containing point‐cloud data."
)
parser.add_argument(
    "--scale_mode",
    type=str,
    default="shape_unit",
    help="Normalization mode for raw point clouds (e.g. 'shape_unit' to fit unit sphere)."
)
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=128,
    help="Number of point clouds per training batch."
)
parser.add_argument(
    "--val_batch_size",
    type=int,
    default=32,
    help="Number of point clouds per validation batch."
)
parser.add_argument(
    "--rotate",
    type=eval,
    default=False,
    choices=[True, False],
    help="Whether to apply random 3D rotations as data augmentation."
)

# Optimizer and scheduler
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Initial learning rate for the optimizer (e.g. Adam)."
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0,
    help="L2 regularization (weight decay) coefficient."
)
parser.add_argument(
    "--max_grad_norm",
    type=float,
    default=10,
    help="Maximum gradient norm for clipping to prevent exploding gradients."
)
parser.add_argument(
    "--end_lr",
    type=float,
    default=1e-4,
    help="Final learning rate after scheduler annealing."
)
parser.add_argument(
    "--sched_start_epoch",
    type=int,
    default=150 * 1000,
    help="Iteration to start decaying the learning rate from `--lr` to `--end_lr`."
)
parser.add_argument(
    "--sched_end_epoch",
    type=int,
    default=300 * 1000,
    help="Iteration where learning rate reaches `--end_lr`."
)

# Training
parser.add_argument(
    "--seed",
    type=int,
    default=2020,
    help="Random seed for reproducibility (data shuffling, weight init)."
)
parser.add_argument(
    "--logging",
    type=eval,
    default=True,
    choices=[True, False],
    help="Enable/disable console logging of training metrics."
)
parser.add_argument(
    "--log_root",
    type=str,
    default="./logs_ae",
    help="Directory where TensorBoard logs and checkpoints are saved."
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Compute device to use ('cuda' or 'cpu')."
)
parser.add_argument(
    "--max_iters",
    type=int,
    default=1000,
    help="Total number of training iterations (mini‐batches)."
)
parser.add_argument(
    "--val_freq",
    type=float,
    default=100,
    help="Run a validation pass every N iterations."
)
parser.add_argument(
    "--tag",
    type=str,
    default=None,
    help="Optional run tag appended to `log_root` for experiment organization."
)
parser.add_argument(
    "--num_val_batches",
    type=int,
    default=20,
    help="Number of batches to sample during each validation."
)
parser.add_argument(
    "--num_inspect_batches",
    type=int,
    default=20,
    help="Number of batches for inspection/visualization after training."
)
parser.add_argument(
    "--num_inspect_pointclouds",
    type=int,
    default=20,
    help="Number of point clouds to visualize per inspection batch."
)

args = parser.parse_args()

seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(
        args.log_root,
        prefix="AE_",
        postfix="_" + args.tag if args.tag is not None else "",
    )
    logger = get_logger("train", log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger("train", None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
transform = Compose([RandomJitter()])
if args.rotate:
    transform = Compose([RandomRotation(), RandomJitter()])
logger.info("Transform: %s" % repr(transform))
logger.info("Loading datasets...")

train_dset = Wheat3dPointCloud(args.dataset_path, transform=transform, n_points=2048)
val_dset = Wheat3dPointCloud(args.dataset_path, transform=transform, n_points=2048)
train_iter = get_data_iterator(
    DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=0,
    )
)
val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)


# Model
logger.info("Building model...")
if args.resume is not None:
    logger.info("Resuming from checkpoint...")
    ckpt = torch.load(args.resume, weights_only=False)
    model = AutoEncoder(ckpt["args"]).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
else:
    model = AutoEncoder(args).to(args.device)
logger.info(repr(model))



optimizer = torch.optim.Adam(
    model.diffusion.parameters(), lr=args.lr, weight_decay=args.weight_decay
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters)


# Train, validate
def train(it):
    # Load data
    batch = next(train_iter)
    x = batch["pointcloud"].to(args.device)


    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss(x)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    
    loss.item()
    if it % 1000 == 0 or it == 0:
        logger.info(
            "[Train] Iter %04d | Loss %.6f | Grad %.4f " % (it, loss.item(), orig_grad_norm)
        )
    writer.add_scalar("train/loss", loss, it)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it)
    writer.add_scalar("train/grad_norm", orig_grad_norm, it)
    writer.flush()
    


def validate_loss(it):
    all_refs = []
    all_recons = []
    for i, batch in enumerate(tqdm(val_loader, desc="Validate")):
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        ref = batch["pointcloud"].to(args.device)
        shift = batch["shift"].to(args.device)
        scale = batch["scale"].to(args.device)
        with torch.no_grad():
            model.eval()
            code = model.encode(ref)
            recons = model.decode(code, ref.size(1), flexibility=args.flexibility)
        all_refs.append(ref * scale + shift)
        all_recons.append(recons * scale + shift)

    all_refs = torch.cat(all_refs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    metrics = EMD_CD(all_recons, all_refs, batch_size=args.val_batch_size)
    cd, emd = metrics["MMD-CD"].item(), metrics["MMD-EMD"].item()

    logger.info("[Val] Iter %04d | CD %.6f | EMD %.6f  " % (it, cd, emd))
    writer.add_scalar("val/cd", cd, it)
    writer.add_scalar("val/emd", emd, it)
    writer.flush()

    return cd


def validate_inspect(it):
    sum_n = 0
    sum_chamfer = 0
    for batch in tqdm(val_loader, desc="Inspect"):
        x = batch["pointcloud"].to(args.device)
        model.eval()
        code = model.encode(x)
        recons = model.decode(code, x.size(1), flexibility=args.flexibility).detach()

        sum_n += x.size(0)
        # if i >= args.num_inspect_batches:
        #     break  # Inspect only 5 batch

    writer.add_mesh(
        "val/pointcloud", recons[: args.num_inspect_pointclouds], global_step=it
    )
    writer.flush()


# Main loop
logger.info("Start training...")

for param in model.parameters():
    param.requires_grad = False
    

# Unfreeze only the decoder parameters
for param in model.diffusion.parameters():
    param.requires_grad = True


try:
    it = 1
    while it <= args.max_iters:
        
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with torch.no_grad():
                cd_loss = validate_loss(it)
                validate_inspect(it)
            opt_states = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info("Terminating...")
