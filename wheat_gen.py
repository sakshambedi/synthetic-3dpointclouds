import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torchvision.transforms import Compose

from utils.dataset import Wheat3dPointCloud
from utils.misc import get_logger, seed_all
from models.autoencoder import AutoEncoder
from utils.dataaugmentation import RandomJitter, RandomRotation

# 1. add num_images argument
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt',         type=str, default='./pretrained/AE_airplane.pt',
                    help='Path to checkpoint file')
parser.add_argument('--categories', type=list, default=['wheat_plant'])
# parser.add_argument('--categories',   type=str_list, default=['wheat_plant'],
                    # help='Dataset categories')
parser.add_argument('--save_dir',     type=str, default='./results',
                    help='Directory to write outputs')
parser.add_argument('--device',       type=str, default='cuda',
                    help='Compute device')
parser.add_argument('--dataset_path', type=str, default='./data/dataset',
                    help='Root path of the dataset')
parser.add_argument('--batch_size',   type=int, default=128,
                    help='Batch size for data loading')
parser.add_argument('--rotate',       type=eval, default=False, choices=[True, False],
                    help='Apply random 3D rotations (data augmentation)')
parser.add_argument('--num_images',   type=int, default=1,help='Number of reconstructed point-cloud images to save')
parser.add_argument('--num_points',   type=int, default=2048,help='Number of  point-cloud in an image ')
args = parser.parse_args()

# 2. unify save_dir creation, avoid race conditions
timestamp = int(time.time())
subdir = f"AE_{'_'.join(args.categories)}_{timestamp}"
save_dir = os.path.join(args.save_dir, subdir)
os.makedirs(save_dir, exist_ok=True)

# 3. set up logger
logger = get_logger('test', save_dir)
for key, val in vars(args).items():
    logger.info(f"[ARGS {key}] {val!r}")


device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
ckpt   = torch.load(args.ckpt,weights_only=False, map_location=device)
seed_all(ckpt['args'].seed)

# 5. prepare data transforms
transforms = [RandomJitter()]
if args.rotate:
    transforms.insert(0, RandomRotation())
transformer = Compose(transforms)

logger.info('Loading dataset...')
test_dset   = Wheat3dPointCloud(args.dataset_path,
                                transform=transformer,
                                n_points=args.num_points)
test_loader = DataLoader(test_dset,
                         batch_size=args.batch_size,
                         num_workers=0,
                         shuffle=False)

# 6. load model once, set to eval mode
logger.info('Loading model...')
model = AutoEncoder(ckpt['args']).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# 7. collect all reconstructions
all_recons = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Reconstructing'):
        pcs   = batch['pointcloud'].to(device)
        code  = model.encode(pcs)
        recons= model.decode(code,
                             pcs.size(1),
                             flexibility=ckpt['args'].flexibility)
        # undo normalization
        recons = recons * batch['scale'].to(device) + batch['shift'].to(device)
        all_recons.append(recons.cpu())

all_recons = torch.cat(all_recons, dim=0).numpy()

# 8. save raw arrays
np.save(os.path.join(save_dir, 'out.npy'), all_recons)

# 9. helper to save 3D scatterplots
def save_point_cloud_img(pc: np.ndarray, filepath: str):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    ax.axis('off')
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

# 10. output the requested number of images
num = min(args.num_images, all_recons.shape[0])
for idx in range(num):
    fname = os.path.join(save_dir, f"recon_{idx:03d}.png")
    save_point_cloud_img(all_recons[idx], fname)
    logger.info(f"Saved image {idx+1}/{num}: {fname}")
