import os
import time
import argparse
import torch
from tqdm.auto import tqdm
from torchvision.transforms import Compose
from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.autoencoder import *
from evaluation import EMD_CD
from utils.dataaugmentation import RandomJitter, RandomRotation
from utils.dataaugmentation import RandomJitter, RandomRotation

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/AE_airplane.pt')
parser.add_argument('--categories', type=str_list, default=['wheat_plant'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/dataset')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument(
    "--rotate",
    type=eval,
    default=False,
    choices=[True, False],
    help="Whether to apply random 3D rotations as data augmentation."
)

args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'AE_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt, weights_only=False)
seed_all(ckpt['args'].seed)


# Datasets and loaders
transformer = Compose([RandomJitter()])
if args.rotate:
    transformer = Compose([RandomRotation(), RandomJitter()])
logger.info('Loading datasets...')
test_dset = Wheat3dPointCloud(args.dataset_path, transform=transformer, n_points=2048)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
logger.info('Loading model...')
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

all_ref = []
all_recons = []
for i, batch in enumerate(tqdm(test_loader)):
    ref = batch['pointcloud'].to(args.device)
    shift = batch['shift'].to(args.device)
    scale = batch['scale'].to(args.device)
    model.eval()
    with torch.no_grad():
        code = model.encode(ref)
        recons = model.decode(code, ref.size(1), flexibility=ckpt['args'].flexibility).detach()

    ref = ref * scale + shift
    recons = recons * scale + shift

    all_ref.append(ref.detach().cpu())
    all_recons.append(recons.detach().cpu())

all_ref = torch.cat(all_ref, dim=0)
all_recons = torch.cat(all_recons, dim=0)

logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), all_ref.numpy())
np.save(os.path.join(save_dir, 'out.npy'), all_recons.numpy())
