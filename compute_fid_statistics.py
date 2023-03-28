import os
import argparse
import torch
import numpy as np
import pickle

from stylegan3.dataset import ImageFolderDataset
from dnnlib.util import open_url
from utils.util import FolderDataset, set_seeds, get_activations


def main(args):
    if not os.path.exists(args.fid_dir):
        os.makedirs(args.fid_dir)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.path.endswith('zip'):
        dataset = ImageFolderDataset(args.path)
    elif os.path.isdir(args.path):
        dataset = FolderDataset(args.path)
    else:
        raise NotImplementedError
    queue = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, pin_memory=True, num_workers=1)

    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        model = pickle.load(f).to(device)
        model.eval()

    act = get_activations(queue, model, device=device,
                          max_samples=args.max_samples)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    file_path = os.path.join(args.fid_dir, args.file)
    np.savez(file_path, mu=mu, sigma=sigma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--fid_dir', type=str, default='assets/stats/')
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    set_seeds(0, 0)

    main(args)
