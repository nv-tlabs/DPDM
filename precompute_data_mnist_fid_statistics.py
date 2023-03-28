import os
import argparse
import torch
import torchvision
import numpy as np
import pickle

from dnnlib.util import open_url
from utils.util import set_seeds, get_activations


class MNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        img = self.data[idx]
        return img


class FashionMNISTDataset(torchvision.datasets.FashionMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        img = self.data[idx]
        return img


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.is_fmnist:
        dataset = FashionMNISTDataset(
            root='toy_data/', train=not args.test, download=True)
        if args.test:
            file_path = os.path.join(args.fid_dir, 'fmnist_test.npz')
        else:
            file_path = os.path.join(args.fid_dir, 'fmnist_train.npz')
    else:
        dataset = MNISTDataset(
            root='toy_data/', train=not args.test, download=True)
        if args.test:
            file_path = os.path.join(args.fid_dir, 'mnist_test.npz')
        else:
            file_path = os.path.join(args.fid_dir, 'mnist_train.npz')
    queue = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size)

    with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
        model = pickle.load(f).to(device)

    act = get_activations(queue, model, device=device,
                          max_samples=len(queue.dataset))
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    np.savez(file_path, mu=mu, sigma=sigma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size per GPU')
    parser.add_argument('--fid_dir', type=str, default='assets/stats/',
                        help='A dir to store fid related files')
    parser.add_argument('--is_fmnist', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    set_seeds(0, 0)

    main(args)
