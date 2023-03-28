import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
import PIL
from torchvision.utils import make_grid
from scipy import linalg
from pathlib import Path

from dataset_tool import is_image_ext


def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size


def set_seeds(rank, seed):
    random.seed(rank + seed)
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        raise ValueError('Directory already exists.')


def add_dimensions(x, n_additional_dims):
    for _ in range(n_additional_dims):
        x = x.unsqueeze(-1)

    return x


def save_checkpoint(ckpt_path, state):
    saved_state = {'model': state['model'].state_dict(),
                   'ema': state['ema'].state_dict(),
                   'optimizer': state['optimizer'].state_dict(),
                   'step': state['step']}
    torch.save(saved_state, ckpt_path)


def save_img(x, filename, figsize=None):
    figsize = figsize if figsize is not None else (6, 6)

    nrow = int(np.sqrt(x.shape[0]))
    image_grid = make_grid(x, nrow)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).cpu())
    plt.savefig(filename, pad_inches=0., bbox_inches='tight')
    plt.close()


def sample_random_image_batch(sampling_shape, sampler, path, device, n_classes=None, name='sample'):
    make_dir(path)

    x = torch.randn(sampling_shape, device=device)
    if n_classes is not None:
        y = torch.randint(n_classes, size=(
            sampling_shape[0],), dtype=torch.int32, device=device)

    x = sampler(x, y)
    x = x / 2. + .5

    save_img(x, os.path.join(path, name + '.png'))
    np.save(os.path.join(path, name), x.cpu())


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    m = np.square(mu1 - mu2).sum()
    s, _ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    fd = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
    return fd


def compute_fid(n_samples, n_gpus, sampling_shape, sampler, inception_model, stats_paths, device, n_classes=None):
    num_samples_per_gpu = int(np.ceil(n_samples / n_gpus))

    def generator(num_samples):
        num_sampling_rounds = int(
            np.ceil(num_samples / sampling_shape[0]))
        for _ in range(num_sampling_rounds):
            x = torch.randn(sampling_shape, device=device)

            if n_classes is not None:
                y = torch.randint(n_classes, size=(
                    sampling_shape[0],), dtype=torch.int32, device=device)
                x = sampler(x, y=y)

            else:
                x = sampler(x)

            x = (x / 2. + .5).clip(0., 1.)
            x = (x * 255.).to(torch.uint8)
            yield x

    act = get_activations(generator(num_samples_per_gpu),
                          inception_model, device=device, max_samples=n_samples)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    m = torch.from_numpy(mu).cuda()
    s = torch.from_numpy(sigma).cuda()
    average_tensor(m)
    average_tensor(s)

    all_pool_mean = m.cpu().numpy()
    all_pool_sigma = s.cpu().numpy()

    fid = []
    for stats_path in stats_paths:
        stats = np.load(stats_path)
        data_pools_mean = stats['mu']
        data_pools_sigma = stats['sigma']
        fid.append(calculate_frechet_distance(data_pools_mean,
                   data_pools_sigma, all_pool_mean, all_pool_sigma))
    return fid


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        self.img = [str(f) for f in sorted(Path(path).rglob('*'))
                    if is_image_ext(f) and os.path.isfile(f)]
        all_labels_path = os.path.join(path, 'all_labels.pt')
        self.label = torch.load(all_labels_path) if os.path.exists(
            all_labels_path) else None
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.img[idx]
        image = np.array(PIL.Image.open(img_path))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        image = image.transpose(2, 0, 1)

        if self.transform is not None:
            image = self.transform(image)

        if self.label is not None:
            return image, self.label[idx]
        else:
            return image

    def __len__(self):
        return len(self.img)


def get_activations(dl, model, device, max_samples):
    pred_arr = []
    total_processed = 0

    print('Starting to sample.')
    for batch in dl:
        # ignore labels
        if isinstance(batch, list):
            batch = batch[0]

        batch = batch.to(device)
        if batch.shape[1] == 1:  # if image is gray scale
            batch = batch.repeat(1, 3, 1, 1)
        elif len(batch.shape) == 3:  # if image is gray scale
            batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)

        with torch.no_grad():
            pred = model(batch.to(device),
                         return_features=True).unsqueeze(-1).unsqueeze(-1)

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr.append(pred)
        total_processed += pred.shape[0]
        if max_samples is not None and total_processed > max_samples:
            print('Max of %d samples reached.' % max_samples)
            break

    pred_arr = np.concatenate(pred_arr, axis=0)
    if max_samples is not None:
        pred_arr = pred_arr[:max_samples]

    return pred_arr
