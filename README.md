## <p align="center">Differentially Private Diffusion Models</p>
<div align="center">
  <a href="https://timudk.github.io/" target="_blank">Tim&nbsp;Dockhorn</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://orcid.org/0000-0001-6579-6044" target="_blank">Tianshi&nbsp;Cao</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://latentspace.cc/" target="_blank">Arash&nbsp;Vahdat</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://karstenkreis.github.io/" target="_blank">Karsten&nbsp;Kreis</a>
  <br> <br>
  <a href="https://arxiv.org/abs/2210.09929" target="_blank">Paper</a> &emsp;
  <a href="https://nv-tlabs.github.io/DPDM/" target="_blank">Project&nbsp;Page</a> 
</div>
<br><br>

## Requirements

DPDM is built using PyTorch 1.11.0 and CUDA 11.3. Please use the following command to install the requirements:
```shell script
pip install -r requirements.txt 
```

## Pretrained checkpoints

We provide pre-trained checkpoints for all models presented in the paper [here](https://drive.google.com/drive/folders/1TQ48z51j0omIHpDsGg7pkGgbB0ivM17V?usp=share_link). You can sample from the checkpoint using the following command:

```shell script
python main.py --mode eval --workdir <new_directory> --config <config_file> -- model.ckpt=<checkpoint_path>
```

| Dataset | Resolution | Epsilon | Config | Checkpoint | Class-conditional |
|:----------|:----------|:----------|:----------|:----------|:----------|
| MNIST | 28 | 0.2 | `configs/mnist_28/sample_eps_0.2.yaml` | `mnist_v_0.2.pth` | Yes |
| MNIST | 28 | 1.0 | `configs/mnist_28/sample_eps_1.0.yaml` | `mnist_edm_1.0.pth` | Yes |
| MNIST | 28 | 10.0 | `configs/mnist_28/sample_eps_10.0.yaml` | `mnist_edm_1.0.pth` | Yes |
| Fashion-MNIST | 28 | 0.2 | `configs/fmnist_28/sample_eps_0.2.yaml` | `mnist_v_0.2.pth` | Yes |
| Fashion-MNIST | 28 | 1.0 | `configs/fmnist_28/sample_eps_1.0.yaml` | `mnist_edm_1.0.pth` | Yes |
| Fashion-MNIST | 28 | 10.0 | `configs/fmnist_28/sample_eps_10.0.yaml` | `mnist_edm_10.0.pth` | Yes |
| CelebA | 32 | 1.0 | `configs/celeba_32/sample_eps_1.0.yaml` | `celeba_edm_1.0.pth` | No |
| CelebA | 32 | 10.0 | `configs/celeba_32/sample_eps_10.0.yaml` | `celeba_edm_10.0.pth` | No |
| CelebA | 64 | 10.0 | `configs/celeba_64/sample_eps_10.0.yaml` | `celeba64_edm_10.0.pth` | No |
| CIFAR-10 | 32 | 10.0 | `configs/cifar10_32/sample_eps_10.0.yaml` | `cifar10_edm_10.0.pth` | Yes |
| ImageNet | 32 | 10.0 | `configs/imagenet_32/sample_eps_10.0.yaml` | `imagenet_edm_10.0.pth` | Yes |


&nbsp;

By default, the above command generates 16 samples using DDIM with 50 steps on a single GPU. You can modify any entry of the config file directly through the command line, e.g., you may append `sampler.num_steps=20` to the above command to change the number of DDIM steps from 50 (default value) to 20. 

For class-conditional sampling, you can set ``sampler.labels`` to a single integer between 0 and `num_classes - 1` (inclusive) to generate all samples from the same class. By default, the configs sets ``sampler.labels=num_classes`` which performs random sampling.

All class-conditional models are trained for classifier-free guidance by dropping out the label 10% of the time. By default, the guidance strength is set to 0 but you may change it by setting ``sampler.guid_scale``.

To reproduce the results in the paper, you may adjust the sampler settings accordingly. For example, to obtain the best FID value for MNIST, eps=1.0, the sampler setting is (Table 13 & Table 14):

```shell script
python main.py --mode eval --workdir <new_directory> --config configs/mnist_28/sample_eps_1.0.yaml -- model.ckpt=<checkpoint_path> sampler.type=edm sampler.s_churn=100. sampler.s_min=0.05 sampler.s_max=50. sampler.num_steps=1000
```

## Training your own models

### Data preparations

First, create the following two folders:
```shell script
mkdir -p data/raw/
mkdir -p data/processed/
```
Afterwards, run the following commands to download and prepare the data used for training. MNIST and FashionMNIST are downloaded and processed automatically.

<details><summary>CIFAR-10</summary>

```shell script
wget -P data/raw/ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
python dataset_tool.py --source data/raw/cifar-10-python.tar.gz --dest data/processed/cifar10.zip
```
</details>

<details><summary>ImageNet</summary>

First download the [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data), then run the following

```shell script
python dataset_tool.py --source==data/raw/imagenet/ILSVRC/Data/CLS-LOC/train --dest=data/processed/imagenet.zip --resolution=32x32 --transform=center-crop
```

</details>

### FID evaluation

Before training, you should compute FID statistics of the data:
```shell script
python precompute_data_mnist_fid_statistics.py
python precompute_data_mnist_fid_statistics.py --test
python precompute_data_mnist_fid_statistics.py --is_fmnist
python precompute_data_mnist_fid_statistics.py --is_fmnist --test
python compute_fid_statistics.py --path data/processed/cifar10.zip --file cifar10.npz 
python compute_fid_statistics.py --path data/processed/imagenet.zip --file imagenet.npz
python compute_fid_statistics.py --path data/processed/celeba.zip --file celeba.npz
```

### DPDM training

We provide configurations to reproduce our models [here](./configs/). You may use the following command for training:

```shell script
python main.py --mode train --workdir <new_directory> --config <dataset>
```

Our models are trained on eight NVIDIA A100 (80Gb) GPUs; you can set the number of GPUs per node via `setup.n_gpus_per_node`. We use large batch sizes which are split into several iterations; to reduce the required GPU memory, you may increase the flag `train.n_splits` (by a multiple of 2).

## Citation
If you find the provided code or checkpoints useful for your research, please consider citing our NeurIPS paper:

```bib
@article{dockhorn2022differentially,
    title={{Differentially Private Diffusion Models}},
    author={Dockhorn, Tim and Cao, Tianshi and Vahdat, Arash and Kreis, Karsten},
    journal={arXiv:2210.09929},
    year={2022}
}
```

## License

Copyright © 2023, NVIDIA Corporation. All rights reserved.

The code of this work is made available under the NVIDIA Source Code License. Please see our main [LICENSE](./LICENSE) file.

Pre-trained checkpooints are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

#### License Dependencies

For any code dependencies related to StyleGAN3 ([`stylegan3/`](./stylegan3/), [`torch_utils/`](./torch_utils/), and [`dnnlib/`](./dnnlib/)), the license is the  Nvidia Source Code License by NVIDIA Corporation, see [StyleGAN3 LICENSE](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).

Our DPDMs are based on the [DDPM++](./higher_score_flow_code/models/score_sde_pytorch) architecture which is realeased under [Apache License 2.0](https://github.com/yang-song/score_sde_pytorch/blob/main/LICENSE).