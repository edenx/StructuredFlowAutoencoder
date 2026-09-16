from __future__ import print_function
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from dataloader.dataloader_mnist import * 
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# https://github.com/google-deepmind/dsprites-dataset

class DisentangledSpritesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir, transform=None):
        """
        Args:
            dir (string): Directory containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.dir = dir
        self.filename = 'dsprites.npz'
        self.filepath = f'{self.dir}/{self.filename}'
        dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')

        # print('Keys in the dataset:', dataset_zip.keys())
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]

        # print('Metadata: \n', self.metadata)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.float32)
        # sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        if self.transform:
            sample = self.transform(sample)
        return sample, self.latents_values[idx], self.latents_classes[idx]



def get_dspirit(num_classes, dat_dir, num_per_class, num_per_class_test, logtran=True, val_split=0.9):
    if not logtran:
        tran = transforms.Compose([
            # transforms.Resize((14, 14)),  # Pad 2 pixels on each side (left, top, right, bottom)
            transforms.ToTensor(),         # Convert image to PyTorch tensor
            # transforms.Lambda(lambda x: x.double())
            # transforms.Normalize((0.5,), (0.5,)) # normalize to between (-1,1)
            # LogitTrans(),
        ])
    else:
        tran =transforms.Compose([
            # transforms.Resize((14, 14)),  # Pad 2 pixels on each side (left, top, right, bottom)
            transforms.ToTensor(),         # Convert image to PyTorch tensor
            # transforms.Lambda(lambda x: x.double())
            # transforms.Normalize((0.5,), (0.5,)) # normalize to between (-1,1)
            LogitTrans(),
        ])
    dataset = DisentangledSpritesDataset(dat_dir, transform=tran)

    # train_sampler = StratifiedSampler(num_classes, train_dataset, samples_per_class=num_per_class)
    # test_sampler = StratifiedSampler(num_classes, test_dataset, samples_per_class=num_per_class_test)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Create data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(val_indices)


    return train_dataset, test_dataset, train_sampler, test_sampler




