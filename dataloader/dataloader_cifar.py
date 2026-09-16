import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler

from dataloader.dataloader_mnist import StratifiedSampler, LogitTrans, SigmoidTrans

torch.set_default_dtype(torch.float64)

class Transback():
    def __init__(self):
        pass
    def __call_(self, x):
        return x / 2 + 0.5

# transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])


# inv_transform = transforms.Compose([
#     Transback()
#     ])

transform = transforms.Compose([
    # transforms.Resize((14, 14)),  # Pad 2 pixels on each side (left, top, right, bottom)
    transforms.ToTensor(),         # Convert image to PyTorch tensor
    # transforms.Lambda(lambda x: x.double())
    # transforms.Normalize((0.5,), (0.5,)) # normalize to between (-1,1)
    transforms.Lambda(lambda x: torch.clamp(x, 0.01, 0.99)),
    LogitTrans(),
])

inv_transform = transforms.Compose([
    # transforms.Resize((14, 14)),  # Pad 2 pixels on each side (left, top, right, bottom)
    # transforms.Normalize((0.5,), (0.5,)) # normalize to between (-1,1)
    SigmoidTrans(),
])

def get_cifar(num_classes, dat_dir, num_per_class, num_per_class_test, download=False):
    
    train_dataset = CIFAR10(
                        root = dat_dir,
                        train = True,
                        download = download,
                        transform = transform
                    )
    test_dataset = CIFAR10(
                        root = dat_dir+"_test",
                        train = False,
                        download = download,
                        transform = transform
                    )
    # train_sampler = DistributedSampler(train_dataset)
    # test_sampler = DistributedSampler(test_dataset)

    train_sampler = StratifiedSampler(num_classes, train_dataset, samples_per_class=num_per_class)
    test_sampler = StratifiedSampler(num_classes, test_dataset, samples_per_class=num_per_class_test)

    return train_dataset, test_dataset, train_sampler, test_sampler

