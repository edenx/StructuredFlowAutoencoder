import torch
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
import torchvision

import numpy as np
import os
import gzip
from PIL import Image # For converting NumPy array to PIL Image in __getitem__
import shutil # For managing directories

torch.set_default_dtype(torch.float64)

# transform = transforms.Compose([transforms.ToTensor()])
# transform = transforms.Compose([
#     transforms.Pad((2, 2, 2, 2)),  # Pad 2 pixels on each side (left, top, right, bottom)
#     transforms.ToTensor()          # Convert image to PyTorch tensor
# ])
class LogitTrans():
    def __init__(self, a=0.01, b=0.99):
        self.a = a
        self.b = b

    def __call__(self, x):
        x_scaled = self.a + (self.b - self.a) * x
        return torch.logit(x_scaled)

class SigmoidTrans():
    def __init__(self, a=0.01, b=0.99):
        self.a = a
        self.b = b

    def __call__(self, x):
        x_scaled = torch.sigmoid(x)
        # return x_scaled
        # return x
        return torch.clamp((x_scaled - self.a) / (self.b - self.a), 0, 1)


transform = transforms.Compose([
    # transforms.Resize((14, 14)),  # Pad 2 pixels on each side (left, top, right, bottom)
    transforms.ToTensor(),         # Convert image to PyTorch tensor
    # transforms.Lambda(lambda x: x.double())
    # transforms.Normalize((0.5,), (0.5,)) # normalize to between (-1,1)
    LogitTrans(),
])

inv_transform = transforms.Compose([
    # transforms.Resize((14, 14)),  # Pad 2 pixels on each side (left, top, right, bottom)
    # transforms.Normalize((0.5,), (0.5,)) # normalize to between (-1,1)
    SigmoidTrans(),
])

class StratifiedSampler(Sampler):
    def __init__(self, num_classes, data_source, samples_per_class, data="MNIST", split="mnist", replace=False):
        self.data_source = data_source
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.replace=replace
        if data == "EMNIST" and split == "letters":
            self.indices_per_class = {i: np.where(np.array(data_source.labels_data_np) == i)[0] for i in range(1,self.num_classes)}
        elif data == "MNIST":
            self.indices_per_class = {i: np.where(np.array(data_source.targets) == i)[0] for i in range(self.num_classes)}
        else:
            self.indices_per_class = {i: np.where(np.array(data_source.labels_data_np) == i)[0] for i in range(self.num_classes)}
        
    def __iter__(self):
        indices = []
        for class_id, indices_class in self.indices_per_class.items():
            indices.extend(np.random.choice(indices_class, self.samples_per_class, replace=self.replace))
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.samples_per_class * len(self.indices_per_class)


def get_mnist(num_classes, dat_dir, num_per_class, num_per_class_test, download=False, logtran=True, split="mnist", replace=False):
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
    train_dataset = MNIST(dat_dir, train=True, download=download,
                          transform=tran
                          )

    test_dataset = MNIST(dat_dir+"_test", train=False, download=download,
                               transform=tran
                               )

    train_sampler = StratifiedSampler(num_classes, train_dataset, samples_per_class=num_per_class, split=split)
    test_sampler = StratifiedSampler(num_classes, test_dataset, samples_per_class=num_per_class_test, split=split)

    return train_dataset, test_dataset, train_sampler, test_sampler


def get_emnist(num_classes, dat_dir, num_per_class, num_per_class_test, split="mnist", download=False, logtran=True, replace=False):
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

    train_dataset = EMNIST(dat_dir, train=True, 
                          transform=tran, split=split
                          )

    # test_dataset = EMNIST(dat_dir, train=False, download=download,
    #                            transform=transform, split=split
    #                            )

    train_sampler = StratifiedSampler(num_classes, train_dataset, samples_per_class=num_per_class, data="EMNIST", split=split, replace=replace)
    # test_sampler = StratifiedSampler(num_classes, test_dataset, samples_per_class=num_per_class_test)

    return train_dataset, train_sampler



# --- IDX File Reading Utilities ---
def _read_idx_int(buffer):
    """Reads a 4-byte big-endian integer from the buffer."""
    return int.from_bytes(buffer.read(4), 'big')

def read_emnist_label_file(path: str) -> np.ndarray:
    """
    Reads an EMNIST label file in IDX format.
    Based on torchvision.datasets.mnist.read_label_file.
    """
    with gzip.open(path, 'rb') as f:
        magic = _read_idx_int(f)
        if magic != 2049: # Magic number for label files is 0x00000801
            raise ValueError(f"Invalid magic number {magic} in label file {path}. Expected 2049.")
        num_items = _read_idx_int(f)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        if len(labels) != num_items:
            raise ValueError(f"Number of items mismatch in label file {path}: "
                             f"header says {num_items}, but found {len(labels)} labels.")
        return labels

def read_emnist_image_file(path: str) -> np.ndarray:
    """
    Reads an EMNIST image file in IDX format.
    Applies the necessary transpose to orient images correctly (like MNIST).
    Based on torchvision.datasets.mnist.read_image_file and EMNIST processing.
    """
    with gzip.open(path, 'rb') as f:
        magic = _read_idx_int(f)
        if magic != 2051: # Magic number for image files is 0x00000803
            raise ValueError(f"Invalid magic number {magic} in image file {path}. Expected 2051.")
        num_images = _read_idx_int(f)
        num_rows = _read_idx_int(f)
        num_cols = _read_idx_int(f)
        
        images_data = np.frombuffer(f.read(), dtype=np.uint8)
        if len(images_data) != num_images * num_rows * num_cols:
            raise ValueError(f"Image data size mismatch in {path}: "
                             f"header implies {num_images * num_rows * num_cols} bytes, "
                             f"but found {len(images_data)} bytes.")
        
        # Reshape to (num_images, num_rows, num_cols)
        images = images_data.reshape(num_images, num_rows, num_cols)
        
        # EMNIST images in the raw files are transposed compared to how they are
        # usually displayed or used (e.g., in MNIST).
        # torchvision.datasets.EMNIST applies a permute(0, 2, 1) to the loaded tensor,
        # which means swapping the last two dimensions (rows and columns for each image).
        # For a NumPy array (N, H, W), this is equivalent to transposing each image: (N, W, H)
        images_transposed = images.transpose(0, 2, 1)
        
        return images_transposed


class EMNIST(Dataset):
    def __init__(self, root: str,
                 split: str = 'digits',    # Default to 'digits' as requested
                 train: bool = True,
                 transform: callable = None,
                 target_transform: callable = None
                 ):
        """
        Custom PyTorch Dataset class for EMNIST, structured similarly to the PINWHEEL example.
        It loads data directly from raw EMNIST IDX files (individual .gz files for images and labels)
        assumed to be already downloaded and extracted into the 'root/EMNIST/raw/' directory.

        Args:
            root (str): Root directory where the 'EMNIST' folder (containing 'raw' and
                        'processed' subfolders) exists. This is the PARENT of the 'EMNIST' folder.
                        For example, if your files are in 'my_data/EMNIST/raw/', root should be 'my_data'.
            split (str, optional): The EMNIST dataset split to use. Valid options are:
                                 'byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist'.
                                 Defaults to 'digits'.
            train (bool, optional): If True, creates dataset from the EMNIST training set,
                                    otherwise from the EMNIST test set. Defaults to True.
            transform (callable, optional): A function/transform that takes in a PIL image
                                         and returns a transformed version. If None,
                                         it defaults to ToTensor() in __getitem__.
                                         Defaults to None.
            target_transform (callable, optional): A function/transform that takes in the
                                                 target and transforms it. Defaults to None.
        """
        self.root = root
        self.split = split
        self.train = train
        self.custom_transform = transform
        self.custom_target_transform = target_transform

        self.emnist_root_dir = os.path.join(self.root, "EMNIST")
        self.raw_folder = os.path.join(self.emnist_root_dir, "raw")
        # Processed folder is not strictly used by this manual loader for loading,
        # but good to define if any caching were to be added later.
        self.processed_folder = os.path.join(self.emnist_root_dir, "processed")

        image_file_name = f"emnist-{self.split}-{'train' if self.train else 'test'}-images-idx3-ubyte.gz"
        label_file_name = f"emnist-{self.split}-{'train' if self.train else 'test'}-labels-idx1-ubyte.gz"
        
        self.image_file_path = os.path.join(self.raw_folder, image_file_name)
        self.label_file_path = os.path.join(self.raw_folder, label_file_name)

        print(f"Attempting to load EMNIST data directly from IDX files:")
        print(f"  Image file: {self.image_file_path}")
        print(f"  Label file: {self.label_file_path}")

        if not os.path.exists(self.image_file_path) or not os.path.exists(self.label_file_path):
            print(f"Error: Raw EMNIST files not found for split='{self.split}', train={self.train}.")
            self._provide_specific_file_guidance(missing_files_override=True) # Pass override
            raise FileNotFoundError(f"Required EMNIST raw files not found. "
                                    f"Image: '{self.image_file_path}', Label: '{self.label_file_path}'")
        
        try:
            # Load images and labels directly from the .gz IDX files
            # These will be NumPy arrays.
            self.images_data_np = read_emnist_image_file(self.image_file_path) # (N, H, W), uint8, transposed
            self.labels_data_np = read_emnist_label_file(self.label_file_path) # (N,), uint8
            
            self.length = len(self.labels_data_np)
            if len(self.images_data_np) != self.length:
                raise ValueError("Number of images and labels mismatch after loading.")

            print(f"Successfully loaded raw EMNIST data: split='{split}', train={train}. Found {self.length} samples.")

        except Exception as e:
            print(f"Error loading or processing raw EMNIST files for split='{split}', train={train}: {e}")
            self._provide_specific_file_guidance()
            raise  # Re-raise the error to halt execution

    def _provide_specific_file_guidance(self, missing_files_override=False):
        """Helper to print expected file paths if loading fails."""
        print(f"\n--- EMNIST File Guidance ---")
        print(f"Expected raw files should be in: '{self.raw_folder}'")
        if not os.path.isdir(self.raw_folder) and not missing_files_override:
            print(f"Error: The raw folder '{self.raw_folder}' does not exist or is not a directory.")
            print(f"Please ensure your 'root' parameter ('{self.root}') is correct and that the "
                  f"'{self.root}/EMNIST/raw/' directory structure exists.")
            return

        prefix = f"emnist-{self.split}-{'train' if self.train else 'test'}"
        img_file_name = f"{prefix}-images-idx3-ubyte.gz"
        lbl_file_name = f"{prefix}-labels-idx1-ubyte.gz"
        expected_img_path = os.path.join(self.raw_folder, img_file_name)
        expected_lbl_path = os.path.join(self.raw_folder, lbl_file_name)

        print(f"  Expected image file: '{expected_img_path}' (Exists: {os.path.exists(expected_img_path)})")
        print(f"  Expected label file: '{expected_lbl_path}' (Exists: {os.path.exists(expected_lbl_path)})")
        
        if not os.path.exists(expected_img_path) or not os.path.exists(expected_lbl_path):
             print("\n  One or both essential raw files are missing.")
        elif not missing_files_override : # Files exist, but there was still an error
             print("\n  Essential raw files appear to exist.")
             print(f"  The error might be due to corrupted files or an unexpected file format.")
        
        print("\nPlease ensure you have:")
        print(f"1. Manually downloaded 'gzip.zip' from the official NIST source: "
              "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip")
        print(f"2. EXTRACTED ALL individual 'emnist-*-...gz' files and 'emnist-*-mapping.txt' files "
              f"from 'gzip.zip' DIRECTLY into the '{self.raw_folder}' directory.")
        print(f"   Do NOT place them in a sub-folder within '{self.raw_folder}'.")
        print(f"3. Double-check that your 'root' path ('{self.root}') and 'split' name ('{self.split}') are correct.")
        print(f"--- End Guidance ---")


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Retrieve the NumPy image data and integer label
        image_np = self.images_data_np[idx] # This is (H, W) uint8, already correctly oriented
        label_int = self.labels_data_np[idx].item() # .item() to get Python int from np.uint8

        # Convert NumPy array to PIL Image (mode 'L' for grayscale)
        # EMNIST images are uint8, so mode 'L' is appropriate.
        image_pil = Image.fromarray(image_np, mode='L')

        # Apply image transformation
        image_tensor = None
        if self.custom_transform:
            image_tensor = self.custom_transform(image_pil)
        else:
            # Default transform: Convert PIL image to PyTorch tensor
            image_tensor = tv_transforms.ToTensor()(image_pil)

        # Apply target transformation
        label_transformed = None
        if self.custom_target_transform:
            label_transformed = self.custom_target_transform(label_int)
        else:
            label_transformed = label_int
        
        # Ensure correct return types
        return image_tensor.float(), torch.tensor(label_transformed, dtype=torch.long)


