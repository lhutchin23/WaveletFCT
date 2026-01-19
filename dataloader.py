"""
Extracts data from CAMUS dataset
Does Image Augmentation + Normalization
"""

import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torchvision.transforms.functional as TF
import tqdm
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# set randomseed's for everything
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageAugmentation:
    """
    Image Augmentation
    Rotations + Flips + color-jitter
    Chose not to do cropping because that could lose important structures
    """

    def __init__(
        self,
        rotation_angle=15,
        flip_prob=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
    ):
        self.rotation_angle = rotation_angle
        self.flip_prob = flip_prob
        self.jitter = v2.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, image, mask):
        if self.rotation_angle > 0:
            angle = random.uniform(-self.rotation_angle, self.rotation_angle)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < self.flip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        image = self.jitter(image)
        return image, mask


class DatasetHelpers(Dataset):
    """
    Helper functions
    """

    def __init__(self, images, masks, transform=None):
        super().__init__()

        if isinstance(images, torch.Tensor):
            images = images.numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.numpy()

        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()
        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


def onehot_encoding(masks, num_classes):
    """
    Convert masks to one-hot encoding
    One-hot masks(N, num_classes, H, W)
    """
    N, _, H, W = masks.shape
    onehot = np.zeros((N, num_classes, H, W), dtype=np.float32)
    for i in range(num_classes):
        onehot[:, i, :, :] = (masks[:, 0, :, :] == i).astype(np.float32)

    return onehot


def normalize_images(images, method="minmax"):
    """
    Normalize images to [0, 1] range

    images: (N, C, H, W) numpy array
    method: 'minmax' or 'zscore'
    """
    normalized = np.zeros_like(images, dtype=np.float32)

    for i in range(len(images)):
        img = images[i]

        if method == "minmax":
            min_val = img.min()
            max_val = img.max()
            if max_val > min_val:
                normalized[i] = (img - min_val) / (max_val - min_val)
            else:
                normalized[i] = img

        elif method == "zscore":
            mean = img.mean()
            std = img.std()
            if std > 0:
                normalized[i] = (img - mean) / std
            else:
                normalized[i] = img

    return normalized


def load_camus_dataset(data_root="./DATA", target_size=(224, 224)):
    """
    Loads CAMUS dataset, I use both 2CH and 4CH veiw as well as both ED and ES phase.
    input:
    target_size, resize images to this size (224 x 224)
    output:
    normalized images (N,1,H,W)
    one-hot encoded masks (N,4,H,W)
    metadata, patient info
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"CAMUS data not found at {data_root}")
    patient_dirs = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("patient")]
    )

    print(f"Loading Camus Data from {data_root}, with {len(patient_dirs)}")

    views = ["2CH", "4CH"]
    phases = ["ED", "ES"]
    images_list, masks_list, metadata_list = [], [], []
    for patient_dir in tqdm.tqdm(patient_dirs, desc="Loading"):
        patient_id = patient_dir.name

        for v in views:
            for p in phases:
                img_file = patient_dir / f"{patient_id}_{v}_{p}.nii.gz"
                mask_file = patient_dir / f"{patient_id}_{v}_{p}_gt.nii.gz"

                if img_file.exists() and mask_file.exists():
                    try:
                        img_data = nib.load(str(img_file)).get_fdata()  # type: ignore
                        mask_data = nib.load(str(mask_file)).get_fdata()  # type: ignore

                        if img_data.ndim == 3:
                            img_data = img_data[:, :, 0]
                            mask_data = mask_data[:, :, 0]

                        img_norm = (img_data - img_data.min()) / (
                            img_data.max() - img_data.min() + 1e-8
                        )

                        if target_size:
                            H, W = img_data.shape
                            zh, zw = target_size[0] / H, target_size[1] / W
                            img_norm = zoom(img_norm, (zh, zw), order=1)
                            mask_data = zoom(mask_data, (zh, zw), order=0)

                        img_norm = img_norm[np.newaxis, :, :]
                        mask_data = mask_data[np.newaxis, :, :]

                        images_list.append(img_norm)
                        masks_list.append(mask_data.astype(np.int32))
                        metadata_list.append(
                            {"patient_id": patient_id, "view": v, "phase": p}
                        )
                    except Exception as e:
                        print(f"Error loading {img_file}: {e}")

    images = np.array(images_list, dtype=np.float32)
    masks = np.array(masks_list, dtype=np.int32)

    print(f"Loaded {len(images)} images, shape: {images.shape}")

    masks_onehot = onehot_encoding(masks, num_classes=4)

    return images, masks_onehot, metadata_list


def load_official_split(data_root):
    """
    Load official CAMUS train/val/test split from subgroup files.
    Returns sets of patient IDs for each split.
    """
    data_root = Path(data_root)

    train_file = data_root / "subgroup_training.txt"
    val_file = data_root / "subgroup_validation.txt"
    test_file = data_root / "subgroup_testing.txt"

    def read_patients(filepath):
        if not filepath.exists():
            raise FileNotFoundError(f"Split file not found: {filepath}")
        with open(filepath, 'r') as f:
            return set(line.strip() for line in f if line.strip())

    train_patients = read_patients(train_file)
    val_patients = read_patients(val_file)
    test_patients = read_patients(test_file)

    print(f"Official split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test patients")

    return train_patients, val_patients, test_patients


def create_dataloaders(
    images, masks, metadata, batch_size=8, train_split=0.7, val_split=0.15, num_workers=0,
    split_type="random", data_root=None
):
    """
    Create train/val/test dataloaders with augmentation for training set.

    takes in:
    train_split: fraction for training (only used if split_type="random")
    val_split: fraction for validation (only used if split_type="random")
    num_workers: number of dataloader workers
    *
    split_type: "random" for 70/15/15 split, "official" for CAMUS official split
    *
    """
    n_samples = len(images)

    if split_type == "official":
        if data_root is None:
            raise ValueError("data_root is required for official split")

        train_patients, val_patients, test_patients = load_official_split(data_root)

        train_idx, val_idx, test_idx = [], [], []
        for i, meta in enumerate(metadata):
            patient_id = meta["patient_id"]
            if patient_id in train_patients:
                train_idx.append(i)
            elif patient_id in val_patients:
                val_idx.append(i)
            elif patient_id in test_patients:
                test_idx.append(i)
            else:
                print(f"Warning: {patient_id} not found in any split file, skipping")

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)

    else:  # random split
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

    train_images, train_masks = images[train_idx], masks[train_idx]
    val_images, val_masks = images[val_idx], masks[val_idx]
    test_images, test_masks = images[test_idx], masks[test_idx]

    train_transform = ImageAugmentation(rotation_angle=15, flip_prob=0.5)

    train_dataset = DatasetHelpers(train_images, train_masks, transform=train_transform)
    val_dataset = DatasetHelpers(val_images, val_masks, transform=None)
    test_dataset = DatasetHelpers(test_images, test_masks, transform=None)
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}