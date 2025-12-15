import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import random

class DF2023Dataset(Dataset):
    def __init__(self, root, transform=None, subset_fraction=1.0, seed=42):
        """
        Args:
            root (str): Path to DF2023_V15 root.
            transform: torchvision transforms to apply.
            subset_fraction (float): fraction of data to keep (e.g. 0.1 = 10%).
            seed (int): random seed for reproducibility.
        """
        self.root = root
        self.transform = transform

        train_dir = os.path.join(root, "DF2023_V15_train")
        img_dir = os.path.join(train_dir, "COCO_V15")
        mask_dir = os.path.join(train_dir, "COCO_V15_GT")

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Could not find {img_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Could not find {mask_dir}")

        self.img_paths, self.mask_paths = [], []

        all_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        all_files.sort()

        if subset_fraction < 1.0:
            random.seed(seed)
            random.shuffle(all_files)
            keep_n = int(len(all_files) * subset_fraction)
            all_files = all_files[:keep_n]
            self.subset_filenames = list(all_files)

            print(f"Using subset: {keep_n} / {len(os.listdir(img_dir))} images ({subset_fraction*100:.1f}%)")

        for fname in all_files:
            img_path = os.path.join(img_dir, fname)
            base = os.path.splitext(fname)[0]
            mask_name = base + "_GT.png"
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                self.img_paths.append(img_path)
                self.mask_paths.append(mask_path)
            else:
                self.img_paths.append(img_path)
                self.mask_paths.append("blank")

        num_tampered = len([m for m in self.mask_paths if m != "blank"])
        num_authentic = len([m for m in self.mask_paths if m == "blank"])
        print(f"[DF2023] Found {num_tampered} tampered pairs and {num_authentic} authentic images.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if mask_path != "blank":
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask, dtype=np.uint8)
            mask_np = (mask_np > 30).astype(np.uint8) * 255
            mask = Image.fromarray(mask_np)
        else:
            mask = Image.new("L", img.size, 0)

        if self.transform:
            img = self.transform(img)
            mask = transforms.functional.resize(mask, img.shape[1:], transforms.InterpolationMode.NEAREST)
            mask = transforms.functional.to_tensor(mask)
        return img, mask

def get_df2023_dataloaders(root, batch_size=4, val_split=0.1, subset_fraction=1.0, seed=42, num_workers=4):
    """
    Returns train_loader, val_loader, dataset
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset = DF2023Dataset(root=root, transform=transform, subset_fraction=subset_fraction, seed=seed)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, dataset
