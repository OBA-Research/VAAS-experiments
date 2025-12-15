import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
import numpy as np


class Casia2Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.img_paths = []
        self.mask_paths = []

        au_dir = os.path.join(root, "Au")
        tp_dir = os.path.join(root, "Tp")
        gt_dir = os.path.join(root, "CASIA 2 Groundtruth")

        if os.path.exists(au_dir):
            for fname in os.listdir(au_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.img_paths.append(os.path.join(au_dir, fname))
                    self.mask_paths.append("blank")

        if os.path.exists(tp_dir):
            for fname in os.listdir(tp_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    base = os.path.splitext(fname)[0]
                    mask_candidates = [
                        os.path.join(gt_dir, base + "_gt.png"),
                        os.path.join(gt_dir, base + ".png"),
                        os.path.join(gt_dir, base + "_mask.png"),
                        os.path.join(gt_dir, base + "_gt.jpg"),
                    ]
                    mask_path = next((m for m in mask_candidates if os.path.exists(m)), None)

                    self.img_paths.append(os.path.join(tp_dir, fname))
                    if mask_path:
                        self.mask_paths.append(mask_path)
                    else:
                        self.mask_paths.append("blank")

        num_tampered = len([m for m in self.mask_paths if m != "blank"])
        num_authentic = len([m for m in self.mask_paths if m == "blank"])
        print(f"[CASIA2] Found {num_tampered} tampered pairs and {num_authentic} authentic images.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB")

        if mask_path != "blank":
            mask = Image.open(mask_path)
            if mask.mode == "RGBA":
                mask = mask.getchannel("A")
            else:
                mask = mask.convert("L")

            mask_np = np.array(mask, dtype=np.uint8)
            mask_np = (mask_np > 30).astype(np.uint8) * 255

            if np.count_nonzero(mask_np) == 0:
                mask_np = 255 - mask_np

            mask = Image.fromarray(mask_np)
        else:
            mask = Image.new("L", img.size, 0)

        if self.transform:
            img = self.transform(img)
            mask = transforms.functional.resize(mask, img.shape[1:], transforms.InterpolationMode.NEAREST)
            mask = transforms.functional.to_tensor(mask)

        return img, mask



def get_casia2_dataloaders(root, batch_size=4, val_split=0.1, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset = Casia2Dataset(root=root, transform=transform)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    print("Performing oversampling of tampered samples...")
    weights = []
    for i in range(len(train_ds)):
        _, mask = train_ds[i]
        if mask.sum() > 0:
            weights.append(3.0) 
        else:
            weights.append(1.0)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, dataset

