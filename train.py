from vaas.fx.fx_model import FxViT
from vaas.fx.fx_utils import compute_reference_stats
from vaas.px.px_model import PatchConsistencySegformer

import random
import warnings
warnings.filterwarnings("ignore", message="Some weights of SegformerForSemanticSegmentation")

from tqdm import tqdm
from vaas.utils.seed import seed_everything
from vaas.utils.helpers import save_json, check_CUDA_available

from training.losses import hybrid_seg_loss, dice_loss_from_logits
from training.metrics import compute_segmentation_metrics

from evaluation.visualization import visualize_results
from vaas.fusion.hybrid_score import compute_scores

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torch
import numpy as np
import os
import csv
import time
import json
from datetime import datetime
from PIL import Image



try:
    import pynvml

    nvml_available = True
except ImportError:
    pynvml = None
    nvml_available = False

seed_everything(42)
device = check_CUDA_available()
print(f"Using device: {device}")


def train_patch_model(
    train_loader,
    val_loader,
    epochs,
    lr,
    patience,
    checkpoint_dir,
    loss_type="bce",
    pos_weight=15.0,
    metric_threshold=0.5,
    dice_weight=0.5,
    focal_alpha=0.25,
    focal_gamma=2.0,
):
    """Train the Px (SegFormer) model with configurable loss and metrics."""

    print(f"Loss type: {loss_type} (Dice weight={dice_weight})")
    if loss_type == "bce":
        print(f"BCE pos_weight = {pos_weight}")
    else:
        print(
            f"   Focal α={focal_alpha}, γ={focal_gamma}, "
            f"Dice weight={dice_weight}, pos_weight={pos_weight}"
        )
    print(f"Metric threshold: {metric_threshold}")

    model = PatchConsistencySegformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",        
        factor=0.5,
        patience=3,
        cooldown=1
    )

    if pos_weight is not None and pos_weight > 0:
        pos_weight_tensor = torch.tensor([pos_weight], device=device)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        bce_loss = nn.BCEWithLogitsLoss()

    def compute_loss(logits_up, masks):
        """Return scalar loss for a batch given logits and masks."""
        targets = masks.float()
        if loss_type == "focal":
            probs = torch.sigmoid(logits_up)
            bce = F.binary_cross_entropy_with_logits(
                logits_up, targets, reduction="none"
            )
            pt = torch.where(targets > 0.5, probs, 1.0 - probs)
            focal = focal_alpha * (1.0 - pt) ** focal_gamma * bce
            focal = focal.mean()
            dice = dice_loss_from_logits(logits_up, targets)
            return focal + dice_weight * dice
        else:
            return hybrid_seg_loss(
                logits_up,
                targets,
                bce_loss_fn=bce_loss,
                dice_weight=dice_weight,
            )

    best_f1 = 0.0
    patience_counter = 0

    log_file = os.path.join(checkpoint_dir, "training_log_px.csv")
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model_px.pth")
    last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint_px.pth")


    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Epoch",
                    "Train_Loss",
                    "Val_Loss",
                    "IoU",
                    "F1",
                    "Precision",
                    "Recall",
                    "LR",
                    "Epoch_Time_min",
                ]
            )

    if torch.cuda.is_available() and nvml_available and pynvml is not None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        total_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1e9
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device: {device_name} ({total_mem:.2f} GB total)")
    else:
        total_mem = 0.0

    print("Starting fresh training (ignoring previous checkpoints).")

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            logits_up = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )  

            loss = compute_loss(logits_up, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_f1 = 0.0
        val_prec = 0.0
        val_rec = 0.0
        loss_batches = 0
        metric_batches = 0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Val {epoch+1}/{epochs}", leave=False):
                imgs, masks = imgs.to(device), masks.to(device)

                logits = model(imgs)
                logits_up = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                batch_loss = compute_loss(logits_up, masks)
                val_loss += batch_loss.item()
                loss_batches += 1

                flat_mask = masks.view(masks.size(0), -1)
                non_empty = (flat_mask.sum(dim=1) > 0)
                if non_empty.any():
                    iou_b, f1_b, prec_b, rec_b = compute_segmentation_metrics(
                        logits_up[non_empty],
                        masks[non_empty],
                        threshold=metric_threshold,
                    )
                    val_iou += iou_b
                    val_f1 += f1_b
                    val_prec += prec_b
                    val_rec += rec_b
                    metric_batches += 1

        if loss_batches > 0:
            val_loss /= loss_batches
        else:
            val_loss = 0.0

        if metric_batches > 0:
            val_iou /= metric_batches
            val_f1 /= metric_batches
            val_prec /= metric_batches
            val_rec /= metric_batches
        else:
            val_iou = val_f1 = val_prec = val_rec = 0.0

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time_min = (time.time() - epoch_start) / 60.0

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"IoU: {val_iou:.3f} | F1: {val_f1:.3f} | "
            f"Precision: {val_prec:.3f} | Recall: {val_rec:.3f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time_min:.2f} min"
        )

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    round(train_loss, 4),
                    round(val_loss, 4),
                    round(val_iou, 4),
                    round(val_f1, 4),
                    round(val_prec, 4),
                    round(val_rec, 4),
                    current_lr,
                    round(epoch_time_min, 2),
                ]
            )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
            },
            last_ckpt_path,
        )

        if val_f1 > best_f1:
            delta = val_f1 - best_f1
            best_f1 = val_f1
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                },
                best_ckpt_path,
            )
            with open(os.path.join(checkpoint_dir, "bestF1.txt"), "w") as f:
                f.write(f"{best_f1:.3f}\n")


            print(
                f"F1 improved by {delta:.3f} -> {best_f1:.3f}. "
                f"Saved best model to {best_ckpt_path}\n"
            )
        else:
            patience_counter += 1
            print(f"No improvement in F1 for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered: no F1 improvement for "
                    f"{patience} consecutive epochs.\n"
                )
                break

    return model



def main():
    import argparse

    from dataset.casia2_dataset_loader import get_casia2_dataloaders
    from dataset.df2023_dataset_loader import get_df2023_dataloaders

    seed_everything(42)

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate VAAS Px module with Fx-based reference stats."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CASIA2",
        choices=["CASIA2", "DF2023"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to dataset root.",
    )
    parser.add_argument(
        "--exp-id",
        type=str,
        default="default_exp",
        help="Experiment identifier string.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Base directory for experiment outputs.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable forensic-aware data augmentation.",
    )

    parser.add_argument(
        "--loss-type",
        type=str,
        default="bce",
        choices=["bce", "focal"],
        help="Loss type for Px segmentation: 'bce' for BCE+Dice, 'focal' for Focal+Dice.",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=15.0,
        help="Positive class weight for BCE/Focal loss (foreground tampered pixels).",
    )
    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=0.5,
        help="Threshold on sigmoid outputs used for metrics (IoU/F1/Precision/Recall).",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.5,
        help="Weight for Dice loss term (default: 0.5).",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Alpha parameter for Focal loss (default: 0.25).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal loss (default: 2.0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for combining global (ViT) and patch-level (SegFormer) scores.",
    )

    args = parser.parse_args()


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", f"{args.dataset}_{args.exp_id}_{timestamp}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_config = vars(args)
    train_config["timestamp"] = timestamp

    config_path = os.path.join(checkpoint_dir, "train_args.json")
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=4)

    print(f"Saved training config to {config_path}")

    fx_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    print(f"Selected dataset: {args.dataset}")
    if args.dataset == "CASIA2":
        print(f"Loading CASIA2 from: {args.dataset_root}")
        train_loader, val_loader, full_dataset = get_casia2_dataloaders(
            args.dataset_root,
            batch_size=args.batch_size,
            val_split=0.1,
            num_workers=4,
        )
    else:
        print(f"Loading DF2023 from: {args.dataset_root}")
        train_loader, val_loader, full_dataset = get_df2023_dataloaders(
            args.dataset_root,
            batch_size=args.batch_size,
            val_split=0.1,
            subset_fraction=0.1,
            num_workers=2,
        )
        subset_file = os.path.join(checkpoint_dir, "df2023_subset_filenames.json")

        if hasattr(full_dataset, "subset_filenames"):
            save_json({"subset_filenames": full_dataset.subset_filenames}, subset_file)
            print(f"Saved DF2023 subset filenames to {subset_file}")
        else:
            print("Full DF2023 dataset used (no subset).")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.dataset}_{args.exp_id}_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Logs and checkpoints will be saved in: {checkpoint_dir}")


    vit_model = FxViT().to(device)
    vit_model.eval()
    model_px = train_patch_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        checkpoint_dir=checkpoint_dir,
        loss_type=args.loss_type,
        pos_weight=args.pos_weight,
        metric_threshold=args.metric_threshold,
        dice_weight=args.dice_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )

    mu_ref, sigma_ref = compute_reference_stats(
        full_dataset,       
        vit_model,          
        device,
        fx_transform
    )

    ref_stat_path = os.path.join(checkpoint_dir, "ref_stats.pth")
    torch.save({"mu_ref": mu_ref, "sigma_ref": sigma_ref}, ref_stat_path)
    print(
        f"Saved reference stats to {ref_stat_path} "
        f"(mu_ref={mu_ref:.4f}, sigma_ref={sigma_ref:.4f})"
    )

    print("Running sample inference on validation data...")
    val_dataset = val_loader.dataset

    if hasattr(val_dataset, "dataset") and hasattr(val_dataset, "indices"):
        base_dataset = val_dataset.dataset
        first_idx = val_dataset.indices[0]
        img_path = base_dataset.img_paths[first_idx]
        mask_path = base_dataset.mask_paths[first_idx]
    else:
        img_path = val_dataset.img_paths[0]
        mask_path = val_dataset.mask_paths[0]

    img = Image.open(img_path).convert("RGB")
    if mask_path == "blank":
        mask = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
    else:
        mask = Image.open(mask_path).convert("L")

    s_f, s_p, s_h, pred_map = compute_scores(
        img=img,
        mask=mask,
        model_px=model_px,
        vit_model=vit_model,
        mu_ref=mu_ref,
        sigma_ref=sigma_ref,
        transform=fx_transform,
        alpha=args.alpha,
    )

    vis_path = os.path.join(checkpoint_dir, f"{args.dataset}_overlay.png")

    visualize_results(
    img,
    mask,
    pred_map,
    vit_model,
    fx_transform,
    s_h,
    save_path=vis_path,
    vis_mode="both"
)


    print(
        f"[INFO] S_F={s_f:.4f}, S_P={s_p:.4f}, S_H={s_h:.4f} "
        f"(visualization at {vis_path})"
    )


if __name__ == "__main__":
    main()

    