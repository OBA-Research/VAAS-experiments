import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from datetime import datetime


from dataset.casia2_dataset_loader import get_casia2_dataloaders
from dataset.df2023_dataset_loader import get_df2023_dataloaders

from vaas.px.px_model import PatchConsistencySegformer


@torch.no_grad()
def compute_metrics(pred, gt):
    """Binary metrics given flat arrays (0/1)."""
    f1 = f1_score(gt, pred, zero_division=0)
    prec = precision_score(gt, pred, zero_division=0)
    rec = recall_score(gt, pred, zero_division=0)
    inter = np.logical_and(pred == 1, gt == 1).sum()
    union = np.logical_or(pred == 1, gt == 1).sum()
    iou = inter / (union + 1e-8)
    return {"F1": f1, "Precision": prec, "Recall": rec, "IoU": iou}


def sweep_thresholds(
    dataset_root,
    checkpoint_dir,
    thresholds=np.linspace(0.1, 0.9, 9),
    max_samples=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Running threshold sweep on device: {device}")
    model_px = PatchConsistencySegformer().to(device)
    ckpt = torch.load(os.path.join(checkpoint_dir, "best_model_px.pth"), map_location=device)
    model_px.load_state_dict(ckpt["model_state_dict"])
    model_px.eval()

    if "casia" in dataset_root.lower():
        print("Detected CASIA2 dataset.")
        _, val_loader, full_dataset = get_casia2_dataloaders(dataset_root, batch_size=1, val_split=0.1)
    elif "df2023" in dataset_root.lower():
        print("Detected DF2023 dataset.")
        _, val_loader, full_dataset = get_df2023_dataloaders(dataset_root, batch_size=1, subset_fraction=0.1)
    else:
        raise ValueError(f"Unknown dataset type for root: {dataset_root}")


    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    results = []
    for t in thresholds:
        all_metrics = {"F1": [], "Precision": [], "Recall": [], "IoU": []}
        for idx in tqdm(range(len(full_dataset)), desc=f"Threshold {t:.2f}"):
            img_path = full_dataset.img_paths[idx]
            mask_path = full_dataset.mask_paths[idx]
            if mask_path == "blank":
                continue

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            img_t = transform(img).unsqueeze(0).to(device)
            mask_np = (np.array(mask.resize((224, 224))) > 128).astype(np.uint8).flatten()

            pred = torch.sigmoid(model_px(img_t))
            pred = torch.nn.functional.interpolate(
                pred, size=(224, 224), mode="bilinear", align_corners=False
            )
            pred_np = pred.squeeze().detach().cpu().numpy()
            pred_bin = (pred_np > t).astype(np.uint8).flatten()

            metrics = compute_metrics(pred_bin, mask_np)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

            if max_samples and len(all_metrics["F1"]) >= max_samples:
                break

        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        avg_metrics["Threshold"] = t
        results.append(avg_metrics)


    print("\nThreshold sweep results:")
    print(f"{'Thresh':>7} | {'F1':>6} | {'IoU':>6} | {'Prec':>6} | {'Rec':>6}")
    for r in results:
        print(f"{r['Threshold']:7.2f} | {r['F1']:6.3f} | {r['IoU']:6.3f} | {r['Precision']:6.3f} | {r['Recall']:6.3f}")

    best = max(results, key=lambda x: x["F1"])
    print(f"\n--> Best threshold = {best['Threshold']:.2f} (F1={best['F1']:.3f}, IoU={best['IoU']:.3f})<--")

    np.save(os.path.join(checkpoint_dir, "threshold_sweep_results.npy"), results)


    config_path = os.path.join(checkpoint_dir, "config.json")
    config_data = {}

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
        except Exception:
            config_data = {}

    config_data.update({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_threshold": float(best["Threshold"]),
        "best_F1": float(best["F1"]),
        "best_IoU": float(best["IoU"]),
        "best_precision": float(best["Precision"]),
        "best_recall": float(best["Recall"]),
        "sweep_file": "threshold_sweep_results.npy"
    })


    train_cfg_path = os.path.join(checkpoint_dir, "train_args.json")
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path, "r") as f:
            config_data["train_args"] = json.load(f)

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)


    print(f"Saved summary configuration to {config_path}")


    thresholds_list = [r["Threshold"] for r in results]
    f1_list = [r["F1"] for r in results]
    iou_list = [r["IoU"] for r in results]
    prec_list = [r["Precision"] for r in results]
    rec_list = [r["Recall"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds_list, f1_list, "o-", label="F1 Score", color="#ff7b00", lw=2)
    plt.plot(thresholds_list, iou_list, "s--", label="IoU", color="#0077b6", lw=2)
    plt.plot(thresholds_list, prec_list, "d--", label="Precision", color="#55a630", alpha=0.8)
    plt.plot(thresholds_list, rec_list, "x--", label="Recall", color="#e63946", alpha=0.8)

    plt.axvline(best["Threshold"], color="gray", ls=":", lw=1)
    plt.text(
        best["Threshold"] + 0.01,
        max(f1_list) - 0.02,
        f"Best Th={best['Threshold']:.2f}",
        fontsize=9,
        color="gray",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Sweep — Px Model Performance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(checkpoint_dir, "threshold_sweep_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved threshold sweep plot to {plot_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sweep threshold for Px model F1")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    sweep_thresholds(
        dataset_root=args.dataset_root,
        checkpoint_dir=args.checkpoint_dir,
        max_samples=args.max_samples,
    )

