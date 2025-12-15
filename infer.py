import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from vaas.fx.fx_model import FxViT
from vaas.px.px_model import PatchConsistencySegformer
from evaluation.visualization import visualize_results
from vaas.fusion.hybrid_score import compute_scores


from dataset.casia2_dataset_loader import get_casia2_dataloaders
from dataset.df2023_dataset_loader import get_df2023_dataloaders

import random
import json

from vaas.utils.seed import seed_everything
seed_everything(42)


def load_best_threshold(checkpoint_dir, fallback=0.5):
    """
    Loads the best threshold and calibration metrics from config.json if available,
    otherwise falls back to the sweep file or a default threshold.
    Returns:
        threshold (float),
        cfg (dict or None)
    """
    
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            th = cfg.get("best_threshold", fallback)
            print(f"Loaded config summary from {config_path}")
            print(
                f"Using best threshold tau = {th:.2f} "
                f"(F1={cfg.get('best_F1', '—')}, IoU={cfg.get('best_IoU', '—')})"
            )
            return th, cfg
        except Exception as e:
            print(f"Could not read config.json: {e}")

    sweep_path = os.path.join(checkpoint_dir, "threshold_sweep_results.npy")
    if os.path.exists(sweep_path):
        try:
            data = np.load(sweep_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and len(data) > 0:
                results = list(data)
                best = max(results, key=lambda x: x["F1"])
                cfg = {
                    "best_threshold": float(best["Threshold"]),
                    "best_F1": best["F1"],
                    "best_IoU": best["IoU"],
                    "best_precision": best.get("Precision", None),
                    "best_recall": best.get("Recall", None),
                }
                print(f" Loaded best threshold from sweep: {best['Threshold']:.2f}")
                return float(best["Threshold"]), cfg
        except Exception as e:
            print(f"Could not read threshold sweep file: {e}")

    print(f"Using fallback threshold {fallback:.2f}")
    return fallback, None



@torch.no_grad()
def run_inference(
    dataset_root,
    checkpoint_dir,
    output_dir,
    num_samples=10,
    threshold=0.5,
    vis_mode="both",
    alpha=0.5,
):
    """
    Perform inference on multiple validation images using trained Px and Fx models.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from {checkpoint_dir}")
    best_ckpt = torch.load(os.path.join(checkpoint_dir, "best_model_px.pth"), map_location=device)
    ref_stats = torch.load(os.path.join(checkpoint_dir, "ref_stats.pth"), map_location=device)
    mu_ref, sigma_ref = ref_stats["mu_ref"], ref_stats["sigma_ref"]

    model_px = PatchConsistencySegformer().to(device)
    model_px.load_state_dict(best_ckpt["model_state_dict"])
    model_px.eval()

    vit_model = FxViT().to(device)
    vit_model.eval()


    threshold, cfg = load_best_threshold(checkpoint_dir, fallback=threshold)

    os.makedirs(output_dir, exist_ok=True)
    used_config = {
        "used_threshold": threshold,
        "source_checkpoint": checkpoint_dir,
        "metrics": cfg if cfg else "No config metadata found"
    }
    with open(os.path.join(output_dir, "used_config.json"), "w") as f:
        import json
        json.dump(used_config, f, indent=4)
    print(f"Saved inference configuration to {output_dir}/used_config.json")

    if cfg:
        print(f"Model F1={cfg.get('best_F1', '—')}, IoU={cfg.get('best_IoU', '—')}, tau={threshold:.2f}")




    print(f"Loading dataset from: {dataset_root}")
    if(args.dataset.lower() == "casia2"):
        _, val_loader, full_dataset = get_casia2_dataloaders(dataset_root, batch_size=1, val_split=0.1)
    elif(args.dataset.lower() == "df2023"):
        _, val_loader, full_dataset = get_df2023_dataloaders(dataset_root, batch_size=1, val_split=0.1, subset_fraction=0.1)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset}. Expected 'casia2' or 'df2023'.")

    fx_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(), 
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}")

    valid_indices = [i for i, m in enumerate(full_dataset.mask_paths) if m != "blank"]
    total_available = len(valid_indices)
    print(f"Found {total_available} valid (tampered) samples.")

    rng = np.random.default_rng(42)
    selected_indices = rng.choice(valid_indices, size=min(num_samples, total_available), replace=False)
    print(f"Randomly selected {len(selected_indices)} samples for inference (seed={42})")

    for idx in tqdm(selected_indices):
        img_path = full_dataset.img_paths[idx]
        mask_path = full_dataset.mask_paths[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        s_f, s_p, s_h, pred_map = compute_scores(
            img=img,
            mask=mask,
            model_px=model_px,
            vit_model=vit_model,
            mu_ref=mu_ref,
            sigma_ref=sigma_ref,
            transform=fx_transform,
            alpha=alpha,
        )

        vis_path = os.path.join(output_dir, f"infer_{idx:04d}_SF{s_f:.3f}_SP{s_p:.3f}_SH{s_h:.3f}.png")
        visualize_results(
            img=img,
            mask=mask,
            pred_map=pred_map,
            vit_model=vit_model,
            fx_transform=fx_transform,
            s_h=s_h,
            save_path=vis_path,
            vis_mode=vis_mode,
            threshold=threshold,
            cfg=cfg,
            dataset_name=args.dataset.upper()
        )


    print(f"Inference complete. {len(selected_indices)} visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAAS Inference Script")

    parser.add_argument("--dataset-root", type=str, required=True, help="Path to CASIA2 dataset root")
    parser.add_argument("--dataset", type=str,default="casia2", required=True, help="Dataset to use: 'casia2' or 'df2023'")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to the checkpoint folder")
    parser.add_argument("--output-dir", type=str, default="vaas_infer_out", help="Where to save visualizations")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of validation images to visualize")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary Px overlay")
    parser.add_argument("--vis-mode", type=str, default="both", choices=["both", "heatmap", "binary"], help="Visualization mode")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for hybrid S_H combination")

    args = parser.parse_args()
    output_dir = os.path.join(args.checkpoint_dir, args.output_dir)
    run_inference(
        dataset_root=args.dataset_root,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=output_dir,
        num_samples=args.num_samples,
        threshold=args.threshold,
        vis_mode=args.vis_mode,
        alpha=args.alpha,
    )
