import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import os
import torch
import math
from vaas.utils.helpers import save_json, check_CUDA_available 

device= check_CUDA_available()

def visualize_results(
    img,
    mask,
    pred_map,
    vit_model,
    fx_transform,
    s_h,
    save_path="vaas_vis_combined.png",
    threshold=0.5,
    vis_mode="both", 
    cfg=None,
    dataset_name="CASIA2",
):

    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized)


    pred_resized = cv2.resize(pred_map, (224, 224))
    pred_resized = (pred_resized - pred_resized.min()) / (
        pred_resized.max() - pred_resized.min() + 1e-8
    )

    heat_px = cv2.applyColorMap(np.uint8(255 * pred_resized), cv2.COLORMAP_INFERNO)
    heat_px = cv2.cvtColor(heat_px, cv2.COLOR_BGR2RGB)
    overlay_px_heat = cv2.addWeighted(img_np, 0.4, heat_px, 0.8, 0)


    px_binary = (pred_resized > threshold).astype(np.uint8) * 255
    px_color = cv2.applyColorMap(px_binary, cv2.COLORMAP_COOL)
    px_color = cv2.cvtColor(px_color, cv2.COLOR_BGR2RGB)
    overlay_px_bin = cv2.addWeighted(img_np, 0.1, px_color, 0.8, 0)

    img_t = fx_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vit_out = vit_model(img_t, output_attentions=True)
        attn_maps = vit_out.attentions
    attn_mean_layers = torch.stack(
        [a.mean(dim=1)[:, 0, 1:] for a in attn_maps]
    ).mean(dim=0)
    num_patches = attn_mean_layers.shape[-1]
    patch_dim = int(math.sqrt(num_patches))
    attn_values = (
        attn_mean_layers.squeeze().cpu().numpy().reshape(patch_dim, patch_dim)
    )
    attn_values = (attn_values - attn_values.min()) / (
        attn_values.max() - attn_values.min() + 1e-8
    )
    attn_resized = cv2.resize(attn_values, (224, 224))
    heat_fx = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    heat_fx = cv2.cvtColor(heat_fx, cv2.COLOR_BGR2RGB)
    overlay_fx = cv2.addWeighted(img_np, 0.7, heat_fx, 0.5, 0)

    mask_resized = mask.resize((224, 224)).convert("L")
    mask_np = (np.array(mask_resized) > 128).astype(np.uint8) * 255

    if vis_mode == "both":
        titles = [
            "Image",
            "Ground Truth Mask",
            "Binary Heatmap (Px)",
            "Heatmap Overlay (Px)",
            "Attention Map Overlay (Fx)",
            "Hybrid Anomaly Score (S_H)",
        ]
        images = [img_np, mask_np, overlay_px_bin, overlay_px_heat, overlay_fx, None]

    elif vis_mode == "binary":
        titles = [
            "Image",
            "Ground Truth Mask",
            "Binary Heatmap (Px)",
            "Attention Map Overlay (Fx)",
            "Hybrid Anomaly Score (S_H)",
        ]
        images = [img_np, mask_np, overlay_px_bin, overlay_fx, None]

    else: 
        titles = [
            "Image",
            "Ground Truth Mask",
            "Heatmap Overlay (Px)",
            "Attention Map Overlay (Fx)",
            "Hybrid Anomaly Score (S_H)",
        ]
        images = [img_np, mask_np, overlay_px_heat, overlay_fx, None]

    fig, axes = plt.subplots(1, len(titles), figsize=(4 * len(titles), 4))
    plt.subplots_adjust(wspace=0.05)


    for ax, title, im in zip(axes, titles, images):
        ax.axis("off")
        ax.set_title(title, pad=8, fontsize=11, fontweight="semibold")
        if im is not None:
            cmap = "gray" if title == "Ground Truth Mask" else None
            ax.imshow(im, cmap=cmap)


    ax = axes[-1]
    ax.axis("off")
    ax.set_aspect("equal")  
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.2, 1.1)

    base_arc = Wedge(
        center=(0, 0),
        r=1.0,
        theta1=0,
        theta2=180,
        facecolor="lightgray",
        edgecolor="none",
    )
    ax.add_patch(base_arc)

    filled_arc = Wedge(
        center=(0, 0),
        r=1.0,
        theta1=180 * (1 - s_h),
        theta2=180,
        facecolor="#ffa552",
        edgecolor="none",
    )
    ax.add_patch(filled_arc)

    theta = math.radians(180 * (1 - s_h))
    x = 0.8 * math.cos(theta)
    y = 0.8 * math.sin(theta)
    ax.plot([0, x], [0, y], color="black", lw=2)
    ax.scatter([0], [0], color="black", s=40, zorder=5)

    ax.text(
        0,
        -0.35,
        f"{s_h:.3f}",
        ha="center",
        va="center",
        fontsize=14,
        color="#ff7b00",
        fontweight="bold",
    )


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if cfg is not None:
        f1 = cfg.get("best_F1", None)
        iou = cfg.get("best_IoU", None)
        tau = cfg.get("best_threshold", None)
        if f1 is not None and iou is not None and tau is not None:
            text = f"τ*={tau:.2f} | F1={f1:.3f} | IoU={iou:.3f}"         
            fig.suptitle(
            f"{dataset_name} | τ*={tau:.2f} | F1={f1:.3f} | IoU={iou:.3f}",
            fontsize=11,
            color="white",
            fontweight="bold",
            y=0.99,
            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3')
)

    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Visualization saved to {save_path}")