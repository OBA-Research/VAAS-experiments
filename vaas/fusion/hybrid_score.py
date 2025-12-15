import torch
import torch.nn.functional as F
import numpy as np
import math

from vaas.utils.helpers import check_CUDA_available
device = check_CUDA_available()

def compute_scores(
    img,
    mask,
    model_px,
    vit_model,
    mu_ref,
    sigma_ref,
    transform,
    alpha=0.5,
):
    """
    Compute VAAS-style anomaly scores:
        S_P - Patch-level plausibility from SegFormer Px
        S_F - Global attention-based fidelity from ViT Fx
        S_H - Hybrid score combining both

    Returns:
        S_F (float), S_P (float), S_H (float), pred_sig (H,W) numpy array
    """

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model_px(img_t)
        logits = F.interpolate(
            logits,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        pred_sig = torch.sigmoid(logits).squeeze().cpu().numpy()  
        S_P = 1.0 - float(pred_sig.mean()) 


    with torch.no_grad():
        vit_out = vit_model(img_t, output_attentions=True)
        attn_maps = vit_out.attentions

    if attn_maps is None:
        raise RuntimeError(
            "ViT model did not return attentions in compute_scores. "
            "Ensure output_attentions=True and attention implementation is 'eager'."
        )

    attn_mean_layers = torch.stack(
        [a.mean(dim=1)[:, 0, 1:] for a in attn_maps]
    ).mean(dim=0)  
    attn_values = attn_mean_layers.squeeze().cpu().numpy()
    mu = float(np.mean(attn_values))

    delta = abs(mu - mu_ref)
    S_F = math.exp(-delta / (sigma_ref + 1e-8))

    S_H = alpha * S_F + (1.0 - alpha) * S_P

    return S_F, S_P, S_H, pred_sig