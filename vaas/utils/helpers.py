import json
import os
import torch


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def check_CUDA_available():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda")
        print(f" - GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    return device