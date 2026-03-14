import os
import random
import numpy as np
import torch
import importlib.util
from pathlib import Path
import json
import pandas as pd


def set_deterministic(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def in_kaggle() -> bool:
    if importlib.util.find_spec("kaggle_secrets") is not None:
        return True
    if (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("KAGGLE_URL_BASE")
        or os.environ.get("KAGGLE_USER_SECRETS_TOKEN")
    ):
        return True
    if Path("/kaggle/input").exists() or Path("/kaggle/working").exists():
        return True
    return False


def in_colab() -> bool:
    try:
        import google.colab

        return True
    except ImportError:
        return False


def to_jsonable(x):
    if isinstance(x, pd.DataFrame):
        return x.to_dict(orient="records")
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if hasattr(x, "isoformat"):
        return x.isoformat()
    return x


def walk(obj):
    if isinstance(obj, dict):
        return {k: walk(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [walk(v) for v in obj]
    return to_jsonable(obj)


def delete_zip(zip_path: str = "/kaggle/working/outputs.zip"):
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Deleted: {zip_path}")
    else:
        print(f"Not found: {zip_path}")


def zip_res(root: str = "/kaggle/working", zip_name: str = "outputs.zip"):
    import zipfile

    zip_path = os.path.join(root, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for r, _, files in os.walk(root):
            for f in files:
                full = os.path.join(r, f)
                if os.path.abspath(full) == os.path.abspath(zip_path):
                    continue
                z.write(full, arcname=os.path.relpath(full, root))
