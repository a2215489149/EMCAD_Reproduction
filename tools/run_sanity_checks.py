import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.networks import EMCADNet
from utils.dataset_synapse import RandomGenerator, Synapse_dataset


def count_files(path):
    return len(list(path.glob("*")))


def build_report():
    report = {
        "workspace_root": str(ROOT),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "checks": {},
    }

    synapse_root = ROOT / "data" / "synapse"
    clinic_root = ROOT / "data" / "polyp" / "target" / "ClinicDB"
    pretrained_dir = ROOT / "pretrained_pth" / "pvt"

    report["checks"]["paths"] = {
        "synapse_train_npz_exists": (synapse_root / "train_npz").exists(),
        "synapse_test_h5_exists": (synapse_root / "test_vol_h5").exists(),
        "clinicdb_exists": clinic_root.exists(),
        "pvt_b2_exists": (pretrained_dir / "pvt_v2_b2.pth").exists(),
    }

    report["checks"]["counts"] = {
        "synapse_train_npz": count_files(synapse_root / "train_npz"),
        "synapse_test_vol_h5": count_files(synapse_root / "test_vol_h5"),
        "clinicdb_train_images": count_files(clinic_root / "train" / "images"),
        "clinicdb_train_masks": count_files(clinic_root / "train" / "masks"),
        "clinicdb_val_images": count_files(clinic_root / "val" / "images"),
        "clinicdb_val_masks": count_files(clinic_root / "val" / "masks"),
        "clinicdb_test_images": count_files(clinic_root / "test" / "images"),
        "clinicdb_test_masks": count_files(clinic_root / "test" / "masks"),
    }

    db = Synapse_dataset(
        base_dir=str(synapse_root / "train_npz"),
        list_dir=str(ROOT / "lists" / "lists_Synapse"),
        split="train",
        nclass=9,
        transform=transforms.Compose([RandomGenerator((224, 224))]),
    )
    for idx in range(min(len(db), 200)):
        sample = db[idx]
        uniq = np.unique(sample["label"].numpy()).tolist()
        if len(uniq) > 1 or uniq[0] != 0:
            report["checks"]["synapse_sample"] = {
                "index": idx,
                "case_name": sample["case_name"],
                "image_shape": list(sample["image"].shape),
                "label_shape": list(sample["label"].shape),
                "unique_labels": uniq,
            }
            break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_synapse = EMCADNet(
        num_classes=9,
        encoder="pvt_v2_b2",
        pretrain=True,
        pretrained_dir=str(pretrained_dir),
    ).to(device)
    model_polyp = EMCADNet(
        num_classes=1,
        encoder="pvt_v2_b2",
        pretrain=True,
        pretrained_dir=str(pretrained_dir),
    ).to(device)
    model_synapse.eval()
    model_polyp.eval()

    with torch.no_grad():
        synapse_out = model_synapse(torch.randn(1, 1, 224, 224, device=device))
        polyp_out = model_polyp(torch.randn(1, 3, 352, 352, device=device))

    report["checks"]["forward"] = {
        "synapse_output_shapes": [list(t.shape) for t in synapse_out],
        "polyp_output_shapes": [list(t.shape) for t in polyp_out],
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    report = build_report()
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
