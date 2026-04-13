import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
ARTIFACTS_DIR = WORKSPACE / "artifacts"
STATUS_PATH = ARTIFACTS_DIR / "clinicdb_overfit_status.json"
LOG_DIR = ARTIFACTS_DIR / "logs"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from run_reproduction_pipeline import append_summary_row, list_model_dirs, read_polyp_average_row


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"clinicdb_overfit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_path


def write_status(status):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def ensure_within_workspace(path):
    resolved = path.resolve()
    workspace = WORKSPACE.resolve()
    if not str(resolved).startswith(str(workspace)):
        raise RuntimeError(f"Path escaped workspace: {resolved}")
    return resolved


def reset_dir(path):
    ensure_within_workspace(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_subset(source_root, dataset_root, subset_size):
    train_images = sorted((source_root / "train" / "images").glob("*"))
    train_masks = source_root / "train" / "masks"
    samples = []
    for image_path in train_images:
        mask_path = train_masks / image_path.name
        if mask_path.exists():
            samples.append(image_path.name)
        if len(samples) == subset_size:
            break
    if len(samples) < subset_size:
        raise RuntimeError(f"Only found {len(samples)} paired samples, expected {subset_size}")

    for split in ["train", "val", "test"]:
        for kind in ["images", "masks"]:
            (dataset_root / split / kind).mkdir(parents=True, exist_ok=True)

    for name in samples:
        image_src = source_root / "train" / "images" / name
        mask_src = source_root / "train" / "masks" / name
        for split in ["train", "val", "test"]:
            shutil.copy2(image_src, dataset_root / split / "images" / name)
            shutil.copy2(mask_src, dataset_root / split / "masks" / name)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subset_size": subset_size,
        "source_root": str(source_root),
        "dataset_root": str(dataset_root),
        "samples": samples,
        "note": "Same subset copied into train/val/test for overfit sanity experiment.",
    }
    (dataset_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_command(command, step_name):
    logging.info("Starting %s", step_name)
    logging.info("Command: %s", " ".join(command))
    start = time.time()
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail = []
    assert process.stdout is not None
    for line in process.stdout:
        tail.append(line.rstrip())
        if len(tail) > 200:
            tail = tail[-200:]
        logging.info("[%s] %s", step_name, line.rstrip())
    return_code = process.wait()
    elapsed = time.time() - start
    if return_code != 0:
        raise RuntimeError(f"{step_name} failed with return code {return_code}")
    return elapsed, tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(WORKSPACE / ".venv38" / "Scripts" / "python.exe"))
    parser.add_argument("--source-dataset", default=str(ROOT / "data" / "polyp" / "target" / "ClinicDB"))
    parser.add_argument("--subset-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--test-batchsize", type=int, default=4)
    args = parser.parse_args()

    log_path = setup_logging()
    python_exe = Path(args.python)
    source_root = Path(args.source_dataset)
    dataset_name = f"ClinicDBOverfit{args.subset_size}"
    diagnostic_root = ARTIFACTS_DIR / "diagnostic_datasets"
    dataset_root = diagnostic_root / dataset_name
    results = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "log_path": str(log_path),
        "dataset_name": dataset_name,
        "subset_size": args.subset_size,
        "status": "initializing",
    }
    write_status(results)

    logging.info("Preparing ClinicDB overfit sanity experiment")
    logging.info("Python: %s", python_exe)
    logging.info("Source dataset: %s", source_root)
    logging.info("Target subset dataset: %s", dataset_root)

    try:
        reset_dir(dataset_root)
        manifest = build_subset(source_root, dataset_root, args.subset_size)
        results["subset_manifest"] = manifest
        results["status"] = "subset_ready"
        write_status(results)

        before_dirs = list_model_dirs(f"{dataset_name}_")
        train_command = [
            str(python_exe),
            "train_polyp.py",
            "--dataset_name",
            dataset_name,
            "--encoder",
            "pvt_v2_b2",
            "--epoch",
            str(args.epochs),
            "--num_runs",
            "1",
            "--batchsize",
            str(args.batchsize),
            "--test_batchsize",
            str(args.test_batchsize),
            "--img_size",
            "352",
            "--num_workers",
            "0",
            "--test_num_workers",
            "0",
            "--train_path",
            str(dataset_root / "train"),
            "--test_path",
            str(diagnostic_root),
        ]
        results["status"] = "training"
        write_status(results)
        train_elapsed, train_tail = run_command(train_command, "clinicdb_overfit_train")

        after_dirs = list_model_dirs(f"{dataset_name}_")
        new_dirs = sorted(after_dirs - before_dirs)
        if not new_dirs:
            raise RuntimeError("Overfit training finished but no model_pth directory was created")
        run_id = new_dirs[-1]

        test_command = [
            str(python_exe),
            "test_polyp.py",
            "--run_id",
            run_id,
            "--dataset_name",
            dataset_name,
            "--img_size",
            "352",
            "--test_batchsize",
            "1",
            "--num_workers",
            "0",
            "--test_path",
            str(diagnostic_root),
        ]
        results["status"] = "testing"
        write_status(results)
        test_elapsed, _ = run_command(test_command, "clinicdb_overfit_test")
        avg_row, results_path = read_polyp_average_row(run_id)
        dice_pct = float(avg_row["Dice"]) * 100.0
        iou_pct = float(avg_row["IoU"]) * 100.0
        hd95 = float(avg_row["HD95"])
        checkpoint = ROOT / "model_pth" / run_id / f"{run_id}-best.pth"

        append_summary_row(
            task="diagnostic_overfit",
            dataset="ClinicDB",
            model="pvt_v2_b2_emcad",
            run=run_id,
            paper_metric="",
            reproduced_metric=f"{dice_pct:.4f}",
            improved_metric="",
            delta="",
            hardware="NVIDIA GeForce RTX 4070 SUPER",
            elapsed=f"{train_elapsed + test_elapsed:.2f}s",
            command=(
                f"train_polyp.py --dataset_name {dataset_name} --epoch {args.epochs} --num_runs 1 "
                f"--train_path {dataset_root / 'train'} --test_path {diagnostic_root}"
            ),
            checkpoint=str(checkpoint),
            notes=(
                f"automatic overfit sanity; subset_size={args.subset_size}; same subset copied to train/val/test; "
                f"iou={iou_pct:.4f}; hd95={hd95:.4f}; results={results_path}"
            ),
        )

        results.update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "run_id": run_id,
                "train_elapsed_seconds": round(train_elapsed, 2),
                "test_elapsed_seconds": round(test_elapsed, 2),
                "dice_pct": round(dice_pct, 4),
                "iou_pct": round(iou_pct, 4),
                "hd95": round(hd95, 4),
                "checkpoint": str(checkpoint),
                "results_path": str(results_path),
                "train_output_tail": train_tail[-40:],
            }
        )
        write_status(results)
        logging.info("ClinicDB overfit sanity completed with Dice %.4f", dice_pct)
    except Exception as exc:
        logging.exception("ClinicDB overfit sanity failed")
        results.update(
            {
                "status": "failed",
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "error": str(exc),
            }
        )
        write_status(results)
        raise


if __name__ == "__main__":
    main()
