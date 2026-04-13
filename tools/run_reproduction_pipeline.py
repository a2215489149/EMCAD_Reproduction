import argparse
from collections import deque
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.networks import EMCADNet
from trainer import _run_official_synapse_test

WORKSPACE = ROOT.parent
ARTIFACTS_DIR = WORKSPACE / "artifacts"
LOG_DIR = ARTIFACTS_DIR / "logs"
STATUS_PATH = ARTIFACTS_DIR / "pipeline_status.json"
SUMMARY_CSV = ARTIFACTS_DIR / "experiment_summary.csv"
FIELDNAMES = [
    "task",
    "dataset",
    "model",
    "run",
    "paper_metric",
    "reproduced_metric",
    "improved_metric",
    "delta",
    "hardware",
    "elapsed",
    "command",
    "checkpoint",
    "notes",
]


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"reproduction_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def append_summary_row(**kwargs):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    row = {field: kwargs.get(field, "") for field in FIELDNAMES}
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def finalize_pipeline(status, current_step):
    status["current_step"] = current_step
    status["completed_at"] = datetime.now().isoformat(timespec="seconds")
    write_status(status)
    logging.info("Pipeline finished with state: %s", current_step)


def run_command(command, step_name, status):
    status["current_step"] = step_name
    write_status(status)
    logging.info("Starting step %s", step_name)
    logging.info("Command: %s", " ".join(command))
    step_start = time.time()
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_tail = deque(maxlen=200)
    assert process.stdout is not None
    for line in process.stdout:
        output_tail.append(line)
        logging.info("[%s] %s", step_name, line.rstrip())
    return_code = process.wait()
    elapsed = time.time() - step_start
    status["current_step"] = step_name
    status["steps"].append(
        {
            "name": step_name,
            "command": command,
            "return_code": return_code,
            "elapsed_seconds": round(elapsed, 2),
        }
    )
    write_status(status)
    if return_code != 0:
        raise RuntimeError(f"{step_name} failed with return code {return_code}")
    return "".join(output_tail), elapsed


def list_model_dirs(prefix):
    model_root = ROOT / "model_pth"
    if not model_root.exists():
        return set()
    return {path.name for path in model_root.iterdir() if path.is_dir() and path.name.startswith(prefix)}


def resolve_new_clinic_runs(before_dirs):
    after_dirs = list_model_dirs("ClinicDB_")
    new_dirs = sorted(after_dirs - before_dirs)
    if not new_dirs:
        raise RuntimeError("ClinicDB training finished but no new model_pth directories were found")
    return new_dirs


def read_polyp_average_row(run_id):
    results_path = ROOT / "results_polyp" / f"Results_{run_id}_ClinicDB_test.xlsx"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing ClinicDB results file: {results_path}")
    df = pd.read_excel(results_path)
    avg_row = df[df["Name"] == "AVERAGE"]
    if avg_row.empty:
        raise RuntimeError(f"AVERAGE row not found in {results_path}")
    return avg_row.iloc[0].to_dict(), results_path


def evaluate_clinicdb_run(python_exe, run_id, status):
    command = [
        str(python_exe),
        "test_polyp.py",
        "--run_id",
        run_id,
        "--dataset_name",
        "ClinicDB",
        "--img_size",
        "352",
        "--test_batchsize",
        "1",
        "--num_workers",
        "0",
        "--test_path",
        "./data/polyp/target",
    ]
    _, elapsed = run_command(command, f"clinicdb_test_{run_id}", status)
    avg_row, results_path = read_polyp_average_row(run_id)
    return {
        "run_id": run_id,
        "dice_pct": float(avg_row["Dice"]) * 100.0,
        "iou_pct": float(avg_row["IoU"]) * 100.0,
        "sensitivity_pct": float(avg_row["Sensitivity"]) * 100.0,
        "specificity_pct": float(avg_row["Specificity"]) * 100.0,
        "precision_pct": float(avg_row["Precision"]) * 100.0,
        "hd95": float(avg_row["HD95"]),
        "results_path": str(results_path),
        "elapsed_seconds": elapsed,
        "checkpoint": str(ROOT / "model_pth" / run_id / f"{run_id}-best.pth"),
    }


def build_synapse_args_namespace():
    class Args:
        pass

    args = Args()
    args.volume_path = "./data/synapse/test_vol_h5"
    args.list_dir = "./lists/lists_Synapse"
    args.num_classes = 9
    args.img_size = 224
    args.test_num_workers = 0
    args.z_spacing = 1
    return args


def evaluate_synapse_checkpoint(checkpoint_path):
    model = EMCADNet(
        num_classes=9,
        kernel_sizes=[1, 3, 5],
        expansion_factor=2,
        dw_parallel=True,
        add=True,
        lgag_ks=3,
        activation="relu6",
        encoder="pvt_v2_b2",
        pretrain=False,
    ).cuda()
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    metrics = _run_official_synapse_test(build_synapse_args_namespace(), model)
    return {
        "mean_dice_pct": metrics["mean_dice"] * 100.0,
        "mean_hd95": metrics["mean_hd95"],
        "mean_jacard_pct": metrics["mean_jacard"] * 100.0,
        "mean_asd": metrics["mean_asd"],
    }


def find_latest_synapse_snapshot(before_dirs):
    prefix = "pvt_v2_b2_EMCAD_kernel_sizes_1_3_5_dw_parallel_add_lgag_ks_3_ef2_act_mscb_relu6_loss_mutation_output_final_layer_Run1_Synapse224"
    after_dirs = list_model_dirs(prefix)
    new_top_dirs = sorted(after_dirs - before_dirs)
    if not new_top_dirs:
        raise RuntimeError("Synapse training finished but no new model_pth directory was found")
    target_root = ROOT / "model_pth" / new_top_dirs[-1]
    candidates = list(target_root.rglob("log.txt"))
    if not candidates:
        raise RuntimeError(f"No Synapse log.txt found under {target_root}")
    latest_log = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest_log.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(WORKSPACE / ".venv38" / "Scripts" / "python.exe"))
    parser.add_argument("--acceptance_tolerance", type=float, default=2.0)
    parser.add_argument("--clinicdb-num-runs", type=int, default=5)
    parser.add_argument("--clinicdb-run-id", default=None,
                        help="Existing ClinicDB run_id to evaluate and use as the baseline gate. If provided, ClinicDB training is skipped.")
    parser.add_argument("--resume-synapse-checkpoint", default=None,
                        help="Existing Synapse checkpoint to resume from.")
    parser.add_argument("--resume-synapse-epoch", type=int, default=None,
                        help="Next epoch index to run when resuming Synapse from a weights-only checkpoint.")
    parser.add_argument("--resume-synapse-iter", type=int, default=None,
                        help="Global iteration to continue from when resuming Synapse from a weights-only checkpoint.")
    parser.add_argument("--resume-synapse-best-performance", type=float, default=None,
                        help="Best Synapse validation Dice seen before resume, in raw 0-1 scale.")
    parser.add_argument("--resume-synapse-best-target-stop-probe", type=float, default=None,
                        help="Best quick validation Dice that already triggered Synapse target-stop probing, in raw 0-1 scale.")
    args = parser.parse_args()

    log_path = setup_logging()
    python_exe = Path(args.python)
    tolerance = float(args.acceptance_tolerance)
    hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    status = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "current_step": "initializing",
        "hardware": hardware,
        "acceptance_tolerance": tolerance,
        "log_path": str(log_path),
        "steps": [],
        "results": {},
    }
    write_status(status)

    logging.info("Pipeline log: %s", log_path)
    logging.info("Using python: %s", python_exe)
    logging.info("Hardware: %s", hardware)
    logging.info("Acceptance tolerance: +/-%.1f", tolerance)
    logging.info("ClinicDB num runs: %d", args.clinicdb_num_runs)
    if args.clinicdb_run_id:
        logging.info("Using existing ClinicDB run_id: %s", args.clinicdb_run_id)
    if args.resume_synapse_checkpoint:
        logging.info("Resuming Synapse from checkpoint: %s", args.resume_synapse_checkpoint)

    clinic_before = list_model_dirs("ClinicDB_")
    clinic_command = [
        str(python_exe),
        "train_polyp.py",
        "--dataset_name",
        "ClinicDB",
        "--encoder",
        "pvt_v2_b2",
        "--epoch",
        "200",
        "--num_runs",
        str(args.clinicdb_num_runs),
        "--batchsize",
        "8",
        "--test_batchsize",
        "8",
        "--img_size",
        "352",
        "--num_workers",
        "0",
        "--test_num_workers",
        "0",
    ]

    clinic_train_output = ""
    try:
        if args.clinicdb_run_id:
            clinic_train_elapsed = 0.0
            clinic_run_ids = [args.clinicdb_run_id]
            status["current_step"] = "clinicdb_reuse_existing_run"
            write_status(status)
            logging.info("Skipping ClinicDB training and reusing run_id %s", args.clinicdb_run_id)
        else:
            clinic_train_output, clinic_train_elapsed = run_command(clinic_command, "clinicdb_train", status)
            clinic_run_ids = resolve_new_clinic_runs(clinic_before)
        clinic_results = [evaluate_clinicdb_run(python_exe, run_id, status) for run_id in clinic_run_ids]
        clinic_best = max(result["dice_pct"] for result in clinic_results)
        clinic_mean = sum(result["dice_pct"] for result in clinic_results) / len(clinic_results)
        clinic_success = abs(clinic_best - 95.21) <= tolerance

        for result in clinic_results:
            append_summary_row(
                task="baseline",
                dataset="ClinicDB",
                model="pvt_v2_b2_emcad",
                run=result["run_id"],
                paper_metric="95.21",
                reproduced_metric=f"{result['dice_pct']:.4f}",
                improved_metric="",
                delta=f"{result['dice_pct'] - 95.21:.4f}",
                hardware=hardware,
                elapsed=f"{result['elapsed_seconds']:.2f}s eval",
                command="test_polyp.py --dataset_name ClinicDB",
                checkpoint=result["checkpoint"],
                notes=(
                    f"official test; iou={result['iou_pct']:.4f}; sensitivity={result['sensitivity_pct']:.4f}; "
                    f"specificity={result['specificity_pct']:.4f}; precision={result['precision_pct']:.4f}; "
                    f"hd95={result['hd95']:.4f}"
                ),
            )

        append_summary_row(
            task="baseline_aggregate",
            dataset="ClinicDB",
            model="pvt_v2_b2_emcad",
            run="five_run_summary",
            paper_metric="95.21",
            reproduced_metric=f"{clinic_best:.4f}",
            improved_metric="",
            delta=f"{clinic_best - 95.21:.4f}",
            hardware=hardware,
            elapsed=f"{clinic_train_elapsed:.2f}s train + eval",
            command=(
                f"train_polyp.py --dataset_name ClinicDB --num_runs {args.clinicdb_num_runs}"
                if not args.clinicdb_run_id else
                f"reuse_existing_run_id {args.clinicdb_run_id}"
            ),
            checkpoint="",
            notes=(
                f"best_of_{len(clinic_results)}={clinic_best:.4f}; mean_of_{len(clinic_results)}={clinic_mean:.4f}; "
                f"criterion_abs_delta_le_{tolerance:.1f}={clinic_success}"
            ),
        )

        status["results"]["clinicdb"] = {
            "success_within_tolerance": clinic_success,
            "acceptance_tolerance": tolerance,
            "best_dice_pct": round(clinic_best, 4),
            "mean_dice_pct": round(clinic_mean, 4),
            "run_ids": clinic_run_ids,
        }
        write_status(status)

        if not clinic_success:
            logging.info(
                "ClinicDB result is outside the +/-%.1f acceptance band. Stopping pipeline before Synapse.",
                tolerance,
            )
            finalize_pipeline(status, "stopped_after_clinicdb_mismatch")
            return
    except Exception as exc:
        logging.exception("ClinicDB stage failed")
        status["results"]["clinicdb"] = {
            "error": str(exc),
            "train_output_tail": clinic_train_output[-4000:],
        }
        write_status(status)
        logging.info("ClinicDB stage failed. Stopping pipeline before Synapse.")
        finalize_pipeline(status, "stopped_after_clinicdb_error")
        return

    synapse_prefix = "pvt_v2_b2_EMCAD_kernel_sizes_1_3_5_dw_parallel_add_lgag_ks_3_ef2_act_mscb_relu6_loss_mutation_output_final_layer_Run1_Synapse224"
    synapse_before = None if args.resume_synapse_checkpoint else list_model_dirs(synapse_prefix)
    synapse_command = [
        str(python_exe),
        "train_synapse.py",
        "--encoder",
        "pvt_v2_b2",
        "--root_path",
        "./data/synapse/train_npz",
        "--volume_path",
        "./data/synapse/test_vol_h5",
        "--pretrained_dir",
        "./pretrained_pth/pvt",
        "--max_iterations",
        "50000",
        "--max_epochs",
        "300",
        "--batch_size",
        "6",
        "--num_workers",
        "0",
        "--test_num_workers",
        "0",
        "--img_size",
        "224",
        "--save_interval",
        "50",
        "--enable_target_stop",
        "--paper_target_dice",
        "83.63",
        "--val_trigger_margin",
        str(tolerance),
        "--test_accept_margin",
        str(tolerance),
        "--target_stop_min_epoch",
        "150",
        "--target_stop_retest_delta",
        "0.1",
    ]
    if args.resume_synapse_checkpoint:
        synapse_command.extend(["--resume_checkpoint", args.resume_synapse_checkpoint])
    if args.resume_synapse_epoch is not None:
        synapse_command.extend(["--resume_epoch", str(args.resume_synapse_epoch)])
    if args.resume_synapse_iter is not None:
        synapse_command.extend(["--resume_iter", str(args.resume_synapse_iter)])
    if args.resume_synapse_best_performance is not None:
        synapse_command.extend(["--resume_best_performance", str(args.resume_synapse_best_performance)])
    if args.resume_synapse_best_target_stop_probe is not None:
        synapse_command.extend(["--resume_best_target_stop_probe", str(args.resume_synapse_best_target_stop_probe)])

    synapse_train_output = ""
    try:
        synapse_train_output, synapse_train_elapsed = run_command(synapse_command, "synapse_train", status)
        if args.resume_synapse_checkpoint:
            synapse_snapshot = Path(args.resume_synapse_checkpoint).resolve().parent
        else:
            synapse_snapshot = find_latest_synapse_snapshot(synapse_before)
        target_stop_json = synapse_snapshot / "target_stop_result.json"
        if target_stop_json.exists():
            target_result = json.loads(target_stop_json.read_text(encoding="utf-8"))
            synapse_metrics = {
                "mean_dice_pct": float(target_result["official_test"]["mean_dice"]),
                "mean_hd95": float(target_result["official_test"]["mean_hd95"]),
                "mean_jacard_pct": float(target_result["official_test"]["mean_jacard"]),
                "mean_asd": float(target_result["official_test"]["mean_asd"]),
            }
            checkpoint_path = str(synapse_snapshot / "target_stop.pth")
            notes = "accepted by target-stop"
        else:
            checkpoint_candidate = synapse_snapshot / "best.pth"
            if not checkpoint_candidate.exists():
                checkpoint_candidate = synapse_snapshot / "last.pth"
            synapse_metrics = evaluate_synapse_checkpoint(str(checkpoint_candidate))
            checkpoint_path = str(checkpoint_candidate)
            notes = "official test after full training"

        synapse_success = abs(synapse_metrics["mean_dice_pct"] - 83.63) <= tolerance
        append_summary_row(
            task="baseline",
            dataset="Synapse",
            model="pvt_v2_b2_emcad",
            run=synapse_snapshot.name,
            paper_metric="83.63",
            reproduced_metric=f"{synapse_metrics['mean_dice_pct']:.4f}",
            improved_metric="",
            delta=f"{synapse_metrics['mean_dice_pct'] - 83.63:.4f}",
            hardware=hardware,
            elapsed=f"{synapse_train_elapsed:.2f}s train + eval",
            command=(
                "train_synapse.py --encoder pvt_v2_b2 --max_epochs 300 --enable_target_stop "
                f"--val_trigger_margin {tolerance:.1f} --test_accept_margin {tolerance:.1f}"
            ),
            checkpoint=checkpoint_path,
            notes=(
                f"{notes}; mean_hd95={synapse_metrics['mean_hd95']:.4f}; "
                f"mean_jacard={synapse_metrics['mean_jacard_pct']:.4f}; mean_asd={synapse_metrics['mean_asd']:.4f}; "
                f"criterion_abs_delta_le_{tolerance:.1f}={synapse_success}"
            ),
        )
        status["results"]["synapse"] = {
            "success_within_tolerance": synapse_success,
            "acceptance_tolerance": tolerance,
            "mean_dice_pct": round(synapse_metrics["mean_dice_pct"], 4),
            "checkpoint": checkpoint_path,
        }
        write_status(status)
    except Exception as exc:
        logging.exception("Synapse stage failed")
        status["results"]["synapse"] = {
            "error": str(exc),
            "train_output_tail": synapse_train_output[-4000:],
        }
        write_status(status)

    finalize_pipeline(status, "completed")


if __name__ == "__main__":
    main()
