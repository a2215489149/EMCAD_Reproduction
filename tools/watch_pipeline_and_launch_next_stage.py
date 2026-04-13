import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
ARTIFACTS_DIR = WORKSPACE / "artifacts"
LOG_DIR = ARTIFACTS_DIR / "logs"
PIPELINE_STATUS_PATH = ARTIFACTS_DIR / "pipeline_status.json"
SUPERVISOR_STATUS_PATH = ARTIFACTS_DIR / "next_stage_supervisor_status.json"
FINAL_STEPS = {
    "stopped_after_clinicdb_mismatch",
    "stopped_after_clinicdb_error",
    "completed",
}


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"next_stage_supervisor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    SUPERVISOR_STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command):
    logging.info("Launching next-stage command: %s", " ".join(command))
    start = time.time()
    completed = subprocess.run(command, cwd=str(ROOT))
    elapsed = time.time() - start
    return completed.returncode, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(WORKSPACE / ".venv38" / "Scripts" / "python.exe"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    log_path = setup_logging()
    python_exe = Path(args.python)
    status = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "log_path": str(log_path),
        "watching": str(PIPELINE_STATUS_PATH),
        "python": str(python_exe),
        "state": "watching",
    }
    write_status(status)

    logging.info("Watching pipeline status at %s", PIPELINE_STATUS_PATH)
    while True:
        if not PIPELINE_STATUS_PATH.exists():
            time.sleep(args.poll_seconds)
            continue

        pipeline = load_json(PIPELINE_STATUS_PATH)
        current_step = pipeline.get("current_step", "")
        if current_step not in FINAL_STEPS:
            time.sleep(args.poll_seconds)
            continue

        status["state"] = "pipeline_finished"
        status["pipeline_final_step"] = current_step
        status["pipeline_completed_at"] = datetime.now().isoformat(timespec="seconds")
        status["pipeline_results"] = pipeline.get("results", {})
        write_status(status)
        logging.info("Detected final pipeline state: %s", current_step)

        clinic_result = pipeline.get("results", {}).get("clinicdb", {})
        synapse_result = pipeline.get("results", {}).get("synapse", {})

        if not clinic_result.get("success_within_tolerance", False):
            command = [str(python_exe), "tools/run_clinicdb_overfit_sanity.py"]
            status["next_action"] = "clinicdb_overfit_sanity"
            status["next_command"] = command
            write_status(status)
            return_code, elapsed = run_command(command)
            status["state"] = "completed"
            status["next_command_returncode"] = return_code
            status["next_command_elapsed_seconds"] = round(elapsed, 2)
            status["completed_at"] = datetime.now().isoformat(timespec="seconds")
            write_status(status)
            return

        if clinic_result.get("success_within_tolerance", False) and synapse_result.get("success_within_tolerance", False):
            status["state"] = "completed"
            status["next_action"] = "awaiting_improvement_selection"
            status["completed_at"] = datetime.now().isoformat(timespec="seconds")
            write_status(status)
            logging.info("Baseline succeeded on both datasets. Waiting for improvement experiment selection.")
            return

        status["state"] = "completed"
        status["next_action"] = "awaiting_manual_followup"
        status["completed_at"] = datetime.now().isoformat(timespec="seconds")
        write_status(status)
        logging.info("No automated next-stage experiment configured for this final state.")
        return


if __name__ == "__main__":
    main()
