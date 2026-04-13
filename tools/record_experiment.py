import argparse
import csv
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--paper-metric", required=True)
    parser.add_argument("--reproduced-metric", required=True)
    parser.add_argument("--improved-metric", default="")
    parser.add_argument("--delta", default="")
    parser.add_argument("--hardware", default="")
    parser.add_argument("--elapsed", default="")
    parser.add_argument("--command", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "task": args.task,
                "dataset": args.dataset,
                "model": args.model,
                "run": args.run,
                "paper_metric": args.paper_metric,
                "reproduced_metric": args.reproduced_metric,
                "improved_metric": args.improved_metric,
                "delta": args.delta,
                "hardware": args.hardware,
                "elapsed": args.elapsed,
                "command": args.command,
                "checkpoint": args.checkpoint,
                "notes": args.notes,
            }
        )


if __name__ == "__main__":
    main()
