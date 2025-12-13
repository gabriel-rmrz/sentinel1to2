# src/sentinel1to2/tools/parse_args.py

from pathlib import Path

def parse_args(argparse):
    parser = argparse.ArgumentParser(description="Sentinel 1 to Sentinel 2 images translator")

    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Dataset YAML config (paths, target, preprocessing).",
    )

    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=None,
        help="Experiment YAML config (model, training, gan, loss).",
    )

    # keep backwards compatible -c
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Single merged YAML config (legacy).",
    )

    parser.add_argument(
        "-a", "--all_steps",
        action="store_true",
        help="Run all steps preceding the selected step.",
    )

    parser.add_argument(
        "-s",
        "--step",
        choices=["preprocessing", "training", "evaluation", "inference", "performance"],
        required=True,
        help="The step to run.",
    )

    return parser.parse_args()

