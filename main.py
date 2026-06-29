import argparse
import subprocess
import sys


COMMANDS = {
    "train": "src.training.train",
    "train_aux": "src.training.train_aux",
    "evaluate": "src.training.validate",
    "evaluate_aux": "src.training.validate_aux",
    "baseline": "src.evaluation.baseline",
    "ablation": "src.evaluation.ablation",
    "predict": "src.inference.predict",
    "worst_prediction": "src.evaluation.worst_prediction",
    "prediction_plot": "src.visualization.prediction_plot",
    "training_plot": "src.visualization.training_plot",
    "baseline_plot": "src.visualization.baseline_comparison",
    "ablation_plot": "src.visualization.ablation_comparison",
}


def run_module(module_name, extra_args=None):
    command = [
        sys.executable,
        "-m",
        module_name
    ]

    if extra_args:
        command.extend(extra_args)

    try:
        subprocess.run(
            command,
            check=True
        )
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Main entry point for Final Project Deep Learning",
        add_help=False
    )

    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help message and exit"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=COMMANDS.keys(),
        help="Pipeline mode to run"
    )

    args, extra_args = parser.parse_known_args()

    if args.help and args.mode is None:
        parser.print_help()
        return

    if args.mode is None:
        parser.error("the following arguments are required: --mode")

    if args.help:
        extra_args.insert(0, "-h")

    run_module(
        COMMANDS[args.mode],
        extra_args
    )


if __name__ == "__main__":
    main()
