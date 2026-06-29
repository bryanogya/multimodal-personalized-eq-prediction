import argparse
import subprocess
import sys


COMMANDS = {
    "train": "src.training.train",
    "train_aux": "src.training.train_aux",
    "evaluate": "src.evaluation.test",
    "baseline": "src.evaluation.baseline",
    "ablation": "src.evaluation.ablation",
    "predict": "src.inference.predict",
    "worst_prediction": "src.evaluation.worst_prediction",
    "prediction_plot": "src.visualization.prediction_plot",
    "training_plot": "src.visualization.training_plot",
    "baseline_plot": "src.visualization.baseline_plot",
    "ablation_plot": "src.visualization.ablation_plot",
}


def run_module(module_name, extra_args=None):
    command = [
        sys.executable,
        "-m",
        module_name
    ]

    if extra_args:
        command.extend(extra_args)

    subprocess.run(
        command,
        check=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Main entry point for Final Project Deep Learning"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=COMMANDS.keys(),
        help="Pipeline mode to run"
    )

    args = parser.parse_args()    
    extra_args = sys.argv[3:]

    run_module(
        COMMANDS[args.mode],
        extra_args
    )


if __name__ == "__main__":
    main()