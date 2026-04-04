#! /usr/bin/env python3

"""train.py: Trains a new model from config file."""

from pathlib import Path
from time import perf_counter

import utils

with utils.DiscoverSourcePath():
    import Framework
    from Logging import Logger
    from Implementations import Methods as MI
    from Implementations import Datasets as DI


def main(config_path: str = None):
    Framework.setup(config_path=config_path, require_custom_config=True)
    training_instance = MI.get_training_instance(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=Framework.config.TRAINING.LOAD_CHECKPOINT,
    )
    dataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH,
    )
    training_instance.run(dataset)
    Framework.teardown()
    return training_instance


if __name__ == "__main__":
    configs = [
        "fg_bicycle",
        "fg_bonsai",
        "fg_counter",
        "fg_drjohnson",
        "fg_flowers",
        "fg_garden",
        "fg_kitchen",
        "fg_playroom",
        "fg_room",
        "fg_stump",
        "fg_train",
        "fg_treehill",
        "fg_truck",
    ]
    Logger.set_mode(Logger.MODE_VERBOSE)
    rows = ["config_name,training_time_seconds,psnr,ssim,lpips"]
    for config in configs:
        Logger.log_info(f"Starting training for {config}")
        start_time = perf_counter()
        training_instance = main(f"configs/{config}.yaml")
        end_time = perf_counter()
        training_time = end_time - start_time

        # Try to read more accurate training time from timings.txt if available
        try:
            timings_file = training_instance.output_directory / "timings.txt"
            if timings_file.exists():
                with open(timings_file) as f:
                    for line in f:
                        pass  # Get the last line
                # Parse "Time:1234.56"
                if line.startswith("Time:"):
                    training_time = float(line.split(":")[1])
                    Logger.log_info(
                        f"Using accurate training time from timings.txt: {training_time:.2f}s"
                    )
        except Exception as e:
            Logger.log_warning(
                f"Could not read timings.txt for {config}, using perf_counter: {e}"
            )

        training_time_round = round(training_time, 2)

        # Read metrics from output directory
        psnr, ssim, lpips = "N/A", "N/A", "N/A"
        try:
            metrics_file = (
                training_instance.output_directory / "test_30000" / "metrics_8bit.txt"
            )
            if metrics_file.exists():
                with open(metrics_file) as f:
                    for line in f:
                        pass  # Get the last line
                # Parse the metrics line: "PSNR:27.49 SSIM:0.867 LPIPS:0.120"
                metrics_dict = {
                    i[0]: float(i[1]) for i in [j.split(":") for j in line.split(" ")]
                }
                psnr = round(metrics_dict.get("PSNR", 0.0), 2)
                ssim = round(metrics_dict.get("SSIM", 0.0), 3)
                lpips = round(metrics_dict.get("LPIPS", 0.0), 3)
        except Exception as e:
            Logger.log_warning(f"Could not read metrics for {config}: {e}")

        Logger.log_info(
            f"{config}:\t{training_time_round}s, PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}"
        )
        rows.append(f"{config},{training_time_round},{psnr},{ssim},{lpips}")
    csv_path = Path("all_train_times.csv")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    Logger.log_info(f"Wrote results to {csv_path}.")
