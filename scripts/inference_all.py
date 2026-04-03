#! /usr/bin/env python3

"""inference_all.py: Benchmarks all FasterGS output directories and writes aggregate FPS results."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import torch

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Logging import Logger
    from Implementations import Methods as MI
    from Implementations import Datasets as DI


NUM_ITERATIONS = 200
NUM_WARMUP_ITERATIONS = 200


def benchmark_model(*, base_dir: Path, checkpoint_name: str, num_warmup_iterations: int, num_iterations: int) -> tuple[float, float]:
    """Run a benchmark pass for a single model and return (fps, ms_per_image)."""
    Framework.setup(config_path=str(base_dir / 'training_config.yaml'), require_custom_config=True)
    dataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH
    )
    model = MI.get_model(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(base_dir / 'checkpoints' / checkpoint_name),
    ).eval()
    renderer = MI.get_renderer(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        model=model
    )
    if len(dataset.test()) == 0 and len(dataset.train()) == 0:
        raise Framework.InferenceError(f'No images found for benchmarking in {base_dir}.')
    for _ in Logger.log_progress(range(num_warmup_iterations), leave=False, leave=False, desc='Warming Up'):
        for view in dataset:
            renderer.render_image(view, benchmark=True)
    num_test_images = len(dataset)
    torch.cuda.synchronize()
    start_time = perf_counter()
    for _ in Logger.log_progress(range(num_iterations), leave=False, desc='Benchmarking Performance'):
        for view in dataset:
            renderer.render_image(view, benchmark=True)
    torch.cuda.synchronize()
    end_time = perf_counter()
    total_time = end_time - start_time
    total_num_images = num_iterations * num_test_images
    avg_fps = total_num_images / total_time
    avg_ms_per_image = (total_time * 1000) / total_num_images
    Framework.teardown()
    return avg_fps, avg_ms_per_image


def iter_model_directories(models_root: Path) -> list[Path]:
    model_dirs = [path for path in models_root.iterdir() if path.is_dir()]
    model_dirs.sort(key=lambda path: path.name)
    return model_dirs


def main(*, models_root: Path, checkpoint_name: str, csv_path: Path) -> None:
    model_dirs = iter_model_directories(models_root)
    if not model_dirs:
        raise Framework.InferenceError(f'No model directories found in {models_root}.')
    rows = ['model_name,FPS,MS']
    for model_dir in model_dirs:
        Logger.log_info(f'Benchmark {model_dir.name} ({NUM_ITERATIONS} iterations).')
        fps, ms = benchmark_model(base_dir=model_dir, checkpoint_name=checkpoint_name, num_warmup_iterations=NUM_WARMUP_ITERATIONS, num_iterations=NUM_ITERATIONS)
        fps_round = round(fps)
        ms_round = round(ms, 2)
        Logger.log_info(f"{model_dir.name}:\t{fps_round:,} ({ms_round})")
        rows.append(f'{model_dir.name},{fps_round},{ms_round}')
    csv_path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    Logger.log_info(f'Wrote results to {csv_path}.')


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='inference_all.py',
        description='Runs warmup and measured benchmark passes for all FasterGS output directories.'
    )
    parser.add_argument(
        '--models_root', action='store', dest='models_root',
        default=str(Path(__file__).resolve().parents[1] / 'output' / 'FasterGS'),
        metavar='path/to/output/FasterGS', required=False,
        help='Directory containing FasterGS model output directories.'
    )
    parser.add_argument(
        '--checkpoint', action='store', dest='checkpoint_name', default='final.pt',
        metavar='checkpoint_name', required=False,
        help='The name of the checkpoint file to use for inference.'
    )
    parser.add_argument(
        '--csv', action='store', dest='csv_path', default='all_fps.csv',
        metavar='all_fps.csv', required=False,
        help='Output CSV path written in model_name,FPS,MS format.'
    )
    args, _ = parser.parse_known_args()
    Logger.set_mode(Logger.MODE_VERBOSE)
    main(
        models_root=Path(args.models_root),
        checkpoint_name=args.checkpoint_name,
        csv_path=Path(args.csv_path),
    )
