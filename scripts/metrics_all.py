#! /usr/bin/env python3

"""metrics_all.py: Collects PSNR, SSIM, and LPIPS from all FasterGS output directories."""

from argparse import ArgumentParser
from pathlib import Path

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Logging import Logger


def iter_model_directories(models_root: Path) -> list[Path]:
    model_dirs = [path for path in models_root.iterdir() if path.is_dir()]
    model_dirs.sort(key=lambda path: path.name)
    return model_dirs


def find_metrics_file(base_dir: Path) -> Path:
    metrics_files = sorted(base_dir.rglob('metrics_8bit.txt'))
    if not metrics_files:
        raise Framework.InferenceError(
            f'No metrics file found for {base_dir}. Run scripts/inference.py with -m first.'
        )
    test_metrics_files = [path for path in metrics_files if path.parent.name.startswith('test_')]
    if test_metrics_files:
        def test_sort_key(path: Path) -> tuple[int, int, str]:
            suffix = path.parent.name.removeprefix('test_')
            return (int(suffix) if suffix.isdigit() else -1, len(path.parts), str(path))

        return max(test_metrics_files, key=test_sort_key)
    return metrics_files[-1]


def read_metrics(base_dir: Path) -> tuple[float, float, float]:
    metrics_file = find_metrics_file(base_dir)
    with open(metrics_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in reversed(lines):
        if line.startswith('PSNR:'):
            metrics = {name: float(value) for name, value in (item.split(':') for item in line.split())}
            try:
                return metrics['PSNR'], metrics['SSIM'], metrics['LPIPS']
            except KeyError as exc:
                raise Framework.InferenceError(
                    f'Invalid metrics file format in {metrics_file}: missing {exc.args[0]}.'
                ) from exc
    raise Framework.InferenceError(f'No summary metrics line found in {metrics_file}.')


def main(*, models_root: Path, csv_path: Path) -> None:
    model_dirs = iter_model_directories(models_root)
    if not model_dirs:
        raise Framework.InferenceError(f'No model directories found in {models_root}.')
    rows = ['model,psnr,ssim,lpips']
    for model_dir in model_dirs:
        Logger.log_info(f'Collecting metrics for {model_dir.name}.')
        psnr, ssim, lpips = read_metrics(model_dir)
        psnr_round = round(psnr, 2)
        ssim_round = round(ssim, 3)
        lpips_round = round(lpips, 3)
        Logger.log_info(
            f'{model_dir.name}:\tPSNR: {psnr_round}, SSIM: {ssim_round}, LPIPS: {lpips_round}'
        )
        rows.append(f'{model_dir.name},{psnr_round},{ssim_round},{lpips_round}')
    csv_path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    Logger.log_info(f'Wrote results to {csv_path}.')


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='metrics_all.py',
        description='Collects PSNR, SSIM, and LPIPS for all FasterGS output directories.'
    )
    parser.add_argument(
        '--models_root', action='store', dest='models_root',
        default=str(Path(__file__).resolve().parents[1] / 'output' / 'FasterGS'),
        metavar='path/to/output/FasterGS', required=False,
        help='Directory containing FasterGS model output directories.'
    )
    parser.add_argument(
        '--csv', action='store', dest='csv_path', default='all_metrics.csv',
        metavar='all_metrics.csv', required=False,
        help='Output CSV path written in model,psnr,ssim,lpips format.'
    )
    args, _ = parser.parse_known_args()
    Logger.set_mode(Logger.MODE_VERBOSE)
    main(
        models_root=Path(args.models_root),
        csv_path=Path(args.csv_path),
    )
