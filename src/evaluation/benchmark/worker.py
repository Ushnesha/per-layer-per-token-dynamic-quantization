"""Internal per-metric worker; process exit releases its vLLM GPU resources."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


def run_metric(name: str, model_path: str) -> dict:
    """Run metrics sequentially in fresh interpreters; keep one final JSON output."""
    with tempfile.TemporaryDirectory(prefix="benchmark-eval-") as temporary:
        output = Path(temporary) / "metric.json"
        log_dir = Path(__file__).resolve().parents[3] / 'logs' / 'evaluation_workers' / os.environ.get('SLURM_JOB_ID', 'local')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{name}.log'
        print(f'Live {name} log: {log_path}', flush=True)
        from .serving import stop_server
        with log_path.open('w') as log:
            process = subprocess.Popen(
                [sys.executable, '-u', '-m', 'src.evaluation.benchmark.worker', name, model_path, str(output)],
                cwd=Path(__file__).resolve().parents[3], stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=9000 if name == 'ttft' else 7200)
            finally:
                stop_server(process)
        if process.returncode != 0 or not output.is_file():
            raise RuntimeError(
                f"{name} worker failed (exit {process.returncode}): "
                f"{log_path.read_text(errors='replace')[-4000:]}"
            )
        with output.open() as handle:
            result = json.load(handle)
        result['worker_log'] = str(log_path)
        return result


def main() -> None:
    name, model_path, destination = sys.argv[1:]
    try:
        if name == "perplexity":
            from .perplexity import evaluate_perplexity
            result = evaluate_perplexity(model_path)
        elif name == "accuracy":
            from .accuracy import evaluate_accuracy
            result = evaluate_accuracy(model_path)
        elif name == "retrieval":
            from .retrieval import evaluate_retrieval
            result = evaluate_retrieval(model_path)
        elif name == "long_context":
            from .long_context import evaluate_long_context
            result = evaluate_long_context(model_path)
        elif name == 'memory':
            from .peak_memory import evaluate_peak_memory
            from .component_memory import checkpoint_components
            measured = evaluate_peak_memory(model_path, include_components=True)
            runtime = measured.pop('component_snapshot', {'status': 'unsupported', 'reason': 'Runtime snapshot unavailable.'})
            checkpoint = checkpoint_components(Path(model_path))
            result = {'status': 'ok' if measured['status'] == runtime['status'] == checkpoint['status'] == 'ok' else 'failed',
                      'peak_memory': measured,
                      'component_memory': {'status': 'ok' if runtime['status'] == checkpoint['status'] == 'ok' else 'failed',
                                           'checkpoint': checkpoint, 'runtime': runtime,
                                           'coverage': 'persistent tensor storage, live physical KV pages, and temporary PyTorch allocation lifetimes; non-PyTorch workspace attribution excluded'}}
        elif name == "peak_memory":
            from .peak_memory import evaluate_peak_memory
            result = evaluate_peak_memory(model_path)
        elif name == "component_memory":
            from .component_memory import evaluate_component_memory
            result = evaluate_component_memory(model_path)
        elif name == "kernel_time":
            from .kernel_time import evaluate_kernel_time
            result = evaluate_kernel_time(model_path)
        elif name == "ttft":
            from .serving import evaluate_ttft
            result = evaluate_ttft(model_path)
        elif name == "max_context":
            from .max_context import evaluate_max_context
            result = evaluate_max_context(model_path)
        else:
            raise ValueError(f"Unknown metric: {name}")
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc()}
    with Path(destination).open("w") as handle:
        json.dump(result, handle, allow_nan=False)


if __name__ == "__main__":
    main()
