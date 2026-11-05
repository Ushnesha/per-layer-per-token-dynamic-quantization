"""Managed loopback vLLM server and token-aware streaming benchmark client."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import http.client
import json
import math
import os
import random
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

from .perplexity import ENGINE_SETTINGS
from .serving_memory import serving_memory

HOST = '127.0.0.1'
PROMPT_LENGTHS = (512, 2048)
CONCURRENCY_LEVELS = (1, 2, 4, 8, 16, 32)
REQUEST_RATES = (1.0, 4.0, 8.0)
MAX_CLIENT_CONCURRENCY = 32
OUTPUT_TOKENS = 128
WARMUP_REQUESTS = 20
MEASURED_REQUESTS = 200
REPETITIONS = 3
REQUEST_TIMEOUT = 120
STARTUP_TIMEOUT = 600
CAMPAIGN_TIMEOUT = 7200


def percentile(values: list[float], percent: float) -> float:
    if not values or not 0 <= percent <= 100 or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Percentiles require valid nonnegative measurements.')
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_ttft(requests: list[dict]) -> dict:
    successful = [request['ttft_ms'] for request in requests if request['status'] == 'ok']
    counts = {'attempted': len(requests), 'successful': len(successful),
              'failed': len(requests) - len(successful)}
    if not successful:
        return {'status': 'failed', **counts, 'reason': 'No complete valid streaming requests.'}
    return {'status': 'ok', **counts, 'unit': 'ms',
            'p50': percentile(successful, 50), 'p95': percentile(successful, 95),
            'mean': sum(successful) / len(successful),
            'percentile_method': 'linear interpolation at (n-1)*p; successful requests only'}


def sse_events(response, clock=time.perf_counter, deadline=None, transport=None):
    """Yield complete SSE messages as received; never buffer the whole response."""
    data = []
    while True:
        if deadline is not None:
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError('Streaming request deadline exceeded.')
            if transport is not None:
                transport.settimeout(remaining)
        raw = response.readline()
        received = clock()
        if not raw:
            if data:
                raise ValueError('Stream ended inside an SSE message.')
            return
        line = raw.decode('utf-8').rstrip('\r\n')
        if line == '':
            if data:
                yield received, '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))


def streamed_request(port: int, model: str, prompt_ids: list[int], output_tokens=OUTPUT_TOKENS,
                     deadline=None, clock=time.perf_counter, scheduled_seconds=None) -> dict:
    body = json.dumps({
        'model': model, 'prompt': prompt_ids, 'max_tokens': output_tokens, 'temperature': 0.0,
        'seed': 0, 'stream': True, 'stream_options': {'include_usage': True},
        'return_token_ids': True, 'stream_interval': 1, 'ignore_eos': True,
        'add_special_tokens': False, 'echo': False,
    }).encode()
    started = clock()
    request_deadline = min(started + REQUEST_TIMEOUT, deadline) if deadline is not None else started + REQUEST_TIMEOUT
    connection = None
    arrivals, chunk_sizes = [], []
    result = {'status': 'failed', 'dispatch_seconds': started, 'input_tokens': len(prompt_ids)}
    if scheduled_seconds is not None:
        result.update(scheduled_dispatch_seconds=scheduled_seconds,
                      client_dispatch_lag_ms=max(0, (started - scheduled_seconds) * 1000))
    try:
        if request_deadline <= started:
            raise TimeoutError('Serving campaign deadline exceeded before dispatch.')
        connection = http.client.HTTPConnection(HOST, port, timeout=request_deadline - started)
        connection.request('POST', '/v1/completions', body=body,
                           headers={'Content-Type': 'application/json'})
        transport = connection.sock
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f'Completion HTTP {response.status}: {response.read(2000).decode(errors="replace")}')
        if 'text/event-stream' not in (response.getheader('Content-Type') or ''):
            raise ValueError('Completion response is not an SSE stream.')
        usage, finished, done = None, False, False
        for received, payload in sse_events(response, clock, request_deadline, transport):
            if payload == '[DONE]':
                done = True
                stream_finished = received
                break
            event = json.loads(payload)
            if event.get('error'):
                raise RuntimeError(f'Server stream error: {event["error"]}')
            if event.get('usage') is not None:
                usage = event['usage']
            for choice in event.get('choices', []):
                if choice.get('index', 0) != 0:
                    raise ValueError('Expected exactly one completion choice.')
                tokens = choice.get('token_ids') or []
                if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
                    raise ValueError('Invalid streamed token IDs.')
                if tokens:
                    arrivals.append(received - started)
                    chunk_sizes.append(len(tokens))
                if choice.get('finish_reason') is not None:
                    finished = True
        if not done or not finished or not arrivals:
            raise ValueError('Stream did not finish with a generated token and [DONE].')
        if usage is None or usage.get('prompt_tokens') != len(prompt_ids):
            raise ValueError('Missing usage or server prompt-token count differs from the workload.')
        if usage.get('completion_tokens') != output_tokens or sum(chunk_sizes) != output_tokens:
            raise ValueError('Streamed/usage output-token counts differ from the fixed workload.')
        result.update(status='ok', ttft_ms=arrivals[0] * 1000, output_tokens=output_tokens,
                      last_token_seconds=started + arrivals[-1], completion_seconds=stream_finished,
                      stream_finished_seconds=stream_finished, client_processing_finished_seconds=clock())
    except (OSError, TimeoutError, ValueError, RuntimeError, http.client.HTTPException) as error:
        result.update(error_type=type(error).__name__, error=str(error), completion_seconds=clock())
    finally:
        if connection is not None:
            connection.close()
    result['token_arrival_offsets_seconds'] = arrivals
    result['token_chunk_sizes'] = chunk_sizes
    return result


def request_batch(port, model, prompt_ids, concurrency, count, deadline):
    # Client executor waiting is excluded from TTFT. Server/network queuing after
    # actual dispatch is included. This is closed-loop load, not a fixed RPS test.
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(streamed_request, port, model, prompt_ids, deadline=deadline)
                   for _ in range(count)]
        return [future.result() for future in futures]


def arrival_offsets(rate, count, seed):
    if rate <= 0 or not math.isfinite(rate) or type(count) is not int or count <= 0:
        raise ValueError('Arrival schedule requires a positive rate and request count.')
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(count - 1):
        offsets.append(offsets[-1] + rng.expovariate(rate))
    return offsets


def request_rate_batch(port, model, prompt_ids, rate, count, seed, deadline):
    offsets = arrival_offsets(rate, count, seed)
    started = time.perf_counter()
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_CLIENT_CONCURRENCY) as executor:
        for offset in offsets:
            scheduled = started + offset
            while time.perf_counter() < min(scheduled, deadline):
                time.sleep(max(0, min(0.1, min(scheduled, deadline) - time.perf_counter())))
            futures.append(executor.submit(streamed_request, port, model, prompt_ids,
                                            deadline=deadline, scheduled_seconds=scheduled))
    requests = [future.result() for future in futures]
    return requests, {'start_seconds': started, 'end_seconds': time.perf_counter()}


def server_command(model_path: str, port: int, served_model: str) -> list[str]:
    command = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', model_path,
            '--host', HOST, '--port', str(port), '--served-model-name', served_model,
            '--dtype', ENGINE_SETTINGS['dtype'], '--max-model-len', str(ENGINE_SETTINGS['max_model_len']),
            '--gpu-memory-utilization', str(ENGINE_SETTINGS['gpu_memory_utilization']),
            '--seed', str(ENGINE_SETTINGS['seed']), '--no-enable-prefix-caching',
            '--stream-interval', '1', '--generation-config', 'vllm', '--disable-uvicorn-access-log']
    if ENGINE_SETTINGS.get('enforce_eager'):
        command.append('--enforce-eager')
    for key, flag in (('max_num_seqs', '--max-num-seqs'),
                      ('max_num_batched_tokens', '--max-num-batched-tokens'),
                      ('safetensors_load_strategy', '--safetensors-load-strategy')):
        if key in ENGINE_SETTINGS:
            command.extend([flag, str(ENGINE_SETTINGS[key])])
    return command


def wait_until_ready(process, port, served_model, timeout=STARTUP_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'vLLM server exited during startup (exit {process.returncode}).')
        connection = http.client.HTTPConnection(HOST, port, timeout=2)
        try:
            connection.request('GET', '/v1/models')
            response = connection.getresponse()
            if response.status == 200:
                models = json.loads(response.read())
                if any(item.get('id') == served_model for item in models.get('data', [])):
                    return
        except (OSError, ValueError, http.client.HTTPException):
            pass
        finally:
            connection.close()
        time.sleep(0.5)
    raise TimeoutError('vLLM server did not become ready within the startup deadline.')


def stop_server(process):
    # The process group was created solely for this benchmark server/workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    # The parent can exit before a stubborn worker. Kill any remaining members
    # of the dedicated server process group as well.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@contextmanager
def managed_server(model_path: str, log_destination=None):
    with socket.socket() as reservation:
        reservation.bind((HOST, 0))
        port = reservation.getsockname()[1]
    served_model = 'aditya-evaluation-' + uuid.uuid4().hex
    command = server_command(model_path, port, served_model)
    environment = os.environ.copy()
    environment.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    environment.pop('VLLM_API_KEY', None)  # This isolated loopback server needs no external credentials.
    with tempfile.TemporaryDirectory(prefix='benchmark-serving-') as temporary:
        log_path = Path(log_destination) if log_destination is not None else Path(temporary) / 'server.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True,
                                       cwd=Path(__file__).resolve().parents[3])
            try:
                try:
                    wait_until_ready(process, port, served_model)
                except Exception as error:
                    log.flush()
                    raise RuntimeError(f'{error}\n{log_path.read_text(errors="replace")[-6000:]}') from error
                yield {'port': port, 'served_model': served_model, 'command': command}
            finally:
                stop_server(process)


def build_prompt(tokenizer, length):
    body = tokenizer('Describe the history and natural features of the city. ', add_special_tokens=False)['input_ids']
    if not body or tokenizer.bos_token_id is None:
        raise ValueError('Cannot construct the fixed serving prompt.')
    return [tokenizer.bos_token_id] + (body * (length // len(body) + 1))[:length - 1]


def evaluate_ttft(model_path: str) -> dict:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        return {'status': 'unsupported', 'reason': str(error)}
    if os.name != 'posix':
        return {'status': 'unsupported', 'reason': 'Managed serving currently targets Linux/Unix process groups.'}
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    trials, groups, rate_trials = [], {}, []
    with serving_memory() as memory, managed_server(model_path) as server:
        deadline = time.perf_counter() + CAMPAIGN_TIMEOUT
        probe = streamed_request(server['port'], server['served_model'], build_prompt(tokenizer, PROMPT_LENGTHS[0]), deadline=deadline)
        if probe['status'] != 'ok':
            raise RuntimeError(f'Streaming protocol preflight failed: {probe.get("error")}')
        for length in PROMPT_LENGTHS:
            prompt = build_prompt(tokenizer, length)
            for concurrency in CONCURRENCY_LEVELS:
                for repetition in range(REPETITIONS):
                    if time.perf_counter() >= deadline:
                        trials.append({'input_tokens': length, 'concurrency': concurrency,
                                       'repetition': repetition, 'status': 'skipped', 'reason': 'Serving campaign deadline reached.'})
                        continue
                    with memory.capture() as memory_capture:
                        warmup = request_batch(server['port'], server['served_model'], prompt, concurrency, WARMUP_REQUESTS, deadline)
                        memory_capture.set_phase('inference')
                        measured_start = time.perf_counter()
                        requests = request_batch(server['port'], server['served_model'], prompt, concurrency, MEASURED_REQUESTS, deadline)
                        measured_end = time.perf_counter()
                    summary = summarize_ttft(requests)
                    trials.append({'input_tokens': length, 'output_tokens': OUTPUT_TOKENS,
                                   'concurrency': concurrency, 'repetition': repetition,
                                   'status': summary['status'], 'ttft': summary,
                                   'arrival_mode': 'closed_loop', 'request_rate': None,
                                   'measurement_window': {'start_seconds': measured_start, 'end_seconds': measured_end},
                                   'memory': memory_capture.result,
                                   'warmup_failed': sum(row['status'] != 'ok' for row in warmup),
                                   'requests': requests})
                    groups.setdefault((length, concurrency), []).extend(requests)
            for rate_index, rate in enumerate(REQUEST_RATES):
                for repetition in range(REPETITIONS):
                    if time.perf_counter() >= deadline:
                        rate_trials.append({'status': 'skipped', 'reason': 'Serving campaign deadline reached.',
                                            'input_tokens': length, 'request_rate': rate, 'repetition': repetition})
                        continue
                    schedule_seed = rate_index * 1000 + repetition
                    with memory.capture() as memory_capture:
                        warmup = request_batch(server['port'], server['served_model'], prompt,
                                               MAX_CLIENT_CONCURRENCY, WARMUP_REQUESTS, deadline)
                        memory_capture.set_phase('inference')
                        requests, window = request_rate_batch(server['port'], server['served_model'], prompt,
                                                             rate, MEASURED_REQUESTS, schedule_seed, deadline)
                    summary = summarize_ttft(requests)
                    rate_trials.append({
                        'input_tokens': length, 'output_tokens': OUTPUT_TOKENS,
                        'concurrency': MAX_CLIENT_CONCURRENCY, 'repetition': repetition,
                        'arrival_mode': 'poisson', 'request_rate': rate, 'schedule_seed': schedule_seed,
                        'measurement_window': window, 'status': summary['status'], 'ttft': summary,
                        'memory': memory_capture.result,
                        'warmup_failed': sum(row['status'] != 'ok' for row in warmup), 'requests': requests,
                        'maximum_client_dispatch_lag_ms': max((row.get('client_dispatch_lag_ms', 0) for row in requests), default=0),
                    })
        server_details = {'host': HOST, **server}
    summaries = [{'input_tokens': length, 'output_tokens': OUTPUT_TOKENS, 'concurrency': concurrency,
                  **summarize_ttft(requests)} for (length, concurrency), requests in sorted(groups.items())]
    return {
        'status': 'ok' if any(row['status'] == 'ok' for row in [*summaries, *rate_trials]) else 'failed',
        'measurement': 'client dispatch to first received chunk containing generated token IDs',
        'unit': 'ms', 'by_workload': summaries, 'trials': trials,
        'request_rate_trials': rate_trials,
        'workload': {'prompt_lengths': list(PROMPT_LENGTHS), 'output_tokens': OUTPUT_TOKENS,
                     'concurrency_levels': list(CONCURRENCY_LEVELS), 'repetitions': REPETITIONS,
                     'request_rates': list(REQUEST_RATES), 'rate_arrival_distribution': 'poisson',
                     'rate_max_client_concurrency': MAX_CLIENT_CONCURRENCY,
                     'warmup_requests_per_trial': WARMUP_REQUESTS, 'measured_requests_per_trial': MEASURED_REQUESTS,
                     'arrival_mode': 'closed_loop', 'seed': 0, 'temperature': 0.0,
                     'ignore_eos': True, 'prefix_caching': False, 'stream_interval': 1},
        'server': server_details, 'engine_settings': ENGINE_SETTINGS,
        'limitations': [
            'Includes loopback connection setup, server queueing, generation, and stream delivery; excludes client executor waiting.',
            'The first token-bearing chunk can contain multiple tokens; chunk sizes and arrival offsets are retained.',
            'Percentiles exclude failed requests; failure counts and records are retained alongside every workload.',
            'Concurrency trials use closed-loop arrivals; separate rate trials use seeded Poisson schedules.',
            'Rate trial dispatch can lag its schedule when the client pool saturates; per-request lag is recorded.',
            'Adaptive backend serving integration and matched-memory comparisons remain pending.',
        ],
    }
