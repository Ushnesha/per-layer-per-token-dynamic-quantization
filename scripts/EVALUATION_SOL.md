# Run evaluation on Sol

From the repository root, submit one command and replace only the checkpoint path:

```bash
sbatch scripts/evaluation.sbatch /absolute/path/to/model
```

The checkpoint must already be downloaded and readable, with `config.json` and
its weights/tokenizer files. The job evaluates the existing checkpoint; it does
not train, quantize, or download a gated model.

For a quantized checkpoint, optionally select the exact FP16 report:

```bash
sbatch scripts/evaluation.sbatch /path/to/quantized-model \
    --reference /absolute/path/to/fp16-evaluation.json
```

`--reference` is the only extra evaluation option. It controls both quality-loss
and compression comparisons. Compatibility is checked before GPU evaluation;
the file must be a finished FP16 report with matching model architecture,
tokenizer, evaluation settings and software versions. Without it, the evaluator
uses compatible reports already in `results/benchmark/`, or the current FP16 run
as its own reference. Older reports lacking memory traces can support quality
and weight comparisons but cannot supply the new KV/runtime ratios.

There is no adaptive-overhead enable flag. The profiler detects declared runtime
instrumentation automatically. FP16/static inference reports `not_applicable`;
it does not time nonexistent quantization work. The adaptive backend remains a
separate future implementation. Quality across precision budgets requires one
run for each actual checkpoint/precision configuration; this command does not
create those checkpoints. An FP16-only result is labelled `reference_only`.

## Runtime prerequisite

Each account needs a readable, compatible conda-pack environment archive. The
job automatically finds `/scratch/$USER/benchmark-evaluation/archives/llm-serv.tar`
or the newest `llm-serv-*.tar` there. The existing archive on Aditya's account
continues to work without repacking. To use a shared archive elsewhere:

```bash
ADITYA_ENV_ARCHIVE=/path/to/readable/llm-serv.tar \
    sbatch --export=ALL scripts/evaluation.sbatch /path/to/model
```

Do not copy Hugging Face tokens or assume another account's private archive is
readable. For a new account, prepare the archive once in an appropriate CPU
allocation using the evaluation requirements:

```bash
module load mamba/latest
mamba create --yes --prefix /scratch/$USER/benchmark-evaluation/envs/llm-serv \
    --override-channels -c conda-forge python=3.12 pip
source activate /scratch/$USER/benchmark-evaluation/envs/llm-serv
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu129 \
    -r requirements-evaluation.txt
python -m pip check
python -m pip install --no-deps --target /scratch/$USER/benchmark-evaluation/pack-tools \
    conda-pack==0.9.2
mkdir -p /scratch/$USER/benchmark-evaluation/archives
PYTHONPATH=/scratch/$USER/benchmark-evaluation/pack-tools python -m conda_pack.cli \
    --prefix /scratch/$USER/benchmark-evaluation/envs/llm-serv \
    --format tar --output /scratch/$USER/benchmark-evaluation/archives/llm-serv.tar
```

Archive preparation can be slow on shared storage; reuse it until dependencies
change. At runtime, the sbatch file unpacks onto node-local disk, relocates the
environment, verifies package locations, and cleans its local copy on exit.
It fails clearly if the archive is unavailable rather than reinstalling packages
or silently using slow shared-storage imports during the benchmark.

## What the command does

Requests one A100, eight CPUs, 48 GB host RAM, and six hours using account
`grp_jzou22`, partition/QOS `public`. It attempts all 17 metric families using
full dataset and serving workloads. Peak memory and component accounting share
one engine. Kernel timing uses a separate eager profiling run;
those timings are not interchangeable with compiled serving measurements.

Start/end/failure notifications use Slurm's submitting-user recipient by default.
To specify an email explicitly, prepend `--mail-user=you@asu.edu` to the sbatch
arguments. Slurm stdout is `evaluation_<job-id>.out` in the submission directory,
so no pre-existing `logs/` folder is needed.

## Outputs and limitations

A single `evaluation_<timestamp>.json` and matching Markdown table report are
written under `results/benchmark/` and updated after completed stages. Live worker
logs are in `logs/evaluation_workers/<job-id>/`; raw profiler artifacts are in
`results/benchmark/profiles/<job-id>/`.

Check `run_status`, `evaluation_checks`, `metric_coverage`, and individual metric
notes. A `partial` run means core checks completed but some measurements are
unavailable; it is not proof all 17 families are fully implemented. Core failures
or unexpected execution failures produce a nonzero exit code.

Memory accounting and remaining limitations:

- Component accounting counts unique GPU weight/buffer/cache/metadata storages,
  retaining unclassified allocator bytes explicitly. During a separate repeat of
  the fixed 512-input/128-output workload, scheduler block IDs establish occupied
  KV pages, including physical page padding. PyTorch allocator history measures
  temporary allocation lifetimes; CUDA peak counters measure worker allocated
  and reserved peaks. Tracing is outside the uninstrumented NVML window and
  serving benchmark, so its synchronization overhead is not reported as latency.
- Temporary allocation measurements combine activations/intermediates/workspaces;
  they do not isolate each kernel's workspace. CUDA-graph retained buffers remain
  persistent allocations. Allocations outside PyTorch are covered only by sampled
  device totals; their component attribution is unavailable.
- Compression reports measured weight payload/shard ratios, occupied KV pages
  plus GPU block/slot metadata, sampled resident runtime growth, and active runtime
  tensors with unused KV pages excluded. These are distinct representations; the
  report states each scope. A self-reference is 1x, not evidence of compression.
  Automatically sized unused KV pools are never used as a KV compression ratio.
- Live KV accounting currently supports the installed vLLM single-cache-group
  runner. FP16/BF16 KV ratios include its physical pages and GPU metadata;
  quantized/hybrid cache layouts require their backend-specific byte-accounting
  adapter, including any scales/zeros. No adaptive backend was added.
- Precision-budget comparisons require comparable runs of derived checkpoints;
  FP16 self-reference alone does not establish a tradeoff. JSON quality points
  retain weight, occupied-KV, and active-runtime memory axes where available,
  along with pairs within the recorded 5% tolerance. Runtime axes refer to the
  fixed diagnostic workload, not memory peaks for every individual quality task.
- Context capacity is reported as a verified lower bound when higher lengths have
  unstable or inconclusive memory-budget results. The external device-growth cap
  remains 85%; the context probe's engine pool uses 80%, leaving five percentage
  points for startup/driver/graph variation under that cap. The policy is identical
  for all models and recorded in the report. Three fresh-process confirmations
  are still required; reaching the model's supported limit is stated explicitly.
- Checkpoints must be supported by the installed vLLM version; this command does
  not implement arbitrary custom quantized weight/cache backends.

Temporary sanity runners, counter-access reproducers, smoke profiles, and their
sbatch files have been removed. Saved historical reports are retained as data.
