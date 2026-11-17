"""
Per-unit sensitivity profiling for mixed-precision allocation.

Two signals per allocation unit (see src.adaptive.units) and bit-width:

    cheap      One forward pass of the FP16 model with hooks on every Linear.
               For each unit, the output error its weights would cause if quantized:
                   abs = sum ||X (W - Q(W))^T||^2      rel = abs / sum ||X W^T||^2
               All units are scored in the same pass (proposal, Idea 1).
    reference  Quantize one unit at a time, run the model, and measure the
               end-to-end KL(FP16 || quantized) of the next-token distribution and
               the increase in calibration NLL. One pass per unit; the ground truth
               the cheap signal is checked against.

Calibration blocks are the shared pile-val blocks (WikiText stays held out),
each prefixed with BOS like the PPL windows.

Usage:
    python -m src.adaptive.sensitivity [--calib-samples 32] [--ref-samples 8] [--bits 4 8]
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

from src.adaptive.units import Unit, build_units, fake_quantize
from src.config.settings import (
    DEFAULT_MODEL_PATH, CALIBRATION_SEQ_LEN, CALIBRATION_SOURCE, GROUP_SIZE, SEED,
    SENSITIVITY_DIR, SENSITIVITY_CALIB_SAMPLES, SENSITIVITY_REF_SAMPLES,
)
from src.utils.data_loader import get_calibration_data
from src.utils.model_loader import ModelLoader
from src.utils.logger import get_logger

logger = get_logger()


def calibration_batches(tokenizer, num_samples: int, seq_len: int, source: str) -> List[torch.Tensor]:
    """Shared calibration blocks as BOS + (seq_len - 1) tokens, one tensor per block."""
    blocks = get_calibration_data(tokenizer, num_samples=num_samples, seq_len=seq_len, source=source, seed=SEED)
    bos = tokenizer.bos_token_id
    return [torch.tensor([[bos] + b["input_ids"][: seq_len - 1]]) for b in blocks]


@torch.no_grad()
def cheap_sensitivity(model, units: List[Unit], batches: List[torch.Tensor], bits: int) -> Dict[str, dict]:
    """Score every unit's local output error at `bits` in one pass of the FP16 model."""
    named = dict(model.named_modules())
    device = next(model.parameters()).device
    owner = {m: u.name for u in units for m in u.modules}
    err = {u.name: 0.0 for u in units}
    ref = {u.name: 0.0 for u in units}

    # Weight perturbation per Linear, computed once
    delta = {m: named[m].weight - fake_quantize(named[m].weight, bits) for m in owner}

    def hook(name):
        def fn(module, inputs, output):
            e = F.linear(inputs[0], delta[name])
            err[owner[name]] += e.float().pow(2).sum().item()
            ref[owner[name]] += output.float().pow(2).sum().item()
        return fn

    handles = [named[m].register_forward_hook(hook(m)) for m in owner]
    try:
        for ids in tqdm(batches, desc=f"Cheap sensitivity W{bits}"):
            model(input_ids=ids.to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    del delta
    torch.cuda.empty_cache()
    return {n: {"abs": err[n], "rel": err[n] / max(ref[n], 1e-12)} for n in err}


@torch.no_grad()
def _logprobs(model, ids: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(model(input_ids=ids, use_cache=False).logits[0].float(), dim=-1)


@torch.no_grad()
def reference_sensitivity(model, units: List[Unit], batches: List[torch.Tensor], bits: int) -> Dict[str, dict]:
    """End-to-end KL(FP16 || unit quantized) and calibration NLL increase, one unit at a time."""
    named = dict(model.named_modules())
    device = next(model.parameters()).device
    batches = [b.to(device) for b in batches]

    # FP16 reference distributions (fp32 log-probs kept on GPU: ~1 GB per 2048-token block)
    ref_logp = [_logprobs(model, ids) for ids in batches]
    ref_nll = sum(F.nll_loss(lp[:-1], ids[0, 1:], reduction="sum").item() for lp, ids in zip(ref_logp, batches))
    n_tokens = sum(ids.shape[1] - 1 for ids in batches)

    scores = {}
    for unit in tqdm(units, desc=f"Reference sensitivity W{bits}"):
        backup = {m: named[m].weight.data.clone() for m in unit.modules}
        for m in unit.modules:
            named[m].weight.data = fake_quantize(backup[m], bits)
        kl, nll = 0.0, 0.0
        for ids, lp_ref in zip(batches, ref_logp):
            lp = _logprobs(model, ids)
            kl += (lp_ref.exp() * (lp_ref - lp)).sum().item()
            nll += F.nll_loss(lp[:-1], ids[0, 1:], reduction="sum").item()
            del lp
        for m in unit.modules:
            named[m].weight.data = backup[m]
        scores[unit.name] = {
            "kl": kl / sum(ids.shape[1] for ids in batches),
            "delta_nll": (nll - ref_nll) / n_tokens,
        }

    del ref_logp
    torch.cuda.empty_cache()
    return scores


def run(args):
    model_name = args.model or DEFAULT_MODEL_PATH
    logger.info(f"Loading {model_name} (FP16) for sensitivity profiling...")
    model, tokenizer = ModelLoader("fp16", device_map="cuda:0").load(model_name)
    model.eval()

    units = build_units(model, include_lm_head=not args.no_lm_head)
    logger.info(f"{len(units)} allocation units, {sum(u.params for u in units) / 1e9:.2f}B params")

    n = max(args.calib_samples, args.ref_samples)
    batches = calibration_batches(tokenizer, n, args.calib_seq_len, args.calib_source)

    cheap, reference = {}, {}
    started = time.time()
    for bits in args.bits:
        cheap[bits] = cheap_sensitivity(model, units, batches[: args.calib_samples], bits)
    cheap_secs = time.time() - started
    if not args.skip_reference:
        for bits in args.bits:
            reference[bits] = reference_sensitivity(model, units, batches[: args.ref_samples], bits)
    ref_secs = time.time() - started - cheap_secs

    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "seed": SEED,
            "calibration": {"source": args.calib_source, "seq_len": args.calib_seq_len,
                            "cheap_samples": args.calib_samples, "ref_samples": args.ref_samples,
                            "bos_prefixed": True},
            "fake_quant": {"scheme": "RTN symmetric (GPTQ sym grid)", "group_size": GROUP_SIZE},
            "bits": args.bits,
        },
        "runtime_s": {"cheap": round(cheap_secs, 1), "reference": round(ref_secs, 1)},
        "units": [
            {
                "name": u.name, "kind": u.kind, "layer": u.layer, "modules": u.modules, "params": u.params,
                "cheap": {str(b): cheap[b][u.name] for b in cheap},
                "reference": {str(b): reference[b][u.name] for b in reference},
            }
            for u in units
        ],
    }

    # How well does the one-pass signal rank units vs the end-to-end reference?
    if reference:
        agreement = {}
        for b in args.bits:
            kl = [reference[b][u.name]["kl"] for u in units]
            agreement[str(b)] = {
                metric: round(float(spearmanr([cheap[b][u.name][metric] for u in units], kl).statistic), 4)
                for metric in ("abs", "rel")
            }
        results["spearman_cheap_vs_kl"] = agreement
        logger.info(f"Spearman(cheap, reference KL): {agreement}")

    out = Path(args.out) if args.out else SENSITIVITY_DIR / "llama3-8b-instruct.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Sensitivity profile saved to {out} (cheap {cheap_secs:.0f}s, reference {ref_secs:.0f}s)")

    if reference:
        b = args.bits[0]
        top = sorted(units, key=lambda u: reference[b][u.name]["kl"], reverse=True)[:10]
        logger.info(f"Most sensitive units at W{b} (reference KL):")
        for u in top:
            logger.info(f"  {u.name:22} KL={reference[b][u.name]['kl']:.5f}  rel={cheap[b][u.name]['rel']:.5f}")
    return results


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str, default=None, help="FP16 model path (default: Llama-3-8B-Instruct)")
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 8], choices=[4, 8])
    parser.add_argument("--calib-samples", type=int, default=SENSITIVITY_CALIB_SAMPLES,
                        help="Calibration blocks for the cheap one-pass signal")
    parser.add_argument("--ref-samples", type=int, default=SENSITIVITY_REF_SAMPLES,
                        help="Calibration blocks for the per-unit KL reference")
    parser.add_argument("--calib-seq-len", type=int, default=CALIBRATION_SEQ_LEN)
    parser.add_argument("--calib-source", type=str, default=CALIBRATION_SOURCE, choices=["pile", "wikitext"])
    parser.add_argument("--skip-reference", action="store_true", help="Only compute the cheap signal")
    parser.add_argument("--no-lm-head", action="store_true", help="Exclude lm_head from the units")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-unit sensitivity profiling")
    add_arguments(parser)
    run(parser.parse_args())
