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
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from safetensors import safe_open
from scipy.stats import spearmanr
from tqdm import tqdm

from src.adaptive.units import Unit, build_units, fake_quantize
from src.config.settings import (
    DEFAULT_MODEL_PATH, CALIBRATION_SEQ_LEN, CALIBRATION_SOURCE, GROUP_SIZE, GPTQ_OUTPUT_BASE, SEED,
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


class GPTQWeights:
    """Dequantized weights of a saved GPTQ checkpoint, for swapping one unit at a time into the FP16 model.

    Modules the checkpoint left unquantized (lm_head in the static baselines) fall
    back to RTN, which is how the mixed quantizer deploys lm_head.
    """

    def __init__(self, model_path: str | Path, bits: int):
        from src.adaptive.mixed_quantizer import dequantize_gptq
        self._dequantize = dequantize_gptq
        self.path, self.bits = Path(model_path), bits
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            with open(index) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            self.weight_map = {k: "model.safetensors" for k in safe_open_keys(self.path / "model.safetensors")}

    def __call__(self, module: str, weight: torch.Tensor, bits: int) -> torch.Tensor:
        assert bits == self.bits, f"{self.path} holds {self.bits}-bit weights, asked for {bits}"
        if f"{module}.qweight" not in self.weight_map:
            return fake_quantize(weight, bits)
        tensors = {}
        for k in ("qweight", "qzeros", "scales", "g_idx"):
            with safe_open(str(self.path / self.weight_map[f"{module}.{k}"]), "pt", device=str(weight.device)) as f:
                tensors[k] = f.get_tensor(f"{module}.{k}")
        w = self._dequantize(**tensors, bits=bits)
        assert w.shape == weight.shape, f"{module}: {tuple(w.shape)} != {tuple(weight.shape)}"
        return w.to(weight.dtype)


def safe_open_keys(path: Path) -> List[str]:
    with safe_open(str(path), "pt") as f:
        return list(f.keys())


@torch.no_grad()
def reference_sensitivity(model, units: List[Unit], batches: List[torch.Tensor], bits: int,
                          quantized_weight: Callable[[str, torch.Tensor, int], torch.Tensor] | None = None,
                          ) -> Dict[str, dict]:
    """End-to-end KL(FP16 || unit quantized) and calibration NLL increase, one unit at a time.

    `quantized_weight(module, weight, bits)` gives the quantized weight to swap in:
    RTN by default, or a GPTQ checkpoint's dequantized weights (GPTQWeights).
    """
    quantized_weight = quantized_weight or (lambda module, w, b: fake_quantize(w, b))
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
            named[m].weight.data = quantized_weight(m, backup[m], bits)
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


def run_gptq_reference(args):
    """Add a "gptq" signal to an existing profile: the reference KL with each unit's
    weights taken from the static GPTQ W4 / W8 checkpoints instead of RTN, so the
    cost reflects GPTQ's error compensation (which the mixed checkpoints use)."""
    out = Path(args.out) if args.out else SENSITIVITY_DIR / "llama3-8b-instruct.json"
    with open(out) as f:
        profile = json.load(f)

    model_name = args.model or DEFAULT_MODEL_PATH
    logger.info(f"Loading {model_name} (FP16) for GPTQ-weight reference sensitivity...")
    model, tokenizer = ModelLoader("fp16", device_map="cuda:0").load(model_name)
    model.eval()
    units = build_units(model, include_lm_head=not args.no_lm_head)
    batches = calibration_batches(tokenizer, args.ref_samples, args.calib_seq_len, args.calib_source)

    started = time.time()
    gptq = {}
    for bits in args.bits:
        source = GPTQWeights(GPTQ_OUTPUT_BASE / f"llama3-8b-instruct-w{bits}a16", bits)
        gptq[bits] = reference_sensitivity(model, units, batches, bits, quantized_weight=source)

    by_name = {u["name"]: u for u in profile["units"]}
    for name in by_name:
        by_name[name]["gptq"] = {str(b): gptq[b][name] for b in gptq}
    profile["protocol"]["gptq_reference"] = {
        "checkpoints": {str(b): str(GPTQ_OUTPUT_BASE / f"llama3-8b-instruct-w{b}a16") for b in gptq},
        "ref_samples": args.ref_samples, "lm_head": "RTN (as deployed)",
    }
    profile["runtime_s"]["gptq_reference"] = round(time.time() - started, 1)
    profile["spearman_gptq_vs_rtn_kl"] = {
        str(b): round(float(spearmanr([by_name[n]["gptq"][str(b)]["kl"] for n in by_name],
                                      [by_name[n]["reference"][str(b)]["kl"] for n in by_name]).statistic), 4)
        for b in gptq if all(str(b) in u.get("reference", {}) for u in by_name.values())
    }
    with open(out, "w") as f:
        json.dump(profile, f, indent=2)
    logger.info(f"GPTQ reference added to {out}; Spearman(GPTQ KL, RTN KL): {profile['spearman_gptq_vs_rtn_kl']}")

    b = args.bits[0]
    top = sorted(by_name.values(), key=lambda u: u["gptq"][str(b)]["kl"], reverse=True)[:10]
    logger.info(f"Most sensitive units at W{b} (GPTQ weights):")
    for u in top:
        logger.info(f"  {u['name']:22} GPTQ KL={u['gptq'][str(b)]['kl']:.5f}  RTN KL={u['reference'][str(b)]['kl']:.5f}")
    return profile


def run(args):
    if getattr(args, "gptq_reference", False):
        return run_gptq_reference(args)
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
    parser.add_argument("--gptq-reference", action="store_true",
                        help="Add the GPTQ-weight reference KL to an existing profile")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-unit sensitivity profiling")
    add_arguments(parser)
    run(parser.parse_args())
