"""
Per-layer, sensitivity-guided mixed-precision weight quantization.

Modules:
    units        - Allocation units (fused groups vLLM serves at one precision) and fake quantization
    sensitivity  - Per-unit sensitivity profiling (cheap one-pass signal + end-to-end KL reference)
"""
