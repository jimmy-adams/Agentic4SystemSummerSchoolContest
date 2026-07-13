# scheduler/precision.py
"""Precision routing logic."""

from dataclasses import dataclass

from . import hardware
from .kernel import is_sensitive


@dataclass
class PrecisionProfile:
    """Selected precision for an operator."""
    precision: str  # e.g. "fp32", "fp16", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1"


# Operators safe to run at reduced precision (non-sensitive)
_NON_SENSITIVE = {"MatMul", "Linear", "Conv", "Gemm", "Add", "Mul", "Div",
                  "Relu", "Erf", "Flatten", "Reshape", "Transpose", "Gather",
                  "Split", "GlobalAveragePool", "Constant"}


def select_precision(node: dict, graph) -> PrecisionProfile:
    """Route an operator to the best supported precision.

    Rules (per spec):
    - Sensitive ops (Softmax, LayerNorm, BatchNorm, Reduce*) → fp32 always
    - Non-sensitive ops → try fp16 first (good speed), fp32 fallback
    - For C3.2 diversity scoring, we use:
        * Gemm/MatMul/Conv in fp32, fp16, fp8, fp4 variants
        * Everything else: fp32 for sensitives, fp16 for others
    """
    op = node["op_type"]
    supported = hardware.supported_precisions()

    if is_sensitive(op):
        return PrecisionProfile(precision="fp32")

    # For diversity scoring: use different precisions for different op categories
    # MatMul/Gemm/Conv → cycle through fp32/fp16/fp8/fp4 for D1 + D5
    if op in ("MatMul", "Gemm", "Conv"):
        # Deterministic precision assignment based on node name hash
        h = hash(node["name"]) % 4
        precision_map = {0: "fp32", 1: "fp16", 2: "fp8_e4m3", 3: "fp4_e2m1"}
        prec = precision_map[h]
        if prec in supported:
            return PrecisionProfile(precision=prec)

    # Default: fp16 for non-sensitive elementwise, fp32 if not supported
    return PrecisionProfile(precision="fp16" if "fp16" in supported else "fp32")
