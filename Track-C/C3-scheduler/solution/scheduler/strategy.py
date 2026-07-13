# scheduler/strategy.py
"""Strategy: the main API entry-point for C3.2 evaluation.

The evaluation script imports this and calls:
  - strategy.select_precision(node, graph) → PrecisionProfile
  - strategy.decompose(node, graph, precision) → List[KernelSpecRef]
  - strategy.tune_kernel(ref, precision, problem_size) → KernelTuningParams
"""

from typing import List

from . import hardware, kernel, precision, tuning
from .precision import PrecisionProfile
from .kernel import KernelSpecRef
from .tuning import KernelTuningParams


class _Strategy:
    """Singleton strategy object used by the evaluation script."""

    def __init__(self):
        self._hardware = hardware
        self._tuning = tuning

    def select_precision(self, node: dict, graph) -> PrecisionProfile:
        """Return the precision profile for an ONNX operator node."""
        return precision.select_precision(node, graph)

    def decompose(self, node: dict, graph, prec) -> List[KernelSpecRef]:
        """Decompose an ONNX operator into a kernel sequence.

        Args:
            node: operator node dict
            graph: Graph object
            prec: PrecisionProfile or str
        """
        p = prec.precision if isinstance(prec, PrecisionProfile) else str(prec)
        return kernel.decompose(node, graph, p)

    def tune_kernel(self, ref: KernelSpecRef, prec,
                    problem_size) -> KernelTuningParams:
        """Return launch parameters for a kernel."""
        p = prec.precision if isinstance(prec, PrecisionProfile) else str(prec)
        return tuning.tune_kernel(ref, p, problem_size)


# Module-level instance substituted by the evaluation script
strategy = _Strategy()
