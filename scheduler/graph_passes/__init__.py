# scheduler/graph_passes/__init__.py
"""C3.3 Graph pass pipeline.

Usage by evaluation script:
    pipeline = GraphPassPipeline(enable_fusion=True, …)
    pipeline.run(graph)
    # Access: pipeline.pass_results['Fusion']['stats']['fusion_log']
"""

from . import fusion


class GraphPassPipeline:
    """Runs optimization passes on a computation graph."""

    def __init__(self, enable_fusion: bool = True, **kwargs):
        self._enable_fusion = enable_fusion
        self.pass_results = {}

    def run(self, graph):
        """Run all enabled passes on the graph."""
        if self._enable_fusion:
            result = fusion.run_fusion(graph)
            self.pass_results["Fusion"] = result

        # Flatten shape info
        self.pass_results.setdefault("Fusion", {"stats": {"fusion_log": []}})

        return self.pass_results
