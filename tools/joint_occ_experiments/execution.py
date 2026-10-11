"""Reuse existing bounded inference scheduling, with BOTH action probabilities.

The legacy SurfaceExecution intentionally zeros REMOVE, so it cannot be used
unchanged for the new head. Only this subclass overrides the probability readout.
Static CUDA graph capture still checks eager/graph bytes before first use.
"""
from real_motion.surface_ccr_execution import SurfaceExecution


class BothActionsExecution(SurfaceExecution):
    def _forward(self, values, live):
        features, labels, actors, classes, context, base, fallback, legal = values
        encoded = self.head.encode(features, labels, actors, classes, live)
        logits = self.head.decode(encoded, actors, context, base, fallback, legal, live)
        return self.head.probabilities(logits, actors)
