"""Ensemble training pipeline"""
"""Ensemble training pipeline (placeholder)."""

from .base import Pipeline


class EnsemblePipeline:
	def run(self, **kwargs):  # pragma: no cover - placeholder
		return {"ensemble": True, **kwargs}


__all__ = ["EnsemblePipeline"]
