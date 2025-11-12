"""GPU resource manager stubs.

Detects whether GPU acceleration is available and exposes simple helpers.
"""

def is_gpu_available() -> bool:
	try:
		import torch  # type: ignore

		return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
	except Exception:
		return False


__all__ = ["is_gpu_available"]

