"""Shim to prevent stdlib shadowing.

This project previously defined a local ``types.py`` which collided with the
Python standard library module of the same name, breaking imports in GPU
images (early site initialization tried to import ``functools`` which depends
on stdlib ``types.GenericAlias``).  Instead of removing the file immediately
(which may break downstream references), we dynamically load the *real* stdlib
``types`` module source into this module's globals to emulate its behavior.

We then (optionally) import training-specific aliases from
``training_types.py`` under distinct names so callers can migrate.

Planned Removal: once all imports switch to ``training_types`` we can delete
this shim entirely.
"""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

# 1. Locate stdlib 'types.py' file.
_stdlib_types_path: str | None = None
for _p in sys.path:
	# Heuristic: typical stdlib path ends with 'pythonX.Y'
	if _p and os.path.isdir(_p) and _p.endswith((f"python{sys.version_info.major}.{sys.version_info.minor}",)):
		candidate = os.path.join(_p, "types.py")
		if os.path.exists(candidate):
			_stdlib_types_path = candidate
			break

if _stdlib_types_path:
	with open(_stdlib_types_path, "r", encoding="utf-8") as f:
		_code = f.read()
	# Execute stdlib types definitions in current module namespace.
	exec(compile(_code, _stdlib_types_path, "exec"), globals(), globals())
else:
	# Fallback minimal sentinel if path not found.
	class _StdLibTypesLoadError(Exception):
		pass
	raise _StdLibTypesLoadError("Could not locate stdlib types.py to load real definitions.")

# 2. Optionally pull in training-specific aliases (non-shadowing names).
try:  # pragma: no cover - best effort
	from .training_types import Params as TrainingParams, Trainable as TrainingTrainable  # type: ignore
	# Expose for migration without polluting stdlib names.
	globals()["TrainingParams"] = TrainingParams
	globals()["TrainingTrainable"] = TrainingTrainable
	if "__all__" in globals():
		__all__.extend(["TrainingParams", "TrainingTrainable"])  # type: ignore
except Exception:  # noqa: BLE001
	if TYPE_CHECKING:  # make mypy/pyright aware optionally
		from .training_types import Params as TrainingParams, Trainable as TrainingTrainable  # type: ignore
	# Silent: training aliases unavailable.
	pass


