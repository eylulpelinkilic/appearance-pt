"""Reading a training checkpoint without trusting it.

torch.load(weights_only=True) refuses Pointcept training checkpoints because they hold non-tensor objects (optimizer
and scheduler state, config objects). load_inert reads them with an unpickler that rebuilds tensors and ordered dicts
and turns every other global into an Inert stub: no function or class from the file is ever called, so nothing in it
runs. Model weights come out intact; everything else is a placeholder.
"""

from __future__ import annotations

import pickle
import types
from pathlib import Path

import torch

SAFE_GLOBALS = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"),
                ("torch._utils", "_rebuild_parameter"), ("torch", "Size")}


class Inert:
    """Placeholder for any object the checkpoint would otherwise construct. It also absorbs what pickle does to the
    objects it builds: calls (a stubbed global used as a factory), item assignment and appends (dict- and list-like
    classes), and state."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Inert()

    def __setstate__(self, state):
        pass

    def __setitem__(self, key, value):
        pass

    def append(self, value):
        pass

    def extend(self, values):
        pass


class InertUnpickler(pickle.Unpickler):
    stubbed: set[str] = set()  # every global replaced so far, for the report

    def find_class(self, module, name):
        if (module, name) in SAFE_GLOBALS:
            return super().find_class(module, name)
        if module == "torch" and isinstance(getattr(torch, name, None), torch.dtype):
            return getattr(torch, name)
        InertUnpickler.stubbed.add(f"{module}.{name}")
        return Inert


def load_inert(path: Path, mmap: bool = True):
    """torch.load through InertUnpickler (torch still rebuilds the storages itself, memory-mapped by default)."""
    module = types.ModuleType("inert_pickle")
    module.Unpickler, module.Pickler, module.load = InertUnpickler, pickle.Pickler, pickle.load
    return torch.load(path, map_location="cpu", mmap=mmap, weights_only=False, pickle_module=module)
