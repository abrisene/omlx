"""Load the real host PLE implementation without importing GPU packages."""
import ast
import json
import logging
import math
import mmap
import os
import struct
import time
import threading
import types
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py"
)


class NoMLX:
    def __getattr__(self, name):
        raise AssertionError("MLX touched: " + name)


def load_runtime():
    tree = ast.parse(SOURCE.read_text())
    selected = []
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and (
                node.name.startswith("_ple_")
                or node.name in {"_SafeTensorMMap", "DiskBackedShardedEmbedding"}
            )
            or isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name)
                and t.id.startswith("_PLE_")
                and t.id not in {"_PLE_RUNTIME_MODEL_PATH", "_PLE_RUNTIME_MODE"}
                for t in node.targets
            )
            or isinstance(node, ast.If)
            and "register_at_fork" in ast.unparse(node.test)
        ):
            selected.append(node)
    runtime = types.ModuleType("_cpu_ple_runtime")
    namespace = runtime.__dict__
    namespace.update(
        runtime=runtime,
        os=os,
        Lock=threading.Lock,
        RLock=threading.RLock,
        ThreadPoolExecutor=ThreadPoolExecutor,
        Path=Path,
        struct=struct,
        json=json,
        mmap=mmap,
        np=np,
        math=math,
        time=time,
        wait=wait,
        register_ple_resource=lambda *a, **k: None,
        logger=logging.getLogger(__name__),
        mx=NoMLX(),
        nn=types.SimpleNamespace(Module=object),
    )
    package = SOURCE.parent
    for filename, name in [("ple_mmap.py", "_SafeTensorMMap"), ("ple_disk.py", "DiskBackedShardedEmbedding")]:
        tree = ast.parse((package / filename).read_text())
        selected.extend(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    layout_tree = ast.parse((package / "ple_layout.py").read_text())
    selected = [n for n in layout_tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))] + selected
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + selected,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace

