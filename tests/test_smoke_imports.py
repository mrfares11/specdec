"""Every module under src/ imports cleanly on CPU; scripts at least compile."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import src

ROOT = Path(__file__).resolve().parents[1]


def test_all_src_modules_import():
    failures = []
    for info in pkgutil.walk_packages(src.__path__, prefix="src."):
        try:
            importlib.import_module(info.name)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{info.name}: {type(e).__name__}: {e}")
    assert not failures, "import failures:\n" + "\n".join(failures)


def test_scripts_compile():
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    assert len(scripts) >= 4, "core CLI entrypoints missing"
    for path in scripts:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_public_functions_have_docstrings():
    """Spot-check the public API surface promised in the task spec."""
    from src.decode import generate, tree, verify
    from src.data import shards

    for fn in (
        tree.build_tree, tree.build_from_config, tree.register_allocation,
        verify.verify_chain, verify.verify_tree,
        generate.speculative_generate, generate.greedy_generate,
        shards.estimate_bytes,
    ):
        assert fn.__doc__ and fn.__doc__.strip(), f"{fn.__qualname__} lacks a docstring"
    for cls in (tree.DraftTree, verify.VerifyOutcome, shards.ShardWriter, shards.ShardReader):
        assert cls.__doc__ and cls.__doc__.strip()
