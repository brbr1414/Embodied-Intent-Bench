"""Skeleton tests: the package layout and version discipline hold before any logic lands.

These are intentionally structural. Behavioural tests arrive with the branches that add
behaviour; this file only guards the scaffold that every later branch builds on.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys

import pytest

import aerointentbench

EXPECTED_SUBPACKAGES = [
    "aerointentbench.schemas",
    "aerointentbench.simulator",
    "aerointentbench.executor",
    "aerointentbench.policies",
    "aerointentbench.metrics",
    "aerointentbench.tasks",
    "aerointentbench.tasks.human_search_segmentation",
]


def test_package_exposes_version() -> None:
    assert isinstance(aerointentbench.__version__, str)
    assert aerointentbench.__version__


def test_v1_supports_only_schema_version_1_0() -> None:
    """V1 pins a single schema version; loaders must reject anything else."""
    assert aerointentbench.SCHEMA_VERSION == "1.0"


@pytest.mark.parametrize("module_name", EXPECTED_SUBPACKAGES)
def test_subpackage_is_importable(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__doc__, f"{module_name} must document its responsibility and boundaries"


def _imported_root_modules(source: str) -> set[str]:
    """Return the top-level module names imported by a Python source file."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        # level > 0 is a relative (first-party) import and needs no check.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_package_has_no_third_party_runtime_dependencies() -> None:
    """The V1 core must run without a GPU and without heavy ML packages.

    ``pyproject.toml`` declares zero runtime dependencies; this asserts the source
    actually honours that, so a stray ``import numpy`` fails here rather than at a
    user's install. Checked statically over every module in the package, which keeps
    the test independent of what other tests happened to import first.

    The ``aerointentbench.v2`` subpackage is the one deliberate exception: it is the
    visual simulator behind the optional ``[v2]`` extra, and its own boundary is pinned
    by :func:`test_v2_subpackage_only_uses_its_declared_extras` below.
    """
    package_root = pathlib.Path(aerointentbench.__file__).parent
    modules = sorted(
        p for p in package_root.rglob("*.py") if "v2" not in p.relative_to(package_root).parts
    )
    assert modules, "expected to find package modules to scan"

    allowed = sys.stdlib_module_names | {"aerointentbench"}
    for module_path in modules:
        for root in _imported_root_modules(module_path.read_text(encoding="utf-8")):
            assert root in allowed, (
                f"{module_path.relative_to(package_root.parent)} imports non-stdlib module "
                f"{root!r}; V1 declares zero runtime dependencies"
            )


def test_v2_subpackage_only_uses_its_declared_extras() -> None:
    """The optional V2 simulator may use exactly its declared extras -- nothing heavier.

    numpy/Pillow/rasterio are the ``[v2]`` optional dependencies; PyTorch and friends
    stay banned everywhere. And the V1 package must not import V2 eagerly: installing
    aerointentbench without the extra has to keep working, so ``aerointentbench/__init__``
    (scanned by the core test above) must never pull ``aerointentbench.v2`` in.
    """
    package_root = pathlib.Path(aerointentbench.__file__).parent
    v2_root = package_root / "v2"
    modules = sorted(v2_root.rglob("*.py"))
    assert modules, "expected to find v2 modules to scan"

    allowed = sys.stdlib_module_names | {"aerointentbench", "numpy", "PIL", "rasterio"}
    for module_path in modules:
        for root in _imported_root_modules(module_path.read_text(encoding="utf-8")):
            assert root in allowed, (
                f"{module_path.relative_to(package_root.parent)} imports {root!r}, which is "
                "not part of the declared [v2] extras"
            )

    core_init = (package_root / "__init__.py").read_text(encoding="utf-8")
    assert "v2" not in core_init, "the V1 package must not import the v2 subpackage eagerly"


def test_cli_entry_point_is_importable() -> None:
    """The console script declared in pyproject.toml must resolve.

    Behaviour is covered in test_cli.py; this only guards the entry point itself, which a
    packaging change could break without any test noticing.
    """
    from aerointentbench.run_benchmark import build_parser, main

    assert callable(main)
    assert build_parser().prog == "aerointentbench"
