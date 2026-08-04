"""Executable statement of the hexagonal layering.

The rule is one sentence: **dependencies point inward**. `domain` knows only
`ports`; `infra` and `adapters` implement them; nothing inner reaches out.

Without a test this degrades on the first hurried commit — glosa already had
one outward arrow (`document.index` → `studio.projection`) before this file
existed. Docling Studio enforces the same discipline with `pytestarch`; a few
lines of `ast` do the job here without another dependency.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
PACKAGE = SRC / "glosa"

#: layer -> (may import these glosa layers, may import these third-party roots)
LAYERS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # The domain may describe data (pydantic is the schema language of the
    # ChatModel port) but may never reach an implementation or do I/O.
    "glosa.domain": (frozenset({"glosa.domain", "glosa.ports"}), frozenset({"pydantic"})),
    "glosa.ports": (frozenset({"glosa.domain", "glosa.ports"}), frozenset({"pydantic"})),
    "glosa.infra": (
        frozenset({"glosa.domain", "glosa.ports", "glosa.infra"}),
        frozenset({"httpx", "pydantic"}),
    ),
    "glosa.adapters": (
        frozenset({"glosa.domain", "glosa.ports", "glosa.infra", "glosa.adapters"}),
        frozenset({"httpx", "pydantic"}),
    ),
}

STDLIB_OK = frozenset(
    {
        "__future__",
        "abc",
        "argparse",
        "ast",
        "asyncio",
        "collections",
        "collections.abc",
        "dataclasses",
        "enum",
        "hashlib",
        "html",
        "itertools",
        "json",
        "logging",
        "math",
        "pathlib",
        "re",
        "sys",
        "time",
        "typing",
        "unicodedata",
    }
)


def _modules() -> list[tuple[str, pathlib.Path]]:
    out = []
    for path in sorted(PACKAGE.rglob("*.py")):
        name = str(path.relative_to(SRC)).removesuffix(".py").replace("/", ".")
        out.append((name.removesuffix(".__init__"), path))
    return out


def _layer_of(module: str) -> str | None:
    for layer in LAYERS:
        if module == layer or module.startswith(layer + "."):
            return layer
    return None


def _imports(path: pathlib.Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


def _root(module: str) -> str:
    return module.split(".")[0]


MODULES = _modules()
LAYERED = [(name, path) for name, path in MODULES if _layer_of(name) is not None]


@pytest.mark.parametrize(("module", "path"), LAYERED, ids=[m for m, _ in LAYERED])
def test_module_only_imports_inward(module: str, path: pathlib.Path) -> None:
    layer = _layer_of(module)
    assert layer is not None
    allowed_layers, allowed_third_party = LAYERS[layer]

    for imported in _imports(path):
        if imported.startswith("glosa"):
            target = _layer_of(imported)
            assert target is not None, f"{module} imports unlayered {imported}"
            assert target in allowed_layers, (
                f"{module} ({layer}) must not import {imported} ({target}) — "
                "dependencies point inward"
            )
        elif _root(imported) not in STDLIB_OK:
            assert _root(imported) in allowed_third_party, (
                f"{module} ({layer}) must not import {imported}"
            )


def test_the_domain_never_parses_a_document() -> None:
    """Only the Docling adapter knows what a stored document looks like."""
    offenders = [
        module
        for module, path in MODULES
        if "json.loads" in path.read_text(encoding="utf-8")
        and not module.startswith("glosa.infra.docling")
    ]
    assert offenders == [], f"json.loads outside the Docling adapter: {offenders}"


def test_only_the_llm_adapters_speak_http() -> None:
    offenders = [
        module
        for module, path in MODULES
        if "httpx" in _imports(path) and not module.startswith("glosa.infra.llm")
    ]
    assert offenders == [], f"httpx outside the LLM adapters: {offenders}"


def test_only_the_docling_adapter_reads_the_docling_tree() -> None:
    """`tree.py` mirrors Studio's collapse rules; nothing else may use it."""
    offenders = [
        module
        for module, path in MODULES
        if any(i.startswith("glosa.infra.docling.tree") for i in _imports(path))
        and not module.startswith("glosa.infra.docling")
    ]
    assert offenders == [], f"docling tree helpers leaked into: {offenders}"


def test_every_module_belongs_to_a_layer() -> None:
    """`cli` and the package root are composition roots; everything else is layered."""
    unlayered = {module for module, _ in MODULES if _layer_of(module) is None}
    assert unlayered == {"glosa", "glosa.cli"}, unlayered
