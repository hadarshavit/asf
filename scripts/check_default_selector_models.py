#!/usr/bin/env python3
"""Static check ensuring selector defaults rely on predictor wrappers."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTORS_DIR = REPO_ROOT / "asf" / "selectors"
TARGET_PREFIX = "asf.predictors"


@dataclass
class DefaultModel:
    file_path: Path
    class_name: str
    parameter: str
    default_name: str
    default_node: ast.AST
    resolved_module: Optional[str]


def module_name_for_path(path: Path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def resolve_relative_module(base_module: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or base_module

    base_parts = base_module.split(".")
    if node.level > len(base_parts):
        return node.module or ""

    prefix = base_parts[: -node.level]
    module_parts: List[str] = []
    if node.module:
        module_parts = node.module.split(".")
    return ".".join(prefix + module_parts)


def collect_imports(tree: ast.AST, file_path: Path) -> Dict[str, str]:
    imports: Dict[str, str] = {}
    module_name = module_name_for_path(file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = resolve_relative_module(module_name, node)
            for alias in node.names:
                asname = alias.asname or alias.name
                imports[asname] = resolved
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                asname = alias.asname or module.split(".")[-1]
                imports[asname] = module
    return imports


def get_default_name(default: ast.AST) -> Optional[str]:
    if isinstance(default, ast.Name):
        return default.id
    if isinstance(default, ast.Attribute):
        parts: List[str] = []
        node: ast.AST = default
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
    return None


def resolve_module(name: str, imports: Dict[str, str]) -> Optional[str]:
    if "." in name:
        first, *rest = name.split(".")
        base = imports.get(first)
        if base is None:
            return None
        return base + "." + ".".join(rest) if rest else base
    return imports.get(name)


def find_default_models(tree: ast.AST, file_path: Path) -> List[DefaultModel]:
    imports = collect_imports(tree, file_path)
    defaults: List[DefaultModel] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            args = item.args.args
            defaults_nodes = item.args.defaults
            if not defaults_nodes:
                continue
            applicable_args = args[-len(defaults_nodes) :]
            for arg, default in zip(applicable_args, defaults_nodes):
                if arg.arg == "self" or "model" not in arg.arg.lower():
                    continue
                default_name = get_default_name(default)
                if default_name is None:
                    continue
                resolved_module = resolve_module(default_name, imports)
                defaults.append(
                    DefaultModel(
                        file_path=file_path,
                        class_name=node.name,
                        parameter=arg.arg,
                        default_name=default_name,
                        default_node=default,
                        resolved_module=resolved_module,
                    )
                )
    return defaults


def main() -> int:
    violations: List[DefaultModel] = []

    for path in sorted(SELECTORS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        defaults = find_default_models(tree, path)
        print(defaults)
        for default in defaults:
            module = default.resolved_module or ""
            if not module.startswith(TARGET_PREFIX):
                violations.append(default)

    if violations:
        for violation in violations:
            rel = violation.file_path.relative_to(REPO_ROOT)
            lineno = getattr(violation.default_node, "lineno", "?")
            module = violation.resolved_module or "<unknown>"
            print(
                f"{rel}:{lineno}: "
                f"{violation.class_name}.__init__ parameter '{violation.parameter}' has default "
                f"'{violation.default_name}' from module '{module}', expected '{TARGET_PREFIX}*'.",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
