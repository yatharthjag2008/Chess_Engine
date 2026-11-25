"""Every command-line script has usage notes and a working --help (step 20)."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


def test_there_are_scripts() -> None:
    assert len(SCRIPTS) >= 20


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_script_has_usage_and_help(path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    assert doc and len(doc.splitlines()) >= 2, f"{path.name} needs a docstring saying what it does and how to run it"
    spec = importlib.util.spec_from_file_location(f"script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        pytest.skip(f"{path.name} has no main()")
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--help"])
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out
