"""Packaging and documentation contracts: the Docker build context, the README's claims about
the local sfx provider and about ``cast_overrides.json``, and the web UI's inline script."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from bookreader.providers.local.sfx import AUDIOCRAFT_HINT

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_dockerignore_keeps_venv_git_data_and_caches_out_of_the_build_context() -> None:
    dockerignore = ROOT / ".dockerignore"
    assert dockerignore.is_file(), "docker build . would upload .venv, .git and data/ without a .dockerignore"
    patterns = {line.strip() for line in dockerignore.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    for required in (".venv/", ".git/", "data/", "**/__pycache__/", ".pytest_cache/", "tests/"):
        assert required in patterns, f"{required} missing from .dockerignore"


def test_readme_says_audiogen_needs_a_manual_audiocraft_install() -> None:
    """No pip extra installs audiocraft, so the README must not promise AudioGen from `local` alone."""
    extras = re.search(r"\[project\.optional-dependencies\](.*?)\n\[", PYPROJECT, re.S)
    assert extras and "audiocraft" not in "".join(line for line in extras.group(1).splitlines() if not line.lstrip().startswith("#"))
    assert "pip install audiocraft" in AUDIOCRAFT_HINT
    provider_row = next(line for line in README.splitlines() if line.startswith("| `local` |"))
    assert "pip install audiocraft" in provider_row and "procedural" in provider_row
    fallback_row = next(line for line in README.splitlines() if line.startswith("| `BOOKREADER_LOCAL_SFX_FALLBACK` |"))
    assert "pip install audiocraft" in fallback_row and "mock" in fallback_row
    docker = README[README.index("## Docker"):]
    assert "audiocraft" in docker


def test_readme_artifact_layout_does_not_claim_a_first_run_cast_writes_cast_overrides() -> None:
    line = next(line for line in README.splitlines() if "cast_overrides.json" in line and "written by" in line)
    assert "PUT /cast" in line and "--from-stage cast" in line
    assert "(or --cast)" not in line


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_web_ui_inline_script_parses(tmp_path: Path) -> None:
    html = (ROOT / "bookreader" / "web" / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    assert scripts, "index.html has no inline script"
    for index, script in enumerate(scripts):
        path = tmp_path / f"ui{index}.js"
        path.write_text(script, encoding="utf-8")
        result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
