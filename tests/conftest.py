from __future__ import annotations

import re
from collections.abc import Generator
from contextlib import suppress
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "test_out"


def _sanitize_nodeid(nodeid: str) -> str:
    """Create a filesystem-safe name from a pytest node id."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", nodeid).strip("_")


def _prune_empty_artifacts(root: Path) -> None:
    """Remove empty test logs and empty directories under the artifact root."""
    if not root.exists():
        return

    for log_file in root.rglob("test.log"):
        if log_file.is_file() and log_file.stat().st_size == 0:
            log_file.unlink()

    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        reverse=True,
    )
    for directory in directories:
        with suppress(OSError):
            directory.rmdir()

    with suppress(OSError):
        root.rmdir()


@pytest.fixture(scope="session")
def outdir() -> Path:
    """Create and return the shared root output directory for test artifacts."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    return OUTDIR


@pytest.fixture
def test_outdir(outdir: Path, request: pytest.FixtureRequest) -> Path:
    """Create a per-test artifact directory under the shared outdir."""
    test_dir = outdir / _sanitize_nodeid(request.node.nodeid)
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture(autouse=True)
def save_test_logs_and_plots(
    test_outdir: Path,
    caplog: pytest.LogCaptureFixture,
) -> Generator[None, None, None]:
    """Persist captured logs and open matplotlib figures for each test."""
    caplog.set_level("DEBUG")
    yield

    log_text = caplog.text
    if log_text.strip():
        log_file = test_outdir / "test.log"
        log_file.write_text(log_text, encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        if not any(test_outdir.iterdir()):
            test_outdir.rmdir()
        return

    figure_numbers = plt.get_fignums()
    if figure_numbers:
        figures_dir = test_outdir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        for index, fig_num in enumerate(figure_numbers, start=1):
            figure = plt.figure(fig_num)
            figure.savefig(figures_dir / f"figure_{index:02d}.png", dpi=140)
    plt.close("all")

    if not any(test_outdir.iterdir()):
        test_outdir.rmdir()


@pytest.fixture(scope="session", autouse=True)
def prune_test_artifacts_at_session_end(outdir: Path) -> Generator[None, None, None]:
    """Prune empty artifacts at the end of the test session."""
    yield
    _prune_empty_artifacts(outdir)
