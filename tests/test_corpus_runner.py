# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the corpus release runner."""

import logging
import os
import subprocess
import sys
import time
import warnings
from types import SimpleNamespace

import pytest

import run_corpus_test
from pdftopdfa.utils import SUPPORTED_LEVELS


@pytest.fixture(autouse=True)
def restore_process_logging_state():
    """Keep runner CLI configuration from leaking into later unit tests."""
    loggers = [
        logging.getLogger(),
        logging.getLogger("pdftopdfa"),
        logging.getLogger("pikepdf"),
    ]
    logger_state = [
        (logger.level, list(logger.handlers), logger.propagate) for logger in loggers
    ]
    warning_filters = list(warnings.filters)

    yield

    for logger, (level, handlers, propagate) in zip(loggers, logger_state):
        logger.setLevel(level)
        logger.handlers[:] = handlers
        logger.propagate = propagate
    warnings.filters[:] = warning_filters


def _delayed_corpus_result(task):
    """Return a corpus-shaped result after the requested test delay."""
    input_path, level, _, delay = task
    time.sleep(delay)
    return {
        "input": str(input_path),
        "relative_path": str(input_path),
        "level": level,
        "success": True,
        "validation_failed": False,
        "skipped": False,
        "error": None,
        "error_type": None,
        "warnings": [],
        "validation_messages": [],
        "processing_time": delay,
    }


def test_corpus_runner_uses_every_supported_level() -> None:
    """The release runner exercises every PDF/A level exposed by the package."""
    assert run_corpus_test.LEVELS == sorted(SUPPORTED_LEVELS)


def test_default_timeout_covers_observed_long_corpus_conversion() -> None:
    """The default covers the corpus conversion measured at 460.38 seconds."""
    assert run_corpus_test.TASK_TIMEOUT_SECONDS > 460.38


def test_analysis_handles_unicode_path_on_cp1252_console(tmp_path) -> None:
    """The real analysis process cannot abort on a legacy Windows console."""
    project_root = run_corpus_test.Path(__file__).resolve().parents[1]
    code = """
from pathlib import Path
import sys
import run_corpus_test as runner

root = Path(sys.argv[1])
corpus = root / "corpus"
corpus.mkdir()
input_path = corpus / f"{chr(0x534D)}.pdf"
input_path.touch()
runner.CORPUS_DIR = corpus
runner.RESULTS_DIR = root / "results"
runner.RESULTS_DIR.mkdir()
runner.LEVELS = ["2a"]
runner._configure_console_output()
runner.analyze_results(
    [{
        "input": str(input_path),
        "relative_path": input_path.name,
        "level": "2a",
        "success": False,
        "validation_failed": False,
        "skipped": False,
        "error": f"Fehler {chr(0x03A9)}",
        "error_type": "ConversionError",
        "warnings": [],
        "validation_messages": [],
        "processing_time": 1.0,
    }],
    1.0,
)
"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (
                str(project_root),
                str(project_root / "src"),
                env.get("PYTHONPATH"),
            ),
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=project_root,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("cp1252", errors="replace")
    assert b"UnicodeEncodeError" not in result.stderr


def test_output_path_is_stable_and_unique_for_duplicate_stems(
    tmp_path, monkeypatch
) -> None:
    """Corpus-relative paths prevent duplicate basenames from sharing outputs."""
    corpus_dir = tmp_path / "corpus"
    first = corpus_dir / "one" / "document.pdf"
    second = corpus_dir / "two" / "document.pdf"
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(run_corpus_test, "CORPUS_DIR", corpus_dir)

    first_output = run_corpus_test._output_path(first, "2a", output_dir)
    second_output = run_corpus_test._output_path(second, "2a", output_dir)

    assert first_output == run_corpus_test._output_path(first, "2a", output_dir)
    assert first_output != second_output
    assert first_output.parent == output_dir
    assert first_output.name.endswith("_2a.pdf")


def test_validation_skipped_warning_fails_conversion(tmp_path, monkeypatch) -> None:
    """A missing or failed validator cannot be reported as a corpus success."""
    corpus_dir = tmp_path / "corpus"
    input_path = corpus_dir / "input.pdf"
    input_path.parent.mkdir()
    input_path.touch()
    warning = "Validation skipped: veraPDF not available"
    monkeypatch.setattr(run_corpus_test, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(
        run_corpus_test,
        "convert_to_pdfa",
        lambda **kwargs: SimpleNamespace(
            success=True,
            validation_failed=False,
            skipped=False,
            warnings=[warning],
            processing_time=0.1,
            error=None,
        ),
    )

    result = run_corpus_test.convert_single((input_path, "2a", tmp_path))

    assert result["success"] is False
    assert result["validation_failed"] is False
    assert result["error"] == warning
    assert result["error_type"] == "ValidationSkipped"


def test_main_fails_before_scanning_when_verapdf_is_unavailable(
    monkeypatch,
) -> None:
    """The release gate stops before corpus work without a usable validator."""

    def find_all_pdfs():
        raise AssertionError("must not scan")

    monkeypatch.setattr(run_corpus_test, "get_verapdf_version", lambda: None)
    monkeypatch.setattr(run_corpus_test, "find_all_pdfs", find_all_pdfs)

    assert run_corpus_test.main() == 1


def test_task_timeout_terminates_hung_worker_and_keeps_completed_result(
    tmp_path, monkeypatch
) -> None:
    """A timed-out conversion cannot block other completed corpus work."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    monkeypatch.setattr(run_corpus_test, "CORPUS_DIR", corpus_dir)
    tasks = [
        (corpus_dir / "slow.pdf", "2a", tmp_path, 5.0),
        (corpus_dir / "fast.pdf", "3a", tmp_path, 0.0),
    ]

    started = time.monotonic()
    results, _ = run_corpus_test._run_tasks(
        tasks,
        worker=_delayed_corpus_result,
        max_workers=2,
        task_timeout=2,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 4
    assert sum(result["success"] for result in results) == 1
    timeout = next(result for result in results if not result["success"])
    assert timeout["error_type"] == "WorkerTimeout"
    assert timeout["relative_path"] == "slow.pdf"


def test_main_returns_nonzero_when_any_corpus_task_fails(tmp_path, monkeypatch) -> None:
    """A failed conversion makes the corpus command fail as a release gate."""
    corpus_dir = tmp_path / "corpus"
    input_path = corpus_dir / "input.pdf"
    input_path.parent.mkdir()
    input_path.touch()
    monkeypatch.setattr(run_corpus_test, "CORPUS_DIR", corpus_dir)
    failure = run_corpus_test._worker_failure_result(
        (input_path, "2a", tmp_path),
        "validation failed",
        "ValidationFailed",
    )
    monkeypatch.setattr(run_corpus_test, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_corpus_test, "LEVELS", ["2a"])
    monkeypatch.setattr(run_corpus_test, "get_verapdf_version", lambda: "veraPDF")
    monkeypatch.setattr(run_corpus_test, "find_all_pdfs", lambda: [input_path])
    monkeypatch.setattr(
        run_corpus_test,
        "_run_tasks",
        lambda tasks: ([failure], 0.1),
    )
    monkeypatch.setattr(run_corpus_test, "analyze_results", lambda *args: None)

    assert run_corpus_test.main() == 1


def test_main_writes_unicode_reports_as_utf8(tmp_path, monkeypatch) -> None:
    """Unicode corpus paths and errors survive every persisted report."""
    corpus_dir = tmp_path / "corpus"
    input_path = corpus_dir / "文档-😀.pdf"
    input_path.parent.mkdir()
    input_path.touch()
    results_dir = tmp_path / "results"
    error = "Fehler Ω 😀"
    monkeypatch.setattr(run_corpus_test, "CORPUS_DIR", corpus_dir)
    failure = run_corpus_test._worker_failure_result(
        (input_path, "2a", tmp_path),
        error,
        "ConversionError",
    )
    monkeypatch.setattr(run_corpus_test, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(run_corpus_test, "LEVELS", ["2a"])
    monkeypatch.setattr(run_corpus_test, "get_verapdf_version", lambda: "veraPDF")
    monkeypatch.setattr(run_corpus_test, "find_all_pdfs", lambda: [input_path])
    monkeypatch.setattr(
        run_corpus_test,
        "_run_tasks",
        lambda tasks: ([failure], 0.1),
    )

    assert run_corpus_test.main() == 1
    for report_name in (
        "raw_results.json",
        "analysis_report.txt",
        "results_summary.csv",
    ):
        report = (results_dir / report_name).read_text(encoding="utf-8")
        assert "文档-😀.pdf" in report
        assert error in report
