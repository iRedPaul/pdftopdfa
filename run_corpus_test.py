#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Comprehensive corpus test for pdftopdfa converter.

Tests all PDFs in veraPDF-corpus-staging against all supported PDF/A levels.
"""

import csv
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import pikepdf

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdftopdfa.converter import convert_to_pdfa
from pdftopdfa.utils import SUPPORTED_LEVELS
from pdftopdfa.verapdf import get_verapdf_version

CORPUS_DIR = Path(__file__).parent / "veraPDF-corpus-staging"
LEVELS = sorted(SUPPORTED_LEVELS)
RESULTS_DIR = Path(__file__).parent / "corpus_test_results"
MAX_WORKERS = os.cpu_count() or 4
TASK_TIMEOUT_SECONDS = 900


def _configure_console_output():
    """Replace characters unsupported by the active console encoding."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (OSError, TypeError, ValueError):
            pass


def _split_relative_path(relative_path):
    """Split a relative path independent of path separator style."""
    return [part for part in re.split(r"[\\/]+", relative_path) if part]


def _top_level_category(relative_path):
    """Return the top-level corpus category for a relative path."""
    parts = _split_relative_path(relative_path)
    return parts[0] if parts else "unknown"


def _deduplicate_messages(messages):
    """Deduplicate messages while preserving order."""
    return list(dict.fromkeys(messages))


def _warning_to_status_message(warning):
    """Normalize conversion warnings into reportable status messages."""
    if warning.startswith("Conversion skipped:"):
        return f"WARNING: {warning}"
    if warning.startswith("Validation skipped:"):
        return f"WARNING: {warning}"
    return f"INFO: {warning}"


def _build_status_messages(captured_messages, warnings_list, *, skipped=False):
    """Return the report messages for a conversion attempt."""
    messages = _deduplicate_messages(captured_messages)
    if messages:
        return messages

    fallback = []
    if skipped:
        fallback.extend(
            _warning_to_status_message(warning)
            for warning in warnings_list
            if warning.startswith("Conversion skipped:")
        )

    return _deduplicate_messages(fallback)


def _is_skipped_result(result):
    """Return True when a result represents a skipped passthrough conversion."""
    if result.get("skipped"):
        return True

    return any(
        warning.startswith("Conversion skipped:")
        for warning in result.get("warnings", [])
    )


def _output_path(pdf_path, level, output_dir):
    """Build a deterministic output path unique to the corpus-relative source."""
    relative_path = pdf_path.relative_to(CORPUS_DIR)
    path_hash = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()
    safe_name = pdf_path.stem.replace(" ", "_")[:80]
    return Path(output_dir) / f"{safe_name}_{path_hash}_{level}.pdf"


def _result_status_messages(result):
    """Return normalized status messages for a stored result row."""
    return _build_status_messages(
        result.get("validation_messages", []),
        result.get("warnings", []),
        skipped=_is_skipped_result(result),
    )


def _read_pdf_metadata(relative_path):
    """Read lightweight PDF metadata for performance analysis."""
    pdf_path = CORPUS_DIR / Path(relative_path)
    metadata = {
        "file_size": None,
        "page_count": None,
        "encrypted": False,
    }

    try:
        metadata["file_size"] = pdf_path.stat().st_size
    except OSError:
        return metadata

    try:
        with pikepdf.open(pdf_path) as pdf:
            metadata["page_count"] = len(pdf.pages)
    except pikepdf.PasswordError:
        metadata["encrypted"] = True
    except pikepdf.PdfError:
        pass

    return metadata


def _format_file_size(num_bytes):
    """Format a file size in bytes for human-readable reports."""
    if num_bytes is None:
        return "size unknown"

    units = ("B", "KiB", "MiB", "GiB")
    value = float(num_bytes)
    unit = units[0]
    for candidate in units[1:]:
        if value < 1024:
            break
        value /= 1024
        unit = candidate

    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def _collect_slowest_conversion_entries(results, limit=10):
    """Return the slowest individual conversion runs."""
    slowest = sorted(
        results,
        key=lambda item: item.get("processing_time", 0.0),
        reverse=True,
    )[:limit]

    return [
        {
            "relative_path": item["relative_path"],
            "level": item["level"],
            "processing_time": item.get("processing_time", 0.0),
            "success": item["success"],
            "skipped": _is_skipped_result(item),
        }
        for item in slowest
    ]


def _collect_slowest_file_entries(results, limit=10):
    """Return the slowest unique files aggregated across all levels."""
    by_file = defaultdict(list)
    for item in results:
        by_file[item["relative_path"]].append(item)

    slowest = []
    for relative_path, items in by_file.items():
        times = [item.get("processing_time", 0.0) for item in items]
        slowest.append(
            {
                "relative_path": relative_path,
                "runs": len(items),
                "avg_processing_time": sum(times) / len(times),
                "max_processing_time": max(times),
                "levels": sorted({item["level"] for item in items}),
                "all_skipped": all(_is_skipped_result(item) for item in items),
            }
        )

    slowest.sort(
        key=lambda item: (-item["max_processing_time"], -item["avg_processing_time"])
    )
    return slowest[:limit]


def _format_runtime_entry(entry, metadata):
    """Format one runtime analysis entry."""
    page_count = metadata.get("page_count")
    if page_count is None:
        if metadata.get("encrypted"):
            page_text = "encrypted"
        else:
            page_text = "pages unknown"
        secs_per_page = None
    else:
        page_text = f"{page_count} page(s)"
        secs_per_page = (
            entry["processing_time"] / page_count if page_count > 0 else None
        )

    size_text = _format_file_size(metadata.get("file_size"))
    if secs_per_page is None:
        return f"{page_text} | {size_text}"
    return f"{page_text} | {size_text} | {secs_per_page:.4f}s/page"


class _CapturedPdftopdfaLogHandler(logging.Handler):
    """Collect relevant pdftopdfa log messages for a single conversion."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        if not record.name.startswith("pdftopdfa"):
            return

        message = record.getMessage()
        relevant_fragments = (
            "veraPDF validation:",
            "PDF claims PDF/A-",
            "veraPDF validation not available:",
            "veraPDF not available, skipping PDF/A pre-check",
            "Conversion skipped:",
            "Skipping conversion: PDF is already valid PDF/A-",
        )
        if not any(fragment in message for fragment in relevant_fragments):
            return

        self.messages.append(f"{record.levelname}: {message}")


def _init_worker():
    """Initialize subprocess: suppress pikepdf warnings for malformed streams."""
    warnings.filterwarnings(
        "ignore",
        message="Unexpected end of stream",
        module="pikepdf",
    )


def convert_single(args):
    """Convert a single PDF to a given level. Runs in subprocess."""
    pdf_path, level, output_dir = args
    result = {
        "input": str(pdf_path),
        "relative_path": str(pdf_path.relative_to(CORPUS_DIR)),
        "level": level,
        "success": False,
        "validation_failed": False,
        "skipped": False,
        "error": None,
        "error_type": None,
        "warnings": [],
        "validation_messages": [],
        "processing_time": 0.0,
    }

    output_path = _output_path(pdf_path, level, output_dir)

    pdftopdfa_logger = logging.getLogger("pdftopdfa")
    capture_handler = _CapturedPdftopdfaLogHandler()
    previous_level = pdftopdfa_logger.level
    previous_propagate = pdftopdfa_logger.propagate
    pdftopdfa_logger.addHandler(capture_handler)
    pdftopdfa_logger.setLevel(logging.INFO)
    pdftopdfa_logger.propagate = False

    try:
        conv_result = convert_to_pdfa(
            input_path=pdf_path,
            output_path=output_path,
            level=level,
            validate=True,
        )
        result["success"] = conv_result.success and not conv_result.validation_failed
        result["validation_failed"] = conv_result.validation_failed
        result["skipped"] = conv_result.skipped
        result["warnings"] = conv_result.warnings
        result["processing_time"] = conv_result.processing_time
        validation_skipped = next(
            (
                warning
                for warning in conv_result.warnings
                if warning.startswith("Validation skipped:")
            ),
            None,
        )
        if validation_skipped:
            result["success"] = False
            result["error"] = validation_skipped
            result["error_type"] = "ValidationSkipped"
        elif conv_result.validation_failed:
            result["error"] = "veraPDF validation failed"
            result["error_type"] = "ValidationFailed"
        elif conv_result.error:
            result["error"] = conv_result.error
            result["error_type"] = "ConversionResult.error"
    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        result["processing_time"] = 0.0
    finally:
        pdftopdfa_logger.removeHandler(capture_handler)
        pdftopdfa_logger.setLevel(previous_level)
        pdftopdfa_logger.propagate = previous_propagate

    result["validation_messages"] = _build_status_messages(
        capture_handler.messages,
        result["warnings"],
        skipped=result["skipped"],
    )

    # Clean up output file to save disk space
    try:
        if output_path.exists():
            output_path.unlink()
    except OSError:
        pass

    return result


def _worker_failure_result(task, message, error_type):
    """Build a failed result row for a worker exception or timeout."""
    pdf_path, level = task[:2]
    return {
        "input": str(pdf_path),
        "relative_path": str(pdf_path.relative_to(CORPUS_DIR)),
        "level": level,
        "success": False,
        "validation_failed": False,
        "skipped": False,
        "error": message,
        "error_type": error_type,
        "warnings": [],
        "validation_messages": [],
        "processing_time": 0.0,
    }


def _terminate_executor(executor):
    """Stop all worker processes without waiting for hung conversions."""
    terminate_workers = getattr(executor, "terminate_workers", None)
    if terminate_workers is not None:
        terminate_workers()
        return

    # Python 3.12/3.13 have no public immediate-termination API.
    processes = tuple(getattr(executor, "_processes", {}).values())
    for process in processes:
        process.terminate()
    executor.shutdown(wait=False, cancel_futures=True)


def _run_tasks(
    tasks,
    *,
    worker=convert_single,
    max_workers=MAX_WORKERS,
    task_timeout=TASK_TIMEOUT_SECONDS,
):
    """Run a bounded task set and replace a pool when one worker times out."""
    queued = deque(tasks)
    pending = {}
    results = []
    total = len(tasks)
    started = time.monotonic()
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
    )

    try:
        while queued or pending:
            while queued and len(pending) < max_workers:
                task = queued.popleft()
                future = executor.submit(worker, task)
                pending[future] = (task, time.monotonic())

            next_deadline = min(
                task_started + task_timeout for _, task_started in pending.values()
            )
            done, _ = wait(
                pending,
                timeout=max(0.0, next_deadline - time.monotonic()),
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                task, _ = pending.pop(future)
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        _worker_failure_result(
                            task,
                            f"Worker exception: {exc}",
                            "WorkerException",
                        )
                    )

            now = time.monotonic()
            expired = [
                future
                for future, (_, task_started) in pending.items()
                if now - task_started >= task_timeout
            ]
            if expired:
                expired_set = set(expired)
                retry = [
                    task
                    for future, (task, _) in pending.items()
                    if future not in expired_set
                ]
                for future in expired:
                    task, _ = pending[future]
                    results.append(
                        _worker_failure_result(
                            task,
                            f"Worker timeout after {task_timeout:g} seconds",
                            "WorkerTimeout",
                        )
                    )
                pending.clear()
                _terminate_executor(executor)
                executor = None
                queued.extendleft(reversed(retry))
                if queued:
                    executor = ProcessPoolExecutor(
                        max_workers=max_workers,
                        initializer=_init_worker,
                    )

            completed = len(results)
            if completed and (completed % 200 == 0 or completed == total):
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                successes = sum(1 for result in results if result["success"])
                failures = completed - successes
                print(
                    f"  Progress: {completed}/{total} "
                    f"({100 * completed / total:.1f}%) | "
                    f"OK: {successes} | FAIL: {failures} | "
                    f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s"
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    return results, time.monotonic() - started


def find_all_pdfs():
    """Find all PDF files in the corpus directory."""
    pdfs = sorted(CORPUS_DIR.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDF files in corpus")
    return pdfs


def main():
    _configure_console_output()

    # Suppress most logging to keep output clean
    logging.basicConfig(level=logging.WARNING)
    # Suppress pikepdf and pdftopdfa debug logging
    logging.getLogger("pdftopdfa").setLevel(logging.ERROR)
    logging.getLogger("pikepdf").setLevel(logging.ERROR)
    # Suppress pikepdf "Unexpected end of stream" warnings from malformed PDFs
    warnings.filterwarnings(
        "ignore",
        message="Unexpected end of stream",
        module="pikepdf",
    )

    verapdf_version = get_verapdf_version()
    if verapdf_version is None:
        print(
            "ERROR: veraPDF is required for corpus release validation.",
            file=sys.stderr,
        )
        return 1
    print(f"Using {verapdf_version}")

    RESULTS_DIR.mkdir(exist_ok=True)

    pdfs = find_all_pdfs()
    if not pdfs:
        print("No PDFs found!")
        return 1

    total_tasks = len(pdfs) * len(LEVELS)
    print(
        f"Running {total_tasks} conversions ({len(pdfs)} PDFs × {len(LEVELS)} levels)"
    )
    print(f"Using {MAX_WORKERS} workers")
    print()

    # Use a temporary directory for outputs
    with tempfile.TemporaryDirectory(prefix="pdftopdfa_test_") as tmpdir:
        # Build task list
        tasks = []
        for pdf_path in pdfs:
            for level in LEVELS:
                tasks.append((pdf_path, level, tmpdir))

        all_results, total_time = _run_tasks(tasks)

    # Save raw results as JSON
    results_file = RESULTS_DIR / "raw_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nRaw results saved to: {results_file}")

    # Generate analysis
    analyze_results(all_results, total_time)
    return 1 if any(not result["success"] for result in all_results) else 0


def analyze_results(results, total_time):
    """Analyze and print results summary."""
    print("\n" + "=" * 80)
    print("CORPUS TEST RESULTS")
    print("=" * 80)

    total = len(results)
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    print(f"\nTotal conversions: {total}")
    print(f"Successful: {len(successes)} ({100 * len(successes) / total:.1f}%)")
    print(f"Failed: {len(failures)} ({100 * len(failures) / total:.1f}%)")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    skipped = [r for r in results if _is_skipped_result(r)]
    if skipped:
        unique_skipped = len({r["relative_path"] for r in skipped})
        print(
            f"Skipped: {len(skipped)} runs across {unique_skipped} file(s) "
            "(reported as successful passthroughs)"
        )

    # Results by level
    print("\n--- Results by PDF/A Level ---")
    for level in LEVELS:
        level_results = [r for r in results if r["level"] == level]
        level_ok = sum(1 for r in level_results if r["success"])
        level_fail = len(level_results) - level_ok
        print(
            f"  PDF/A-{level}: {level_ok}/{len(level_results)} OK "
            f"({100 * level_ok / len(level_results):.1f}%), "
            f"{level_fail} failures"
        )

    # Results by corpus source directory (top-level)
    print("\n--- Results by Corpus Category ---")
    by_category = defaultdict(lambda: {"ok": 0, "fail": 0})
    for r in results:
        category = _top_level_category(r["relative_path"])
        if r["success"]:
            by_category[category]["ok"] += 1
        else:
            by_category[category]["fail"] += 1

    for cat in sorted(by_category):
        info = by_category[cat]
        total_cat = info["ok"] + info["fail"]
        print(
            f"  {cat}: {info['ok']}/{total_cat} OK "
            f"({100 * info['ok'] / total_cat:.1f}%)"
        )

    # Error type breakdown
    print("\n--- Error Types ---")
    error_types = defaultdict(int)
    for r in failures:
        etype = r.get("error_type") or "Unknown"
        error_types[etype] += 1
    for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")

    if skipped:
        print("\n--- Skipped Conversion Reasons ---")
        skip_reasons = Counter()
        for r in skipped:
            matching_warning = next(
                (
                    warning
                    for warning in r.get("warnings", [])
                    if warning.startswith("Conversion skipped:")
                ),
                "Conversion skipped: reason not recorded",
            )
            skip_reasons[matching_warning] += 1
        for reason, count in skip_reasons.most_common():
            print(f"  [{count}x] {reason}")

    # Most common error messages (top 30)
    print("\n--- Top 30 Error Messages ---")
    error_msgs = defaultdict(int)
    for r in failures:
        msg = r.get("error") or "No error message"
        # Truncate long messages
        if len(msg) > 200:
            msg = msg[:200] + "..."
        error_msgs[msg] += 1
    for msg, count in sorted(error_msgs.items(), key=lambda x: -x[1])[:30]:
        print(f"  [{count}x] {msg}")

    # Failures by level and corpus category
    print("\n--- Failure Breakdown: Level × Category ---")
    cross = defaultdict(lambda: defaultdict(int))
    for r in failures:
        category = _top_level_category(r["relative_path"])
        cross[r["level"]][category] += 1

    for level in LEVELS:
        if cross[level]:
            print(f"\n  PDF/A-{level} failures:")
            for cat in sorted(cross[level]):
                print(f"    {cat}: {cross[level][cat]}")

    print("\n--- Performance Outliers ---")
    slowest_conversions = _collect_slowest_conversion_entries(results, limit=10)
    slowest_files = _collect_slowest_file_entries(results, limit=10)
    metadata_paths = {
        entry["relative_path"] for entry in slowest_conversions + slowest_files
    }
    metadata_by_path = {
        relative_path: _read_pdf_metadata(relative_path)
        for relative_path in metadata_paths
    }

    print("\n  Slowest files across all levels:")
    for entry in slowest_files:
        metadata = metadata_by_path[entry["relative_path"]]
        context = _format_runtime_entry(
            {"processing_time": entry["avg_processing_time"]},
            metadata,
        )
        skip_suffix = " | skipped on all levels" if entry["all_skipped"] else ""
        print(
            f"    max {entry['max_processing_time']:.2f}s | "
            f"avg {entry['avg_processing_time']:.2f}s | "
            f"levels {','.join(entry['levels'])} | "
            f"{context}{skip_suffix} | {entry['relative_path']}"
        )

    print("\n  Slowest individual conversions:")
    for entry in slowest_conversions:
        metadata = metadata_by_path[entry["relative_path"]]
        context = _format_runtime_entry(entry, metadata)
        outcome = "SKIP" if entry["skipped"] else ("OK" if entry["success"] else "FAIL")
        print(
            f"    {entry['processing_time']:.2f}s | PDF/A-{entry['level']} | "
            f"{outcome} | {context} | {entry['relative_path']}"
        )

    # Validation/pre-check log analysis
    print("\n--- veraPDF / Validation Messages ---")
    message_counts = defaultdict(int)
    conversions_with_messages = 0
    for r in results:
        validation_messages = _result_status_messages(r)
        if validation_messages:
            conversions_with_messages += 1
        for message in validation_messages:
            if len(message) > 200:
                message = message[:200] + "..."
            message_counts[message] += 1
    print(f"  Conversions with captured messages: {conversions_with_messages}")
    if message_counts:
        for message, count in sorted(
            message_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )[:20]:
            print(f"  [{count}x] {message}")
    else:
        print("  No validation messages captured")

    # Warnings analysis
    print("\n--- Warnings Summary ---")
    warning_counts = defaultdict(int)
    for r in results:
        for w in r.get("warnings", []):
            if len(w) > 200:
                w = w[:200] + "..."
            warning_counts[w] += 1
    if warning_counts:
        for w, count in sorted(warning_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  [{count}x] {w}")
    else:
        print("  No warnings recorded")

    # Files that fail on ALL levels
    print("\n--- Files Failing on ALL Levels ---")
    file_failures = defaultdict(set)
    for r in failures:
        file_failures[r["relative_path"]].add(r["level"])
    all_level_failures = {
        f: levels for f, levels in file_failures.items() if levels == set(LEVELS)
    }
    print(f"  {len(all_level_failures)} files fail on all {len(LEVELS)} levels")

    # Files that fail on SOME levels only
    partial_failures = {
        f: levels for f, levels in file_failures.items() if levels != set(LEVELS)
    }
    print(f"  {len(partial_failures)} files fail on some levels only")

    if partial_failures:
        print("\n  Partial failure details (first 30):")
        for f, levels in sorted(partial_failures.items())[:30]:
            print(f"    {f}: fails on {sorted(levels)}")

    # Save detailed report
    report_file = RESULTS_DIR / "analysis_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("DETAILED FAILURE REPORT\n")
        f.write("=" * 80 + "\n\n")

        skipped = [r for r in results if _is_skipped_result(r)]
        if skipped:
            f.write("SKIPPED CONVERSIONS\n")
            f.write("=" * 80 + "\n\n")
            skip_reasons = Counter()
            for r in skipped:
                matching_warning = next(
                    (
                        warning
                        for warning in r.get("warnings", [])
                        if warning.startswith("Conversion skipped:")
                    ),
                    "Conversion skipped: reason not recorded",
                )
                skip_reasons[matching_warning] += 1

            f.write(
                f"Skipped runs: {len(skipped)} "
                f"across {len({r['relative_path'] for r in skipped})} file(s)\n\n"
            )
            for reason, count in skip_reasons.most_common():
                f.write(f"--- [{count}x] {reason} ---\n\n")

        slowest_conversions = _collect_slowest_conversion_entries(results, limit=10)
        slowest_files = _collect_slowest_file_entries(results, limit=10)
        metadata_paths = {
            entry["relative_path"] for entry in slowest_conversions + slowest_files
        }
        metadata_by_path = {
            relative_path: _read_pdf_metadata(relative_path)
            for relative_path in metadata_paths
        }

        f.write("PERFORMANCE OUTLIERS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Slowest files across all levels:\n")
        for entry in slowest_files:
            metadata = metadata_by_path[entry["relative_path"]]
            context = _format_runtime_entry(
                {"processing_time": entry["avg_processing_time"]},
                metadata,
            )
            skip_suffix = " | skipped on all levels" if entry["all_skipped"] else ""
            f.write(
                f"  Max {entry['max_processing_time']:.2f}s | "
                f"Avg {entry['avg_processing_time']:.2f}s | "
                f"Levels {','.join(entry['levels'])} | "
                f"{context}{skip_suffix}\n"
            )
            f.write(f"    File: {entry['relative_path']}\n")
        f.write("\nSlowest individual conversions:\n")
        for entry in slowest_conversions:
            metadata = metadata_by_path[entry["relative_path"]]
            context = _format_runtime_entry(entry, metadata)
            outcome = (
                "SKIP" if entry["skipped"] else ("OK" if entry["success"] else "FAIL")
            )
            f.write(
                f"  {entry['processing_time']:.2f}s | PDF/A-{entry['level']} | "
                f"{outcome} | {context}\n"
            )
            f.write(f"    File: {entry['relative_path']}\n")
        f.write("\n")

        validation_message_groups = defaultdict(list)
        for r in sorted(results, key=lambda x: (x["relative_path"], x["level"])):
            for message in _result_status_messages(r):
                validation_message_groups[message].append(r)

        f.write("VERAPDF / VALIDATION MESSAGE REPORT\n")
        f.write("=" * 80 + "\n\n")
        if validation_message_groups:
            for message, items in sorted(
                validation_message_groups.items(),
                key=lambda x: (-len(x[1]), x[0]),
            ):
                f.write(f"--- [{len(items)}x] {message} ---\n")
                for r in items[:50]:
                    f.write(f"  File: {r['relative_path']}\n")
                    f.write(f"  Level: {r['level']}\n\n")
                if len(items) > 50:
                    f.write(f"  ... and {len(items) - 50} more\n\n")
        else:
            f.write("No validation messages captured.\n\n")

        for level in LEVELS:
            level_failures = [r for r in failures if r["level"] == level]
            f.write(f"\n{'=' * 80}\n")
            f.write(f"PDF/A-{level} FAILURES ({len(level_failures)} total)\n")
            f.write(f"{'=' * 80}\n\n")

            # Group by error type
            by_error = defaultdict(list)
            for r in level_failures:
                by_error[r.get("error_type", "Unknown")].append(r)

            for etype in sorted(by_error):
                items = by_error[etype]
                f.write(f"\n--- {etype} ({len(items)} failures) ---\n")
                for r in items[:50]:  # Limit to 50 per error type
                    f.write(f"  File: {r['relative_path']}\n")
                    f.write(f"  Error: {r.get('error', 'N/A')}\n")
                    f.write("\n")
                if len(items) > 50:
                    f.write(f"  ... and {len(items) - 50} more\n\n")

    print(f"\nDetailed report saved to: {report_file}")

    # Save CSV summary
    csv_file = RESULTS_DIR / "results_summary.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "level",
                "success",
                "validation_failed",
                "skipped",
                "error_type",
                "error",
                "processing_time",
                "validation_messages",
            ],
        )
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x["relative_path"], x["level"])):
            writer.writerow(
                {
                    "relative_path": r["relative_path"],
                    "level": r["level"],
                    "success": r["success"],
                    "validation_failed": r.get("validation_failed", False),
                    "skipped": _is_skipped_result(r),
                    "error_type": r.get("error_type", ""),
                    "error": r.get("error", ""),
                    "processing_time": r.get("processing_time", 0),
                    "validation_messages": " | ".join(_result_status_messages(r)),
                }
            )
    print(f"CSV summary saved to: {csv_file}")


if __name__ == "__main__":
    raise SystemExit(main())
