# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Private same-filesystem staging for validated output publication."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .exceptions import ConversionError


@dataclass(frozen=True, slots=True)
class StagedFileSnapshot:
    """Identity and content digest of one regular staged file."""

    device: int
    inode: int
    size: int
    modified_ns: int
    sha256: str


def private_staging_directory(
    parent: Path,
    *,
    prefix: str,
    delete: bool = True,
) -> TemporaryDirectory[str]:
    """Create a mode-0700 staging directory on the destination filesystem.

    The directory is a ``<prefix><random>`` sibling of the output file, so
    publication is a rename within one filesystem. Callers must call
    ``cleanup()`` themselves; an interrupted process can leave the directory
    behind, which is why the prefix identifies the output it belongs to.

    Pass ``delete=False`` when the caller may deliberately abandon the
    directory. A retained staging directory can hold the only surviving copy of
    a destination whose rollback failed, and an armed weakref finalizer would
    delete exactly that copy once the object is collected.
    """

    parent.mkdir(parents=True, exist_ok=True)
    return TemporaryDirectory(prefix=prefix, dir=parent, delete=delete)


def copy_to_private_stage(source: Path, directory: Path, name: str) -> Path:
    """Copy ``source`` through an exclusive file descriptor into ``directory``."""

    destination = directory / name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(destination, flags, 0o600)
    try:
        with source.open("rb") as source_file, os.fdopen(fd, "wb") as output_file:
            fd = -1
            shutil.copyfileobj(source_file, output_file)
        shutil.copystat(source, destination, follow_symlinks=False)
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        if fd >= 0:
            os.close(fd)
    return destination


def staged_file_snapshot(path: Path) -> StagedFileSnapshot:
    """Hash a regular staged file while verifying its pathname identity."""

    try:
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode):
            raise ConversionError(f"Staged output is not a regular file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            opened_stat = os.fstat(stream.fileno())
            if (
                opened_stat.st_dev != path_stat.st_dev
                or opened_stat.st_ino != path_stat.st_ino
            ):
                raise ConversionError(
                    f"Staged output identity changed while opening: {path}"
                )
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            final_stat = os.fstat(stream.fileno())
        final_path_stat = path.lstat()
    except ConversionError:
        raise
    except OSError as exc:
        raise ConversionError(f"Could not inspect staged output {path}: {exc}") from exc

    identity = (path_stat.st_dev, path_stat.st_ino)
    for value in (opened_stat, final_stat, final_path_stat):
        if (value.st_dev, value.st_ino) != identity:
            raise ConversionError(f"Staged output identity changed: {path}")
    if (
        final_stat.st_size != path_stat.st_size
        or final_stat.st_mtime_ns != path_stat.st_mtime_ns
        or final_path_stat.st_size != path_stat.st_size
        or final_path_stat.st_mtime_ns != path_stat.st_mtime_ns
    ):
        raise ConversionError(f"Staged output changed while being inspected: {path}")
    return StagedFileSnapshot(
        device=int(path_stat.st_dev),
        inode=int(path_stat.st_ino),
        size=int(path_stat.st_size),
        modified_ns=int(path_stat.st_mtime_ns),
        sha256=digest.hexdigest(),
    )


def verify_staged_file(path: Path, expected: StagedFileSnapshot) -> None:
    """Fail if the staged pathname or bytes differ from ``expected``."""

    if staged_file_snapshot(path) != expected:
        raise ConversionError(f"Staged output changed after validation: {path}")


def _snapshot_matches_candidate(
    actual: StagedFileSnapshot,
    expected: StagedFileSnapshot,
) -> bool:
    """Return whether ``actual`` is the validated file under a new pathname."""

    return (
        actual.device == expected.device
        and actual.inode == expected.inode
        and actual.size == expected.size
        and actual.sha256 == expected.sha256
    )


def _snapshot_matches_content(
    actual: StagedFileSnapshot,
    expected: StagedFileSnapshot,
) -> bool:
    """Return whether ``actual`` holds the validated bytes under any identity."""

    return actual.size == expected.size and actual.sha256 == expected.sha256


# FAT32, exFAT and several network filesystems reject hard links outright.
# Publication then falls back to an exclusive copy, which loses the file
# identity but keeps both the overwrite protection and the exact bytes.
_UNSUPPORTED_LINK_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, name, None)
        for name in ("EPERM", "ENOSYS", "EXDEV", "EOPNOTSUPP", "ENOTSUP", "EINVAL")
    )
    if value is not None
)
_UNSUPPORTED_LINK_WINERRORS = frozenset({1, 50, 87})


def _link_unsupported(exc: OSError) -> bool:
    """Return whether ``exc`` means the filesystem has no hard links."""

    winerror = getattr(exc, "winerror", None)
    if winerror is not None and winerror in _UNSUPPORTED_LINK_WINERRORS:
        return True
    return exc.errno in _UNSUPPORTED_LINK_ERRNOS


def _stat_matches_snapshot(value: os.stat_result, expected: StagedFileSnapshot) -> bool:
    """Return whether pathname metadata still identifies ``expected`` exactly."""

    return (
        stat.S_ISREG(value.st_mode)
        and value.st_dev == expected.device
        and value.st_ino == expected.inode
        and value.st_size == expected.size
        and value.st_mtime_ns == expected.modified_ns
    )


def _stat_has_snapshot_identity(
    value: os.stat_result,
    expected: StagedFileSnapshot,
) -> bool:
    """Return whether a regular pathname is still the snapshotted file."""

    return (
        stat.S_ISREG(value.st_mode)
        and value.st_dev == expected.device
        and value.st_ino == expected.inode
    )


def _windows_api_path(path: Path) -> str:
    """Return an absolute extended-length path for a Unicode Windows API."""

    value = os.path.abspath(path)
    if value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return f"\\\\?\\UNC\\{value[2:]}"
    return f"\\\\?\\{value}"


def _apply_windows_parent_dacl(
    path: Path,
    parent: Path,
    *,
    template: Path | None = None,
) -> None:
    """Give a staged file its destination-compatible DACL."""

    import ctypes
    from ctypes import wintypes

    class GenericMapping(ctypes.Structure):
        _fields_ = [
            ("generic_read", wintypes.DWORD),
            ("generic_write", wintypes.DWORD),
            ("generic_execute", wintypes.DWORD),
            ("generic_all", wintypes.DWORD),
        ]

    owner_security_information = 0x00000001
    group_security_information = 0x00000002
    dacl_security_information = 0x00000004
    se_file_object = 1
    se_dacl_auto_inherit = 0x00000001
    se_avoid_privilege_check = 0x00000008
    se_avoid_owner_check = 0x00000010
    se_default_owner_from_parent = 0x00000020
    se_default_group_from_parent = 0x00000040
    unsupported_errors = {1, 50, 120}

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    parent_descriptor = ctypes.c_void_p()
    template_descriptor = ctypes.c_void_p()
    child_descriptor = ctypes.c_void_p()

    get_security = advapi32.GetNamedSecurityInfoW
    get_security.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    get_security.restype = wintypes.DWORD
    result = get_security(
        _windows_api_path(parent),
        se_file_object,
        owner_security_information
        | group_security_information
        | dacl_security_information,
        None,
        None,
        None,
        None,
        ctypes.byref(parent_descriptor),
    )
    if result in unsupported_errors:
        return
    if result:
        raise ctypes.WinError(result)

    destroy_security = advapi32.DestroyPrivateObjectSecurity
    destroy_security.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    destroy_security.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p
    try:
        if template is not None:
            result = get_security(
                _windows_api_path(template),
                se_file_object,
                owner_security_information
                | group_security_information
                | dacl_security_information,
                None,
                None,
                None,
                None,
                ctypes.byref(template_descriptor),
            )
            if result in unsupported_errors:
                return
            if result:
                raise ctypes.WinError(result)

        create_security = advapi32.CreatePrivateObjectSecurityEx
        create_security.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.ULONG,
            wintypes.HANDLE,
            ctypes.POINTER(GenericMapping),
        ]
        create_security.restype = wintypes.BOOL
        file_mapping = GenericMapping(
            0x00120089,
            0x00120116,
            0x001200A0,
            0x001F01FF,
        )
        if not create_security(
            parent_descriptor,
            template_descriptor,
            ctypes.byref(child_descriptor),
            None,
            False,
            se_dacl_auto_inherit
            | se_avoid_privilege_check
            | se_avoid_owner_check
            | se_default_owner_from_parent
            | se_default_group_from_parent,
            None,
            ctypes.byref(file_mapping),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        set_security = advapi32.SetFileSecurityW
        set_security.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        set_security.restype = wintypes.BOOL
        if not set_security(
            _windows_api_path(path),
            dacl_security_information,
            child_descriptor,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if child_descriptor.value:
            destroy_security(ctypes.byref(child_descriptor))
        if template_descriptor.value:
            local_free(template_descriptor)
        local_free(parent_descriptor)


def _preserve_posix_destination_metadata(staged: Path, destination: Path) -> None:
    """Apply an existing POSIX destination's ownership and metadata."""

    destination_stat = destination.stat(follow_symlinks=False)
    staged_stat = staged.stat(follow_symlinks=False)
    try:
        if (
            staged_stat.st_uid != destination_stat.st_uid
            or staged_stat.st_gid != destination_stat.st_gid
        ):
            os.chown(
                staged,
                destination_stat.st_uid,
                destination_stat.st_gid,
                follow_symlinks=False,
            )
        shutil.copystat(destination, staged, follow_symlinks=False)
    except OSError as exc:
        raise ConversionError(
            f"Could not preserve destination metadata for {destination}: {exc}"
        ) from exc


def _publish_exclusive_copy(
    staged: Path,
    destination: Path,
    candidate: StagedFileSnapshot,
) -> StagedFileSnapshot:
    """Publish onto a filesystem without hard links, still refusing to overwrite.

    ``copy_to_private_stage`` creates the target with ``O_EXCL``, so a target
    that appeared since the caller's check is reported instead of replaced. The
    published file is a distinct inode, so it is verified by content.
    """

    try:
        copy_to_private_stage(staged, destination.parent, destination.name)
    except FileExistsError as exc:
        raise ConversionError(
            f"Publication target already exists: {destination}"
        ) from exc
    except OSError as exc:
        raise ConversionError(
            f"Could not publish staged output {destination}: {exc}"
        ) from exc

    try:
        published = staged_file_snapshot(destination)
        if not _snapshot_matches_content(published, candidate):
            raise ConversionError(
                f"Published output differs from validated candidate: {destination}"
            )
        staged.unlink()
    except BaseException as exc:
        try:
            destination.unlink(missing_ok=True)
        except OSError as recovery_exc:
            exc.add_note(
                "Publication recovery failed: "
                f"{type(recovery_exc).__name__}: {recovery_exc}"
            )
        raise
    return published


def publish_staged_file(
    staged: Path,
    destination: Path,
    expected: StagedFileSnapshot,
    *,
    backup: Path | None = None,
    require_absent: bool = False,
) -> StagedFileSnapshot:
    """Atomically publish the exact validated file, optionally retaining the target."""

    staged = Path(staged)
    destination = Path(destination)
    backup = Path(backup) if backup is not None else None
    verify_staged_file(staged, expected)
    try:
        destination_stat = destination.lstat()
    except FileNotFoundError:
        destination_stat = None
    except OSError as exc:
        raise ConversionError(
            f"Could not inspect publication target {destination}: {exc}"
        ) from exc

    if destination_stat is not None and not stat.S_ISREG(destination_stat.st_mode):
        raise ConversionError(
            f"Publication target is not a regular file: {destination}"
        )
    if require_absent and destination_stat is not None:
        raise ConversionError(f"Publication target already exists: {destination}")
    if backup is not None and backup.exists():
        raise ConversionError(f"Publication backup already exists: {backup}")

    candidate: StagedFileSnapshot | None = None
    original: StagedFileSnapshot | None = None
    published_by_this_call = False
    if destination_stat is None:
        if sys.platform == "win32":
            _apply_windows_parent_dacl(staged, destination.parent)
        candidate = staged_file_snapshot(staged)
        if not _snapshot_matches_candidate(candidate, expected):
            raise ConversionError(
                f"Staged output changed while preparing publication: {staged}"
            )
        if require_absent:
            linked = False
            try:
                os.link(staged, destination, follow_symlinks=False)
                linked = True
            except FileExistsError as exc:
                raise ConversionError(
                    f"Publication target already exists: {destination}"
                ) from exc
            except OSError as exc:
                if not _link_unsupported(exc):
                    raise ConversionError(
                        f"Could not publish staged output {destination}: {exc}"
                    ) from exc
            if not linked:
                return _publish_exclusive_copy(staged, destination, candidate)
            published_by_this_call = True
        else:
            os.replace(staged, destination)
            published_by_this_call = True
    else:
        if sys.platform == "win32":
            _apply_windows_parent_dacl(
                staged,
                destination.parent,
                template=destination,
            )
        else:
            _preserve_posix_destination_metadata(staged, destination)
        candidate = staged_file_snapshot(staged)
        if not _snapshot_matches_candidate(candidate, expected):
            raise ConversionError(
                f"Staged output changed while preparing publication: {staged}"
            )
        if backup is not None:
            try:
                try:
                    os.link(destination, backup, follow_symlinks=False)
                except OSError as exc:
                    if not _link_unsupported(exc):
                        raise
                    copy_to_private_stage(destination, backup.parent, backup.name)
                original = staged_file_snapshot(backup)
            except OSError as exc:
                raise ConversionError(
                    f"Could not retain publication target {destination}: {exc}"
                ) from exc
            except BaseException:
                backup.unlink(missing_ok=True)
                raise
        os.replace(staged, destination)
        published_by_this_call = True

    try:
        published = staged_file_snapshot(destination)
        if not _snapshot_matches_candidate(published, candidate):
            raise ConversionError(
                f"Published output differs from validated candidate: {destination}"
            )
        if require_absent:
            staged.unlink()
    except BaseException as exc:
        if (
            not published_by_this_call
            or candidate is None
            or (destination_stat is not None and original is None)
        ):
            raise
        try:
            rollback_staged_publication(
                destination,
                candidate,
                original=original,
                backup=backup if original is not None else None,
            )
        except BaseException as recovery_exc:
            exc.add_note(
                "Publication recovery failed: "
                f"{type(recovery_exc).__name__}: {recovery_exc}"
            )
        raise
    return published


def rollback_staged_publication(
    destination: Path,
    candidate: StagedFileSnapshot,
    *,
    original: StagedFileSnapshot | None,
    backup: Path | None,
) -> None:
    """Idempotently restore a target after an interrupted staged publication."""

    destination = Path(destination)
    backup = Path(backup) if backup is not None else None
    try:
        current = staged_file_snapshot(destination)
    except ConversionError:
        try:
            current_stat = destination.lstat()
        except FileNotFoundError:
            current = None
            destination_exists = False
            candidate_is_current = False
        except OSError as exc:
            raise ConversionError(
                f"Could not inspect publication target {destination}: {exc}"
            ) from exc
        else:
            current = None
            destination_exists = True
            candidate_is_current = _stat_has_snapshot_identity(
                current_stat,
                candidate,
            )
            if not stat.S_ISREG(current_stat.st_mode):
                raise ConversionError(
                    f"Publication target changed before rollback: {destination}"
                )
    else:
        destination_exists = True
        candidate_is_current = (
            current.device == candidate.device and current.inode == candidate.inode
        )

    if original is None:
        if candidate_is_current:
            destination.unlink()
        elif destination_exists:
            raise ConversionError(
                f"Publication target changed before rollback: {destination}"
            )
        return

    if backup is None:
        raise ConversionError(f"Publication backup is missing for {destination}")
    if candidate_is_current or not destination_exists:
        if not backup.exists():
            raise ConversionError(f"Publication backup is missing: {backup}")
        verify_staged_file(backup, original)
        os.replace(backup, destination)
    elif backup.exists():
        # A hard-linked backup is literally the target; a copied backup - used
        # where the filesystem has no hard links - only shares its bytes.
        try:
            same_original = os.path.samefile(destination, backup)
        except OSError as exc:
            raise ConversionError(
                f"Could not compare publication backup {backup}: {exc}"
            ) from exc
        if not same_original and not (
            current is not None and _snapshot_matches_content(current, original)
        ):
            raise ConversionError(
                f"Publication target changed before rollback: {destination}"
            )
        backup.unlink()

    try:
        restored_stat = destination.lstat()
    except OSError as exc:
        raise ConversionError(
            f"Could not inspect restored publication target {destination}: {exc}"
        ) from exc
    # A copied backup was never the target's inode, so identity can only be
    # required when it could have been established in the first place.
    if not _stat_matches_snapshot(restored_stat, original) and not (
        stat.S_ISREG(restored_stat.st_mode)
        and _snapshot_matches_content(staged_file_snapshot(destination), original)
    ):
        raise ConversionError(
            f"Restored publication target differs from its backup: {destination}"
        )
