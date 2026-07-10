# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Font loading for PDF/A compliance."""

import logging
import os
import threading
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from ..exceptions import FontEmbeddingError
from .constants import (
    CIDFONT_REPLACEMENT,
    CJK_FONT_INDEX,
    FALLBACK_FONT,
    FONT_REPLACEMENTS,
    WINDOWS_SYSTEM_FONT_POSTSCRIPT_NAMES,
    resolve_standard14_alias,
)
from .utils import check_fstype_restrictions, get_fstype, is_permitted_fstype_notice

if TYPE_CHECKING:
    from fontTools.ttLib import TTFont

logger = logging.getLogger(__name__)
_WINDOWS_ALLOWED_SYSTEM_FONT_NAMES = frozenset(
    name.replace(" ", "").lower() for name in WINDOWS_SYSTEM_FONT_POSTSCRIPT_NAMES
)

# Process-wide cache of the scanned system font index, keyed by the fonts
# directory. Building the index walks every font file under %WINDIR%\Fonts
# and parses it with fontTools, which is expensive; the result is stable
# for the lifetime of the process.
_SYSTEM_FONT_INDEX_LOCK = threading.Lock()
_SYSTEM_FONT_INDEX_CACHE: dict[str, dict[str, tuple[Path, int | None]]] = {}


class FontLoader:
    """Loads and caches font files.

    This helper class handles loading replacement fonts from resources
    and caching them for reuse.
    """

    def __init__(self, font_cache: dict[str, tuple[bytes, "TTFont"]]) -> None:
        """Initializes the FontLoader.

        Args:
            font_cache: Shared cache dictionary for loaded fonts.
        """
        self._font_cache = font_cache
        self._system_font_index: dict[str, tuple[Path, int | None]] | None = None

    def load_standard14_font(self, font_name: str) -> tuple[bytes, "TTFont"]:
        """Loads a bundled replacement font for Standard-14 fonts.

        Args:
            font_name: Name of the Standard-14 font.

        Returns:
            Tuple of (font data as bytes, TTFont object).

        Raises:
            FontEmbeddingError: If the font cannot be loaded.
        """
        from fontTools.ttLib import TTFont

        canonical_name = resolve_standard14_alias(font_name)
        if canonical_name in self._font_cache:
            return self._font_cache[canonical_name]

        replacement_file = FONT_REPLACEMENTS.get(canonical_name)
        if replacement_file is None:
            raise FontEmbeddingError(f"No replacement defined for font '{font_name}'")

        # Load font from resources
        try:
            font_ref = (
                resources.files("pdftopdfa") / "resources" / "fonts" / replacement_file
            )
            font_data = font_ref.read_bytes()
        except Exception as e:
            raise FontEmbeddingError(
                f"Could not load replacement font '{replacement_file}': {e}"
            ) from e

        # Parse font with fonttools
        self._ensure_embedding_allowed(
            font_data,
            font_label=replacement_file,
        )
        tt_font = TTFont(BytesIO(font_data))
        self._font_cache[canonical_name] = (font_data, tt_font)
        return font_data, tt_font

    def load_replacement_font(
        self,
        font_name: str,
        *,
        use_fallback: bool = False,
    ) -> tuple[bytes, "TTFont"]:
        """Load a replacement font, preferring policy-approved Windows fonts."""
        system_match = self._load_matching_system_font(font_name)
        if system_match is not None:
            return system_match

        if use_fallback:
            return self.load_fallback_font()
        return self.load_standard14_font(font_name)

    def load_fallback_font(self) -> tuple[bytes, "TTFont"]:
        """Loads the fallback font (LiberationSans) for unknown fonts.

        Used when a non-embedded font has no specific replacement
        in FONT_REPLACEMENTS.

        Returns:
            Tuple of (font data as bytes, TTFont object).

        Raises:
            FontEmbeddingError: If the font cannot be loaded.
        """
        from fontTools.ttLib import TTFont

        cache_key = "__fallback__"
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        try:
            font_ref = (
                resources.files("pdftopdfa") / "resources" / "fonts" / FALLBACK_FONT
            )
            font_data = font_ref.read_bytes()
        except Exception as e:
            raise FontEmbeddingError(
                f"Could not load fallback font '{FALLBACK_FONT}': {e}"
            ) from e

        self._ensure_embedding_allowed(
            font_data,
            font_label=FALLBACK_FONT,
        )
        tt_font = TTFont(BytesIO(font_data))
        self._font_cache[cache_key] = (font_data, tt_font)
        return font_data, tt_font

    def load_cidfont_replacement_by_ordering(
        self, ordering: str
    ) -> tuple[bytes, "TTFont"]:
        """Loads the CIDFont replacement font for a specific CJK ordering.

        Selects the correct font from the TTC based on CJK_FONT_INDEX.
        Falls back to index 0 (Simplified Chinese) for unknown orderings.

        Args:
            ordering: CIDSystemInfo Ordering value (e.g. "Japan1", "GB1").

        Returns:
            Tuple of (font data as bytes, TTFont object).

        Raises:
            FontEmbeddingError: If the font cannot be loaded.
        """
        from fontTools.ttLib import TTCollection, TTFont

        font_index = CJK_FONT_INDEX.get(ordering, 0)
        cache_key = f"__cidfont_{font_index}__"

        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        # Load font from resources
        try:
            font_ref = (
                resources.files("pdftopdfa")
                / "resources"
                / "fonts"
                / CIDFONT_REPLACEMENT
            )
            font_data = font_ref.read_bytes()
        except Exception as e:
            raise FontEmbeddingError(
                f"Could not load CIDFont replacement font '{CIDFONT_REPLACEMENT}': {e}"
            ) from e

        # Parse font with fonttools (TTC = TrueType Collection)
        from io import BytesIO

        if CIDFONT_REPLACEMENT.endswith(".ttc"):
            ttc = TTCollection(BytesIO(font_data))
            try:
                if font_index >= len(ttc.fonts):
                    font_index = 0
                # Serialize the single font for FontFile2
                buf = BytesIO()
                ttc.fonts[font_index].save(buf)
                font_data = buf.getvalue()
            finally:
                ttc.close()
            self._ensure_embedding_allowed(
                font_data,
                font_label=CIDFONT_REPLACEMENT,
            )
            tt_font = TTFont(BytesIO(font_data))
        else:
            self._ensure_embedding_allowed(
                font_data,
                font_label=CIDFONT_REPLACEMENT,
            )
            tt_font = TTFont(BytesIO(font_data))

        self._font_cache[cache_key] = (font_data, tt_font)
        return font_data, tt_font

    def _load_matching_system_font(
        self,
        font_name: str,
    ) -> tuple[bytes, "TTFont"] | None:
        """Load a policy-approved Windows system font for ``font_name``."""
        from fontTools.ttLib import TTFont

        cache_key = f"__system__::{font_name}"
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        if not self._is_windows_font_request_allowed(font_name):
            return None

        match = self._build_system_font_index().get(
            self._normalize_font_lookup_name(font_name)
        )
        if match is None:
            return None

        font_path, font_number = match
        if not self._is_windows_fonts_path(font_path):
            logger.warning(
                "Ignoring system font '%s' outside %%WINDIR%%\\Fonts: %s",
                font_name,
                font_path,
            )
            return None

        try:
            font_data = self._load_system_font_bytes(font_path, font_number)
            tt_font = TTFont(BytesIO(font_data))
        except Exception:
            return None

        actual_ps_name = self._extract_postscript_name(tt_font)
        if not self._is_windows_font_request_allowed(actual_ps_name):
            logger.warning(
                "Ignoring system font '%s' because resolved PostScript name "
                "'%s' is not allowlisted",
                font_name,
                actual_ps_name or "Unknown",
            )
            tt_font.close()
            return None

        font_label = (
            f"{font_path}#{font_number}" if font_number is not None else str(font_path)
        )
        if not self._ensure_embedding_allowed(
            font_data,
            font_label=font_label,
            raise_on_disallowed=False,
        ):
            tt_font.close()
            return None

        self._font_cache[cache_key] = (font_data, tt_font)
        return font_data, tt_font

    @staticmethod
    def _ensure_embedding_allowed(
        font_data: bytes,
        *,
        font_label: str,
        raise_on_disallowed: bool = True,
    ) -> bool:
        """Validate fsType rights for full-font embedding."""
        fstype = get_fstype(font_data)
        if fstype is None:
            return True

        embedding_allowed, _subsetting_allowed, warnings = check_fstype_restrictions(
            fstype
        )
        if embedding_allowed:
            for warning in warnings:
                log_message = "Font '%s': %s"
                if is_permitted_fstype_notice(warning):
                    logger.info(log_message, font_label, warning)
                else:
                    logger.warning(log_message, font_label, warning)
            return True

        warning_text = ", ".join(warnings) if warnings else f"fsType={fstype}"
        message = f"Font '{font_label}' cannot be embedded as outlines: {warning_text}"
        if raise_on_disallowed:
            raise FontEmbeddingError(message)

        logger.warning(message)
        return False

    @classmethod
    def _normalize_font_lookup_name(cls, font_name: str) -> str:
        """Normalize font names for exact-ish system lookup."""
        return font_name.lstrip("/").replace(" ", "").lower()

    def _build_system_font_index(self) -> dict[str, tuple[Path, int | None]]:
        """Index policy-eligible Windows system fonts by PostScript name.

        The scan result is cached process-wide (keyed by the fonts
        directory) so repeated conversions do not rebuild the index.
        """
        if self._system_font_index is not None:
            return self._system_font_index

        fonts_root = self._get_windows_fonts_dir()
        cache_key = os.path.normcase(str(fonts_root)) if fonts_root is not None else ""

        with _SYSTEM_FONT_INDEX_LOCK:
            index = _SYSTEM_FONT_INDEX_CACHE.get(cache_key)
            if index is None:
                index = self._scan_system_font_files()
                _SYSTEM_FONT_INDEX_CACHE[cache_key] = index

        self._system_font_index = index
        return index

    def _scan_system_font_files(self) -> dict[str, tuple[Path, int | None]]:
        """Scan the system fonts directory and index allowlisted fonts."""
        from fontTools.ttLib import TTCollection, TTFont

        index: dict[str, tuple[Path, int | None]] = {}
        for font_path in self._iter_system_font_files():
            suffix = font_path.suffix.lower()
            try:
                if suffix in {".ttc", ".otc"}:
                    collection = TTCollection(str(font_path))
                    try:
                        for font_number, tt_font in enumerate(collection.fonts):
                            candidate = self._extract_postscript_name(tt_font)
                            if candidate is None:
                                continue
                            normalized = self._normalize_font_lookup_name(candidate)
                            if normalized in _WINDOWS_ALLOWED_SYSTEM_FONT_NAMES:
                                index.setdefault(normalized, (font_path, font_number))
                    finally:
                        collection.close()
                else:
                    tt_font = TTFont(str(font_path), lazy=True)
                    try:
                        candidate = self._extract_postscript_name(tt_font)
                        if candidate is None:
                            continue
                        normalized = self._normalize_font_lookup_name(candidate)
                        if normalized in _WINDOWS_ALLOWED_SYSTEM_FONT_NAMES:
                            index.setdefault(normalized, (font_path, None))
                    finally:
                        tt_font.close()
            except Exception:
                continue

        return index

    @classmethod
    def _extract_postscript_name(cls, tt_font: "TTFont") -> str | None:
        """Extract the actual PostScript name (nameID 6) from a font."""
        name_table = tt_font.get("name")
        if name_table is None:
            return None

        for record in name_table.names:
            if record.nameID != 6:
                continue
            try:
                text = record.toUnicode()
            except Exception:
                continue
            if text:
                return text

        return None

    @staticmethod
    def _iter_system_font_files() -> list[Path]:
        """Return candidate font files from ``%WINDIR%\\Fonts`` on Windows."""
        fonts_root = FontLoader._get_windows_fonts_dir()
        if fonts_root is None or not fonts_root.exists():
            return []

        result: list[Path] = []
        for pattern in ("*.ttf", "*.otf", "*.ttc", "*.otc"):
            result.extend(fonts_root.rglob(pattern))
        return result

    @staticmethod
    def _is_windows_platform() -> bool:
        """Return whether policy-controlled system font loading is enabled."""
        return os.name == "nt"

    @classmethod
    def _get_windows_fonts_dir(cls) -> Path | None:
        """Return the canonical ``%WINDIR%\\Fonts`` directory."""
        if not cls._is_windows_platform():
            return None

        windir = os.environ.get("WINDIR")
        if not windir:
            return None
        return Path(windir) / "Fonts"

    @classmethod
    def _is_windows_fonts_path(cls, font_path: Path) -> bool:
        """Check whether a font file resides under ``%WINDIR%\\Fonts``."""
        fonts_dir = cls._get_windows_fonts_dir()
        if fonts_dir is None:
            return False

        normalized_font_path = os.path.normcase(str(font_path.resolve(strict=False)))
        normalized_fonts_dir = os.path.normcase(str(fonts_dir.resolve(strict=False)))
        try:
            return os.path.commonpath([normalized_font_path, normalized_fonts_dir]) == (
                normalized_fonts_dir
            )
        except ValueError:
            return False

    @classmethod
    def _is_windows_font_request_allowed(cls, font_name: str | None) -> bool:
        """Check whether a requested or resolved PostScript name is allowlisted."""
        if not cls._is_windows_platform() or not font_name:
            return False
        return cls._normalize_font_lookup_name(font_name) in (
            _WINDOWS_ALLOWED_SYSTEM_FONT_NAMES
        )

    @staticmethod
    def _load_system_font_bytes(
        font_path: Path,
        font_number: int | None,
    ) -> bytes:
        """Load and serialize a single font program from a font file or TTC."""
        from fontTools.ttLib import TTCollection

        if font_number is None:
            return font_path.read_bytes()

        collection = TTCollection(str(font_path))
        try:
            buffer = BytesIO()
            collection.fonts[font_number].save(buffer)
            return buffer.getvalue()
        finally:
            collection.close()
