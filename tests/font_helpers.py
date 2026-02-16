# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Shared helper functions for font-related tests."""


def _liberation_fonts_available() -> bool:
    """Checks if Liberation fonts are available."""
    try:
        from importlib import resources

        font_ref = (
            resources.files("pdftopdfa")
            / "resources"
            / "fonts"
            / "LiberationSans-Regular.ttf"
        )
        font_ref.read_bytes()
        return True
    except Exception:
        return False


def _noto_cjk_font_available() -> bool:
    """Checks if Noto Sans CJK Font is available."""
    try:
        from importlib import resources

        font_ref = (
            resources.files("pdftopdfa")
            / "resources"
            / "fonts"
            / "NotoSansCJK-Regular.ttc"
        )
        font_ref.read_bytes()
        return True
    except Exception:
        return False
