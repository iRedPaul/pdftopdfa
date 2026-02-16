# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for fonts/cid_unicode.py — CID-to-Unicode mapping loading."""


class TestCidUnicode:
    """Tests for cid_unicode module (CID-to-Unicode mapping loading)."""

    def test_load_japan1(self):
        """Japan1 ordering loads a non-empty mapping."""
        from pdftopdfa.fonts.cid_unicode import get_cid_to_unicode

        mapping = get_cid_to_unicode("Japan1")
        assert mapping is not None
        assert len(mapping) > 1000

    def test_load_gb1(self):
        """GB1 ordering loads a non-empty mapping."""
        from pdftopdfa.fonts.cid_unicode import get_cid_to_unicode

        mapping = get_cid_to_unicode("GB1")
        assert mapping is not None
        assert len(mapping) > 1000

    def test_load_cns1(self):
        """CNS1 ordering loads a non-empty mapping."""
        from pdftopdfa.fonts.cid_unicode import get_cid_to_unicode

        mapping = get_cid_to_unicode("CNS1")
        assert mapping is not None
        assert len(mapping) > 1000

    def test_load_korea1(self):
        """Korea1 ordering loads a non-empty mapping."""
        from pdftopdfa.fonts.cid_unicode import get_cid_to_unicode

        mapping = get_cid_to_unicode("Korea1")
        assert mapping is not None
        assert len(mapping) > 1000

    def test_unknown_ordering_returns_none(self):
        """Unknown ordering returns None."""
        from pdftopdfa.fonts.cid_unicode import get_cid_to_unicode

        assert get_cid_to_unicode("UnknownOrdering") is None

    def test_identity_ordering_returns_none(self):
        """Identity ordering is not a real CID collection."""
        from pdftopdfa.fonts.cid_unicode import get_cid_to_unicode

        assert get_cid_to_unicode("Identity") is None

    def test_caching(self):
        """Second call returns same object (cached)."""
        from pdftopdfa.fonts.cid_unicode import get_cid_to_unicode

        first = get_cid_to_unicode("Japan1")
        second = get_cid_to_unicode("Japan1")
        assert first is second

    def test_mapping_values_are_valid_unicode(self):
        """All values should be valid BMP Unicode codepoints."""
        from pdftopdfa.fonts.cid_unicode import get_cid_to_unicode

        mapping = get_cid_to_unicode("Japan1")
        assert mapping is not None
        for cid, unicode_val in mapping.items():
            assert 0 <= cid <= 0xFFFF
            assert 0 < unicode_val <= 0xFFFF
