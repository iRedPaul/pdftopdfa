# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""CID-to-Unicode mappings for Adobe CID-keyed font collections.

Provides pre-built CID-to-Unicode mappings for bare CFF CID-keyed fonts
that lack a cmap table. Data is derived from Adobe's cmap-resources project
(BSD-3-Clause licensed).
"""

import functools
import gzip
import logging
import struct
from importlib.resources import files

logger = logging.getLogger(__name__)

# Mapping from CIDSystemInfo Ordering to compressed binary resource filename
_ORDERING_TO_RESOURCE: dict[str, str] = {
    "Japan1": "adobe_japan1_utf16.bin.gz",
    "GB1": "adobe_gb1_utf16.bin.gz",
    "CNS1": "adobe_cns1_utf16.bin.gz",
    "Korea1": "adobe_korea1_utf16.bin.gz",
}


@functools.cache
def get_cid_to_unicode(ordering: str) -> dict[int, int] | None:
    """Get CID-to-Unicode mapping for a CID collection ordering.

    Args:
        ordering: CIDSystemInfo Ordering value (e.g. "Japan1", "GB1").

    Returns:
        Dict mapping CID (int) to Unicode codepoint (int), or None if
        the ordering is not recognized.
    """
    if ordering not in _ORDERING_TO_RESOURCE:
        return None

    return _load_mapping(ordering)


def _load_mapping(ordering: str) -> dict[int, int]:
    """Load and decompress a CID-to-Unicode binary mapping file.

    Binary format: sequence of (uint16_be CID, uint16_be Unicode) pairs.

    Args:
        ordering: CIDSystemInfo Ordering value.

    Returns:
        Dict mapping CID to Unicode codepoint.
    """
    resource_name = _ORDERING_TO_RESOURCE[ordering]
    resource_dir = files("pdftopdfa") / "resources" / "cid_unicode"
    resource_path = resource_dir.joinpath(resource_name)

    data = gzip.decompress(resource_path.read_bytes())

    mapping: dict[int, int] = {}
    for i in range(0, len(data), 4):
        cid, unicode_val = struct.unpack(">HH", data[i : i + 4])
        if unicode_val not in (0x0000, 0xFEFF, 0xFFFE, 0xFFFF):
            mapping[cid] = unicode_val

    logger.debug("Loaded %d CID->Unicode entries for %s", len(mapping), ordering)
    return mapping
