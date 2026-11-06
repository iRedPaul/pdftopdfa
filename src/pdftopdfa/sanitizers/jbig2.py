# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Validate JBIG2 globals references without changing image compression."""

import warnings

from pikepdf import Array, Dictionary, Name, Pdf, Stream, StreamDecodeLevel

from ..exceptions import ConversionError


def get_jbig2_globals_streams(pdf: Pdf) -> list[Stream]:
    """Collect distinct globals streams referenced by JBIG2 decode parameters.

    A single filter may use a parameter dictionary or a singleton parameter
    array; filter chains require corresponding arrays (ISO 32000-1, Table 5).
    Missing or null parameters are optional.
    A present /JBIG2Globals value must be a stream (ISO 32000-1, Table 12).
    Neither image data nor globals payloads are decoded by this helper.

    Args:
        pdf: Open PDF whose stream references are inspected.

    Returns:
        Referenced globals streams, with shared streams returned once.

    Raises:
        ConversionError: A JBIG2 parameter structure or globals reference is
            malformed.
    """
    streams: list[Stream] = []
    seen: set[tuple[int, int]] = set()
    for image in pdf.objects:
        if not isinstance(image, Stream):
            continue
        filters = image.get("/Filter")
        filter_list = list(filters) if isinstance(filters, Array) else [filters]
        indices = [
            index
            for index, name in enumerate(filter_list)
            if isinstance(name, Name) and name == Name.JBIG2Decode
        ]
        if not indices:
            continue

        parms = image.get("/DecodeParms")
        if parms is None:
            continue
        if isinstance(parms, Array):
            if len(parms) != len(filter_list):
                raise ConversionError(
                    f"Invalid /DecodeParms for JBIG2 stream {image.objgen}: "
                    "expected one parameter entry per filter"
                )
            entries = [parms[index] for index in indices]
        else:
            if len(filter_list) != 1:
                raise ConversionError(
                    f"Invalid /DecodeParms for JBIG2 stream {image.objgen}: "
                    "expected one parameter entry per filter"
                )
            entries = [parms]

        for entry in entries:
            if entry is None:
                continue
            if not isinstance(entry, Dictionary):
                raise ConversionError(
                    f"Invalid /DecodeParms for JBIG2 stream {image.objgen}: "
                    "expected a dictionary or null"
                )
            globals_stream = entry.get("/JBIG2Globals")
            if globals_stream is None:
                continue
            if not isinstance(globals_stream, Stream):
                raise ConversionError(
                    f"Invalid /JBIG2Globals in stream {image.objgen}: "
                    "expected a stream reference"
                )
            if globals_stream.objgen not in seen:
                seen.add(globals_stream.objgen)
                streams.append(globals_stream)
    return streams


def validate_jbig2_globals(pdf: Pdf) -> list[Stream]:
    """Validate globals references and decode only their outer stream filters.

    This checks that the compressed container can be read; it does not parse
    JBIG2 segments, invoke a JBIG2 image decoder, or rewrite either stream.

    Args:
        pdf: Open PDF whose globals references and outer encoding are checked.

    Returns:
        Distinct globals streams whose outer filters were decoded successfully.

    Raises:
        ConversionError: A reference is malformed or a globals stream cannot be
            decoded with the generalized PDF stream filters.
    """
    streams = get_jbig2_globals_streams(pdf)
    for stream in streams:
        try:
            stream.read_bytes(StreamDecodeLevel.generalized)
        except Exception as exc:
            raise ConversionError(
                f"Cannot decode /JBIG2Globals stream {stream.objgen}: {exc}"
            ) from exc
    return streams


def convert_jbig2_external_globals(pdf: Pdf) -> dict[str, int]:
    """Preserve legacy imports while leaving JBIG2 streams unchanged.

    Deprecated: PDF/A permits JBIG2 globals and refinement coding. Global
    segments must remain separate, so this compatibility function performs no
    conversion, validation, or decoding. Use sanitize_for_pdfa for structural
    validation during conversion.

    Args:
        pdf: PDF retained as an argument for compatibility; it is not modified.

    Returns:
        The legacy converted, reencoded, and failed counters, all zero.
    """
    warnings.warn(
        "convert_jbig2_external_globals is deprecated and no longer modifies "
        "JBIG2 streams; use sanitize_for_pdfa for PDF/A sanitization",
        DeprecationWarning,
        stacklevel=2,
    )
    return {"converted": 0, "reencoded": 0, "failed": 0}
