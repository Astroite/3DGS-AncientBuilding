"""Standard SH3 Gaussian Splatting PLY layout and header reader."""

from __future__ import annotations

import re
from pathlib import Path


FLOAT_PROPERTIES = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *(f"f_rest_{index}" for index in range(45)),
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def read_ply_header(path: Path) -> tuple[bytes, int, list[str]]:
    with path.open("rb") as stream:
        header = bytearray()
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header has no end_header marker")
            header.extend(line)
            if line.strip() == b"end_header":
                break
    text = header.decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError("Only binary_little_endian PLY 1.0 is supported")
    match = re.search(r"^element vertex (\d+)$", text, re.MULTILINE)
    if match is None:
        raise ValueError("PLY header has no vertex count")
    properties = re.findall(r"^property float (\S+)$", text, re.MULTILINE)
    return bytes(header), int(match.group(1)), properties
