"""Bake Nerfstudio's Z-up Gaussian PLY into a portable Y-up PLY."""

from __future__ import annotations

import argparse
from pathlib import Path

from gsdb.ply import rotate_gaussian_ply_y_up


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = rotate_gaussian_ply_y_up(args.source, args.destination)
    print(f"Wrote {count:,} Gaussians to {args.destination}")


if __name__ == "__main__":
    main()
