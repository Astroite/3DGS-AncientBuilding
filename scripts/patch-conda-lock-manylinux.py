from __future__ import annotations

from pathlib import Path

import conda_lock.pypi_solver


path = Path(conda_lock.pypi_solver.__file__).resolve()
old = 'MANYLINUX_TAGS = ["1", "2010", "2014", "_2_17", "_2_18", "_2_24", "_2_28"]'
new = (
    'MANYLINUX_TAGS = ["1", "2010", "2014", "_2_17", "_2_18", "_2_24", '
    '"_2_28", "_2_29", "_2_30", "_2_31", "_2_32", "_2_33", "_2_34", "_2_35"]'
)
content = path.read_text(encoding="utf-8")
if new in content:
    print(f"conda-lock manylinux compatibility patch already applied: {path}")
elif old in content:
    path.write_text(content.replace(old, new, 1), encoding="utf-8")
    print(f"Applied conda-lock manylinux compatibility patch: {path}")
else:
    raise SystemExit(
        f"Refusing to patch unexpected conda-lock source: {path}. "
        "The installed version may already contain a different upstream fix."
    )
