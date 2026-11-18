"""Edit text files without disturbing their line endings (much of this repository is CRLF)."""

from pathlib import Path


def edit(path, *pairs, count=1):
    """Replace each old text with its new text; assert each is there the expected number of times."""
    p = Path(path)
    raw = p.read_bytes()
    crlf = b"\r\n" in raw
    s = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in pairs:
        n = s.count(old)
        assert n == count, f"{path}: {n} matches for {old.splitlines()[0][:70]!r}"
        s = s.replace(old, new)
    p.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
    return p
