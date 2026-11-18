"""Build PkgTool from LibOrbisPkg's source, with its PlayGo estimate fixed.

Two things are wrong with the PkgTool.Core that OpenOrbis ships, and both are properties of the tool, not of this
repository:

* It is a self-contained **.NET Core 3.0** binary, so its crypto layer dlopens `libssl.so.1.1`, which no
  distribution newer than Ubuntu 22.04 has (see pc/ci_toolchain.py, which fetches it and puts it where the loader
  looks).
* It sizes the **PlayGo chunk-hash table** from an estimate aligned to 0x80000, and on a package of several GiB
  that estimate comes out a chunk or two short: `pkg_build` stops with *"Playgo Chunk hash file was not allocated
  enough space. Report this as a bug"* (LibOrbisPkg#64). It is the difference between the 41 MB package, which
  builds, and the 4.9 GB one, which does not.

The same source builds against the .NET that the runner already has (8), which fixes the first, and the allocation
is over-sized by a line of the patch below, which fixes the second (the tool shrinks the table back to its real
length as soon as the sizes are known, so the extra room is only ever reserved, never written).

The result is `toolchain/pkgtool/PkgTool.Core`, used by build.py for `pkg_build` when it is there. Without network
or without the SDK this does nothing and the bundled tool is used as before -- and then a full-size package fails
as it did.
"""

import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLCHAIN = Path(os.environ.get("GARLIC_TOOLCHAIN") or ROOT / "toolchain")
SRC = TOOLCHAIN / "liborbispkg"
OUT = TOOLCHAIN / "pkgtool"
TARBALL = "https://codeload.github.com/maxton/LibOrbisPkg/tar.gz/refs/heads/master"

# The one line that matters. `e.DataSize` is the room reserved inside the PKG for the chunk hashes; the code above
# works it out from an estimate and then, a few lines later, checks it against the size the package really has.
PATCH_FROM = "          e.DataSize = (uint)(pkgSize / 0x10000L) * 4;"
PATCH_TO = ("          // station-ai: the estimate above lands a chunk or two short on a package of several GiB,\n"
            "          // and pkg_build then dies with \"Playgo Chunk hash file was not allocated enough space\"\n"
            "          // (LibOrbisPkg#64). Reserve 1 MiB more; the table is shrunk to its real length below.\n"
            "          e.DataSize = (uint)(pkgSize / 0x10000L) * 4 + 0x100000;")


def log(msg):
    print(msg, flush=True)


def dotnet():
    for name in ("dotnet", "dotnet.exe"):
        p = shutil.which(name)
        if p:
            return p
    return None


def fetch():
    if (SRC / "PkgTool" / "Program.cs").exists():
        return
    log(f"  get {TARBALL}")
    tmp = TOOLCHAIN / "downloads" / "liborbispkg.tar.gz"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(TARBALL, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    shutil.rmtree(SRC, ignore_errors=True)
    with tarfile.open(tmp) as t:
        t.extractall(SRC)                                    # LibOrbisPkg-master/...
    inner = next(p for p in SRC.iterdir() if p.is_dir())
    for p in inner.iterdir():
        shutil.move(str(p), str(SRC / p.name))
    inner.rmdir()


def patch():
    """Over-size the PlayGo reservation and target the .NET that is installed."""
    f = SRC / "LibOrbisPkg" / "PKG" / "PkgBuilder.cs"
    text = f.read_text(encoding="utf-8-sig")
    if PATCH_TO.splitlines()[-1].strip() not in text:
        assert PATCH_FROM in text, "PkgBuilder.cs: the line to patch is not there"
        text = text.replace(PATCH_FROM, PATCH_TO)
        f.write_text("\ufeff" + text, encoding="utf-8")
    for proj in ("LibOrbisPkg.Core", "PkgTool.Core"):
        p = SRC / proj / f"{proj}.csproj"
        s = p.read_text(encoding="utf-8-sig")
        s = re.sub(r"<TargetFramework>[^<]*</TargetFramework>", "<TargetFramework>net8.0</TargetFramework>", s)
        if proj == "PkgTool.Core" and "<AssemblyName>" not in s:
            s = s.replace("<OutputType>Exe</OutputType>", "<OutputType>Exe</OutputType>\n    <AssemblyName>PkgTool</AssemblyName>")
        p.write_text("\ufeff" + s, encoding="utf-8")


def main():
    exe = dotnet()
    log("PkgTool from LibOrbisPkg's source" + (f" ({exe})" if exe else ""))
    if not exe:
        log("  no dotnet here: the bundled PkgTool.Core is used (a 5 GB package will not build)")
        return 0
    try:
        fetch()
        patch()
    except Exception as ex:                                   # noqa: BLE001
        log(f"  {ex}")
        return 0
    r = subprocess.run([exe, "publish", str(SRC / "PkgTool.Core" / "PkgTool.Core.csproj"), "-c", "Release",
                        "-o", str(OUT), "--nologo"],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr, flush=True)
    tool = OUT / ("PkgTool.exe" if os.name == "nt" else "PkgTool")
    if r.returncode or not tool.exists():
        log(f"  the build did not produce {tool}: the bundled PkgTool.Core is used")
        return 0
    log(f"  built {tool} ({tool.stat().st_size} bytes)")


if __name__ == "__main__":
    raise SystemExit(main())
