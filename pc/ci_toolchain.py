"""Fetch the tools build_chat.py needs, into toolchain/ (or $GARLIC_TOOLCHAIN), on Linux or macOS.

build.py expects:

  toolchain/OpenOrbis/PS4Toolchain/   bin/linux/{create-fself,create-gp4,PkgTool.Core}, lib/, include/, link.x,
                                      samples/hello_world/{sce_sys/about/right.sprx, sce_module/*.prx}
  toolchain/zig/zig                   the Zig binary, with its lib/ next to it: its clang and lld do the compiling
                                      and the linking (the app is built for x86_64-freebsd-none, the PS4's OS)

The OpenOrbis release is a tarball or a zip whose top folder has changed between versions, so this script looks for
the tool rather than assuming a layout, and says what it found. Run it again and it does nothing.

  python pc/ci_toolchain.py            install (or repair) the tools in toolchain/
  python pc/ci_toolchain.py --probe    just say what is there and what the release contains
"""

import argparse
import ctypes
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLCHAIN = Path(os.environ.get("GARLIC_TOOLCHAIN") or ROOT / "toolchain")
OO = TOOLCHAIN / "OpenOrbis" / "PS4Toolchain"
ZIG_DIR = TOOLCHAIN / "zig"
SSL11 = TOOLCHAIN / "libssl1.1"          # for PkgTool.Core, see libssl11()
SSL11_POOLS = ["https://old-releases.ubuntu.com/ubuntu/pool/main/o/openssl/",
               "https://archive.ubuntu.com/ubuntu/pool/main/o/openssl/",
               "https://security.ubuntu.com/ubuntu/pool/main/o/openssl/",
               "https://mirrors.kernel.org/ubuntu/pool/main/o/openssl/",
               "https://deb.debian.org/debian/pool/main/o/openssl/",
               "http://old-releases.ubuntu.com/ubuntu/pool/main/o/openssl/",
               "http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/",
               "http://security.ubuntu.com/ubuntu/pool/main/o/openssl/"]

OO_RELEASE = "https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain/releases/download/v0.5.4/toolchain-llvm-18.tar.gz"
OO_SAMPLES = "https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain/releases/download/v0.5.3/samples-pkg-llvm-18.zip"
ZIGS = ["https://ziglang.org/download/0.14.1/zig-linux-x86_64-0.14.1.tar.xz",
        "https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz"]
WANTED_SAMPLES = ["right.sprx"]


def log(*a):
    print(*a, flush=True)


def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size:
        log(f"  cached {dest.name} ({dest.stat().st_size / 2**20:.0f} MiB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  get {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    tmp.rename(dest)
    log(f"  -> {dest} ({dest.stat().st_size / 2**20:.0f} MiB)")
    return dest


def extract(archive, into):
    archive, into = Path(archive), Path(into)
    into.mkdir(parents=True, exist_ok=True)
    log(f"  extract {archive.name} -> {into}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            t.extractall(into)


def find(root, name):
    for p in Path(root).rglob(name):
        if p.is_file():
            return p
    return None


def probe():
    log(f"toolchain dir: {TOOLCHAIN}")
    for label, p in [("OpenOrbis", OO), ("zig", ZIG_DIR), ("PS4Toolchain/bin/linux", OO / "bin" / "linux")]:
        log(f"  {label:24s} {'yes' if p.exists() else 'no '} {p}")
    for name in ("create-fself", "create-gp4", "PkgTool.Core", "PkgTool.Core.dll"):
        p = OO / "bin" / "linux" / name
        log(f"    bin/linux/{name:16s} {'yes' if p.exists() else 'no'}")
    for name in ("link.x", "lib", "include", "samples/hello_world"):
        p = OO / name
        log(f"    {name:28s} {'yes' if p.exists() else 'no'}")
    if (OO / "samples" / "hello_world").exists():
        for pat in ("sce_sys/about/*.sprx", "sce_module/*.prx"):
            log(f"      {pat:26s} {[p.name for p in (OO / 'samples' / 'hello_world').glob(pat)]}")
    z = ZIG_DIR / "zig"
    log(f"    zig binary {z} {'yes' if z.exists() else 'no'}")
    lib = SSL11 / "usr" / "lib" / "x86_64-linux-gnu"
    log(f"    libssl.so.1.1 {'yes' if (lib / 'libssl.so.1.1').exists() else 'no '} {lib}")
    if (lib / "libssl.so.1.1").exists():
        log(f"      {[(p.name, p.stat().st_size) for p in lib.iterdir()]}")
    log(f"    the system's own libssl 1.1 {'yes' if system_ssl() else 'no'}")


def libssl11():
    """PkgTool.Core is a self-contained .NET Core 3.0 binary, and its crypto layer dlopens libssl.so.1.1. Every
    distribution newer than 22.04 ships OpenSSL 3 instead, and then `PkgTool.Core pkg_build` dies with SIGABRT and
    no message at all (its other commands, which need no crypto, still work). Take the library out of a package
    and let build.py put it on the loader's path."""
    lib = SSL11 / "usr" / "lib" / "x86_64-linux-gnu"
    if os.name == "nt":
        return
    if (lib / "libssl.so.1.1").exists():
        expose_ssl11(lib)
        return
    log("libssl 1.1 (PkgTool.Core's crypto library)")
    for pool in SSL11_POOLS:
        try:
            html = urllib.request.urlopen(pool, timeout=60).read().decode("utf-8", "replace")
        except Exception as ex:  # noqa: BLE001
            log(f"  {pool}: {ex}")
            continue
        names = sorted(set(re.findall(r"libssl1\.1_[^\"'<>]*_amd64\.deb", html)))
        if not names:
            log(f"  {pool}: no libssl1.1 package")
            continue
        name = names[-1]                                   # the listings are in version order
        try:
            deb = download(urllib.parse.urljoin(pool, name), TOOLCHAIN / "downloads" / name)
            lib.mkdir(parents=True, exist_ok=True)
            subprocess.run(["dpkg-deb", "-x", str(deb), str(SSL11)], check=True)
            if (lib / "libssl.so.1.1").exists():
                log(f"  {name} -> {lib}")
                expose_ssl11(lib)
                return
            log(f"  {name}: extracted, but not into {lib}")
        except Exception as ex:  # noqa: BLE001
            log(f"  {name}: {ex}")
    if system_ssl():
        log("  no copy of our own, but the system has libssl.so.1.1: PkgTool.Core can use that one")
        return
    raise SystemExit("no libssl 1.1 anywhere, and PkgTool.Core cannot build a package without it:\n"
                     "  it dlopens libssl.so.1.1 (Ubuntu 20.04 and older ship it; 22.04 and newer do not).\n"
                     "  The pools tried are SSL11_POOLS in pc/ci_toolchain.py.")


def expose_ssl11(lib):
    """PkgTool.Core is a self-extracting .NET host, and the process it finally runs in does not see the
    LD_LIBRARY_PATH that build.py sets (the host drops it: the loader then searches only the directories on the
    distribution's own path). So put the two libraries where the loader looks by default. That needs root; CI has
    passwordless sudo, and without it the LD_LIBRARY_PATH route is still worth a try."""
    dest = Path("/usr/lib/x86_64-linux-gnu")
    names = ("libssl.so.1.1", "libcrypto.so.1.1")
    if all((dest / n).exists() for n in names):
        return
    for cmd in (["cp"], ["sudo", "-n", "cp"]):
        try:
            subprocess.run([*cmd, *[str(lib / n) for n in names], str(dest) + "/"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log(f"  copied {', '.join(names)} into {dest} (PkgTool.Core's loader searches there)")
            return
        except Exception:  # noqa: BLE001
            continue
    log(f"  could not copy into {dest}: PkgTool.Core may not find libssl 1.1 "
        f"(cp {lib}/libssl*.so.1.1 {dest}/ as root)")


def system_ssl():
    """Is there a libssl.so.1.1 on the system's own loader path? Then a copy of ours is not needed."""
    try:
        ctypes.CDLL("libssl.so.1.1")
        return True
    except OSError:
        return False


def install():
    log("OpenOrbis PS4 toolchain")
    archive = download(OO_RELEASE, TOOLCHAIN / "downloads" / Path(OO_RELEASE).name)
    raw = TOOLCHAIN / "raw_openorbis"
    if not raw.exists():
        extract(archive, raw)
    # The tool is somewhere inside: find it, then make toolchain/OpenOrbis/PS4Toolchain point at its folder.
    fself = find(raw, "create-fself")
    if not fself:
        raise SystemExit(f"create-fself is not in {archive}: {sorted(p.name for p in raw.iterdir())}")
    bin_linux = fself.parent
    if bin_linux.name not in ("linux", "windows"):
        cand = fself.parent / "linux" / "create-fself"
        if cand.exists():
            bin_linux = cand.parent
    ps4tc = bin_linux.parent.parent
    log(f"  toolchain root: {ps4tc}")
    if OO.exists() or OO.is_symlink():
        if OO.resolve() != ps4tc.resolve():
            shutil.rmtree(OO, ignore_errors=True) if not OO.is_symlink() else OO.unlink()
    if not OO.exists():
        OO.parent.mkdir(parents=True, exist_ok=True)
        try:
            OO.symlink_to(ps4tc.resolve(), target_is_directory=True)
            log(f"  linked {OO} -> {ps4tc}")
        except OSError:
            shutil.copytree(ps4tc, OO)
    for tool in ("create-fself", "create-gp4", "PkgTool.Core"):
        p = OO / "bin" / "linux" / tool
        if p.exists():
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # The package needs two files that live in the toolchain's sample packages, not in every release.
    have = OO / "samples" / "hello_world"
    if not (have / "sce_sys" / "about" / "right.sprx").exists() or not list((have / "sce_module").glob("*.prx")):
        log("samples/hello_world: getting the sample package")
        sarch = download(OO_SAMPLES, TOOLCHAIN / "downloads" / Path(OO_SAMPLES).name)
        sraw = TOOLCHAIN / "raw_samples"
        if not sraw.exists():
            extract(sarch, sraw)
        sprx = find(sraw, "right.sprx")
        prx = list(Path(sraw).rglob("sce_module/*.prx"))
        if not sprx or not prx:
            raise SystemExit(f"the sample package has no right.sprx / sce_module prx: {sprx}, {prx}")
        (have / "sce_sys" / "about").mkdir(parents=True, exist_ok=True)
        shutil.copy(sprx, have / "sce_sys" / "about" / "right.sprx")
        (have / "sce_module").mkdir(parents=True, exist_ok=True)
        for p in prx:
            shutil.copy(p, have / "sce_module" / p.name)
        log(f"  copied {[p.name for p in prx]} and right.sprx")
    log("Zig")
    if not (ZIG_DIR / "zig").exists():
        err = None
        for url in ZIGS:
            try:
                archive = download(url, TOOLCHAIN / "downloads" / Path(url).name)
                tmp = TOOLCHAIN / "raw_zig"
                extract(archive, tmp)
                inner = next(p for p in tmp.iterdir() if p.is_dir() and (p / "zig").exists())
                ZIG_DIR.mkdir(parents=True, exist_ok=True)
                for p in inner.iterdir():
                    dest = ZIG_DIR / p.name
                    if dest.exists():
                        continue
                    shutil.move(str(p), str(dest))
                err = None
                break
            except Exception as ex:  # noqa: BLE001
                err = ex
                log(f"  {url}: {ex}")
        if err:
            raise SystemExit(f"no Zig: {err}")
    (ZIG_DIR / "zig").chmod((ZIG_DIR / "zig").stat().st_mode | stat.S_IXUSR)
    libssl11()
    probe()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", help="say what is there, change nothing")
    args = ap.parse_args()
    if args.probe:
        return probe()
    return install()


if __name__ == "__main__":
    sys.exit(main() or 0)
