"""Shared build settings: where the tools are, the compiler flags and the connection to the console.

The app itself is built by build_chat.py. The tools are expected in toolchain/ (or in the folder named by the
GARLIC_TOOLCHAIN environment variable):
  OpenOrbis/PS4Toolchain   the OpenOrbis PS4 Toolchain (headers, libraries, link script, package tools)
  zig/zig.exe              Zig: its clang and lld compile the app and the GPU kernels
  dotnet/dotnet.exe        the .NET runtime that OpenOrbis' PkgTool needs
"""

import ftplib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLCHAIN = Path(os.environ.get("GARLIC_TOOLCHAIN") or ROOT / "toolchain")
OO = TOOLCHAIN / "OpenOrbis" / "PS4Toolchain"
WIN = os.name == "nt"
EXE = ".exe" if WIN else ""                       # the tools' names on Windows and on Linux
ZIG = TOOLCHAIN / "zig" / f"zig{EXE}"
DOTNET = TOOLCHAIN / "dotnet" / "dotnet.exe"      # Windows only: PkgTool.Core.dll runs on it there
# PkgTool.Core is .NET Core 3.0: its crypto needs libssl.so.1.1, which pc/ci_toolchain.py fetches (see there)
SSL11 = TOOLCHAIN / "libssl1.1" / "usr" / "lib" / "x86_64-linux-gnu"
BIN = OO / ("bin/windows" if WIN else "bin/linux")
BUILD = ROOT / "build"
PKGDIR = BUILD / "pkg"

# The console's address for the uploads over FTP (GoldHEN's FTP server): set PS4_IP, and PS4_FTP_PORT if it is not 2121.
PS4_IP = os.environ.get("PS4_IP", "")
FTP_PORT = int(os.environ.get("PS4_FTP_PORT") or 2121)

CFLAGS = ["-target", "x86_64-freebsd-none", "-march=btver2", "-O2", "-fPIC", "-funwind-tables",
          "-fno-stack-protector", "-fshort-wchar", "-nostdinc", "-isystem", str(OO / "include"), "-Wall", "-c"]


ENV = dict(os.environ, DOTNET_ROLL_FORWARD="Major", OO_PS4_TOOLCHAIN=str(OO),
           DOTNET_SYSTEM_GLOBALIZATION_INVARIANT="1")   # PkgTool.Core needs no ICU library (Linux)
if not WIN and (SSL11 / "libssl.so.1.1").exists():
    ENV["LD_LIBRARY_PATH"] = f"{SSL11}:{ENV.get('LD_LIBRARY_PATH', '')}".rstrip(":")


def run(cmd, env=None, **kw):
    print(">", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, env=env or ENV, **kw)


def pkgtool(*args):
    """Run PkgTool.Core. Its crypto layer is a .NET shim that dlopens `libssl.so.1.1`, which no distribution newer
    than Ubuntu 22.04 ships; pc/ci_toolchain.py fetches the library, and here it is handed over at the moment the
    tool runs, not at import (the toolchain may be fetched after this module is read). Two ways, because the .NET
    host does not always pass LD_LIBRARY_PATH on to the process that does the work: the variable first, then the
    libraries preloaded, so they are part of the process before anything else looks for them."""
    if WIN:
        return run([DOTNET, BIN / "PkgTool.Core.dll", *args], cwd=PKGDIR)
    libssl, libcrypto = SSL11 / "libssl.so.1.1", SSL11 / "libcrypto.so.1.1"
    env = ENV
    tries = [env]
    if libssl.exists():
        env = dict(ENV, LD_LIBRARY_PATH=f"{SSL11}:{ENV.get('LD_LIBRARY_PATH', '')}".rstrip(":"))
        tries = [env, dict(env, LD_PRELOAD=f"{libssl}:{libcrypto}")]
    print("libssl.so.1.1:", "yes" if libssl.exists() else "no ", SSL11, flush=True)
    for i, e in enumerate(tries):
        try:
            if i:
                print("> the same command, with libssl 1.1 preloaded", flush=True)
            return run([BIN / "PkgTool.Core", *args], env=e, cwd=PKGDIR)
        except subprocess.CalledProcessError:
            if i == len(tries) - 1:
                raise


def pkg_build(outdir):
    """`pkg_build` is the one PkgTool.Core command that needs libssl 1.1 (see pc/ci_toolchain.py)."""
    return pkgtool("pkg_build", "pkg.gp4", str(outdir))


def ps4_ftp(timeout=30):
    """A logged-in FTP connection to the console."""
    if not PS4_IP:
        sys.exit("Set the PS4_IP environment variable to your console's address (and start GoldHEN's FTP server).")
    ftp = ftplib.FTP()
    ftp.connect(PS4_IP, FTP_PORT, timeout=timeout)
    ftp.login()
    return ftp
