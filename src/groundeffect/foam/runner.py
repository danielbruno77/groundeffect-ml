"""Run OpenFOAM either natively or inside the official OpenCFD container.

The same pipeline runs on a laptop (Docker, arm64 or amd64), in GitHub Actions
and on an HPC node with a native install: only the launcher changes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_IMAGE = os.environ.get("GROUNDEFFECT_FOAM_IMAGE", "opencfd/openfoam-default:2512")


@dataclass(frozen=True)
class FoamRunner:
    mode: str = "auto"  # "auto" | "native" | "docker"
    image: str = DEFAULT_IMAGE

    def resolved_mode(self) -> str:
        if self.mode != "auto":
            return self.mode
        return "native" if shutil.which("simpleFoam") else "docker"

    def run(self, case: Path, script: str = "./Allrun", timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        case = case.resolve()
        if self.resolved_mode() == "native":
            cmd = ["bash", "-c", script]
            return subprocess.run(cmd, cwd=case, capture_output=True, text=True, timeout=timeout)
        uid_gid = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else None
        cmd = ["docker", "run", "--rm", "-v", f"{case}:/case", "-w", "/case"]
        if uid_gid:
            cmd += ["--user", uid_gid, "-e", "HOME=/tmp"]
        # The image's entrypoint sources the OpenFOAM environment for login shells.
        cmd += ["--entrypoint", "bash", self.image, "-lc", f"source /usr/lib/openfoam/openfoam*/etc/bashrc; {script}"]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
