#!/usr/bin/env python3
"""
Build the final DMG with both the .app and the ReadMe.txt visible.
Run after `briefcase build macOS --adhoc-sign`.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "build/macapp/macos/app/Doc Cleaner.app"
README = ROOT / "resources/ReadMe.txt"
OUT = ROOT / "dist/Doc Cleaner-1.3.0.dmg"

if not APP.exists():
    sys.exit(f"ERROR: .app not found at {APP}\nRun: briefcase build macOS --adhoc-sign")

if not README.exists():
    sys.exit(f"ERROR: ReadMe.txt not found at {README}")

OUT.parent.mkdir(exist_ok=True)
if OUT.exists():
    OUT.unlink()

import dmgbuild

dmgbuild.build_dmg(
    str(OUT),
    "Doc Cleaner",
    settings={
        "files": [str(APP), str(README)],
        "symlinks": {"Applications": "/Applications"},
        "icon_locations": {
            "Doc Cleaner.app": (150, 185),
            "Applications":    (430, 185),
            "ReadMe.txt":      (290, 370),
        },
        "background": "builtin-arrow",
        "window_rect":  ((200, 120), (600, 500)),
        "icon_size":    100,
        "text_size":    13,
        "format":       "UDZO",
    }
)

size = OUT.stat().st_size // (1024 * 1024)
print(f"Done: {OUT.name} ({size} MB)")
print("Contents: Doc Cleaner.app + ReadMe.txt + Applications symlink")
