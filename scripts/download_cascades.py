"""
scripts/download_cascades.py
==============================
Standalone script to download all required Haar cascade XML files.

Usage:
    python scripts/download_cascades.py

Bug fix note:
    haarcascade_mcs_mouth.xml exists ONLY in the 2.4.13.7 tag,
    not in the master branch. This script uses the correct URLs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.cascade_utils import download_cascades

if __name__ == "__main__":
    results = download_cascades(ROOT / "cascades")
    print(f"\n✅ {len(results)} cascade(s) ready in ./cascades/")
