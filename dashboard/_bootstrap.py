"""Make the repo-root packages importable from the dashboard.

Streamlit Cloud installs only `requirements.txt` (no editable `pip install -e .`)
and puts only the entry-script directory (`dashboard/`) on `sys.path`. The
dashboard imports `src` and, transitively, `basel_common`, both of which live at
the repo root. Importing this module first inserts the repo root on `sys.path`
so those imports resolve. No-op locally where the packages are already installed.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
