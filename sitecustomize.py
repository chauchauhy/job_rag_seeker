"""Startup workaround for a hanging Windows WMI COM query.

On some Windows machines the WMI service responds very slowly (or not at all),
which makes CPython's ``platform.system()`` / ``platform.win32_ver()`` block
indefinitely inside ``_wmi.exec_query``. Streamlit calls ``platform.system()``
at import time, so ``streamlit run`` freezes before the app script even loads.

Setting the internal ``_wmi`` hook to ``None`` makes ``platform`` use its fast
``sys.getwindowsversion()`` / ``ver`` fallback instead of WMI.

This file is copied into ``.venv/Lib/site-packages/`` by ``run.ps1`` so it is
auto-imported on every interpreter startup in the project virtualenv.
"""

import platform
import sys

if sys.platform == "win32":
    platform._wmi = None
