"""Compatibility import for historical experiment scripts.

The persistent serving implementation is step_7_i_migraphx_server.py.
"""
from __future__ import annotations

import sys
import step_7_i_migraphx_server as _server

if __name__ == "__main__":
    _server.main()
else:
    sys.modules[__name__] = _server
