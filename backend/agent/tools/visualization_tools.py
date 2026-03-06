"""
Visualization execution tool.
Runs LLM-generated Python code (matplotlib) in a sandboxed exec()
and returns the resulting chart as a base64-encoded PNG.
"""
import io
import re
import json
import base64
import logging
from langchain_core.tools import tool
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_log = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    if not text or not text.strip():
        return ""
    s = text.strip()
    m = re.search(r"```(?:\w*)\s*([\s\S]*?)```", s)
    if m:
        s = m.group(1).strip()
    return s


@tool
def visualization_execution(code: str) -> str:
    """
    Execute Python visualization code and return the chart as a base64 PNG image.

    The code should use matplotlib.pyplot (available as ``plt``).
    pandas (``pd``) and numpy (``np``) are also available.
    Do NOT call ``plt.show()``; the current figure is captured automatically.

    Args:
        code: Python source code that produces a matplotlib figure.

    Returns:
        JSON string with ``description`` (short summary for LLM context)
        and ``image_markdown`` (markdown image with embedded base64 data-URI).
    """
    code = _strip_code_fences(code)
    if not code:
        return json.dumps({"description": "No code provided", "image_markdown": ""})

    exec_globals: dict = {"__builtins__": __builtins__}
    exec_globals["plt"] = plt
    exec_globals["matplotlib"] = matplotlib
    if pd is not None:
        exec_globals["pd"] = pd
    if np is not None:
        exec_globals["np"] = np
    exec_globals["io"] = io

    plt.close("all")

    try:
        exec(code, exec_globals)

        fig = plt.gcf()
        if not fig.get_axes():
            return json.dumps({
                "description": "Code ran but produced no figure axes",
                "image_markdown": "",
            })

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close("all")

        title = fig._suptitle.get_text() if fig._suptitle else ""
        if not title:
            axes = fig.get_axes()
            title = axes[0].get_title() if axes else ""
        desc = f"Chart generated: {title}" if title else "Chart generated successfully"

        return json.dumps({
            "description": desc,
            "image_markdown": f"![chart](data:image/png;base64,{img_b64})",
        })
    except Exception as e:
        plt.close("all")
        _log.warning("Visualization execution error: %s", e)
        return json.dumps({
            "description": f"Error generating chart: {e}",
            "image_markdown": "",
        })
