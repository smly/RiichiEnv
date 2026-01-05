import json
import os
import uuid
from typing import Any, List

from IPython.display import HTML

# Load the compiled JS once (or lazy load)
_VIEWER_JS = None

def _get_viewer_js() -> str:
    global _VIEWER_JS
    if _VIEWER_JS is None:
        p = os.path.join(os.path.dirname(__file__), "assets", "viewer.js")
        if not os.path.exists(p):
            return "console.error('RiichiEnv Viewer JS not found. Please build tools/replay-visualizer first.');"
        with open(p, "r", encoding="utf-8") as f:
            _VIEWER_JS = f.read()
    return _VIEWER_JS

def show_replay(log: List[dict[str, Any]]) -> HTML:
    """
    Displays a replay viewer for the given MJAI log in a Jupyter Notebook.
    The viewer is self-contained (JS embedded) and works without a kernel connection after rendering.
    """
    
    unique_id = f"riichienv-viewer-{uuid.uuid4()}"
    log_json = json.dumps(log)
    viewer_js = _get_viewer_js()
    
    html_content = f"""
    <div id="{unique_id}" style="width: 100%; min-height: 600px; border: 1px solid #ddd;">
        Loading RiichiEnv Replay...
    </div>
    <script>
    (function() {{
        {viewer_js}
        
        const logData = {log_json};
        
        // Wait for DOM or just run if deferred
        if (window.RiichiEnvViewer) {{
            new window.RiichiEnvViewer("{unique_id}", logData);
        }} else {{
            console.error("RiichiEnvViewer global not found");
        }}
    }})();
    </script>
    """
    
    return HTML(html_content)
