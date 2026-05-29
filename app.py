"""
app.py (root-level)
====================
HuggingFace Spaces entry-point alias.
Simply re-exports the Gradio `demo` object from app/app.py.
"""

from app.app import demo  # noqa: F401

if __name__ == "__main__":
    demo.launch()
