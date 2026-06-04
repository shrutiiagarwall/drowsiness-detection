"""Root-level app.py — HuggingFace Spaces entry point."""
from app.app import demo  # noqa: F401

if __name__ == "__main__":
    demo.launch()
