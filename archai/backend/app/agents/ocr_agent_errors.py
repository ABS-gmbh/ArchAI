"""The OCR agent's error, in its own module so the tile code it uses can raise it.
"""

from __future__ import annotations



class OCRAgentError(RuntimeError):
    """Raised when OCR extraction fails."""
