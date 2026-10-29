"""The SAIA OCR agent's error, in its own module so the services it uses can raise it.
"""

from __future__ import annotations



class SaiaOCRAgentError(RuntimeError):
    """Raised when SAIA OCR extraction fails."""
