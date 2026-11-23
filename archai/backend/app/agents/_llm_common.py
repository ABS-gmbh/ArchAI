"""Helpers the OCR agents shared as verbatim copies (audit AG-17).

Only code that was identical in behaviour lives here. The agents' JSON
extractors, image downscalers and repair loops still differ from one another
(see AG-17), and merging them would change what each agent accepts.
"""

from __future__ import annotations

from typing import Any


def is_image_too_large_error(exc: Exception) -> bool:
    """True when *exc* is the provider or PIL refusing an image for its size."""
    text = str(exc).lower()
    return (
        ("image size" in text and "exceeds limit" in text)
        or "decompression bomb" in text
        or ("too large" in text and "pixels" in text)
    )


def chat_completion_preferring_json_object(client: Any, *, messages: list[Any], **params: Any) -> dict[str, Any]:
    """``client.chat_completion`` asking for a JSON object, retried without when the model rejects the format.

    Only an error naming ``response_format`` or ``json_object`` triggers the
    retry; any other error is raised.
    """
    try:
        return client.chat_completion(response_format={"type": "json_object"}, messages=messages, **params)
    except Exception as exc:
        text = str(exc).lower()
        if "response_format" not in text and "json_object" not in text:
            raise
        return client.chat_completion(messages=messages, **params)
