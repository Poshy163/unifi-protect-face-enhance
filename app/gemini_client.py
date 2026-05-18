"""Optional Gemini integration for face identification.

If a GEMINI_API_KEY env var is set, the webapp exposes /api/ai/suggest which
asks Gemini Flash Lite to match an unnamed face against the known identities.

This is best-effort and the user always confirms with Y/N before any merge.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

# Lazy import — the package is optional. If it's missing, this module still
# imports and `is_enabled()` returns False.
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except Exception:
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    _GENAI_AVAILABLE = False


_DEFAULT_MODEL = "gemini-2.5-flash-lite"


class GeminiClient:
    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL):
        if not _GENAI_AVAILABLE:
            raise RuntimeError("google-genai is not installed")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        # Per-process cache: (group_id, identity-fingerprint) -> result.
        # Identity-fingerprint is a hash of identity ids so the cache is
        # invalidated if you rename/add/remove identities.
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def suggest(self, query_image: bytes, identities: list[dict],
                cache_key: str | None = None) -> dict[str, Any]:
        """Ask Gemini which identity (if any) the query face matches.

        identities: list of {"id", "name", "image"} where image is JPEG bytes.

        Returns: {"identityId": str|None, "name": str|None, "confidence": float, "reason": str}
        """
        if cache_key is not None:
            fp = ",".join(sorted(i["id"] for i in identities))
            with self._lock:
                hit = self._cache.get((cache_key, fp))
                if hit is not None:
                    return hit

        parts: list[Any] = []
        # Query face first.
        parts.append("This is the QUERY face we want to identify:")
        parts.append(genai_types.Part.from_bytes(data=query_image, mime_type="image/jpeg"))
        parts.append(
            "Below are reference photos of known people. For each, the name "
            "comes before the image."
        )
        for i, ident in enumerate(identities, start=1):
            parts.append(f"REFERENCE {i}: {ident['name']}")
            parts.append(genai_types.Part.from_bytes(data=ident["image"], mime_type="image/jpeg"))

        parts.append(
            "Which reference person, if any, is the same person as the query face? "
            "Consider face shape, features, age, gender. Return strict JSON only.\n"
            'Schema: {"name": <reference name or "UNKNOWN">, "confidence": <0.0-1.0>, "reason": <short>}\n'
            "If confidence is below 0.6, set name to UNKNOWN."
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=parts,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                # Lower thinking budget for speed/cost. Flash-lite is cheap.
                temperature=0.1,
            ),
        )

        text = response.text or ""
        try:
            obj = json.loads(text)
        except Exception:
            obj = {"name": "UNKNOWN", "confidence": 0.0, "reason": f"unparseable: {text[:120]}"}

        match_name = (obj.get("name") or "").strip()
        if match_name.upper() == "UNKNOWN" or not match_name:
            result = {"identityId": None, "name": None,
                      "confidence": float(obj.get("confidence", 0.0) or 0.0),
                      "reason": obj.get("reason") or ""}
        else:
            # Map back to identity id (case-insensitive).
            ident = next((i for i in identities if i["name"].lower() == match_name.lower()), None)
            if not ident:
                # Sometimes Gemini answers a slightly different spelling.
                ident = next((i for i in identities
                              if match_name.lower() in i["name"].lower()
                              or i["name"].lower() in match_name.lower()), None)
            result = {
                "identityId": ident["id"] if ident else None,
                "name": ident["name"] if ident else match_name,
                "confidence": float(obj.get("confidence", 0.0) or 0.0),
                "reason": obj.get("reason") or "",
            }

        if cache_key is not None:
            with self._lock:
                self._cache[(cache_key, fp)] = result
        return result


def build_from_env() -> GeminiClient | None:
    """Return a client if GEMINI_API_KEY is set and the SDK is installed."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or not _GENAI_AVAILABLE:
        return None
    model = os.getenv("GEMINI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    return GeminiClient(api_key, model)


def is_available() -> bool:
    return _GENAI_AVAILABLE
