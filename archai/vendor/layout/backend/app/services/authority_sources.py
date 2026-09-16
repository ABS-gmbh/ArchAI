"""External authority source helpers for VIAF and GeoNames.

These helpers keep direct source lookups separate from the main
authority-linking orchestrator so the resolver can query multiple
authorities in parallel.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote_plus

from app.config import settings

log = logging.getLogger("archai.authority_sources")

_USER_AGENT = "Archai-OCR-Pipeline/1.0 (research; mailto:archai@example.com)"


def _http_json(url: str, params: dict[str, Any], *, timeout: int = 10) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    query = "&".join(
        f"{quote_plus(str(key))}={quote_plus(str(value))}"
        for key, value in params.items()
        if value is not None and str(value) != ""
    )
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        log.debug("Authority source HTTP error for %s: %s", url, exc)
        return {}
    except Exception as exc:  # noqa: BLE001
        log.debug("Authority source request failed for %s: %s", url, exc)
        return {}


def _fetch_viaf_record(viaf_id: str) -> dict[str, Any]:
    viaf_id = str(viaf_id or "").strip()
    if not viaf_id:
        return {}
    return _http_json(f"https://viaf.org/viaf/{viaf_id}/viaf.json", {}, timeout=10)


def search_viaf(query: str, *, k: int = 5, ent_type: str = "") -> list[dict[str, Any]]:
    if ent_type not in {"person", "work"}:
        return []
    payload = _http_json(
        "https://viaf.org/viaf/AutoSuggest",
        {"query": query},
        timeout=10,
    )
    rows = payload.get("result") or []
    if not isinstance(rows, list):
        return []

    out: list[dict[str, Any]] = []
    for row in rows[:k]:
        if not isinstance(row, dict):
            continue
        viaf_id = str(row.get("viafid") or "").strip()
        label = str(row.get("term") or "").strip()
        if not viaf_id or not label:
            continue
        record = _fetch_viaf_record(viaf_id)
        aliases: list[dict[str, str]] = []
        seen: set[str] = set()
        main_headings = record.get("mainHeadings", {}).get("data", [])
        if isinstance(main_headings, dict):
            main_headings = [main_headings]
        for item in main_headings if isinstance(main_headings, list) else []:
            if not isinstance(item, dict):
                continue
            value = str(item.get("text") or "").strip()
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            aliases.append({"lang": "", "value": value})

        name_type = str(record.get("nameType") or row.get("nametype") or "").strip()
        titles = record.get("titles")
        description_parts = [part for part in [name_type, label] if part]
        out.append(
            {
                "source": "viaf",
                "authority_id": viaf_id,
                "qid": "",
                "viaf_id": viaf_id,
                "geonames_id": "",
                "label": label,
                "description": " | ".join(description_parts) or "VIAF authority record",
                "url": f"https://viaf.org/viaf/{viaf_id}",
                "aliases": aliases,
                "instance_of_qids": viaf_type_qids(name_type),
                "canonical_label": label,
                "canonical_description": "VIAF authority record",
                "lat": None,
                "lon": None,
                "country_qids": [],
                "admin_qids": [],
                "country_name": "",
                "admin1_name": "",
                "parent_location": "",
                "titles": titles,
                "name_type": name_type,
                "source_confidence": 0.92,
            }
        )
    return out


# ── Native type assertions from non-Wikidata sources ──────────────────
#
# is_type_compatible is precision-first: with no P31 values it returns False so
# untyped Wikidata items are never auto-linked. VIAF and GeoNames records were
# built with instance_of_qids=[], so EVERY one of them was judged incompatible
# and hard-gated out - the two sources were queried on every person, work and
# place mention and could never contribute a link.
#
# They do assert a type, just not as a P31 QID. Translating their own
# vocabularies restores the signal without weakening the gate for Wikidata.

_GEONAMES_FCODE_QIDS: dict[str, str] = {
    "PPLC": "Q5119",       # capital
    "PPLA": "Q515",        # seat of a first-order admin division
    "PPLA2": "Q515",
    "PPLA3": "Q515",
    "PPLA4": "Q515",
    "PPL": "Q486972",      # populated place
    "PPLL": "Q532",        # village
    "PPLX": "Q486972",
    "ADM1": "Q56061",      # administrative territorial entity
    "ADM2": "Q56061",
    "ADM3": "Q56061",
    "ADM4": "Q56061",
    "ADMD": "Q56061",
    "PCLI": "Q6256",       # independent political entity (country)
    "PCL": "Q6256",
    "RGN": "Q82794",       # geographic region
    "ISL": "Q23442",       # island
    "MT": "Q8502",         # mountain
    "MTS": "Q46831",       # mountain range
    "STM": "Q4022",        # river
}

_GEONAMES_CLASS_QIDS: dict[str, str] = {
    "P": "Q486972",        # city, village
    "A": "Q56061",         # country, state, region
    "H": "Q618123",        # stream, lake
    "T": "Q618123",        # mountain, hill, rock
    "L": "Q82794",         # parks, area
    "S": "Q618123",        # spot, building, farm
    "V": "Q618123",        # forest, heath
    "R": "Q618123",        # road, railroad
    "U": "Q618123",        # undersea
}

_VIAF_NAMETYPE_QIDS: dict[str, str] = {
    "personal": "Q5",          # human
    "corporate": "Q43229",     # organization
    "geographic": "Q486972",   # human settlement
    "uniformtitle": "Q7725634",  # literary work
    "title": "Q7725634",
    "work": "Q7725634",
    "expression": "Q7725634",
}


def geonames_type_qids(feature_code: str, feature_class: str = "") -> list[str]:
    """Translate a GeoNames feature code/class into compatible Wikidata QIDs."""
    code = str(feature_code or "").strip().upper()
    if code in _GEONAMES_FCODE_QIDS:
        return [_GEONAMES_FCODE_QIDS[code]]
    cls = str(feature_class or "").strip().upper()[:1]
    if cls in _GEONAMES_CLASS_QIDS:
        return [_GEONAMES_CLASS_QIDS[cls]]
    # A GeoNames hit is a place by construction, even when the code is unknown.
    return ["Q2221906"]  # geographic location


def viaf_type_qids(name_type: str) -> list[str]:
    """Translate a VIAF nameType into compatible Wikidata QIDs."""
    key = str(name_type or "").strip().lower().replace(" ", "").replace("-", "")
    qid = _VIAF_NAMETYPE_QIDS.get(key)
    return [qid] if qid else []


def search_geonames(query: str, *, k: int = 5, ent_type: str = "") -> list[dict[str, Any]]:
    if ent_type != "place":
        return []
    username = str(settings.geonames_username or "").strip()
    if not username:
        return []
    base_url = str(settings.geonames_base_url or "http://api.geonames.org").rstrip("/")
    payload = _http_json(
        f"{base_url}/searchJSON",
        {
            "q": query,
            "maxRows": max(1, min(int(k), 10)),
            "style": "FULL",
            "username": username,
        },
        timeout=int(settings.geonames_timeout_seconds or 10),
    )
    rows = payload.get("geonames") or []
    if not isinstance(rows, list):
        return []

    out: list[dict[str, Any]] = []
    for row in rows[:k]:
        if not isinstance(row, dict):
            continue
        geoname_id = str(row.get("geonameId") or "").strip()
        label = str(row.get("toponymName") or row.get("name") or "").strip()
        if not geoname_id or not label:
            continue
        country_name = str(row.get("countryName") or "").strip()
        admin1_name = str(row.get("adminName1") or "").strip()
        fcode_name = str(row.get("fcodeName") or row.get("fclName") or "place").strip()
        parent_location = " > ".join(part for part in (admin1_name, country_name) if part)
        out.append(
            {
                "source": "geonames",
                "authority_id": geoname_id,
                "qid": "",
                "viaf_id": "",
                "geonames_id": geoname_id,
                "label": label,
                "description": " | ".join(part for part in (fcode_name, parent_location) if part) or "GeoNames place",
                "url": f"https://www.geonames.org/{geoname_id}",
                "aliases": [{"lang": "", "value": label}],
                "instance_of_qids": geonames_type_qids(
                    str(row.get("fcode") or ""), str(row.get("fcl") or "")
                ),
                "canonical_label": label,
                "canonical_description": fcode_name or "GeoNames place",
                "lat": row.get("lat"),
                "lon": row.get("lng"),
                "country_qids": [],
                "admin_qids": [],
                "country_name": country_name,
                "admin1_name": admin1_name,
                "parent_location": parent_location,
                "feature_code": str(row.get("fcode") or "").strip(),
                "source_confidence": 0.95,
            }
        )
    return out
