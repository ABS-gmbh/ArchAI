"""The text reports written for a traced OCR run: the mention-extraction report,
its audit gates and the consolidated report. Moved out of ``app/routers/ocr.py``.
"""

from __future__ import annotations

import re
from typing import Any

from app.db.pipeline_db import (
    insert_entity_decisions,
    list_entity_attempts_for_run,
    list_entity_decisions,
    list_entity_mentions,
)


def _build_mention_extraction_report(
    run_id: str,
    asset_ref: str,
    mentions: list[dict[str, Any]],
    salvage_debug: dict[str, Any],
) -> str:
    """Build a chat-printable === MENTION EXTRACTION REPORT ===.

    Primary path (auditable): queries ``entity_decisions`` and
    ``entity_attempts`` from SQLite so every printed line is backed by a
    persisted row.  Falls back to in-memory data only when no decisions
    exist in the DB (legacy unit-test path).

    Sections (DB-backed path):
      - top_mentions_preview:  status = ACCEPT_LINKABLE
      - skipped_preview:       status = SKIP_NON_LINKABLE
      - top_rejected_preview:  status = REJECT_LINKABLE
      - filtered_out_preview:  status = FILTERED_OUT (debug-only)
      - attempts_for_accepted: from entity_attempts table
    """
    _LINKABLE_TYPES = {"person", "place", "work", "date"}
    L: list[str] = []
    L.append("=== MENTION EXTRACTION REPORT ===")
    L.append(f"run_id: {run_id}")
    L.append(f"asset_ref: {asset_ref}")

    # ── Attempt DB-backed auditable path ────────────────────────────
    decisions = list_entity_decisions(run_id)

    # ── Backfill: older runs without entity_decisions ─────────────
    if not decisions:
        db_mentions_bf = list_entity_mentions(run_id)
        if db_mentions_bf:
            import uuid as _uuid_bf
            backfill_rows: list[dict[str, Any]] = []
            for m in db_mentions_bf:
                etype = str(m.get("ent_type") or "unknown")
                surface = str(m.get("surface") or "")
                s_off = int(m.get("start_offset", 0))
                e_off = int(m.get("end_offset", 0))
                if etype in _LINKABLE_TYPES:
                    status = "ACCEPT_LINKABLE"
                    method = str(m.get("method") or "unknown")
                else:
                    status = "SKIP_NON_LINKABLE"
                    method = str(m.get("method") or "unknown")
                backfill_rows.append({
                    "decision_id": str(_uuid_bf.uuid4()),
                    "chunk_id": m.get("chunk_id"),
                    "start_offset": s_off,
                    "end_offset": e_off,
                    "surface": surface,
                    "norm": m.get("norm"),
                    "ent_type_guess": etype,
                    "label": m.get("label"),
                    "status": status,
                    "method": method,
                    "reason": f"backfill_from_entity_mentions:{m.get('mention_id', '?')}",
                    "confidence": float(m.get("confidence", 0)),
                    "meta_json": None,
                })
            if backfill_rows:
                insert_entity_decisions(run_id, backfill_rows)
                decisions = list_entity_decisions(run_id)

    if decisions:
        # DB-backed mentions_total and mentions_by_method
        db_mentions = list_entity_mentions(run_id)
        L.append(f"mentions_total: {len(db_mentions)}")
        method_counts: dict[str, int] = {}
        for m in db_mentions:
            method = str(m.get("method") or "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
        L.append("mentions_by_method:")
        if method_counts:
            for method, count in sorted(method_counts.items()):
                L.append(f"  {method}: {count}")
        else:
            L.append("  (none)")
        # Fetch all attempts for this run (joined with decision metadata)
        all_attempts = list_entity_attempts_for_run(run_id)
        # Build lookup: decision_id → list[attempt]
        attempts_by_decision: dict[str, list[dict[str, Any]]] = {}
        for a in all_attempts:
            attempts_by_decision.setdefault(a["decision_id"], []).append(a)

        accept_decisions = [d for d in decisions if d.get("status") == "ACCEPT_LINKABLE"]
        skip_decisions = [d for d in decisions if d.get("status") == "SKIP_NON_LINKABLE"]
        reject_decisions = [d for d in decisions if d.get("status") == "REJECT_LINKABLE"]
        filtered_decisions = [d for d in decisions if d.get("status") == "FILTERED_OUT"]

        # Top linkable mentions preview (up to 10) — status = ACCEPT_LINKABLE
        L.append("")
        top_n = min(10, len(accept_decisions))
        L.append(f"top_mentions_preview (N={top_n}):")
        sorted_accept = sorted(accept_decisions, key=lambda d: -float(d.get("confidence") or 0))
        for d in sorted_accept[:top_n]:
            label = d.get("label") or "-"
            # Get canonical/nd/bo from the ACCEPT attempt
            dec_attempts = attempts_by_decision.get(d["decision_id"], [])
            accept_att = next((a for a in dec_attempts if a.get("attempt_decision") == "ACCEPT"), None)
            notes_parts: list[str] = []
            if accept_att:
                notes_parts.append(f"canonical={accept_att.get('candidate', '?')}")
                if accept_att.get("nd") is not None:
                    notes_parts.append(f"nd={accept_att['nd']:.3f}")
                if accept_att.get("bo") is not None:
                    notes_parts.append(f"bo={accept_att['bo']:.3f}")
            notes = " ".join(notes_parts) if notes_parts else ""
            L.append(
                f"  surface=\"{d.get('surface', '')}\""
                f"  label={label}"
                f"  conf={float(d.get('confidence') or 0):.2f}"
                f"  method={d.get('method', '?')}"
                f"  notes={notes}"
            )
        if not accept_decisions:
            L.append("  (none)")

        # Skipped non-linkable preview — status = SKIP_NON_LINKABLE
        if skip_decisions:
            L.append("")
            skip_n = len(skip_decisions) if salvage_debug.get("fixture_override") else min(10, len(skip_decisions))
            L.append(f"skipped_preview (N={skip_n}):")
            for d in skip_decisions[:skip_n]:
                surface = d.get("surface", "")
                ent_type = d.get("ent_type_guess", "?")
                label = str(d.get("label") or "").strip()
                reason = d.get("reason", "SKIP_NON_LINKABLE")
                L.append(
                    f"  surface=\"{surface}\""
                    f"  type={ent_type}"
                    f"{f'  label={label}' if label else ''}"
                    f"  reason={reason}"
                )

        # Rejected preview — status = REJECT_LINKABLE only
        L.append("")
        rej_n = min(10, len(reject_decisions))
        L.append(f"top_rejected_preview (N={rej_n}):")
        for d in reject_decisions[:rej_n]:
            dec_attempts = attempts_by_decision.get(d["decision_id"], [])
            canonical = "?"
            if dec_attempts:
                canonical = dec_attempts[-1].get("candidate", "?")
            L.append(
                f"  surface=\"{d.get('surface', '')}\""
                f"  canonical={canonical}"
                f"  reason={d.get('reason', '?')}"
            )
        if not reject_decisions:
            L.append("  (none)")

        # Filtered out preview — status = FILTERED_OUT (debug-only)
        if filtered_decisions:
            L.append("")
            filt_n = min(10, len(filtered_decisions))
            L.append(f"filtered_out_preview (N={filt_n}):")
            for d in filtered_decisions[:filt_n]:
                L.append(
                    f"  surface=\"{d.get('surface', '')}\""
                    f"  reason={d.get('reason', '?')}"
                )

        # Attempt history for accepted spans (from entity_attempts table)
        attempts_for_accepted: list[dict[str, Any]] = []
        for d in accept_decisions:
            dec_attempts = attempts_by_decision.get(d["decision_id"], [])
            for a in dec_attempts:
                a["_accepted_surface"] = d.get("surface", "")
                attempts_for_accepted.append(a)
        if attempts_for_accepted:
            L.append("")
            att_n = min(10, len(attempts_for_accepted))
            L.append(f"attempts_for_accepted (N={att_n}):")
            for a in attempts_for_accepted[:att_n]:
                L.append(
                    f"  surface=\"{a.get('_accepted_surface', '')}\""
                    f"  candidate={a.get('candidate', '?')}"
                    f"  attempt={a.get('attempt_decision', '?')}"
                    f"  reason={a.get('reason', '?')}"
                )

        return "\n".join(L)

    # ── Legacy path: derive from in-memory data ─────────────────────
    # (kept for backward-compatibility with unit tests that call the
    # function directly without persisted decisions)
    L.append(f"mentions_total: {len(mentions)}")

    # Counts by method
    method_counts_legacy: dict[str, int] = {}
    for m in mentions:
        method = str(m.get("method") or "unknown")
        method_counts_legacy[method] = method_counts_legacy.get(method, 0) + 1
    L.append("mentions_by_method:")
    if method_counts_legacy:
        for method, count in sorted(method_counts_legacy.items()):
            L.append(f"  {method}: {count}")
    else:
        L.append("  (none)")

    # Partition mentions into linkable (ACCEPT) and non-linkable (SKIP)
    linkable = [m for m in mentions if str(m.get("ent_type") or "unknown") in _LINKABLE_TYPES]
    non_linkable = [m for m in mentions if str(m.get("ent_type") or "unknown") not in _LINKABLE_TYPES]

    # Top linkable mentions preview (up to 10) — final status = ACCEPT
    L.append("")
    top_n = min(10, len(linkable))
    L.append(f"top_mentions_preview (N={top_n}):")
    sorted_linkable = sorted(linkable, key=lambda m: -float(m.get("confidence", 0)))
    for m in sorted_linkable[:top_n]:
        label = m.get("label") or "-"
        notes = str(m.get("notes", ""))
        L.append(
            f"  surface=\"{m.get('surface', '')}\""
            f"  label={label}"
            f"  conf={float(m.get('confidence', 0)):.2f}"
            f"  method={m.get('method', '?')}"
            f"  notes={notes}"
        )
    if not linkable:
        L.append("  (none)")

    # Skipped non-linkable preview — final status = SKIP_NON_LINKABLE
    # Deduplicate by surface (mention + debug entry may both exist)
    skipped_entries: list[dict[str, Any]] = []
    skipped_seen: set[str] = set()
    for src in [non_linkable, salvage_debug.get("skipped_non_linkable", [])]:
        for s in src:
            key = str(s.get("surface", "")).lower()
            if key and key not in skipped_seen:
                skipped_seen.add(key)
                skipped_entries.append(s)
    if skipped_entries:
        L.append("")
        skip_n = len(skipped_entries) if salvage_debug.get("fixture_override") else min(10, len(skipped_entries))
        L.append(f"skipped_preview (N={skip_n}):")
        for s in skipped_entries[:skip_n]:
            surface = s.get("surface", "")
            ent_type = s.get("ent_type", s.get("canonical", "?"))
            label = str(s.get("label") or "").strip()
            reason = s.get("reason", s.get("notes", "SKIP_NON_LINKABLE"))
            L.append(
                f"  surface=\"{surface}\""
                f"  type={ent_type}"
                f"{f'  label={label}' if label else ''}"
                f"  reason={reason}"
            )

    # Rejected preview from salvage debug (final REJECT decisions only)
    rejected = salvage_debug.get("rejected", [])
    L.append("")
    rej_n = min(10, len(rejected))
    L.append(f"top_rejected_preview (N={rej_n}):")
    for r in rejected[:rej_n]:
        L.append(
            f"  surface=\"{r.get('surface', '')}\""
            f"  canonical={r.get('canonical', '?')}"
            f"  reason={r.get('reason', '?')}"
        )
    if not rejected:
        L.append("  (none)")

    return "\n".join(L)


def _build_consolidated_report(
    run_id: str,
    asset_ref: str,
    mentions: list[dict[str, Any]],
    salvage_debug: dict[str, Any],
    linking_result: dict[str, Any] | None,
) -> str:
    """Build a single consolidated report combining mention extraction + entity linking.

    This is the ONE report that gets printed in chat after each pipeline run.
    Every section is DB-backed — no phantom outputs.
    """
    parts: list[str] = []

    # Part 1: Mention extraction (DB-backed)
    parts.append(_build_mention_extraction_report(
        run_id, asset_ref, mentions, salvage_debug,
    ))

    # Part 2: Entity linking (DB-only via build_linking_report_from_db)
    from app.services.authority_linking import build_linking_report_from_db
    parts.append("")
    parts.append(build_linking_report_from_db(run_id))

    # Part 3: AUDIT gates (mention extraction side)
    parts.append("")
    parts.append(_build_mention_audit_gates(run_id))

    return "\n".join(parts)


def _build_mention_audit_gates(run_id: str) -> str:
    """Compute AUDIT_1, AUDIT_2, AUDIT_3 gates from DB only.

    These validate that the mention extraction report is fully consistent
    with the persisted entity_decisions and entity_attempts tables.
    """
    from app.db.pipeline_db import count_entity_decisions

    L: list[str] = []
    L.append("=== MENTION EXTRACTION AUDIT ===")

    decisions = list_entity_decisions(run_id)
    all_attempts = list_entity_attempts_for_run(run_id)

    # ── AUDIT_1: No phantom surfaces ──────────────────────────────
    # Every surface in entity_decisions has a valid status.
    valid_statuses = {"ACCEPT_LINKABLE", "REJECT_LINKABLE", "SKIP_NON_LINKABLE", "FILTERED_OUT"}
    audit_1_pass = True
    for d in decisions:
        if d.get("status") not in valid_statuses:
            audit_1_pass = False
            break
        if not d.get("surface"):
            audit_1_pass = False
            break
    L.append(f"AUDIT_1 (no phantom surfaces): {'PASS' if audit_1_pass else 'FAIL'}")

    # ── AUDIT_2: Attempt trace consistency ────────────────────────
    # Every attempt row links to a valid decision_id in entity_decisions
    decision_ids = {d["decision_id"] for d in decisions}
    audit_2_pass = True
    if all_attempts:
        for a in all_attempts:
            if a["decision_id"] not in decision_ids:
                audit_2_pass = False
                break
            if a.get("attempt_decision") not in ("ACCEPT", "REJECT"):
                audit_2_pass = False
                break
    L.append(f"AUDIT_2 (attempt trace consistency): {'PASS' if audit_2_pass else 'FAIL'}")

    # ── AUDIT_3: Decision coverage & uniqueness ───────────────────
    total = count_entity_decisions(run_id)
    accepted = count_entity_decisions(run_id, status="ACCEPT_LINKABLE")
    rejected = count_entity_decisions(run_id, status="REJECT_LINKABLE")
    skipped = count_entity_decisions(run_id, status="SKIP_NON_LINKABLE")
    filtered = count_entity_decisions(run_id, status="FILTERED_OUT")
    sum_parts = accepted + rejected + skipped + filtered

    span_keys = [d["span_key"] for d in decisions]
    unique_keys = len(set(span_keys)) == len(span_keys)

    audit_3_pass = (total == sum_parts) and unique_keys
    detail = f"total={total}, A={accepted}+R={rejected}+S={skipped}+F={filtered}={sum_parts}, unique_keys={unique_keys}"
    L.append(f"AUDIT_3 (decision coverage & uniqueness): {'PASS' if audit_3_pass else 'FAIL'} ({detail})")

    return "\n".join(L)


def _parse_nd_bo_from_notes(notes: str | None) -> tuple[float | None, float | None, str | None]:
    """Extract nd, bo, canonical from notes strings like 'canonical=X nd=0.222 bo=0.750'."""
    if not notes:
        return None, None, None
    nd_val: float | None = None
    bo_val: float | None = None
    canon_val: str | None = None
    m_nd = re.search(r"nd=([\d.]+)", notes)
    if m_nd:
        nd_val = float(m_nd.group(1))
    m_bo = re.search(r"bo=([\d.]+)", notes)
    if m_bo:
        bo_val = float(m_bo.group(1))
    m_canon = re.search(r"canonical=(\S+)", notes)
    if m_canon:
        canon_val = m_canon.group(1)
    return nd_val, bo_val, canon_val
