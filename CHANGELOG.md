# Changelog

Versioning starts here. Older changes are summarized in the initial entry.

## 0.3.0 — 2026-06-25

- **Feature: ⚡ Auto-match** — a one-shot bulk identity matcher. Instead of
  the one-card-at-a-time Y/N walk, it classifies every loaded unnamed face
  against your named identities in parallel (new `POST /api/ai/suggest-batch`,
  fanned out across `AI_BATCH_WORKERS` threads) and presents a single review
  list. Confident matches (≥85%) come pre-checked, so the common case is one
  click to merge dozens of faces. The step-through **🤖 AI suggest** mode is
  unchanged for when you want to review individually. Concurrency makes the
  wall-clock time roughly one Gemini call deep instead of N calls deep.
- **Fix (coverage): we now harvest *every* face Protect has seen.** Verified
  against a live instance, `/events` is hard-capped at ~1000 results (~30h)
  per request and ignores the `page` param, so the old single-request harvest
  only saw ~8h of history and surfaced ~90 phantom groups. The harvest now
  walks a **sliding window** (`end = oldest - 1`, repeated `PHANTOM_WINDOWS`
  times, default 4 ≈ ~4 days) — surfacing 600+ phantom groups on the test
  system. It also reads the group id from the thumbnail `labels` array
  (`group:<id>`) in addition to the `group` object, catching ~6% more groups
  that only encode it there, and uses `groupType:degraded` from labels to
  flag low-quality clusters reliably. New tunables: `PHANTOM_WINDOWS`,
  `PHANTOM_EVENT_LIMIT`, `PHANTOM_CACHE_TTL`.
- **UI: the "Face Triage" title is now a home button.** Click it to leave the
  identity detail view, clear any selection, and scroll back to the top of
  the triage grid.
- **UI: added a favicon** (served at `/favicon.svg` and `/favicon.ico`).

## 0.2.1 — 2026-05-19

- **Fix (critical): new faces from Protect didn't appear in triage.** The
  phantom-group harvest was hitting `/events` without an `orderDirection`
  parameter — Protect defaults to ASC, so `limit=500` over the 7-day window
  returned the OLDEST 500 events and skipped everything recent. Verified on a
  live Protect 7.1.55 instance: 0 phantoms harvested without `DESC`, 18+ with
  it. Added `orderDirection=DESC` to the events request.
- **Fix: 🤖 Let AI pick** appeared to do nothing — it would say "AI picked
  one" but no tile got the blue halo. Root cause: the candidate-grid endpoint
  returns top 8 detections ranked by `(has_enhanced, confidence, recency)`,
  but the AI suggest endpoint ranked top 12 by `confidence` only. AI was
  picking ids that weren't in the displayed grid. The UI now sends the
  displayed `enhancedImageId`s to the AI endpoint, which restricts its pick
  to exactly those tiles.

## 0.2.0 — 2026-05-19

- UI redesigned with a UniFi-inspired palette:
  - Navy base (`#0b1220`) with subtle radial gradient backdrop
  - Vibrant cyan-blue accent (`#0094ff`) matching UniFi Protect
  - Brighter status colors: emerald `#22d37c`, amber `#ffa726`, coral
    `#ff4d5e`
  - Title bar gets a glowing status dot + light gradient text
  - Cards lift higher on hover with a blue glow; selection ring uses a
    halo effect
  - Primary buttons gain a UniFi-style gradient + outer glow
  - Drag/drop "merge target" identity rows now glow blue, not green,
    matching the accent
- Pill, sidebar, and modal styling refined to match the new palette.

## 0.1.3 — 2026-05-19

- Fix: delete on phantom face groups appeared to do nothing in the UI
  because the phantom-harvest cache (was 10s TTL) wasn't being invalidated
  on mutations. The actual `DELETE /recognition/face/groups/{id}` API call
  was succeeding silently. Now `merge`, `rename`, `delete`, `reassign` all
  invalidate the phantom cache.
- Phantom cache TTL dropped 10s → 4s for snappier baseline freshness.
- Fix: new face events from Protect didn't appear in triage without a
  manual refresh. Added a 30s auto-refresh of the unenrolled grid that
  runs only while the triage view is focused, no selection is active, no
  modal is open, and AI suggest mode isn't running.

## 0.1.2 — 2026-05-19

- Fix: "🤖 AI pick" badge on the best-avatar tile was getting clipped by
  the candidate grid's scroll container. Moved inside the tile as a
  corner-pill overlay with backdrop blur — no longer cut off.

## 0.1.1 — 2026-05-19

- Add explicit version tracking. `app/version.py` is the source of truth.
- Expose `GET /api/version` returning `{version, buildDate}`.
- Show version next to "Face Triage" in the webapp header.
- Log version on enhancer startup so it's visible in `docker logs`.

## 0.1.0 — 2026-05-19

Initial versioned release. Snapshot of the project at the time versioning
was introduced. Highlights:

- **Enhancer loop**: long-lived background sweep that auto-enhances every
  face detection via the AI Key's `/recognition/face/detections/{id}/image/enhance`
  endpoint, with adaptive throttling based on the AI Key task queue.
- **Face Triage webapp** on port 8080:
  - Lists named identities + unnamed clusters (including phantom groups
    harvested from recent events).
  - Click / drag-drop / 1–9 hotkey to merge unnamed clusters into named identities.
  - Per-identity detail view with detection grid, move/unassign, delete identity.
  - **🎯 Best avatar** picker with optional **🤖 Let AI pick** that previews
    Gemini's choice on a candidate tile before you commit.
  - **🤖 AI suggest** mode that walks unnamed faces and proposes identity
    matches via Gemini, with Y/N confirm and next-card pre-fetching.
  - Enhancer status pill in the header with click-through detail modal and
    a Run-now button.
  - Retroactive processing trigger for the AI Key.
- Modern dark UI (Linear/Vercel-inspired): indigo accent, refined cards,
  modal backdrop blur, smooth motion.
- Show low-quality / show-degraded toggle (defaults on).
- Avatar cache-busting on every mutation so photos refresh immediately.
- Performance: parallel image fetching, server-side group/reference caches,
  client-side infinite scroll on grids.
