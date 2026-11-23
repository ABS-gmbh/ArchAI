# Upstream provenance

The application in `archai/` began as a copy of
[emanuskript/layout](https://github.com/emanuskript/layout). This file records
what the repository's own history shows about that copy, and what it does not.

| | |
|---|---|
| Upstream repository | `https://github.com/emanuskript/layout`, recorded as `remote "origin"` in a stray `archai/vendor/layout/.git/config` (as the audit found it, HYG-13), which also named the branches `main` and `feat/ui-improvements` and had an `[lfs]` section |
| Upstream commit | **Not recorded.** The stray `.git` held only that config file (no `HEAD`, no objects, no refs), so the commit cannot be recovered from this repository |
| Brought into this repository | `4203a98` (2026-04-27), "Refactor repo into unified ArchAI monorepo with full docs", under `archai/vendor/layout/` |
| Moved to `archai/` | `1a5c949` (2026-10-22); the stray `.git` directory did not move with it |
| Upstream traces left | `archai/tools/layout_comparison/` still carries absolute paths from the machine its scripts were written on (`/home/hasan/layout/...`) |
| Licence | `archai/LICENSE` (MIT, Mohamed Basuony) was added with the first commit, `a1372c0`. The upstream project's own licence was not recorded with the copy |

Everything changed since the copy is in this repository's history from
`4203a98` on; what differs from upstream before that point cannot be told
apart.

To pin the upstream commit, clone the upstream repository and find the commit
whose tree is closest to `git show 4203a98:archai/vendor/layout` (for
example, the fewest differing files across its history), then record it
here. Before redistributing the application, check the upstream licence and
keep its notice if it requires one.
