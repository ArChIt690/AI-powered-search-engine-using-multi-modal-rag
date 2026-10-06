# Share the search engine over Tailscale (HTTPS): plan and outcome

> Approved 2026-10-05; built 2026-10-06. A portfolio project showing internal company search: users only ask
> questions, admins add documents, nothing is on the public internet.

## Built
- `tailscale` service (profile `tailscale`): `tailscale/tailscale:v1.102.4`, hostname `search`, userspace mode,
  `TS_AUTHKEY` from `.env`, state volume, `tailscale/serve.json`: HTTPS 443 → `frontend:8501`, 8443 → `admin:8501`,
  Funnel off (tailnet only).
- All host ports bound to `127.0.0.1` (Elasticsearch, Redis, API, both pages).
- `frontend/app.py` = search page (no uploads; "Signed in as" from Tailscale); `frontend/admin.py` = admin page
  (uploads, file list; only `SEARCH_ADMINS` through Tailscale; open on the machine itself); `frontend/shared.py`.
- `tailscale/policy.example.hujson`: optional ACL (everyone → 443, admins → 8443, tag `tag:search-engine`).
  The user couldn't apply it, so the tag is optional (`TS_EXTRA_ARGS`) and the admin page relies on the
  `SEARCH_ADMINS` identity check.

## Checked
- The `search` machine joined the tailnet (`search.taild9e042.ts.net`, 100.83.179.72); serve status shows both
  pages, tailnet only.
- `https://search.taild9e042.ts.net` and `:8443` return 200 with a verified certificate (first request 36 s while
  the certificate was issued, then ~0.03 s).
- Elasticsearch, Redis, API and the raw page ports are closed on the `search` machine and on every address of the
  laptop.
- Fixed in `.env`: spaces around `=` and a typo in the admin email.
