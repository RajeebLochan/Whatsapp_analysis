# Security incident runbook — exposed API key and chat data in git history

**Audience:** the repository owner (these steps need owner access and judgement).
**Status:** exposure confirmed on 2026-11-04 (see `AUDIT_REPORT.md` F-01, F-02, F-09).
Nothing in this runbook was executed by the remediation branch; no history was
rewritten, nothing was force-pushed, no deployed data was touched and the exposed
key was never used or tested.

## What was exposed

| Item | Where | Since | Notes |
|---|---|---|---|
| API key (64 hex chars, `TOGETHER_API_KEY` in `.env`) | Commit `88e21e7` (2025-07-15) and its descendants until `3fa0a62` (2025-07-18) replaced the value | 2025-07-15 | The repository is **public**; assume the key was harvested. |
| Uploaded WhatsApp chat exports (filenames + full text) | `uploaded_files.db` in commits from `88e21e7` onward, including the audited HEAD `459cf1a` | 2025-07-15 | Up to 9 live rows in history; deleted rows still recoverable from SQLite free pages in later commits. Contents were **not** read during the audit. |
| Same database on servers | Any deployment that ran the old code (Streamlit Community Cloud demo, Azure App Service `whatsappanalyze`) | First deployment | Every upload to the live app was written to `uploaded_files.db` on that server. |
| Compiled `__pycache__/*.pyc` | History | 2025-07 | No secrets detected; remove for hygiene. |

Removing the files from the current branch (done in this PR) **does not** remove
them from history, from clones, from forks or from GitHub's caches.

## 1. Contain the credential (do first, today)

1. Sign in to Together AI → *Settings → API keys*. **Revoke** every key that existed
   on or before 2025-07-18 (or that matches the old `.env`).
2. Create a new key only if a shared deployment key is really needed. Store it only in
   the hosting provider's secret store (Streamlit *Secrets*, Azure *App settings*),
   never in the repository. Leave `WA_ENABLE_SHARED_AI_KEY` unset/false unless step 5
   is in place.
3. Review the provider's usage and billing history from 2025-07-15 to today for
   unexpected requests; dispute charges if needed.
4. Set provider-side spending limits and alerts.
5. Before enabling a shared key, add access control to the deployment (e.g.
   Streamlit Cloud viewer authentication) — the per-session request cap in the app
   is not a spending control.

## 2. Assess the chat-data exposure

1. Decide, without redistributing it, whose data the database contained: your own test
   chats, or chats uploaded by visitors of the live demo. Inspect it locally and
   offline only (e.g. `sqlite3` row counts and filenames), then delete local copies.
2. If third-party data was exposed, take advice on notification duties that may apply
   (for example India's DPDP Act 2023, GDPR for EU residents) and notify affected
   people where required or appropriate.
3. Record what you found, when, and what you decided.

## 3. Clean the deployed servers

1. **Streamlit Community Cloud:** redeploy from the fixed `main` (the new code creates
   no database). Rebooting/redeploying replaces the container filesystem; confirm the
   old `uploaded_files.db` is gone.
2. **Azure App Service `whatsappanalyze`:** open Kudu/SSH, locate
   `uploaded_files.db` under the site directory (e.g. `/home/site/wwwroot`), handle it
   according to step 2 (preserve evidence if needed), then delete it. Redeploy the
   fixed build. Check for backups/snapshots of the App Service that contain the file.
3. Set the App Service startup command to
   `python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0`.

## 4. Purge git history (owner only, coordinated)

This rewrites every commit hash. Merge or close open pull requests first (including the
remediation PR), tell collaborators, and make a private backup.

```bash
pip install git-filter-repo
git clone --mirror https://github.com/RajeebLochan/Whatsapp_analysis.git wa-mirror.git
cp -r wa-mirror.git wa-mirror-backup.git          # keep privately, delete after verification
cd wa-mirror.git
git filter-repo --invert-paths \
  --path .env \
  --path uploaded_files.db \
  --path-glob '__pycache__/*' \
  --path-glob '*.pyc'
# verify: both commands must print nothing
git log --all --oneline -- .env uploaded_files.db
git log --all -p | grep -E '[0-9a-f]{64}' | head
git remote add origin https://github.com/RajeebLochan/Whatsapp_analysis.git
git push --force --all origin
git push --force --tags origin
```

Afterwards:

1. GitHub keeps hidden `refs/pull/*` refs and cached views of old commits. Contact
   GitHub Support ("remove sensitive data") with the affected commit SHAs
   (`88e21e7`, `073ad37`, `d183573`, `ede69d2`, `d95dff9`, `3fa0a62`, `459cf1a` and
   any other commit containing the files) and ask them to purge cached views and
   pull-request refs. Before rewriting, list every commit whose tree contains the
   files (all 17 commits reachable from `459cf1a` did at audit time):

   ```bash
   git rev-list --all | while read c; do
     git ls-tree -r --name-only "$c" | grep -qxE '\.env|uploaded_files\.db' && echo "$c"
   done
   ```
2. Check for forks or clones created since publication (0 forks at audit time) and
   ask their owners to delete them.
3. Everyone with a clone must re-clone; old clones still contain the data.
4. Delete the private backup once you have verified the purge.

## 5. Verify and prevent recurrence

1. Run a secret scanner over the rewritten history, e.g.
   `gitleaks detect --source . --log-opts="--all"`.
2. Enable GitHub *secret scanning* and *push protection* for the repository.
3. Keep `.gitignore` entries for `.env*`, `*.db`, `.streamlit/secrets.toml`.
4. Consider a pre-commit hook (gitleaks) for contributors.
5. Re-read the privacy section of `README.md` before each release and keep it
   accurate.

## Checklist

- [ ] Old Together key(s) revoked; usage reviewed; spending limits set
- [ ] Chat-data exposure assessed; notification decision recorded
- [ ] `uploaded_files.db` removed from Streamlit Cloud and Azure (and backups)
- [ ] Fixed version deployed; startup command verified
- [ ] History purged; GitHub Support request filed; forks/clones handled
- [ ] Secret scanning and push protection enabled
