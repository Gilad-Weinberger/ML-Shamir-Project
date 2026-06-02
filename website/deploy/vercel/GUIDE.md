# Vercel + Supabase Deployment Guide

Deploy the Django grape leaf assessment app to **Vercel** with **Supabase Storage** for upload previews and **Hugging Face Space** for model inference (no PyTorch on Vercel).

---

## Architecture

```
User → Vercel (Django UI)
         ├─► Supabase Storage (upload preview URL)
         └─► HF Space Gradio API (prediction)
```

Complete the [Hugging Face guide](../huggingface/GUIDE.md) first so `HF_INFERENCE_URL` points to a running Space.

---

## Prerequisites

- GitHub repo with this project pushed
- [Vercel](https://vercel.com) account
- [Supabase](https://supabase.com) account
- HF Space running (from HF deploy kit)

---

## Step 1 — Supabase project & storage bucket

1. Create a project at [supabase.com/dashboard](https://supabase.com/dashboard)
2. Open **Database → Extensions** → enable **pg_cron**, **pg_net**, and **http**
3. Open **SQL Editor** → **New query**
4. Paste and run [`supabase_setup.sql`](supabase_setup.sql)
5. In the same file, **uncomment** the `vault.create_secret` block, replace placeholders with your **Project URL** and **service_role** key, and run that block once

This creates:

| Item | Detail |
|------|--------|
| **Bucket** | `leaf-uploads` — **public** read for preview URLs |
| **File types** | JPEG, PNG, WebP, GIF (max 10 MB) |
| **Cron job** | `purge-leaf-uploads-every-48h` — deletes **all** images via the **Storage API** every **48 hours** (00:00 UTC on odd calendar days: 1st, 3rd, 5th, …) |
| **Vault** | `supabase_project_url` + `supabase_service_role_key` — required for the purge function (never commit real keys) |

6. Confirm bucket `leaf-uploads` exists under **Storage**
7. Confirm cron job under **Database → Cron** (or run):

```sql
select jobid, jobname, schedule, command
from cron.job
where jobname = 'purge-leaf-uploads-every-48h';
```

To add or reset only the cron job (after Vault + purge function exist), use [`supabase_cron_cleanup.sql`](supabase_cron_cleanup.sql). To recreate only the purge function, use [`supabase_purge_function.sql`](supabase_purge_function.sql).

Copy from **Project Settings → API**:

| Setting | Env var |
|---------|---------|
| Project URL | `SUPABASE_URL` |
| `service_role` key (secret) | `SUPABASE_SERVICE_ROLE_KEY` |

Use the **service role** key server-side only (never expose in frontend code).

---

## Step 2 — Hugging Face inference URL

From your running HF Space, set:

```
HF_INFERENCE_URL=https://YOUR_USERNAME-grape-leaf-inference.hf.space
```

If the Space or model repo is private, also set `HF_TOKEN`.

---

## Step 3 — Import project to Vercel

1. Go to [vercel.com/new](https://vercel.com/new) → import your GitHub repo
2. **Framework Preset:** **Django** (or leave empty and use `website/vercel.json` → `"framework": "django"`)
3. **Root Directory:** `website` (required)
4. In **Settings → Build and Deployment**, leave **Install Command**, **Build Command**, and **Output Directory** **empty** (use platform defaults).
5. Turn **off** “Include source files outside of the Root Directory” (if shown).

`website/vercel.json` sets `"framework": "django"`. Vercel then:

- Installs deps with **uv** from `pyproject.toml` / `requirements.txt`
- Runs **`collectstatic` automatically** (because `STATIC_ROOT` is set — do not add a manual collectstatic build command)
- Deploys a **Python function** (no `public` output folder)

Dependencies are pinned in **`requirements.txt`** and mirrored in **`pyproject.toml`** `[project.dependencies]`.

Entrypoint: `pyproject.toml` → `[tool.vercel] entrypoint = "website.wsgi:application"`.

Do **not** add a `vercel.json` or `.vercelignore` at the **repo root** — that conflicts with Root Directory `website`.

---

## Step 4 — Environment variables

Set these in **two places** depending on where you run the app:

| Where you run | Where to put env vars |
|---------------|------------------------|
| **Production (Vercel)** | Vercel dashboard → your project → **Settings → Environment Variables** |
| **Local (`py manage.py runserver`)** | Copy **`website/.env.example`** → **`website/.env.local`** (auto-loaded by Django) |

Use [`website/.env.example`](../../.env.example) as the local template. For Vercel, set the same keys in the dashboard (**Production** / **Preview**).

---

### Variable-by-variable guide

#### `DJANGO_SECRET_KEY`

| | |
|---|---|
| **What it is** | Random secret Django uses for sessions, CSRF, etc. |
| **Where to get it** | Generate yourself — any long random string (50+ chars). Example PowerShell: `[System.Convert]::ToBase64String((1..48 \| ForEach-Object { Get-Random -Maximum 256 }))` |
| **Where to put it** | **Vercel:** Settings → Environment Variables → name `DJANGO_SECRET_KEY`, value = your string. **Local:** export in terminal or add to `.env`. |
| **Example** | `k8x#m2p9...` (never commit the real value) |
| **Required** | Yes (production) |

---

#### `DEBUG`

| | |
|---|---|
| **What it is** | Turns Django debug mode on/off. |
| **Where to get it** | You choose the value — not from a third-party dashboard. |
| **Where to put it** | **Vercel:** `DEBUG` = `False`. **Local:** omit (defaults to `True`) or set `True`. |
| **Example** | `False` |
| **Required** | Yes on Vercel |

---

#### `ALLOWED_HOSTS`

| | |
|---|---|
| **What it is** | Comma-separated list of hostnames Django will accept. |
| **Where to get it** | Your Vercel URL after first deploy (e.g. `ml-shamir-project.vercel.app`) plus any custom domain. |
| **Where to put it** | **Vercel:** one variable, comma-separated. **Local:** usually leave empty for `localhost`. |
| **Example** | `.vercel.app,my-app.vercel.app,leaf.example.com` |
| **Required** | Yes on Vercel |

Tip: `.vercel.app` matches any `*.vercel.app` subdomain.

---

#### `SUPABASE_URL`

| | |
|---|---|
| **What it is** | Your Supabase project API URL. |
| **Where to get it** | [Supabase Dashboard](https://supabase.com/dashboard) → your project → **Project Settings** (gear) → **API** → **Project URL** |
| **Where to put it** | **Vercel:** `SUPABASE_URL`. **Local:** same, if testing Supabase uploads locally. |
| **Example** | `https://abcdefghijklmnop.supabase.co` |
| **Required** | Yes (for upload previews in production) |

---

#### `SUPABASE_SERVICE_ROLE_KEY`

| | |
|---|---|
| **What it is** | Server-side API key with full access (bypasses RLS). Used by Django to upload images. |
| **Where to get it** | Supabase → **Project Settings → API** → **Project API keys** → **`service_role`** → **Reveal** (secret) |
| **Where to put it** | **Vercel:** `SUPABASE_SERVICE_ROLE_KEY` (mark as sensitive). **Local:** same — never commit to git. |
| **Example** | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` |
| **Required** | Yes (production) |

**Never** use the `anon` key here. **Never** expose `service_role` in frontend code or public repos.

---

#### `SUPABASE_STORAGE_BUCKET`

| | |
|---|---|
| **What it is** | Name of the Storage bucket for leaf upload previews. |
| **Where to get it** | You define it when running [`supabase_setup.sql`](supabase_setup.sql) — default is `leaf-uploads`. Confirm under **Storage** in Supabase. |
| **Where to put it** | **Vercel:** `SUPABASE_STORAGE_BUCKET` = `leaf-uploads`. **Local:** same. |
| **Example** | `leaf-uploads` |
| **Required** | Yes (production) |

---

#### `HF_INFERENCE_URL`

| | |
|---|---|
| **What it is** | Base URL of your Hugging Face **Space** (Gradio inference app). |
| **Where to get it** | [huggingface.co/spaces](https://huggingface.co/spaces) → your Space (e.g. `GiladWeinberger/grape-leaf-inference`) → copy the public URL **without** a trailing path. |
| **Where to put it** | **Vercel:** `HF_INFERENCE_URL`. **Local:** same, to test remote inference without a local `.pth`. |
| **Example** | `https://GiladWeinberger-grape-leaf-inference.hf.space` |
| **Required** | Yes on Vercel (no PyTorch on serverless) |

See [HF guide](../huggingface/GUIDE.md) if the Space is not running yet.

---

#### `HF_TOKEN`

| | |
|---|---|
| **What it is** | Hugging Face access token so Django can call a **private** Space or private model repo. |
| **Where to get it** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **Create new token** → **Read** access is enough for inference |
| **Where to put it** | **Vercel:** `HF_TOKEN` (sensitive). **Local:** same. |
| **Example** | `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` |
| **Required** | Yes if your Space or model repo is **private**; optional if both are public |

---

#### `HF_MODEL_REPO` and `HF_MODEL_FILE` (optional on Vercel)

| | |
|---|---|
| **What they are** | Hugging Face **model** repo id and weights filename — used to download a `.pth` locally, not for Vercel inference. |
| **Where to get them** | Your model repo (e.g. `GiladWeinberger/grape-leaf-5deg`) and file `grape_leaf_model_5deg.pth` after running `upload_model.py`. |
| **Where to put them** | **Local only** (optional). **Not needed on Vercel** when `HF_INFERENCE_URL` is set. |
| **Example** | `HF_MODEL_REPO=GiladWeinberger/grape-leaf-5deg`, `HF_MODEL_FILE=grape_leaf_model_5deg.pth` |

---

### Quick checklist — Vercel dashboard

1. Open [vercel.com](https://vercel.com) → your project → **Settings → Environment Variables**
2. Add each variable below for **Production** (and **Preview** if desired):

| Variable | Get value from |
|----------|----------------|
| `DJANGO_SECRET_KEY` | Generate random string |
| `DEBUG` | Type `False` |
| `ALLOWED_HOSTS` | Your `*.vercel.app` URL (+ custom domain) |
| `SUPABASE_URL` | Supabase → Settings → API → Project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase → Settings → API → service_role key |
| `SUPABASE_STORAGE_BUCKET` | `leaf-uploads` (from SQL setup) |
| `HF_INFERENCE_URL` | Your HF Space URL |
| `HF_TOKEN` | huggingface.co/settings/tokens (if private) |

3. **Redeploy** after adding or changing variables (Deployments → ⋮ → Redeploy).

---

### Quick checklist — local testing

1. Copy env template:

```bash
cd website
copy deploy\vercel\.env.example .env
```

2. Edit **`website/.env.local`** with your Supabase and HF keys (see variable guide above).

3. Install and run:

```bash
pip install -r requirements-local.txt
py manage.py runserver
```

4. Upload a leaf image — the preview URL should start with `https://...supabase.co/storage/v1/object/public/leaf-uploads/...`

---

## Step 5 — Deploy

1. Click **Deploy**
2. Wait for build (`collectstatic` + dependency install)
3. Open the deployment URL (e.g. `https://your-app.vercel.app`)

---

## Step 6 — Verify

1. Open the app URL
2. Upload a leaf image
3. Confirm:
   - Preview image loads from a Supabase `supabase.co/storage/...` URL
   - Prediction percentage appears (may take 30–60s on first request if Space was sleeping)
   - CSS/styles load correctly

---

## Environment variables reference (summary)

| Variable | Where to get it | Where to put it |
|----------|-----------------|-----------------|
| `DJANGO_SECRET_KEY` | Generate a random string | Vercel env vars; local terminal / `.env` |
| `DEBUG` | Use `False` (production) | Vercel env vars |
| `ALLOWED_HOSTS` | Your Vercel app URL / domain | Vercel env vars |
| `SUPABASE_URL` | Supabase → Settings → API → Project URL | Vercel + local (if testing uploads) |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase → Settings → API → **service_role** | Vercel + local (keep secret) |
| `SUPABASE_STORAGE_BUCKET` | `leaf-uploads` from `supabase_setup.sql` | Vercel + local |
| `HF_INFERENCE_URL` | HF Space page URL (base, no trailing `/`) | Vercel + local |
| `HF_TOKEN` | huggingface.co/settings/tokens | Vercel + local (if Space/repo private) |
| `HF_MODEL_REPO` | HF model repo after upload | Local only (optional) |
| `HF_MODEL_FILE` | Weights filename in repo | Local only (optional) |

Full details: [Step 4 — Environment variables](#step-4--environment-variables).

---

## Important notes

### Upload retention (48-hour purge)

Uploaded leaf previews are **temporary**. A Supabase **pg_cron** job deletes **every file** in `leaf-uploads` on a 48-hour cycle. After a purge, old preview URLs in the browser will break — that is expected.

To run a one-off purge manually (uses Storage API — do **not** use `DELETE FROM storage.objects`):

```sql
select public.purge_leaf_uploads_storage();
```

To change the schedule, edit the cron expression in `supabase_setup.sql` and re-run [`supabase_cron_cleanup.sql`](supabase_cron_cleanup.sql).

If cron logs **"Direct deletion from storage tables is not allowed"**, your job still uses the old SQL `DELETE`. Run [`supabase_purge_function.sql`](supabase_purge_function.sql), add Vault secrets, then [`supabase_cron_cleanup.sql`](supabase_cron_cleanup.sql).

### No persistent disk on Vercel

Do **not** rely on `MEDIA_ROOT` in production. Supabase must be configured or upload previews will fail.

### No PyTorch on Vercel

Inference runs on your HF Space. Root `requirements.txt` excludes `torch` (use `requirements-local.txt` locally).

### Cold starts

HF free-tier Spaces sleep when idle. The first prediction after idle can take 30–60 seconds.

### Custom domain

Add your domain in Vercel → **Domains**, then append it to `ALLOWED_HOSTS`.

### Local development

Copy [`website/.env.example`](../../.env.example) to **`website/.env.local`** and fill in your Supabase + HF values. Django loads `.env.local` (or `.env`) on startup.

```bash
cd website
copy .env.example .env.local   # Windows
# edit .env.local with your keys
pip install -r requirements-local.txt
py manage.py runserver
```

Uploads go to **Supabase Storage** only (no local `media/` folder). If Supabase env vars are missing, the app shows a configuration error instead of saving locally.

For local inference without Vercel, either set `HF_INFERENCE_URL` or keep a `.pth` in `website/base/`. See root `commands.txt` and the HF guide.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `DisallowedHost` | Add your Vercel URL/domain to `ALLOWED_HOSTS` |
| Static CSS missing | Redeploy after `CompressedStaticFilesStorage` in settings; confirm `/static/css/base/home.css` loads in browser Network tab |
| `Invalid API key` on upload | **Supabase** error → fix `SUPABASE_SERVICE_ROLE_KEY` (JWT `eyJ...`, not `hf_`). **HF** error → remove `HF_TOKEN` if Space is public, or set a valid Read `hf_` token |
| Upload fails | Verify Supabase env vars and run `supabase_setup.sql`; enable **pg_cron**, **pg_net**, and **http** |
| Cron: "Direct deletion from storage tables is not allowed" | Re-run `supabase_purge_function.sql` + Vault secrets + `supabase_cron_cleanup.sql` (Supabase blocks raw `DELETE` on `storage.objects`) |
| Cron: missing Vault secrets | Run the `vault.create_secret` block in `supabase_setup.sql` with your Project URL and service_role key |
| Preview URL broken after 2 days | Expected — cron purges all leaf uploads every 48 hours |
| No prediction / timeout | Wake HF Space in browser; check `HF_INFERENCE_URL` |
| 401 from HF | Set `HF_TOKEN` in Vercel env |
| Build too large | Log must show slim deps (no `torch`). Clear dashboard Install Command override. Remove repo-root `vercel.json` if present. |
| `externally-managed-environment` | Remove custom `pip install` Install Command; use empty install + uv (see `pyproject.toml`). |
| `No Output Directory named "public"` | Set Framework Preset to **Django** (not Other). Remove manual `buildCommand` / clear Output Directory in dashboard. |
| Log shows `/website/...` in ignore | Usually an old **repo-root** `.vercelignore`; delete it and redeploy with only `website/.vercelignore` |

---

## Verification checklist

- [ ] Deploy succeeds without torch in the bundle
- [ ] Upload shows Supabase public image URL in the UI
- [ ] Prediction returns via HF Space
- [ ] Static CSS loads (WhiteNoise + collectstatic)

---

## Files in this folder

| File | Purpose |
|------|---------|
| `GUIDE.md` | This guide |
| `website/.env.example` | Local env template (copy to `website/.env.local`) |
| `deploy/vercel/.env.example` | Same keys, Vercel-oriented comments |
| `supabase_setup.sql` | Public bucket, policies, purge function, 48h purge cron |
| `supabase_purge_function.sql` | Purge function + Vault template (if setup already ran) |
| `supabase_cron_cleanup.sql` | Reschedule cron only (after purge function + Vault exist) |
| `requirements-vercel.txt` | Mirror of root `requirements.txt` (Vercel / production) |
| `requirements-local.txt` | Full local stack (torch, matplotlib, training) |
| `build.sh` | Install + collectstatic |
| `vercel.json` | Vercel config template |
