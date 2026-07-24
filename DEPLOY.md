# Deploying to Streamlit Community Cloud

This app is prepped for **free** hosting on Streamlit Community Cloud. Follow
these steps once; after that, every `git push` auto-redeploys.

## What's already set up

- `requirements.txt` — dependency list Streamlit Cloud installs from
- `.streamlit/config.toml` — headless server config
- **Auto-ingest on first launch** — the app builds `chroma_store/` from the
  committed data on its first run, so the vector store isn't in the repo
- `.gitignore` — keeps `.env` and `.streamlit/secrets.toml` (your key) out of git

## Step 1 — Put the code on GitHub

Create a **new repository** at https://github.com/new (public is required for
the free tier — the code and synthetic data are safe to share; your key is not
in the repo).

Then, from the project folder:

```bash
git init
git add .
git commit -m "Telecom RAG chatbot"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> Before pushing, confirm your key is NOT staged:
> ```bash
> git status --short | grep -E "\.env|secrets\.toml"   # should print nothing
> ```

## Step 2 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings → Secrets** and paste:

   ```toml
   GROQ_API_KEY = "gsk_your-new-key-here"
   ```

5. Click **Deploy**.

The first build takes a few minutes (installing PyTorch, downloading the
embedding model, building the vector store). After that it's live at
`https://<your-app-name>.streamlit.app`.

## Updating later

Edit code locally, then:

```bash
git add .
git commit -m "…"
git push
```

Streamlit redeploys automatically. To change the Groq key, edit it in the
Streamlit dashboard under **Settings → Secrets** (never in the repo).

## Notes

- **Memory:** the free tier has ~1 GB RAM. PyTorch + `all-MiniLM-L6-v2` fits,
  but if the app ever restarts under load, that's the likely cause.
- **Cost:** none. Groq's free tier rate-limits rather than charges, so a
  traffic spike slows the app — it never bills you.
