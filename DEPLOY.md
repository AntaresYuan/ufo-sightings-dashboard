# Deployment Guide — GitHub Pages

A 10-minute path to a live URL you can submit. You'll need a free GitHub
account; everything else is point-and-click.

## Step 1 — Create a new public repository

1. Go to <https://github.com/new>
2. **Repository name:** something like `ufo-sightings-dashboard` (lowercase,
   no spaces). The URL of your live demo will be
   `https://<your-username>.github.io/<repo-name>/`.
3. **Visibility:** Public (required for free GitHub Pages).
4. **Initialize this repository with: README** — leave _unchecked_.
   (You're about to upload your own files.)
5. Click **Create repository**.

## Step 2 — Upload the files

On the new repo page, click **"uploading an existing file"** (the link is in
the empty-repo prompt), then drag in everything from the `site/` folder
plus the data CSV:

```
index.html
build_final.py
README.md
csv-data/ufo-scrubbed-geocoded-time-standardized.csv
```

Scroll down and click **Commit changes**.

> **Tip:** If GitHub complains the CSV is too large (it shouldn't —
> ~12 MB is fine), you can either skip it (the live demo doesn't need it,
> only the regenerate-from-source workflow does) or use Git LFS.

## Step 3 — Turn on GitHub Pages

1. In the repo, click **Settings** (top tab).
2. Left sidebar → **Pages**.
3. Under **Build and deployment**:
   - **Source:** Deploy from a branch
   - **Branch:** `main`, folder `/ (root)`
   - Click **Save**.
4. Wait ~1–2 minutes. The page will refresh with a green box showing your
   site URL: `https://<your-username>.github.io/<repo-name>/`

## Step 4 — Verify

Visit the URL. You should see the dashboard title, the lede paragraph,
and three sections of visualizations. The first load can take 3–5
seconds while the page fetches Vega-Lite from the CDN.

If something looks broken:
- **Map area is empty?** The U.S. TopoJSON loads from a public CDN
  (`cdn.jsdelivr.net`). Most networks allow this; a few blocked corporate
  networks won't.
- **Page is blank?** Open browser dev tools → Console for the error.
  Most likely a syntax error in `index.html` if you edited it.

## Step 5 — Submit

Copy the live URL and paste it in the Canvas submission box, along with
a link to the repo itself for source code:

```
Live demo: https://<your-username>.github.io/<repo-name>/
Source:    https://github.com/<your-username>/<repo-name>
```

## Updating later

Edit any file directly in the GitHub web UI (pencil icon), commit, and
the live site will update within a minute. Or regenerate locally with
`python3 build_final.py` and re-upload `index.html`.
