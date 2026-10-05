# Deploying the public demo

Live instance: https://lgla6kmbujefu7glenjaqt.streamlit.app/

The hosted demo runs on **Streamlit Community Cloud** (free). It deploys straight from GitHub, so
there's nothing to build. Each visitor either runs the free offline mode or pastes their own
Anthropic API key, so the demo never spends your credits.

## One-time setup (about 5 minutes)

1. Merge the `ada-v4-agent-team` branch into `main` on GitHub.
2. Go to **https://share.streamlit.io** and click **Continue with GitHub**.
3. Click **Create app → Deploy a public app from GitHub** and fill in:

   | Field | Value |
   |---|---|
   | Repository | `sid23git/ada-agent-team` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | App URL | e.g. `ada-agent-team` → `https://ada-agent-team.streamlit.app` |

4. Open **Advanced settings**:
   - **Python version:** `3.12`
   - **Secrets:** paste exactly this line:

     ```toml
     ADA_PUBLIC_DEMO = "1"
     ```

     Do **not** put your own `ANTHROPIC_API_KEY` here. Public mode ignores it anyway, so
     strangers can never spend your credits.
5. Click **Deploy**. The first build installs dependencies and takes about 3–5 minutes.

## What public mode changes

| Setting | Local | Public demo (`ADA_PUBLIC_DEMO=1`) |
|---|---|---|
| API key | `.env` / environment | visitor's own key, used for their session's runs only |
| Default agents | live if a key is set | offline (free, instant) |
| Budget cap | $10 | $2 per run |
| Upload size | unlimited | 25 MB, first 200,000 rows |
| `run_python` tool | on | **off**: an uploaded CSV is untrusted input, and text inside it could try to steer an agent into misusing a code tool |

Each investigation gets its own Anthropic client, so concurrent visitors can never use each
other's key. `tests/test_investigation.py` and `tests/test_app.py` both check this.

## After it's live

- Put the URL at the top of the README, replacing the placeholder under the title.
- Put it on your LinkedIn post and in your resume's project line.
- The free tier sleeps after a few days without visitors. The first visitor after that waits
  about 30 seconds while it wakes up.
