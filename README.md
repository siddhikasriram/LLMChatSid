

## Architecture

```
User → Streamlit UI → RetrievalQA (k=3, stuff)
                         ├─ Chroma (volume)
                         └─ OpenAI gpt-3.5-turbo

Ingest (one-shot) → PDF/CSV/TXT loaders → chunk (size=100, overlap=5)
                 → Instructor-XL (CPU) → Chroma volume
```

Hugging Face model weights are cached on a Docker volume. Chroma stays on disk (not a separate Chroma server).

## Quick start (Docker)

1. Copy the env file and add your OpenAI key:

   ```sh
   cp .env.example .env
   ```

2. Build vectors from everything under `data/` (includes a sample chef resume PDF):

   ```sh
   docker compose --profile ingest up ingest
   ```

   The first run downloads Instructor-XL (~1.5GB) and is slow on CPU.

3. Start the chat UI:

   ```sh
   docker compose up app
   ```

4. Open [http://localhost:8501](http://localhost:8501).

Re-run ingest after you add or replace files in `data/`. Ingest rebuilds the Chroma collection each time.

## Adding documents

Drop `.pdf`, `.csv`, or `.txt` files anywhere under `data/` (subfolders are walked recursively), then re-run the ingest service.

## Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required for the chat UI |
| `OPENAI_MODEL` | Default `gpt-3.5-turbo` |
| `EMBEDDING_MODEL` | Default `hkunlp/instructor-xl` |
| `EMBEDDING_DEVICE` | Default `cpu` |
| `CHROMA_PERSIST_DIR` | Vector store path (`/app/chroma_db` in Docker) |
| `DATA_DIR` | Files to ingest (`/app/data` in Docker) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Same splitter settings as the prototype |
| `RETRIEVER_K` | Default `3` |

`.env` is gitignored. Commit `.env.example` only.

## Local run (without Docker)

Use local paths instead of `/app/...`:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

In `.env` set:

```
DATA_DIR=./data
CHROMA_PERSIST_DIR=./chroma_db
```

Then:

```sh
python -m app.ingest
streamlit run app/ui.py
```

## Project layout

```
app/            # ingest, loaders, RAG chain, Streamlit UI
data/sample/    # one checked-in PDF so the first ingest works
```

## Security

Never commit `.env`.

## Author

Siddhika Sriram — [Email](mailto:siddhikasriram.ss@gmail.com), [LinkedIn](https://www.linkedin.com/in/sid-sriram/)
