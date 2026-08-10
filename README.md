# Smart Column Mapper

An interactive web application that maps columns from uploaded files (CSV, XLSX, XLS) to a canonical set of data columns. The mapping system uses a hybrid approach: deterministic rule-based synonym matching, optionally refined by LLM recommendations via Ollama (`gpt-oss:120b`).

---

## Features

- **Hybrid Mapping Engine**: Rule-based synonym matching runs first and always; an optional "Enhance with AI" step sends the headers and sample rows to Ollama for a second opinion.
- **AI Refinement**: When AI enhancement is triggered, the returned mapping overwrites the dropdown selections and the changed fields flash to show what moved. Every mapping stays editable afterwards.
- **One-to-One Validation**: Both the rule-based and AI mappings are validated so that no canonical field claims a source column already taken by another.
- **Mapped Data Preview**: The processed result renders as a table so you can confirm the mapping before using the output.
- **Safe JSON Serialization**: Automated sanitization of pandas `NaN` / `NaT` values to prevent serialization crashes.

---

## Project Structure

```text
├── main.py               # FastAPI application & mapping engine
├── static/
│   └── index.html        # Single-page frontend (TailwindCSS via CDN)
├── tests/                # pytest suite (see Testing below)
├── testdata/             # Local datasets, git-ignored — see testdata/README.md
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Runtime + test dependencies
├── pytest.ini            # Test configuration & markers
├── .env.example          # Template for environment variables
├── .env                  # Your local config (git-ignored, create from the example)
└── .gitignore            # Git exclusion rules
```

---

## Requirements

- **Python 3.10+** (developed and tested against 3.13)
- An internet connection for the frontend — TailwindCSS and the Inter font load from CDNs.
- An Ollama API key, only if you want the AI enhancement step.

---

## Setup & Installation

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate the Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

The app runs without any configuration. To enable the AI enhancement step, copy the template and add your key:

```bash
cp .env.example .env
```

```env
OLLAMA_API_KEY=your_ollama_api_key_here
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_API_KEY` | _(unset)_ | Bearer token for the Ollama API. Without it, `/enhance/` returns an empty AI mapping. |
| `OLLAMA_HOST` | `https://ollama.com` | Defaults to Ollama Cloud. Point at `http://localhost:11434` to use a local Ollama instance. |
| `SERVER_HOST` | `127.0.0.1` | Address uvicorn binds to. |
| `SERVER_PORT` | `8080` | Port uvicorn binds to. |

---

## Running the Application

With the virtual environment active:

```bash
python main.py
```

The application will start on: **http://127.0.0.1:8080**

---

## Usage

1. Upload a `.csv`, `.xlsx`, or `.xls` file.
2. Click **Analyze Columns** — the rule-based engine pre-fills the mapping dropdowns.
3. *(Optional)* Click **Enhance with AI** to have the LLM revise the mapping using the headers and the first 10 rows of data.
4. Adjust any dropdown manually, then process the file to see the mapped preview.

---

## API Endpoints

- `GET /` — Serves the frontend column mapper interface.
- `POST /analyze/` — Inspects the uploaded file headers (first 10 rows) and returns the canonical column definitions, the file's columns, the rule-based mapping, and sample data.
- `POST /enhance/` — Sends headers plus sample rows to Ollama and returns both the `ai_mapping` and the `auto_mapping` for comparison.
- `POST /process/` — Reconstructs and returns the sanitized canonical data across the full file, based on the chosen column mappings.

---

## Testing

Install the test dependencies and run the suite:

```bash
pip install -r requirements-dev.txt
pytest
```

Tests drive the app through FastAPI's `TestClient`, so no server needs to be
running and no port is required. The Ollama client is stubbed throughout, so the
suite never reaches the network and needs no `OLLAMA_API_KEY`.

```bash
pytest -m "not realdata"  # unit + endpoint tests only (fast)
pytest -m realdata        # only the tests using the files in testdata/
```

Tests marked `realdata` exercise the datasets in [testdata/](testdata/). Those
files are git-ignored, and the tests skip cleanly when they are absent — see
[testdata/README.md](testdata/README.md) for download links.

---

## Troubleshooting

**AI enhancement returns nothing / logs `Illegal header value b'Bearer '`**
`OLLAMA_API_KEY` is unset. The app degrades gracefully — the rule-based mapping still works — but AI suggestions require a valid key in `.env`. Restart the server after adding it.

**Port 8080 already in use**
Set `SERVER_PORT` in your `.env` to a free port.
