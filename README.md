# Smart Column Mapper

An interactive web application that maps columns from uploaded files (CSV, XLSX, XLS) to a canonical set of data columns. The mapping system uses a hybrid approach: deterministic rule-based synonym matching, optionally refined by LLM recommendations via Ollama (`gpt-oss:120b`).

---

## Features

- **Multi-Format Ingestion**: Reads CSV, XLSX, XLS and SQLite databases through the same mapping flow.
- **SQLite Table Selection**: Lists every table and view in an uploaded database and maps one at a time. Left to itself it picks the table matching the most canonical fields, so a large lookup table never wins over the transaction table; a dropdown switches tables and remaps.
- **Grain-Safe Auto-Join**: Optionally widens the chosen table with columns from related tables, joining only where the row count provably cannot change. See below.
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

1. Upload a `.csv`, `.xlsx`, `.xls`, `.sqlite`, `.sqlite3` or `.db` file.
2. Click **Analyze Columns** — the rule-based engine pre-fills the mapping dropdowns.
3. *For databases*, a **Database table** dropdown appears when the file holds more than one table. Switching tables re-analyzes against that table.
4. *(Optional)* Click **Enhance with AI** to have the LLM revise the mapping using the headers and the first 10 rows of data.
5. Adjust any dropdown manually, then process the file to see the mapped preview.

---

## API Endpoints

- `GET /` — Serves the frontend column mapper interface.
- `POST /analyze/` — Inspects the uploaded file headers (first 10 rows) and returns the canonical column definitions, the file's columns, the rule-based mapping, sample data, and — for databases — `tables` and `selected_table`.
- `POST /enhance/` — Sends headers plus sample rows to Ollama and returns both the `ai_mapping` and the `auto_mapping` for comparison.
- `POST /process/` — Reconstructs and returns the sanitized canonical data across the full file, based on the chosen column mappings.

All three accept two optional form fields for database uploads: `table`, naming
the table to read, and `auto_join`, widening it with related tables. Omit
`table` and the app chooses; name a table that isn't there and the response is a
400 listing the ones that are. Every response reports `selected_table` and
`joined_tables`. For non-database uploads `tables` and `joined_tables` come back
empty and `selected_table` is `null`.

### Auto-join

Pass `auto_join=true` (or tick **Pull in related tables**) and the chosen table
is widened with columns from tables that join onto it. A table is only folded in
when both of these hold:

1. **The join key is unique in the target.** That makes the relationship
   many-to-one, so each source row matches at most one target row and the row
   count provably cannot change.
2. **The key matches most sampled rows.** Uniqueness alone cannot distinguish a
   real relationship from a column name two unrelated tables happen to share.

Both conditions exist because a fan-out join is quietly destructive rather than
loudly broken. On olist, joining `order_payments` onto `order_items` adds only
4% more rows but reports **14,209,115.34** in revenue instead of the true
**13,591,643.70** — a plausible number that is wrong. The second condition
catches the opposite failure: `leads_closed` shares `seller_id` with
`order_items` and is grain-safe, but matches 4.5% of it and belongs to a
different business process entirely.

Uploading `olist.sqlite` with auto-join expands `order_items` from 7 columns to
30 by folding in `orders`, `products`, `sellers`, `customers` and
`product_category_name_translation`, leaving the row count at 112,650 and the
revenue total unchanged. `joined_tables` in the response names what was used.

The temp copy of the database is indexed on each join key first — uploads carry
no indexes, and `LIMIT` does not rescue an unindexed join. That one step takes
analysis of the olist join from ~19s to ~3s.

Its blind spot is relevance, not correctness: a table can be grain-safe and
well-matched yet still make no business sense to join. `joined_tables` is
reported so you can see what happened and switch the base table if it is wrong.

### Supported formats

| Extension | Read via | Notes |
| --- | --- | --- |
| `.csv` | `pandas.read_csv` | |
| `.xlsx`, `.xls` | `pandas.read_excel` | First sheet only |
| `.sqlite`, `.sqlite3`, `.db` | `sqlite3` + `pandas.read_sql_query` | One table or view per pass; spooled to a temp file since sqlite3 needs a real path |

Anything else is rejected with a 400 naming the formats that are accepted.

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
