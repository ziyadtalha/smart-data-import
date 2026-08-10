# Smart Column Mapper

An interactive web application that maps columns from uploaded files (CSV, XLSX, XLS) to a canonical set of data columns. The mapping system uses a hybrid approach: deterministic rule-based synonym matching, optionally refined by LLM recommendations via Ollama (`gpt-oss:120b`).

---

## Features

- **Multi-Format Ingestion**: Reads CSV, XLSX, XLS and SQLite databases through the same mapping flow.
- **SQLite Table Selection**: Lists every table and view in an uploaded database and maps one at a time. Left to itself it picks the table matching the most canonical fields, so a large lookup table never wins over the transaction table; a dropdown switches tables and remaps.
- **Grain-Safe Auto-Join**: Optionally widens the chosen table with columns from related tables, joining only where the row count provably cannot change. See below.
- **Three-Stage Mapping**: Synonym rules first, then a local embedding matcher, then — only if the user asks — an LLM. Each stage is more expensive than the last and only sees what the previous one could not resolve.
- **AI Refinement**: When AI enhancement is triggered, the returned mapping overwrites the dropdown selections and the changed fields flash to show what moved. Every mapping stays editable afterwards.
- **One-to-One Validation**: Both the rule-based and AI mappings are validated so that no canonical field claims a source column already taken by another.
- **Mapped Data Preview**: The processed result renders as a table so you can confirm the mapping before using the output.
- **Safe JSON Serialization**: Automated sanitization of pandas `NaN` / `NaT` values to prevent serialization crashes.

---

## Project Structure

```text
├── main.py               # FastAPI application & mapping engine
├── semantic.py           # Stage 2: local embedding matcher
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

### How a column gets mapped

| Stage | Mechanism | Cost | Marked in the UI as |
| --- | --- | --- | --- |
| 1 — Rules | Normalized header matched against canonical names and synonyms | Free | *(nothing — it is certain)* |
| 2 — Semantic | Cosine similarity against canonical field names, descriptions and synonyms, via a local embedding model | Free per request | **similar** |
| 3 — LLM | Headers plus 10 sample rows sent to Ollama, validated against the schema | Per-token, opt-in | **AI** |

Each stage only sees the headers and canonical fields the previous stage left
unresolved, and no stage overrides an earlier one. Stage 3 runs only when the
user clicks **Enhance with AI**.

Stage 2 runs `BAAI/bge-small-en-v1.5` locally through fastembed's ONNX build —
no network call, no token cost, nothing leaves the machine. The model is
downloaded once (~130 MB) on first use and lazily loaded, so startup and
rule-only mapping stay instant. If `fastembed` is not installed the stage is
skipped and stage 1 still works.

Four constraints keep it from guessing:

**Identifier headers are excluded.** Every identifier embeds close to every
other one — `product_id` scores 0.751 against `product_name`, and `seller_id`
scores 0.720 against `customer_id`, both higher than genuine matches elsewhere.
Headers ending in `id`, `code`, `number` and the like are left to stage 1,
which handles the ones that really are canonical.

**The threshold sits above the worst wrong match, with margin.** Calibrated
against the headers in `testdata/`, the closest wrong call that survives the
identifier guard is `order_status` → `order_date` at 0.650 — two
order-prefixed phrases the model cannot pull apart. The threshold of 0.68 costs
real recall: `Timestamp` (0.609) and `basket_size` (0.558) are correct matches
this stage declines rather than guesses at, and stage 3 picks them up.

**A header proposes only its single best field — no consolation prizes.** A
joined olist row carries both `product_category_name` and
`product_category_name_english`. Both are plainly categories, and assigning the
runner-up to whatever is still free put it on `product_name` at 0.780. If a
header's best field is already taken, the header is dropped instead.

**A match whose values contradict the declared type is rejected.** Header text
alone cannot separate `product_name` from `product_name_lenght`, which holds
the length of the name and scores 0.712 — that genuinely is what its name
resembles. The sampled values settle it: `product_name` is declared a `string`
and the column holds integers. Only clear contradictions count, and dates are
exempt because they arrive as strings as often as not.

**Several columns may want the same field, and only one gets it.** Wide tables
make near-duplicate columns common, so proposals are resolved by score and the
losers are dropped.

#### Worked example: a database upload

The embedding model never sees tables. Everything database-specific happens
first and flattens the upload into one wide set of column names; from there a
database is indistinguishable from a CSV header row.

```
olist.sqlite (11 tables)
  1. choose a table          -> order_items        rules only, no model
  2. join related tables     -> +5 tables          uniqueness + match rate, no model
  3. 30 column names         -> from here, identical to a CSV
  4. rules claim what they can -> order_id, price, customer_id
  5. the 27 leftovers go to the model
```

Of those 27, four are dropped as identifiers (`order_item_id`, `product_id`,
`seller_id`, `customer_unique_id`). The remaining 23 each name their best
canonical field, and the filters do the rest:

| Column | Best guess | Score | Outcome |
| --- | --- | --- | --- |
| `product_category_name` | `category` | 0.829 | **kept** |
| `product_category_name_english` | `category` | 0.791 | lost the field to the line above |
| `product_name_lenght` | `product_name` | 0.712 | rejected — integers cannot fill a `string` |
| `order_purchase_timestamp` | `order_date` | 0.709 | **kept** |
| `order_delivered_customer_date` | `order_date` | 0.691 | lost the field |
| `order_delivered_carrier_date` | `order_date` | 0.686 | lost the field |
| `order_status` | `order_date` | 0.650 | below threshold |
| `product_weight_g` | `quantity` | 0.597 | below threshold |

Five proposals cleared the filters, contesting two fields, so two survive:
30 columns become 3 rule matches and 2 semantic ones.

Note how thin the winning margin is on the date group — 0.018 over the
runner-up. It picked correctly, since purchase time really is the order date,
but that is not a margin to trust blindly. It is the reason semantic matches
are badged for review rather than applied silently.

Stage 2 applies to every upload, database included. It runs once, on the
columns finally selected — table scoring and join planning stay on the rules,
since they run the mapper once per candidate table and embedding there would
scale each request with the size of the schema.

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
pytest -m embedding       # only the tests that load the real embedding model
```

Stage 2's tests stub the encoder with hand-written similarities, so the bulk of
the suite runs offline and without `fastembed`. The `embedding` marker covers
the handful that load the real model, including the traps it must refuse.

Tests marked `realdata` exercise the datasets in [testdata/](testdata/). Those
files are git-ignored, and the tests skip cleanly when they are absent — see
[testdata/README.md](testdata/README.md) for download links.

---

## Known limitations

Things that work as designed but are worth knowing before relying on them.

**Nothing is remembered between uploads.** Every upload re-runs all three
stages from scratch. The same POS export mapped last week pays the same cost
again, and a mapping the user corrected by hand is discarded when the page
resets. A hash of the header row, mapped to a confirmed mapping, would make
repeat uploads instant and let corrections accumulate.

**The whole file is re-sent on every call.** `sqlite3` needs a real path, so
each request spools the upload to a temp file and deletes it afterwards. A
113 MB database analysed, enhanced and processed moves ~450 MB in total. An
upload token that keeps the spooled file for the session would remove that.

**`/process/` returns the entire dataset in one JSON response** — 112,650 rows
for olist, taking ~17s with auto-join. There is no pagination or streaming.

**`/process/` trusts the mapping it is given.** `validate_one_to_one_mapping`
guards the rule, semantic and LLM paths, but not the user-supplied mapping, so
a hand-crafted request can point two canonical fields at one column. The UI
cannot produce that state.

**Excel reads the first sheet only.** A multi-sheet workbook is the same
problem as a multi-table database, and could reuse the table picker.

**Canonical descriptions were written for humans, not embeddings.** Stage 2
compares headers against them directly, which makes them the highest-leverage
knob in the system — `order_date` reading "Date the order was placed or
processed" attracts every date column in a schema, and `quantity` attracts
every numeric measurement (`product_weight_g` scores 0.597 against it).
Rewriting them to be discriminative rather than merely accurate would raise
recall at no runtime cost.

**Auto-join guarantees correctness, not relevance.** A table can be grain-safe
and well-matched yet make no business sense to join. `joined_tables` is
reported so you can see what happened.

---

## Troubleshooting

**AI enhancement returns nothing / logs `Illegal header value b'Bearer '`**
`OLLAMA_API_KEY` is unset. The app degrades gracefully — the rule-based mapping still works — but AI suggestions require a valid key in `.env`. Restart the server after adding it.

**Port 8080 already in use**
Set `SERVER_PORT` in your `.env` to a free port.
