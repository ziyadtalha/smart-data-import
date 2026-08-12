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
├── tools/
│   └── tune_semantic.py  # Calibration harness for stage 2's threshold
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
other one — on olist, `seller_id` scores 0.670 against `customer_id` and
`product_id` 0.665, near enough to the threshold to land a foreign key on the
wrong field. Headers ending in `id`, `code`, `number` and the like are left to
stage 1, which handles the ones that really are canonical.

**The threshold sits above the worst wrong match, with margin.** Calibrated by
sweeping it over 103 hand-labelled headers drawn from every table in
`testdata/` and taking the widest band that admits no wrong match at all. That
band is `[0.665, 0.679)`, bounded below by the hardest wrong call —
`return_date` → `order_date` at 0.660, in a rental table whose real order date
is `rental_date` — and above by the weakest correct one, `payment_value` →
`total_price` at 0.679. The threshold of 0.67 sits in the middle. It still
costs recall: `rental_date` (0.643), `Timestamp` (0.640) and `begin_date_time`
(0.639) are correct matches this stage declines rather than guesses at, because
`return_date` and `end_date_time` sit right on top of them. Stage 3 picks those
up.

**A header proposes only its single best field — no consolation prizes.** A
joined olist row carries both `product_category_name` and
`product_category_name_english`. Both are plainly categories, and assigning the
runner-up to whatever is still free put it on `product_name` at 0.741. If a
header's best field is already taken, the header is dropped instead.

**A match whose values contradict the declared type is rejected.** Header text
alone cannot separate `product_name` from `product_name_lenght`, which holds
the length of the name and scores 0.689 — that genuinely is what its name
resembles. The sampled values settle it: `product_name` is declared a `string`
and the column holds integers. Only clear contradictions count, and dates are
exempt because they arrive as strings as often as not.

#### The canonical descriptions are part of the matcher

Stage 2 embeds each field's name, description and synonyms together and
compares headers against the result, so the descriptions are not documentation
— their wording decides what matches. Two habits keep them working:

- **Be discriminative, not merely accurate.** `order_date` once read "Date the
  order was placed or processed". Perfectly true, and it attracted every
  `*_date` column in the schema. It now names the moment of purchase and gives
  an example value.
- **Never write what a field is not.** Embeddings have no negation: "not the
  delivery date" puts *delivery date* into the vector and pulls those columns
  closer.

Rewriting the descriptions on those two rules, with no code change, took
stage 2 from 5 correct matches to 9 on the labelled set — and lowered the
threshold at the same time, because the wrong matches fell further than the
right ones. `basket_size` → `quantity` went from 0.558 to 0.697 when
`quantity`'s description stopped reading "Quantity of units purchased" (which
attracted every numeric measurement) and started reading "How many items went
into the basket". Reword them only alongside a re-run of that calibration.

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
  4. rules claim what they can -> order_id, product_id, price, customer_id
  5. the 23 leftovers go to the model
```

Six of the 30 are dropped as identifiers (`order_id`, `order_item_id`,
`product_id`, `seller_id`, `customer_id`, `customer_unique_id`) and one more
was claimed by a rule. The remaining 23 each name their best canonical field,
and the filters do the rest:

| Column | Best guess | Score | Outcome |
| --- | --- | --- | --- |
| `product_category_name` | `category` | 0.821 | **kept** |
| `product_category_name_english` | `category` | 0.789 | lost the field to the line above |
| `order_purchase_timestamp` | `order_date` | 0.754 | **kept** |
| `order_delivered_customer_date` | `order_date` | 0.708 | lost the field |
| `product_name_lenght` | `product_name` | 0.689 | rejected — integers cannot fill a `string` |
| `order_delivered_carrier_date` | `order_date` | 0.687 | lost the field |
| `order_estimated_delivery_date` | `order_date` | 0.687 | lost the field |
| `customer_state` | `customer_id` | 0.670 | lost the field — a rule already had it |
| `customer_zip_code_prefix` | `customer_phone` | 0.652 | below threshold |
| `order_status` | `order_date` | 0.647 | below threshold |
| `customer_city` | `customer_phone` | 0.636 | below threshold |
| `shipping_limit_date` | `order_date` | 0.634 | below threshold |
| `product_weight_g` | `quantity` | 0.587 | below threshold |

Six proposals cleared the threshold, contesting two fields, so two survive:
30 columns become 4 rule matches and 2 semantic ones.

The date group is the one to watch. `order_purchase_timestamp` beat
`order_delivered_customer_date` by 0.046, and it picked correctly — purchase
time really is the order date — but four `*_date` columns were bunched inside
0.07 of each other. That is not a margin to trust blindly, and it is why
semantic matches are badged for review rather than applied silently.

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

### Recalibrating stage 2

The tests pin the behaviour that matters, but the threshold and the canonical
descriptions were chosen by measurement, and that measurement is reproducible:

```bash
python tools/tune_semantic.py            # sweep the threshold
python tools/tune_semantic.py --detail 0.67
```

It runs rules-then-embeddings over 103 hand-labelled headers from every table
in `testdata/` and prints, for each cut, how many correct matches survive and
what the worst wrong match is. The number to maximise is the correct-match
count at the highest cut that still admits zero wrong ones. Run it after
touching a description, a synonym list, the threshold, **or the set of
canonical fields**; it needs the datasets and `fastembed`, and reports on
whatever subset of `testdata/` is present.

Adding or removing a field is easy to overlook as a calibration change, but
every header is scored against the whole set, so a new field changes what
unrelated headers are compared to. When `product_id` was added the sweep came
back byte-identical — same clean band `[0.665, 0.680)`, same 9 correct matches,
same worst wrong match — because no header's best field became `product_id`.
That is the outcome to check for, not to assume.

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

**Stage 2 cannot separate a sale from its neighbouring event.** `return_date`
outscores `rental_date`, and `end_date_time` outscores `begin_date_time`, so
the matcher declines both rather than take the wrong one. Descriptions cannot
fix this — the field *name* `order date` is equidistant from both — and it is
the single thing capping the threshold. Stage 3 resolves these.

**The threshold is calibrated against 103 headers from five datasets.** That is
enough to be honest about the trade-off, not enough to be a benchmark. A schema
unlike anything in `testdata/` may sit differently against it.

**Auto-join guarantees correctness, not relevance.** A table can be grain-safe
and well-matched yet make no business sense to join. `joined_tables` is
reported so you can see what happened.

---

## Troubleshooting

**AI enhancement returns nothing / logs `Illegal header value b'Bearer '`**
`OLLAMA_API_KEY` is unset. The app degrades gracefully — the rule-based mapping still works — but AI suggestions require a valid key in `.env`. Restart the server after adding it.

**Port 8080 already in use**
Set `SERVER_PORT` in your `.env` to a free port.
