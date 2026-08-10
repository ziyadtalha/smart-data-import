import io
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
import ollama
import pandas as pd

import semantic
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

# Load environment variables from .env file if present
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")

app = FastAPI()

# The descriptions are not just documentation: stage 2 embeds them and compares
# column headers against the result, so their wording decides what matches. Two
# habits keep them working, both learned by measurement:
#
#   * Be discriminative, not merely accurate. "Date the order was placed"
#     is true, and it attracted every *_date column in the schema.
#   * Never write what a field is not. Embeddings have no negation — "not the
#     delivery date" puts "delivery date" into the vector and pulls it closer.
#
# Concrete example values do most of the work. Reword these only alongside a
# run of tools/tune_semantic.py, which measures what the change costs or buys.
CANONICAL_COLUMNS = [
    {
        "name": "order_id",
        "description": "Unique identifier for the order or transaction.",
        "synonyms": ["order id", "orderid", "id", "order_number"],
        "type": "string",
    },
    {
        "name": "order_date",
        "description": "The moment the purchase was made and the basket was paid for, for example 2023-04-17.",
        "synonyms": ["order_date", "orderdate", "date", "sale_date"],
        "type": "date",
    },
    {
        "name": "customer_id",
        "description": "Unique identifier for the customer or buyer account.",
        "synonyms": [
            "customer_id",
            "customerid",
            "client_id",
            "buyer_id",
            "customer_unique_id",
            "customer_number",
        ],
        "type": "string",
    },
    {
        "name": "customer_name",
        "description": "The person's name, first name and surname, such as Jane Alvarez.",
        "synonyms": ["customer", "customer_name", "client", "customerName"],
        "type": "string",
    },
    {
        "name": "customer_email",
        "description": "Where to email this person, such as jane.alvarez@example.com.",
        "synonyms": ["email", "email_address", "mail"],
        "type": "string",
    },
    {
        "name": "customer_phone",
        "description": "The digits dialled to ring this person, for example +1 555 0134.",
        "synonyms": ["phone", "mobile", "tel", "phone_number"],
        "type": "string",
    },
    {
        "name": "category",
        "description": "The product's group in the catalogue, such as beverages, toys or electronics.",
        "synonyms": ["category", "product_category", "department"],
        "type": "string",
    },
    {
        "name": "product_name",
        "description": "What the item is called on the receipt, such as blue cotton t-shirt.",
        "synonyms": ["product", "product_name", "item", "item_name"],
        "type": "string",
    },
    {
        "name": "quantity",
        "description": "How many items went into the basket, for example 1, 2 or 3.",
        "synonyms": ["qty", "quantity", "units"],
        "type": "integer",
    },
    {
        "name": "unit_price",
        "description": "The price of one item on its own, for example 4.90 each.",
        "synonyms": ["price", "unit_price", "price_per_item", "item_price"],
        "type": "decimal",
    },
    {
        "name": "total_price",
        "description": "Total monetary amount charged for the transaction.",
        # "amount" lives here rather than on unit_price: in point-of-sale
        # exports it is the basket total, not the per-unit figure.
        "synonyms": ["total", "amount", "total_amount", "amount_paid", "net_sales", "sale_total", "transaction_total"],
        "type": "decimal",
    },
]

CANONICAL_COLUMN_NAMES = [column["name"] for column in CANONICAL_COLUMNS]

def normalize_string(val: str) -> str:
    return str(val).strip().lower().replace(" ", "_").replace("-", "_")

CANONICAL_COLUMN_LOOKUP = {}
for column in CANONICAL_COLUMNS:
    CANONICAL_COLUMN_LOOKUP[column["name"]] = column["name"]
    for alias in column["synonyms"]:
        CANONICAL_COLUMN_LOOKUP[normalize_string(alias)] = column["name"]


def auto_map_columns(file_columns: list[str]) -> dict[str, str]:
    """Generates initial mapping dictionary from file columns to canonical columns."""
    mapping: dict[str, str] = {}
    used_columns: set[str] = set()
    for col in file_columns:
        norm = normalize_string(col)
        canonical_name = CANONICAL_COLUMN_LOOKUP.get(norm)
        if (
            canonical_name
            and canonical_name not in mapping
            and col not in used_columns
        ):
            mapping[canonical_name] = col
            used_columns.add(col)
    return mapping


def column_kinds(df: pd.DataFrame) -> dict[str, str]:
    """Classify each sampled column so stage 2 can reject impossible matches."""
    kinds: dict[str, str] = {}
    for name in df.columns:
        values = df[name].dropna()
        if values.empty:
            kinds[name] = "unknown"
        elif pd.api.types.is_numeric_dtype(values):
            kinds[name] = "number"
        elif pd.api.types.is_datetime64_any_dtype(values):
            kinds[name] = "datetime"
        else:
            kinds[name] = "text"
    return kinds


def suggest_mapping(
    file_columns: list[str],
    use_semantic: bool = True,
    sample: pd.DataFrame | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Run stage 2: exact and synonym rules, then embedding similarity.

    Returns the mapping plus, per canonical field, whether it came from the
    rules or from the semantic matcher — the caller surfaces that difference
    because a semantic match is a suggestion rather than a certainty.
    """
    mapping = auto_map_columns(file_columns)
    sources = {field: "rule" for field in mapping}

    if not use_semantic or not semantic.is_available():
        return mapping, sources

    try:
        matched = semantic.match(
            file_columns,
            CANONICAL_COLUMNS,
            taken_fields=set(mapping),
            taken_columns=set(mapping.values()),
            column_kinds=column_kinds(sample) if sample is not None else None,
        )
    except Exception as e:
        # A missing model download or offline host must not break the upload.
        print(f"Semantic matching skipped: {e}")
        return mapping, sources

    for field, (column, _score) in matched.items():
        mapping[field] = column
        sources[field] = "semantic"

    return mapping, sources


def validate_one_to_one_mapping(
    mapping: dict[str, str],
    file_columns: list[str],
) -> dict[str, str]:
    """Keep only valid one-to-one mappings: one canonical field to one source column."""
    cleaned: dict[str, str] = {}
    used_columns: set[str] = set()

    for canonical_name, source_col in mapping.items():
        if canonical_name not in CANONICAL_COLUMN_NAMES:
            continue
        if source_col not in file_columns:
            continue
        if source_col in used_columns:
            continue

        cleaned[canonical_name] = source_col
        used_columns.add(source_col)

    return cleaned


def prompt_for_mapping(
    file_columns: list[str],
    sample_data: list[dict] | None = None,
    model: str = "gpt-oss:120b",
    auto_mapping: dict[str, str] | None = None,
) -> dict[str, str]:
    sample_section = ""
    if sample_data:
        sample_section = f"\n    Here are the first few rows of actual data to help you understand what each column contains:\n    {json.dumps(sample_data, indent=2, default=str)}\n"

    auto_mapping_section = ""
    if auto_mapping:
        auto_mapping_section = (
            "\n    The rule-based auto mapping is:\n"
            f"    {json.dumps(auto_mapping, indent=2, sort_keys=True)}\n"
            "    Use it as the starting point, but do not assign the same file column to more than one canonical column. "
            "Each canonical column must map to at most one file column and each file column must map to at most one canonical column.\n"
        )

    prompt = f"""You are an AI assistant that maps file columns to canonical columns.
    Canonical columns definitions:
    {json.dumps(CANONICAL_COLUMNS, indent=2)}
    File columns: {file_columns}
    {sample_section}
    {auto_mapping_section}
    You should map the file columns to the canonical columns based on exact names, common synonyms, and the actual data values.
    Return the mapping as a JSON object, where the keys are the canonical column names and the values are the corresponding file columns.
    Use one-to-one mapping only: one canonical column can map to one file column, and one file column cannot map to multiple canonical columns.
    If a canonical column has no matching file column, omit it or set it to null.

    Example output: 
    {{
        "first_name": "fname",
        "last_name": "lname",
        "email": "email_address",
        "phone_number": "phone"
    }}

    Respond ONLY with the JSON object. Do not include any explanation or markdown formatting."""

    messages = [
        {
            'role': 'user',
            'content': prompt,
        },
    ]

    try:
        client = ollama.Client(
            host=os.getenv('OLLAMA_HOST', 'https://ollama.com'),
            headers={'Authorization': 'Bearer ' + (os.environ.get('OLLAMA_API_KEY') or '')}
        )
        
        full_response = ""
        for part in client.chat(model, messages=messages, stream=True):
            content = part.get('message', {}).get('content', '')
            full_response += content
            print(content, end='', flush=True)
        print()
        
        # Clean markdown code block wraps from response if present
        cleaned = full_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
            
        mapping = json.loads(cleaned)
        if isinstance(mapping, dict):
            return validate_one_to_one_mapping(mapping, file_columns)
    except Exception as e:
        print(f"LLM mapping failed or was skipped: {e}")

    return {}


CSV_EXTENSIONS = (".csv",)
EXCEL_EXTENSIONS = (".xlsx", ".xls")
SQLITE_EXTENSIONS = (".sqlite", ".sqlite3", ".db")
UNSUPPORTED_FILE_MESSAGE = (
    "Please upload a .csv, .xlsx, .xls, .sqlite, .sqlite3 or .db file."
)

# Ceiling on how far an auto-join will expand, base table included.
MAX_AUTO_JOIN_TABLES = 8

# A real relationship matches most rows. A column name two tables happen to
# share matches almost none, so anything below this is treated as coincidence.
MIN_JOIN_MATCH_RATE = 0.5
JOIN_SAMPLE_ROWS = 2000


@dataclass
class LoadResult:
    """A parsed upload plus whatever the source format knows about itself."""
    frame: pd.DataFrame
    tables: list[str] = field(default_factory=list)
    selected_table: str | None = None
    joined_tables: list[str] = field(default_factory=list)


def quote_identifier(name: str) -> str:
    """Quote a SQLite table name for interpolation into a query."""
    return '"' + name.replace('"', '""') + '"'


def list_sqlite_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [row[0] for row in rows]


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [
        row[1]
        for row in connection.execute(f"PRAGMA table_info({quote_identifier(table)})")
    ]


def column_is_unique(connection: sqlite3.Connection, table: str, column: str) -> bool:
    """True when every non-null value in the column appears exactly once.

    Phrased as a search for one duplicate rather than COUNT(DISTINCT) so that
    a non-unique column stops at its first repeat instead of sorting the whole
    table. Call it after indexing the column and both paths stay cheap.
    """
    quoted = quote_identifier(column)
    quoted_table = quote_identifier(table)
    try:
        has_rows = connection.execute(
            f"SELECT EXISTS(SELECT 1 FROM {quoted_table} WHERE {quoted} IS NOT NULL)"
        ).fetchone()[0]
        if not has_rows:
            return False
        has_duplicate = connection.execute(
            f"SELECT EXISTS(SELECT 1 FROM {quoted_table} "
            f"WHERE {quoted} IS NOT NULL "
            f"GROUP BY {quoted} HAVING COUNT(*) > 1)"
        ).fetchone()[0]
    except sqlite3.Error:
        return False
    return not has_duplicate


def create_join_index(
    connection: sqlite3.Connection,
    table: str,
    column: str,
) -> None:
    """Index a join key on the temp copy of the upload.

    Uploads arrive without indexes, and LIMIT does not rescue an unindexed
    join — SQLite still scans to find the first rows. Indexing costs well
    under a second and turns a multi-second join into a few milliseconds.
    """
    index_name = quote_identifier(f"smart_map_ix_{table}_{column}")
    try:
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {quote_identifier(table)}({quote_identifier(column)})"
        )
    except sqlite3.Error:
        # An unindexable target (a view, say) just stays slow.
        pass


def join_match_rate(
    connection: sqlite3.Connection,
    source: str,
    target: str,
    key: str,
) -> float:
    """Fraction of sampled source rows that find a partner in the target."""
    quoted_key = quote_identifier(key)
    try:
        total, matched = connection.execute(
            f"SELECT COUNT(*), SUM(EXISTS("
            f"  SELECT 1 FROM {quote_identifier(target)} t "
            f"  WHERE t.{quoted_key} = s.{quoted_key}"
            f")) FROM ("
            f"  SELECT {quoted_key} FROM {quote_identifier(source)} "
            f"  LIMIT {JOIN_SAMPLE_ROWS}"
            f") s"
        ).fetchone()
    except sqlite3.Error:
        return 0.0
    return (matched or 0) / total if total else 0.0


def plan_auto_join(
    connection: sqlite3.Connection,
    base_table: str,
    table_names: list[str],
) -> list[tuple[str, str, str]]:
    """Plan LEFT JOINs that cannot change the row count.

    Two conditions must both hold. The key must be unique in the target, which
    makes the relationship many-to-one so each source row matches at most one
    target row and the grain of the base table survives — joining
    order_payments onto order_items adds only 4% more rows but inflates summed
    revenue by 5%, which looks plausible and is wrong. The key must also match
    most rows, because uniqueness alone cannot tell a real relationship from a
    column name two unrelated tables happen to share: olist's leads_closed
    shares seller_id with order_items but matches only 4.5% of it.

    Returns (source_table, target_table, join_key) in join order.
    """
    columns = {name: table_columns(connection, name) for name in table_names}
    reached = [base_table]
    steps: list[tuple[str, str, str]] = []

    while len(reached) < MAX_AUTO_JOIN_TABLES:
        step = None
        for source in reached:
            for target in table_names:
                if target in reached:
                    continue
                for key in columns[source]:
                    if key not in columns[target]:
                        continue
                    # Index first: both checks below read through it.
                    create_join_index(connection, target, key)
                    if not column_is_unique(connection, target, key):
                        continue
                    if join_match_rate(connection, source, target, key) < (
                        MIN_JOIN_MATCH_RATE
                    ):
                        continue
                    step = (source, target, key)
                    break
                if step:
                    break
            if step:
                break

        if step is None:
            break

        steps.append(step)
        reached.append(step[1])

    return steps


def build_auto_join_query(
    connection: sqlite3.Connection,
    base_table: str,
    steps: list[tuple[str, str, str]],
) -> str:
    """Compose the SELECT for a planned join, keeping column names unique."""
    aliases = {base_table: "t0"}
    selected: list[str] = []
    taken: set[str] = set()

    for column in table_columns(connection, base_table):
        selected.append(f"t0.{quote_identifier(column)} AS {quote_identifier(column)}")
        taken.add(column)

    joins: list[str] = []
    for position, (source, target, key) in enumerate(steps, start=1):
        alias = f"t{position}"
        aliases[target] = alias
        joins.append(
            f"LEFT JOIN {quote_identifier(target)} {alias} "
            f"ON {alias}.{quote_identifier(key)} = "
            f"{aliases[source]}.{quote_identifier(key)}"
        )

        for column in table_columns(connection, target):
            if column == key:
                continue
            # Qualify a name that the base table or an earlier join already used.
            name = column if column not in taken else f"{target}_{column}"
            if name in taken:
                continue
            selected.append(
                f"{alias}.{quote_identifier(column)} AS {quote_identifier(name)}"
            )
            taken.add(name)

    return (
        f"SELECT {', '.join(selected)} "
        f"FROM {quote_identifier(base_table)} t0 " + " ".join(joins)
    )


def choose_sqlite_table(
    connection: sqlite3.Connection,
    table_names: list[str],
) -> str:
    """Pick the table whose columns map to the most canonical fields.

    Row count breaks ties, which favours transaction tables over the lookup
    tables that tend to share their column names.
    """
    best_name = table_names[0]
    best_score = (-1, -1)

    for name in table_names:
        columns = [
            row[1]
            for row in connection.execute(f"PRAGMA table_info({quote_identifier(name)})")
        ]
        try:
            row_count = connection.execute(
                f"SELECT COUNT(*) FROM {quote_identifier(name)}"
            ).fetchone()[0]
        except sqlite3.Error:
            row_count = 0

        score = (len(auto_map_columns(columns)), row_count)
        if score > best_score:
            best_name, best_score = name, score

    return best_name


def read_sqlite(
    contents: bytes,
    table: str | None,
    limit: int | None,
    auto_join: bool = False,
) -> LoadResult:
    """Read one table out of an uploaded SQLite database.

    sqlite3 needs a real path, so the upload is spooled to a temp file that is
    removed once the connection closes. Because that copy is disposable, the
    auto-join is free to index it.
    """
    handle, temp_path = tempfile.mkstemp(suffix=".sqlite")
    try:
        with os.fdopen(handle, "wb") as temp_file:
            temp_file.write(contents)

        connection = sqlite3.connect(temp_path)
        try:
            table_names = list_sqlite_tables(connection)
            if not table_names:
                raise HTTPException(
                    status_code=400, detail="This database contains no tables."
                )
            if table and table not in table_names:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Table '{table}' not found. "
                        f"Available tables: {', '.join(table_names)}."
                    ),
                )

            selected = table or choose_sqlite_table(connection, table_names)
            joined_tables: list[str] = []

            if auto_join:
                steps = plan_auto_join(connection, selected, table_names)
                query = build_auto_join_query(connection, selected, steps)
                joined_tables = [target for _, target, _ in steps]
            else:
                query = f"SELECT * FROM {quote_identifier(selected)}"

            if limit:
                query += f" LIMIT {int(limit)}"
            frame = pd.read_sql_query(query, connection)
        finally:
            connection.close()
    finally:
        os.unlink(temp_path)

    return LoadResult(frame, table_names, selected, joined_tables)


def load_dataframe(
    filename: str | None,
    contents: bytes,
    *,
    limit: int | None = None,
    table: str | None = None,
    auto_join: bool = False,
) -> LoadResult:
    """Read an upload into a DataFrame.

    Database uploads also report their tables, the table that was read and any
    tables folded in by the auto-join. Those stay empty for flat formats.
    """
    name = (filename or "").lower()

    try:
        if name.endswith(SQLITE_EXTENSIONS):
            return read_sqlite(contents, table, limit, auto_join)
        if name.endswith(CSV_EXTENSIONS):
            return LoadResult(pd.read_csv(io.BytesIO(contents), nrows=limit))
        if name.endswith(EXCEL_EXTENSIONS):
            return LoadResult(pd.read_excel(io.BytesIO(contents), nrows=limit))
        raise HTTPException(status_code=400, detail=UNSUPPORTED_FILE_MESSAGE)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")


@app.get("/")
async def get_frontend():
    return FileResponse("static/index.html")

@app.post("/analyze/")
async def analyze_file(
    file: UploadFile = File(...),
    table: str | None = Form(None),
    auto_join: bool = Form(False),
):
    contents = await file.read()
    loaded = load_dataframe(
        file.filename, contents, limit=10, table=table, auto_join=auto_join
    )
    df = loaded.frame

    file_columns = list(df.columns)
    auto_suggested, mapping_sources = suggest_mapping(file_columns, sample=df)

    # Include sample data for preview (convert NaN to None for JSON)
    sample_df = df.astype(object).where(pd.notnull(df), None)

    return {
        "canonical_columns": CANONICAL_COLUMNS,
        "file_columns": file_columns,
        "suggested_mapping": auto_suggested,
        "mapping_sources": mapping_sources,
        "sample_data": sample_df.to_dict(orient="records"),
        "tables": loaded.tables,
        "selected_table": loaded.selected_table,
        "joined_tables": loaded.joined_tables,
    }

@app.post("/enhance/")
async def enhance_mapping(
    file: UploadFile = File(...),
    table: str | None = Form(None),
    auto_join: bool = Form(False),
):
    contents = await file.read()
    loaded = load_dataframe(
        file.filename, contents, limit=10, table=table, auto_join=auto_join
    )
    df = loaded.frame

    file_columns = list(df.columns)
    sample_data = df.to_dict(orient='records')
    auto_mapping, mapping_sources = suggest_mapping(file_columns, sample=df)
    ai_mapping = prompt_for_mapping(
        file_columns,
        sample_data=sample_data,
        auto_mapping=auto_mapping,
    )
    return {
        "ai_mapping": ai_mapping,
        "auto_mapping": auto_mapping,
        "mapping_sources": mapping_sources,
        "selected_table": loaded.selected_table,
        "joined_tables": loaded.joined_tables,
    }

@app.post("/process/")
async def process_file(
    file: UploadFile = File(...),
    mappings_json: str = Form(...),
    table: str | None = Form(None),
    auto_join: bool = Form(False),
):
    contents = await file.read()

    try:
        user_mappings = json.loads(mappings_json)
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid mappings configuration."
        )

    loaded = load_dataframe(
        file.filename, contents, table=table, auto_join=auto_join
    )
    df = loaded.frame

    # Reconstruct data matching user choice: {canonical_col: file_col}
    result_df = pd.DataFrame(index=df.index)
    for field in CANONICAL_COLUMNS:
        canonical_name = field["name"]
        source_col = user_mappings.get(canonical_name)
        if source_col and source_col in df.columns:
            result_df[canonical_name] = df[source_col]
        else:
            result_df[canonical_name] = None

    result_df = result_df.astype(object).where(pd.notnull(result_df), None)

    return {
        "mapped_columns_count": len(user_mappings),
        "data": result_df.to_dict(orient="records"),
        "selected_table": loaded.selected_table,
        "joined_tables": loaded.joined_tables,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv('SERVER_HOST', '127.0.0.1'),
        port=int(os.getenv('SERVER_PORT', '8080')),
    )