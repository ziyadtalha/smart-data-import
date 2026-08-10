import io
import json
import os
import ollama
import pandas as pd
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

CANONICAL_COLUMNS = [
    {
        "name": "order_id",
        "description": "Unique identifier for the order or transaction.",
        "synonyms": ["order id", "orderid", "id", "order_number"],
        "type": "string",
    },
    {
        "name": "order_date",
        "description": "Date the order was placed or processed.",
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
        "description": "Name of the customer or buyer.",
        "synonyms": ["customer", "customer_name", "client", "customerName"],
        "type": "string",
    },
    {
        "name": "customer_email",
        "description": "Customer email address.",
        "synonyms": ["email", "email_address", "mail"],
        "type": "string",
    },
    {
        "name": "customer_phone",
        "description": "Customer phone number.",
        "synonyms": ["phone", "mobile", "tel", "phone_number"],
        "type": "string",
    },
    {
        "name": "category",
        "description": "Product category or department classification.",
        "synonyms": ["category", "product_category", "department"],
        "type": "string",
    },
    {
        "name": "product_name",
        "description": "Name of the sold product or item.",
        "synonyms": ["product", "product_name", "item", "item_name"],
        "type": "string",
    },
    {
        "name": "quantity",
        "description": "Quantity of units purchased.",
        "synonyms": ["qty", "quantity", "units"],
        "type": "integer",
    },
    {
        "name": "unit_price",
        "description": "Price per unit before total calculation.",
        "synonyms": ["amount", "price", "unit_price"],
        "type": "decimal",
    },
    {
        "name": "total_price",
        "description": "Total monetary amount charged for the transaction.",
        "synonyms": ["total", "total_amount", "amount_paid", "net_sales", "sale_total", "transaction_total"],
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


@app.get("/")
async def get_frontend():
    return FileResponse("static/index.html")

@app.post("/analyze/")
async def analyze_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    contents = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents), nrows=10)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents), nrows=10)
        else:
            raise HTTPException(
                status_code=400, detail="Please upload a .csv or .xlsx file."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to read file: {str(e)}"
        )

    file_columns = list(df.columns)
    auto_suggested = auto_map_columns(file_columns)

    # Include sample data for preview (convert NaN to None for JSON)
    sample_df = df.astype(object).where(pd.notnull(df), None)

    return {
        "canonical_columns": CANONICAL_COLUMNS,
        "file_columns": file_columns,
        "suggested_mapping": auto_suggested,
        "sample_data": sample_df.to_dict(orient="records"),
    }

@app.post("/enhance/")
async def enhance_mapping(file: UploadFile = File(...)):
    filename = file.filename.lower()
    contents = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents), nrows=10)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents), nrows=10)
        else:
            raise HTTPException(
                status_code=400, detail="Please upload a .csv or .xlsx file."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to read file: {str(e)}"
        )

    file_columns = list(df.columns)
    sample_data = df.head(10).to_dict(orient='records')
    auto_mapping = auto_map_columns(file_columns)
    ai_mapping = prompt_for_mapping(
        file_columns,
        sample_data=sample_data,
        auto_mapping=auto_mapping,
    )
    return {"ai_mapping": ai_mapping, "auto_mapping": auto_mapping}

@app.post("/process/")
async def process_file(
    file: UploadFile = File(...), mappings_json: str = Form(...)
):
    import json

    filename = file.filename.lower()
    contents = await file.read()

    try:
        user_mappings = json.loads(mappings_json)
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid mappings configuration."
        )

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400, detail="Please upload a .csv or .xlsx file."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to read file: {str(e)}"
        )

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
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv('SERVER_HOST', '127.0.0.1'),
        port=int(os.getenv('SERVER_PORT', '8080')),
    )