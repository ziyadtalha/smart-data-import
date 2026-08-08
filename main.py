import io
import json
import os
import ollama
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

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

CANONICAL_COLUMNS = ["first_name", "last_name", "email", "phone_number"]

COLUMN_MAPPING = {
    "fname": "first_name",
    "given_name": "first_name",
    "first": "first_name",
    "lname": "last_name",
    "surname": "last_name",
    "last": "last_name",
    "email_address": "email",
    "mail": "email",
    "phone": "phone_number",
    "mobile": "phone_number",
    "tel": "phone_number",
}


def normalize_string(val: str) -> str:
    return str(val).strip().lower().replace(" ", "_").replace("-", "_")


def auto_map_columns(file_columns: list[str]) -> dict[str, str]:
    """Generates initial mapping dictionary from file columns to canonical columns."""
    mapping = {}
    for col in file_columns:
        norm = normalize_string(col)
        if norm in CANONICAL_COLUMNS:
            mapping[norm] = col
        elif norm in COLUMN_MAPPING:
            canonical = COLUMN_MAPPING[norm]
            if canonical not in mapping:
                mapping[canonical] = col
    return mapping


def prompt_for_mapping(
    file_columns: list[str],
    model: str = "gpt-oss:120b",
) -> dict[str, str]:
    prompt = f"""You are an AI assistant that maps file columns to canonical columns.
    Canonical columns: {CANONICAL_COLUMNS}
    File columns: {file_columns}

    You should map the file columns to the canonical columns.
    Return the mapping as a JSON object, where the keys are the canonical columns and the values are the corresponding file columns.
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
            cleaned_mapping = {}
            for k, v in mapping.items():
                if k in CANONICAL_COLUMNS and v in file_columns:
                    cleaned_mapping[k] = v
            return cleaned_mapping
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
            df = pd.read_csv(io.BytesIO(contents), nrows=1)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents), nrows=1)
        else:
            raise HTTPException(
                status_code=400, detail="Please upload a .csv or .xlsx file."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to read file: {str(e)}"
        )

    file_columns = list(df.columns)
    ai_suggested = prompt_for_mapping(file_columns)
    auto_suggested = auto_map_columns(file_columns)

    # Merge suggestions, prioritizing AI suggestions
    suggested = {}
    for canonical in CANONICAL_COLUMNS:
        if canonical in ai_suggested and ai_suggested[canonical]:
            suggested[canonical] = ai_suggested[canonical]
        elif canonical in auto_suggested and auto_suggested[canonical]:
            suggested[canonical] = auto_suggested[canonical]

    return {
        "canonical_columns": CANONICAL_COLUMNS,
        "file_columns": file_columns,
        "suggested_mapping": suggested,
        "ai_mapping": ai_suggested,
        "auto_mapping": auto_suggested,
    }


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
    for canonical in CANONICAL_COLUMNS:
        source_col = user_mappings.get(canonical)
        if source_col and source_col in df.columns:
            result_df[canonical] = df[source_col]
        else:
            result_df[canonical] = None

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