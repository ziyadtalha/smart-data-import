# Smart Column Mapper

An interactive, high-performance web application designed to map columns from uploaded files (CSV, XLSX, XLS) to a canonical set of data columns. The mapping system uses a hybrid approach: deterministic rule-based pattern matching combined with LLM-powered AI recommendations via Ollama (`gpt-oss:120b`).

---

## Features

- **Hybrid Mapping Engine**: Combines fuzzy rule-based matchers with Ollama LLM suggestions.
- **Side-by-Side Comparison**: Displays AI recommendations alongside rule-based suggestions.
- **Glassmorphic Web Interface**: Modern, responsive, and responsive preview table design built using TailwindCSS and the Inter font.
- **Safe JSON Serialization**: Automated sanitization of pandas `NaN` / `NaT` values to prevent serialization crashes.

---

## Project Structure

```text
├── main.py          # FastAPI application & single-page frontend
├── test_data.csv    # Sample data file for testing
├── .env             # Environment variables (API keys)
└── .gitignore       # Git exclusion rules
```

---

## Setup & Installation

Follow these steps to set up the project locally:

### 1. Create a Virtual Environment
Initialize a clean Python environment:
```bash
python3 -m venv .venv
```

### 2. Activate the Environment
Activate the environment in your shell:
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries analyzed from the workspace environment:
```bash
pip install fastapi uvicorn pandas python-multipart openpyxl ollama
```

### 4. Configuration
Create a `.env` file in the project root directory to configure the Ollama API:
```env
OLLAMA_API_KEY=your_ollama_api_key_here
```

---

## Running the Application

Ensure your virtual environment is active, then run:

```bash
python main.py
```

The application will start on: **http://localhost:8000**

---

## API Endpoints

- `GET /` — Serves the frontend column mapper interface.
- `POST /analyze/` — Inspects the uploaded file headers and returns both AI & rule-based mapping recommendations.
- `POST /process/` — Reconstructs and returns the sanitized canonical data based on the chosen column mappings.
