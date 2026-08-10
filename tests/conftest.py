"""Shared fixtures.

Tests drive the app through FastAPI's TestClient rather than a live server, so
they need no running process and no free port.
"""
import io
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402

TESTDATA = ROOT / "testdata"


@pytest.fixture(scope="session", autouse=True)
def repo_root_cwd():
    """`GET /` serves static/index.html by relative path, so the CWD must be the repo root."""
    previous = os.getcwd()
    os.chdir(ROOT)
    yield ROOT
    os.chdir(previous)


@pytest.fixture(scope="session")
def client(repo_root_cwd):
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sales_frame():
    """25 rows with two deliberate blanks, to exercise NaN sanitization."""
    return pd.DataFrame([
        {
            "Order ID": 1000 + i,
            "Sale Date": f"2026-01-{(i % 28) + 1:02d}",
            "Client": f"Customer {i}",
            "Email Address": "" if i == 3 else f"user{i}@example.com",
            "Mobile": "" if i == 5 else f"555-{i:04d}",
            "Department": "Electronics" if i % 2 else "Office",
            "Item": f"Product {i}",
            "Qty": i % 5 + 1,
            "Price": round(10 + i * 1.25, 2),
            "Net Sales": round((i % 5 + 1) * (10 + i * 1.25), 2),
        }
        for i in range(25)
    ])


@pytest.fixture(scope="session")
def sales_csv(sales_frame):
    return sales_frame.to_csv(index=False).encode()


@pytest.fixture(scope="session")
def sales_xlsx(sales_frame):
    buffer = io.BytesIO()
    sales_frame.to_excel(buffer, index=False)
    return buffer.getvalue()


@pytest.fixture
def analyze(client):
    def _analyze(data, name="data.csv"):
        return client.post("/analyze/", files={"file": (name, data, "application/octet-stream")})
    return _analyze


@pytest.fixture
def process(client):
    def _process(data, mappings_json, name="data.csv"):
        return client.post(
            "/process/",
            files={"file": (name, data, "application/octet-stream")},
            data={"mappings_json": mappings_json},
        )
    return _process


@pytest.fixture
def enhance(client):
    def _enhance(data, name="data.csv"):
        return client.post("/enhance/", files={"file": (name, data, "application/octet-stream")})
    return _enhance


@pytest.fixture
def fake_ollama(monkeypatch):
    """Replace ollama.Client so tests never reach the network.

    `install(text)` makes the model stream back `text`; `install(error=...)`
    makes the call blow up so the graceful-degradation path can be checked.
    """
    def install(text="", *, error=None, chunks=3):
        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def chat(self, model, messages=None, stream=False):
                if error is not None:
                    raise error
                step = max(1, -(-len(text) // chunks))
                for start in range(0, len(text), step):
                    yield {"message": {"content": text[start:start + step]}}

        monkeypatch.setattr(main.ollama, "Client", FakeClient)

    return install
