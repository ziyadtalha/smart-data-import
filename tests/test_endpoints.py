"""Tests for the /, /analyze/ and /process/ endpoints."""
import json

import pytest

import main

CANONICAL_ORDER = main.CANONICAL_COLUMN_NAMES


def test_root_serves_the_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower()


# --- /analyze/ -------------------------------------------------------------

def test_analyze_reports_headers_and_mapping(analyze, sales_csv):
    body = analyze(sales_csv).json()
    assert body["file_columns"] == [
        "Order ID", "Sale Date", "Client", "Email Address", "Mobile",
        "Department", "Item", "Qty", "Price", "Net Sales",
    ]
    assert body["suggested_mapping"] == {
        "order_id": "Order ID",
        "order_date": "Sale Date",
        "customer_name": "Client",
        "customer_email": "Email Address",
        "customer_phone": "Mobile",
        "category": "Department",
        "product_name": "Item",
        "quantity": "Qty",
        "unit_price": "Price",
        "total_price": "Net Sales",
    }


def test_analyze_returns_canonical_definitions(analyze, sales_csv):
    body = analyze(sales_csv).json()
    assert [c["name"] for c in body["canonical_columns"]] == CANONICAL_ORDER


def test_analyze_caps_the_sample_at_ten_rows(analyze, sales_csv):
    body = analyze(sales_csv).json()
    assert len(body["sample_data"]) == 10


def test_analyze_converts_blanks_to_null(analyze, sales_csv):
    body = analyze(sales_csv).json()
    assert body["sample_data"][3]["Email Address"] is None
    assert body["sample_data"][5]["Mobile"] is None


def test_analyze_never_emits_a_bare_nan_token(analyze, sales_csv):
    """`NaN` is valid Python json but not valid JSON; browsers reject it."""
    assert "NaN" not in analyze(sales_csv).text


def test_analyze_reads_xlsx_identically_to_csv(analyze, sales_csv, sales_xlsx):
    from_csv = analyze(sales_csv).json()
    from_xlsx = analyze(sales_xlsx, name="data.xlsx").json()
    assert from_xlsx["file_columns"] == from_csv["file_columns"]
    assert from_xlsx["suggested_mapping"] == from_csv["suggested_mapping"]


@pytest.mark.parametrize("header", [
    "total_price", "Total Price", "TOTAL_PRICE",
    "customer_email", "Customer Email",
    "customer_phone", "CUSTOMER PHONE",
    "customer_id", "Customer ID",
])
def test_headers_named_after_canonical_columns_do_not_crash(analyze, header):
    """Regression: these headers used to raise TypeError and return a 500."""
    response = analyze(f"{header}\nsome value\n".encode())
    assert response.status_code == 200
    assert response.json()["suggested_mapping"] == {main.normalize_string(header): header}


def test_analyze_rejects_unsupported_extensions(analyze):
    assert analyze(b"hello", name="notes.txt").status_code == 400


def test_analyze_rejects_an_unparseable_file(analyze):
    assert analyze(b"", name="empty.csv").status_code == 400


# --- /process/ -------------------------------------------------------------

def test_process_reads_every_row_not_just_the_sample(process, sales_csv, sales_frame):
    """/analyze/ samples 10 rows; /process/ must still cover the whole file."""
    body = process(sales_csv, json.dumps({"order_id": "Order ID"})).json()
    assert len(body["data"]) == len(sales_frame) == 25


def test_process_emits_every_canonical_column(process, sales_csv):
    body = process(sales_csv, json.dumps({"order_id": "Order ID"})).json()
    assert list(body["data"][0].keys()) == CANONICAL_ORDER


def test_process_carries_values_across(process, sales_csv):
    mapping = {"order_id": "Order ID", "customer_name": "Client", "total_price": "Net Sales"}
    body = process(sales_csv, json.dumps(mapping)).json()
    assert body["data"][0]["order_id"] == 1000
    assert body["data"][0]["customer_name"] == "Customer 0"
    assert body["data"][0]["total_price"] == 10.0


def test_process_nulls_unmapped_fields(process, sales_csv):
    body = process(sales_csv, json.dumps({"order_id": "Order ID"})).json()
    assert body["data"][0]["category"] is None
    assert body["data"][0]["quantity"] is None


def test_process_reports_the_mapped_column_count(process, sales_csv):
    mapping = {"order_id": "Order ID", "customer_name": "Client"}
    body = process(sales_csv, json.dumps(mapping)).json()
    assert body["mapped_columns_count"] == 2


def test_process_preserves_blanks_as_null(process, sales_csv):
    body = process(sales_csv, json.dumps({"customer_email": "Email Address"})).json()
    assert body["data"][3]["customer_email"] is None


def test_process_ignores_a_source_column_that_does_not_exist(process, sales_csv):
    body = process(sales_csv, json.dumps({"order_id": "Ghost Column"})).json()
    assert body["data"][0]["order_id"] is None


def test_process_rejects_malformed_mappings_json(process, sales_csv):
    response = process(sales_csv, "{not json")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid mappings configuration."


def test_process_rejects_unsupported_extensions(process, sales_csv):
    assert process(sales_csv, json.dumps({}), name="notes.txt").status_code == 400


def test_process_accepts_xlsx(process, sales_xlsx):
    response = process(sales_xlsx, json.dumps({"order_id": "Order ID"}), name="data.xlsx")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 25
