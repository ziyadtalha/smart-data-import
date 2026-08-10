"""Tests against the real datasets in testdata/.

Those files are large and git-ignored, so every test here skips when the file
is absent. See testdata/README.md for download links.
"""
import json
import sqlite3

import pandas as pd
import pytest

from conftest import TESTDATA

POS_CSV = TESTDATA / "simulated_pos_data_with_seasonal_trends.csv"
OLIST_DB = TESTDATA / "olist.sqlite"

pytestmark = pytest.mark.realdata


# --- simulated restaurant POS ---------------------------------------------

@pytest.fixture(scope="module")
def pos_bytes():
    if not POS_CSV.exists():
        pytest.skip(f"{POS_CSV.name} not present")
    return POS_CSV.read_bytes()


def test_pos_file_analyzes(analyze, pos_bytes):
    """This file's Total_Price header used to return a 500."""
    response = analyze(pos_bytes, name=POS_CSV.name)
    assert response.status_code == 200


def test_pos_headers_map_as_expected(analyze, pos_bytes):
    mapping = analyze(pos_bytes, name=POS_CSV.name).json()["suggested_mapping"]
    assert mapping["customer_id"] == "Customer_ID"
    assert mapping["customer_name"] == "Customer_Name"
    assert mapping["product_name"] == "Item_Name"
    assert mapping["quantity"] == "Quantity"
    assert mapping["total_price"] == "Total_Price"


def test_pos_payment_method_stays_unmapped(analyze, pos_bytes):
    """There is no canonical field for it; inventing one would be wrong."""
    mapping = analyze(pos_bytes, name=POS_CSV.name).json()["suggested_mapping"]
    assert "Payment_Method" not in mapping.values()


def test_pos_processes_every_row(process, pos_bytes):
    expected = len(pd.read_csv(POS_CSV))
    mapping = json.dumps({"customer_id": "Customer_ID", "total_price": "Total_Price"})
    body = process(pos_bytes, mapping, name=POS_CSV.name).json()
    assert len(body["data"]) == expected


# --- olist e-commerce database --------------------------------------------

OLIST_QUERY = """
SELECT
    o.order_id,
    o.order_purchase_timestamp,
    c.customer_unique_id,
    c.customer_city,
    t.product_category_name_english,
    oi.order_item_id,
    oi.price,
    oi.freight_value
FROM order_items oi
JOIN orders o    ON o.order_id = oi.order_id
JOIN customers c ON c.customer_id = o.customer_id
LEFT JOIN products p ON p.product_id = oi.product_id
LEFT JOIN product_category_name_translation t
       ON t.product_category_name = p.product_category_name
LIMIT 500
"""


@pytest.fixture(scope="module")
def olist_csv_bytes():
    if not OLIST_DB.exists():
        pytest.skip(f"{OLIST_DB.name} not present")
    connection = sqlite3.connect(OLIST_DB)
    try:
        frame = pd.read_sql_query(OLIST_QUERY, connection)
    finally:
        connection.close()
    return frame.to_csv(index=False).encode()


def test_sqlite_uploads_are_rejected(analyze):
    """The endpoints read CSV/XLSX only; a .sqlite must be exported first."""
    response = analyze(b"SQLite format 3\x00", name="olist.sqlite")
    assert response.status_code == 400


def test_olist_export_analyzes(analyze, olist_csv_bytes):
    assert analyze(olist_csv_bytes, name="olist.csv").status_code == 200


def test_olist_headers_map_as_expected(analyze, olist_csv_bytes):
    mapping = analyze(olist_csv_bytes, name="olist.csv").json()["suggested_mapping"]
    assert mapping["order_id"] == "order_id"
    assert mapping["customer_id"] == "customer_unique_id"
    assert mapping["unit_price"] == "price"


def test_olist_nulls_survive_as_json_null(analyze, olist_csv_bytes):
    """Some rows have no category translation."""
    response = analyze(olist_csv_bytes, name="olist.csv")
    assert "NaN" not in response.text


def test_olist_processes_without_losing_rows(process, olist_csv_bytes):
    expected = len(olist_csv_bytes.decode().strip().splitlines()) - 1
    mapping = json.dumps({"order_id": "order_id", "unit_price": "price"})
    body = process(olist_csv_bytes, mapping, name="olist.csv").json()
    assert len(body["data"]) == expected
