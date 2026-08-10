"""Tests against the real datasets in testdata/.

Those files are large and git-ignored, so every test here skips when the file
is absent. See testdata/README.md for download links.
"""
import json
import sqlite3

import pandas as pd
import pytest

import main
from conftest import TESTDATA

POS_CSV = TESTDATA / "simulated_pos_data_with_seasonal_trends.csv"
OLIST_DB = TESTDATA / "olist.sqlite"
POS_TRANSACTIONS = TESTDATA / "pos_transactions.csv"
POS_OPERATOR_LOGS = TESTDATA / "pos_operator_logs.csv"

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


# --- supermarket POS (Open Source Point of Sale) ---------------------------

@pytest.fixture(scope="module")
def transactions_bytes():
    if not POS_TRANSACTIONS.exists():
        pytest.skip(f"{POS_TRANSACTIONS.name} not present")
    return POS_TRANSACTIONS.read_bytes()


def test_transactions_amount_maps_to_total_not_unit_price(analyze, transactions_bytes):
    """The row is basket_size=23, amount=112.71 — a basket total, not a unit price."""
    mapping = analyze(transactions_bytes, name=POS_TRANSACTIONS.name).json()[
        "suggested_mapping"
    ]
    assert mapping["total_price"] == "amount"
    assert "unit_price" not in mapping


def test_transactions_id_maps_to_order_id(analyze, transactions_bytes):
    mapping = analyze(transactions_bytes, name=POS_TRANSACTIONS.name).json()[
        "suggested_mapping"
    ]
    assert mapping["order_id"] == "id"


def test_transactions_process_every_row(process, transactions_bytes):
    expected = len(pd.read_csv(POS_TRANSACTIONS))
    body = process(
        transactions_bytes,
        json.dumps({"order_id": "id", "total_price": "amount"}),
        name=POS_TRANSACTIONS.name,
    ).json()
    assert len(body["data"]) == expected == 163269


def test_operator_logs_analyze(analyze):
    if not POS_OPERATOR_LOGS.exists():
        pytest.skip(f"{POS_OPERATOR_LOGS.name} not present")
    response = analyze(POS_OPERATOR_LOGS.read_bytes(), name=POS_OPERATOR_LOGS.name)
    assert response.status_code == 200
    assert response.json()["file_columns"] == [
        "id", "Workstation_Group_ID", "Workstation_ID", "begin_date_time", "operator_id"
    ]


# --- olist e-commerce database --------------------------------------------

SAKILA_DB = TESTDATA / "sqlite-sakila.db"


@pytest.fixture(scope="module")
def sakila_bytes():
    if not SAKILA_DB.exists():
        pytest.skip(f"{SAKILA_DB.name} not present")
    return SAKILA_DB.read_bytes()


def test_sakila_selects_the_payment_table(analyze, sakila_bytes):
    """21 tables, and the transaction table wins on canonical matches."""
    body = analyze(sakila_bytes, name=SAKILA_DB.name).json()
    assert len(body["tables"]) == 21
    assert body["selected_table"] == "payment"


def test_sakila_amount_maps_to_total_price(analyze, sakila_bytes):
    """A second dataset where 'amount' is a transaction total, not a unit price."""
    mapping = analyze(sakila_bytes, name=SAKILA_DB.name).json()["suggested_mapping"]
    assert mapping["total_price"] == "amount"
    assert mapping["customer_id"] == "customer_id"


def test_sakila_auto_join_reaches_contact_details(client, sakila_bytes):
    """email lives on customer and phone on address, two hops from payment."""
    response = client.post(
        "/analyze/",
        files={"file": (SAKILA_DB.name, sakila_bytes, "application/octet-stream")},
        data={"auto_join": "true"},
    )
    body = response.json()
    assert "customer" in body["joined_tables"]
    assert "address" in body["joined_tables"]
    assert body["suggested_mapping"]["customer_email"] == "email"
    assert body["suggested_mapping"]["customer_phone"] == "phone"


def test_sakila_auto_join_respects_the_table_ceiling(client, sakila_bytes):
    response = client.post(
        "/analyze/",
        files={"file": (SAKILA_DB.name, sakila_bytes, "application/octet-stream")},
        data={"auto_join": "true"},
    )
    joined = response.json()["joined_tables"]
    assert len(joined) == main.MAX_AUTO_JOIN_TABLES - 1


def test_sakila_auto_join_preserves_payments(client, sakila_bytes):
    def totals(**form):
        response = client.post(
            "/process/",
            files={"file": (SAKILA_DB.name, sakila_bytes, "application/octet-stream")},
            data={"mappings_json": json.dumps({"total_price": "amount"}), **form},
        )
        rows = response.json()["data"]
        return len(rows), round(
            sum(r["total_price"] for r in rows if r["total_price"] is not None), 2
        )

    assert totals() == totals(auto_join="true") == (16049, 67416.51)


def test_sakila_declares_foreign_keys_but_they_are_not_required(sakila_bytes, tmp_path):
    """Sakila declares FKs where olist declares none; the planner ignores both."""
    path = tmp_path / "sakila.db"
    path.write_bytes(sakila_bytes)
    connection = sqlite3.connect(path)
    try:
        declared = list(connection.execute("PRAGMA foreign_key_list(payment)"))
    finally:
        connection.close()
    assert declared, "expected sakila to declare foreign keys"


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


@pytest.fixture(scope="module")
def olist_db_bytes():
    if not OLIST_DB.exists():
        pytest.skip(f"{OLIST_DB.name} not present")
    return OLIST_DB.read_bytes()


def test_olist_database_uploads_directly(analyze, olist_db_bytes):
    response = analyze(olist_db_bytes, name=OLIST_DB.name)
    assert response.status_code == 200


def test_olist_lists_all_eleven_tables(analyze, olist_db_bytes):
    body = analyze(olist_db_bytes, name=OLIST_DB.name).json()
    assert len(body["tables"]) == 11
    assert "order_items" in body["tables"]
    assert "geolocation" in body["tables"]


def test_olist_auto_selects_order_items(analyze, olist_db_bytes):
    """geolocation has ten times the rows but maps to nothing."""
    body = analyze(olist_db_bytes, name=OLIST_DB.name).json()
    assert body["selected_table"] == "order_items"
    assert body["suggested_mapping"] == {"order_id": "order_id", "unit_price": "price"}


def test_olist_orders_table_maps_ids(analyze, client, olist_db_bytes):
    response = client.post(
        "/analyze/",
        files={"file": (OLIST_DB.name, olist_db_bytes, "application/octet-stream")},
        data={"table": "orders"},
    )
    body = response.json()
    assert body["selected_table"] == "orders"
    assert body["suggested_mapping"]["order_id"] == "order_id"
    assert body["suggested_mapping"]["customer_id"] == "customer_id"


def test_olist_auto_join_expands_order_items(analyze, client, olist_db_bytes):
    response = client.post(
        "/analyze/",
        files={"file": (OLIST_DB.name, olist_db_bytes, "application/octet-stream")},
        data={"auto_join": "true"},
    )
    body = response.json()
    assert body["selected_table"] == "order_items"
    assert set(body["joined_tables"]) == {
        "orders", "products", "sellers", "customers",
        "product_category_name_translation",
    }
    assert "order_purchase_timestamp" in body["file_columns"]
    assert "product_category_name_english" in body["file_columns"]


def test_olist_auto_join_refuses_the_fan_out_tables(analyze, client, olist_db_bytes):
    """order_payments and order_reviews have several rows per order."""
    response = client.post(
        "/analyze/",
        files={"file": (OLIST_DB.name, olist_db_bytes, "application/octet-stream")},
        data={"auto_join": "true"},
    )
    joined = response.json()["joined_tables"]
    assert "order_payments" not in joined
    assert "order_reviews" not in joined
    assert "geolocation" not in joined


def test_olist_auto_join_refuses_the_unrelated_leads_tables(client, olist_db_bytes):
    """leads_closed shares seller_id with order_items but matches only 4.5% of it."""
    response = client.post(
        "/analyze/",
        files={"file": (OLIST_DB.name, olist_db_bytes, "application/octet-stream")},
        data={"auto_join": "true"},
    )
    joined = response.json()["joined_tables"]
    assert "leads_closed" not in joined
    assert "leads_qualified" not in joined


def test_olist_auto_join_keeps_revenue_intact(client, olist_db_bytes):
    """Joining order_payments would report 14.2M instead of 13.6M."""
    def revenue(**form):
        response = client.post(
            "/process/",
            files={"file": (OLIST_DB.name, olist_db_bytes, "application/octet-stream")},
            data={"mappings_json": json.dumps({"unit_price": "price"}),
                  "table": "order_items", **form},
        )
        rows = response.json()["data"]
        return len(rows), round(
            sum(r["unit_price"] for r in rows if r["unit_price"] is not None), 2
        )

    plain_rows, plain_total = revenue()
    joined_rows, joined_total = revenue(auto_join="true")
    assert plain_rows == joined_rows == 112650
    assert plain_total == joined_total == 13591643.70


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
