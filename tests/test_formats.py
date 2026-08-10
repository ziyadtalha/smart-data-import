"""Tests for the SQLite upload path, including table selection and auto-join.

All fixtures are synthesised here, so these run without anything in testdata/.
"""
import json
import sqlite3

import pytest

# --- SQLite fixtures -------------------------------------------------------


def build_db(path, tables):
    """tables: {name: (create_sql, [rows])}"""
    connection = sqlite3.connect(path)
    try:
        for name, (create_sql, rows) in tables.items():
            connection.execute(create_sql)
            if rows:
                placeholders = ",".join("?" * len(rows[0]))
                connection.executemany(
                    f'INSERT INTO "{name}" VALUES ({placeholders})', rows
                )
        connection.commit()
    finally:
        connection.close()
    return path.read_bytes()


@pytest.fixture(scope="module")
def shop_db(tmp_path_factory):
    """A 'sales' table worth mapping, plus a bigger table that maps to nothing."""
    path = tmp_path_factory.mktemp("sqlite") / "shop.sqlite"
    return build_db(path, {
        "geo": (
            "CREATE TABLE geo (lat REAL, lng REAL)",
            [(float(i), float(i)) for i in range(200)],
        ),
        "sales": (
            "CREATE TABLE sales (order_id TEXT, customer_name TEXT, qty INTEGER, "
            "total_price REAL)",
            [(f"order-{i}", f"Customer {i}", i, i * 1.5) for i in range(20)],
        ),
    })


@pytest.fixture(scope="module")
def single_table_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("sqlite") / "one.sqlite"
    return build_db(path, {
        "orders": (
            "CREATE TABLE orders (order_id TEXT, total_price REAL)",
            [("a", 1.0), ("b", 2.0)],
        ),
    })


@pytest.fixture(scope="module")
def empty_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("sqlite") / "empty.sqlite"
    return build_db(path, {})


# --- table discovery and selection ----------------------------------------

def test_lists_every_table(analyze, shop_db):
    body = analyze(shop_db, name="shop.sqlite").json()
    assert body["tables"] == ["geo", "sales"]


def test_picks_the_table_matching_the_most_canonical_columns(analyze, shop_db):
    """'geo' has ten times the rows but maps to nothing, so 'sales' must win."""
    body = analyze(shop_db, name="shop.sqlite").json()
    assert body["selected_table"] == "sales"
    assert body["suggested_mapping"] == {
        "order_id": "order_id",
        "customer_name": "customer_name",
        "quantity": "qty",
        "total_price": "total_price",
    }


def test_row_count_breaks_ties_between_equal_matches(analyze, tmp_path):
    """Two tables map equally well; the larger one is the likelier fact table."""
    data = build_db(tmp_path / "tie.sqlite", {
        "small": (
            "CREATE TABLE small (order_id TEXT)", [("a",)],
        ),
        "large": (
            "CREATE TABLE large (order_id TEXT)", [(f"o{i}",) for i in range(50)],
        ),
    })
    assert analyze(data, name="tie.sqlite").json()["selected_table"] == "large"


def test_an_explicit_table_overrides_the_heuristic(analyze, client, shop_db):
    response = client.post(
        "/analyze/",
        files={"file": ("shop.sqlite", shop_db, "application/octet-stream")},
        data={"table": "geo"},
    )
    body = response.json()
    assert body["selected_table"] == "geo"
    assert body["file_columns"] == ["lat", "lng"]


def test_a_single_table_database_selects_that_table(analyze, single_table_db):
    body = analyze(single_table_db, name="one.sqlite").json()
    assert body["tables"] == ["orders"]
    assert body["selected_table"] == "orders"


def test_views_are_offered_alongside_tables(analyze, tmp_path):
    path = tmp_path / "view.sqlite"
    data = build_db(path, {
        "orders": (
            "CREATE TABLE orders (order_id TEXT, total_price REAL)",
            [("a", 1.0)],
        ),
    })
    connection = sqlite3.connect(path)
    connection.execute("CREATE VIEW recent AS SELECT * FROM orders")
    connection.commit()
    connection.close()
    body = analyze(path.read_bytes(), name="view.sqlite").json()
    assert "recent" in body["tables"]


@pytest.mark.parametrize("extension", [".sqlite", ".sqlite3", ".db"])
def test_every_database_extension_is_accepted(analyze, shop_db, extension):
    response = analyze(shop_db, name=f"shop{extension}")
    assert response.status_code == 200
    assert response.json()["selected_table"] == "sales"


# --- database error handling ----------------------------------------------

def test_unknown_table_is_rejected_with_the_available_names(analyze, client, shop_db):
    response = client.post(
        "/analyze/",
        files={"file": ("shop.sqlite", shop_db, "application/octet-stream")},
        data={"table": "does_not_exist"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "does_not_exist" in detail and "sales" in detail


def test_a_table_name_carrying_sql_is_rejected_not_executed(client, shop_db):
    """The name is checked against sqlite_master before it reaches a query."""
    response = client.post(
        "/analyze/",
        files={"file": ("shop.sqlite", shop_db, "application/octet-stream")},
        data={"table": "sales; DROP TABLE sales;--"},
    )
    assert response.status_code == 400


def test_a_database_with_no_tables_is_rejected(analyze, empty_db):
    response = analyze(empty_db, name="empty.sqlite")
    assert response.status_code == 400
    assert "no tables" in response.json()["detail"]


def test_a_file_that_is_not_a_database_is_rejected(analyze):
    response = analyze(b"SQLite format 3\x00 truncated", name="broken.sqlite")
    assert response.status_code == 400


# --- database end to end ---------------------------------------------------

def test_enhance_reports_the_table_it_used(enhance, fake_ollama, shop_db):
    fake_ollama(json.dumps({"order_id": "order_id"}))
    body = enhance(shop_db, name="shop.sqlite").json()
    assert body["selected_table"] == "sales"
    assert body["auto_mapping"]["order_id"] == "order_id"


def test_process_reads_the_whole_selected_table(process, shop_db):
    mapping = json.dumps({"order_id": "order_id", "total_price": "total_price"})
    body = process(shop_db, mapping, name="shop.sqlite").json()
    assert body["selected_table"] == "sales"
    assert len(body["data"]) == 20


def test_process_honours_an_explicit_table(client, shop_db):
    response = client.post(
        "/process/",
        files={"file": ("shop.sqlite", shop_db, "application/octet-stream")},
        data={"mappings_json": json.dumps({}), "table": "geo"},
    )
    body = response.json()
    assert body["selected_table"] == "geo"
    assert len(body["data"]) == 200


# --- auto-join -------------------------------------------------------------

@pytest.fixture(scope="module")
def joinable_db(tmp_path_factory):
    """A star schema with one safe lookup, one fan-out trap and one coincidence.

    - products joins many-to-one on product_id       -> safe, should be joined
    - payments has several rows per sale             -> would fan out, must be skipped
    - promos shares promo_id but matches almost none -> coincidence, must be skipped
    """
    path = tmp_path_factory.mktemp("sqlite") / "star.sqlite"
    sales = [(f"s{i}", f"p{i % 5}", f"promo{i}", 10.0) for i in range(100)]
    products = [(f"p{i}", f"Widget {i}", f"cat{i}") for i in range(5)]
    payments = [(f"s{i}", n, 5.0) for i in range(100) for n in (1, 2)]
    promos = [("promo0", "Launch offer")]
    return build_db(path, {
        "sales": (
            "CREATE TABLE sales (sale_id TEXT, product_id TEXT, promo_id TEXT, "
            "total_price REAL)", sales,
        ),
        "products": (
            "CREATE TABLE products (product_id TEXT, product_name TEXT, category TEXT)",
            products,
        ),
        "payments": (
            "CREATE TABLE payments (sale_id TEXT, seq INTEGER, amount REAL)", payments,
        ),
        "promos": (
            "CREATE TABLE promos (promo_id TEXT, description TEXT)", promos,
        ),
    })


def join_analyze(client, data, **form):
    return client.post(
        "/analyze/",
        files={"file": ("star.sqlite", data, "application/octet-stream")},
        data={"table": "sales", **form},
    )


def test_auto_join_is_off_by_default(client, joinable_db):
    body = join_analyze(client, joinable_db).json()
    assert body["joined_tables"] == []
    assert body["file_columns"] == ["sale_id", "product_id", "promo_id", "total_price"]


def test_auto_join_pulls_in_a_many_to_one_lookup(client, joinable_db):
    body = join_analyze(client, joinable_db, auto_join="true").json()
    assert "products" in body["joined_tables"]
    assert "product_name" in body["file_columns"]
    assert "category" in body["file_columns"]


def test_auto_join_skips_a_table_that_would_duplicate_rows(client, joinable_db):
    """payments has two rows per sale; joining it would inflate every total."""
    body = join_analyze(client, joinable_db, auto_join="true").json()
    assert "payments" not in body["joined_tables"]


def test_auto_join_skips_a_coincidental_key_match(client, joinable_db):
    """promos shares promo_id but covers 1 of 100 sales, so it is not a relationship."""
    body = join_analyze(client, joinable_db, auto_join="true").json()
    assert "promos" not in body["joined_tables"]


def test_auto_join_preserves_the_row_count(client, joinable_db):
    plain = client.post(
        "/process/",
        files={"file": ("star.sqlite", joinable_db, "application/octet-stream")},
        data={"mappings_json": json.dumps({"total_price": "total_price"}),
              "table": "sales"},
    ).json()
    joined = client.post(
        "/process/",
        files={"file": ("star.sqlite", joinable_db, "application/octet-stream")},
        data={"mappings_json": json.dumps({"total_price": "total_price"}),
              "table": "sales", "auto_join": "true"},
    ).json()
    assert len(plain["data"]) == len(joined["data"]) == 100


def test_auto_join_does_not_inflate_totals(client, joinable_db):
    """The whole point: a fan-out join would double this sum."""
    body = client.post(
        "/process/",
        files={"file": ("star.sqlite", joinable_db, "application/octet-stream")},
        data={"mappings_json": json.dumps({"total_price": "total_price"}),
              "table": "sales", "auto_join": "true"},
    ).json()
    assert sum(row["total_price"] for row in body["data"]) == 1000.0


def test_auto_join_improves_the_mapping(client, joinable_db):
    without = join_analyze(client, joinable_db).json()["suggested_mapping"]
    with_join = join_analyze(client, joinable_db, auto_join="true").json()["suggested_mapping"]
    assert len(with_join) > len(without)
    assert with_join["product_name"] == "product_name"
    assert with_join["category"] == "category"


def test_auto_join_keeps_column_names_unique(client, joinable_db):
    columns = join_analyze(client, joinable_db, auto_join="true").json()["file_columns"]
    assert len(columns) == len(set(columns))


def test_auto_join_is_reported_on_every_endpoint(client, joinable_db, fake_ollama):
    fake_ollama(json.dumps({}))
    for endpoint, extra in [
        ("/analyze/", {}),
        ("/enhance/", {}),
        ("/process/", {"mappings_json": json.dumps({})}),
    ]:
        response = client.post(
            endpoint,
            files={"file": ("star.sqlite", joinable_db, "application/octet-stream")},
            data={"table": "sales", "auto_join": "true", **extra},
        )
        assert response.json()["joined_tables"] == ["products"], endpoint


def test_auto_join_on_a_flat_file_is_a_no_op(analyze, sales_csv):
    body = analyze(sales_csv).json()
    assert body["joined_tables"] == []


# --- unsupported formats ---------------------------------------------------

@pytest.mark.parametrize("name", ["sales.xml", "notes.txt", "report.json", "data.parquet"])
def test_unsupported_extensions_are_rejected(analyze, name):
    response = analyze(b"<sales><sale><Qty>1</Qty></sale></sales>", name=name)
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Please upload a .csv, .xlsx, .xls, .sqlite, .sqlite3 or .db file."
    )


# --- flat formats keep empty table metadata --------------------------------

def test_csv_carries_no_table_metadata(analyze, sales_csv):
    body = analyze(sales_csv).json()
    assert body["tables"] == []
    assert body["selected_table"] is None
