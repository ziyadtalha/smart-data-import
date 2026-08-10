"""Tests for stage 2's embedding matcher.

The encoder is stubbed with deterministic vectors, so these run offline, in
milliseconds, and without fastembed installed. The tests marked `embedding`
exercise the real model and skip when it is unavailable.
"""
import sqlite3

import numpy as np
import pytest

import main
import semantic


@pytest.fixture
def stub_encoder(monkeypatch):
    """Drive the matcher with hand-written similarities.

    `install({"header": {"canonical_field": score}})` makes the encoder return
    vectors whose dot products reproduce those scores, so a test can state the
    similarity it wants to reason about instead of guessing at real embeddings.
    """
    def install(table, default=0.1):
        fields = [c["name"] for c in main.CANONICAL_COLUMNS]
        axes = {name: index for index, name in enumerate(fields)}
        size = len(fields)

        def fake_encode(texts):
            vectors = []
            for text in texts:
                vector = np.zeros(size, dtype=np.float32)
                canonical = next(
                    (
                        c["name"]
                        for c in main.CANONICAL_COLUMNS
                        if text.startswith(c["name"].replace("_", " ") + ":")
                    ),
                    None,
                )
                if canonical is not None:
                    vector[axes[canonical]] = 1.0
                else:
                    scores = table.get(text, {})
                    for field, score in scores.items():
                        vector[axes[field]] = score
                    leftover = 1.0 - float(np.sum(vector ** 2))
                    if leftover > 0:
                        # Park the remaining magnitude on an unused axis so the
                        # vector is unit length without disturbing the scores.
                        vector = np.append(vector, np.sqrt(leftover))
                    else:
                        vector = np.append(vector, 0.0)
                if len(vector) == size:
                    vector = np.append(vector, 0.0)
                vectors.append(vector)

            matrix = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            return matrix / np.maximum(norms, 1e-12)

        semantic._canonical_cache.clear()
        monkeypatch.setattr(semantic, "encode", fake_encode)
        monkeypatch.setattr(semantic, "is_available", lambda: True)

    yield install
    semantic._canonical_cache.clear()


# --- header normalization --------------------------------------------------

@pytest.mark.parametrize("header,expected", [
    ("Amount_Paid", "amount paid"),
    ("basket_size", "basket size"),
    ("WorkstationGroupID", "workstation group id"),
    ("NET_SALES_USD", "net_sales_usd".replace("_", " ")),
    ("order-purchase.timestamp", "order purchase timestamp"),
])
def test_humanize_splits_headers_into_phrases(header, expected):
    assert semantic.humanize(header) == expected


# --- identifier guard ------------------------------------------------------

@pytest.mark.parametrize("header", [
    "product_id", "seller_id", "OperatorID", "WorkstationGroupID",
    "inventory_id", "order_number", "postal_code",
])
def test_identifier_headers_are_excluded(header):
    assert semantic.looks_like_identifier(header)


@pytest.mark.parametrize("header", [
    "basket_size", "begin_date_time", "Item_Category", "amount", "Timestamp",
])
def test_ordinary_headers_are_not_treated_as_identifiers(header):
    assert not semantic.looks_like_identifier(header)


def test_identifier_headers_never_produce_a_semantic_match(stub_encoder):
    """Even at a similarity of 1.0, an identifier must not be matched."""
    stub_encoder({"product id": {"product_name": 1.0}})
    assert semantic.match(["product_id"], main.CANONICAL_COLUMNS) == {}


# --- thresholding ----------------------------------------------------------

def test_a_score_above_the_threshold_matches(stub_encoder):
    stub_encoder({"item category": {"category": 0.90}})
    matched = semantic.match(["Item_Category"], main.CANONICAL_COLUMNS)
    assert matched["category"][0] == "Item_Category"


def test_a_score_below_the_threshold_is_dropped(stub_encoder):
    stub_encoder({"mystery col": {"category": 0.50}})
    assert semantic.match(["mystery col"], main.CANONICAL_COLUMNS) == {}


def test_the_threshold_is_inclusive_of_better_scores(stub_encoder):
    stub_encoder({"a": {"category": semantic.SIMILARITY_THRESHOLD + 0.01}})
    assert "category" in semantic.match(["a"], main.CANONICAL_COLUMNS)


# --- one-to-one ------------------------------------------------------------

def test_the_best_scoring_header_wins_a_contested_field(stub_encoder):
    stub_encoder({
        "first guess": {"category": 0.80},
        "second guess": {"category": 0.95},
    })
    matched = semantic.match(
        ["first guess", "second guess"], main.CANONICAL_COLUMNS
    )
    assert matched["category"][0] == "second guess"
    assert len(matched) == 1


def test_a_header_is_never_assigned_twice(stub_encoder):
    stub_encoder({"one header": {"category": 0.95, "product_name": 0.90}})
    matched = semantic.match(["one header"], main.CANONICAL_COLUMNS)
    assert list(matched) == ["category"]


def test_a_header_whose_best_field_is_taken_is_dropped_not_reassigned(stub_encoder):
    """No consolation prizes.

    olist carries product_category_name and product_category_name_english.
    Both are plainly categories; the runner-up used to slide onto product_name
    at 0.780 purely because category was gone.
    """
    stub_encoder({
        "cat pt": {"category": 0.90, "product_name": 0.80},
        "cat en": {"category": 0.85, "product_name": 0.78},
    })
    matched = semantic.match(["cat pt", "cat en"], main.CANONICAL_COLUMNS)
    assert matched["category"][0] == "cat pt"
    assert "product_name" not in matched


def test_a_header_is_judged_against_filled_fields_too(stub_encoder):
    """If a header most resembles a field the rules already filled, it has
    nothing confident to say about whatever is left over."""
    stub_encoder({"spare": {"category": 0.95, "product_name": 0.85}})
    matched = semantic.match(
        ["spare"], main.CANONICAL_COLUMNS, taken_fields={"category"}
    )
    assert matched == {}


# --- type guard ------------------------------------------------------------

def test_a_numeric_column_cannot_fill_a_string_field(stub_encoder):
    """product_name_lenght holds the length of the name, and scores 0.712."""
    stub_encoder({"product name lenght": {"product_name": 0.95}})
    matched = semantic.match(
        ["product_name_lenght"],
        main.CANONICAL_COLUMNS,
        column_kinds={"product_name_lenght": "number"},
    )
    assert matched == {}


def test_a_text_column_cannot_fill_a_decimal_field(stub_encoder):
    stub_encoder({"payment method": {"total_price": 0.95}})
    matched = semantic.match(
        ["Payment_Method"],
        main.CANONICAL_COLUMNS,
        column_kinds={"Payment_Method": "text"},
    )
    assert matched == {}


def test_a_text_column_still_fills_a_string_field(stub_encoder):
    stub_encoder({"item cat": {"category": 0.95}})
    matched = semantic.match(
        ["item cat"], main.CANONICAL_COLUMNS, column_kinds={"item cat": "text"}
    )
    assert matched["category"][0] == "item cat"


def test_dates_are_not_type_checked(stub_encoder):
    """Dates arrive as strings as often as not, so text must stay acceptable."""
    stub_encoder({"purchase stamp": {"order_date": 0.95}})
    matched = semantic.match(
        ["purchase stamp"],
        main.CANONICAL_COLUMNS,
        column_kinds={"purchase stamp": "text"},
    )
    assert matched["order_date"][0] == "purchase stamp"


def test_an_all_null_column_is_not_rejected_on_type(stub_encoder):
    stub_encoder({"item cat": {"category": 0.95}})
    matched = semantic.match(
        ["item cat"], main.CANONICAL_COLUMNS, column_kinds={"item cat": "unknown"}
    )
    assert "category" in matched


@pytest.mark.parametrize("canonical_type,kind,conflicts", [
    ("string", "number", True),
    ("string", "text", False),
    ("string", "unknown", False),
    ("decimal", "text", True),
    ("integer", "datetime", True),
    ("integer", "number", False),
    ("date", "text", False),
    ("date", "number", False),
])
def test_type_conflict_rules(canonical_type, kind, conflicts):
    assert semantic.type_conflicts(canonical_type, kind) is conflicts


def test_column_kinds_classifies_a_sample(sales_frame):
    kinds = main.column_kinds(sales_frame)
    assert kinds["Client"] == "text"
    assert kinds["Qty"] == "number"
    assert kinds["Price"] == "number"


def test_fields_already_taken_by_rules_are_left_alone(stub_encoder):
    stub_encoder({"spare": {"category": 0.95}})
    matched = semantic.match(
        ["spare"], main.CANONICAL_COLUMNS, taken_fields={"category"}
    )
    assert matched == {}


def test_columns_already_used_by_rules_are_left_alone(stub_encoder):
    stub_encoder({"spare": {"category": 0.95}})
    matched = semantic.match(
        ["spare"], main.CANONICAL_COLUMNS, taken_columns={"spare"}
    )
    assert matched == {}


# --- integration with the rule stage ---------------------------------------

def test_suggest_mapping_labels_where_each_field_came_from(stub_encoder):
    stub_encoder({"item category": {"category": 0.95}})
    mapping, sources = main.suggest_mapping(["Order ID", "Item_Category"])
    assert mapping == {"order_id": "Order ID", "category": "Item_Category"}
    assert sources == {"order_id": "rule", "category": "semantic"}


def test_semantic_never_overrides_a_rule_match(stub_encoder):
    stub_encoder({"category": {"product_name": 0.99}})
    mapping, sources = main.suggest_mapping(["category"])
    assert mapping == {"category": "category"}
    assert sources["category"] == "rule"


def test_suggest_mapping_can_be_turned_off(stub_encoder):
    stub_encoder({"item category": {"category": 0.95}})
    mapping, sources = main.suggest_mapping(["Item_Category"], use_semantic=False)
    assert mapping == {}
    assert sources == {}


def test_missing_backend_falls_back_to_rules(monkeypatch):
    monkeypatch.setattr(semantic, "is_available", lambda: False)
    mapping, sources = main.suggest_mapping(["Order ID", "Item_Category"])
    assert mapping == {"order_id": "Order ID"}
    assert sources == {"order_id": "rule"}


def test_a_broken_encoder_falls_back_to_rules(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(semantic, "is_available", lambda: True)
    monkeypatch.setattr(semantic, "match", explode)
    mapping, sources = main.suggest_mapping(["Order ID", "Item_Category"])
    assert mapping == {"order_id": "Order ID"}


# --- endpoint --------------------------------------------------------------

def test_analyze_reports_mapping_sources(analyze, sales_csv):
    body = analyze(sales_csv).json()
    assert set(body["mapping_sources"]) == set(body["suggested_mapping"])
    assert all(v in {"rule", "semantic"} for v in body["mapping_sources"].values())


def build_db(path, statements):
    connection = sqlite3.connect(path)
    try:
        for statement in statements:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()
    return path.read_bytes()


def test_database_uploads_get_the_semantic_stage(analyze, tmp_path, stub_encoder):
    stub_encoder({"item cat": {"category": 0.99}})
    data = build_db(tmp_path / "one.sqlite", [
        "CREATE TABLE sales (order_id TEXT, item_cat TEXT)",
        "INSERT INTO sales VALUES ('a', 'Drinks')",
    ])
    body = analyze(data, name="one.sqlite").json()
    assert body["selected_table"] == "sales"
    assert body["suggested_mapping"]["category"] == "item_cat"
    assert body["mapping_sources"]["category"] == "semantic"


def test_table_scoring_does_not_embed_once_per_table(analyze, tmp_path, monkeypatch):
    """Choosing among tables must stay on the rules.

    The scorer runs the mapper once per table, so embedding there would scale
    the request with the size of the schema. Stage 2 runs once, on the table
    that was actually chosen.
    """
    calls = {"n": 0}

    def counting_match(*args, **kwargs):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(semantic, "is_available", lambda: True)
    monkeypatch.setattr(semantic, "match", counting_match)

    data = build_db(tmp_path / "many.sqlite", [
        "CREATE TABLE a (order_id TEXT, note TEXT)",
        "CREATE TABLE b (customer_name TEXT, note TEXT)",
        "CREATE TABLE c (quantity INTEGER, note TEXT)",
        "CREATE TABLE d (total_price REAL, note TEXT)",
        "INSERT INTO a VALUES ('x', 'n')",
    ])
    analyze(data, name="many.sqlite")
    assert calls["n"] == 1, "semantic stage ran more than once for a 4-table database"


# --- the real model --------------------------------------------------------

@pytest.mark.embedding
@pytest.mark.parametrize("header,expected", [
    ("Item_Category", "category"),
    ("product_category_name_english", "category"),
    ("order_purchase_timestamp", "order_date"),
    ("payment_date", "order_date"),
    ("basket_size", "quantity"),
    ("payment_value", "total_price"),
    ("total_sales", "total_price"),
    ("first_name", "customer_name"),
])
def test_real_model_matches_known_headers(header, expected):
    if not semantic.is_available():
        pytest.skip("fastembed not installed")
    matched = semantic.match([header], main.CANONICAL_COLUMNS)
    assert expected in matched, f"{header} should have matched {expected}"


@pytest.mark.embedding
@pytest.mark.parametrize("header", [
    "order_status", "Payment_Method", "freight_value", "last_update",
    "release_year", "special_features", "return_date", "customer_city",
    "shipping_limit_date", "product_weight_g", "declared_monthly_revenue",
    "district",
])
def test_real_model_refuses_known_traps(header):
    """return_date vs order_date is the closest call at 0.660.

    It is what sets the threshold: no wrong match on the labelled headers in
    testdata/ scores higher, and every correct one scores at least 0.679.
    """
    if not semantic.is_available():
        pytest.skip("fastembed not installed")
    assert semantic.match([header], main.CANONICAL_COLUMNS) == {}


@pytest.mark.embedding
def test_real_model_declines_a_date_it_cannot_separate_from_a_return():
    """In sakila's rental table, rental_date is the sale and return_date is not.

    They sit 0.017 apart and the wrong one is on top, so stage 2 takes neither
    rather than guess. The opt-in LLM stage exists for exactly this.
    """
    if not semantic.is_available():
        pytest.skip("fastembed not installed")
    matched = semantic.match(["rental_date", "return_date"], main.CANONICAL_COLUMNS)
    assert "order_date" not in matched


@pytest.mark.embedding
def test_real_model_refuses_a_length_column_by_type():
    """product_name_lenght scores 0.712 against product_name on name alone."""
    if not semantic.is_available():
        pytest.skip("fastembed not installed")
    columns = ["product_name_lenght"]
    assert "product_name" in semantic.match(columns, main.CANONICAL_COLUMNS)
    assert semantic.match(
        columns, main.CANONICAL_COLUMNS,
        column_kinds={"product_name_lenght": "number"},
    ) == {}


@pytest.mark.embedding
def test_real_model_keeps_only_one_of_two_category_columns():
    """Both olist category columns resemble `category`; only one may win."""
    if not semantic.is_available():
        pytest.skip("fastembed not installed")
    matched = semantic.match(
        ["product_category_name", "product_category_name_english"],
        main.CANONICAL_COLUMNS,
    )
    assert list(matched) == ["category"]
