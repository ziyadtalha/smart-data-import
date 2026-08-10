"""Unit tests for the rule-based mapping engine."""
import pytest

import main


# --- normalization ---------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Order ID", "order_id"),
    ("  Qty  ", "qty"),
    ("customer-name", "customer_name"),
    ("TOTAL_AMOUNT", "total_amount"),
    ("Customer_ID", "customer_id"),
])
def test_normalize_string(raw, expected):
    assert main.normalize_string(raw) == expected


# --- lookup table integrity ------------------------------------------------

def test_lookup_values_are_always_canonical_name_strings():
    """Regression guard.

    The lookup once stored the whole column dict under the canonical name while
    storing the name string under each synonym. Seven of ten fields had a
    synonym that normalized back to the canonical name and overwrote the dict,
    which hid the inconsistency; the rest kept the dict and made
    auto_map_columns raise `TypeError: unhashable type: 'dict'`.
    """
    for key, value in main.CANONICAL_COLUMN_LOOKUP.items():
        assert isinstance(value, str), f"{key!r} maps to {type(value).__name__}, not str"
        assert value in main.CANONICAL_COLUMN_NAMES


@pytest.mark.parametrize("name", main.CANONICAL_COLUMN_NAMES)
def test_canonical_name_resolves_to_itself(name):
    assert main.CANONICAL_COLUMN_LOOKUP[name] == name


def test_no_synonym_is_claimed_by_two_canonical_columns():
    """A shared synonym would silently resolve to whichever field was defined last."""
    owners = {}
    for column in main.CANONICAL_COLUMNS:
        for alias in column["synonyms"]:
            key = main.normalize_string(alias)
            assert key not in owners, (
                f"synonym {alias!r} claimed by both {owners.get(key)!r} and {column['name']!r}"
            )
            owners[key] = column["name"]


def test_every_canonical_column_is_fully_defined():
    for column in main.CANONICAL_COLUMNS:
        assert set(column) == {"name", "description", "synonyms", "type"}
        assert column["name"] and column["description"]
        assert isinstance(column["synonyms"], list) and column["synonyms"]


# --- auto_map_columns ------------------------------------------------------

def test_auto_map_matches_exact_names_and_synonyms():
    mapping = main.auto_map_columns(["Order ID", "Client", "Qty", "Net Sales"])
    assert mapping == {
        "order_id": "Order ID",
        "customer_name": "Client",
        "quantity": "Qty",
        "total_price": "Net Sales",
    }


def test_auto_map_ignores_unrecognized_headers():
    assert main.auto_map_columns(["Payment_Method", "Loyalty Tier"]) == {}


def test_auto_map_never_reuses_a_source_column():
    """'price' and 'unit_price' both name unit_price; only the first may win."""
    mapping = main.auto_map_columns(["price", "unit_price"])
    assert mapping == {"unit_price": "price"}
    assert len(set(mapping.values())) == len(mapping)


@pytest.mark.parametrize("header", ["amount", "Amount", "AMOUNT"])
def test_amount_is_a_total_not_a_unit_price(header):
    """In point-of-sale exports 'amount' is the basket total.

    pos_transactions.csv pairs basket_size=23 with amount=112.71, so reading it
    as a per-unit price understates the row by a factor of the basket size.
    """
    assert main.auto_map_columns([header]) == {"total_price": header}


def test_unit_price_still_wins_its_own_synonyms():
    mapping = main.auto_map_columns(["Price_Per_Item", "Total_Price"])
    assert mapping == {"unit_price": "Price_Per_Item", "total_price": "Total_Price"}


def test_auto_map_handles_empty_input():
    assert main.auto_map_columns([]) == {}


# --- customer_id -----------------------------------------------------------

def test_customer_id_is_a_canonical_column():
    assert "customer_id" in main.CANONICAL_COLUMN_NAMES


@pytest.mark.parametrize("header", [
    "customer_id", "Customer_ID", "CUSTOMER ID", "customerid",
    "client_id", "buyer_id", "customer_unique_id", "customer_number",
])
def test_customer_id_synonyms_resolve(header):
    assert main.auto_map_columns([header]) == {"customer_id": header}


def test_customer_id_does_not_shadow_customer_name():
    mapping = main.auto_map_columns(["Customer_ID", "Customer_Name"])
    assert mapping == {"customer_id": "Customer_ID", "customer_name": "Customer_Name"}


def test_customer_id_does_not_shadow_order_id():
    mapping = main.auto_map_columns(["Order ID", "Customer_ID"])
    assert mapping == {"order_id": "Order ID", "customer_id": "Customer_ID"}


# --- validate_one_to_one_mapping -------------------------------------------

def test_validate_drops_unknown_canonical_field():
    assert main.validate_one_to_one_mapping({"not_a_field": "A"}, ["A"]) == {}


def test_validate_drops_source_column_absent_from_file():
    assert main.validate_one_to_one_mapping({"order_id": "Ghost"}, ["A"]) == {}


def test_validate_drops_duplicate_source_column():
    cleaned = main.validate_one_to_one_mapping(
        {"order_id": "A", "customer_id": "A"}, ["A"]
    )
    assert cleaned == {"order_id": "A"}


def test_validate_keeps_a_clean_mapping_intact():
    mapping = {"order_id": "A", "customer_id": "B"}
    assert main.validate_one_to_one_mapping(mapping, ["A", "B"]) == mapping
