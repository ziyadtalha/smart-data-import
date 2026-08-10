"""Tests for the LLM-backed /enhance/ endpoint.

Every test stubs ollama.Client, so the suite never touches the network and
needs no OLLAMA_API_KEY.
"""
import json

import main

TWO_COLUMN_CSV = b"Order ID,Widget Code\n1,abc\n2,def\n"
FILE_COLUMNS = ["Order ID", "Widget Code"]


# --- prompt_for_mapping ----------------------------------------------------

def test_parses_a_plain_json_reply(fake_ollama):
    fake_ollama(json.dumps({"order_id": "Order ID", "product_name": "Widget Code"}))
    assert main.prompt_for_mapping(FILE_COLUMNS) == {
        "order_id": "Order ID",
        "product_name": "Widget Code",
    }


def test_strips_markdown_code_fences(fake_ollama):
    fake_ollama('```json\n{"order_id": "Order ID"}\n```')
    assert main.prompt_for_mapping(FILE_COLUMNS) == {"order_id": "Order ID"}


def test_reassembles_a_reply_split_across_stream_chunks(fake_ollama):
    fake_ollama(json.dumps({"order_id": "Order ID"}), chunks=25)
    assert main.prompt_for_mapping(FILE_COLUMNS) == {"order_id": "Order ID"}


def test_returns_empty_when_the_reply_is_not_json(fake_ollama):
    fake_ollama("I think the first column is the order identifier.")
    assert main.prompt_for_mapping(FILE_COLUMNS) == {}


def test_returns_empty_when_the_reply_is_json_but_not_an_object(fake_ollama):
    fake_ollama(json.dumps(["order_id", "product_name"]))
    assert main.prompt_for_mapping(FILE_COLUMNS) == {}


def test_returns_empty_when_the_api_call_fails(fake_ollama):
    fake_ollama(error=RuntimeError("connection refused"))
    assert main.prompt_for_mapping(FILE_COLUMNS) == {}


def test_drops_columns_the_model_invented(fake_ollama):
    fake_ollama(json.dumps({"order_id": "Order ID", "customer_name": "Nonexistent"}))
    assert main.prompt_for_mapping(FILE_COLUMNS) == {"order_id": "Order ID"}


def test_drops_canonical_fields_the_model_invented(fake_ollama):
    fake_ollama(json.dumps({"order_id": "Order ID", "not_a_field": "Widget Code"}))
    assert main.prompt_for_mapping(FILE_COLUMNS) == {"order_id": "Order ID"}


def test_enforces_one_to_one_on_the_model_output(fake_ollama):
    """The model is told not to reuse a column, but the result is validated anyway."""
    fake_ollama(json.dumps({"order_id": "Order ID", "customer_id": "Order ID"}))
    assert main.prompt_for_mapping(FILE_COLUMNS) == {"order_id": "Order ID"}


# --- /enhance/ -------------------------------------------------------------

def test_enhance_returns_both_mappings(enhance, fake_ollama):
    fake_ollama(json.dumps({"order_id": "Order ID", "product_name": "Widget Code"}))
    body = enhance(TWO_COLUMN_CSV).json()
    assert body["auto_mapping"] == {"order_id": "Order ID"}
    assert body["ai_mapping"] == {"order_id": "Order ID", "product_name": "Widget Code"}


def test_enhance_can_fill_fields_the_rules_missed(enhance, fake_ollama):
    fake_ollama(json.dumps({"order_id": "Order ID", "product_name": "Widget Code"}))
    body = enhance(TWO_COLUMN_CSV).json()
    assert len(body["ai_mapping"]) > len(body["auto_mapping"])


def test_enhance_degrades_gracefully_when_the_model_is_unavailable(enhance, fake_ollama):
    """No key, no network, bad reply: the rule-based mapping must still come back."""
    fake_ollama(error=RuntimeError("Illegal header value b'Bearer '"))
    response = enhance(TWO_COLUMN_CSV)
    assert response.status_code == 200
    body = response.json()
    assert body["ai_mapping"] == {}
    assert body["auto_mapping"] == {"order_id": "Order ID"}


def test_enhance_rejects_unsupported_extensions(client, fake_ollama):
    fake_ollama(json.dumps({}))
    response = client.post(
        "/enhance/", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


def test_enhance_accepts_xlsx(enhance, fake_ollama, sales_xlsx):
    fake_ollama(json.dumps({"order_id": "Order ID"}))
    response = enhance(sales_xlsx, name="data.xlsx")
    assert response.status_code == 200
    assert response.json()["ai_mapping"] == {"order_id": "Order ID"}
