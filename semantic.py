"""Stage 2's last step: embedding similarity against canonical field descriptions.

This is a synonym matcher, not a language model. It runs locally through
fastembed's ONNX build of BAAI/bge-small-en-v1.5, costs nothing per request and
never sends data anywhere. The LLM in main.prompt_for_mapping is a separate,
opt-in stage that the user triggers by hand.

The whole module degrades to a no-op when fastembed is not installed, so the
rule-based mapping keeps working on a minimal install.
"""
import re
import threading

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Cosine similarity below this is treated as no match.
#
# Calibrated by sweeping the threshold over 103 hand-labelled headers drawn from
# every table in testdata/, and taking the widest band that admits no wrong
# match at all. That band is [0.665, 0.679), bounded below by the hardest wrong
# match — "return_date" -> order_date at 0.660, in a rental table where the
# genuine order date is "rental_date" — and above by the weakest correct one,
# "payment_value" -> total_price at 0.679. 0.67 sits in the middle of it.
#
# What it still costs: "Timestamp" (0.640), "rental_date" (0.643) and
# "begin_date_time" (0.639) are correct matches this stage declines rather than
# guess at, because "return_date" and "end_date_time" sit right on top of them.
# The opt-in LLM stage exists for those.
SIMILARITY_THRESHOLD = 0.67

_model = None
_model_lock = threading.Lock()
_canonical_cache: dict[tuple[str, ...], np.ndarray] = {}


def is_available() -> bool:
    """True when the embedding backend can be imported."""
    try:
        import fastembed  # noqa: F401
    except Exception:
        return False
    return True


def _get_model():
    """Load the model once, on first use.

    Kept out of import so the app starts instantly and works offline until
    somebody actually asks for a semantic match.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from fastembed import TextEmbedding

                _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """Embed texts and L2-normalize, so a dot product is cosine similarity."""
    vectors = np.array(list(_get_model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def humanize(header: str) -> str:
    """Turn a column header into the kind of phrase the model was trained on."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(header))
    text = re.sub(r"[_\-.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def looks_like_identifier(header: str) -> bool:
    """True for foreign-key style headers, which must not be matched by meaning.

    Every identifier embeds close to every other one: on olist, seller_id
    scores 0.670 against customer_id and product_id 0.665, near enough to the
    threshold to land a foreign key on the wrong field. Identifier headers that
    really are canonical (id, order_id, customer_id) are caught by the synonym
    rules before this stage runs, so excluding them here costs nothing.
    """
    return re.search(r"(^|\s)(id|ids|identifier|no|number|code)$", humanize(header)) is not None


NUMERIC_FIELD_TYPES = {"integer", "decimal"}


def type_conflicts(canonical_type: str, kind: str | None) -> bool:
    """True when the sampled values cannot be what the canonical field declares.

    Header text alone cannot tell product_name from product_name_lenght — the
    latter holds the length of the name and scores 0.689 against product_name,
    because that is genuinely what its name resembles. The values settle it:
    product_name is declared a string and the column is full of integers.

    Only clear contradictions are rejected. Dates are left alone because they
    arrive as strings as often as not.
    """
    if kind in (None, "unknown"):
        return False
    if canonical_type == "string" and kind == "number":
        return True
    if canonical_type in NUMERIC_FIELD_TYPES and kind in ("text", "datetime"):
        return True
    return False


def canonical_text(column: dict) -> str:
    """Name, description and synonyms together.

    Measured over the labelled headers in testdata/, at each variant's best
    threshold that admits no wrong match: name alone and name plus synonyms
    never reach one, because some wrong match outscores every right match at
    every cut. Name plus description reaches 9 correct matches, and adding the
    synonyms reaches the same 9 at a lower threshold — 0.665 rather than 0.695
    — so the wrong matches sit further below the cut. The synonyms are kept for
    that margin rather than for reach.
    """
    return (
        f"{column['name'].replace('_', ' ')}: {column['description']} "
        f"Also called: {', '.join(column['synonyms'])}."
    )


def _canonical_vectors(canonical_columns: list[dict]) -> np.ndarray:
    texts = tuple(canonical_text(column) for column in canonical_columns)
    if texts not in _canonical_cache:
        _canonical_cache[texts] = encode(list(texts))
    return _canonical_cache[texts]


def match(
    file_columns: list[str],
    canonical_columns: list[dict],
    *,
    taken_fields: set[str] | None = None,
    taken_columns: set[str] | None = None,
    column_kinds: dict[str, str] | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, tuple[str, float]]:
    """Match leftover headers to leftover canonical fields by meaning.

    Returns {canonical_name: (file_column, score)}. Assignment is greedy from
    the highest score down, so the mapping stays one-to-one.
    """
    taken_fields = set(taken_fields or ())
    taken_columns = set(taken_columns or ())

    candidates = [
        column
        for column in file_columns
        if column not in taken_columns and not looks_like_identifier(column)
    ]
    if not candidates:
        return {}

    # Score against every canonical field, including ones already filled, so a
    # header is judged on what it actually resembles rather than on what
    # happens to be left.
    field_vectors = _canonical_vectors(canonical_columns)
    header_vectors = encode([humanize(column) for column in candidates])
    scores = header_vectors @ field_vectors.T

    # Each header proposes only its single best field. Without this, a header
    # whose best field is already spoken for slides onto its second choice:
    # olist has both product_category_name and product_category_name_english,
    # both plainly categories, and the runner-up would land on product_name at
    # 0.741. A consolation prize is not evidence.
    kinds = column_kinds or {}
    proposals = []
    for row, column in enumerate(candidates):
        best = int(np.argmax(scores[row]))
        field = canonical_columns[best]
        if field["name"] in taken_fields:
            continue
        if type_conflicts(field["type"], kinds.get(column)):
            continue
        proposals.append((float(scores[row][best]), column, field["name"]))

    matched: dict[str, tuple[str, float]] = {}
    for score, column, field in sorted(proposals, key=lambda item: -item[0]):
        if score < threshold or field in matched:
            continue
        matched[field] = (column, score)

    return matched
