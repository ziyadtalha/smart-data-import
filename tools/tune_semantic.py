"""Calibration harness for stage 2's embedding matcher.

Stage 2's threshold and the canonical descriptions were both chosen by
measurement rather than taste. This is the measurement. It runs the real
pipeline -- rules, then semantic.match -- over hand-labelled headers drawn from
every table in testdata/, sweeps the threshold, and reports how many correct
matches survive at each cut and what the worst wrong match is.

    python tools/tune_semantic.py                # sweep the threshold
    python tools/tune_semantic.py --detail 0.67  # what happens at one cut

The number to optimize is the count of correct matches at the *highest* cut
that still admits zero wrong ones. Rewriting the descriptions in April took
that from 5 to 9 and lowered the cut from 0.68 to 0.67.

Requires the datasets in testdata/ (git-ignored -- see testdata/README.md) and
fastembed. Contexts whose file is missing are skipped, so a partial download
still gives a partial answer.
"""
import argparse
import copy
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

import main
import semantic

TESTDATA = ROOT / "testdata"

# Genuinely ambiguous headers. Scored neither way, but listed rather than
# omitted so the exclusion is a decision instead of an oversight.
IGNORE = "__ignore__"

# Labels are keyed by context -- a CSV file, or a "database::table" -- because
# the right answer depends on it: `first_name` in sakila's `actor` table is not
# a customer name, and the same header in `customer` is. A label of None means
# the header must not be matched at all. Headers not listed are not scored.
LABELS = {
    "pos_operator_logs.csv": {
        "begin_date_time": "order_date",
    },
    "pos_transactions.csv": {
        "begin_date_time": "order_date",
        "end_date_time": IGNORE,   # the same transaction's close; either is defensible
        "basket_size": "quantity",
        "t_cash": None,
        "t_card": None,
    },
    "simulated_pos_data_with_seasonal_trends.csv": {
        "Timestamp": "order_date",
        "Item_Category": "category",
        "Payment_Method": None,
    },
    "olist.sqlite::customers": {
        "customer_zip_code_prefix": None,
        "customer_city": None,
        "customer_state": None,
    },
    "olist.sqlite::geolocation": {
        "geolocation_lat": None,
        "geolocation_lng": None,
        "geolocation_city": None,
        "geolocation_state": None,
    },
    "olist.sqlite::leads_closed": {
        "won_date": IGNORE,        # a deal closed, not an order placed
        "business_segment": None,
        "lead_type": None,
        "lead_behaviour_profile": None,
        "has_company": None,
        "has_gtin": None,
        "average_stock": None,
        "business_type": None,
        "declared_product_catalog_size": None,
        "declared_monthly_revenue": None,   # revenue, but not a transaction total
    },
    "olist.sqlite::leads_qualified": {
        "first_contact_date": None,
        "origin": None,
    },
    "olist.sqlite::order_items": {
        "shipping_limit_date": None,   # a shipping deadline, not the order date
        "freight_value": None,
    },
    "olist.sqlite::order_payments": {
        "payment_value": "total_price",
        "payment_sequential": None,
        "payment_type": None,
        "payment_installments": None,
    },
    "olist.sqlite::order_reviews": {
        "review_score": None,
        "review_comment_title": None,
        "review_comment_message": None,
        "review_creation_date": None,
        "review_answer_timestamp": None,
    },
    "olist.sqlite::orders": {
        "order_purchase_timestamp": "order_date",
        "order_status": None,
        "order_approved_at": IGNORE,   # close enough to the purchase to argue
        "order_delivered_carrier_date": None,
        "order_delivered_customer_date": None,
        "order_estimated_delivery_date": None,
    },
    "olist.sqlite::product_category_name_translation": {
        "product_category_name": "category",
        "product_category_name_english": "category",
    },
    "olist.sqlite::products": {
        "product_name_lenght": None,
        "product_description_lenght": None,
        "product_photos_qty": None,
        "product_weight_g": None,
        "product_length_cm": None,
        "product_height_cm": None,
        "product_width_cm": None,
    },
    "olist.sqlite::sellers": {
        "seller_zip_code_prefix": None,
        "seller_city": None,
        "seller_state": None,
    },
    "sqlite-sakila.db::actor": {
        "first_name": IGNORE,      # an actor, but the model cannot see the table
        "last_name": IGNORE,
        "last_update": None,
    },
    "sqlite-sakila.db::address": {
        "address": None,
        "address2": None,
        "district": None,
        "last_update": None,
    },
    "sqlite-sakila.db::city": {"city": None, "last_update": None},
    "sqlite-sakila.db::country": {"country": None, "last_update": None},
    "sqlite-sakila.db::customer": {
        "first_name": "customer_name",
        "last_name": IGNORE,       # only one of the pair can win the field
        "create_date": IGNORE,     # account creation, not an order
        "active": None,
        "last_update": None,
    },
    "sqlite-sakila.db::customer_list": {
        "name": "customer_name",
        "address": None,
        "city": None,
        "country": None,
        "notes": None,
    },
    "sqlite-sakila.db::film": {
        "title": IGNORE,           # the rented product, arguably product_name
        "description": None,
        "release_year": None,
        "rental_duration": None,
        "rental_rate": "unit_price",
        "length": None,
        "replacement_cost": None,  # a cost, not what was charged
        "rating": None,
        "special_features": None,
        "last_update": None,
    },
    "sqlite-sakila.db::film_list": {
        "title": IGNORE,
        "description": None,
        "length": None,
        "rating": None,
        "actors": None,
    },
    "sqlite-sakila.db::inventory": {"last_update": None},
    "sqlite-sakila.db::payment": {
        "payment_date": "order_date",
        "last_update": None,
    },
    "sqlite-sakila.db::rental": {
        "rental_date": "order_date",
        "return_date": None,
        "last_update": None,
    },
    "sqlite-sakila.db::sales_by_film_category": {
        "total_sales": "total_price",
    },
    "sqlite-sakila.db::sales_by_store": {
        "total_sales": "total_price",
        "store": None,
        "manager": None,
    },
    "sqlite-sakila.db::staff": {
        "first_name": IGNORE,
        "last_name": IGNORE,
        "email": IGNORE,           # a staff address, but indistinguishable
        "picture": None,
        "active": None,
        "username": None,
        "password": None,
        "last_update": None,
    },
    "sqlite-sakila.db::staff_list": {
        "name": IGNORE,
        "address": None,
        "city": None,
        "country": None,
    },
    "sqlite-sakila.db::store": {"last_update": None},
}


def load_contexts() -> dict[str, tuple[list[str], pd.DataFrame]]:
    """Read a sample of each labelled context, skipping absent files."""
    contexts = {}
    for name in LABELS:
        if "::" in name:
            database, table = name.split("::")
            path = TESTDATA / database
            if not path.exists():
                continue
            connection = sqlite3.connect(path)
            try:
                frame = pd.read_sql_query(
                    f"SELECT * FROM {main.quote_identifier(table)} LIMIT 50", connection
                )
            finally:
                connection.close()
        else:
            path = TESTDATA / name
            if not path.exists():
                continue
            frame = pd.read_csv(path, nrows=50)
        contexts[name] = (list(frame.columns), frame)
    return contexts


def run(canonical, contexts, threshold):
    """Score one set of canonical definitions at one threshold."""
    hits, misses, false_positives = [], [], []

    for context, (columns, frame) in contexts.items():
        labels = LABELS[context]
        rules = main.auto_map_columns(columns)

        matched = semantic.match(
            columns,
            canonical,
            taken_fields=set(rules),
            taken_columns=set(rules.values()),
            column_kinds=main.column_kinds(frame),
            threshold=threshold,
        )
        mapping = {field: column for field, (column, _) in matched.items()}
        scores = {field: score for field, (_, score) in matched.items()}

        # Several headers may be labelled with the same field -- olist has two
        # category columns -- and only one can win. Any of them winning counts.
        wanted: dict[str, set[str]] = {}
        for header, label in labels.items():
            if label not in (None, IGNORE):
                wanted.setdefault(label, set()).add(header)

        for field, candidates in wanted.items():
            if field in rules:
                continue  # stage 1 already had it; not stage 2's to win
            got = mapping.get(field)
            if got in candidates:
                hits.append((context, field, got, round(scores[field], 3)))
            else:
                misses.append((context, field, sorted(candidates), got))

        for field, column in mapping.items():
            label = labels.get(column, "__unlabelled__")
            if label is None:
                false_positives.append(
                    (context, column, field, round(scores[field], 3), "must not match")
                )
            elif label not in (IGNORE, "__unlabelled__") and label != field:
                false_positives.append(
                    (context, column, field, round(scores[field], 3),
                     f"belongs on {label}")
                )

    return hits, misses, false_positives


def report_detail(canonical, contexts, threshold):
    hits, misses, fps = run(canonical, contexts, threshold)
    print(f"--- threshold {threshold} ---")

    print(f"\nHITS ({len(hits)})")
    for context, field, column, score in sorted(hits, key=lambda row: -row[3]):
        print(f"  {score:.3f}  {column:<34} -> {field:<15} [{context}]")

    print(f"\nMISSES ({len(misses)})")
    for context, field, candidates, got in misses:
        note = f"went to {got}" if got else "nothing matched"
        print(f"         {'/'.join(candidates):<34} -> {field:<15} "
              f"({note}) [{context}]")

    print(f"\nFALSE POSITIVES ({len(fps)})")
    for context, column, field, score, why in sorted(fps, key=lambda row: -row[3]):
        print(f"  {score:.3f}  {column:<34} -> {field:<15} {why} [{context}]")


def report_sweep(canonical, contexts):
    print(f"{'thresh':>7} {'hits':>5} {'miss':>5} {'FP':>4}   worst false positive")
    best = (0, None)
    for step in range(500, 861, 5):
        threshold = step / 1000
        hits, misses, fps = run(canonical, contexts, threshold)
        worst = max(fps, key=lambda row: row[3], default=None)
        note = f"{worst[1]} -> {worst[2]} @ {worst[3]:.3f}" if worst else "-"
        print(f"{threshold:>7.3f} {len(hits):>5} {len(misses):>5} {len(fps):>4}   {note}")
        if not fps and len(hits) >= best[0]:
            best = (len(hits), threshold)

    hits_at, threshold = best
    if threshold is None:
        print("\nNo threshold admits zero false positives.")
        return
    print(f"\nBest clean cut: {threshold:.3f} with {hits_at} correct matches "
          f"(module is set to {semantic.SIMILARITY_THRESHOLD}).")


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=float, metavar="THRESHOLD",
                        help="explain one threshold instead of sweeping")
    args = parser.parse_args()

    if not semantic.is_available():
        sys.exit("fastembed is not installed; nothing to calibrate.")

    contexts = load_contexts()
    if not contexts:
        sys.exit(f"No datasets found in {TESTDATA}. See testdata/README.md.")

    scored = sum(
        1 for context in contexts
        for label in LABELS[context].values() if label is not IGNORE
    )
    print(f"{len(contexts)} contexts, {scored} labelled headers\n")

    canonical = copy.deepcopy(main.CANONICAL_COLUMNS)
    if args.detail is not None:
        report_detail(canonical, contexts, args.detail)
    else:
        report_sweep(canonical, contexts)


if __name__ == "__main__":
    main_cli()
