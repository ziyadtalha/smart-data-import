# Test data

These files are **not committed** — they are large and carry their own upstream
licences, so `.gitignore` excludes them. Download them here if you want to run
the `realdata` test suite; tests skip cleanly when a file is missing.

| File | Source |
| --- | --- |
| `simulated_pos_data_with_seasonal_trends.csv` | [POS Data — Simulated Restaurant Data](https://www.kaggle.com/datasets/ganeshabbitota/pos-data-simulated-restaurant-data) |
| `olist.sqlite` | [E-Commerce Dataset by Olist as an SQLite Database](https://www.kaggle.com/datasets/terencicp/e-commerce-dataset-by-olist-as-an-sqlite-database) |
| `pos_transactions.csv`, `pos_operator_logs.csv` | [Open Source Point of Sale Dataset](https://www.kaggle.com/datasets/mehdiakoudadd/open-source-point-of-sale-dataset) |
| `sqlite-sakila.db` | Sakila sample database (MySQL's sample schema, ported to SQLite) |

## Notes

**The Open Source POS download ships a `data-04-00067 (1).xml` that is not
data.** It is the JATS-encoded journal article describing the dataset (MDPI
*Data*, DOI 10.3390/data4020067). The two CSVs are the whole dataset. The app
does not read XML.

**`amount` means the basket total in `pos_transactions.csv`** — the first row
pairs `basket_size=23` with `amount=112.71`. That file is why `amount` is a
synonym of `total_price` rather than `unit_price`. Sakila's `payment.amount`
agrees.

**`sqlite-sakila.db` is the auto-join's other end of the spectrum.** It declares
real foreign keys where olist declares none, and it has 21 tables, so the join
hits the `MAX_AUTO_JOIN_TABLES` ceiling. The planner ignores declared keys in
both cases and reaches the same answer from uniqueness and match rate alone:
starting at `payment` it walks out to `customer` and `address` for email and
phone, taking the mapping from 2 fields to 4 with all 16,049 payments and their
67,416.51 total intact.

**`olist.sqlite` uploads directly.** The endpoints accept `.sqlite`/`.db`, list
the tables they find, and map one table at a time. With no `table` given, the
app picks the table whose columns match the most canonical fields — for olist
that is `order_items`, chosen over `geolocation` despite the latter having ten
times the rows.

**Olist spreads a sale across several tables**, so no single one fills the
canonical schema — `order_items` alone maps 2 fields. Tick **Pull in related
tables** (`auto_join=true`) and it widens to 30 columns via `orders`,
`products`, `sellers`, `customers` and the category translation, which takes the
AI-assisted mapping from 2 to 5 fields with the row count and revenue total
unchanged.

Two of its tables are traps that the join rules exist to refuse:
`order_payments` has several payments per order, so joining it inflates revenue
by 5%; `leads_closed` shares `seller_id` with `order_items` but matches only
4.5% of it and describes seller acquisition rather than sales.
`tests/test_real_data.py` asserts both stay out.

**Neither dataset fills the canonical schema completely.** Olist is anonymized,
so it has no customer name, email, or phone; the restaurant POS file has no
order identifier or email. Fields left unmapped in those tests are genuinely
absent from the source, not mapping failures.

## Running the tests

```bash
pytest                    # everything; realdata tests skip if files are absent
pytest -m realdata        # only the tests that use these files
pytest -m "not realdata"  # skip them entirely
```
