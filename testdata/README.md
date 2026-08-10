# Test data

These files are **not committed** — they are large and carry their own upstream
licences, so `.gitignore` excludes them. Download them here if you want to run
the `realdata` test suite; tests skip cleanly when a file is missing.

| File | Source |
| --- | --- |
| `simulated_pos_data_with_seasonal_trends.csv` | [POS Data — Simulated Restaurant Data](https://www.kaggle.com/datasets/ganeshabbitota/pos-data-simulated-restaurant-data) |
| `olist.sqlite` | [E-Commerce Dataset by Olist as an SQLite Database](https://www.kaggle.com/datasets/terencicp/e-commerce-dataset-by-olist-as-an-sqlite-database) |
| _(not yet downloaded)_ | [Open Source Point of Sale Dataset](https://www.kaggle.com/datasets/mehdiakoudadd/open-source-point-of-sale-dataset) |

## Notes

**`olist.sqlite` cannot be uploaded directly.** The app reads CSV/XLSX/XLS only,
so a `.sqlite` upload is rejected with a 400. It also needs joins before it
looks anything like the canonical schema — orders, customers, order_items and
the category translation table all live separately. `tests/test_real_data.py`
exports a joined slice to CSV in memory and feeds that through the endpoints.

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
