"""Tests for data preprocessing — CSV cleaning, label mapping, text normalisation."""

import csv
import io
import tempfile
from pathlib import Path

import pytest

from pos_classifier.data.preprocessing import (
    _clean_category,
    _clean_text,
    load_query_data,
    load_training_data,
)


TRAINING_CSV_CONTENT = """\
product_description,Category
cookies cakes Chocolate Sandwich Cookies,Dry Goods & Pantry Staples;;;;
Sparkling Orange Juice,Beverages;;;;
"Cheesecake, Chocolate Truffle",Fresh & Perishable Items;;;;
Dish Soap Lemon,Household & Personal Care;;;;
Craft Beer IPA,Specialty & Miscellaneous;;;;
Unknown Product,UNKNOWN_CATEGORY
,Beverages
"""

# 2 rows per class (10 valid) + 1 same-class duplicate + 1 cross-class duplicate
# → 12 raw rows, 10 unique after dedup (cross-class duplicate keeps first label)
TRAINING_CSV_LARGE = """\
product_description,Category
Apple Juice,Beverages
Orange Juice,Beverages
Milk,Fresh & Perishable Items
Eggs,Fresh & Perishable Items
Bread,Dry Goods & Pantry Staples
Pasta,Dry Goods & Pantry Staples
Shampoo,Household & Personal Care
Soap,Household & Personal Care
Hot Sauce,Specialty & Miscellaneous
Artisan Cheese,Specialty & Miscellaneous
Apple Juice,Beverages
Milk,Specialty & Miscellaneous
"""

QUERY_CSV_CONTENT = """\
product_description,HUMAN_VERIFIED_Category
Chocolate Sandwich Cookies,Dry Goods & Pantry Staples
"Pie, Apple",Fresh & Perishable Items
Organic Honey,
"""


@pytest.fixture()
def training_csv(tmp_path: Path) -> Path:
    p = tmp_path / "training.csv"
    p.write_text(TRAINING_CSV_CONTENT, encoding="utf-8")
    return p


@pytest.fixture()
def large_training_csv(tmp_path: Path) -> Path:
    p = tmp_path / "training_large.csv"
    p.write_text(TRAINING_CSV_LARGE, encoding="utf-8")
    return p


@pytest.fixture()
def query_csv(tmp_path: Path) -> Path:
    p = tmp_path / "query.csv"
    p.write_text(QUERY_CSV_CONTENT, encoding="utf-8")
    return p


class TestCleaning:
    def test_strip_semicolons(self):
        assert _clean_category("Beverages;;;;") == "Beverages"

    def test_strip_quote_and_semicolons(self):
        assert _clean_category('"Dry Goods & Pantry Staples";;;;') == "Dry Goods & Pantry Staples"

    def test_nkfc_normalisation(self):
        # non-breaking hyphen → regular hyphen after NFKC
        raw = "1‑Ply"
        cleaned = _clean_text(raw)
        assert "‑" not in cleaned

    def test_empty_string(self):
        assert _clean_category("") == ""
        assert _clean_category("   ") == ""


class TestLoadTrainingData:
    def test_loads_valid_rows(self, training_csv):
        texts, labels = load_training_data(training_csv)
        assert len(texts) == 5  # 5 valid rows, 1 unknown cat, 1 empty text

    def test_label_values_in_range(self, training_csv):
        _, labels = load_training_data(training_csv)
        assert all(0 <= l <= 4 for l in labels)

    def test_subset_parameter(self, training_csv):
        texts, labels = load_training_data(training_csv, subset=2)
        assert len(texts) == 2
        assert len(labels) == 2

    def test_comma_in_description(self, training_csv):
        texts, _ = load_training_data(training_csv)
        # "Cheesecake, Chocolate Truffle" should survive as full description
        assert any("Cheesecake" in t for t in texts)

    def test_skips_unknown_category(self, training_csv):
        texts, _ = load_training_data(training_csv)
        assert not any("Unknown Product" in t for t in texts)

    def test_skips_empty_description(self, training_csv):
        texts, _ = load_training_data(training_csv)
        assert "" not in texts


class TestDeduplication:
    def test_same_class_duplicate_removed(self, large_training_csv):
        # "Apple Juice,Beverages" appears twice — only 1 should survive
        texts, labels = load_training_data(large_training_csv)
        assert texts.count("Apple Juice") == 1

    def test_cross_class_duplicate_keeps_first_label(self, large_training_csv):
        # "Milk" first appears as Fresh & Perishable Items (label 1),
        # then as Specialty & Miscellaneous (label 4) — first label wins
        from pos_classifier.config import LABEL_MAP
        texts, labels = load_training_data(large_training_csv)
        idx = texts.index("Milk")
        assert labels[idx] == LABEL_MAP["Fresh & Perishable Items"]

    def test_total_rows_after_dedup(self, large_training_csv):
        # 12 raw rows − 2 duplicates = 10 unique
        texts, labels = load_training_data(large_training_csv)
        assert len(texts) == 10

    def test_no_duplicate_descriptions_in_output(self, large_training_csv):
        texts, _ = load_training_data(large_training_csv)
        assert len(texts) == len(set(texts))


class TestStratifiedSubset:
    def test_subset_size(self, large_training_csv):
        texts, labels = load_training_data(large_training_csv, subset=6)
        assert len(texts) == 6

    def test_all_classes_represented(self, large_training_csv):
        # 10 rows, 2 per class, subset=5 — stratification should give 1 per class
        from pos_classifier.config import NUM_LABELS
        _, labels = load_training_data(large_training_csv, subset=5)
        assert len(set(labels)) == NUM_LABELS

    def test_fallback_when_unstratifiable(self, training_csv):
        # training_csv has 1 row per class — stratification impossible, must not raise
        texts, labels = load_training_data(training_csv, subset=2)
        assert len(texts) == 2
        assert len(labels) == 2


class TestLoadQueryData:
    def test_loads_rows(self, query_csv):
        texts, labels = load_query_data(query_csv)
        assert len(texts) == 3

    def test_verified_label_present(self, query_csv):
        _, labels = load_query_data(query_csv)
        assert labels[0] is not None      # "Dry Goods & Pantry Staples"
        assert labels[1] is not None      # "Fresh & Perishable Items"

    def test_missing_label_is_none(self, query_csv):
        _, labels = load_query_data(query_csv)
        assert labels[2] is None          # "Organic Honey" has no label

    def test_comma_in_description(self, query_csv):
        texts, _ = load_query_data(query_csv)
        assert any("Pie" in t for t in texts)
