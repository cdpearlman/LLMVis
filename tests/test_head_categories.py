"""Test that head_categories.json is correctly parsed into category stores."""
import json
import os
import pytest


@pytest.fixture
def head_categories():
    json_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'head_categories.json')
    with open(json_path, 'r') as f:
        return json.load(f)


def build_categories_store(head_categories_data):
    """Replicate the category store construction from app.py."""
    categories_store = {}
    for cat_key, cat_data in head_categories_data['categories'].items():
        categories_store[cat_key] = {
            'display_name': cat_data.get('display_name', cat_key),
            'heads': [{'layer': h['layer'], 'head': h['head']}
                      for h in cat_data.get('top_heads', [])]
        }
    return categories_store


class TestHeadCategoriesStore:
    def test_all_categories_have_heads(self, head_categories):
        """Every category in the JSON must produce a non-empty heads list."""
        for model_key, model_data in head_categories.items():
            if 'categories' not in model_data:
                continue
            store = build_categories_store(model_data)
            for cat_key, cat_val in store.items():
                assert len(cat_val['heads']) > 0, (
                    f"Category '{cat_key}' in model '{model_key}' has no heads — "
                    f"check that the JSON field name matches the parsing code"
                )

    def test_heads_have_layer_and_head_keys(self, head_categories):
        """Each head entry must have 'layer' and 'head' integer fields."""
        for model_key, model_data in head_categories.items():
            if 'categories' not in model_data:
                continue
            store = build_categories_store(model_data)
            for cat_key, cat_val in store.items():
                for h in cat_val['heads']:
                    assert 'layer' in h and 'head' in h, (
                        f"Head entry in '{cat_key}' missing layer/head keys"
                    )
                    assert isinstance(h['layer'], int)
                    assert isinstance(h['head'], int)

    def test_field_name_is_top_heads_not_heads(self, head_categories):
        """Guard against regression: JSON field must be 'top_heads', not 'heads'."""
        for model_key, model_data in head_categories.items():
            if 'categories' not in model_data:
                continue
            for cat_key, cat_data in model_data['categories'].items():
                assert 'top_heads' in cat_data, (
                    f"Category '{cat_key}' uses wrong field name — expected 'top_heads'"
                )
                # 'heads' should NOT exist as a field in the JSON
                assert 'heads' not in cat_data, (
                    f"Category '{cat_key}' has ambiguous 'heads' field — use 'top_heads' only"
                )
