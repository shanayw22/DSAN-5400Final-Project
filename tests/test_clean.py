import pytest
import pandas as pd
from unittest.mock import patch
from mbfc_scrape.clean import get_norm_bias_label, get_norm_factual_label


# Making mock dataframe to test with functions in "clean.py"
# valid_labels = ["right", "high", "left", "center", "mixed"]
# pd.DataFrame({"label": valid_labels})

# testing "get_norm_bias_label" function with set inputs
def test_get_norm_bias():
    assert get_norm_bias_label("pro-science") == "neutral"
    assert get_norm_bias_label("right-center") == "right-center"
    assert get_norm_bias_label("left-center") == "left-center"

# testing "get_norm_factual_label" function with set inputs
def test_get_norm_factual():
    assert get_norm_factual_label("high") == "high"
    assert get_norm_factual_label("low") == "low"
    assert get_norm_factual_label("mostly factual") == "mixed"