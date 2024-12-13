import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from newspaper import Article
from db_access_module.db_access import get_gkg_data, get_base_url, fetch_article_content


mock_df = pd.DataFrame({
    "DocumentIdentifier": ["http://news.org/article0", "http://news.org/article1"],
    "V2Themes": ["PALE; ISR", "RUSS; UKR"],
    "Date": ["20220101", "20240101"],
})

mock_mbfc = pd.DataFrame({
    "source": ["news.org"]
})

@patch("db_access_module.db_access.client.query")
def test_getting_gkg_data(some_query):
    mock_of_query = MagicMock()
    some_query.return_value = mock_of_query
    mock_of_query.result.return_value.to_arrow.return_value.to_pandas.return_value = mock_df

    df = get_gkg_data(min_date="20220101", max_date="20240101")
    pd.testing.assert_frame_equal(df, mock_df)

def test_getting_base_url():
    url = "http://www.news.org/article1"
    expected_base = "news.org"
    assert get_base_url(url) == expected_base

@patch("db_access_module.db_access.Article")
def test_fetching_articles_content(url):
    mock_article = Article(url)
    mock_instance = mock_article.return_value
    mock_instance.download.return_value = None
    mock_instance.parse.return_value = None
    mock_instance.text = "Mock content"

    # mock df
    df = pd.DataFrame({"DocumentIdentifier": ["http://www.news.org/article1"]})

    content_df = fetch_article_content(df, "DocumentIdentifier", "content")

    assert "content" in content_df.columns
    assert content_df["content"].iloc[0] == "Mock content"