from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from newspaper import Article
from db_access_module.db_access import get_gkg_data, get_base_url, fetch_article_content


# Making mock dataframe to test with functions in "db_access.py"
mock_df = pd.DataFrame({
    "DocumentIdentifier": ["http://news.org/article0", "http://news.org/article1"],
    "V2Themes": ["PALE; ISR", "RUSS; UKR"],
    "Date": ["20220101", "20240101"],
})

mock_mbfc = pd.DataFrame({
    "source": ["news.org"]
})

# Mock Google- client instance ; from Chat- GPT: how to test with seraphic file
@pytest.fixture
def mock_bigquery_client():
    with patch("google.cloud.bigquery.Client.from_service_account_json") as mock_from_service_account_json:
        mock_client_instance = MagicMock()
        mock_from_service_account_json.return_value = mock_client_instance 
        yield mock_client_instance


# Mock query to test "get_gkg_data" function
def test_getting_gkg_data(mock_bigquery_client):
    mock_bigquery_client.query.return_value.result.return_value = [
        {"key": "value"}
    ]
    df = get_gkg_data(min_date="20220101", max_date="20240101")
    mock_bigquery_client.query.assert_called_once()
    assert df == [{"key": "value"}]

# Mock url to extract url for "get_base_url" function
def test_getting_base_url():
    url = "http://www.news.org/article1"
    expected_base = "news.org"
    assert get_base_url(url) == expected_base

# Mock url to make mock dataframe column in "fetch_article_content" function
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