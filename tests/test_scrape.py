import pytest
from unittest.mock import patch, MagicMock
from mbfc_scrape.scrape import check_response_ok, get_url_domain, get_matched_text


# Making a mock response to test the "check_response_ok" function
def test_check_response_ok():
    mock_r = MagicMock()
    false_response = mock_r.status_code != 200 & "html" not in mock_r.headers['content-type']
    assert check_response_ok(false_response) is False

    true_response = mock_r.status_code == 200 & "html" in mock_r.headers['content-type']
    assert check_response_ok(true_response) is True

# Making a mock url example to test the "get_url_domain" function
def test_get_url_domain():
    assert get_url_domain("https://www.news.org") == "news.org"
    assert get_url_domain("invalid-url") is None

# Making a mock match case to test the "get_matched_text" function
def test_get_matched_text():
    mock_match = MagicMock()
    mock_match.group.return_value = "  Mock Match Case "
    assert get_matched_text(mock_match) == "mock match case"
    assert get_matched_text(None) == ""

# Making a mock response for mock request to test "scrape.py" file
@patch("mbfc_scrape.scrape.requests.get")
def test_scrape_file(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "html"}
    mock_response.text = """
        <table id="mbfc-table">
            <a href="https://www.news.org">Mock Article Source</a>
        </table>
    """
    mock_get.return_value = mock_response