import io
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from modelchef.corpus import viewer_headers, viewer_page


def test_optional_token_only_appears_in_request_headers(monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'hf_test_value')
    assert viewer_headers()['Authorization'] == 'Bearer hf_test_value'
    monkeypatch.setenv('HF_TOKEN', 'invalid token')
    with pytest.raises(ValueError, match='whitespace') as error:
        viewer_headers()
    assert 'invalid token' not in str(error.value)


def test_rate_limit_retries_same_page_and_honors_retry_after(monkeypatch):
    request = Request('https://datasets-server.huggingface.co/rows')
    calls, pauses, messages = [], [], []

    def open_page(actual, timeout):
        calls.append(actual)
        if len(calls) == 1:
            raise HTTPError(actual.full_url, 429, 'Too Many Requests', {'Retry-After': '12'}, io.BytesIO())
        return io.BytesIO(b'page')

    monkeypatch.setattr('modelchef.corpus.urlopen', open_page)
    monkeypatch.setattr('modelchef.corpus.time.sleep', pauses.append)
    assert viewer_page(request, 30, 10, messages.append) == b'page'
    assert calls == [request, request]
    assert pauses == [12]
    assert 'retrying the same page' in messages[0]


def test_retries_are_bounded_and_permanent_errors_are_not_retried(monkeypatch):
    request = Request('https://datasets-server.huggingface.co/rows')
    calls = []

    def limited(actual, timeout):
        calls.append(actual)
        raise HTTPError(actual.full_url, 429, 'Too Many Requests', {}, io.BytesIO())

    monkeypatch.setattr('modelchef.corpus.urlopen', limited)
    monkeypatch.setattr('modelchef.corpus.time.sleep', lambda seconds: None)
    with pytest.raises(HTTPError):
        viewer_page(request, 30, 10, lambda message: None)
    assert len(calls) == 7

    def denied(actual, timeout):
        raise HTTPError(actual.full_url, 401, 'Unauthorized', {}, io.BytesIO())

    monkeypatch.setattr('modelchef.corpus.urlopen', denied)
    monkeypatch.setattr('modelchef.corpus.time.sleep', lambda seconds: pytest.fail('Permanent errors must not sleep'))
    with pytest.raises(HTTPError, match='401'):
        viewer_page(request, 30, 10, lambda message: None)


def test_long_server_cooldown_is_not_ignored(monkeypatch):
    def limited(request, timeout):
        raise HTTPError(request.full_url, 429, 'Too Many Requests', {'Retry-After': '600'}, io.BytesIO())

    monkeypatch.setattr('modelchef.corpus.urlopen', limited)
    monkeypatch.setattr('modelchef.corpus.time.sleep', lambda seconds: pytest.fail('Cooldown exceeds retry budget'))
    with pytest.raises(ValueError, match='cooldown'):
        viewer_page(Request('https://datasets-server.huggingface.co/rows'), 30, 10, lambda message: None)
