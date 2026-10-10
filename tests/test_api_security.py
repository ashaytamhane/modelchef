import asyncio
import json

import pytest

from modelchef.api_security import SecurityMiddleware


def test_security_boundary(monkeypatch):
    token = 'x' * 48
    monkeypatch.setenv('TEST_KEY', token)
    monkeypatch.setenv('MODELCHEF_API_SECURITY', json.dumps({
        'api_key_env': 'TEST_KEY', 'requests_per_minute': 20,
        'max_concurrent_requests': 1, 'max_body_bytes': 100,
        'request_timeout_seconds': 1}))
    calls = []

    async def app(scope, receive, send):
        calls.append(await receive())
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await send({'type': 'http.response.body', 'body': b'ok'})

    middleware = SecurityMiddleware(app)

    async def request(path='/v1/completions', auth=True, body=b'{"prompt":"Hi"}'):
        output = []
        async def receive():
            return {'type': 'http.request', 'body': body}
        async def send(message):
            output.append(message)
        await middleware({'type': 'http', 'method': 'POST', 'path': path,
                          'headers': [(b'authorization', ('Bearer ' + token).encode())] if auth else []}, receive, send)
        return output[0]['status']

    assert asyncio.run(request(auth=False)) == 401
    assert asyncio.run(request('/invocations')) == 404
    assert asyncio.run(request(body=b'x' * 101)) == 413
    assert asyncio.run(request(body=b'{"messages":[{"content":[{"image_url":"http://private"}]}]}')) == 400
    assert not calls
    assert asyncio.run(request()) == 200
    middleware.active = 1
    assert asyncio.run(request()) == 429
    middleware.active = 0
    middleware.settings['requests_per_minute'] = 1
    assert asyncio.run(request()) == 429


def test_weak_secret_rejected(monkeypatch):
    monkeypatch.setenv('MODELCHEF_API_SECURITY', '{"api_key_env":"TEST_KEY"}')
    monkeypatch.setenv('TEST_KEY', 'weak')
    with pytest.raises(ValueError, match='32'):
        SecurityMiddleware(None)


def test_timeout_releases_capacity(monkeypatch):
    monkeypatch.setenv('TEST_KEY', 'x' * 48)
    monkeypatch.setenv('MODELCHEF_API_SECURITY', json.dumps({
        'api_key_env': 'TEST_KEY', 'requests_per_minute': 20,
        'max_concurrent_requests': 1, 'max_body_bytes': 100,
        'request_timeout_seconds': 0.01}))
    async def app(scope, receive, send):
        await asyncio.sleep(1)
    middleware = SecurityMiddleware(app)
    output = []
    async def receive():
        return {'type': 'http.request', 'body': b'{"prompt":"Hi"}'}
    async def send(message):
        output.append(message)
    asyncio.run(middleware({'type': 'http', 'method': 'POST', 'path': '/v1/completions',
                           'headers': [(b'authorization', b'Bearer ' + b'x' * 48)]}, receive, send))
    assert output[0]['status'] == 504
    assert middleware.active == 0


@pytest.mark.parametrize('payload', [
    {'prompt': 'Hi', 'n': 1000},
    {'prompt': 'Hi', 'n': True},
    {'prompt': ['Hi', 'again']},
    {'prompt': []},
    {'prompt': 'Hi', 'max_tokens': 1000000},
    {'prompt': 'Hi', 'max_tokens': None},
    {'prompt': 'Hi', 'max_tokens': True},
    {'prompt': 'Hi', 'max_tokens': 1, 'max_completion_tokens': 1},
    {'messages': [{'role': 'user', 'content': 'Hi', 'tool_calls': []}]},
])
def test_generation_abuse_is_rejected_before_engine(monkeypatch, payload):
    monkeypatch.setenv('TEST_KEY', 'x' * 48)
    monkeypatch.setenv('MODELCHEF_API_SECURITY', json.dumps({
        'api_key_env': 'TEST_KEY', 'requests_per_minute': 20,
        'max_concurrent_requests': 1, 'max_body_bytes': 1024,
        'request_timeout_seconds': 1}))
    async def app(scope, receive, send):
        pytest.fail('An oversized generation must not reach vLLM')
    async def receive():
        return {'type': 'http.request', 'body': json.dumps(payload).encode()}
    output = []
    async def send(message):
        output.append(message)
    asyncio.run(SecurityMiddleware(app)({'type': 'http', 'method': 'POST',
        'path': '/v1/chat/completions' if 'messages' in payload else '/v1/completions',
        'headers': [(b'authorization', b'Bearer ' + b'x' * 48)]}, receive, send))
    assert output[0]['status'] == 400


def test_missing_generation_limit_is_bounded_and_nonfinite_json_rejected(monkeypatch):
    monkeypatch.setenv('TEST_KEY', 'x' * 48)
    monkeypatch.setenv('MODELCHEF_API_SECURITY', json.dumps({
        'api_key_env': 'TEST_KEY', 'requests_per_minute': 20,
        'max_concurrent_requests': 1, 'max_body_bytes': 1024,
        'request_timeout_seconds': 1, 'max_output_tokens': 32}))
    received = []
    async def app(scope, receive, send):
        received.append(json.loads((await receive())['body']))
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await send({'type': 'http.response.body', 'body': b'ok'})
    middleware = SecurityMiddleware(app)
    async def request(body):
        output = []
        async def receive():
            return {'type': 'http.request', 'body': body}
        async def send(message):
            output.append(message)
        await middleware({'type': 'http', 'method': 'POST', 'path': '/v1/completions',
            'headers': [(b'authorization', b'Bearer ' + b'x' * 48)]}, receive, send)
        return output[0]['status']
    assert asyncio.run(request(b'{"prompt":"Hi"}')) == 200
    assert received[0]['max_tokens'] == 32
    assert asyncio.run(request(b'{"prompt":"Hi","temperature":NaN}')) == 400
    assert len(received) == 1
