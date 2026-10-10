"""Fail-closed ASGI boundary for a single vLLM API process; secrets stay in environment."""
import asyncio
import hmac
import json
import os
import time
from collections import deque


class SecurityMiddleware:
    def __init__(self, app):
        self.app = app
        self.settings = json.loads(os.environ['MODELCHEF_API_SECURITY'])
        self.secret = os.environ[self.settings['api_key_env']]
        if len(self.secret) < 32 or any(c.isspace() for c in self.secret):
            raise ValueError('API token must contain at least 32 non-whitespace characters')
        self.arrivals = deque()
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            if scope['type'] == 'websocket':
                await send({'type': 'websocket.close', 'code': 1008})
                return
            return await self.app(scope, receive, send)

        async def reject(status):
            await send({'type': 'http.response.start', 'status': status,
                        'headers': [(b'content-type', b'application/json'),
                                    (b'cache-control', b'no-store')]})
            await send({'type': 'http.response.body', 'body': b'{"error":"Request rejected"}'})

        headers = scope.get('headers', [])
        auth = [v for k, v in headers if k.lower() == b'authorization']
        if len(auth) != 1 or not hmac.compare_digest(auth[0], ('Bearer ' + self.secret).encode()):
            return await reject(401)
        # Exact routes: block admin, metrics, alternate inference and URL-fetching APIs.
        routes = {'/v1/models': 'GET', '/v1/completions': 'POST', '/v1/chat/completions': 'POST'}
        if routes.get(scope['path']) != scope['method']:
            return await reject(404)
        now = time.monotonic()
        while self.arrivals and self.arrivals[0] <= now - 60:
            self.arrivals.popleft()
        if len(self.arrivals) >= self.settings['requests_per_minute'] or self.active >= self.settings['max_concurrent_requests']:
            return await reject(429)
        self.arrivals.append(now)
        self.active += 1
        started = False

        async def guarded_send(message):
            nonlocal started
            if message['type'] == 'http.response.start':
                started = True
                message = dict(message)
                message['headers'] = list(message.get('headers', [])) + [
                    (b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff')]
            await send(message)

        try:
            async with asyncio.timeout(self.settings['request_timeout_seconds']):
                body = bytearray()
                while True:
                    message = await receive()
                    if message['type'] == 'http.disconnect':
                        return
                    body.extend(message.get('body', b''))
                    if len(body) > self.settings['max_body_bytes']:
                        return await reject(413)
                    if not message.get('more_body', False):
                        break
                if scope['method'] == 'POST':
                    try:
                        payload = json.loads(body)
                    except (ValueError, UnicodeDecodeError):
                        return await reject(400)
                    if not isinstance(payload, dict):
                        return await reject(400)
                    allowed = {'model', 'prompt', 'messages', 'max_tokens', 'max_completion_tokens',
                               'temperature', 'top_p', 'top_k', 'stream', 'stream_options',
                               'stop', 'seed', 'frequency_penalty', 'presence_penalty', 'n'}
                    if set(payload) - allowed:
                        return await reject(400)
                    if scope['path'] == '/v1/completions':
                        prompt = payload.get('prompt')
                        if not isinstance(prompt, str) and not (isinstance(prompt, list) and
                                all(isinstance(p, str) for p in prompt)):
                            return await reject(400)
                    # Text-only boundary prevents user-controlled multimodal URL/file fetching.
                    messages = payload.get('messages', [])
                    if not isinstance(messages, list) or any(not isinstance(m, dict) or
                            not isinstance(m.get('content'), str) for m in messages):
                        return await reject(400)
                delivered = False

                async def replay():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                    return await receive()

                await self.app(scope, replay, guarded_send)
        except TimeoutError:
            if not started:
                await reject(504)
            else:
                await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
        finally:
            self.active -= 1
