"""Talk to a ModelChef vLLM deployment through its authenticated HTTPS API."""

import json
import os
import ssl
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def completion(endpoint, model, prompt, *, api_key_env='MODELCHEF_API_KEY',
               max_tokens=64, temperature=0.7, ca_cert=None):
    if not endpoint.startswith('https://'):
        raise ValueError('Remote chat requires an HTTPS endpoint')
    if not prompt:
        raise ValueError('Prompt is required')
    if not 1 <= max_tokens <= 1024 or not 0 < temperature <= 2:
        raise ValueError('max_tokens must be 1-1024 and temperature must be in (0, 2]')
    token = os.environ.get(api_key_env, '')
    if len(token) < 32 or any(character.isspace() for character in token):
        raise ValueError(f'Set {api_key_env} to the serving API key')
    base = endpoint.rstrip('/')
    context = ssl.create_default_context(cafile=str(ca_cert) if ca_cert else None)
    if not model or model == 'auto':
        models_request = Request(base + '/models', method='GET', headers={
            'Authorization': 'Bearer ' + token})
        try:
            with urlopen(models_request, timeout=30, context=context) as response:
                available = json.load(response)['data']
        except HTTPError as error:
            raise ValueError(f'Serving API returned HTTP {error.code}') from None
        except URLError as error:
            raise ValueError(f'Cannot reach the serving API: {error.reason}') from None
        if len(available) != 1 or not available[0].get('id'):
            raise ValueError('Select a model explicitly with --model')
        model = available[0]['id']
    url = base + '/completions'
    body = json.dumps({'model': model, 'prompt': prompt, 'max_tokens': max_tokens,
                       'temperature': temperature}).encode('utf-8')
    request = Request(url, data=body, method='POST', headers={
        'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=120, context=context) as response:
            result = json.load(response)
    except HTTPError as error:
        raise ValueError(f'Serving API returned HTTP {error.code}') from None
    except URLError as error:
        raise ValueError(f'Cannot reach the serving API: {error.reason}') from None
    choices = result.get('choices', [])
    if not choices or not isinstance(choices[0].get('text'), str):
        raise ValueError('Serving API returned no completion text')
    return choices[0]['text']


def chat(endpoint, model='auto', *, api_key_env='MODELCHEF_API_KEY', prompt=None,
         max_tokens=64, temperature=0.7, ca_cert=None):
    def ask(message):
        return completion(endpoint, model, message, api_key_env=api_key_env,
                          max_tokens=max_tokens, temperature=temperature, ca_cert=ca_cert)

    if prompt is not None:
        print(ask(prompt), flush=True)
        return
    print('Connected to ModelChef serving. Enter a prompt; /exit quits.', flush=True)
    while True:
        try:
            message = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if message == '/exit':
            return
        if message:
            print('Model: ' + ask(message), flush=True)
