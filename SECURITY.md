# Security and deployment boundaries

ModelChef is an experimental model-development platform. The built-in API controls provide TLS, a shared bearer token and request limits; they do not provide per-user identity, distributed quotas or protection against network-level denial of service. Review a deployment before exposing it publicly. Keep vLLM behind a private network or a gateway when stronger controls are needed.

## Report a vulnerability

Use the repository's **Security → Advisories → Report a vulnerability** option when available. If private reporting is unavailable, open an issue requesting a private contact without posting an exploit, credential or affected account details. Never include API keys, AWS credentials, TLS private keys or sensitive model/data examples in public reports.

## Protect model inputs and credentials

- Store AWS credentials in the standard external credential store and API tokens in the configured environment variable or AWS Secrets Manager. Keep secrets outside recipes and source files.
- PII filtering applies to processed training inputs. Raw run snapshots and dataset caches retain their original text. Use appropriate local access permissions, retention and deletion policies for sensitive inputs.
- Pretrained loading disables remote model code and implicit model downloads. Local checkpoints and CUDA plugins still require trusted sources; selecting a plugin executes its Python/native code.
- Temporary AWS training has termination deadlines and a recovery ledger. Persistent serving deliberately stays running until stopped or deleted. Budget estimates and timeout mechanisms cannot guarantee an exact cloud bill during external failures.

The test suite exercises security validation and simulated cloud lifecycle failures. It does not certify a live public deployment or a model's behavior under adversarial prompts.
