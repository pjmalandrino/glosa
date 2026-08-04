"""Domain layer — pure logic over the ports.

No I/O, no HTTP client, and no knowledge of how a document is serialized.
Anything in here can be exercised with nothing but value objects and a fake
`ChatModel`. Enforced by `tests/test_architecture.py`.
"""
