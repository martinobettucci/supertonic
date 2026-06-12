from __future__ import annotations

import pytest

from benchmark_api import ensure_local_url
from serve_api import ensure_loopback


def test_loopback_guards_accept_localhost_addresses():
    for host in ("localhost", "127.0.0.1", "::1"):
        ensure_loopback(host)
    for url in (
        "http://localhost:8765",
        "http://127.0.0.1:8765",
        "http://[::1]:8765",
    ):
        ensure_local_url(url)


def test_loopback_guards_reject_external_or_deceptive_addresses():
    for host in ("0.0.0.0", "192.0.2.1", "example.com"):
        with pytest.raises(SystemExit):
            ensure_loopback(host)
    for url in (
        "https://127.0.0.1:8765",
        "http://0.0.0.0:8765",
        "http://127.0.0.1.example:8765",
        "http://example.com:8765",
    ):
        with pytest.raises(SystemExit):
            ensure_local_url(url)
