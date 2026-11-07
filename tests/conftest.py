"""Keep the portable suite offline while allowing its loopback mock servers."""

import os
import socket

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.fixture(autouse=True)
def offline_connections(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    original = socket.create_connection
    original_connect = socket.socket.connect
    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        if host not in {"127.0.0.1", "::1", "localhost", None}:
            raise OSError("External DNS is disabled in the offline test suite")
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

    def connect(address, *args, **kwargs):
        if address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise OSError("External network is disabled in the offline test suite")
        return original(address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", connect)

    def socket_connect(sock, address):
        if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise OSError("External network is disabled in the offline test suite")
        return original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", socket_connect)
