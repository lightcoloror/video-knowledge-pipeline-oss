"""Opt-in pytest audit guard for local evidence regressions (no service health proof).

Load with -p evidence_offline_guard and tests on PYTHONPATH. The audit hook
also covers socket functions imported before pytest; test-local mocks remain
usable. It applies to this Python process, not arbitrary child executables.
"""
import sys
import socket
from collections import Counter

_blocked = Counter()
_ipc = Counter()
_EVENTS = frozenset({'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname',
                     'socket.gethostbyaddr', 'socket.sendto', 'socket.sendmsg'})


def _network_audit(event, args):
    # CPython on Windows implements asyncio's self-pipe with this authenticated
    # transient loopback socketpair. Admit only its exact csock -> lsock pair,
    # never generic loopback health probes or arbitrary local destinations.
    if event == 'socket.connect':
        caller = sys._getframe(1)
        fallback = getattr(socket, '_fallback_socketpair', None)
        if caller.f_code is getattr(fallback, '__code__', None):
            local = caller.f_locals
            if (len(args) == 2 and args[0] is local.get('csock')
                    and args[1] == local['lsock'].getsockname()
                    and args[1][0] in ('127.0.0.1', '::1')):
                _ipc['stdlib_socketpair'] += 1
                return
    if event in _EVENTS:
        _blocked[event] += 1
        raise OSError('Evidence regression offline guard: real network operation denied')


def pytest_sessionstart(session):
    sys.addaudithook(_network_audit)


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line('Evidence offline guard: ' + repr(dict(_blocked))
        + '; IPC: ' + repr(dict(_ipc))
        + '; Python network attempts denied, not provider-health validation.')
