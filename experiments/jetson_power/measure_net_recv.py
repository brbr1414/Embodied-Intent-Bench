#!/usr/bin/env python3
"""Discard receiver for measure_net.py: accept one connection, read, count, drop.

Runs on the far host (writes nothing to disk — the Orin root has ~5 MB free).

Usage: measure_net_recv.py <port>
"""

import socket
import sys


def main():
    port = int(sys.argv[1])
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(1)
    server.settimeout(1800)
    conn, addr = server.accept()
    conn.settimeout(120)
    total = 0
    while True:
        try:
            data = conn.recv(1 << 16)
        except socket.timeout:  # noqa: UP041 — runs on Python 3.8, where this is not TimeoutError
            break
        if not data:
            break
        total += len(data)
    print(f"received {total / 1e6:.1f} MB from {addr[0]}")


if __name__ == "__main__":
    main()
