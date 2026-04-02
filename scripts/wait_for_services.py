#!/usr/bin/env python3
"""
Wait for services to be ready

This script waits for PostgreSQL and other services to be ready
before the devcontainer is considered fully initialized.
"""

import time
import socket
import sys

def wait_for_port(host, port, service_name, timeout=60):
    """Wait for a port to be open."""
    print(f"⏳ Waiting for {service_name} on {host}:{port}...")

    start_time = time.time()
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                print(f"✓ {service_name} is ready!")
                return True

        except socket.error:
            pass

        if time.time() - start_time > timeout:
            print(f"✗ Timeout waiting for {service_name}")
            return False

        time.sleep(2)

    return False

def main():
    print("="*60)
    print("Ike SemOps - Waiting for Services")
    print("="*60)
    print()

    services = [
        ("localhost", 5432, "PostgreSQL"),
        ("localhost", 5678, "n8n"),
        ("localhost", 6333, "Qdrant"),
    ]

    all_ready = True
    for host, port, name in services:
        if not wait_for_port(host, port, name):
            all_ready = False

    print()
    if all_ready:
        print("✓ All services are ready!")
        print()
        print("📍 Service URLs:")
        print("  • Supabase Studio:  http://localhost:8000")
        print("  • n8n:              http://localhost:5678")
        print("  • Qdrant Dashboard: http://localhost:6333/dashboard")
        print()
        print("🚀 Ready to code!")
        print()
        return 0
    else:
        print("⚠ Some services are not ready yet")
        print("  You may need to wait a bit longer or check Docker logs")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
