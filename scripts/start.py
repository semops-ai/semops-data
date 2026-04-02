#!/usr/bin/env python3
"""
Start Ike SemOps services

This script:
1. Clones/updates Supabase repository
2. Starts all services (Supabase + n8n + Qdrant)
3. Waits for services to be ready
4. Optionally initializes Phase 1 schema

Usage:
    python scripts/start.py
    python scripts/start.py --init-schema  # Also run schema initialization
"""

import os
import subprocess
import time
import argparse
import sys

def run_command(cmd, cwd=None, check=True):
    """Run a shell command."""
    print(f"→ Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"✗ Command failed: {result.stderr}")
        sys.exit(1)
    return result

def clone_supabase_repo():
    """Clone the Supabase repository using sparse checkout if not already present."""
    if not os.path.exists("supabase"):
        print("\n📦 Cloning Supabase repository...")
        run_command([
            "git", "clone", "--filter=blob:none", "--no-checkout",
            "https://github.com/supabase/supabase.git"
        ])
        os.chdir("supabase")
        run_command(["git", "sparse-checkout", "init", "--cone"])
        run_command(["git", "sparse-checkout", "set", "docker"])
        run_command(["git", "checkout", "master"])
        os.chdir("..")
        print("✓ Supabase repository cloned")
    else:
        print("\n📦 Supabase repository exists, updating...")
        result = run_command(["git", "pull"], cwd="supabase", check=False)
        if result.returncode == 0:
            print("✓ Supabase repository updated")
        else:
            print("⚠ Could not update Supabase repo (might be fine)")

def prepare_supabase_env():
    """Copy .env to supabase/docker/.env"""
    env_path = os.path.join("supabase", "docker", ".env")
    root_env = ".env"

    if not os.path.exists(root_env):
        print("✗ Error: .env file not found in root directory!")
        print("  Please create a .env file with required variables.")
        sys.exit(1)

    print(f"\n📋 Copying .env to {env_path}...")
    import shutil
    shutil.copyfile(root_env, env_path)
    print("✓ Environment configured")

def check_docker():
    """Check if Docker is running."""
    print("\n🐳 Checking Docker...")
    result = run_command(["docker", "info"], check=False)
    if result.returncode != 0:
        print("✗ Docker is not running!")
        print("  Please start Docker Engine and try again.")
        sys.exit(1)
    print("✓ Docker is running")

def start_services():
    """Start all services using docker compose."""
    print("\n🚀 Starting services...")
    print("  This may take a few minutes on first run...")

    run_command(["docker", "compose", "up", "-d"])

    print("\n✓ Services started!")
    print("\n  Waiting for services to initialize...")
    time.sleep(15)  # Give services time to start

def check_services():
    """Check if services are running."""
    print("\n🔍 Checking service health...")

    services = {
        "PostgreSQL (direct)": ("localhost", 5434),
        "Supabase Studio": ("localhost", 8000),
        "n8n": ("localhost", 5678),
        "Qdrant": ("localhost", 6333),
    }

    all_good = True
    for name, (host, port) in services.items():
        # Try to connect
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            print(f"  ✓ {name} is running on port {port}")
        else:
            print(f"  ⚠ {name} not yet ready on port {port}")
            all_good = False

    return all_good

def show_summary():
    """Show summary and next steps."""
    print("\n" + "="*70)
    print("🎉 Ike SemOps Services Started!")
    print("="*70)

    print("\n📍 Access your services:")
    print("  • Supabase Studio:  http://localhost:8000")
    print("  • n8n:              http://localhost:5678")
    print("  • Qdrant Dashboard: http://localhost:6333/dashboard")

    print("\n🔐 Supabase Studio Login:")
    print("  Username: (from DASHBOARD_USERNAME in .env)")
    print("  Password: (from DASHBOARD_PASSWORD in .env)")

    print("\n🗄️  Database Connection (direct, bypasses pooler):")
    print("  Host:     localhost")
    print("  Port:     5434")
    print("  Database: postgres")
    print("  User:     postgres")
    print("  Password: (from POSTGRES_PASSWORD in .env)")
    print("  Note:     Port 5432 is Supavisor (pooler). Use 5434 for scripts/agents.")

    print("\n🚀 Next Steps:")
    print("  1. Initialize Phase 1 schema:")
    print("     python scripts/init_schema.py")
    print("\n  2. Open Supabase Studio and explore the database")
    print("  3. Open n8n and create your first workflow")
    print("  4. Check the README.md for examples and documentation")

    print("\n⏹️  To stop services:")
    print("     docker compose down")
    print("\n")

def main():
    parser = argparse.ArgumentParser(
        description='Start Ike SemOps services'
    )
    parser.add_argument(
        '--init-schema',
        action='store_true',
        help='Also initialize Phase 1 schema after starting services'
    )
    args = parser.parse_args()

    print("="*70)
    print("Ike SemOps - Service Startup")
    print("="*70)

    # Check prerequisites
    check_docker()

    # Setup Supabase
    clone_supabase_repo()
    prepare_supabase_env()

    # Start services
    start_services()

    # Check services
    if check_services():
        print("\n✓ All services are ready!")
    else:
        print("\n⚠ Some services may still be initializing...")
        print("  Wait a few more seconds and check the URLs above")

    # Initialize schema if requested
    if args.init_schema:
        print("\n🗄️  Initializing Phase 1 schema...")
        result = run_command(
            ["python", "scripts/init_schema.py"],
            check=False
        )
        if result.returncode == 0:
            print("✓ Schema initialized successfully!")
        else:
            print("⚠ Schema initialization had issues - you can run it manually:")
            print("  python scripts/init_schema.py")

    show_summary()

if __name__ == "__main__":
    main()
