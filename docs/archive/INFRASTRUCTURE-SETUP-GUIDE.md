# Infrastructure Repository Setup Guide

## Overview
This guide covers best practices for creating a shared Docker infrastructure repository that can support multiple application projects, with considerations for dual-boot (Windows/Linux) environments.

---

## Step-by-Step Setup Plan

### 1. Choose Your Primary Development OS
**Recommendation: Linux**

**Why Linux?**
- Docker runs natively (better performance)
- Better file permission handling
- No WSL2 overhead
- Industry standard for containerized development
- Simpler path syntax (` vs `C:\Users\...`)

**Decision Point:** Pick ONE OS for infrastructure development to avoid data sync issues.

### 2. Create Infrastructure Repository Structure

```bash
# Recommended folder structure
infrastructure/
├── docker-compose.yml
├── .env.example
├── .env (git-ignored)
├── README.md
├── backups/
│   └── .gitkeep
└── scripts/
    ├── backup-databases.sh
    └── restore-databases.sh
```

### 3. Use Named Docker Volumes (Initially)

**Start simple - don't specify host folders:**

```yaml
services:
  postgres:
    image: supabase/postgres:15.1.0.117
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  supabase-storage:
    image: supabase/storage-api:latest
    volumes:
      - supabase_storage:/var/lib/storage
    ports:
      - "5000:5000"

volumes:
  postgres_data:
    # Docker manages this automatically
  supabase_storage:
    # No host path specified
```

**Benefits:**
- Works the same on Windows or Linux
- No permission issues
- Simple to start
- Easy to backup/restore with Docker commands

**Later:** Once you've chosen your primary OS, you can convert to host-mounted volumes if needed.

### 4. Expose Ports for Application Access

```yaml
services:
  postgres:
    ports:
      - "5432:5432"  # Applications connect via localhost:5432

  supabase:
    ports:
      - "54321:8000"  # API accessible at localhost:54321

  n8n:
    ports:
      - "5678:5678"  # UI at localhost:5678
```

**Your application repos then connect via:**
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/postgres
SUPABASE_URL=http://localhost:54321
N8N_URL=http://localhost:5678
```

---

## Cross-OS "Watch Out For" List

### ❌ DON'Ts - Dual Boot Pitfalls

1. **DON'T use host-mounted volumes across OSes**
   ```yaml
   # This breaks across Windows/Linux
   volumes:
     - C:/Data/postgres:/var/lib/postgresql/data  # Windows only
     -   # Linux only
   ```

2. **DON'T expect data to transfer automatically**
   - Each OS has separate Docker installations
   - Named volumes are OS-specific
   - You must explicitly export/import data

3. **DON'T use NTFS for shared partitions with database volumes**
   - NTFS permissions cause issues with PostgreSQL in Linux
   - Performance degradation
   - Potential data corruption risks

4. **DON'T hard-code OS-specific paths in docker-compose.yml**
   ```yaml
   # BAD - OS-specific
   - C:\Users\Me\backups:/backups

   # GOOD - Use named volumes or environment variables
   - backup_data:/backups
   ```

5. **DON'T forget to stop containers before switching OS**
   ```bash
   # Always run before rebooting
   docker-compose down
   ```
   - Prevents potential corruption
   - Ensures clean shutdown

### ✅ DOs - Best Practices

1. **DO use named Docker volumes**
   - Portable across Docker versions
   - OS-independent syntax
   - Proper permission handling

2. **DO create database dumps for OS transfers**
   ```bash
   # Export before switching OS
   docker exec postgres pg_dumpall -U postgres > backup.sql
   ```

3. **DO keep .env files in sync**
   - Use the same credentials across both OSes
   - Store .env.example in git
   - Manually copy .env when needed

4. **DO use consistent port mappings**
   - Same ports on both OSes
   - Prevents connection string changes

5. **DO document which OS you're primarily using**
   - Add to README.md
   - Helps future you remember

---

## Docker Volume & Network Options

### Option 1: Infrastructure Repo + Named Volumes (Recommended to Start)

**Setup:**
```yaml
# infrastructure/docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  postgres_data:
    name: infra_postgres_data
```

**Benefits:**
- ✅ Simple setup
- ✅ Works on any OS
- ✅ Docker manages volume lifecycle
- ✅ Persists across container recreations
- ✅ Easy backup with Docker commands

**Use Case:** Best for starting out, single-OS development

---

### Option 2: Infrastructure Repo + External Named Volumes

**Setup:**
```bash
# Create volumes once
docker volume create shared_postgres_data
docker volume create shared_supabase_storage
```

```yaml
# infrastructure/docker-compose.yml
services:
  postgres:
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
    external: true
    name: shared_postgres_data
```

**Benefits:**
- ✅ Volumes survive `docker-compose down -v`
- ✅ Can be shared across multiple compose projects
- ✅ Explicit volume lifecycle management
- ✅ Prevents accidental deletion

**Use Case:** When you have multiple infrastructure setups or want extra protection

---

### Option 3: Infrastructure Repo + Host-Mounted Volumes

**Setup (Linux):**
```yaml
# infrastructure/docker-compose.yml
services:
  postgres:
    volumes:
      - 
      - 
    ports:
      - "5432:5432"

  supabase-storage:
    volumes:
      - 
```

**Benefits:**
- ✅ Easy to locate and browse data
- ✅ Simple backup (copy folders)
- ✅ Can use different drives/partitions
- ✅ Survives Docker reinstalls
- ✅ Direct file access for inspection

**Drawbacks:**
- ⚠️ Must manage permissions manually
- ⚠️ OS-specific paths (not portable)
- ⚠️ Need to create directories first

**Use Case:** When you've committed to one OS and want direct file access

---

### Option 4: Infrastructure Repo + Shared Docker Network

**For connecting multiple Docker Compose projects together:**

**Setup:**
```bash
# Create shared network once
docker network create infrastructure_network
```

**Infrastructure repo:**
```yaml
# infrastructure/docker-compose.yml
services:
  postgres:
    networks:
      - infra_network
    # No port exposure needed if apps are on same network

networks:
  infra_network:
    external: true
    name: infrastructure_network
```

**Application repo:**
```yaml
# my-app/docker-compose.yml
services:
  api:
    networks:
      - infra_network
    environment:
      # Use service name from infrastructure repo
      DATABASE_URL: postgresql://postgres@postgres:5432/db

networks:
  infra_network:
    external: true
    name: infrastructure_network
```

**Benefits:**
- ✅ Containers can reference each other by service name
- ✅ No port exposure to host needed (more secure)
- ✅ Isolated network communication
- ✅ Multiple apps can share infrastructure

**Drawbacks:**
- ⚠️ More complex setup
- ⚠️ Apps must be containerized
- ⚠️ Harder to debug (can't use localhost)

**Use Case:** When all your applications run in Docker and you want service mesh-like connectivity

---

### Option 5: Infrastructure Repo + Exposed Ports (Simplest for Multiple Apps)

**Setup:**
```yaml
# infrastructure/docker-compose.yml
services:
  postgres:
    ports:
      - "5432:5432"

  supabase:
    ports:
      - "54321:8000"

  n8n:
    ports:
      - "5678:5678"
```

**Application repos (Docker or native):**
```env
# .env in any app
DATABASE_URL=postgresql://postgres:password@localhost:5432/postgres
SUPABASE_URL=http://localhost:54321
```

**Benefits:**
- ✅ Simplest to understand
- ✅ Works with containerized AND native apps
- ✅ Easy to test (use pgAdmin, Postman, etc.)
- ✅ No network configuration needed
- ✅ Each app repo is independent

**Use Case:** Most common pattern, best for mixed development (some Docker, some native)

---

## Comparison Matrix

| Approach | Complexity | Cross-OS | Multi-App | Native App Access | Best For |
|----------|------------|----------|-----------|-------------------|----------|
| Named Volumes | Low | ✅ Yes | ✅ Yes | ✅ Yes (via ports) | Starting out |
| External Volumes | Medium | ✅ Yes | ✅ Yes | ✅ Yes (via ports) | Production-like |
| Host-Mounted | Medium | ❌ No | ✅ Yes | ✅ Yes (via ports) | Single OS committed |
| Shared Network | High | ✅ Yes | ✅ Yes | ❌ No | Full Docker ecosystem |
| Exposed Ports | Low | ✅ Yes | ✅ Yes | ✅ Yes | **Recommended** |

---

## Backup and Restore Strategies

### Strategy 1: Volume-Level Backup (Complete Snapshot)

**What it does:** Copies the entire Docker volume like cloning a hard drive

**Backup:**
```bash
# Create backup of entire volume
docker run --rm \
  -v postgres_data:/data \
  -v $(pwd)/backups:/backup \
  ubuntu tar czf /backup/postgres-volume-$(date +%Y%m%d-%H%M%S).tar.gz -C /data .
```

**Restore:**
```bash
# Restore entire volume
docker run --rm \
  -v postgres_data:/data \
  -v $(pwd)/backups:/backup \
  ubuntu tar xzf /backup/postgres-volume-20250131-120000.tar.gz -C /data
```

**Pros:**
- ✅ Complete snapshot including all metadata
- ✅ Exact filesystem state preserved
- ✅ Works for any volume type

**Cons:**
- ❌ Larger file size
- ❌ OS/version specific
- ❌ Not human-readable
- ❌ Harder to restore selectively

**Use Case:** Disaster recovery, complete system migration

---

### Strategy 2: Database-Level Backup (Logical Export) **[RECOMMENDED]**

**What it does:** Exports database structure and data as SQL statements

**Backup (Postgres):**
```bash
# Full database dump (structure + data + users)
docker exec -t postgres pg_dumpall -c -U postgres > backups/dump-$(date +%Y%m%d-%H%M%S).sql

# Single database
docker exec -t postgres pg_dump -U postgres dbname > backups/dbname-$(date +%Y%m%d-%H%M%S).sql

# Compressed
docker exec -t postgres pg_dump -U postgres dbname | gzip > backups/dbname-$(date +%Y%m%d-%H%M%S).sql.gz
```

**Restore:**
```bash
# Restore full dump
cat backups/dump-20250131-120000.sql | docker exec -i postgres psql -U postgres

# Restore compressed
gunzip < backups/dbname.sql.gz | docker exec -i postgres psql -U postgres -d dbname

# Restore single database
docker exec -i postgres psql -U postgres -d dbname < backups/dbname.sql
```

**Pros:**
- ✅ **Cross-OS compatible** (Windows ↔ Linux)
- ✅ Human-readable SQL
- ✅ Version-independent (mostly)
- ✅ Smaller file size
- ✅ Can selectively restore tables
- ✅ Can be version-controlled (if small enough)

**Cons:**
- ❌ Doesn't include system configs
- ❌ Requires database to be running

**Use Case:** Moving between OSes, migrations, daily backups, version control

---

### Strategy 3: Automated Backup Script

**Create:** `scripts/backup-databases.sh`

```bash
#!/bin/bash
set -e

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "Starting database backup at $TIMESTAMP"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Backup Postgres
echo "Backing up PostgreSQL..."
docker exec postgres pg_dumpall -c -U postgres | gzip > "$BACKUP_DIR/postgres-$TIMESTAMP.sql.gz"

# Backup Supabase Storage (if using volumes)
echo "Backing up Supabase storage..."
docker run --rm \
  -v supabase_storage:/data \
  -v "$(pwd)/$BACKUP_DIR:/backup" \
  ubuntu tar czf "/backup/supabase-storage-$TIMESTAMP.tar.gz" -C /data .

# Keep only last 7 days of backups
echo "Cleaning old backups..."
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/*-$TIMESTAMP.*"
```

**Make executable:**
```bash
chmod +x scripts/backup-databases.sh
```

**Run manually:**
```bash
./scripts/backup-databases.sh
```

**Automate with cron (Linux):**
```bash
# Daily backup at 2 AM
0 2 * * * cd /path/to/infrastructure && ./scripts/backup-databases.sh >> ./backups/backup.log 2>&1
```

---

### Strategy 4: Cross-OS Transfer Workflow

**Scenario:** You developed in Linux, now switching to Windows for a session

**Export from Linux:**
```bash
# 1. Create database dump
docker exec postgres pg_dumpall -U postgres > transfer-backup.sql

# 2. Copy to shared location
cp transfer-backup.sql /mnt/c/Shared/  # Or USB drive, cloud, etc.
```

**Import to Windows:**
```bash
# 1. Start infrastructure in Windows
docker-compose up -d

# 2. Wait for Postgres to be ready
docker exec postgres pg_isready -U postgres

# 3. Restore database
cat /c/Shared/transfer-backup.sql | docker exec -i postgres psql -U postgres

# 4. Verify
docker exec -it postgres psql -U postgres -c "\dt"
```

**Pros:**
- ✅ Works perfectly across OSes
- ✅ Clean data transfer
- ✅ No permission issues

**Cons:**
- ❌ Manual process
- ❌ Need to remember to export before switching

---

### Strategy 5: Direct Volume Access (Advanced)

**Locate volume on disk:**
```bash
# Find where Docker stores the volume
docker volume inspect postgres_data
```

**Output:**
```json
[
    {
        "Name": "postgres_data",
        "Mountpoint": "/var/lib/docker/volumes/postgres_data/_data"
    }
]
```

**Backup by copying directory:**
```bash
# Linux (requires sudo)
sudo cp -r /var/lib/docker/volumes/postgres_data/_data ./backups/postgres-data-backup

# Or create tar
sudo tar czf ./backups/volume-backup.tar.gz -C /var/lib/docker/volumes/postgres_data/_data .
```

**Pros:**
- ✅ Direct access to files
- ✅ Fast backup

**Cons:**
- ❌ Requires root/admin access
- ❌ Must stop containers first (risk of corruption)
- ❌ OS-specific paths
- ❌ Doesn't work with Docker Desktop on Windows (uses VM)

**Use Case:** Emergency recovery only

---

## Backup Comparison

| Method | Cross-OS | Size | Speed | Ease | Recommended |
|--------|----------|------|-------|------|-------------|
| Volume-level | ❌ No | Large | Fast | Medium | Disaster recovery |
| Database dump | ✅ Yes | Small | Medium | Easy | **Daily use** |
| Automated script | ✅ Yes | Small | Medium | Easy | **Production** |
| Direct copy | ❌ No | Large | Fast | Hard | Emergency only |

---

## Recommended Workflow

### Initial Setup (One Time)

1. **Choose Linux as primary OS** for infrastructure development

2. **Create infrastructure repository:**
   ```bash
   mkdir infrastructure
   cd infrastructure
   git init
   ```

3. **Create docker-compose.yml with named volumes:**
   ```yaml
   version: '3.8'
   services:
     postgres:
       image: supabase/postgres:15.1.0.117
       volumes:
         - postgres_data:/var/lib/postgresql/data
       environment:
         POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
       ports:
         - "5432:5432"

   volumes:
     postgres_data:
   ```

4. **Create .env file:**
   ```env
   POSTGRES_PASSWORD=your_secure_password
   # Add other service configs
   ```

5. **Create backup script** (see Strategy 3 above)

6. **Document in README.md:**
   - How to start services
   - Connection strings
   - Backup/restore procedures

### Daily Development

**Starting work:**
```bash
cd infrastructure
docker-compose up -d
```

**Stopping work:**
```bash
docker-compose down  # Volumes persist!
```

**Weekly backup:**
```bash
./scripts/backup-databases.sh
```

### When Switching to Windows (Occasional)

**Before leaving Linux:**
```bash
# Create portable dump
docker exec postgres pg_dumpall -U postgres > /mnt/c/Shared/latest-dump.sql
docker-compose down
```

**In Windows:**
```bash
# Start fresh infrastructure
cd C:\path\to\infrastructure
docker-compose up -d

# Restore data
cat /c/Shared/latest-dump.sql | docker exec -i postgres psql -U postgres
```

---

## Summary Recommendations

### For Your Dual-Boot Scenario:

1. ✅ **Use Linux** as primary infrastructure OS
2. ✅ **Use named Docker volumes** (not host-mounted)
3. ✅ **Expose ports** for application access (simplest)
4. ✅ **Use `pg_dumpall`** for cross-OS transfers
5. ✅ **Automate backups** with a script
6. ✅ **Keep .env synced** across both OSes

### Volume Strategy:
- **Start:** Named volumes (simple, portable)
- **Later:** Can migrate to host-mounted if you commit to one OS

### Network Strategy:
- **Use:** Exposed ports (`5432:5432`, etc.)
- **Why:** Works with Docker and native apps, simplest debugging

### Backup Strategy:
- **Daily:** Automated `pg_dump` script
- **Before OS switch:** Manual `pg_dumpall` export
- **Emergency:** Volume-level backup

This approach gives you maximum flexibility while keeping things simple!
