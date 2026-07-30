#!/usr/bin/env bash
# Track F / F1: Postgres 16 + pgvector on the inf2 host. Idempotent by
# construction: every step probes state before mutating it, so rerunning
# after a partial failure (or on an already-provisioned box) is safe.
#
# Design mirrored from the source repo's scripts/setup_postgres_pgvector.py:
# same probe-then-create pattern for role (pg_roles), database (pg_database)
# and extension (CREATE EXTENSION IF NOT EXISTS), same env-overridable
# credentials. Differences: role/db are np_rag (this repo's namespace), and
# the schema is applied from schema.sql THROUGH the exact DSN the lanes use,
# so a green run here proves the connection string ingest.py/query.py rely on.
set -euo pipefail

PG_USER="${PG_USER:-np_rag}"
PG_DB="${PG_DB:-np_rag}"
PG_PASSWORD="${PG_PASSWORD:-np_rag}"   # demo appliance on a private, SSM-only box; override via env
PG_PORT="${PG_PORT:-5432}"
DSN="postgresql://$PG_USER:$PG_PASSWORD@localhost:$PG_PORT/$PG_DB"
RAG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_SQL="$RAG_DIR/schema.sql"

SUDO="sudo"
[ "$(id -u)" = "0" ] && SUDO=""
as_pg() { sudo -u postgres "$@"; }     # works from root and from sudo-capable users

echo "=== setup_pg: postgresql-16 + pgvector -> $DSN ==="

# ---------------------------------------------------------------- 1. packages
if ! dpkg -s postgresql-16 >/dev/null 2>&1 || ! dpkg -s postgresql-16-pgvector >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -y
  if ! apt-cache show postgresql-16 >/dev/null 2>&1; then
    # The inf2 DLAMI is Ubuntu 22.04, whose stock apt only carries
    # postgresql-14; postgresql-16 and postgresql-16-pgvector come from the
    # PGDG repo. The enrolment helper ships in postgresql-common and is
    # itself idempotent. TODO-VERIFY on-box: outbound HTTPS to
    # apt.postgresql.org from the private subnet (S3/pip already work via
    # the NAT path, so this is expected to as well).
    $SUDO apt-get install -y --no-install-recommends postgresql-common ca-certificates
    $SUDO /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
    $SUDO apt-get update -y
  fi
  $SUDO apt-get install -y postgresql-16 postgresql-16-pgvector
else
  echo "--- packages already installed (skip apt) ---"
fi

# ----------------------------------------------------------------- 2. service
# TODO-VERIFY on-box: on a machine with a pre-existing older cluster the new
# 16 cluster can land on port 5433 (`pg_lsclusters`); this box is expected
# clean, so 5432 is asserted below by pg_isready.
if ! pg_isready -q -p "$PG_PORT" 2>/dev/null; then
  $SUDO systemctl start postgresql 2>/dev/null || $SUDO service postgresql start
fi
for _ in $(seq 1 30); do
  pg_isready -q -p "$PG_PORT" 2>/dev/null && break
  sleep 1
done
pg_isready -q -p "$PG_PORT" || { echo "postgres never became ready on :$PG_PORT" >&2; exit 1; }

# ----------------------------------------------------- 3. role / db / extension
# Same idempotency probes as the source repo: SELECT from pg_roles /
# pg_database, ALTER instead of CREATE when the object already exists.
if as_pg psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$PG_USER'" | grep -q 1; then
  as_pg psql -qc "ALTER ROLE $PG_USER WITH LOGIN PASSWORD '$PG_PASSWORD'"
  echo "--- role $PG_USER exists (password refreshed) ---"
else
  as_pg psql -qc "CREATE ROLE $PG_USER WITH LOGIN PASSWORD '$PG_PASSWORD'"
  echo "--- role $PG_USER created ---"
fi

if as_pg psql -tAc "SELECT 1 FROM pg_database WHERE datname='$PG_DB'" | grep -q 1; then
  echo "--- database $PG_DB exists ---"
else
  as_pg createdb -O "$PG_USER" "$PG_DB"
  echo "--- database $PG_DB created (owner $PG_USER) ---"
fi

# Extension needs superuser; np_rag only needs to USE it.
as_pg psql -d "$PG_DB" -qc "CREATE EXTENSION IF NOT EXISTS vector"
as_pg psql -d "$PG_DB" -qc "GRANT ALL ON SCHEMA public TO $PG_USER"

# ------------------------------------------------------------------ 4. schema
# Applied AS np_rag over TCP: exercises the lanes' DSN end to end.
psql "$DSN" -v ON_ERROR_STOP=1 -q -f "$SCHEMA_SQL"
echo "--- schema.sql applied (rerun-safe: IF NOT EXISTS throughout) ---"

# ------------------------------------------------------------------ 5. verify
echo "--- verify ---"
echo "server  : $(psql "$DSN" -tAc 'SELECT version();')"
EXTV="$(psql "$DSN" -tAc "SELECT extversion FROM pg_extension WHERE extname='vector';")"
echo "pgvector: extversion=${EXTV:-MISSING}"
[ -n "$EXTV" ] || { echo "pgvector extension missing after setup" >&2; exit 1; }
echo "chunks  : $(psql "$DSN" -tAc 'SELECT count(*) FROM chunks;') rows"
echo "=== setup_pg OK ==="
