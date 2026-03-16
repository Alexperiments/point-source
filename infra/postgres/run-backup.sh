#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/backend/.env.production}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/infra/postgres/backup.log}"
CONTAINER_NAME="${CONTAINER_NAME:-postgres}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Starting Postgres base backup"

  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a

  docker exec \
    -e PGUSER="$DATABASE_USER" \
    -e PGPASSWORD="$DATABASE_PASSWORD" \
    -e PGDATABASE="$DATABASE_NAME" \
    -e PGHOST="/var/run/postgresql" \
    "$CONTAINER_NAME" \
    sh -lc 'envdir /etc/wal-g.d/env wal-g backup-push /home/postgres/pgdata/data'

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Finished Postgres base backup"
} >>"$LOG_FILE" 2>&1
