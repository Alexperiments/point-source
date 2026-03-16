#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/backend/.env.production}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/infra/postgres/backup.log}"
CONTAINER_NAME="${CONTAINER_NAME:-postgres}"
LAST_LSN_FILE="${LAST_LSN_FILE:-$REPO_ROOT/infra/postgres/.last_wal_lsn}"

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

  get_current_lsn() {
    docker exec \
      -e PGUSER="$DATABASE_USER" \
      -e PGPASSWORD="$DATABASE_PASSWORD" \
      -e PGDATABASE="$DATABASE_NAME" \
      -e PGHOST="/var/run/postgresql" \
      "$CONTAINER_NAME" \
      sh -lc 'psql -tA -c "SELECT pg_current_wal_lsn();"'
  }

  if [ "${FORCE:-0}" = "1" ]; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] FORCE=1, running backup-push regardless of LSN"
    WAL_LSN="$(get_current_lsn)"
  else
    WAL_LSN="$(get_current_lsn)"
    if [ -f "$LAST_LSN_FILE" ]; then
      LAST_LSN="$(cat "$LAST_LSN_FILE")"
    else
      LAST_LSN=""
    fi
    if [ "$WAL_LSN" = "$LAST_LSN" ] && [ -n "$LAST_LSN" ]; then
      echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] WAL LSN unchanged ($WAL_LSN), skipping backup-push"
      exit 0
    fi
  fi

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] WAL LSN current: $WAL_LSN"

  docker exec \
    -e PGUSER="$DATABASE_USER" \
    -e PGPASSWORD="$DATABASE_PASSWORD" \
    -e PGDATABASE="$DATABASE_NAME" \
    -e PGHOST="/var/run/postgresql" \
    "$CONTAINER_NAME" \
    sh -lc 'envdir /etc/wal-g.d/env wal-g backup-push /home/postgres/pgdata/data'

  echo "$WAL_LSN" > "$LAST_LSN_FILE"
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Wrote last WAL LSN to $LAST_LSN_FILE"
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Finished Postgres base backup"
} >>"$LOG_FILE" 2>&1
