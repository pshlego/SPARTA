#!/bin/bash
set -e

# PostgreSQL details
PGHOST="localhost"
PGPORT="5432"
PGUSER="$POSTGRES_USER" # make sure this is set in your environment
PGDB="$POSTGRES_DB"     # make sure this is set in your environment

# Wait for Postgres to become available.
echo "Waiting for PostgreSQL to start on port $PGPORT..."
until psql -U "$PGUSER" -c '\q' 2>/dev/null; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done
>&2 echo "Postgres is up - executing command"

# Create expanded database
psql -v ON_ERROR_STOP=1 --username "$PGUSER" --dbname "$PGDB" <<-EOSQL
    CREATE DATABASE nba;
EOSQL

# Construct text database
psql -v ON_ERROR_STOP=1 --username "$PGUSER" --dbname "nba" -f /docker-entrypoint-initdb.d/nba/grounding_tables/init.sql