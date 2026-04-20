#!/bin/bash
set -e

# Wait for Postgres
echo "Waiting for PostgreSQL..."
for i in {1..30}; do
  if nc -z metastore-db 5432 2>/dev/null; then
    echo "PostgreSQL is ready!"
    break
  fi
  echo "Retry $i: PostgreSQL not ready yet..."
  sleep 1
done

# Initialize Hive schema if not already done  
echo "Checking/Initializing Hive Metastore schema..."
schematool -dbType postgres -initSchema || echo "Schema already exists or initialization handled"

# Start the metastore service
echo "Starting Hive Metastore service on port 9083..."
exec hive --service metastore
