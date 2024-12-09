#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the directory of the Docker Compose file
ROOT_DIR="$(git rev-parse --show-toplevel)"
DOCKER_DIR="$ROOT_DIR/docker"
COMPOSE_FILE="$DOCKER_DIR/compose.yml"

docker compose -f "$COMPOSE_FILE" up --build frontend playground

docker compose -f "$COMPOSE_FILE" up --build backend caddy