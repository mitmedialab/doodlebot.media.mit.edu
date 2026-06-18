#!/usr/bin/env bash
#
# Single entry point for managing the server service.
# Run from anywhere — paths are resolved relative to this script.
#
# Usage:
#   ./cli.sh start              Build the image and (re)start the server, detached.
#                               Safe to run on an already-running stack: it
#                               force-recreates the container from the fresh
#                               image, so you do NOT need to stop first.
#   ./cli.sh stop               Stop and remove the server container and network.
#   ./cli.sh restart            Clean slate: stop (down), then start. Unlike a
#                               bare start, this also tears down the network and
#                               removes orphan containers. Use it when networking
#                               changed, a service was removed, or to recover from
#                               a wedged/stale-container state — not for routine
#                               redeploys, where start alone suffices.
#   ./cli.sh logs [N|all]       Follow logs; show the last N lines first (default 100).
#   ./cli.sh status             Show the service's container status and port mappings.
#   ./cli.sh help               Show this message.
#
# Examples:
#   ./cli.sh start              # bring the server up with a fresh build
#   ./cli.sh logs               # tail -f, last 100 lines first
#   ./cli.sh logs 500           # tail -f, last 500 lines first
#   ./cli.sh logs all           # tail -f, full history first
#   ./cli.sh stop               # tear the stack down cleanly

set -euo pipefail

# Resolve this script to an absolute path BEFORE changing directories, so it can
# still be read (e.g. by usage()) regardless of where it was invoked from.
SCRIPT="$(readlink -f "$0")"

# Always operate from the directory holding this script / compose.yml.
cd "$(dirname "${SCRIPT}")"

SERVICE="server"

# `docker compose` (v2 plugin) with a fallback to the legacy `docker-compose`.
if docker compose version >/dev/null 2>&1; then
  compose() { docker compose "$@"; }
elif command -v docker-compose >/dev/null 2>&1; then
  compose() { docker-compose "$@"; }
else
  echo "error: neither 'docker compose' nor 'docker-compose' is available" >&2
  exit 1
fi

usage() {
  # Print the leading comment block (between the shebang and `set -euo`),
  # stripped of the leading "# " / "#" so it reads as plain help text.
  sed -n '3,/^set /{/^set /d; s/^#\( \|$\)//; p;}' "${SCRIPT}"
}

# Print an error, then the usage text, and exit with the usage-error code (2).
# Used for anything the caller can fix by reading the help: missing command,
# unknown command, or a subcommand given bad arguments.
die() {
  echo "error: $*" >&2
  echo >&2
  usage >&2
  exit 2
}

# Guard for subcommands that accept no arguments. $1 is the command name; any
# further arguments are the (unexpected) ones the user passed.
expect_no_args() {
  local name="$1"; shift
  [ "$#" -eq 0 ] || die "'${name}' takes no arguments (got: $*)"
}

# compose.yml bind-mounts ./presets.json as a *file*. If it's missing, Docker
# silently creates a directory in its place, which then breaks the server's JSON
# read. Ensure it exists as a file (seeded with an empty JSON object) first.
ensure_presets_file() {
  local presets="presets.json"
  if [ -d "${presets}" ]; then
    die "'${presets}' is a directory (likely created by a failed bind mount); remove it and retry"
  fi
  if [ ! -f "${presets}" ]; then
    echo "==> Creating missing ${presets} (seeded with {})"
    echo '{}' > "${presets}"
  fi
}

cmd_start() {
  expect_no_args start "$@"
  ensure_presets_file
  echo "==> Building ${SERVICE} image"
  compose build "${SERVICE}"

  echo "==> Starting ${SERVICE} service"
  # --force-recreate ensures the freshly built image is used even if config is
  # unchanged; -d runs it detached. On an already-running stack this replaces
  # the live container in place, so calling start without a prior stop is fine.
  # It does NOT, however, tear down the network or prune orphans (that's stop's
  # job) — which is the whole reason `restart` exists as a separate command.
  compose up -d --force-recreate "${SERVICE}"

  echo "==> ${SERVICE} is up. Recent logs:"
  compose logs --tail=20 "${SERVICE}"
}

cmd_stop() {
  expect_no_args stop "$@"
  echo "==> Stopping and removing ${SERVICE} (and its network)"
  # `down` (not `stop`) also removes the container, avoiding a stale
  # half-created container that can hold the published host port hostage.
  compose down --remove-orphans
}

cmd_restart() {
  expect_no_args restart "$@"
  # Distinct from a bare `start`: the `down` in cmd_stop removes the container,
  # the network, and any orphans before cmd_start rebuilds them — a full reset,
  # not the in-place container swap that `start`/--force-recreate performs.
  cmd_stop
  cmd_start
}

cmd_logs() {
  [ "$#" -le 1 ] || die "'logs' takes at most one argument (got $#: $*)"
  local tail="${1:-100}"
  # Accept either the literal "all" or a positive integer — these are the only
  # values docker compose's --tail understands.
  if [ "${tail}" != "all" ] && ! [[ "${tail}" =~ ^[1-9][0-9]*$ ]]; then
    die "'logs' count must be a positive integer or 'all' (got: '${tail}')"
  fi
  echo "==> Following ${SERVICE} logs (tail=${tail}). Press Ctrl-C to stop."
  compose logs --follow --tail="${tail}" "${SERVICE}"
}

cmd_status() {
  expect_no_args status "$@"
  compose ps "${SERVICE}"
}

main() {
  # No command at all: show help, but treat it as a usage error so scripts can
  # detect the empty invocation.
  [ "$#" -gt 0 ] || die "no command provided"

  local sub="$1"; shift
  case "${sub}" in
    start)            cmd_start "$@" ;;
    stop)             cmd_stop "$@" ;;
    restart)          cmd_restart "$@" ;;
    logs)             cmd_logs "$@" ;;
    status|ps)        cmd_status "$@" ;;
    help|-h|--help)   usage ;;
    *)                die "unknown command '${sub}'" ;;
  esac
}

main "$@"
