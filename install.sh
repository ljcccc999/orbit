#!/bin/sh
set -eu

MINIMUM_MEMORY_GB=10
INSTALL_ROOT="${ORBIT_INSTALL_ROOT:-$HOME/.orbit}"
RUNTIME_DIR="$INSTALL_ROOT/runtime"
BIN_DIR="${ORBIT_BIN_DIR:-$HOME/.local/bin}"
ARCHIVE_URL="${ORBIT_ARCHIVE_URL:-https://github.com/ljcccc999/orbit/archive/refs/heads/main.tar.gz}"
TEMP_DIR=""

cleanup() {
  if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
  fi
}
trap cleanup EXIT INT TERM

download() {
  url="$1"
  destination="$2"
  attempt=1
  while [ "$attempt" -le 5 ]; do
    if curl -fL --connect-timeout 20 --max-time 300 "$url" -o "$destination"; then
      return 0
    fi
    if [ "$attempt" -lt 5 ]; then
      printf 'Download interrupted. Retrying (%s/5)…\n' "$attempt" >&2
      sleep $((attempt * 2))
    fi
    attempt=$((attempt + 1))
  done
  printf 'Download failed after 5 attempts. Check the network connection and run the installer again.\n' >&2
  return 1
}

memory_gb() {
  if command -v sysctl >/dev/null 2>&1; then
    bytes=$(sysctl -n hw.memsize 2>/dev/null || true)
    if [ -n "$bytes" ]; then
      awk -v value="$bytes" 'BEGIN { printf "%.0f", value / 1000000000 }'
      return
    fi
  fi
  if [ -r /proc/meminfo ]; then
    awk '/MemTotal/ { printf "%.0f", $2 / 1000000; exit }' /proc/meminfo
    return
  fi
  printf '0'
}

MEMORY_GB=$(memory_gb)
if [ "$MEMORY_GB" -gt 0 ] && [ "$MEMORY_GB" -le "$MINIMUM_MEMORY_GB" ]; then
  printf 'Orbit requires more than %s GB of memory. This computer reports about %s GB.\n' "$MINIMUM_MEMORY_GB" "$MEMORY_GB" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  printf 'Python 3.9 or newer is required. Install Python and run this command again.\n' >&2
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
  printf 'Python 3.9 or newer is required.\n' >&2
  exit 1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || true)
if [ -n "${ORBIT_SOURCE_DIR:-}" ] && [ -f "$ORBIT_SOURCE_DIR/pyproject.toml" ]; then
  SOURCE_DIR="$ORBIT_SOURCE_DIR"
elif [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ] && [ -d "$SCRIPT_DIR/orbit" ]; then
  SOURCE_DIR="$SCRIPT_DIR"
else
  if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
    printf 'curl and tar are required for the one-line installer.\n' >&2
    exit 1
  fi
  TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/orbit-install.XXXXXX")
  printf 'Downloading Orbit…\n'
  download "$ARCHIVE_URL" "$TEMP_DIR/orbit.tar.gz"
  tar -xzf "$TEMP_DIR/orbit.tar.gz" -C "$TEMP_DIR"
  SOURCE_DIR=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 2 -name pyproject.toml -print | head -n 1 | sed 's#/pyproject.toml$##')
  if [ -z "$SOURCE_DIR" ]; then
    printf 'The Orbit archive is invalid.\n' >&2
    exit 1
  fi
fi

mkdir -p "$INSTALL_ROOT" "$BIN_DIR"
if [ ! -x "$RUNTIME_DIR/bin/python" ]; then
  printf 'Creating the local Orbit runtime…\n'
  python3 -m venv "$RUNTIME_DIR"
fi

printf 'Installing Orbit and its local AI runtime…\n'
"$RUNTIME_DIR/bin/python" -m pip install --retries 8 --timeout 60 --upgrade pip
"$RUNTIME_DIR/bin/python" -m pip install --retries 8 --timeout 60 --upgrade "$SOURCE_DIR"
ln -sf "$RUNTIME_DIR/bin/orbit" "$BIN_DIR/orbit"

printf '\nOrbit is installed. Models and training data stay in %s.\n' "$INSTALL_ROOT"
if ! printf '%s' ":$PATH:" | grep -q ":$BIN_DIR:"; then
  printf 'For later launches, add %s to PATH or run %s/orbit.\n' "$BIN_DIR" "$BIN_DIR"
fi
printf 'Opening the local web interface…\n'
exec "$RUNTIME_DIR/bin/orbit"
