#!/bin/sh
set -eu

MINIMUM_MEMORY_GB=10
INSTALL_ROOT="${ORBIT_INSTALL_ROOT:-$HOME/.orbit}"
RUNTIME_DIR="$INSTALL_ROOT/runtime"
BIN_DIR="${ORBIT_BIN_DIR:-}"
ARCHIVE_URL="${ORBIT_ARCHIVE_URL:-https://github.com/ljcccc999/orbit/archive/refs/heads/main.tar.gz}"
TEMP_DIR=""
NEW_RUNTIME=""

cleanup() {
  if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
  fi
  if [ -n "$NEW_RUNTIME" ] && [ -d "$NEW_RUNTIME" ]; then
    rm -rf "$NEW_RUNTIME"
  fi
}
trap cleanup EXIT INT TERM

choose_bin_dir() {
  for candidate in /usr/local/bin /opt/homebrew/bin "$HOME/.local/bin"; do
    case ":$PATH:" in
      *":$candidate:"*)
        if [ -d "$candidate" ] && [ -w "$candidate" ]; then
          printf '%s' "$candidate"
          return
        fi
        ;;
    esac
  done
  printf '%s' "$HOME/.local/bin"
}

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

PYTHON_BIN=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PYTHON_BIN=$(command -v "$candidate")
    break
  fi
done

if [ -z "$BIN_DIR" ]; then
  BIN_DIR=$(choose_bin_dir)
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
RUNTIME_OK=false
if [ -x "$RUNTIME_DIR/bin/python" ] && "$RUNTIME_DIR/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  RUNTIME_OK=true
fi

if [ "$RUNTIME_OK" = false ]; then
  printf 'Creating the local Orbit runtime…\n'
  NEW_RUNTIME="$INSTALL_ROOT/runtime.new.$$"
  if [ -n "$PYTHON_BIN" ]; then
    "$PYTHON_BIN" -m venv "$NEW_RUNTIME"
  else
    if ! command -v curl >/dev/null 2>&1; then
      printf 'Orbit needs curl once to install its private Python 3.11 runtime.\n' >&2
      exit 1
    fi
    if [ -z "$TEMP_DIR" ]; then
      TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/orbit-install.XXXXXX")
    fi
    printf 'Installing a private Python 3.11 runtime for Orbit…\n'
    download "https://astral.sh/uv/install.sh" "$TEMP_DIR/uv-install.sh"
    UV_DIR="$INSTALL_ROOT/tools"
    env UV_UNMANAGED_INSTALL="$UV_DIR" sh "$TEMP_DIR/uv-install.sh"
    "$UV_DIR/uv" venv --python 3.11 "$NEW_RUNTIME"
  fi
  printf 'Installing Orbit and its local AI runtime…\n'
  "$NEW_RUNTIME/bin/python" -m pip install --retries 8 --timeout 60 --upgrade pip
  "$NEW_RUNTIME/bin/python" -m pip install --retries 8 --timeout 60 --upgrade "$SOURCE_DIR"
  if [ -d "$RUNTIME_DIR" ]; then
    OLD_RUNTIME="$INSTALL_ROOT/runtime.old.$$"
    mv "$RUNTIME_DIR" "$OLD_RUNTIME"
    ln -s "$NEW_RUNTIME" "$RUNTIME_DIR"
    NEW_RUNTIME=""
    rm -rf "$OLD_RUNTIME"
  else
    ln -s "$NEW_RUNTIME" "$RUNTIME_DIR"
    NEW_RUNTIME=""
  fi
else
  printf 'Installing Orbit and its local AI runtime…\n'
  "$RUNTIME_DIR/bin/python" -m pip install --retries 8 --timeout 60 --upgrade pip
  "$RUNTIME_DIR/bin/python" -m pip install --retries 8 --timeout 60 --upgrade "$SOURCE_DIR"
fi
ln -sf "$RUNTIME_DIR/bin/orbit" "$BIN_DIR/orbit"

printf '\nOrbit is installed. Models and training data stay in %s.\n' "$INSTALL_ROOT"
if ! printf '%s' ":$PATH:" | grep -q ":$BIN_DIR:"; then
  printf 'For later launches, add %s to PATH or run %s/orbit.\n' "$BIN_DIR" "$BIN_DIR"
fi
printf 'Starting the crash-recovering local API and opening the interface…\n'
if [ "${ORBIT_NO_BROWSER:-0}" = "1" ]; then
  exec "$RUNTIME_DIR/bin/orbit" start
fi
exec "$RUNTIME_DIR/bin/orbit" open
