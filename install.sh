#!/bin/sh
# install.sh — Ollama KV Split installer for Linux (x86_64 / arm64).
#
# Fetches the latest release tarball from
#   https://github.com/strnad/ollama-kv-split/releases/latest
# and extracts it into /usr/. Registers a systemd unit named
# ollama.service. Stops the upstream ollama.service first if it is running, so
# the two installations don't fight over port 11434.
#
# Env overrides:
#   OLLAMA_VERSION  — pin a specific release tag (e.g. v0.21.0-split.1)
#   OLLAMA_REPO     — full repo path (default: strnad/ollama-kv-split)
#   OLLAMA_INSTALL_DIR — install prefix (default: /usr)
#   OLLAMA_FORCE_CPU=1 — skip CUDA detection and install the CPU tarball
#
# Usage:
#   curl -fsSL https://github.com/strnad/ollama-kv-split/releases/latest/download/install.sh | sh

set -eu

main() {

red="$( (/usr/bin/tput bold || :; /usr/bin/tput setaf 1 || :) 2>&-)"
plain="$( (/usr/bin/tput sgr0 || :) 2>&-)"

status() { echo ">>> $*" >&2; }
error() { echo "${red}ERROR:${plain} $*" >&2; exit 1; }
warning() { echo "${red}WARNING:${plain} $*" >&2; }

TEMP_DIR=$(mktemp -d)
cleanup() { rm -rf "$TEMP_DIR"; }
trap cleanup EXIT

available() { command -v "$1" >/dev/null 2>&1; }
require() {
    MISSING=''
    for TOOL in "$@"; do
        if ! available "$TOOL"; then
            MISSING="$MISSING $TOOL"
        fi
    done
    echo "$MISSING"
}

OS="$(uname -s)"
[ "$OS" = "Linux" ] || error "install.sh only supports Linux. For macOS or Windows use upstream ollama.com."

ARCH=$(uname -m)
case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) error "Unsupported architecture: $ARCH" ;;
esac

NEEDS=$(require curl tar gzip)
if [ -n "$NEEDS" ]; then
    status "ERROR: missing required tools:"
    for NEED in $NEEDS; do echo "  - $NEED"; done
    exit 1
fi

REPO="${OLLAMA_REPO:-strnad/ollama-kv-split}"
INSTALL_DIR="${OLLAMA_INSTALL_DIR:-/usr}"

# CUDA detection
FLAVOR="cpu"
if [ "${OLLAMA_FORCE_CPU:-0}" != "1" ] && available nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
    FLAVOR="cuda"
    status "NVIDIA GPU detected — installing CUDA build"
else
    status "No NVIDIA GPU detected — installing CPU build"
fi

# Resolve version (latest unless pinned)
if [ -n "${OLLAMA_VERSION:-}" ]; then
    RELEASE_PATH="download/${OLLAMA_VERSION}"
else
    RELEASE_PATH="latest/download"
fi

ASSET="ollama-linux-${ARCH}.tgz"
if [ "$FLAVOR" = "cuda" ]; then
    ASSET="ollama-linux-${ARCH}-cuda.tgz"
fi

URL="https://github.com/${REPO}/releases/${RELEASE_PATH}/${ASSET}"
status "downloading ${URL}"
if ! curl --fail --location --progress-bar -o "$TEMP_DIR/$ASSET" "$URL"; then
    if [ "$FLAVOR" = "cuda" ]; then
        warning "CUDA tarball not found — falling back to CPU build"
        ASSET="ollama-linux-${ARCH}.tgz"
        URL="https://github.com/${REPO}/releases/${RELEASE_PATH}/${ASSET}"
        status "downloading ${URL}"
        curl --fail --location --progress-bar -o "$TEMP_DIR/$ASSET" "$URL" \
            || error "download failed for $URL"
    else
        error "download failed for $URL"
    fi
fi

# Clear any stale shared libraries from a previous install before extracting.
# The tarball ships its own lib/ollama/ tree; leftover files from older builds
# or upstream Ollama would be linked-in-time and crash the new binary.
SUDO=''
if [ "$(id -u)" -ne 0 ]; then
    if available sudo; then
        SUDO='sudo'
    else
        error "root privileges are required. Install sudo or run as root."
    fi
fi

if [ -d "$INSTALL_DIR/lib/ollama" ]; then
    status "clearing stale $INSTALL_DIR/lib/ollama"
    $SUDO rm -rf "$INSTALL_DIR/lib/ollama"
fi

# Stop upstream ollama if present — single daemon, single port.
if systemctl list-unit-files ollama.service >/dev/null 2>&1; then
    if systemctl is-active --quiet ollama.service 2>/dev/null; then
        status "stopping existing ollama.service"
        $SUDO systemctl stop ollama.service || true
    fi
fi

status "extracting tarball into ${INSTALL_DIR}"
$SUDO tar -C "$INSTALL_DIR" -xzf "$TEMP_DIR/$ASSET"

# systemd unit
if available systemctl; then
    UNIT="/etc/systemd/system/ollama.service"
    status "installing systemd unit at ${UNIT}"
    if ! id ollama >/dev/null 2>&1; then
        $SUDO useradd --system --create-home --home-dir /usr/share/ollama \
            --shell /bin/false --comment "Ollama KV Split" ollama || true
    fi
    $SUDO sh -c "cat > $UNIT" <<EOF
[Unit]
Description=Ollama KV Split
After=network-online.target

[Service]
ExecStart=$INSTALL_DIR/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=$PATH"
Environment="HOME=/usr/share/ollama"

[Install]
WantedBy=default.target
EOF
    $SUDO systemctl daemon-reload
    $SUDO systemctl enable ollama.service
    $SUDO systemctl restart ollama.service
    status "ollama service is running on http://127.0.0.1:11434"
else
    status "systemd not detected — run '$INSTALL_DIR/bin/ollama serve' manually"
fi

status "installed: $($INSTALL_DIR/bin/ollama --version 2>&1 | head -1)"
}

main "$@"
