#!/usr/bin/env bash
# Git Bash / WSL entry point
set -e
cd "$(dirname "$0")"

if [ ! -f "python/python.exe" ]; then
    echo "[LOI] Khong tim thay python/python.exe"
    exit 1
fi

echo "Dang khoi dong (build graph mat ~40 giay)..."
./python/python.exe launcher.py
