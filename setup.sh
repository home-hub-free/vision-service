#!/usr/bin/env bash
# vision-service setup — venv + deps + (optional) systemd unit.
# Mirrors the other service setup.sh scripts. The null/CPU build needs no GPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "[vision] creating venv + installing web/MQTT deps (null build) ..."
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

[ -f .env ] || { cp .env.example .env; echo "[vision] wrote .env (edit HUB_SERVICE_TOKEN)"; }

echo "[vision] done. Run locally:"
echo "    .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8130"
echo "Install the service:"
echo "    sudo cp homehub-vision.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now homehub-vision"
echo "For real identity (M1/M2) see requirements.txt + DECISIONS.md."
