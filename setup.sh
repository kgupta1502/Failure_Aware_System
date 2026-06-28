#!/bin/bash
# System Risk Index v2 Setup Script

echo "[*] Creating clean Python virtual environment..."
python3 -m venv venv

echo "[*] Activating virtual environment..."
source venv/bin/activate

echo "[*] Upgrading package manager..."
pip install --upgrade pip

echo "[*] Installing project dependencies from requirements.txt..."
pip install -r requirements.txt

echo "[*] Setup complete! To activate your environment later, run: source venv/bin/activate"