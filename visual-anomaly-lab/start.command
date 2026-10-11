#!/bin/zsh
cd "${0:A:h}"
export PYTHONDONTWRITEBYTECODE=1
exec "${0:A:h:h}/.venv/bin/python" -B server.py
