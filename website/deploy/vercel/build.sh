#!/usr/bin/env bash
set -euo pipefail

pip install -r deploy/vercel/requirements-vercel.txt
python manage.py collectstatic --noinput
