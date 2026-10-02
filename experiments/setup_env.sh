#!/usr/bin/env bash
# setup_env.sh -- one-time environment for the LLM measurement client.
#
# This client does NOT need a GPU: inference runs on the server behind $API_BASE.
# It needs python3, numpy, scipy and outbound network. Nothing else.
#
# Why a venv: a broken numpy in ~/.local/lib/pythonX.Y/site-packages shadows every
# other install and fails with "No module named 'numpy._utils'". A venv disables
# user site-packages, so ~/.local stops interfering without deleting anything.
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/graphrag-certified}"

echo "python: $(command -v python3)  ($(python3 -V 2>&1))"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1          # belt and braces: never read ~/.local

python -m pip install --upgrade pip
python -m pip install numpy scipy

echo
echo "== verifying =="
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'from', numpy.__file__)"
python -c "import scipy; print('scipy', scipy.__version__)"
python -c "import sys; print('user-site enabled:', __import__('site').ENABLE_USER_SITE)"

echo
echo "venv ready: $VENV"
echo "activate it in every new shell with:"
echo "    source $VENV/bin/activate && export PYTHONNOUSERSITE=1"
