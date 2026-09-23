#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${repo_dir}/jupyterbook"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif [[ -x "${HOME}/anaconda3/bin/python3" ]]; then
  python_bin="${HOME}/anaconda3/bin/python3"
else
  python_bin="$(command -v python3)"
fi

if [[ ! -x "${venv_dir}/bin/python" ]] || ! "${venv_dir}/bin/python" -c 'import jupyter_book' >/dev/null 2>&1; then
  "${python_bin}" -m venv --clear "${venv_dir}"
fi

"${venv_dir}/bin/python" -m pip install --requirement "${repo_dir}/requirements.txt"

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  source "${venv_dir}/bin/activate"
  echo "Activated ${venv_dir} using ${python_bin}."
elif [[ "$#" -gt 0 ]]; then
  exec "${venv_dir}/bin/jupyter-book" "$@"
else
  echo "Virtual environment ready: ${venv_dir}"
  echo "Activate it in the current shell with: source ./start.sh"
  echo "Or build directly with: ./start.sh build notes/"
fi
