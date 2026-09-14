#!/usr/bin/env bash
# Source this file before invoking the Python 3.8 GenieDrive environment.
# Some managed GPU images inject their system Python 3.12 torch libraries into
# every conda environment. Mixing that libtorch_python.so with CPython 3.8
# produces an undefined PyObject_CallOneArg symbol.

_geniedrive_strip_python312_paths() {
  local raw="${1:-}"
  local item
  local cleaned=""
  local parts=()
  IFS=':' read -r -a parts <<< "${raw}"
  for item in "${parts[@]}"; do
    [[ -z "${item}" ]] && continue
    case "${item}" in
      *python3.12/dist-packages*|*python3.12/site-packages*) continue ;;
    esac
    cleaned="${cleaned:+${cleaned}:}${item}"
  done
  printf '%s' "${cleaned}"
}

_clean_pythonpath="$(_geniedrive_strip_python312_paths "${PYTHONPATH:-}")"
_clean_librarypath="$(_geniedrive_strip_python312_paths "${LD_LIBRARY_PATH:-}")"

if [[ -n "${_clean_pythonpath}" ]]; then
  export PYTHONPATH="${_clean_pythonpath}"
else
  unset PYTHONPATH
fi
if [[ -n "${_clean_librarypath}" ]]; then
  export LD_LIBRARY_PATH="${_clean_librarypath}"
else
  unset LD_LIBRARY_PATH
fi
unset LD_PRELOAD

unset _clean_pythonpath _clean_librarypath
unset -f _geniedrive_strip_python312_paths
