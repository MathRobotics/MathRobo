#!/usr/bin/env bash
# Run from any directory; keep a separate uv environment for each Python.
set -uo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$root" || exit 1

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [3.9 3.10 ...]"
  echo "Default: 3.9 3.10 3.11 3.12 3.13 3.14"
  echo "Uses uv.lock; uv may download Python and dependencies. No publishing."
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required and must be on PATH." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- 3.9 3.10 3.11 3.12 3.13 3.14
fi
for version in "$@"; do
  if [[ ! "$version" =~ ^3\.[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Invalid Python version: $version (expected e.g. 3.12 or 3.12.8)" >&2
    exit 2
  fi
done

trap 'exit 130' INT
trap 'exit 143' TERM
results=()
failed=0
for version in "$@"; do
  echo
  echo "=== Python $version ==="
  UV_PROJECT_ENVIRONMENT="$root/.venv-py$version" \
    uv run --python "$version" --locked --extra dev --no-default-groups \
      python -m pytest tests
  status=$?
  if [[ $status -eq 130 || $status -eq 143 ]]; then
    exit "$status"
  elif [[ $status -eq 0 ]]; then
    results+=("Python $version: PASS")
  else
    results+=("Python $version: FAIL (exit $status; see output above)")
    failed=1
  fi
done

echo
echo "=== Results ==="
printf '%s\n' "${results[@]}"
exit "$failed"
