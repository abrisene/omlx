#!/usr/bin/env bash
set -euo pipefail

push_after=0
build_mode=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)
      push_after=1
      ;;
    --build)
      build_mode="skip-venv"
      ;;
    --full-build)
      build_mode="full"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--push] [--build|--full-build]" >&2
      exit 1
      ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git fetch upstream
git switch dev
git rebase upstream/main

if [[ "$push_after" -eq 1 ]]; then
  git push --force-with-lease origin dev
fi

if [[ "$build_mode" == "skip-venv" ]]; then
  (
    cd packaging
    python3.11 build.py --skip-venv
  )
elif [[ "$build_mode" == "full" ]]; then
  (
    cd packaging
    python3.11 build.py
  )
fi

echo "dev is now rebased onto upstream/main"
