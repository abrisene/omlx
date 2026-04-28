#!/usr/bin/env bash
set -euo pipefail

push_after=0
if [[ "${1:-}" == "--push" ]]; then
  push_after=1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git fetch upstream
git switch dev
git rebase upstream/main

if [[ "$push_after" -eq 1 ]]; then
  git push --force-with-lease origin dev
fi

echo "dev is now rebased onto upstream/main"
