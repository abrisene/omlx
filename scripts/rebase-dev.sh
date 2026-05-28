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

# Build flow (post Swift rewrite, omlx upstream a1ade10+):
#   packaging/build.py only produces the venvstacks Python layers
#   (--venvstacks-only is now its only run mode; --skip-venv was retired).
#   The user-facing oMLX.app is built by apps/omlx-mac/Scripts/build.sh,
#   which auto-rebuilds the venvstacks donor when pyproject/venvstacks/uv.lock
#   fingerprints drift.
#   --build      -> rebuild the Swift bundle, reusing the existing export
#   --full-build -> force a fresh venvstacks rebuild + bundle
if [[ "$build_mode" == "skip-venv" ]]; then
  apps/omlx-mac/Scripts/build.sh release --no-rebuild-donor
elif [[ "$build_mode" == "full" ]]; then
  apps/omlx-mac/Scripts/build.sh release --rebuild-donor
fi

echo "dev is now rebased onto upstream/main"
