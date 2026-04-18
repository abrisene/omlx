#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/update-local-build.sh [options]

Update the local oMLX checkout, rebuild the app, reinstall it in /Applications,
and preserve current user settings/state via backups.

Options:
  --full-build     Rebuild venvstacks layers instead of using --skip-venv
  --no-rebase      Do not rebase the current branch onto updated main
  --no-restart     Do not relaunch the app after reinstall
  --sync-origin-main
                   Force-push local main to origin/main after syncing from upstream
  --macos-target V Set packaging/build.py --macos-target (e.g. 26.0)
  -h, --help       Show this help

Default behavior:
  1. git fetch upstream + origin
  2. reset local main to upstream/main
  3. optionally rebase the current branch onto main (unless already on main or --no-rebase)
  4. backup ~/.omlx and app config metadata
  5. rebuild the app
  6. replace /Applications/oMLX.app
  7. relaunch the app
EOF
}

FULL_BUILD=0
DO_REBASE=1
DO_RESTART=1
SYNC_ORIGIN_MAIN=0
MACOS_TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-build) FULL_BUILD=1; shift ;;
    --no-rebase) DO_REBASE=0; shift ;;
    --no-restart) DO_RESTART=0; shift ;;
    --sync-origin-main) SYNC_ORIGIN_MAIN=1; shift ;;
    --macos-target)
      MACOS_TARGET="${2:-}"
      if [[ -z "$MACOS_TARGET" ]]; then
        echo "error: --macos-target requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGING_DIR="$REPO_ROOT/packaging"
APP_PATH="/Applications/oMLX.app"
APP_SUPPORT_DIR="$HOME/Library/Application Support/oMLX"
BACKUP_ROOT="$APP_SUPPORT_DIR/backups/update-local-build-$(date +%Y%m%d-%H%M%S)"
CURRENT_BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"

mkdir -p "$BACKUP_ROOT"

echo "==> Repo: $REPO_ROOT"
echo "==> Branch: $CURRENT_BRANCH"
echo "==> Backup: $BACKUP_ROOT"

if ! git -C "$REPO_ROOT" remote | grep -qx upstream; then
  echo "error: expected an 'upstream' remote to exist" >&2
  exit 1
fi
if ! git -C "$REPO_ROOT" remote | grep -qx origin; then
  echo "error: expected an 'origin' remote to exist" >&2
  exit 1
fi

cp -f "$HOME/.omlx/settings.json" "$BACKUP_ROOT/settings.json" 2>/dev/null || true
cp -f "$HOME/.omlx/model_settings.json" "$BACKUP_ROOT/model_settings.json" 2>/dev/null || true
cp -f "$HOME/Library/Application Support/oMLX/config.json" "$BACKUP_ROOT/app-config.json" 2>/dev/null || true

{
  echo "timestamp: $(date)"
  echo "repo: $REPO_ROOT"
  echo "branch_before: $CURRENT_BRANCH"
  echo "commit_before: $(git -C "$REPO_ROOT" rev-parse HEAD)"
} > "$BACKUP_ROOT/BUILD.txt"

echo "==> Updating git refs"
git -C "$REPO_ROOT" fetch upstream
git -C "$REPO_ROOT" fetch origin

if [[ "$CURRENT_BRANCH" != "main" ]]; then
  git -C "$REPO_ROOT" switch main >/dev/null
fi
echo "==> Resetting local main to upstream/main"
git -C "$REPO_ROOT" reset --hard upstream/main >/dev/null

if [[ "$SYNC_ORIGIN_MAIN" -eq 1 ]]; then
  echo "==> Syncing origin/main to upstream/main"
  git -C "$REPO_ROOT" push --force-with-lease origin main
fi

if [[ "$CURRENT_BRANCH" != "main" ]]; then
  git -C "$REPO_ROOT" switch "$CURRENT_BRANCH" >/dev/null
  if [[ "$DO_REBASE" -eq 1 ]]; then
    echo "==> Rebasing $CURRENT_BRANCH onto main"
    git -C "$REPO_ROOT" rebase main
  fi
fi

echo "==> Stopping current app/server"
osascript -e 'tell application "oMLX" to quit' >/dev/null 2>&1 || true
for _ in $(seq 1 20); do
  if ! pgrep -f '/Applications/oMLX.app/Contents/MacOS/oMLX|/Applications/oMLX.app/Contents/MacOS/python3 -m omlx.cli serve' >/dev/null; then
    break
  fi
  sleep 1
done
pkill -f '/Applications/oMLX.app/Contents/MacOS/python3 -m omlx.cli serve' >/dev/null 2>&1 || true
pkill -f '/Applications/oMLX.app/Contents/MacOS/oMLX' >/dev/null 2>&1 || true
sleep 2

echo "==> Building app"
export PATH="$HOME/.local/bin:$PATH"
BUILD_ARGS=()
if [[ "$FULL_BUILD" -eq 0 ]]; then
  BUILD_ARGS+=(--skip-venv)
fi
if [[ -n "$MACOS_TARGET" ]]; then
  BUILD_ARGS+=(--macos-target "$MACOS_TARGET")
fi
(cd "$PACKAGING_DIR" && python3.11 build.py "${BUILD_ARGS[@]}")

if [[ -d "$APP_PATH" ]]; then
  echo "==> Backing up installed app"
  mv "$APP_PATH" "$BACKUP_ROOT/oMLX.app.previous"
fi

echo "==> Installing rebuilt app"
/usr/bin/ditto "$PACKAGING_DIR/dist/oMLX.app" "$APP_PATH"
xattr -dr com.apple.quarantine "$APP_PATH" >/dev/null 2>&1 || true

if [[ "$DO_RESTART" -eq 1 ]]; then
  echo "==> Relaunching app"
  open "$APP_PATH"
fi

{
  echo "branch_after: $(git -C "$REPO_ROOT" branch --show-current)"
  echo "commit_after: $(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "full_build: $FULL_BUILD"
  echo "rebased: $DO_REBASE"
  echo "restarted: $DO_RESTART"
  echo "sync_origin_main: $SYNC_ORIGIN_MAIN"
  echo "macos_target: ${MACOS_TARGET:-default}"
} >> "$BACKUP_ROOT/BUILD.txt"

echo "==> Done"
echo "Backup: $BACKUP_ROOT"
echo "Commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
