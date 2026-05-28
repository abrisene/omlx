# Agent / Claude Notes

This file is the canonical shared instructions doc for local agents working in this repo.

`AGENTS.md` and `CLAUDE.md` should both symlink to this file.

## Branch workflow

### `dev` is the rolling integration branch

- Use `dev` as the local working integration branch.
- Keep `dev` continuously rebased onto `upstream/main`.
- Land local feature work onto `dev`, then continue rebasing `dev` forward.

### Feature branch policy

- Do focused work on feature branches when useful.
- Rebase or merge feature branches into `dev`.
- Avoid treating `main` as the active integration branch.

## Rebase routine

Short manual routine:

```bash
git fetch upstream
git switch dev
git rebase upstream/main
# Build the user-facing app (Swift bundle); auto-rebuilds the venvstacks
# donor only when dependency fingerprints drifted:
apps/omlx-mac/Scripts/build.sh release
```

If you want to update the remote integration branch too:

```bash
git push --force-with-lease origin dev
```

> **Build flow changed (upstream a1ade10+, Swift rewrite).** The PyObjC menubar
> app and the `build.py` `.app`/DMG pipeline are retired. `packaging/build.py`
> now only produces the venvstacks Python layers — its single run mode is
> `--venvstacks-only` (`--skip-venv` no longer exists). The macOS `oMLX.app`
> is built by `apps/omlx-mac/Scripts/build.sh`, which fingerprints
> `pyproject.toml` / `venvstacks.toml` / `uv.lock` and rebuilds the export
> automatically when they drift. Note `packaging/venvstacks.toml` is now
> auto-generated from `pyproject.toml` (`requirements = []`) — edit pins in
> `pyproject.toml`, not by hand.
>
> **Use Python 3.11+ for the build.** `build.sh` drives `build.py` with
> whatever `python3` resolves to; if that's the system 3.9 the donor build and
> fingerprint check fail (`TypeError: unsupported operand type(s) for |`). Set
> `PYTHON_BIN=$(command -v python3.11)` (the `rebase-dev.sh` helper does this
> automatically).

To build just the Python layers (e.g. to refresh a dev venv after a dep bump):

```bash
python3.11 packaging/build.py --venvstacks-only
```

To force a fresh venvstacks rebuild plus the bundle:

```bash
apps/omlx-mac/Scripts/build.sh release --rebuild-donor
```

## Helper script

Use:

```bash
./scripts/rebase-dev.sh --build
```

Or to rebase and push:

```bash
./scripts/rebase-dev.sh --build --push
```

For a full rebuild after rebasing:

```bash
./scripts/rebase-dev.sh --full-build
```

## Current intent

- `dev` is the place where upstream changes and local DeepSeek / integration work should meet.
- Keep `dev` healthy and current.
- Archive older one-off integration branches instead of continuing to use them as the main landing branch.
- When `dev` changes, build the latest app bundle so local testing stays on the
  current integration state.
