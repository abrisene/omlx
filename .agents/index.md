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
```

If you want to update the remote integration branch too:

```bash
git push --force-with-lease origin dev
```

## Helper script

Use:

```bash
./scripts/rebase-dev.sh
```

Or to rebase and push:

```bash
./scripts/rebase-dev.sh --push
```

## Current intent

- `dev` is the place where upstream changes and local DeepSeek / integration work should meet.
- Keep `dev` healthy and current.
- Archive older one-off integration branches instead of continuing to use them as the main landing branch.
