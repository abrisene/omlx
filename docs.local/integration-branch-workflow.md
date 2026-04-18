# Integration Branch Workflow

## Repository policy

This repo uses:

- `upstream` = canonical source repository
- `origin` = personal fork

Branch roles:

- `main`
  - should stay at parity with `upstream/main`
  - should not accumulate local integration-only commits
- `integration/local-patches`
  - carries local integrations, experiments, tuning changes, docs, and app workflow helpers
  - should be periodically rebased onto `main`

## Remote layout

Expected remotes:

```bash
origin   git@github.com:abrisene/omlx.git
upstream git@github.com:jundot/omlx.git
```

## Canonical update flow

### Refresh local `main` from upstream

```bash
git switch main
git fetch upstream
git reset --hard upstream/main
```

### Keep fork `main` in sync too

```bash
git push --force-with-lease origin main
```

### Rebase the integration branch

```bash
git switch integration/local-patches
git rebase main
```

### Push the integration branch

```bash
git push --force-with-lease origin integration/local-patches
```

## Local rebuild helper

Helper script:

```bash
scripts/update-local-build.sh
```

Default behavior:

1. fetch `upstream` and `origin`
2. reset local `main` to `upstream/main`
3. switch back to the current branch
4. rebase the current branch onto `main` unless `--no-rebase` is used
5. backup local app/settings metadata
6. rebuild the app
7. reinstall `/Applications/oMLX.app`
8. relaunch the app

### Optional sync of fork main

If you also want to update `origin/main` during the helper run:

```bash
./scripts/update-local-build.sh --sync-origin-main
```

This will force-push local `main` to the fork after syncing from upstream.

## Operational guidance

- Treat `main` as disposable/tracking state for upstream parity.
- Treat `integration/local-patches` as the durable local work branch.
- Prefer rebasing local integration work onto updated `main` rather than merging `main` back into the integration branch.
- Use the helper script for app rebuild/reinstall cycles.

## Current local branch of record

At the time of writing, active local integration work lives on:

```bash
integration/local-patches
```
