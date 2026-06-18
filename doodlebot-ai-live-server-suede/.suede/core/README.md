# [Suede](https://github.com/pmalacho-mit/suede) Core (`release`)

Utilities for managing / interacting with the release branch of a suede dependency.

## [upstream](./upstream)

Propose this dependency's **local changes back to the library**, as a reviewable
PR against the library's `main`.

Use it when you've edited a vendored dependency in place and want those edits to
become a contribution to the library itself (rather than living only in your repo).

### Usage

```bash
<path>/.suede/upstream              # if executable
bash <path>/.suede/upstream         # otherwise
```

First commit the changes you want to send — the working tree must be clean.

### What it does

1. Splits the dependency's local commits out via `git subrepo` and pushes them to
   a deterministic branch on the library's remote:
   `downstream/<owner>/<repo>-<your-commit>`.
2. A [GitHub Action]() on the library rebuilds that branch as a `main`-shaped PR head
   and opens the pull request for the maintainers to test, fix, and merge.
3. Your local state is restored afterward, so a later `git subrepo pull` stays
   safe. The `release` branch is **never** modified, so other consumers are
   unaffected.

Each commit becomes its own snapshot/branch/PR; re-running on the same commit is a
no-op (it detects the already-open proposal).

### Notes

- It's a thin bootstrapper: the real logic is hosted at `https://suede.sh/upstream`
  so it can evolve without re-shipping dependencies. Override the host (for forks
  or testing) with `SUEDE_UPSTREAM_URL`.
- Requires `git`, `curl`, and [`git-subrepo`](https://github.com/ingydotnet/git-subrepo).
- Pass `-r`/`--remote <name>` to push to a remote other than the one tracked in
  the dependency's `.gitrepo`.